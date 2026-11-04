"""Clip sampling for supervised training/eval: random-window training clips,
sliding-window inference clips. Ground-truth segments are converted from
video-absolute seconds to window-local frame indices here; everything
downstream (model, metrics) works in those coordinates (metrics.py converts
back to seconds using each video's fps before scoring).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .annotations import AnnotationSet, VideoRecord

WINDOW_FRAMES = 160  # total temporal window length sampled from the video
CHUNK_FRAMES = 16  # frames per V-JEPA2 encoder forward pass (tubelet input)
RESOLUTION = 224  # matches the domain-adaptive-pretrain resolution (224px)


def _load_decord_reader(video_path: Path):
    import decord  # imported lazily: only required when actually reading video

    decord.bridge.set_bridge("native")
    return decord.VideoReader(str(video_path), num_threads=1)


def _resize_crop(frames_hwc: np.ndarray, resolution: int) -> np.ndarray:
    """frames_hwc: [T, H, W, 3] uint8. Resize short side then center-crop to
    a square `resolution`. Uses a minimal, dependency-light implementation
    (PIL) rather than a video-specific augmentation library."""
    from PIL import Image

    out = np.empty((frames_hwc.shape[0], resolution, resolution, 3), dtype=np.uint8)
    for t in range(frames_hwc.shape[0]):
        img = Image.fromarray(frames_hwc[t])
        w, h = img.size
        scale = resolution / min(w, h)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))))
        w, h = img.size
        left = (w - resolution) // 2
        top = (h - resolution) // 2
        img = img.crop((left, top, left + resolution, top + resolution))
        out[t] = np.asarray(img)
    return out


def _to_tensor(frames_hwc: np.ndarray) -> torch.Tensor:
    """[T,H,W,3] uint8 -> [3,T,H,W] float32 in [0,255] (normalization is the
    backbone's job, matching how the ImageNet mean/std are applied inside
    the data preprocessor in the original pipeline)."""
    frames = torch.from_numpy(frames_hwc).permute(3, 0, 1, 2).contiguous().float()
    return frames


def _read_window(video_path: Path, start_frame: int, num_frames: int, total_frames: int, resolution: int) -> torch.Tensor:
    reader = _load_decord_reader(video_path)
    indices = [min(start_frame + i, total_frames - 1) for i in range(num_frames)]
    frames = reader.get_batch(indices).asnumpy()  # [T,H,W,3]
    frames = _resize_crop(frames, resolution)
    return _to_tensor(frames)


@dataclass
class TrainSample:
    frames: torch.Tensor  # [3, WINDOW_FRAMES, RESOLUTION, RESOLUTION]
    gt_segments: torch.Tensor  # [N, 2] window-local frame indices (float)
    gt_labels: torch.Tensor  # [N] int64 class indices
    key: str


class ClipTrainDataset(Dataset):
    """Random-window training clips. One sample per __getitem__ call, drawn
    uniformly from `records` (videos with zero ground-truth segments --
    "background", e.g. an ungradable eye -- are included and simply
    contribute no positive targets)."""

    def __init__(self, records: list[VideoRecord], classes: list[str], video_root: str,
                 window_frames: int = WINDOW_FRAMES, resolution: int = RESOLUTION, seed: int = 0):
        if not records:
            raise ValueError("ClipTrainDataset received zero records")
        self.records = records
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.video_root = Path(video_root)
        self.window_frames = window_frames
        self.resolution = resolution
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> TrainSample:
        record = self.records[idx % len(self.records)]
        total_frames = max(1, round(record.duration * record.fps))
        window = min(self.window_frames, total_frames)
        start_frame = int(self._rng.integers(0, max(1, total_frames - window + 1)))

        frames = _read_window(
            self.video_root / record.relative_path, start_frame, self.window_frames, total_frames, self.resolution
        )

        segments, labels = [], []
        for ann in record.annotations:
            seg_start_s, seg_end_s = ann["segment"]
            seg_start_f = seg_start_s * record.fps - start_frame
            seg_end_f = seg_end_s * record.fps - start_frame
            # Clip to the sampled window; drop segments with no overlap.
            clipped_start = max(0.0, seg_start_f)
            clipped_end = min(float(self.window_frames), seg_end_f)
            if clipped_end <= clipped_start:
                continue
            segments.append([clipped_start, clipped_end])
            labels.append(self.class_to_idx[ann["label"]])

        gt_segments = torch.tensor(segments, dtype=torch.float32) if segments else torch.zeros((0, 2))
        gt_labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        return TrainSample(frames=frames, gt_segments=gt_segments, gt_labels=gt_labels, key=record.key)


def train_collate(batch: list[TrainSample]) -> dict:
    return {
        "frames": torch.stack([b.frames for b in batch]),
        "gt_segments": [b.gt_segments for b in batch],
        "gt_labels": [b.gt_labels for b in batch],
        "keys": [b.key for b in batch],
    }


@dataclass
class InferenceWindow:
    frames: torch.Tensor
    window_start_frame: int
    window_frames: int
    key: str
    fps: float
    duration: float


class ClipSlidingDataset(Dataset):
    """Sliding-window inference clips covering the full duration of every
    video in `records`. One window per __getitem__ call; `evaluate.py` is
    responsible for stitching per-window predictions back into one
    video-level prediction list using `window_start_frame` and `fps`."""

    def __init__(self, records: list[VideoRecord], video_root: str,
                 window_frames: int = WINDOW_FRAMES, overlap_ratio: float = 0.5, resolution: int = RESOLUTION):
        self.video_root = Path(video_root)
        self.window_frames = window_frames
        self.resolution = resolution
        self._windows: list[tuple[VideoRecord, int]] = []
        stride = max(1, round(window_frames * (1 - overlap_ratio)))
        for record in records:
            total_frames = max(1, round(record.duration * record.fps))
            if total_frames <= window_frames:
                self._windows.append((record, 0))
                continue
            start = 0
            while start < total_frames:
                self._windows.append((record, start))
                if start + window_frames >= total_frames:
                    break
                start += stride

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> InferenceWindow:
        record, start_frame = self._windows[idx]
        total_frames = max(1, round(record.duration * record.fps))
        frames = _read_window(
            self.video_root / record.relative_path, start_frame, self.window_frames, total_frames, self.resolution
        )
        return InferenceWindow(
            frames=frames, window_start_frame=start_frame, window_frames=self.window_frames,
            key=record.key, fps=record.fps, duration=record.duration,
        )
