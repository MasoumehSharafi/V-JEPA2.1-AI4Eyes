"""Ties the frozen V-JEPA2.1 backbone to the trainable ActionFormer-style
head: chunks the input window into backbone-sized clips, pools + upsamples
back to one feature per input frame, then runs the multi-scale head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import VJEPA21Backbone
from .heads import (
    DEFAULT_REGRESSION_RANGES,
    DEFAULT_STRIDES,
    ActionFormerHead,
    FPN1D,
    Target,
    assign_targets,
    diou_loss_1d,
    generate_points,
    nms_1d,
    sigmoid_focal_loss,
)

CHUNK_FRAMES = 16  # must match backbone's num_frames/tubelet config


class TemporalActionDetector(nn.Module):
    def __init__(
        self,
        backbone: VJEPA21Backbone,
        num_classes: int,
        window_frames: int = 160,
        head_channels: int = 512,
        num_levels: int = len(DEFAULT_STRIDES),
        strides: tuple = DEFAULT_STRIDES,
        regression_ranges: tuple = DEFAULT_REGRESSION_RANGES,
        center_sample_radius: float = 1.5,
        loss_normalizer_momentum: float = 0.9,
    ):
        super().__init__()
        if window_frames % CHUNK_FRAMES != 0:
            raise ValueError(f"window_frames={window_frames} must be a multiple of CHUNK_FRAMES={CHUNK_FRAMES}")
        self.backbone = backbone
        self.window_frames = window_frames
        self.num_chunks = window_frames // CHUNK_FRAMES
        self.num_classes = num_classes
        self.strides = strides
        self.regression_ranges = regression_ranges
        self.center_sample_radius = center_sample_radius

        self.fpn = FPN1D(in_channels=backbone.embed_dim, out_channels=head_channels, num_levels=num_levels)
        self.head = ActionFormerHead(in_channels=head_channels, num_classes=num_classes)

        self.register_buffer("loss_normalizer", torch.tensor(100.0), persistent=False)
        self.loss_normalizer_momentum = loss_normalizer_momentum

    def extract_sequence(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: [B, 3, window_frames, H, W] -> [B, C, window_frames]
        (one backbone-derived feature vector per input frame, via chunked
        backbone forward + spatial mean-pool + temporal interpolation)."""
        batch, channels, total_frames, height, width = frames.shape
        chunks = frames.view(batch, channels, self.num_chunks, CHUNK_FRAMES, height, width)
        chunks = chunks.permute(0, 2, 1, 3, 4, 5).reshape(batch * self.num_chunks, channels, CHUNK_FRAMES, height, width)

        feats = self.backbone(chunks)  # [B*num_chunks, C, T_chunk, H_tok, W_tok]
        feats = feats.mean(dim=(3, 4))  # spatial mean pool -> [B*num_chunks, C, T_chunk]

        embed_dim, t_chunk = feats.shape[1], feats.shape[2]
        feats = feats.view(batch, self.num_chunks, embed_dim, t_chunk)
        feats = feats.permute(0, 2, 1, 3).reshape(batch, embed_dim, self.num_chunks * t_chunk)

        feats = F.interpolate(feats, size=self.window_frames, mode="linear", align_corners=False)
        return feats  # [B, C, window_frames]

    def forward_train(self, frames: torch.Tensor, gt_segments: list[torch.Tensor], gt_labels: list[torch.Tensor]) -> dict:
        sequence = self.extract_sequence(frames)
        levels = self.fpn(sequence)
        cls_outputs, reg_outputs = self.head(levels)

        level_lengths = [lvl.shape[-1] for lvl in cls_outputs]
        points = generate_points(level_lengths, self.strides, frames.device)

        cls_logits = torch.cat([c.permute(0, 2, 1).reshape(c.shape[0], -1, self.num_classes) for c in cls_outputs], dim=1)
        reg_preds = torch.cat([r.permute(0, 2, 1).reshape(r.shape[0], -1, 2) for r in reg_outputs], dim=1)

        total_cls_loss, total_reg_loss, total_pos = 0.0, 0.0, 0
        for b in range(frames.shape[0]):
            targets = [
                Target(start=seg[0].item(), end=seg[1].item(), label=int(label))
                for seg, label in zip(gt_segments[b], gt_labels[b])
            ]
            cls_t, reg_t, pos_mask = assign_targets(
                points, self.regression_ranges, self.strides, targets, self.num_classes, self.center_sample_radius
            )
            total_cls_loss = total_cls_loss + sigmoid_focal_loss(cls_logits[b], cls_t)
            n_pos = int(pos_mask.sum().item())
            total_pos += n_pos
            if n_pos > 0:
                total_reg_loss = total_reg_loss + diou_loss_1d(reg_preds[b][pos_mask], reg_t[pos_mask])

        with torch.no_grad():
            normalizer = max(total_pos, 1)
            self.loss_normalizer.mul_(self.loss_normalizer_momentum).add_(
                normalizer * (1 - self.loss_normalizer_momentum)
            )
        denom = self.loss_normalizer.clamp(min=1.0)
        cls_loss = total_cls_loss / denom
        reg_loss = total_reg_loss / denom
        return {"cls_loss": cls_loss, "reg_loss": reg_loss, "loss": cls_loss + reg_loss}

    @torch.no_grad()
    def forward_test(self, frames: torch.Tensor, score_threshold: float = 0.05, nms_iou: float = 0.5, max_detections: int = 200):
        """Returns a list (len == batch size) of lists of
        {"segment": [start, end], "label": int, "score": float}, in
        WINDOW-LOCAL FRAME coordinates (the caller maps these back to
        video-level seconds -- see evaluate.py)."""
        sequence = self.extract_sequence(frames)
        levels = self.fpn(sequence)
        cls_outputs, reg_outputs = self.head(levels)
        level_lengths = [lvl.shape[-1] for lvl in cls_outputs]
        points = generate_points(level_lengths, self.strides, frames.device)
        all_points = torch.cat(points)

        cls_logits = torch.cat([c.permute(0, 2, 1).reshape(c.shape[0], -1, self.num_classes) for c in cls_outputs], dim=1)
        reg_preds = torch.cat([r.permute(0, 2, 1).reshape(r.shape[0], -1, 2) for r in reg_outputs], dim=1)
        scores_all = torch.sigmoid(cls_logits)  # [B, N, num_classes]

        results = []
        for b in range(frames.shape[0]):
            starts = all_points - reg_preds[b, :, 0]
            ends = all_points + reg_preds[b, :, 1]
            segments = torch.stack([starts, ends], dim=1).clamp(min=0, max=float(self.window_frames))

            detections = []
            for cls_idx in range(self.num_classes):
                cls_scores = scores_all[b, :, cls_idx]
                keep_mask = cls_scores >= score_threshold
                if not keep_mask.any():
                    continue
                cls_segments = segments[keep_mask]
                cls_conf = cls_scores[keep_mask]
                keep = nms_1d(cls_segments, cls_conf, iou_threshold=nms_iou, max_segments=max_detections)
                for i in keep.tolist():
                    detections.append(
                        {
                            "segment": [cls_segments[i, 0].item(), cls_segments[i, 1].item()],
                            "label": cls_idx,
                            "score": cls_conf[i].item(),
                        }
                    )
            results.append(detections)
        return results
