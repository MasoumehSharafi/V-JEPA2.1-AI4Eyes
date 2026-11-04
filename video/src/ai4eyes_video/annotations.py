"""Per-test labeled annotations: the supervised-training/eval format.

One JSON file per test type (NITBUT / INVASIVE_TBUT / BLINK_MECHANICS),
built from the manifest + the shared split CSV (see manifest.py) plus
whatever ground-truth segments were labeled for that test. Never
committed (see ../../.gitignore); this module only defines the schema
and a reader/validator.

Schema:
{
  "version": "ai4eyes-video-1.0",
  "test_type": "NITBUT",
  "classes": ["NITBUT"],
  "database": {
    "<key>": {
      "duration": 20.03,              # seconds
      "fps": 29.97,
      "split": "train" | "val" | "test",   # copied from the split CSV,
                                             # never recomputed here
      "patient_id": "...",
      "exam_id": "...",
      "eye": "OD" | "OS",
      "relative_path": "videos/....mp4",    # relative to config paths().video_root
      "annotations": [{"segment": [1.2, 5.6], "label": "NITBUT"}]
      # 0 annotations = no gradable event in this video (background/
      # ungradable); kept, never dropped.
    },
    ...
  }
}
"""
import json
from dataclasses import dataclass, field
from pathlib import Path


class AnnotationsError(ValueError):
    pass


@dataclass(frozen=True)
class VideoRecord:
    key: str
    duration: float
    fps: float
    split: str
    patient_id: str
    exam_id: str
    eye: str
    relative_path: str
    annotations: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class AnnotationSet:
    test_type: str
    classes: list[str]
    records: dict[str, VideoRecord]

    def split(self, split: str) -> list[VideoRecord]:
        return [r for r in self.records.values() if r.split == split]


def read_annotations(path: str) -> AnnotationSet:
    payload = json.loads(Path(path).read_text())
    for key in ("version", "test_type", "classes", "database"):
        if key not in payload:
            raise AnnotationsError(f"{path}: missing top-level key {key!r}")

    records: dict[str, VideoRecord] = {}
    for key, entry in payload["database"].items():
        for required in ("duration", "fps", "split", "patient_id", "exam_id", "eye", "relative_path"):
            if required not in entry:
                raise AnnotationsError(f"{path}: database[{key!r}] missing {required!r}")
        annotations = entry.get("annotations", [])
        for ann in annotations:
            if "segment" not in ann or "label" not in ann:
                raise AnnotationsError(f"{path}: database[{key!r}] has a malformed annotation: {ann!r}")
            start, end = ann["segment"]
            if not (end > start >= 0):
                raise AnnotationsError(
                    f"{path}: database[{key!r}] has an invalid segment {ann['segment']!r}"
                )
            if ann["label"] not in payload["classes"]:
                raise AnnotationsError(
                    f"{path}: database[{key!r}] label {ann['label']!r} not in classes {payload['classes']}"
                )
        records[key] = VideoRecord(
            key=key,
            duration=float(entry["duration"]),
            fps=float(entry["fps"]),
            split=entry["split"],
            patient_id=entry["patient_id"],
            exam_id=entry["exam_id"],
            eye=entry["eye"],
            relative_path=entry["relative_path"],
            annotations=annotations,
        )
    return AnnotationSet(test_type=payload["test_type"], classes=payload["classes"], records=records)
