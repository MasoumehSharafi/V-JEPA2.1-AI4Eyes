"""The unlabelled-data manifest: one row per media item.

This is the join format meant to outlive any one pipeline run -- the SSL
pretraining dataset reads clips from it today, and a later *joint*
image+video pretraining can simply concatenate a video manifest and an
image manifest, since both share this same schema. See ../README.md
("Manifest schema") for the full documented format.

Columns (CSV, header required):
    key          unique id for this media item, e.g. "<exam_id>-<eye>"
    media_type   "video" | "image"
    test_type    e.g. "NITBUT" | "INVASIVE_TBUT" | "BLINK_MECHANICS"
                 (image test types are added later by the image pipeline)
    patient_id   patient identifier
    exam_id      exam identifier
    eye          "OD" | "OS" (required for the OD/OS split-integrity rule
                 below; leave blank only for media with no laterality,
                 e.g. a future whole-face image type)
    split        "train" | "val" | "test" -- copied verbatim from the
                 shared, authoritative split CSV (see `read_split_csv`
                 below). This column is NEVER recomputed by this package;
                 it is read once and passed through.
    relative_path  path to the media file, relative to a configured
                   video/image root (see config.paths().video_root) --
                   never an absolute machine path or an S3 URI.

No patient/exam id, label, or path in this file is real -- this module
only defines the schema and validation; actual manifests are
environment-specific and are never committed (see ../../.gitignore).
"""
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REQUIRED_COLUMNS = (
    "key",
    "media_type",
    "test_type",
    "patient_id",
    "exam_id",
    "eye",
    "split",
    "relative_path",
)
VALID_MEDIA_TYPES = {"video", "image"}
VALID_SPLITS = {"train", "val", "test"}
VALID_EYES = {"OD", "OS", ""}


@dataclass(frozen=True)
class MediaItem:
    key: str
    media_type: str
    test_type: str
    patient_id: str
    exam_id: str
    eye: str
    split: str
    relative_path: str


class ManifestError(ValueError):
    """Raised for any manifest schema or clinical-rule violation."""


def read_manifest(path: str) -> list[MediaItem]:
    """Reads and validates a manifest CSV. Raises ManifestError on any
    schema violation or clinical-rule breach (see `validate_od_os_rule`)."""
    rows: list[MediaItem] = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ManifestError(f"Manifest {path} missing columns: {sorted(missing)}")
        for line_no, raw in enumerate(reader, start=2):
            rows.append(_parse_row(raw, path, line_no))
    validate_od_os_rule(rows)
    return rows


def _parse_row(raw: dict, path: str, line_no: int) -> MediaItem:
    item = MediaItem(
        key=raw["key"].strip(),
        media_type=raw["media_type"].strip(),
        test_type=raw["test_type"].strip(),
        patient_id=raw["patient_id"].strip(),
        exam_id=raw["exam_id"].strip(),
        eye=raw["eye"].strip(),
        split=raw["split"].strip(),
        relative_path=raw["relative_path"].strip(),
    )
    if not item.key:
        raise ManifestError(f"{path}:{line_no}: empty key")
    if item.media_type not in VALID_MEDIA_TYPES:
        raise ManifestError(
            f"{path}:{line_no}: media_type must be one of {VALID_MEDIA_TYPES}, got {item.media_type!r}"
        )
    if item.split not in VALID_SPLITS:
        raise ManifestError(
            f"{path}:{line_no}: split must be one of {VALID_SPLITS}, got {item.split!r}"
        )
    if item.eye not in VALID_EYES:
        raise ManifestError(
            f"{path}:{line_no}: eye must be one of {VALID_EYES}, got {item.eye!r}"
        )
    if item.relative_path.startswith("/") or item.relative_path.startswith("s3://"):
        raise ManifestError(
            f"{path}:{line_no}: relative_path must be relative, got {item.relative_path!r}"
        )
    return item


def validate_od_os_rule(items: list[MediaItem]) -> None:
    """Clinical rule: a single patient's OD and OS must never land in
    different splits. Raises ManifestError (naming the offending patient)
    on any violation -- never silently drops or reassigns a row."""
    splits_by_patient: dict[str, set[str]] = {}
    for item in items:
        if not item.eye:
            continue  # no laterality to check (e.g. a future whole-face image)
        splits_by_patient.setdefault(item.patient_id, set()).add(item.split)
    offenders = {pid: splits for pid, splits in splits_by_patient.items() if len(splits) > 1}
    if offenders:
        detail = ", ".join(f"{pid}: {sorted(splits)}" for pid, splits in sorted(offenders.items()))
        raise ManifestError(
            f"OD/OS split-integrity violation -- these patients have eyes in "
            f"more than one split ({detail})"
        )


def read_split_csv(path: str) -> dict[str, str]:
    """Reads the shared, authoritative split CSV (key -> split). This is
    READ-ONLY: the split is never recomputed or redrawn here or anywhere
    else in this package -- any new manifest must be built by joining
    against this file's assignments, not by re-deriving a split."""
    assignments: dict[str, str] = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if "key" not in (reader.fieldnames or []) or "split" not in (reader.fieldnames or []):
            raise ManifestError(f"Split CSV {path} must have at least 'key' and 'split' columns")
        for line_no, raw in enumerate(reader, start=2):
            split = raw["split"].strip()
            if split not in VALID_SPLITS:
                raise ManifestError(f"{path}:{line_no}: split must be one of {VALID_SPLITS}, got {split!r}")
            assignments[raw["key"].strip()] = split
    return assignments


def filter_manifest(
    items: list[MediaItem],
    test_type: Optional[str] = None,
    media_type: Optional[str] = None,
    split: Optional[str] = None,
) -> list[MediaItem]:
    """Convenience filter, e.g. `filter_manifest(items, test_type="NITBUT", split="train")`."""
    out = items
    if test_type is not None:
        out = [i for i in out if i.test_type == test_type]
    if media_type is not None:
        out = [i for i in out if i.media_type == media_type]
    if split is not None:
        out = [i for i in out if i.split == split]
    return out


def resolve_path(item: MediaItem, media_root: str) -> Path:
    return Path(media_root) / item.relative_path
