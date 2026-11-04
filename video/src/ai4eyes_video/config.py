"""Central environment-variable resolution.

No path, bucket name, or credential in this package is ever hardcoded --
everything here has a documented default that assumes a local, relative
layout, and every default can be overridden by the matching environment
variable. See ../README.md ("Configuration") for the full table.
"""
import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Paths:
    # Where raw (never-committed) data lives: the clinical CSV export,
    # the shared split CSV, per-test annotation JSON, and video files.
    data_root: str
    split_csv: str
    manifest_csv: str
    annotations_dir: str
    video_root: str

    # S3 source (only used by data-prep download helpers; never hardcoded).
    s3_bucket: str
    s3_prefix: str

    # Fast local scratch (e.g. an ephemeral NVMe volume) for active
    # training runs -- NOT assumed to be durable storage.
    work_dir: str

    # Durable checkpoint storage (outside the scratch volume).
    checkpoint_dir: str

    # The cloned facebookresearch/vjepa2 checkout (see scripts/setup_vjepa.sh)
    # and the backbone checkpoint currently selected for training/eval.
    vjepa2_repo: str
    vjepa2_checkpoint: str


def paths() -> Paths:
    """Reads environment variables fresh on every call (tests rely on this:
    they monkeypatch os.environ and expect a new `paths()` call to see it)."""
    return Paths(
        data_root=_env("AI4EYES_DATA_ROOT", "./data"),
        split_csv=_env("AI4EYES_SPLIT_CSV", "./data/split.csv"),
        manifest_csv=_env("AI4EYES_MANIFEST", "./data/manifest.csv"),
        annotations_dir=_env("AI4EYES_ANNOTATIONS_DIR", "./data/annotations"),
        video_root=_env("AI4EYES_VIDEO_ROOT", "./data/videos"),
        s3_bucket=_env("AI4EYES_S3_BUCKET", ""),
        s3_prefix=_env("AI4EYES_S3_PREFIX", ""),
        work_dir=_env("AI4EYES_WORK_DIR", "./work"),
        checkpoint_dir=_env("AI4EYES_CHECKPOINT_DIR", "./checkpoints"),
        vjepa2_repo=_env("VJEPA2_REPO", "./vjepa2"),
        vjepa2_checkpoint=_env(
            "VJEPA2_CHECKPOINT", "./checkpoints/vjepa2_1_vitl_dist_vitG_384.pt"
        ),
    )
