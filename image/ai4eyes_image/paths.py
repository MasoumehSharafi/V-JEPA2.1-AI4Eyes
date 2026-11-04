"""Every location the pipeline reads or writes, set by environment variables (see ../env.example).

Nothing here is machine-specific: defaults are folders inside image/, and all of them are git-ignored.
"""
import os
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]          # .../image


def _path(var, default):
    return Path(os.environ.get(var) or default).expanduser().resolve()


DATA_DIR = _path("AI4EYES_DATA_DIR", PKG_ROOT / "data")
LABELS_CSV = _path("AI4EYES_LABELS_CSV", DATA_DIR / "labels.csv")          # one row per (exam, eye, test, labeller)
SPLIT_CSV = _path("AI4EYES_SPLIT_CSV", DATA_DIR / "split.csv")             # patient_pid, split (train / val / test)
IMAGES_DIR = _path("AI4EYES_IMAGES_DIR", DATA_DIR / "images")              # <exam_id>/<file> as in S3
SSL_MANIFEST = _path("AI4EYES_SSL_MANIFEST", DATA_DIR / "ssl_manifest.csv")
CHECKPOINT_DIR = _path("AI4EYES_CHECKPOINT_DIR", PKG_ROOT / "checkpoints")
OUTPUT_DIR = _path("AI4EYES_OUTPUT_DIR", PKG_ROOT / "outputs")
WORK_DIR = _path("AI4EYES_WORK_DIR", PKG_ROOT / "work")                    # fast scratch disk (e.g. local NVMe)
VJEPA2_DIR = _path("AI4EYES_VJEPA2_DIR", PKG_ROOT / "third_party" / "vjepa2")
S3_BUCKET = os.environ.get("AI4EYES_S3_BUCKET", "")


def require_bucket():
    if not S3_BUCKET:
        raise SystemExit("set AI4EYES_S3_BUCKET (see env.example)")
    return S3_BUCKET
