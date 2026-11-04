#!/usr/bin/env bash
# Labelled data: download the media (S3, read-only), extract the lower-lid frames, cache the cornea / ring masks,
# then write (or verify) the shared patient split.
#   scripts/prepare_data.sh [--priority-ids FILE]     # FILE: see ai4eyes_image/data/frames.py
set -euo pipefail
. "$(dirname "$0")/_env.sh"
step "download labelled media"
$PY -m ai4eyes_image.data.download
step "lower-lid frames from the lipid-layer videos"
$PY -m ai4eyes_image.data.frames "$@"
step "cornea + fixation-ring masks of the staining photos (cached next to each image)"
$PY - <<'PYEOF'
import pandas as pd
from ai4eyes_image import paths
from ai4eyes_image.preprocess.masks import build_masks
from ai4eyes_image.signs import labels
d = labels("corneal_staining", pd.read_csv(paths.LABELS_CSV, low_memory=False))
new = build_masks(sorted(set(d.image_path)))
print(f"{len(new)} masks computed, {sum(not r['ok'] for r in new)} fell back to the centre square")
PYEOF
step "shared patient split (written once, verified afterwards)"
$PY -m ai4eyes_image.data.split
