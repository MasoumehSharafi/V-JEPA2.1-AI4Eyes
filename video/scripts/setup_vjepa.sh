#!/usr/bin/env bash
#
# Clones the official facebookresearch/vjepa2 repo at a pinned commit and
# downloads the official V-JEPA2.1 checkpoint we build on. Never vendors
# vjepa2's code or weights into this repo -- both land in gitignored
# locations (VJEPA2_REPO, AI4EYES_CHECKPOINT_DIR).
#
# Model / resolution used throughout this project (the image pipeline
# should match): vjepa2_1_vit_large_384 -- V-JEPA 2.1 ViT-L/16, distilled
# from a ViT-giant teacher, pretrained at 384px. Our domain-adaptive
# pretraining (scripts/pretrain_ssl.sh) and all supervised fine-tuning
# (scripts/train_supervised.sh) run at 224px/16 frames per clip, not 384px
# -- see video/README.md ("Model variant") for why.
#
# Usage: bash video/scripts/setup_vjepa.sh
# Env vars (all optional, see video/README.md for the full table):
#   VJEPA2_REPO            default ./vjepa2
#   VJEPA2_COMMIT          default 204698b45b3712590f06245fbfba32d3be539812
#   AI4EYES_CHECKPOINT_DIR default ./checkpoints
set -euo pipefail

VJEPA2_REPO="${VJEPA2_REPO:-./vjepa2}"
VJEPA2_COMMIT="${VJEPA2_COMMIT:-204698b45b3712590f06245fbfba32d3be539812}"
AI4EYES_CHECKPOINT_DIR="${AI4EYES_CHECKPOINT_DIR:-./checkpoints}"
CHECKPOINT_URL="https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"
CHECKPOINT_NAME="vjepa2_1_vitl_dist_vitG_384.pt"

echo "== Cloning facebookresearch/vjepa2 @ ${VJEPA2_COMMIT} into ${VJEPA2_REPO} =="
if [[ -d "${VJEPA2_REPO}/.git" ]]; then
    echo "Already cloned at ${VJEPA2_REPO}; fetching and checking out the pinned commit."
    git -C "${VJEPA2_REPO}" fetch origin
else
    git clone https://github.com/facebookresearch/vjepa2.git "${VJEPA2_REPO}"
fi
git -C "${VJEPA2_REPO}" checkout "${VJEPA2_COMMIT}"

echo "== Applying patches (see patches/README.md) =="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for patch in "${SCRIPT_DIR}/../patches"/*.patch; do
    [[ -e "${patch}" ]] || continue
    if git -C "${VJEPA2_REPO}" apply --reverse --check "${patch}" 2>/dev/null; then
        echo "Already applied: $(basename "${patch}")"
    else
        git -C "${VJEPA2_REPO}" apply "${patch}"
        echo "Applied: $(basename "${patch}")"
    fi
done

echo "== Installing vjepa2's own requirements =="
pip install -r "${VJEPA2_REPO}/requirements.txt"

echo "== Downloading official checkpoint: ${CHECKPOINT_NAME} =="
mkdir -p "${AI4EYES_CHECKPOINT_DIR}"
DEST="${AI4EYES_CHECKPOINT_DIR}/${CHECKPOINT_NAME}"
if [[ -f "${DEST}" ]]; then
    echo "Already present: ${DEST}"
else
    curl -fL "${CHECKPOINT_URL}" -o "${DEST}.part"
    mv "${DEST}.part" "${DEST}"
fi

echo
echo "Done."
echo "VJEPA2_REPO=${VJEPA2_REPO}"
echo "VJEPA2_CHECKPOINT=${DEST}"
echo "Export these (or rely on the matching defaults) before running the other scripts."
