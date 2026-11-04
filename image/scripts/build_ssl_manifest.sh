#!/usr/bin/env bash
# SSL manifest: every unlabelled image of every image sign (labelled exams excluded) → $AI4EYES_SSL_MANIFEST.
# Reads S3 only; lipid-layer videos are downloaded to $AI4EYES_WORK_DIR for the lower-lid frames.
#   scripts/build_ssl_manifest.sh [--types staining meniscus meibography lid_margin lower_lid]
set -euo pipefail
. "$(dirname "$0")/_env.sh"
step "SSL manifest"
$PY -m ai4eyes_image.pretrain.manifest "$@"
