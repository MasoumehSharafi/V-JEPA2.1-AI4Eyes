"""SSL manifest: every unlabelled image of every image sign, pooled, with its crop box. Metadata only.

    python -m ai4eyes_image.pretrain.manifest [--types staining meniscus ...]   → $AI4EYES_SSL_MANIFEST

Schema (one row per image; a joint image + video pretraining can concatenate manifests of this form):
  key          S3 key, or a relative id for local files
  exam_id, eye
  split        train / val by an md5 hash of exam_id (val_pct 5; unused by V-JEPA SSL, which trains on all rows)
  ring_ok, ring_cx, ring_cy, ring_r, ring_dots   fixation-ring fit (staining only; inpainted at load time)
  crop_ok, box_x0, box_y0, box_side              square crop box in native pixels (may leave the frame: zero-pad)
  image_type   staining | meniscus | meibography | lid_margin | lower_lid
  local_path   set for files read from disk instead of S3 (lower-lid frames extracted from lipid-layer videos)

Every exam in the labels CSV is excluded, so no labelled train / val / test image is ever seen in SSL. Unlabelled
lipid-layer videos are downloaded to $AI4EYES_WORK_DIR/ssl_videos and one frame per video (LOWER_LID_T s) is written
to $AI4EYES_DATA_DIR/ssl_frames/lower_lid/.
"""
import argparse
import hashlib
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd

from ai4eyes_image import paths
from ai4eyes_image.data.frames import read_frame
from ai4eyes_image.data.s3 import list_keys, s3_client, s3_read_image
from ai4eyes_image.preprocess.masks import cornea_mask, fit_ring
from ai4eyes_image.preprocess.transforms import square_crop_box

log = logging.getLogger("manifest")
VAL_PCT = 5
LOWER_LID_T = 1.2          # s
COLUMNS = ["key", "exam_id", "eye", "split", "ring_ok", "ring_cx", "ring_cy", "ring_r", "ring_dots", "crop_ok",
           "box_x0", "box_y0", "box_side", "image_type", "local_path"]
# image type → (S3 file names, crop)
TYPES = {
    "staining": (["corneal-staining_OD.png", "corneal-staining_OS.png"], "cornea"),
    "meniscus": (["meniscus_OD.png", "meniscus_OS.png"], "fullframe"),
    "meibography": (["meibography-lower_OD.png", "meibography-lower_OS.png",
                     "meibography-upper_OD.png", "meibography-upper_OS.png"], "fullframe"),
    "lid_margin": (["lid-margin_OD.png", "lid-margin_OS.png"], "fullframe"),
    "lower_lid": (["lipid-layer_OD.mp4", "lipid-layer_OS.mp4"], "video_frame"),
}


def split_for(exam_id, val_pct=VAL_PCT):
    return "val" if int(hashlib.md5(exam_id.encode()).hexdigest(), 16) % 100 < val_pct else "train"


def fullframe_box(h, w):
    """Square box around the whole frame — the same box the downstream full-frame bank crops."""
    return square_crop_box(np.ones((h, w), bool))


def no_ring():
    return {"ring_ok": False, "ring_cx": np.nan, "ring_cy": np.nan, "ring_r": np.nan, "ring_dots": 0}


def image_row(bucket, key, crop_mode, _client={}):
    client = _client.setdefault("c", s3_client())
    exam_id, name = key.rsplit("/", 1)
    row = {"key": key, "exam_id": exam_id, "eye": name[-6:-4], "split": split_for(exam_id)}
    try:
        img = s3_read_image(client, bucket, key)
    except IOError as e:          # e.g. an MP4 stored under a .png name — skip it
        return {"key": key, "skip_reason": str(e)}
    h, w = img.shape[:2]
    if crop_mode == "fullframe":
        x0, y0, side = fullframe_box(h, w)
        return {**row, **no_ring(), "crop_ok": True, "box_x0": x0, "box_y0": y0, "box_side": side}
    ring = fit_ring(img)
    row.update(ring_ok=ring is not None, ring_cx=ring.center[0] if ring else np.nan,
               ring_cy=ring.center[1] if ring else np.nan, ring_r=ring.radius if ring else np.nan,
               ring_dots=ring.n_dots if ring else 0)
    box, ok = None, False
    if ring is not None:
        cmask, info = cornea_mask(img, ring)
        if cmask.any() and info.get("ok"):
            box, ok = square_crop_box(cmask), True
    if box is None:  # fallback: centre square of the frame
        side = min(h, w)
        box = ((w - side) // 2, (h - side) // 2, side)
    return {**row, "crop_ok": ok, "box_x0": box[0], "box_y0": box[1], "box_side": box[2]}


def frame_row(bucket, key, _client={}):
    """Unlabelled lipid-layer video → one lower-lid frame on disk → manifest row with local_path."""
    client = _client.setdefault("c", s3_client())
    exam_id, name = key.split("/")
    eye = name[-6:-4]
    vid = paths.WORK_DIR / "ssl_videos" / exam_id / name
    if not vid.exists():
        vid.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, key, str(vid))
    rel = f"{exam_id}/lower-lid_{eye}_t{LOWER_LID_T:.1f}.png"
    p = paths.DATA_DIR / "ssl_frames" / "lower_lid" / rel
    if not p.exists():
        img = read_frame(vid, LOWER_LID_T)
        if img is None:
            return {"key": key, "skip_reason": "unreadable video"}
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), img)
    h, w = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).shape
    x0, y0, side = fullframe_box(h, w)
    return {"key": f"lower_lid_frames/{rel}", "exam_id": exam_id, "eye": eye, "split": split_for(exam_id), **no_ring(),
            "crop_ok": True, "box_x0": x0, "box_y0": y0, "box_side": side, "local_path": str(p)}


def build(types):
    bucket = paths.require_bucket()
    labelled = set(pd.read_csv(paths.LABELS_CSV, usecols=["exam_id"]).exam_id)
    parts = []
    for t in types:
        names, mode = TYPES[t]
        keys = [k for k in list_keys(bucket, names) if k.rsplit("/", 1)[0] not in labelled]
        t0 = time.time()
        with ThreadPoolExecutor(16) as ex:
            if mode == "video_frame":
                rows = list(ex.map(lambda k: frame_row(bucket, k), keys))
            else:
                rows = list(ex.map(lambda k: image_row(bucket, k, mode), keys))
        skipped = [r for r in rows if "skip_reason" in r]
        m = pd.DataFrame([r for r in rows if "skip_reason" not in r]).assign(image_type=t)
        log.info("%s: %d unlabelled files, %d rows, %d skipped (%.0f s)", t, len(keys), len(m), len(skipped),
                 time.time() - t0)
        parts.append(m)
    m = pd.concat(parts, ignore_index=True).reindex(columns=COLUMNS)
    assert m.key.is_unique and not set(m.exam_id) & labelled, "a labelled exam is in the SSL manifest"
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", nargs="*", default=list(TYPES), choices=list(TYPES))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    m = build(a.types)
    paths.SSL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(paths.SSL_MANIFEST, index=False)
    log.info("%d images, %d exams → %s\n%s", len(m), m.exam_id.nunique(), paths.SSL_MANIFEST,
             m.image_type.value_counts().to_string())


if __name__ == "__main__":
    main()
