"""The 8 image signs: which test type, which label column, how raw grades become classes, which input crop.

`labels(sign, df)` turns the labels CSV into one row per label row of that sign (an eye graded by two labellers
appears twice), with `y` (class index, or height in mm for tear meniscus) and `image_path`.
"""
import json
from dataclasses import dataclass

import numpy as np

from ai4eyes_image import paths


@dataclass(frozen=True)
class Sign:
    name: str
    test_type: str
    label_col: str
    kind: str                 # "classification" | "regression"
    crop: str                 # "cornea" (staining: cornea crop, ring inpainted) | "fullframe"
    classes: tuple = ()
    grade_to_class: dict = None
    reject_to: int = None     # class for REJECTED (ungradable) eyes; None = REJECTED eyes are dropped
    image_type: str = ""      # SSL image type of its photos (ssl manifest `image_type`)


# Raw 0-5 staining grade, 6 = REJECTED → 3 classes: 0 / 1 / (≥2 or ungradable)
STAINING_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2}

SIGNS = {s.name: s for s in [
    Sign("corneal_staining", "CORNEAL_STAINING", "corneal_staining_van_bijsterveld", "classification", "cornea",
         ("0", "1", ">=2/ungr"), STAINING_TO_CLASS, image_type="staining"),
    Sign("hyperemia_nasal", "HYPEREMIA", "hyperemia_nasal_grade", "classification", "fullframe",
         ("1", "2", ">=3"), {1: 0, 2: 1, 3: 2, 4: 2}, image_type="meniscus"),
    Sign("hyperemia_temporal", "HYPEREMIA", "hyperemia_temporal_grade", "classification", "fullframe",
         ("<=1", "2", "3"), {0: 0, 1: 0, 2: 1, 3: 2}, image_type="meniscus"),
    Sign("meibography_lower", "MEIBOGRAPHY_LOWER", "meibography_meiboscale", "classification", "fullframe",
         ("0", "1", "2", "3+4+ungr"), {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}, reject_to=3, image_type="meibography"),
    Sign("meibography_upper", "MEIBOGRAPHY_UPPER", "meibography_meiboscale", "classification", "fullframe",
         ("0+1", "2", "3+ungr"), {0: 0, 1: 0, 2: 1, 3: 2}, reject_to=2, image_type="meibography"),
    Sign("upper_lid_margin", "UPPER_LID_MARGIN", "upper_lid_margin_conditions", "classification", "fullframe",
         ("none", "blepharitis", "vascular", "other"), image_type="lid_margin"),
    Sign("lower_lid_margin", "LOWER_LID_MARGIN", "lower_lid_margin_conditions", "classification", "fullframe",
         ("none", "MGD/blepharitis", "vascular", "other"), image_type="lower_lid"),
    Sign("tear_meniscus", "MENISCUS", "meniscus_height_mm", "regression", "fullframe", image_type="meniscus"),
]}

# Lid margin: multi-label condition lists → 4 groups; an eye takes the first group that applies (1 > 2 > 3).
_UPPER_BLEPH = ("collarette", "blephar", "scal", "flaky", "demodex")
_UPPER_VASC = ("telangiect", "hyperemia")
_LOWER_MGD = ("capping", "collarette", "blephar", "scal", "flaky", "demodex")
_LOWER_VASC = ("telangiect", "hyperemia", "inflam")

# The lower lid margin is graded on one frame of the lipid-layer video (ai4eyes_image.data.frames)
LOWER_LID_FRAME = "lower-lid-margin_{eye}.png"


def _lid_group(conditions, group1, group2):
    c = [x.lower() for x in conditions]
    if c == ["none"]:
        return 0
    if any(k in x for x in c for k in group1):
        return 1
    if any(k in x for x in c for k in group2):
        return 2
    return 3


def upper_lid_group(conditions):
    return _lid_group(conditions, _UPPER_BLEPH, _UPPER_VASC)


def lower_lid_group(conditions):
    return _lid_group(conditions, _LOWER_MGD, _LOWER_VASC)


def labels(sign, df):
    """Label rows of one sign from the labels CSV → copy with `label_raw`, `y`, `image_path`."""
    s = SIGNS[sign]
    rej = df.eye_status == "REJECTED"
    if s.name == "corneal_staining":
        d = df[df.test_type == s.test_type].copy()
        d = d[(d.eye_status == "REJECTED") | d[s.label_col].notna()]
        raw = np.where(d.eye_status == "REJECTED", 6, d[s.label_col].fillna(-1)).astype(int)
        d["label_raw"] = np.where(raw == 6, "ungradable", raw.astype(str))
        d["y"] = raw
        d["y"] = d.y.map(s.grade_to_class)
    elif s.kind == "regression":
        d = df[(df.test_type == s.test_type) & ~rej & df[s.label_col].notna()].copy()
        d["label_raw"], d["y"] = d[s.label_col], d[s.label_col].astype(float)
    elif s.name in ("upper_lid_margin", "lower_lid_margin"):
        group = upper_lid_group if s.name == "upper_lid_margin" else lower_lid_group
        d = df[(df.test_type == s.test_type) & ~rej & df[s.label_col].notna()].copy()
        d["label_raw"] = d[s.label_col]
        d["y"] = d.label_raw.map(lambda v: group(json.loads(v)))
        if s.name == "lower_lid_margin":
            d["image_path"] = [str(paths.IMAGES_DIR / e / LOWER_LID_FRAME.format(eye=y)) for e, y in zip(d.exam_id, d.eye)]
    else:   # graded signs (hyperemia, meibography)
        d = df[df.test_type == s.test_type].copy()
        r = d.eye_status == "REJECTED"
        d = d[r | d[s.label_col].notna()]
        r = d.eye_status == "REJECTED"
        if s.reject_to is None:
            d, r = d[~r], r[~r]
        d["label_raw"] = np.where(r, "ungradable", d[s.label_col].astype(str))
        # a grade outside the map (e.g. nasal 0, upper meibography 4) joins the nearest end class
        g2c = s.grade_to_class
        lo, hi = min(g2c), max(g2c)
        d["y"] = [s.reject_to if x else g2c[min(max(int(v), lo), hi)] for x, v in zip(r, d[s.label_col])]
    if "image_path" not in d:
        d["image_path"] = [str(paths.IMAGES_DIR / k) for k in d.media_s3_key]
    return d.reset_index(drop=True)
