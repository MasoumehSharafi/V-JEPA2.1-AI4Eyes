import json

import numpy as np
import pandas as pd
import pytest

TESTS = {   # test type → (label column, value maker)
    "CORNEAL_STAINING": ("corneal_staining_van_bijsterveld", lambda r: float(r.integers(0, 4))),
    "HYPEREMIA": (None, None),
    "MENISCUS": ("meniscus_height_mm", lambda r: float(r.uniform(0.15, 0.6))),
    "MEIBOGRAPHY_LOWER": ("meibography_meiboscale", lambda r: float(r.integers(0, 5))),
    "MEIBOGRAPHY_UPPER": ("meibography_meiboscale", lambda r: float(r.integers(0, 4))),
    "UPPER_LID_MARGIN": ("upper_lid_margin_conditions",
                         lambda r: json.dumps([["None"], ["Collarettes"], ["Lid margin telangiectasia"], ["makeup"]][r.integers(0, 4)])),
    "LOWER_LID_MARGIN": ("lower_lid_margin_conditions",
                         lambda r: json.dumps([["None"], ["Orifice capping"], ["Inflamed margin"], ["Lash loss"]][r.integers(0, 4)])),
}


@pytest.fixture
def labels_df():
    """Synthetic labels CSV: 120 patients × OD/OS × every image test type (no real data)."""
    r = np.random.default_rng(0)
    rows = []
    for p in range(120):
        exam = f"exam{p:03d}"
        for eye in ("OD", "OS"):
            for tt, (col, make) in TESTS.items():
                row = {"sample_id": f"{exam}-{eye}-{tt}", "patient_pid": f"pat{p:03d}", "exam_id": exam, "eye": eye,
                       "test_type": tt, "eye_status": "REJECTED" if r.random() < 0.05 else "COMPLETE",
                       "media_s3_key": f"{exam}/{tt.lower()}_{eye}.png"}
                if tt == "HYPEREMIA":
                    row["hyperemia_nasal_grade"], row["hyperemia_temporal_grade"] = float(r.integers(1, 5)), float(r.integers(0, 4))
                elif row["eye_status"] == "COMPLETE":
                    row[col] = make(r)
                rows.append(row)
    return pd.DataFrame(rows)
