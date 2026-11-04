import pandas as pd

from ai4eyes_image.signs import SIGNS, labels, lower_lid_group, upper_lid_group


def test_lid_groups_first_match_wins():
    assert upper_lid_group(["None"]) == 0
    assert upper_lid_group(["Collarettes", "Lid margin telangiectasia"]) == 1      # blepharitis before vascular
    assert upper_lid_group(["Lid margin hyperemia"]) == 2
    assert upper_lid_group(["makeup debris"]) == 3
    assert lower_lid_group(["Orifice capping"]) == 1
    assert lower_lid_group(["Inflamed margin"]) == 2


def test_every_sign_labels(labels_df):
    for s, sign in SIGNS.items():
        d = labels(s, labels_df)
        assert len(d) and d.sample_id.is_unique and d.image_path.notna().all()
        if sign.kind == "classification":
            assert set(d.y) <= set(range(len(sign.classes))), s


def test_rejected_eyes(labels_df):
    rej = set(labels_df.sample_id[labels_df.eye_status == "REJECTED"])
    st = labels("corneal_staining", labels_df)
    assert (st[st.sample_id.isin(rej)].y == 2).all()                 # ungradable joins class 2 (≥2 / ungradable)
    assert not labels("hyperemia_nasal", labels_df).sample_id.isin(rej).any()   # dropped
    assert (labels("meibography_lower", labels_df).set_index("sample_id").y.reindex(sorted(rej)).dropna() == 3).all()


def test_out_of_map_grades_join_end_class():
    df = pd.DataFrame({"sample_id": ["a", "b"], "patient_pid": ["p", "p"], "exam_id": ["e", "e"], "eye": ["OD", "OS"],
                       "test_type": "HYPEREMIA", "eye_status": "COMPLETE", "media_s3_key": ["e/x_OD.png", "e/x_OS.png"],
                       "hyperemia_nasal_grade": [0.0, 4.0]})
    assert labels("hyperemia_nasal", df).y.tolist() == [0, 2]        # nasal 0 → class "1", 4 → ">=3"
