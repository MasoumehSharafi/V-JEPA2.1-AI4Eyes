import pandas as pd

from ai4eyes_image.data import split as sp


def test_split_is_patient_level_and_deterministic(labels_df, monkeypatch):
    monkeypatch.setattr(sp, "N_DRAWS", 20)
    a, b = sp.make_split(labels_df), sp.make_split(labels_df)
    assert a.equals(b)
    assert a.patient_pid.is_unique                                     # one split per patient → OD/OS together
    assert a.split.value_counts().to_dict() == {"train": 30, "test": 60, "val": 30}
    eyes = labels_df[["patient_pid", "exam_id", "eye"]].drop_duplicates().merge(a, on="patient_pid")
    assert (eyes.groupby("exam_id").split.nunique() == 1).all()
