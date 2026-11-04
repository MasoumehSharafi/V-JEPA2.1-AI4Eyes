"""Metric function tests against tiny hand-computed synthetic examples.
No data, no GPU, no files -- everything is built in-memory."""
import pytest

from ai4eyes_video.annotations import AnnotationSet, VideoRecord
from ai4eyes_video.metrics import count_metrics, gradability_metrics, temporal_iou


def test_temporal_iou_basic():
    assert temporal_iou([0, 10], [0, 10]) == pytest.approx(1.0)
    assert temporal_iou([0, 10], [20, 30]) == pytest.approx(0.0)
    assert temporal_iou([0, 2], [1, 3]) == pytest.approx(1 / 3)


def _nitbut_set():
    records = {
        "v1": VideoRecord("v1", 10.0, 30.0, "test", "p1", "e1", "OD", "v1.mp4",
                           [{"segment": [2.0, 5.0], "label": "NITBUT"}]),
        "v2": VideoRecord("v2", 10.0, 30.0, "test", "p2", "e2", "OD", "v2.mp4", []),
        "v3": VideoRecord("v3", 10.0, 30.0, "test", "p3", "e3", "OD", "v3.mp4",
                           [{"segment": [1.0, 3.0], "label": "NITBUT"}]),
    }
    return AnnotationSet(test_type="NITBUT", classes=["NITBUT"], records=records)


def test_gradability_metrics():
    gt = _nitbut_set()
    predictions = {
        "v1": [{"segment": [2.1, 4.9], "score": 0.8}],
        "v2": [],  # correctly predicted ungradable
        "v3": [{"segment": [0.5, 1.0], "score": 0.05}],  # below threshold -> predicted ungradable (a miss)
    }
    report = gradability_metrics(gt, predictions, split="test", score_threshold=0.1)

    assert report["n"] == 3
    assert report["tp"] == 1 and report["tn"] == 1 and report["fn"] == 1 and report["fp"] == 0
    assert report["accuracy"] == pytest.approx(2 / 3)
    assert report["precision"] == pytest.approx(1.0)
    assert report["recall"] == pytest.approx(0.5)
    assert report["f1"] == pytest.approx(2 / 3)
    assert report["mean_tiou"] == pytest.approx(2.8 / 3.0, abs=1e-6)
    assert report["start_mae"] == pytest.approx(0.1, abs=1e-6)
    assert report["end_mae"] == pytest.approx(0.1, abs=1e-6)
    assert report["duration_mae"] == pytest.approx(0.2, abs=1e-6)


def test_gradability_metrics_all_ungradable_has_no_tiou_keys():
    records = {"v1": VideoRecord("v1", 10.0, 30.0, "test", "p1", "e1", "OD", "v1.mp4", [])}
    gt = AnnotationSet(test_type="NITBUT", classes=["NITBUT"], records=records)
    report = gradability_metrics(gt, {"v1": []}, split="test")
    assert "mean_tiou" not in report
    assert report["accuracy"] == pytest.approx(1.0)


def _blink_set():
    records = {
        "v1": VideoRecord(
            "v1", 10.0, 30.0, "test", "p1", "e1", "OD", "v1.mp4",
            [
                {"segment": [1.0, 2.0], "label": "Blink Complete"},
                {"segment": [5.0, 6.0], "label": "Blink Complete"},
                {"segment": [3.0, 4.0], "label": "Blink Incomplete"},
            ],
        )
    }
    return AnnotationSet(test_type="BLINK_MECHANICS", classes=["Blink Complete", "Blink Incomplete"], records=records)


def test_count_metrics():
    gt = _blink_set()
    predictions = {
        "v1": [
            {"segment": [1.1, 2.1], "label": 0, "score": 0.9},  # matches gt [1,2] (Complete)
            {"segment": [8.0, 9.0], "label": 0, "score": 0.5},  # no match -> FP (Complete)
            {"segment": [3.1, 4.1], "label": 1, "score": 0.8},  # matches gt [3,4] (Incomplete)
        ]
    }
    report = count_metrics(gt, predictions, split="test", score_threshold=0.1, iou_threshold=0.5)

    complete = report["per_class"]["Blink Complete"]
    assert complete["tp"] == 1 and complete["fp"] == 1 and complete["fn"] == 1
    assert complete["precision"] == pytest.approx(0.5)
    assert complete["recall"] == pytest.approx(0.5)
    assert complete["count_exact_match"] == pytest.approx(1.0)  # gt=2, pred=2 -> exact match even though not perfectly matched

    incomplete = report["per_class"]["Blink Incomplete"]
    assert incomplete["tp"] == 1 and incomplete["fp"] == 0 and incomplete["fn"] == 0
    assert incomplete["precision"] == pytest.approx(1.0)
    assert incomplete["recall"] == pytest.approx(1.0)
    assert incomplete["count_exact_match"] == pytest.approx(1.0)


def test_count_metrics_below_threshold_is_excluded():
    gt = _blink_set()
    predictions = {"v1": [{"segment": [1.1, 2.1], "label": 0, "score": 0.01}]}
    report = count_metrics(gt, predictions, split="test", score_threshold=0.1, iou_threshold=0.5)
    complete = report["per_class"]["Blink Complete"]
    assert complete["tp"] == 0 and complete["fn"] == 2  # the low-score prediction never counted
