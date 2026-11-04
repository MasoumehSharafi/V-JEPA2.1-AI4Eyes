"""Manifest schema + clinical-rule tests. No data, no GPU -- everything
here is tiny synthetic CSV content written to a pytest tmp_path."""
import csv

import pytest

from ai4eyes_video.manifest import ManifestError, read_manifest, read_split_csv, validate_od_os_rule, MediaItem

COLUMNS = ["key", "media_type", "test_type", "patient_id", "exam_id", "eye", "split", "relative_path"]


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def row(key, patient="p1", exam="e1", eye="OD", split="train", media_type="video", test_type="NITBUT"):
    return {
        "key": key, "media_type": media_type, "test_type": test_type, "patient_id": patient,
        "exam_id": exam, "eye": eye, "split": split, "relative_path": f"videos/{key}.mp4",
    }


def test_valid_manifest_parses(tmp_path):
    path = tmp_path / "manifest.csv"
    write_csv(path, [row("e1-OD", eye="OD"), row("e1-OS", eye="OS")])
    items = read_manifest(str(path))
    assert len(items) == 2
    assert items[0].key == "e1-OD"
    assert items[0].media_type == "video"


def test_missing_column_raises(tmp_path):
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[c for c in COLUMNS if c != "split"])
        writer.writeheader()
    with pytest.raises(ManifestError, match="missing columns"):
        read_manifest(str(path))


def test_invalid_split_raises(tmp_path):
    path = tmp_path / "manifest.csv"
    write_csv(path, [row("e1-OD", split="bogus")])
    with pytest.raises(ManifestError, match="split must be one of"):
        read_manifest(str(path))


def test_invalid_media_type_raises(tmp_path):
    path = tmp_path / "manifest.csv"
    write_csv(path, [row("e1-OD", media_type="audio")])
    with pytest.raises(ManifestError, match="media_type must be one of"):
        read_manifest(str(path))


def test_absolute_path_rejected(tmp_path):
    path = tmp_path / "manifest.csv"
    bad = row("e1-OD")
    bad["relative_path"] = "/abs/path.mp4"
    write_csv(path, [bad])
    with pytest.raises(ManifestError, match="must be relative"):
        read_manifest(str(path))


def test_od_os_same_split_is_fine():
    items = [
        MediaItem("e1-OD", "video", "NITBUT", "p1", "e1", "OD", "train", "a.mp4"),
        MediaItem("e1-OS", "video", "NITBUT", "p1", "e1", "OS", "train", "b.mp4"),
    ]
    validate_od_os_rule(items)  # should not raise


def test_od_os_split_violation_raises():
    items = [
        MediaItem("e1-OD", "video", "NITBUT", "p1", "e1", "OD", "train", "a.mp4"),
        MediaItem("e1-OS", "video", "NITBUT", "p1", "e1", "OS", "test", "b.mp4"),
    ]
    with pytest.raises(ManifestError, match="OD/OS split-integrity"):
        validate_od_os_rule(items)


def test_read_manifest_enforces_od_os_rule(tmp_path):
    path = tmp_path / "manifest.csv"
    write_csv(path, [row("e1-OD", eye="OD", split="train"), row("e1-OS", eye="OS", split="test")])
    with pytest.raises(ManifestError, match="OD/OS split-integrity"):
        read_manifest(str(path))


def test_read_split_csv(tmp_path):
    path = tmp_path / "split.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key", "split"])
        writer.writeheader()
        writer.writerow({"key": "e1-OD", "split": "test"})
        writer.writerow({"key": "e1-OS", "split": "test"})
    assignments = read_split_csv(str(path))
    assert assignments == {"e1-OD": "test", "e1-OS": "test"}


def test_read_split_csv_rejects_invalid_split(tmp_path):
    path = tmp_path / "split.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key", "split"])
        writer.writeheader()
        writer.writerow({"key": "e1-OD", "split": "bogus"})
    with pytest.raises(ManifestError):
        read_split_csv(str(path))
