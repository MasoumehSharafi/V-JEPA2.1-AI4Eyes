"""Builds the unlabelled manifest (manifest.py schema) and a per-test
annotations.json (annotations.py schema) from a raw clinical export CSV
and the shared, authoritative split CSV. The split is READ, never redrawn
(see manifest.read_split_csv) -- this script has no random-sampling
train/val/test logic at all.

Because the raw export's exact columns change over time, column names are
CLI arguments rather than hardcoded; defaults match the schema used in this
project's most recent export. Run with --dry-run to validate without
writing anything.

Two label modes, matching the two shapes of ground truth this project has:
  --label-mode single-segment   one optional (start,end) event per video
                                 (NITBUT, Invasive-TBUT: a gradable video has
                                 exactly one event; "ungradable" = none)
  --label-mode point-events     a JSON column of {"time": t, "label": name}
                                 keyframes, each converted into a fixed-width
                                 pseudo-segment centered on it (Blink
                                 Mechanics: point-in-time blink events, no
                                 true start/end in the source annotation)

Usage:
    python -m ai4eyes_video.prepare_data \\
        --raw-csv /path/to/latest_export.csv --split-csv /path/to/split.csv \\
        --test-type NITBUT --label-mode single-segment \\
        --out-manifest data/manifest_nitbut.csv --out-annotations data/annotations/nitbut.json
"""
import argparse
import csv
import json
from pathlib import Path

from .manifest import VALID_SPLITS, ManifestError, read_split_csv, validate_od_os_rule
from .manifest import MediaItem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-csv", required=True, help="Raw clinical export CSV (never committed)")
    parser.add_argument("--split-csv", required=True, help="Shared, authoritative split CSV (key,split columns)")
    parser.add_argument("--test-type", required=True)
    parser.add_argument("--label-mode", required=True, choices=["single-segment", "point-events"])
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--out-annotations", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate only; write nothing")

    # Column-name mapping (defaults match the most recent export schema).
    parser.add_argument("--col-test-type", default="test_type")
    parser.add_argument("--col-patient-id", default="patient_pid")
    parser.add_argument("--col-exam-id", default="exam_id")
    parser.add_argument("--col-eye", default="eye")
    parser.add_argument("--col-eye-status", default="eye_status")
    parser.add_argument("--col-media-path", default="media_relative_path")
    parser.add_argument("--col-fps", default="fps")
    parser.add_argument("--col-duration", default="duration_seconds")
    parser.add_argument("--col-segment-start", default="gt_start_seconds")
    parser.add_argument("--col-segment-end", default="gt_end_seconds")
    parser.add_argument("--col-events-json", default="events_json",
                         help="point-events mode: JSON list of {time, label}")
    parser.add_argument("--event-window-seconds", type=float, default=0.3,
                         help="point-events mode: pseudo-segment width centered on each event")
    parser.add_argument("--classes", nargs="+", default=None,
                         help="point-events mode: explicit class list (order defines label indices)")
    return parser


def _key(exam_id: str, eye: str) -> str:
    return f"{exam_id}-{eye}"


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    split_by_key = read_split_csv(args.split_csv)

    with open(args.raw_csv, newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r[args.col_test_type].strip() == args.test_type]
    if not rows:
        raise ManifestError(f"No rows with {args.col_test_type}={args.test_type!r} in {args.raw_csv}")

    manifest_items: list[MediaItem] = []
    database: dict[str, dict] = {}
    classes: list[str] = args.classes or ([args.test_type] if args.label_mode == "single-segment" else [])

    for row in rows:
        exam_id = row[args.col_exam_id].strip()
        eye = row[args.col_eye].strip()
        patient_id = row[args.col_patient_id].strip()
        key = _key(exam_id, eye)
        split = split_by_key.get(key)
        if split is None:
            raise ManifestError(f"Key {key!r} not found in split CSV {args.split_csv} -- the split must be read, not re-derived")
        if split not in VALID_SPLITS:
            raise ManifestError(f"Unexpected split {split!r} for key {key!r}")

        relative_path = row[args.col_media_path].strip()
        manifest_items.append(
            MediaItem(
                key=key, media_type="video", test_type=args.test_type, patient_id=patient_id,
                exam_id=exam_id, eye=eye, split=split, relative_path=relative_path,
            )
        )

        duration = float(row[args.col_duration])
        fps = float(row[args.col_fps])
        annotations = []
        if args.label_mode == "single-segment":
            is_gradable = row.get(args.col_eye_status, "").strip().upper() == "COMPLETE"
            if is_gradable:
                start = float(row[args.col_segment_start])
                end = float(row[args.col_segment_end])
                annotations.append({"segment": [start, end], "label": args.test_type})
        else:  # point-events
            events = json.loads(row[args.col_events_json]) if row.get(args.col_events_json) else []
            half = args.event_window_seconds / 2
            for event in events:
                label = event["label"]
                if label not in classes:
                    raise ManifestError(f"Event label {label!r} not in --classes {classes}")
                center = float(event["time"])
                annotations.append({"segment": [max(0.0, center - half), min(duration, center + half)], "label": label})

        database[key] = {
            "duration": duration, "fps": fps, "split": split, "patient_id": patient_id,
            "exam_id": exam_id, "eye": eye, "relative_path": relative_path, "annotations": annotations,
        }

    validate_od_os_rule(manifest_items)
    print(f"Validated {len(manifest_items)} videos, {len({m.patient_id for m in manifest_items})} patients -- OD/OS split-integrity OK")

    if args.dry_run:
        print("--dry-run: nothing written")
        return

    manifest_path = Path(args.out_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key", "media_type", "test_type", "patient_id", "exam_id", "eye", "split", "relative_path"])
        writer.writeheader()
        for item in manifest_items:
            writer.writerow(item.__dict__)
    print(f"Wrote manifest: {manifest_path}")

    annotations_path = Path(args.out_annotations)
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    annotations_path.write_text(
        json.dumps({"version": "ai4eyes-video-1.0", "test_type": args.test_type, "classes": classes, "database": database}, indent=2)
    )
    print(f"Wrote annotations: {annotations_path}")


if __name__ == "__main__":
    main()
