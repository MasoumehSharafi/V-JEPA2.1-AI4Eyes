"""Converts our documented manifest (manifest.py schema) into the plain
space-delimited "<path> <label>" format the official vjepa2 trainer
expects for SSL pretraining (label is always 0 -- unlabeled). Excludes
the `test` split by construction (SSL pretraining must never see
downstream-eval patients -- the equivalent of the leakage guard used for
the original DAPT runs) and supports excluding an explicit extra patient
list for defense in depth.

Usage:
    python -m ai4eyes_video.export_vjepa_manifest \\
        --manifest data/manifest.csv --video-root data/videos \\
        --test-type NITBUT --out data/vjepa_train.csv
"""
import argparse
from pathlib import Path

from .manifest import filter_manifest, read_manifest, resolve_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--test-type", default=None, help="Restrict to one test_type; omit to use all (joint pretraining)")
    parser.add_argument("--media-type", default="video", choices=["video", "image"])
    parser.add_argument("--exclude-patients-file", default=None,
                         help="Optional: a text file, one patient_id per line, to additionally exclude")
    parser.add_argument("--out", required=True)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    items = read_manifest(args.manifest)
    items = filter_manifest(items, test_type=args.test_type, media_type=args.media_type)

    excluded = set()
    if args.exclude_patients_file:
        excluded = {line.strip() for line in Path(args.exclude_patients_file).read_text().splitlines() if line.strip()}

    kept = [i for i in items if i.split != "test" and i.patient_id not in excluded]
    dropped_test = sum(1 for i in items if i.split == "test")
    dropped_excluded = sum(1 for i in items if i.split != "test" and i.patient_id in excluded)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        for item in kept:
            handle.write(f"{resolve_path(item, args.video_root)} 0\n")

    print(f"Wrote {len(kept)} rows to {out_path} "
          f"(excluded {dropped_test} test-split rows, {dropped_excluded} additionally-excluded-patient rows)")


if __name__ == "__main__":
    main()
