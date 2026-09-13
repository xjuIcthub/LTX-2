#!/usr/bin/env python3
"""Validate a task directory's prompt CSV and optional output layout.

Examples::

    uv run python tasks/validate.py sep14
    uv run python tasks/validate.py sep14 --check-videos
"""

from __future__ import annotations

# ruff: noqa: T201 -- this CLI prints a human-readable validation report.
import argparse
import json
import sys

try:
    from ._common import TaskFormatError, validate_task_dir
except ImportError:  # pragma: no cover - supports ``python tasks/validate.py``
    from _common import TaskFormatError, validate_task_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", help="Task directory name or path, for example sep14")
    parser.add_argument(
        "--check-videos",
        action="store_true",
        help="Require one output directory per CSV and video_000.mp4-style files for every row",
    )
    parser.add_argument("--json", action="store_true", help="Print a machine-readable JSON report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = validate_task_dir(args.task, check_videos=args.check_videos)
    except TaskFormatError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(
            f"{'OK' if report.ok else 'FAILED'}: {report.task_dir} | "
            f"csv={report.csv_count} rows={report.row_count} "
            f"video_dirs={report.video_dir_count} videos={report.video_count}"
        )
        for name, rows in report.files:
            print(f"  {name}: {rows} rows")
        for error in report.errors:
            print(f"ERROR: {error}", file=sys.stderr)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
