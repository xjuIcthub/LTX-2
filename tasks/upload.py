#!/usr/bin/env python3
"""Upload task CSVs and generated MP4s to ``xjuIcthub/tasks`` on Hugging Face.

The local directory ``tasks/sep14`` is uploaded at ``sep14/`` in the dataset
repository, so the remote layout is, for example,
``sep14/spatial_relationship-multi_instance/video_000.mp4``.

Examples::

    uv run python tasks/upload.py sep14
    uv run python tasks/upload.py --all
    uv run python tasks/upload.py sep14 --dry-run
"""

from __future__ import annotations

# ruff: noqa: T201, PLC0415 -- standalone CLI keeps the optional Hub dependency lazy.
import argparse
import json
import sys
from pathlib import Path

try:
    from ._common import TASKS_ROOT, TaskFormatError, resolve_task_dir, validate_task_dir
except ImportError:  # pragma: no cover - supports ``python tasks/upload.py``
    from _common import TASKS_ROOT, TaskFormatError, resolve_task_dir, validate_task_dir


DEFAULT_REPO_ID = "xjuIcthub/tasks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", nargs="*", help="Task names or paths; omit with --all")
    parser.add_argument("--all", action="store_true", help="Upload every child task directory under tasks/")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--repo-type", default="dataset", choices=("dataset", "model", "space"))
    parser.add_argument("--revision", default=None)
    parser.add_argument("--commit-message", default="Upload generated task videos")
    parser.add_argument("--token", default=None, help="Hugging Face token; prefer HF_TOKEN or cached login")
    parser.add_argument("--allow-partial", action="store_true", help="Upload files even when videos are incomplete")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _discover_tasks() -> list[Path]:
    if not TASKS_ROOT.exists():
        return []
    return sorted(
        (path for path in TASKS_ROOT.iterdir() if path.is_dir() and not path.name.startswith(".")),
        key=lambda path: path.name.casefold(),
    )


def _files_to_upload(task_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in task_dir.rglob("*")
            if path.is_file()
            and not path.name.startswith(".")
            and not path.name.endswith(".partial.mp4")
            and path.suffix.lower() in {".csv", ".mp4"}
        ),
        key=lambda path: path.relative_to(task_dir).as_posix().casefold(),
    )


def _resolve_selected(args: argparse.Namespace) -> list[Path]:
    if args.all and args.tasks:
        raise TaskFormatError("Use task names or --all, not both")
    if not args.all and not args.tasks:
        raise TaskFormatError("Provide at least one task name, or use --all")
    if args.all:
        selected = _discover_tasks()
        if not selected:
            raise TaskFormatError(f"No task directories found under {TASKS_ROOT}")
        return selected
    return [resolve_task_dir(task) for task in args.tasks]


def _validate_for_upload(task_dir: Path, allow_partial: bool) -> None:
    report = validate_task_dir(task_dir, check_videos=not allow_partial)
    if not report.ok:
        raise TaskFormatError(" | ".join(report.errors))


def _upload_with_api(
    task_dir: Path,
    *,
    repo_id: str,
    repo_type: str,
    revision: str | None,
    commit_message: str,
    token: str | None,
) -> object:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for uploads; run `uv sync` or install huggingface-hub") from exc
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    return api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=task_dir,
        path_in_repo=task_dir.name,
        revision=revision,
        commit_message=commit_message,
        allow_patterns=["*.csv", "*.mp4", "**/*.csv", "**/*.mp4"],
        ignore_patterns=["*.partial.mp4", "*.tmp", "*.jsonl"],
    )


def main() -> int:
    args = parse_args()
    try:
        task_dirs = _resolve_selected(args)
        for task_dir in task_dirs:
            _validate_for_upload(task_dir, args.allow_partial)
    except TaskFormatError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    plans: list[dict[str, object]] = []
    for task_dir in task_dirs:
        files = _files_to_upload(task_dir)
        if not files:
            print(f"ERROR: {task_dir.name} contains no CSV or MP4 files", file=sys.stderr)
            return 1
        plans.append(
            {
                "task": task_dir.name,
                "local": str(task_dir),
                "remote": f"{args.repo_id}/{task_dir.name}",
                "files": len(files),
                "csv": sum(path.suffix.lower() == ".csv" for path in files),
                "videos": sum(path.suffix.lower() == ".mp4" for path in files),
            }
        )
    if args.dry_run:
        print(json.dumps(plans, ensure_ascii=False, indent=2))
        return 0

    for task_dir, plan in zip(task_dirs, plans, strict=True):
        print(f"Uploading {plan['task']}: {plan['csv']} CSV + {plan['videos']} MP4 to {plan['remote']}")
        try:
            result = _upload_with_api(
                task_dir,
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                revision=args.revision,
                commit_message=args.commit_message,
                token=args.token,
            )
        except Exception as exc:
            print(f"ERROR: upload failed for {task_dir.name}: {exc}", file=sys.stderr)
            return 1
        commit_url = getattr(result, "commit_url", None)
        print(f"Uploaded {task_dir.name}" + (f": {commit_url}" if commit_url else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
