"""Shared task-directory and prompt-CSV helpers."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# ruff: noqa: PLR0912 -- validation deliberately reports several independent layout conditions.

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_ROOT = REPO_ROOT / "tasks"
TASK_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class TaskFormatError(ValueError):
    """Raised when a task directory or prompt CSV is not in the task format."""


@dataclass(frozen=True)
class PromptRecord:
    """One normalized prompt row."""

    id: str
    prompt: str
    row_number: int


@dataclass(frozen=True)
class TaskValidation:
    """Validation result suitable for both CLI output and automation."""

    task_dir: Path
    csv_count: int
    row_count: int
    video_dir_count: int
    video_count: int
    errors: tuple[str, ...]
    files: tuple[tuple[str, int], ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, object]:
        return {
            "task_dir": str(self.task_dir),
            "csv_count": self.csv_count,
            "row_count": self.row_count,
            "video_dir_count": self.video_dir_count,
            "video_count": self.video_count,
            "files": [{"csv": name, "rows": rows} for name, rows in self.files],
            "errors": list(self.errors),
            "ok": self.ok,
        }


def _task_root(tasks_root: Path | None = None) -> Path:
    return (tasks_root or TASKS_ROOT).resolve()


def resolve_task_dir(task: str | Path, tasks_root: Path | None = None) -> Path:
    """Resolve a task name or path while keeping it below the tasks root."""
    root = _task_root(tasks_root)
    raw = Path(task)
    if raw.is_absolute():
        candidate = raw
    elif raw.parts and raw.parts[0] == root.name:
        candidate = REPO_ROOT / raw
    else:
        candidate = root / raw
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise TaskFormatError(f"Task must be inside {root}: {candidate}") from exc
    if candidate == root:
        raise TaskFormatError("A task name is required; the tasks root is not a task")
    if not TASK_NAME_PATTERN.fullmatch(candidate.name):
        raise TaskFormatError(f"Invalid task name {candidate.name!r}; use letters, numbers, '.', '_' or '-'.")
    return candidate


def prompt_csv_files(task_dir: Path) -> list[Path]:
    """Return deterministic, top-level prompt CSV files."""
    return sorted(
        (path for path in task_dir.iterdir() if path.is_file() and path.suffix.lower() == ".csv"),
        key=lambda path: path.name.casefold(),
    )


def read_prompt_csv(path: Path) -> list[PromptRecord]:
    """Read a normalized UTF-8 CSV with the exact ``id,prompt`` schema."""
    records: list[PromptRecord] = []
    seen_ids: set[str] = set()
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise TaskFormatError(f"Cannot open {path}: {exc}") from exc
    with handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise TaskFormatError(f"{path.name}: file is empty; expected header id,prompt") from exc
        if header != ["id", "prompt"]:
            raise TaskFormatError(f"{path.name}: expected exact header id,prompt, got {header!r}")
        for row_number, row in enumerate(reader, start=2):
            if not row or all(not field.strip() for field in row):
                raise TaskFormatError(f"{path.name}:{row_number}: blank row")
            if len(row) != 2:
                raise TaskFormatError(
                    f"{path.name}:{row_number}: expected 2 columns, got {len(row)}; quote commas inside prompt text"
                )
            record_id, prompt = row[0].strip(), row[1].strip()
            if not record_id:
                raise TaskFormatError(f"{path.name}:{row_number}: id is empty")
            if record_id in seen_ids:
                raise TaskFormatError(f"{path.name}:{row_number}: duplicate id {record_id!r}")
            if not prompt:
                raise TaskFormatError(f"{path.name}:{row_number}: prompt is empty")
            seen_ids.add(record_id)
            records.append(PromptRecord(id=record_id, prompt=prompt, row_number=row_number))
    if not records:
        raise TaskFormatError(f"{path.name}: no prompt rows found")
    return records


def _video_names(output_dir: Path) -> list[str]:
    return sorted(
        (path.name for path in output_dir.iterdir() if path.is_file() and path.suffix.lower() == ".mp4"),
        key=str.casefold,
    )


def _expected_video_names(row_count: int) -> list[str]:
    return [f"video_{index:03d}.mp4" for index in range(row_count)]


def validate_task_dir(task: str | Path, *, check_videos: bool = False) -> TaskValidation:
    """Validate CSV layout and, optionally, generated-video layout."""
    errors: list[str] = []
    try:
        task_dir = resolve_task_dir(task)
    except TaskFormatError as exc:
        return TaskValidation(Path(task), 0, 0, 0, 0, (str(exc),), ())

    if not task_dir.exists():
        return TaskValidation(task_dir, 0, 0, 0, 0, (f"Task directory does not exist: {task_dir}",), ())
    if not task_dir.is_dir():
        return TaskValidation(task_dir, 0, 0, 0, 0, (f"Task path is not a directory: {task_dir}",), ())

    csv_files = prompt_csv_files(task_dir)
    if not csv_files:
        errors.append(f"{task_dir}: no top-level .csv files found")
    files: list[tuple[str, int]] = []
    rows_by_stem: dict[str, list[PromptRecord]] = {}
    for path in csv_files:
        try:
            records = read_prompt_csv(path)
        except TaskFormatError as exc:
            errors.append(str(exc))
            continue
        rows_by_stem[path.stem] = records
        files.append((path.name, len(records)))

    top_level_unexpected = [
        path.name
        for path in task_dir.iterdir()
        if not path.name.startswith(".") and path.is_file() and path.suffix.lower() != ".csv"
    ]
    if top_level_unexpected:
        errors.append("Unexpected top-level files: " + ", ".join(sorted(top_level_unexpected)))

    output_dirs = {path.name: path for path in task_dir.iterdir() if not path.name.startswith(".") and path.is_dir()}
    expected_stems = set(rows_by_stem)
    unexpected_dirs = sorted(set(output_dirs) - expected_stems)
    if unexpected_dirs:
        errors.append("Unexpected output directories: " + ", ".join(unexpected_dirs))

    video_dir_count = sum(stem in output_dirs for stem in expected_stems)
    video_count = 0
    if check_videos:
        missing_dirs = sorted(expected_stems - set(output_dirs))
        if missing_dirs:
            errors.append("Missing output directories: " + ", ".join(missing_dirs))
        for stem, records in rows_by_stem.items():
            output_dir = output_dirs.get(stem)
            if output_dir is None:
                continue
            actual = _video_names(output_dir)
            expected = _expected_video_names(len(records))
            video_count += len(actual)
            if actual != expected:
                errors.append(f"{stem}: expected videos {expected[0]}..{expected[-1]}, got {actual or 'none'}")
    else:
        video_count = sum(len(_video_names(path)) for path in output_dirs.values())

    return TaskValidation(
        task_dir=task_dir,
        csv_count=len(csv_files),
        row_count=sum(len(records) for records in rows_by_stem.values()),
        video_dir_count=video_dir_count,
        video_count=video_count,
        errors=tuple(errors),
        files=tuple(files),
    )


def iter_task_records(task_dir: Path) -> Iterable[tuple[Path, int, PromptRecord]]:
    """Yield ``(csv_path, zero_based_index, record)`` in stable task order."""
    for csv_path in prompt_csv_files(task_dir):
        for index, record in enumerate(read_prompt_csv(csv_path)):
            yield csv_path, index, record
