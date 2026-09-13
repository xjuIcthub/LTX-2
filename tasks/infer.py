#!/usr/bin/env python3
"""Run the repository's HQ text-to-video pipeline for every CSV in a task.

The default profile matches the H200 run used for the September prompt batch:
full BF16 LTX-2.3 dev weights, no quantization, the HQ Res2s sampler, 15 stage-1
steps plus 3 stage-2 steps, 1920x1088 output, 24 fps, and a five-second MP4
with audio. Run with one visible GPU, for example::

    CUDA_VISIBLE_DEVICES=0 uv run python tasks/infer.py sep14

For multi-GPU inference, use one process per visible GPU. The first workers
receive round-robin groups of videos; the last worker scans every unfinished
video and consumes the remainder as a fallback::

    CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python tasks/infer.py sep14 --num-processes 4

Each top-level CSV writes to a same-name directory containing
``video_000.mp4``, ``video_001.mp4``, and so on. Existing files are skipped
unless ``--overwrite`` is supplied.
"""

from __future__ import annotations

# ruff: noqa: T201, PLC0415 -- standalone long-running CLI with lazy model imports.
import argparse
import dataclasses
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from ._common import (
        REPO_ROOT,
        PromptRecord,
        TaskFormatError,
        prompt_csv_files,
        read_prompt_csv,
        resolve_task_dir,
        validate_task_dir,
    )
except ImportError:  # pragma: no cover - supports ``python tasks/infer.py``
    from _common import (
        REPO_ROOT,
        PromptRecord,
        TaskFormatError,
        prompt_csv_files,
        read_prompt_csv,
        resolve_task_dir,
        validate_task_dir,
    )


_NOT_LOADED = object()


class _CachedBuilder:
    """Keep a model builder's first result resident across prompts."""

    def __init__(self, builder: object, label: str) -> None:
        self._builder = builder
        self._label = label
        self._model: object = _NOT_LOADED

    def __getattr__(self, name: str) -> object:
        return getattr(self._builder, name)

    def build(self, *args: object, **kwargs: object) -> object:
        if self._model is _NOT_LOADED:
            logging.info("Loading persistent %s", self._label)
            build = self._builder.build  # type: ignore[attr-defined]
            self._model = build(*args, **kwargs)
        return self._model


def _cached_factory(factory: object, label: str) -> object:
    model: object = _NOT_LOADED

    def build_once(*args: object, **kwargs: object) -> object:
        nonlocal model
        if model is _NOT_LOADED:
            logging.info("Loading persistent %s", label)
            model = factory(*args, **kwargs)  # type: ignore[operator]
        return model

    return build_once


def enable_persistent_weights(pipeline: object) -> None:
    """Cache component weights so one task does not reload the 22B model per row."""
    pipeline.prompt_encoder._build_text_encoder = _cached_factory(  # type: ignore[attr-defined]
        pipeline.prompt_encoder._build_text_encoder, "Gemma text encoder"
    )
    pipeline.prompt_encoder._build_embeddings_processor = _cached_factory(  # type: ignore[attr-defined]
        pipeline.prompt_encoder._build_embeddings_processor, "embeddings processor"
    )
    pipeline.stage_1._build_transformer = _cached_factory(  # type: ignore[attr-defined]
        pipeline.stage_1._build_transformer, "stage-1 transformer"
    )
    pipeline.stage_2._build_transformer = _cached_factory(  # type: ignore[attr-defined]
        pipeline.stage_2._build_transformer, "stage-2 transformer"
    )
    pipeline.upsampler._encoder_builder = _CachedBuilder(  # type: ignore[attr-defined]
        pipeline.upsampler._encoder_builder, "upsampler video encoder"
    )
    pipeline.upsampler._upsampler_builder = _CachedBuilder(  # type: ignore[attr-defined]
        pipeline.upsampler._upsampler_builder, "spatial upsampler"
    )
    pipeline.video_decoder._decoder_builder = _CachedBuilder(  # type: ignore[attr-defined]
        pipeline.video_decoder._decoder_builder, "video decoder"
    )
    pipeline.audio_decoder._decoder_builder = _CachedBuilder(  # type: ignore[attr-defined]
        pipeline.audio_decoder._decoder_builder, "audio decoder"
    )
    pipeline.audio_decoder._vocoder_builder = _CachedBuilder(  # type: ignore[attr-defined]
        pipeline.audio_decoder._vocoder_builder, "audio vocoder"
    )
    # No task CSV supplies image conditioning. Avoid building the unused image encoder.
    pipeline.image_conditioner = lambda _fn: []  # type: ignore[attr-defined]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", help="Task directory name or path, for example sep14")
    parser.add_argument(
        "--checkpoint-path", type=Path, default=REPO_ROOT / "models/ltx-2.3/ltx-2.3-22b-dev.safetensors"
    )
    parser.add_argument(
        "--distilled-lora-path",
        type=Path,
        default=REPO_ROOT / "models/ltx-2.3/ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
    )
    parser.add_argument(
        "--spatial-upsampler-path",
        type=Path,
        default=REPO_ROOT / "models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    )
    parser.add_argument("--gemma-root", type=Path, default=REPO_ROOT / "models/gemma-3-12b")
    parser.add_argument("--height", type=int, default=1088)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--num-frames", type=int, default=121, help="Model frames; must be 8*k+1")
    parser.add_argument("--export-frames", type=int, default=120, help="Frames written to the five-second MP4")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--num-inference-steps", type=int, default=15)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--crf", type=int, default=14)
    parser.add_argument("--preset", default="slow")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--enhance-prompt", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reload-weights-each-video", action="store_true")
    parser.add_argument("--quantization", choices=("none", "fp8-cast"), default="none")
    parser.add_argument("--offload", choices=("none", "cpu"), default="none")
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument(
        "--num-processes",
        type=int,
        default=1,
        help="Number of one-GPU worker processes; 0 uses every visible GPU",
    )
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="Comma-separated GPU IDs to use (overrides CUDA_VISIBLE_DEVICES for worker selection)",
    )
    parser.add_argument("--max-retries", type=int, default=2, help="Attempts per video within one run")
    parser.add_argument("--_worker-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-count", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_run-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _validate_inference_args(args: argparse.Namespace, task_dir: Path) -> None:
    report = validate_task_dir(task_dir)
    if not report.ok:
        raise TaskFormatError("Task validation failed: " + " | ".join(report.errors))
    if args.height % 64 or args.width % 64:
        raise ValueError("height and width must be divisible by 64")
    if (args.num_frames - 1) % 8:
        raise ValueError("num_frames must satisfy num_frames = 8*k+1")
    if args.export_frames < 1 or args.export_frames > args.num_frames:
        raise ValueError("export_frames must be between 1 and num_frames")
    if not args.fps.is_integer():
        raise ValueError("fps must be an integer for the current video encoder")
    if abs(args.export_frames / args.fps - 5) > 1e-9:
        raise ValueError("The default task profile requires export_frames / fps == 5 seconds")
    if args.num_inference_steps < 1 or args.max_batch_size < 1:
        raise ValueError("num-inference-steps and max-batch-size must be positive")
    if args.num_processes < 0:
        raise ValueError("num-processes must be non-negative; use 0 for all visible GPUs")
    if args.max_retries < 1:
        raise ValueError("max-retries must be positive")
    if not args.dry_run:
        for path in (args.checkpoint_path, args.distilled_lora_path, args.spatial_upsampler_path, args.gemma_root):
            if not path.exists():
                raise FileNotFoundError(path)


def _trim_audio(audio: object, duration_seconds: float) -> object:
    if audio is None:
        return None
    sample_count = round(audio.sampling_rate * duration_seconds)  # type: ignore[attr-defined]
    waveform = audio.waveform  # type: ignore[attr-defined]
    if waveform.ndim != 2:
        raise ValueError(f"Expected stereo audio waveform, got shape {tuple(waveform.shape)}")
    if waveform.shape[0] == 2:
        waveform = waveform[:, :sample_count]
    elif waveform.shape[1] == 2:
        waveform = waveform[:sample_count, :]
    else:
        raise ValueError(f"Expected stereo audio waveform, got shape {tuple(waveform.shape)}")
    return dataclasses.replace(audio, waveform=waveform)


def _export_frames(video: object, count: int) -> object:
    remaining = count
    for chunk in video:  # type: ignore[union-attr]
        take = min(remaining, len(chunk))
        if take:
            yield chunk[:take]
            remaining -= take
    if remaining:
        raise RuntimeError(f"Decoder returned {remaining} fewer frames")


def _build_pipeline(args: argparse.Namespace) -> object:
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_pipelines.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline
    from ltx_pipelines.utils.allocator_trim_strategy import AllocatorTrimStrategy
    from ltx_pipelines.utils.quantization_factory import QuantizationKind
    from ltx_pipelines.utils.types import OffloadMode

    checkpoint_path = str(args.checkpoint_path.resolve())
    quantization = (
        QuantizationKind(args.quantization).to_policy(checkpoint_path) if args.quantization != "none" else None
    )
    persistent_weights = not args.reload_weights_each_video
    pipeline = TI2VidTwoStagesHQPipeline(
        checkpoint_path=checkpoint_path,
        distilled_lora=[
            LoraPathStrengthAndSDOps(
                path=str(args.distilled_lora_path.resolve()),
                strength=1.0,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            )
        ],
        distilled_lora_strength_stage_1=0.25,
        distilled_lora_strength_stage_2=0.5,
        spatial_upsampler_path=str(args.spatial_upsampler_path.resolve()),
        gemma_root=str(args.gemma_root.resolve()),
        loras=(),
        quantization=quantization,
        offload_mode=OffloadMode(args.offload),
        alloc_trim_strategy=AllocatorTrimStrategy.DEFER if persistent_weights else AllocatorTrimStrategy.TRIM,
    )
    if persistent_weights:
        enable_persistent_weights(pipeline)
    return pipeline


WorkItem = tuple[Path, int, PromptRecord]


def _all_work_items(task_dir: Path) -> list[WorkItem]:
    """Read every prompt once in deterministic CSV/row order."""
    return [
        (csv_path, index, record)
        for csv_path in prompt_csv_files(task_dir)
        for index, record in enumerate(read_prompt_csv(csv_path))
    ]


def _output_path(task_dir: Path, csv_path: Path, index: int) -> Path:
    return task_dir / csv_path.stem / f"video_{index:03d}.mp4"


def _video_lock_path(output_path: Path) -> Path:
    return output_path.with_suffix(".lock")


def _read_lock_state(handle: object) -> list[str]:
    handle.seek(0)  # type: ignore[attr-defined]
    return handle.read().strip().split()  # type: ignore[attr-defined]


def _write_lock_state(handle: object, state: str) -> None:
    handle.seek(0)  # type: ignore[attr-defined]
    handle.truncate()  # type: ignore[attr-defined]
    handle.write(state + "\n")  # type: ignore[attr-defined]
    handle.flush()  # type: ignore[attr-defined]


def _release_video_lock(handle: object) -> None:
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
    handle.close()  # type: ignore[attr-defined]


def _acquire_video_lock(lock_path: Path) -> object | None:
    """Try to claim one output path without leaving stale claims after a crash."""
    import fcntl

    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def _state_matches_run(lock_path: Path, run_id: str, terminal_states: set[str]) -> bool:
    try:
        state = lock_path.read_text(encoding="utf-8").strip().split()
    except FileNotFoundError:
        return False
    return len(state) >= 2 and state[1] == run_id and state[0] in terminal_states


def _claim_item_lock(
    output_path: Path,
    *,
    overwrite: bool,
    run_id: str,
) -> object | None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _video_lock_path(output_path)
    handle = _acquire_video_lock(lock_path)
    if handle is None:
        return None
    state = _read_lock_state(handle)
    if len(state) >= 2 and state[1] == run_id and state[0] in {"done", "exhausted"}:
        _release_video_lock(handle)
        return None
    if output_path.is_file() and not overwrite:
        _release_video_lock(handle)
        return None
    return handle


def _record_item_failure(handle: object, run_id: str, max_retries: int) -> None:
    state = _read_lock_state(handle)
    attempts = 0
    if len(state) >= 3 and state[0] == "failed" and state[1] == run_id:
        try:
            attempts = int(state[2])
        except ValueError:
            attempts = 0
    attempts += 1
    status = "exhausted" if attempts >= max_retries else "failed"
    _write_lock_state(handle, f"{status} {run_id} {attempts}")


def _append_manifest(path: Path, payload: dict[str, object]) -> None:
    import fcntl

    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _validate_output(path: Path, args: argparse.Namespace) -> dict[str, object]:
    import av

    with av.open(str(path)) as container:
        video_stream = next(stream for stream in container.streams if stream.type == "video")
        audio_streams = sum(stream.type == "audio" for stream in container.streams)
        decoded_frames = sum(1 for _ in container.decode(video=video_stream))
        if video_stream.duration is None or video_stream.time_base is None:
            raise RuntimeError(f"Video stream has no duration: {path}")
        duration = float(video_stream.duration * video_stream.time_base)
        result = {
            "width": video_stream.width,
            "height": video_stream.height,
            "fps": float(video_stream.average_rate),
            "duration": duration,
            "frames": decoded_frames,
            "audio_streams": audio_streams,
        }
    if result["width"] != args.width or result["height"] != args.height:
        raise RuntimeError(f"Unexpected output dimensions for {path}: {result}")
    if result["frames"] != args.export_frames or abs(result["duration"] - 5) > 0.01:
        raise RuntimeError(f"Unexpected output duration/frames for {path}: {result}")
    if result["audio_streams"] != 1:
        raise RuntimeError(f"Expected one audio stream for {path}: {result}")
    return result


def _run_video_item(  # noqa: PLR0913 -- the per-video call needs the shared HQ runtime context.
    *,
    args: argparse.Namespace,
    task_dir: Path,
    item: WorkItem,
    global_index: int,
    pipeline: object,
    tiling_config: object,
    video_chunks: int,
    params: object,
    negative_prompt: str,
    worker_index: int,
    worker_count: int,
    gpu_label: str,
) -> None:
    import torch

    csv_path, index, record = item
    output_path = _output_path(task_dir, csv_path, index)
    temporary_path = output_path.with_name(f".video_{index:03d}.partial.mp4")
    temporary_path.unlink(missing_ok=True)
    seed = args.seed_base + global_index
    print(
        f"worker={worker_index}/{worker_count} gpu={gpu_label} run {csv_path.name}:{index} "
        f"id={record.id!r} seed={seed}",
        flush=True,
    )
    started = time.monotonic()
    with torch.inference_mode():
        video, audio = pipeline(
            prompt=record.prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.fps,
            num_inference_steps=args.num_inference_steps,
            video_guider_params=params.video_guider_params,
            audio_guider_params=params.audio_guider_params,
            images=[],
            tiling_config=tiling_config,
            enhance_prompt=args.enhance_prompt,
            max_batch_size=args.max_batch_size,
        )
        audio = _trim_audio(audio, 5)
        from ltx_pipelines.utils.media_io import encode_video

        encode_video(
            video=_export_frames(video, args.export_frames),
            fps=int(args.fps),
            audio=audio,
            output_path=str(temporary_path),
            video_chunks_number=video_chunks,
            crf=args.crf,
            preset=args.preset,
        )
    validation = _validate_output(temporary_path, args)
    temporary_path.replace(output_path)
    elapsed = time.monotonic() - started
    payload = {
        "state": "complete",
        "csv": csv_path.name,
        "row_index": index,
        "id": record.id,
        "prompt": record.prompt,
        "output": output_path.name,
        "seed": seed,
        "elapsed_seconds": round(elapsed, 3),
        "pipeline": "TI2VidTwoStagesHQPipeline",
        "num_inference_steps": args.num_inference_steps,
        "stage_2_steps": 3,
        "width": args.width,
        "height": args.height,
        "num_frames": args.num_frames,
        "export_frames": args.export_frames,
        "fps": args.fps,
        "audio": True,
        "quantization": args.quantization,
        "offload": args.offload,
        "persistent_weights": not args.reload_weights_each_video,
        "worker_index": worker_index,
        "worker_count": worker_count,
        "gpu": gpu_label,
        "validation": validation,
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    _append_manifest(output_path.parent / "manifest.jsonl", payload)
    print(f"worker={worker_index}/{worker_count} saved {output_path} in {elapsed:.1f}s", flush=True)


def _run_worker(
    args: argparse.Namespace,
    task_dir: Path,
    items: list[WorkItem],
    *,
    worker_index: int,
    worker_count: int,
    run_id: str,
) -> int:
    import torch

    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count != 1:
        raise RuntimeError(
            f"Worker {worker_index} expected exactly one visible GPU, found {visible_gpu_count}. "
            "The parent must assign one CUDA device per process."
        )

    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, LTX_2_3_HQ_PARAMS

    gpu_label = os.environ.get("LTX_TASK_GPU_LABEL", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    print(f"worker={worker_index}/{worker_count} gpu={gpu_label} loading HQ pipeline", flush=True)
    pipeline = _build_pipeline(args)
    tiling_config = TilingConfig.default()
    video_chunks = get_video_chunks_number(args.num_frames, tiling_config)
    params = LTX_2_3_HQ_PARAMS
    negative_prompt = args.negative_prompt or DEFAULT_NEGATIVE_PROMPT
    run_started = time.monotonic()
    if worker_index == worker_count - 1:
        # The final worker scans the complete queue and takes any remainder
        # left by the round-robin assignments of the other workers.
        candidate_items = list(enumerate(items))
    else:
        candidate_items = [
            (global_index, item)
            for global_index, item in enumerate(items)
            if global_index % worker_count == worker_index
        ]
    while True:
        pending_seen = False
        claimed_this_round = False
        for global_index, item in candidate_items:
            csv_path, index, _record = item
            output_path = _output_path(task_dir, csv_path, index)
            lock_path = _video_lock_path(output_path)
            if output_path.is_file() and not args.overwrite:
                continue
            if args.overwrite and _state_matches_run(lock_path, run_id, {"done", "exhausted"}):
                continue
            pending_seen = True
            lock_handle = _claim_item_lock(output_path, overwrite=args.overwrite, run_id=run_id)
            if lock_handle is None:
                continue
            claimed_this_round = True
            try:
                _run_video_item(
                    args=args,
                    task_dir=task_dir,
                    item=item,
                    global_index=global_index,
                    pipeline=pipeline,
                    tiling_config=tiling_config,
                    video_chunks=video_chunks,
                    params=params,
                    negative_prompt=negative_prompt,
                    worker_index=worker_index,
                    worker_count=worker_count,
                    gpu_label=gpu_label,
                )
                _write_lock_state(lock_handle, f"done {run_id}")
            except Exception:
                logging.exception(
                    "worker=%s failed %s:%s; retrying up to %s times",
                    worker_index,
                    csv_path.name,
                    index,
                    args.max_retries,
                )
                _record_item_failure(lock_handle, run_id, args.max_retries)
            finally:
                _release_video_lock(lock_handle)
        if not pending_seen:
            break
        if not claimed_this_round:
            # Another worker currently owns the remaining files. Locks release
            # automatically if that process crashes, so keep polling as a fallback.
            time.sleep(1)
    print(f"worker={worker_index}/{worker_count} complete in {time.monotonic() - run_started:.1f}s", flush=True)
    return 0


def _gpu_ids(args: argparse.Namespace) -> list[str]:
    configured = args.gpu_ids if args.gpu_ids is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is not None:
        gpu_ids = [value.strip() for value in configured.split(",") if value.strip()]
        if not gpu_ids or gpu_ids == ["-1"]:
            raise RuntimeError("No CUDA devices are selected; set CUDA_VISIBLE_DEVICES or --gpu-ids")
        return gpu_ids
    import torch

    count = torch.cuda.device_count()
    if count < 1:
        raise RuntimeError("No CUDA devices found")
    return [str(index) for index in range(count)]


def _run_multi_process(
    args: argparse.Namespace,
    task_dir: Path,
    items: list[WorkItem],
) -> int:
    gpu_ids = _gpu_ids(args)
    worker_count = len(gpu_ids) if args.num_processes == 0 else args.num_processes
    if worker_count > len(gpu_ids):
        raise ValueError(f"Requested {worker_count} processes but only {len(gpu_ids)} GPUs are visible: {gpu_ids}")
    selected_gpu_ids = gpu_ids[:worker_count]
    run_id = args._run_id or f"{time.time_ns()}-{os.getpid()}"
    print(
        f"Launching {worker_count} one-GPU workers over {len(items)} videos; "
        f"last worker is the dynamic remainder fallback (run={run_id})",
        flush=True,
    )
    script_path = str(Path(__file__).resolve())
    child_args = [argument for argument in sys.argv[1:] if not argument.startswith("--_worker-")]
    child_args = [argument for argument in child_args if not argument.startswith("--_run-id")]
    processes: list[subprocess.Popen[bytes]] = []
    for worker_index, gpu_id in enumerate(selected_gpu_ids):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["LTX_TASK_GPU_LABEL"] = gpu_id
        env["PYTHONUNBUFFERED"] = "1"
        command = [
            sys.executable,
            script_path,
            *child_args,
            "--_worker-index",
            str(worker_index),
            "--_worker-count",
            str(worker_count),
            "--_run-id",
            run_id,
        ]
        print(f"  worker={worker_index}/{worker_count} -> CUDA_VISIBLE_DEVICES={gpu_id}", flush=True)
        processes.append(subprocess.Popen(command, env=env))
    exit_codes = [process.wait() for process in processes]
    print(f"Worker exit codes: {exit_codes}", flush=True)
    final_report = validate_task_dir(task_dir, check_videos=True)
    if not final_report.ok:
        for error in final_report.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if args.overwrite:
        unfinished = [
            f"{csv_path.name}:{index}"
            for csv_path, index, _record in items
            if not _state_matches_run(_video_lock_path(_output_path(task_dir, csv_path, index)), run_id, {"done"})
        ]
        if unfinished:
            print(
                "ERROR: overwrite run did not complete every video: " + ", ".join(unfinished),
                file=sys.stderr,
            )
            return 1
    return 0


def main() -> int:
    args = parse_args()
    task_dir = resolve_task_dir(args.task)
    _validate_inference_args(args, task_dir)
    report = validate_task_dir(task_dir)
    if not report.ok:
        raise TaskFormatError("Task validation failed: " + " | ".join(report.errors))

    csv_files = prompt_csv_files(task_dir)
    items = _all_work_items(task_dir)
    pending = [item for item in items if args.overwrite or not _output_path(task_dir, item[0], item[1]).is_file()]
    is_worker = args._worker_index is not None
    if not is_worker:
        print(
            f"Task {task_dir.name}: csv={len(csv_files)} rows={report.row_count} pending={len(pending)} "
            f"profile=HQ {args.width}x{args.height} {args.export_frames} frames @ {args.fps:g} fps "
            f"steps={args.num_inference_steps}+3 audio=on workers={args.num_processes or 'auto'}"
        )
        if args.dry_run or not pending:
            for csv_path in csv_files:
                records = read_prompt_csv(csv_path)
                output_dir = task_dir / csv_path.stem
                print(f"  {csv_path.name}: {len(records)} prompts -> {output_dir.name}/video_000.mp4 ...")
            return 0 if args.dry_run else int(not validate_task_dir(task_dir, check_videos=True).ok)
        return _run_multi_process(args, task_dir, items)

    if args._worker_count is None or args._worker_index < 0 or args._worker_index >= args._worker_count:
        raise ValueError("Internal worker index/count is invalid")
    return _run_worker(
        args,
        task_dir,
        items,
        worker_index=args._worker_index,
        worker_count=args._worker_count,
        run_id=args._run_id or f"{time.time_ns()}-{os.getpid()}",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
