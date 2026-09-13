#!/usr/bin/env python3
"""Run the repository's HQ text-to-video pipeline for every CSV in a task.

The default profile matches the H200 run used for the September prompt batch:
full BF16 LTX-2.3 dev weights, no quantization, the HQ Res2s sampler, 15 stage-1
steps plus 3 stage-2 steps, 1920x1088 output, 24 fps, and a five-second MP4
with audio. Run with one visible GPU, for example::

    CUDA_VISIBLE_DEVICES=0 uv run python tasks/infer.py sep14

Each top-level CSV writes to a same-name directory containing
``video_000.mp4``, ``video_001.mp4``, and so on. Existing files are skipped
unless ``--overwrite`` is supplied.
"""

from __future__ import annotations

# ruff: noqa: T201, PLC0415, PLR0915 -- standalone long-running CLI with lazy model imports.
import argparse
import dataclasses
import hashlib
import json
import logging
import time
from pathlib import Path

try:
    from ._common import (
        REPO_ROOT,
        TaskFormatError,
        prompt_csv_files,
        read_prompt_csv,
        resolve_task_dir,
        validate_task_dir,
    )
except ImportError:  # pragma: no cover - supports ``python tasks/infer.py``
    from _common import (
        REPO_ROOT,
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


def _append_manifest(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


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


def main() -> int:
    args = parse_args()
    task_dir = resolve_task_dir(args.task)
    _validate_inference_args(args, task_dir)
    report = validate_task_dir(task_dir)
    if not report.ok:
        raise TaskFormatError("Task validation failed: " + " | ".join(report.errors))

    csv_files = prompt_csv_files(task_dir)
    pending = [
        (csv_path, index)
        for csv_path in csv_files
        for index, _record in enumerate(read_prompt_csv(csv_path))
        if args.overwrite or not (task_dir / csv_path.stem / f"video_{index:03d}.mp4").is_file()
    ]
    print(
        f"Task {task_dir.name}: csv={len(csv_files)} rows={report.row_count} pending={len(pending)} "
        f"profile=HQ {args.width}x{args.height} {args.export_frames} frames @ {args.fps:g} fps "
        f"steps={args.num_inference_steps}+3 audio=on"
    )
    if args.dry_run or not pending:
        for csv_path in csv_files:
            records = read_prompt_csv(csv_path)
            output_dir = task_dir / csv_path.stem
            print(f"  {csv_path.name}: {len(records)} prompts -> {output_dir.name}/video_000.mp4 ...")
        return 0

    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one visible GPU, found {torch.cuda.device_count()}. "
            "Set CUDA_VISIBLE_DEVICES to one GPU before launching this task."
        )

    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, LTX_2_3_HQ_PARAMS
    from ltx_pipelines.utils.media_io import encode_video

    pipeline = _build_pipeline(args)
    tiling_config = TilingConfig.default()
    video_chunks = get_video_chunks_number(args.num_frames, tiling_config)
    params = LTX_2_3_HQ_PARAMS
    negative_prompt = args.negative_prompt or DEFAULT_NEGATIVE_PROMPT
    run_started = time.monotonic()
    global_index = 0
    for csv_path in csv_files:
        records = read_prompt_csv(csv_path)
        output_dir = task_dir / csv_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.jsonl"
        for index, record in enumerate(records):
            output_path = output_dir / f"video_{index:03d}.mp4"
            if output_path.is_file() and not args.overwrite:
                print(f"skip {csv_path.name}:{index} -> {output_path.name}")
                global_index += 1
                continue
            temporary_path = output_dir / f".video_{index:03d}.partial.mp4"
            temporary_path.unlink(missing_ok=True)
            seed = args.seed_base + global_index
            print(f"run {csv_path.name}:{index} id={record.id!r} seed={seed}")
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
                "validation": validation,
                "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            }
            _append_manifest(manifest_path, payload)
            print(f"saved {output_path} in {elapsed:.1f}s")
            global_index += 1
    print(f"Completed task {task_dir.name} in {time.monotonic() - run_started:.1f}s")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
