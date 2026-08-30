from __future__ import annotations

# ruff: noqa: T201 -- this long-running CLI must stream human-readable progress.
import argparse
import csv
import fcntl
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

import torch

from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline
from ltx_pipelines.utils.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, LTX_2_3_HQ_PARAMS
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.quantization_factory import QuantizationKind
from ltx_pipelines.utils.types import OffloadMode

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAME = Path(__file__).name
_NOT_LOADED = object()


class _CachedBuilder:
    """Keep the first model built by an LTX builder resident for later prompts."""

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


def _cached_factory(factory: Callable[..., object], label: str) -> Callable[..., object]:
    model: object = _NOT_LOADED

    def build_once(*args: object, **kwargs: object) -> object:
        nonlocal model
        if model is _NOT_LOADED:
            logging.info("Loading persistent %s", label)
            model = factory(*args, **kwargs)
        return model

    return build_once


def enable_persistent_weights(pipeline: TI2VidTwoStagesHQPipeline) -> None:
    """Cache every model used by this text-only, video-only batch on the H200."""
    pipeline.prompt_encoder._build_text_encoder = _cached_factory(  # type: ignore[method-assign]
        pipeline.prompt_encoder._build_text_encoder,
        "Gemma text encoder",
    )
    pipeline.prompt_encoder._build_embeddings_processor = _cached_factory(  # type: ignore[method-assign]
        pipeline.prompt_encoder._build_embeddings_processor,
        "embeddings processor",
    )
    pipeline.stage_1._build_transformer = _cached_factory(  # type: ignore[method-assign]
        pipeline.stage_1._build_transformer,
        "stage-1 transformer",
    )
    pipeline.stage_2._build_transformer = _cached_factory(  # type: ignore[method-assign]
        pipeline.stage_2._build_transformer,
        "stage-2 transformer",
    )
    pipeline.upsampler._encoder_builder = _CachedBuilder(
        pipeline.upsampler._encoder_builder,
        "upsampler video encoder",
    )
    pipeline.upsampler._upsampler_builder = _CachedBuilder(
        pipeline.upsampler._upsampler_builder,
        "spatial upsampler",
    )
    pipeline.video_decoder._decoder_builder = _CachedBuilder(
        pipeline.video_decoder._decoder_builder,
        "video decoder",
    )

    # This batch is text-to-video only, so image conditioning is always empty.
    # Audio latents still participate in diffusion, but decoding them is unnecessary
    # because the requested MP4 files contain no audio stream.
    pipeline.image_conditioner = lambda _fn: []  # type: ignore[assignment]
    pipeline.audio_decoder = lambda _latent: None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a sharded 256-prompt VBench batch with the official LTX-2.3 HQ pipeline."
    )
    parser.add_argument("--prompts-csv", type=Path, default=REPO_ROOT / "VBench-origin_256_prompts.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "vbench_origin_256")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=REPO_ROOT / "models/ltx-2.3/ltx-2.3-22b-dev.safetensors",
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
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--num-inference-steps", type=int, default=15)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--quantization", choices=("fp8-cast", "none"), default="fp8-cast")
    parser.add_argument("--offload", choices=("none", "cpu"), default="none")
    parser.add_argument("--distilled-lora-strength-stage-1", type=float, default=0.25)
    parser.add_argument("--distilled-lora-strength-stage-2", type=float, default=0.5)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None, help="Exclusive end index; defaults to all prompts.")
    parser.add_argument("--enhance-prompt", action="store_true")
    parser.add_argument(
        "--reload-weights-each-video",
        action="store_true",
        help="Use the original low-memory behavior instead of keeping model weights resident.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "prompt" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a 'prompt' column")
        rows = list(reader)
    if len(rows) != 256:
        raise ValueError(f"Expected 256 prompts in {path}, found {len(rows)}")
    if any(not row["prompt"].strip() for row in rows):
        raise ValueError(f"{path} contains an empty prompt")
    return rows


def is_complete(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 1024


def append_manifest(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def selected_indices(args: argparse.Namespace, row_count: int) -> list[int]:
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    start = max(0, args.start_index)
    end = row_count if args.end_index is None else min(row_count, args.end_index)
    if start >= end:
        raise ValueError(f"Empty index range [{start}, {end})")
    indices = [index for index in range(start, end) if index % args.num_shards == args.shard_index]
    if not indices:
        raise ValueError(f"Shard {args.shard_index}/{args.num_shards} has no indices in [{start}, {end})")
    return indices


def validate_args(args: argparse.Namespace) -> None:
    if torch.cuda.device_count() != 1 and not args.dry_run:
        raise RuntimeError(
            f"Expected exactly one visible GPU per HQ shard, found {torch.cuda.device_count()}. "
            "Launch one process per GPU."
        )
    if args.height % 64 or args.width % 64:
        raise ValueError("LTX dimensions must be divisible by 64")
    if (args.num_frames - 1) % 8:
        raise ValueError("LTX num_frames must satisfy num_frames = 8 * k + 1")
    if args.num_inference_steps < 1:
        raise ValueError("--num-inference-steps must be positive")
    for strength in (args.distilled_lora_strength_stage_1, args.distilled_lora_strength_stage_2):
        if not 0.0 <= strength <= 1.0:
            raise ValueError("Distilled LoRA strengths must be between 0 and 1")
    for path in (args.checkpoint_path, args.distilled_lora_path, args.spatial_upsampler_path, args.gemma_root):
        if not path.exists() and not args.dry_run:
            raise FileNotFoundError(path)


def acquire_shard_lock(output_dir: Path, shard_index: int) -> TextIO:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / f".{SCRIPT_NAME}.shard_{shard_index}.lock"
    handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"Shard {shard_index} is already running for {output_dir}") from None
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    return handle


def build_pipeline(args: argparse.Namespace) -> TI2VidTwoStagesHQPipeline:
    checkpoint_path = str(args.checkpoint_path.resolve())
    distilled_lora = [
        LoraPathStrengthAndSDOps(
            path=str(args.distilled_lora_path.resolve()),
            strength=1.0,
            sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
        )
    ]
    quantization = (
        QuantizationKind(args.quantization).to_policy(checkpoint_path) if args.quantization != "none" else None
    )
    persistent_weights = not args.reload_weights_each_video
    pipeline = TI2VidTwoStagesHQPipeline(
        checkpoint_path=checkpoint_path,
        distilled_lora=distilled_lora,
        distilled_lora_strength_stage_1=args.distilled_lora_strength_stage_1,
        distilled_lora_strength_stage_2=args.distilled_lora_strength_stage_2,
        spatial_upsampler_path=str(args.spatial_upsampler_path.resolve()),
        gemma_root=str(args.gemma_root.resolve()),
        loras=(),
        quantization=quantization,
        registry=None,
        offload_mode=OffloadMode(args.offload),
        alloc_trim_strategy=AllocatorTrimStrategy.DEFER if persistent_weights else AllocatorTrimStrategy.TRIM,
    )
    if persistent_weights:
        enable_persistent_weights(pipeline)
    return pipeline


def main() -> None:
    args = parse_args()
    rows = load_rows(args.prompts_csv.resolve())
    indices = selected_indices(args, len(rows))
    validate_args(args)
    output_dir = args.output_dir.resolve()

    print(
        f"[{SCRIPT_NAME}] HQ shard={args.shard_index}/{args.num_shards} prompts={len(indices)} "
        f"frames={args.num_frames} size={args.width}x{args.height} fps={args.fps:g} "
        f"steps={args.num_inference_steps}+3 audio=off "
        f"weights={'reload' if args.reload_weights_each_video else 'persistent'}",
        flush=True,
    )
    if args.dry_run:
        for index in indices[:5]:
            print(f"video_{index:03d}.mp4 <- {rows[index]['prompt']}")
        return

    lock_handle = acquire_shard_lock(output_dir, args.shard_index)
    try:
        if not args.overwrite and all(is_complete(output_dir / f"video_{index:03d}.mp4") for index in indices):
            print(f"[{SCRIPT_NAME}] shard {args.shard_index} is already complete; exiting", flush=True)
            return

        pipeline = build_pipeline(args)
        manifest_path = output_dir / f"manifest.shard_{args.shard_index}.jsonl"
        tiling_config = TilingConfig.default()
        video_chunks = get_video_chunks_number(args.num_frames, tiling_config)
        params = LTX_2_3_HQ_PARAMS

        for index in indices:
            row = rows[index]
            output_path = output_dir / f"video_{index:03d}.mp4"
            if is_complete(output_path) and not args.overwrite:
                print(f"[{index + 1:03d}/{len(rows)}] skip {output_path.name}", flush=True)
                continue

            temp_path = output_dir / f".video_{index:03d}.partial.mp4"
            temp_path.unlink(missing_ok=True)
            seed = args.seed_base + index
            print(f"[{index + 1:03d}/{len(rows)}] seed={seed} prompt={row['prompt']!r}", flush=True)
            started = time.monotonic()
            with torch.inference_mode():
                video, _audio = pipeline(
                    prompt=row["prompt"],
                    negative_prompt=args.negative_prompt,
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
                )
                encode_video(
                    video=video,
                    fps=int(args.fps),
                    audio=None,
                    output_path=str(temp_path),
                    video_chunks_number=video_chunks,
                )
            if not is_complete(temp_path):
                raise RuntimeError(f"LTX HQ did not create a valid file: {temp_path}")
            temp_path.replace(output_path)
            elapsed = time.monotonic() - started
            append_manifest(
                manifest_path,
                {
                    "index": index,
                    "record_id": row.get("record_id"),
                    "prompt": row["prompt"],
                    "output": output_path.name,
                    "seed": seed,
                    "elapsed_seconds": round(elapsed, 3),
                    "pipeline": "TI2VidTwoStagesHQPipeline",
                    "num_inference_steps": args.num_inference_steps,
                    "stage_2_steps": 3,
                    "distilled_lora_strength_stage_1": args.distilled_lora_strength_stage_1,
                    "distilled_lora_strength_stage_2": args.distilled_lora_strength_stage_2,
                    "num_frames": args.num_frames,
                    "width": args.width,
                    "height": args.height,
                    "fps": args.fps,
                    "audio": False,
                    "persistent_weights": not args.reload_weights_each_video,
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                },
            )
            print(f"[{index + 1:03d}/{len(rows)}] saved {output_path.name} in {elapsed:.1f}s", flush=True)
    finally:
        lock_handle.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
