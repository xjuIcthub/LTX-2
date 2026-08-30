from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from datetime import timedelta
from pathlib import Path

import torch

from ltx_pipelines.multigpu.controller import MGPUController
from ltx_pipelines.ti2vid_two_stages_mgpu import TI2VidTwoStagesRunner
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, LTX_2_3_PARAMS


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAME = Path(__file__).name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all 256 VBench prompts with the persistent two-GPU LTX-2 pipeline."
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
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None, help="Exclusive end index; defaults to all prompts.")
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


def validate_args(args: argparse.Namespace, row_count: int) -> tuple[int, int]:
    if torch.cuda.device_count() != 2 and not args.dry_run:
        raise RuntimeError(
            f"Expected exactly two visible GPUs, found {torch.cuda.device_count()}. "
            "Launch with CUDA_VISIBLE_DEVICES=0,1."
        )
    if args.height % 64 or args.width % 64:
        raise ValueError("The two-stage LTX pipeline requires height and width divisible by 64")
    if (args.num_frames - 1) % 8:
        raise ValueError("LTX num_frames must satisfy num_frames = 8 * k + 1")
    start = max(0, args.start_index)
    end = row_count if args.end_index is None else min(row_count, args.end_index)
    if start >= end:
        raise ValueError(f"Empty index range [{start}, {end})")
    for path in (args.checkpoint_path, args.distilled_lora_path, args.spatial_upsampler_path, args.gemma_root):
        if not path.exists() and not args.dry_run:
            raise FileNotFoundError(path)
    return start, end


def main() -> None:
    args = parse_args()
    rows = load_rows(args.prompts_csv.resolve())
    start, end = validate_args(args, len(rows))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    print(
        f"[{SCRIPT_NAME}] prompts={len(rows)} range=[{start}, {end}) "
        f"frames={args.num_frames} size={args.width}x{args.height} fps={args.fps:g} audio=off"
    )
    if args.dry_run:
        for index in range(start, min(end, start + 3)):
            print(f"video_{index:03d}.mp4 <- {rows[index]['prompt']}")
        return

    vae_queue = torch.multiprocessing.get_context("spawn").SimpleQueue()
    controller = MGPUController(TI2VidTwoStagesRunner, num_gpus=2)
    controller.start(
        timeout=timedelta(minutes=45),
        checkpoint_path=str(args.checkpoint_path.resolve()),
        gemma_root=str(args.gemma_root.resolve()),
        spatial_upsampler_path=str(args.spatial_upsampler_path.resolve()),
        vae_queue=vae_queue,
        distilled_lora_path=str(args.distilled_lora_path.resolve()),
    )
    try:
        for index in range(start, end):
            row = rows[index]
            output_path = output_dir / f"video_{index:03d}.mp4"
            if is_complete(output_path) and not args.overwrite:
                print(f"[{index + 1:03d}/{len(rows)}] skip {output_path.name}")
                continue

            temp_path = output_dir / f".video_{index:03d}.partial.mp4"
            temp_path.unlink(missing_ok=True)
            seed = args.seed_base + index
            print(f"[{index + 1:03d}/{len(rows)}] seed={seed} prompt={row['prompt']!r}", flush=True)
            started = time.monotonic()
            stream = controller.stream(
                output_path=str(temp_path),
                prompt=row["prompt"],
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                seed=seed,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                frame_rate=args.fps,
                num_inference_steps=args.num_inference_steps,
                video_guider_params=LTX_2_3_PARAMS.video_guider_params,
                audio_guider_params=LTX_2_3_PARAMS.audio_guider_params,
                include_audio=False,
                images=[],
            )
            try:
                for _ in stream:
                    pass
            finally:
                stream.drain()
            if not is_complete(temp_path):
                raise RuntimeError(f"LTX did not create a valid file: {temp_path}")
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
                    "num_frames": args.num_frames,
                    "width": args.width,
                    "height": args.height,
                    "fps": args.fps,
                    "audio": False,
                },
            )
            print(f"[{index + 1:03d}/{len(rows)}] saved {output_path.name} in {elapsed:.1f}s", flush=True)
    finally:
        controller.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
