#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export HF_HOME="${HF_HOME:-/data/chenjiayu/wenbiao_zhao/hf_home}"

model_dir="${MODEL_DIR:-models/ltx-2.3}"
gemma_dir="${GEMMA_DIR:-models/gemma-3-12b}"
output_path="${OUTPUT_PATH:-outputs/ltx-2.3-smoke.mp4}"
prompt="${PROMPT:-A red fox walks through a snowy pine forest while soft wind moves the branches. The camera slowly tracks alongside the fox. Natural forest ambience and quiet footsteps are audible.}"

mkdir -p "$(dirname "$output_path")"

uv run --frozen --no-dev python -m ltx_pipelines.distilled \
  --distilled-checkpoint-path "$model_dir/ltx-2.3-22b-distilled-1.1.safetensors" \
  --spatial-upsampler-path "$model_dir/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" \
  --gemma-root "$gemma_dir" \
  --height 512 \
  --width 768 \
  --num-frames 49 \
  --frame-rate 24 \
  --seed 42 \
  --output-path "$output_path" \
  --prompt "$prompt"
