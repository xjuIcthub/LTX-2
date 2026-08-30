#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export HF_HOME="${HF_HOME:-/data/chenjiayu/wenbiao_zhao/hf_home}"

uv_bin="${UV_BIN:-/data/chenjiayu/.local/bin/uv}"
model_dir="${MODEL_DIR:-models/ltx-2.3}"
gemma_dir="${GEMMA_DIR:-models/gemma-3-12b}"
output_path="${OUTPUT_PATH:-outputs/ltx-2.3-smoke.mp4}"
prompt="${PROMPT:-A red fox walks through a snowy pine forest while soft wind moves the branches. The camera slowly tracks alongside the fox. Natural forest ambience and quiet footsteps are audible.}"
offload_mode="${OFFLOAD_MODE:-cpu}"

mkdir -p "$(dirname "$output_path")"

distilled_checkpoint="$model_dir/ltx-2.3-22b-distilled-1.1.safetensors"
dev_checkpoint="$model_dir/ltx-2.3-22b-dev.safetensors"
distilled_lora="$model_dir/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
spatial_upsampler="$model_dir/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

if [[ -f "$distilled_checkpoint" ]]; then
  "$uv_bin" run --frozen --no-dev python -m ltx_pipelines.distilled \
    --distilled-checkpoint-path "$distilled_checkpoint" \
    --spatial-upsampler-path "$spatial_upsampler" \
    --gemma-root "$gemma_dir" \
    --height 512 \
    --width 768 \
    --num-frames 49 \
    --frame-rate 24 \
    --offload "$offload_mode" \
    --seed 42 \
    --output-path "$output_path" \
    --prompt "$prompt"
else
  "$uv_bin" run --frozen --no-dev python -m ltx_pipelines.ti2vid_two_stages \
    --checkpoint-path "$dev_checkpoint" \
    --distilled-lora "$distilled_lora" 1.0 \
    --spatial-upsampler-path "$spatial_upsampler" \
    --gemma-root "$gemma_dir" \
    --height 512 \
    --width 768 \
    --num-frames 49 \
    --frame-rate 24 \
    --num-inference-steps 8 \
    --offload "$offload_mode" \
    --seed 42 \
    --output-path "$output_path" \
    --prompt "$prompt"
fi
