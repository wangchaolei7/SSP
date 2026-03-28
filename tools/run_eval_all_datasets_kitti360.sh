#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Edit parameters here.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC=${NPROC:-1}
CHECKPOINT_NAME="29-12_12-52"
SAVE_DIR="/data1/wangcl/project/SSP/kitti360"
CHECKPOINT_FOLDER="video/"
CHECKPOINT_PATH="/data1/wangcl/project/SSP/kitti360/video/29-12_12-52/epoch_0010.pth.tar"
SPLIT="val"
# Set to 1 for metrics only (no visualization output).
METRICS_ONLY=0
# Extra args passed to eval.vis.video (leave empty if unused).
MAX_FRAMES=100
EXTRA_ARGS="--report-fps --max-frames ${MAX_FRAMES}"
OUTPUT_BASE="FPS_cityscapes_origin"

CITY_ROOT_ORIGIN="${CITY_ROOT_ORIGIN:-/home/wangcl/data/open_video_DGSS/cityscapes_sequence}"
CITY_ROOT_CORR="${CITY_ROOT_CORR:-/home/wangcl/data/open_video_DGSS/cityscapes_sequence/leftImg8bit_sequence_Corruptions}"
CITY_ROOT_LABELS="${CITY_ROOT_LABELS:-/home/wangcl/data/open_video_DGSS/cityscapes_sequence/gtFine}"

run_eval() {
  local name="$1"
  shift
  echo "[eval] ${name}"
  local common_args=()
  if [[ "${METRICS_ONLY}" == "1" ]]; then
    common_args+=(--metrics-only)
  fi
  if [[ -n "${CHECKPOINT_PATH}" ]]; then
    common_args+=(--checkpoint-path "${CHECKPOINT_PATH}")
  fi
  if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    common_args+=(${EXTRA_ARGS})
  fi
  torchrun --nproc_per_node="${NPROC}" -m eval.vis.video "${CHECKPOINT_NAME}" \
    --save-dir "${SAVE_DIR}" \
    --checkpoint-folder "${CHECKPOINT_FOLDER}" \
    --split "${SPLIT}" \
    --output-subdir "${name}" \
    "${common_args[@]}" \
    "$@"
}

# run_eval "apollo" --dataset apolloscape
# run_eval "camvid" --dataset camvid
# run_eval "kitti360" --dataset kitti360

run_eval "${OUTPUT_BASE}/size_1024x2048" \
  --corruption origin_leftImg8bit_sequence \
  --city-root-images "${CITY_ROOT_ORIGIN}" \
  --city-root-labels "${CITY_ROOT_LABELS}" \
  --input-size 1024 2048

run_eval "${OUTPUT_BASE}/size_512x1024" \
  --corruption origin_leftImg8bit_sequence \
  --city-root-images "${CITY_ROOT_ORIGIN}" \
  --city-root-labels "${CITY_ROOT_LABELS}" \
  --input-size 512 1024
