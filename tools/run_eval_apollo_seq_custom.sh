#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Edit parameters here.
NPROC="${NPROC:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"
CHECKPOINT_NAME="26-12_00-23"
SAVE_DIR="/data1/wangcl/project/SSP/apollo"
CHECKPOINT_FOLDER="video/"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
SPLIT="val"
# Set to 1 for metrics only (no visualization output).
METRICS_ONLY=1
# Extra args passed to eval.vis.video (leave empty if unused).
EXTRA_ARGS="--no-gif"

CAMVID_SEQ_PATH="${CAMVID_SEQ_PATH:-/home/wangcl/data/open_video_DGSS/CamVid/val/images/Seq05VD}"
KITTI360_SEQ_PATHS=(
  "/home/wangcl/data/open_video_DGSS/kitti360_sequence/val/data_2d_raw/2013_05_28_drive_0007_sync/image_00"
  "/home/wangcl/data/open_video_DGSS/kitti360_sequence/val/data_2d_raw/2013_05_28_drive_0009_sync/image_00"
)

CITY_ROOT_ORIGIN="${CITY_ROOT_ORIGIN:-/home/wangcl/data/open_video_DGSS/cityscapes_sequence}"
CITY_ROOT_CORR="${CITY_ROOT_CORR:-/home/wangcl/data/open_video_DGSS/cityscapes_sequence/leftImg8bit_sequence_Corruptions}"
CITY_ROOT_LABELS="${CITY_ROOT_LABELS:-/home/wangcl/data/open_video_DGSS/cityscapes_sequence/gtFine}"

CITY_SEQ_ORIGIN_PATHS=(
  "/data1/wangcl/dataset/open_video_DGSS/cityscapes_sequence/origin_leftImg8bit_sequence/munster/seq2"
)
CITY_SEQ_CORR_PATHS=(
  "/data1/wangcl/dataset/open_video_DGSS/cityscapes_sequence/leftImg8bit_sequence_Corruptions/fog/lindau/seq42"
  "/data1/wangcl/dataset/open_video_DGSS/cityscapes_sequence/leftImg8bit_sequence_Corruptions/frost/munster/seq35"
  "/data1/wangcl/dataset/open_video_DGSS/cityscapes_sequence/leftImg8bit_sequence_Corruptions/snow/munster/seq107"
  "/data1/wangcl/dataset/open_video_DGSS/cityscapes_sequence/leftImg8bit_sequence_Corruptions/spatter/munster/seq88"
)


seq_key_from_path() {
  local p="$1"
  local base
  base="$(basename "${p}")"
  if [[ "${base}" == "image_00" || "${base}" == "image_01" ]]; then
    basename "$(dirname "${p}")"
  else
    echo "${base}"
  fi
}

run_eval() {
  local name="$1"
  shift
  echo "[eval] ${name}"
  local common_args=()
  if [[ "${METRICS_ONLY}" == "0" ]]; then
    common_args+=(--metrics-only)
  fi
  if [[ -n "${CHECKPOINT_PATH}" ]]; then
    common_args+=(--checkpoint-path "${CHECKPOINT_PATH}")
  fi
  if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    common_args+=(${EXTRA_ARGS})
  fi
  torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" -m eval.vis.video "${CHECKPOINT_NAME}" \
    --save-dir "${SAVE_DIR}" \
    --checkpoint-folder "${CHECKPOINT_FOLDER}" \
    --split "${SPLIT}" \
    --output-subdir "${name}" \
    "${common_args[@]}" \
    "$@"
}

# camvid_seq="$(seq_key_from_path "${CAMVID_SEQ_PATH}")"
# run_eval "camvid_${camvid_seq}" \
#   --dataset camvid \
#   --city-seq "${camvid_seq}"

# for seq_path in "${KITTI360_SEQ_PATHS[@]}"; do
#   kitti_seq="$(seq_key_from_path "${seq_path}")"
#   run_eval "kitti360_${kitti_seq}" \
#     --dataset kitti360 \
#     --city-seq "${kitti_seq}"
# done

for seq_path in "${CITY_SEQ_ORIGIN_PATHS[@]}"; do
  seq_city="$(basename "$(dirname "${seq_path}")")"
  seq_name="$(basename "${seq_path}")"
  city_seq="${seq_city}/${seq_name}"
  run_eval "cityscapes_origin_${seq_city}_${seq_name}" \
    --corruption origin_leftImg8bit_sequence \
    --city-root-images "${CITY_ROOT_ORIGIN}" \
    --city-root-labels "${CITY_ROOT_LABELS}" \
    --city-seq "${city_seq}"
done

for seq_path in "${CITY_SEQ_CORR_PATHS[@]}"; do
  corruption="$(basename "$(dirname "$(dirname "${seq_path}")")")"
  seq_city="$(basename "$(dirname "${seq_path}")")"
  seq_name="$(basename "${seq_path}")"
  city_seq="${seq_city}/${seq_name}"
  run_eval "cityscapes_${corruption}_${seq_city}_${seq_name}" \
    --dataset cityscapes_seq_corrupt \
    --corruption "${corruption}" \
    --city-root-images "${CITY_ROOT_CORR}" \
    --city-root-labels "${CITY_ROOT_LABELS}" \
    --city-seq "${city_seq}"
done
