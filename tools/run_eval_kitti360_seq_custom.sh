#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Edit parameters here.
NPROC="${NPROC:-6}"
CHECKPOINT_NAME="29-12_12-52"
SAVE_DIR="/data1/wangcl/project/SSP/kitti360"
CHECKPOINT_FOLDER="video/"
CHECKPOINT_PATH="/data1/wangcl/project/SSP/kitti360/video/29-12_12-52/epoch_0010.pth.tar"
SPLIT="val"
# Set to 1 for metrics only (no visualization output).
METRICS_ONLY=1
# Extra args passed to eval.vis.video (leave empty if unused).
EXTRA_ARGS="--no-gif"

CITY_ROOT_ORIGIN="${CITY_ROOT_ORIGIN:-/home/wangcl/data/open_video_DGSS/cityscapes_sequence}"
CITY_ROOT_CORR="${CITY_ROOT_CORR:-/home/wangcl/data/open_video_DGSS/cityscapes_sequence/leftImg8bit_sequence_Corruptions}"
CITY_ROOT_LABELS="${CITY_ROOT_LABELS:-/home/wangcl/data/open_video_DGSS/cityscapes_sequence/gtFine}"

# Cityscapes sequences to infer (full paths). Edit as needed.
CITY_SEQ_ORIGIN_PATHS=(
  "/home/wangcl/data/open_video_DGSS/cityscapes_sequence/origin_leftImg8bit_sequence/frankfurt/seq1"
)
CITY_SEQ_CORR_PATHS=(
  "/home/wangcl/data/open_video_DGSS/cityscapes_sequence/leftImg8bit_sequence_Corruptions/fog/munster/seq35"
  "/home/wangcl/data/open_video_DGSS/cityscapes_sequence/leftImg8bit_sequence_Corruptions/frost/lindau/seq2"
  "/home/wangcl/data/open_video_DGSS/cityscapes_sequence/leftImg8bit_sequence_Corruptions/snow/frankfurt/seq108"
  "/home/wangcl/data/open_video_DGSS/cityscapes_sequence/leftImg8bit_sequence_Corruptions/spatter/munster/seq136"
)

CAMVID_SEQ_PATH="${CAMVID_SEQ_PATH:-/home/wangcl/data/open_video_DGSS/CamVid/val/images/Seq05VD}"
CAMVID_SEQ_NAME="$(basename "${CAMVID_SEQ_PATH}")"

APOLLO_SEQ_PATH_1="${APOLLO_SEQ_PATH_1:-/home/wangcl/data/open_video_DGSS/ApolloScape/val/ColorImage/Record046}"
APOLLO_SEQ_PATH_2="${APOLLO_SEQ_PATH_2:-/home/wangcl/data/open_video_DGSS/ApolloScape/val/ColorImage/Record053}"
APOLLO_SEQ_NAME_1="$(basename "${APOLLO_SEQ_PATH_1}")"
APOLLO_SEQ_NAME_2="$(basename "${APOLLO_SEQ_PATH_2}")"

#if [[ "${METRICS_ONLY}" == "0" ]]; then 0或1表示是否只计算指标
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
  torchrun --nproc_per_node="${NPROC}" -m eval.vis.video "${CHECKPOINT_NAME}" \
    --save-dir "${SAVE_DIR}" \
    --checkpoint-folder "${CHECKPOINT_FOLDER}" \
    --split "${SPLIT}" \
    --output-subdir "${name}" \
    --write-res \
    "${common_args[@]}" \
    "$@"
}

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

# run_eval "camvid_${CAMVID_SEQ_NAME}" \
#   --dataset camvid \
#   --city-seq "${CAMVID_SEQ_PATH}"

# run_eval "apollo_${APOLLO_SEQ_NAME_1}" \
#   --dataset apolloscape \
#   --city-seq "${APOLLO_SEQ_PATH_1}"

# run_eval "apollo_${APOLLO_SEQ_NAME_2}" \
#   --dataset apolloscape \
#   --city-seq "${APOLLO_SEQ_PATH_2}"
