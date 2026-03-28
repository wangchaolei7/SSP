#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
NPROC="${NPROC:-6}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29530}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-29-12_12-52}"
SAVE_DIR="${SAVE_DIR:-/data1/wangcl/project/SSP/kitti360}"
CHECKPOINT_FOLDER="${CHECKPOINT_FOLDER:-video/}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/data1/wangcl/project/SSP/kitti360/video/29-12_12-52/epoch_0010.pth.tar}"
SPLIT="${SPLIT:-val}"
METRICS_ONLY="${METRICS_ONLY:-1}"
EXTRA_ARGS="${EXTRA_ARGS:---no-gif}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-cross_domain}"
TARGETS="${TARGETS:-uavid,vspw}"
TORCHRUN_BIN="${TORCHRUN_BIN:-}"

if [[ -z "${TORCHRUN_BIN}" ]]; then
  if command -v torchrun >/dev/null 2>&1; then
    TORCHRUN_BIN="$(command -v torchrun)"
  elif [[ -x /usr/local/anaconda3/envs/SSP/bin/torchrun ]]; then
    TORCHRUN_BIN="/usr/local/anaconda3/envs/SSP/bin/torchrun"
  else
    echo "torchrun not found. Set TORCHRUN_BIN or activate the SSP environment." >&2
    exit 1
  fi
fi

run_eval() {
  local name="$1"
  local master_port="$2"
  shift 2
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
  "${TORCHRUN_BIN}" --nproc_per_node="${NPROC}" --master_port="${master_port}" -m eval.vis.video "${CHECKPOINT_NAME}" \
    --save-dir "${SAVE_DIR}" \
    --checkpoint-folder "${CHECKPOINT_FOLDER}" \
    --split "${SPLIT}" \
    --output-subdir "${name}" \
    "${common_args[@]}" \
    "$@"
}

if [[ ",${TARGETS}," == *",uavid,"* ]]; then
  run_eval "${OUTPUT_PREFIX}/uavid_val" "${MASTER_PORT_BASE}" --dataset uavid_crossdomain
fi

if [[ ",${TARGETS}," == *",vspw,"* ]]; then
  run_eval "${OUTPUT_PREFIX}/vspw_val" "$((MASTER_PORT_BASE + 1))" --dataset vspw_crossdomain
fi
