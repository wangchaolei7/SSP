#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TARGETS="${TARGETS:-uavid,vspw}"

echo "[eval] camvid source"
TARGETS="${TARGETS}" bash "${SCRIPT_DIR}/run_eval_cross_domain_camvid.sh"

echo "[eval] apollo source"
TARGETS="${TARGETS}" bash "${SCRIPT_DIR}/run_eval_cross_domain_apollo.sh"

echo "[eval] kitti360 source"
TARGETS="${TARGETS}" bash "${SCRIPT_DIR}/run_eval_cross_domain_kitti360.sh"
