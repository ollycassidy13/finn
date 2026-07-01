#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
AUDIT_MODULE="${AUDIT_MODULE:-full_siglip.audit_siglip2_86m_quant_improvement_goal}"
AUDIT_JSON="${AUDIT_JSON:-${REPO}/full_siglip/build/siglip2_86m_quant_improvement_status.estimate_select.json}"
AUDIT_MD="${AUDIT_MD:-${REPO}/full_siglip/build/siglip2_86m_quant_improvement_status.estimate_select.md}"
AUDIT_BUILD_ROOT="${AUDIT_BUILD_ROOT:-full_siglip/build}"
AUDIT_FP32_REPORT="${AUDIT_FP32_REPORT:-full_siglip/build/static_siglip2_86m_patch16_224_fp32_baseline/fp32_static_imagenet_report.json}"

select_qonnx() {
  if [[ -n "${QONNX:-}" ]]; then
    printf '%s\n' "${QONNX}"
    return
  fi
  "${PYTHON_BIN}" -m "${AUDIT_MODULE}" \
    --build-root "${AUDIT_BUILD_ROOT}" \
    --fp32-report "${AUDIT_FP32_REPORT}" \
    --json-out "${AUDIT_JSON}" \
    --md-out "${AUDIT_MD}" >/dev/null || true
  local audit_qonnx
  audit_qonnx="$("${PYTHON_BIN}" - "${AUDIT_JSON}" <<'PY' || true
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
with path.open("r", encoding="utf-8") as f:
    report = json.load(f)
best = report.get("best_qat_report")
if not isinstance(best, dict) or best.get("ok") is not True:
    raise SystemExit(1)
for raw in best.get("qonnx_files") or []:
    qonnx = Path(raw)
    if qonnx.is_file():
        print(qonnx)
        raise SystemExit(0)
raise SystemExit(1)
PY
  )"
  if [[ -n "${audit_qonnx}" ]]; then
    printf '%s\n' "${audit_qonnx}"
    return
  fi
  echo "missing 86M accepted QONNX; set QONNX=... or wait for the 50k export" >&2
  return 1
}

QONNX="$(select_qonnx)"
if [[ ! -f "${QONNX}" ]]; then
  echo "selected 86M QONNX does not exist: ${QONNX}" >&2
  exit 1
fi
if [[ "${SELECT_QONNX_ONLY:-0}" == "1" ]]; then
  printf '%s\n' "${QONNX}"
  exit 0
fi
QONNX_DIR="$(dirname -- "${QONNX}")"
BASE_NAME="$(basename -- "${QONNX_DIR}")"
report_meta() {
  local key="$1"
  local fallback="$2"
  "${PYTHON_BIN}" - "${QONNX_DIR}/qat_report.json" "${key}" "${fallback}" <<'PY'
import json
import sys
from pathlib import Path

path, key, fallback = sys.argv[1:4]
if not Path(path).is_file():
    print(fallback)
    raise SystemExit(0)
with open(path, "r", encoding="utf-8") as f:
    report = json.load(f)
print(report.get(key, fallback))
PY
}
WEIGHT_BITS="${WEIGHT_BITS:-$(report_meta weight_bits 6)}"
ACT_BITS="${ACT_BITS:-$(report_meta act_bits 8)}"
PREP_DIR="${PREP_DIR:-${QONNX_DIR}_split_skipconv}"
TEMP_DIR="${TEMP_DIR:-${REPO}/scripts_finn/finn_temp_${BASE_NAME}_mlo_est}"
LOG="${LOG:-${REPO}/full_siglip/logs/${BASE_NAME}_finn_estimate.log}"
CLOCK_NS="${CLOCK_NS:-3.9}"
TARGET_FPS="${TARGET_FPS:-1200}"
LOOP_TARGET_FPS="${LOOP_TARGET_FPS:-120}"
MVAU_WWIDTH_MAX="${MVAU_WWIDTH_MAX:-72}"
MVAU_HLS_NODES="${MVAU_HLS_NODES:-MVAU_0,MVAU_97,MVAU_98}"
MVAU_PUMPED_EXCLUDE_NODES="${MVAU_PUMPED_EXCLUDE_NODES:-}"
SIGLIP2_86M_STREAM_FEEDBACK="${SIGLIP2_86M_STREAM_FEEDBACK:-1}"
FOLDING_CONFIG_FILE="${FOLDING_CONFIG_FILE:-}"

if [[ "${SIGLIP2_86M_STREAM_FEEDBACK}" == "1" ]]; then
  EST_DIR="${EST_DIR:-${QONNX_DIR}_mlo_fps${TARGET_FPS}_loopfps${LOOP_TARGET_FPS}_w${MVAU_WWIDTH_MAX}_hlstop_stream_feedback_est}"
else
  EST_DIR="${EST_DIR:-${QONNX_DIR}_mlo_fps${TARGET_FPS}_w${MVAU_WWIDTH_MAX}_pumped_embedding_est}"
fi

mkdir -p "${TEMP_DIR}" "$(dirname -- "${LOG}")"

# Xilinx settings scripts reference optional shell variables such as PYTHONPATH.
set +u
# shellcheck disable=SC1091
source "${REPO}/vivado-env-setup-vck190.sh"
set -u
export FINN_HOST_BUILD_DIR="${TEMP_DIR}"
export IMAGENET_ROOT="${IMAGENET_ROOT:-/proj/xlabs_t3/users/ml-workspace/datasets/imagenet/raw-images/ILSVRC2012}"

if [[ ! -f "${PREP_DIR}/convert_to_hw_shuffle.onnx" ]]; then
  echo "=== Preparing 86M QONNX for FINN: ${QONNX} -> ${PREP_DIR} ===" | tee -a "${LOG}"
  FINN_DOCKER_PREBUILT=1 FINN_SKIP_DEP_REPOS=1 ./run-docker.sh \
    python -m full_siglip.prepare_model \
      --input "${QONNX}" \
      --output-dir "${PREP_DIR}" \
      --board VCK190 \
      --clock-ns "${CLOCK_NS}" \
      --target-fps "${TARGET_FPS}" \
      --stop-after dataflow_partition \
      --split-streamline \
      --split-convert-to-hw \
      --checkpoint-stage convert_to_hw_shuffle \
      --checkpoint-stage dataflow_partition \
      --no-save-checkpoints \
      --skip-conv-bias-extract \
    2>&1 | tee -a "${LOG}"
else
  echo "=== Reusing prepared FINN graph: ${PREP_DIR}/convert_to_hw_shuffle.onnx ===" | tee -a "${LOG}"
fi

build_args=(
  python -m full_siglip.build_static
  --input "${PREP_DIR}/convert_to_hw_shuffle.onnx"
  --output-dir "${EST_DIR}"
  --mode estimate
  --board VCK190
  --clock-ns "${CLOCK_NS}"
  --target-fps "${TARGET_FPS}"
  --mlo
  --depth 12
  --weight-bits "${WEIGHT_BITS}"
  --act-bits "${ACT_BITS}"
  --mvau-wwidth-max "${MVAU_WWIDTH_MAX}"
  --mvau-hls-nodes "${MVAU_HLS_NODES}"
  --mvau-pumped-compute
  --no-auto-fifo-depths
  --no-save-intermediate-models
  --output-mode embedding
)

if [[ "${LOOP_TARGET_FPS}" != "0" && -n "${LOOP_TARGET_FPS}" ]]; then
  build_args+=(--loop-target-fps "${LOOP_TARGET_FPS}")
fi

if [[ -n "${FOLDING_CONFIG_FILE}" ]]; then
  build_args+=(--folding-config-file "${FOLDING_CONFIG_FILE}")
fi

if [[ -n "${MVAU_PUMPED_EXCLUDE_NODES}" ]]; then
  build_args+=(--mvau-pumped-exclude-nodes "${MVAU_PUMPED_EXCLUDE_NODES}")
fi

if [[ "${SIGLIP2_86M_STREAM_FEEDBACK}" == "1" ]]; then
  build_args+=(
    --generate-siglip2-86m-overlapped-scheduler-spec
    --overlapped-scheduler-implementation-kind builtin_stream_feedback
  )
fi

echo "=== Running 86M FINN estimate: ${EST_DIR} ===" | tee -a "${LOG}"
FINN_DOCKER_PREBUILT=1 FINN_SKIP_DEP_REPOS=1 ./run-docker.sh \
  "${build_args[@]}" \
  2>&1 | tee -a "${LOG}"

echo "=== 86M FINN estimate artifacts ===" | tee -a "${LOG}"
ls -lh "${EST_DIR}/report" 2>/dev/null | tee -a "${LOG}" || true
