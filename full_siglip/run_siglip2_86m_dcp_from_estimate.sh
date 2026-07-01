#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
AUDIT_MODULE="${AUDIT_MODULE:-full_siglip.audit_siglip2_86m_quant_improvement_goal}"
AUDIT_JSON="${AUDIT_JSON:-full_siglip/build/siglip2_86m_quant_improvement_status.local_dcp_preflight.json}"
AUDIT_MD="${AUDIT_MD:-full_siglip/build/siglip2_86m_quant_improvement_status.local_dcp_preflight.md}"
AUDIT_BUILD_ROOT="${AUDIT_BUILD_ROOT:-full_siglip/build}"
AUDIT_FP32_REPORT="${AUDIT_FP32_REPORT:-full_siglip/build/static_siglip2_86m_patch16_224_fp32_baseline/fp32_static_imagenet_report.json}"
AUDIT_BASELINE_ESTIMATE="${AUDIT_BASELINE_ESTIMATE:-}"
AUDIT_BASELINE_DCP_TIMING="${AUDIT_BASELINE_DCP_TIMING:-}"
CLOCK_NS="${CLOCK_NS:-3.999}"
DCP_SUFFIX="${DCP_SUFFIX:-dcp_256mhz}"
TARGET_FPS="${TARGET_FPS:-1200}"
LOOP_TARGET_FPS="${LOOP_TARGET_FPS:-120}"
MVAU_WWIDTH_MAX="${MVAU_WWIDTH_MAX:-72}"
MVAU_HLS_NODES="${MVAU_HLS_NODES:-MVAU_0,MVAU_97,MVAU_98}"
MVAU_PUMPED_EXCLUDE_NODES="${MVAU_PUMPED_EXCLUDE_NODES:-}"
SIGLIP2_86M_STREAM_FEEDBACK="${SIGLIP2_86M_STREAM_FEEDBACK:-1}"
TEMP_DIR="${TEMP_DIR:-}"
LOG="${LOG:-${REPO}/full_siglip/logs/siglip2_86m_dcp_from_estimate.log}"
WEIGHT_BITS="${WEIGHT_BITS:-}"
ACT_BITS="${ACT_BITS:-}"
FINN_VIVADO_ROUTE_DIRECTIVE="${FINN_VIVADO_ROUTE_DIRECTIVE:-Explore}"
FINN_VIVADO_PHYS_OPT_DIRECTIVE="${FINN_VIVADO_PHYS_OPT_DIRECTIVE:-AggressiveExplore}"
FINN_VIVADO_POST_ROUTE_PHYS_OPT_DIRECTIVE="${FINN_VIVADO_POST_ROUTE_PHYS_OPT_DIRECTIVE:-ExploreWithAggressiveHoldFix}"
POSTROUTE_RECOVERY="${POSTROUTE_RECOVERY:-1}"
POSTROUTE_PHYS_OPT_DIRECTIVE="${POSTROUTE_PHYS_OPT_DIRECTIVE:-ExploreWithAggressiveHoldFix}"
POSTROUTE_ROUTE_DIRECTIVE="${POSTROUTE_ROUTE_DIRECTIVE:-Explore}"
POSTROUTE_FINAL_PHYS_OPT_DIRECTIVE="${POSTROUTE_FINAL_PHYS_OPT_DIRECTIVE:-ExploreWithAggressiveHoldFix}"
export DCP_SUFFIX
export FINN_VIVADO_ROUTE_DIRECTIVE
export FINN_VIVADO_PHYS_OPT_DIRECTIVE
export FINN_VIVADO_POST_ROUTE_PHYS_OPT_DIRECTIVE

explicit_preflight() {
  if [[ -z "${EST_DIR:-}" ]]; then
    return 1
  fi
  local build_path="${EST_DIR}"
  local name
  name="$(basename -- "${build_path}")"
  FOLDING_CONFIG="${FOLDING_CONFIG:-${build_path}/auto_folding_config.json}"
  if [[ -z "${INPUT_MODEL:-}" ]]; then
    if [[ "${name}" != *"_mlo_"* ]]; then
      echo "cannot infer QONNX export dir from explicit EST_DIR name: ${name}" >&2
      return 1
    fi
    local export_name
    export_name="${name%%_mlo_*}"
    local export_dir
    export_dir="$(dirname -- "${build_path}")/${export_name}"
    INPUT_MODEL="${INPUT_MODEL:-${export_dir}_split_skipconv/convert_to_hw_shuffle.onnx}"
  fi
  if [[ -z "${DCP_DIR:-}" ]]; then
    local dcp_suffix
    dcp_suffix="$(printf '%s' "${DCP_SUFFIX}" | sed 's/^_*//; s/_*$//')"
    if [[ "${name}" == *_est ]]; then
      DCP_DIR="$(dirname -- "${build_path}")/${name%_est}_${dcp_suffix}"
    else
      DCP_DIR="$(dirname -- "${build_path}")/${name}_${dcp_suffix}"
    fi
  fi
  [[ -f "${FOLDING_CONFIG}" ]] || {
    echo "missing explicit folding config ${FOLDING_CONFIG}" >&2
    return 1
  }
  [[ -f "${INPUT_MODEL}" ]] || {
    echo "missing explicit input model ${INPUT_MODEL}" >&2
    return 1
  }
  return 0
}

read_preflight() {
  "${PYTHON_BIN}" - "${AUDIT_JSON}" <<'PY'
import json
import os
import sys
from pathlib import Path

with open(sys.argv[1], "r", encoding="utf-8") as f:
    report = json.load(f)
estimate = report.get("best_estimate")
if not isinstance(estimate, dict):
    raise SystemExit("missing best_estimate")
if estimate.get("latency_lower_than_baseline") is not True:
    raise SystemExit(f"best estimate latency is not lower than W6A8 baseline: {estimate}")
if estimate.get("cycles_lower_than_baseline") is not True:
    raise SystemExit(f"best estimate cycles are not lower than W6A8 baseline: {estimate}")
build_dir = estimate.get("build_dir")
if not isinstance(build_dir, str) or not build_dir:
    raise SystemExit("missing best_estimate.build_dir")
build_path = Path(build_dir)
if not (build_path / "auto_folding_config.json").is_file():
    raise SystemExit(f"missing {build_path / 'auto_folding_config.json'}")
name = build_path.name
if "_mlo_" not in name:
    raise SystemExit(f"cannot infer QONNX export dir from estimate name: {name}")
export_name = name.split("_mlo_", 1)[0]
export_dir = build_path.parent / export_name
prep_dir = Path(str(export_dir) + "_split_skipconv")
input_model = prep_dir / "convert_to_hw_shuffle.onnx"
if not input_model.is_file():
    raise SystemExit(f"missing prepared graph {input_model}")
dcp_suffix = os.environ.get("DCP_SUFFIX", "dcp_256mhz").strip("_")
if name.endswith("_est"):
    dcp_dir = build_path.with_name(f"{name[:-4]}_{dcp_suffix}")
else:
    dcp_dir = build_path.with_name(f"{name}_{dcp_suffix}")
print(f"EST_DIR={build_path}")
print(f"FOLDING_CONFIG={build_path / 'auto_folding_config.json'}")
print(f"INPUT_MODEL={input_model}")
print(f"DCP_DIR={dcp_dir}")
PY
}

run_audit() {
  local json_out="$1"
  local md_out="$2"
  local audit_args=(
    -m "${AUDIT_MODULE}"
    --build-root "${AUDIT_BUILD_ROOT}"
    --fp32-report "${AUDIT_FP32_REPORT}"
    --clock-ns "${CLOCK_NS}"
    --json-out "${json_out}"
    --md-out "${md_out}"
  )
  if [[ -n "${AUDIT_BASELINE_ESTIMATE}" ]]; then
    audit_args+=(--baseline-estimate "${AUDIT_BASELINE_ESTIMATE}")
  fi
  if [[ -n "${AUDIT_BASELINE_DCP_TIMING}" ]]; then
    audit_args+=(--baseline-dcp-timing "${AUDIT_BASELINE_DCP_TIMING}")
  fi
  "${PYTHON_BIN}" "${audit_args[@]}" >/dev/null || true
}

audit_has_timing_clean_dcp() {
  local json_out="$1"
  "${PYTHON_BIN}" - "${json_out}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
with path.open("r", encoding="utf-8") as f:
    report = json.load(f)
raise SystemExit(0 if isinstance(report.get("timing_clean_dcp"), dict) else 1)
PY
}

prepare_postroute_scheduler_evidence() {
  local source_dir="$1"
  local postroute_dir="$2"
  "${PYTHON_BIN}" - "${source_dir}" "${postroute_dir}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

source_dir = Path(sys.argv[1])
postroute_dir = Path(sys.argv[2])
postroute_dir.mkdir(parents=True, exist_ok=True)

for name in (
    "siglip2_86m_overlapped_scheduler_spec.json",
    "overlapped_scheduler_materialization.json",
    "overlapped_scheduler_annotated.onnx",
):
    src = source_dir / name
    if src.is_file():
        shutil.copy2(src, postroute_dir / name)

mat_path = postroute_dir / "overlapped_scheduler_materialization.json"
if mat_path.is_file():
    with mat_path.open("r", encoding="utf-8") as f:
        materialization = json.load(f)
    if isinstance(materialization, dict):
        source = materialization.setdefault("source", {})
        if isinstance(source, dict):
            source["scheduler_spec"] = str(
                (postroute_dir / "siglip2_86m_overlapped_scheduler_spec.json").resolve()
            )
        annotated = postroute_dir / "overlapped_scheduler_annotated.onnx"
        if annotated.is_file():
            materialization["annotated_graph"] = str(annotated)
            materialization["annotated_graph_abs"] = str(annotated.resolve())
    mat_path.write_text(json.dumps(materialization, indent=2) + "\n", encoding="utf-8")
PY
}

run_postroute_recovery() {
  local postroute_dir="${DCP_DIR}_postroute_explorehold"
  local postroute_stitched="${postroute_dir}/stitched_ip"
  local routed_dcp="${DCP_DIR}/stitched_ip/finn_design_routed.dcp"
  local source_xdc="${DCP_DIR}/stitched_ip/finn_design.xdc"

  if [[ "${POSTROUTE_RECOVERY}" != "1" ]]; then
    echo "postroute_recovery=disabled" | tee -a "${LOG}"
    return 1
  fi
  if [[ ! -s "${routed_dcp}" || ! -f "${source_xdc}" ]]; then
    echo "postroute_recovery=skipped missing routed_dcp_or_xdc" | tee -a "${LOG}"
    return 1
  fi
  mkdir -p "${postroute_stitched}"
  prepare_postroute_scheduler_evidence "${DCP_DIR}" "${postroute_dir}"
  echo "=== Running 86M post-route recovery ===" | tee -a "${LOG}"
  echo "postroute_output=${postroute_dir}" | tee -a "${LOG}"
  echo "postroute_phys_opt_directive=${POSTROUTE_PHYS_OPT_DIRECTIVE}" | tee -a "${LOG}"
  echo "postroute_route_directive=${POSTROUTE_ROUTE_DIRECTIVE}" | tee -a "${LOG}"
  echo "postroute_final_phys_opt_directive=${POSTROUTE_FINAL_PHYS_OPT_DIRECTIVE}" | tee -a "${LOG}"
  vivado -mode batch \
    -source full_siglip/postroute_physopt_stitched_ip_ooc.tcl \
    -tclargs \
      "${routed_dcp}" \
      "${source_xdc}" \
      "${postroute_stitched}" \
      "${CLOCK_NS}" \
      "${POSTROUTE_PHYS_OPT_DIRECTIVE}" \
      "${POSTROUTE_ROUTE_DIRECTIVE}" \
      "${POSTROUTE_FINAL_PHYS_OPT_DIRECTIVE}" \
    2>&1 | tee -a "${LOG}"
}

if ! explicit_preflight; then
  run_audit "${AUDIT_JSON}" "${AUDIT_MD}"

  if ! preflight_output="$(read_preflight)"; then
    exit 1
  fi
  eval "${preflight_output}"
fi

if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
  printf 'EST_DIR=%s\n' "${EST_DIR}"
  printf 'FOLDING_CONFIG=%s\n' "${FOLDING_CONFIG}"
  printf 'INPUT_MODEL=%s\n' "${INPUT_MODEL}"
  printf 'DCP_DIR=%s\n' "${DCP_DIR}"
  exit 0
fi

scheduler_meta() {
  local key="$1"
  local fallback="$2"
  "${PYTHON_BIN}" - "${EST_DIR}/siglip2_86m_overlapped_scheduler_spec.json" "${key}" "${fallback}" <<'PY'
import json
import sys
from pathlib import Path

path, key, fallback = sys.argv[1:4]
if not Path(path).is_file():
    print(fallback)
    raise SystemExit(0)
with open(path, "r", encoding="utf-8") as f:
    report = json.load(f)
scope = report.get("exact_scope") if isinstance(report, dict) else {}
scope = scope if isinstance(scope, dict) else {}
print(scope.get(key, fallback))
PY
}
WEIGHT_BITS="${WEIGHT_BITS:-$(scheduler_meta weight_bits 6)}"
ACT_BITS="${ACT_BITS:-$(scheduler_meta activation_bits 8)}"
if [[ -z "${TEMP_DIR}" ]]; then
  temp_name="$(basename -- "${DCP_DIR}")"
  TEMP_DIR="${REPO}/scripts_finn/finn_temp_${temp_name}"
fi

mkdir -p "${TEMP_DIR}" "$(dirname -- "${LOG}")"

# Xilinx settings scripts reference optional shell variables such as PYTHONPATH.
set +u
# shellcheck disable=SC1091
source "${REPO}/vivado-env-setup-vck190.sh"
set -u
export FINN_HOST_BUILD_DIR="${TEMP_DIR}"
export IMAGENET_ROOT="${IMAGENET_ROOT:-/proj/xlabs_t3/users/ml-workspace/datasets/imagenet/raw-images/ILSVRC2012}"

echo "=== Running 86M DCP from estimate ===" | tee -a "${LOG}"
echo "input=${INPUT_MODEL}" | tee -a "${LOG}"
echo "estimate=${EST_DIR}" | tee -a "${LOG}"
echo "folding=${FOLDING_CONFIG}" | tee -a "${LOG}"
echo "output=${DCP_DIR}" | tee -a "${LOG}"
echo "vivado_route_directive=${FINN_VIVADO_ROUTE_DIRECTIVE}" | tee -a "${LOG}"
echo "vivado_phys_opt_directive=${FINN_VIVADO_PHYS_OPT_DIRECTIVE}" | tee -a "${LOG}"
echo "vivado_post_route_phys_opt_directive=${FINN_VIVADO_POST_ROUTE_PHYS_OPT_DIRECTIVE}" | tee -a "${LOG}"

build_args=(
  python -m full_siglip.build_static \
  --input "${INPUT_MODEL}" \
  --output-dir "${DCP_DIR}" \
  --mode dcp \
  --board VCK190 \
  --clock-ns "${CLOCK_NS}" \
  --target-fps "${TARGET_FPS}" \
  --mlo \
  --depth 12 \
  --weight-bits "${WEIGHT_BITS}" \
  --act-bits "${ACT_BITS}" \
  --mvau-wwidth-max "${MVAU_WWIDTH_MAX}" \
  --mvau-hls-nodes "${MVAU_HLS_NODES}" \
  --folding-config-file "${FOLDING_CONFIG}" \
  --mvau-pumped-compute \
  --no-auto-fifo-depths \
  --no-save-intermediate-models \
  --output-mode embedding
)

if [[ "${LOOP_TARGET_FPS}" != "0" && -n "${LOOP_TARGET_FPS}" ]]; then
  build_args+=(--loop-target-fps "${LOOP_TARGET_FPS}")
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

FINN_DOCKER_PREBUILT=1 FINN_SKIP_DEP_REPOS=1 ./run-docker.sh \
  "${build_args[@]}" \
  2>&1 | tee -a "${LOG}"

run_audit \
  full_siglip/build/siglip2_86m_quant_improvement_status.after_dcp.json \
  full_siglip/build/siglip2_86m_quant_improvement_status.after_dcp.md

if ! audit_has_timing_clean_dcp full_siglip/build/siglip2_86m_quant_improvement_status.after_dcp.json; then
  if run_postroute_recovery; then
    run_audit \
      full_siglip/build/siglip2_86m_quant_improvement_status.after_postroute.json \
      full_siglip/build/siglip2_86m_quant_improvement_status.after_postroute.md
  fi
fi
