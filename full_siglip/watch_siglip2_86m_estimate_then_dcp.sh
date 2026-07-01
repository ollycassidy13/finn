#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
POLL_SECONDS="${POLL_SECONDS:-300}"
CLOCK_NS="${CLOCK_NS:-3.999}"
DCP_SESSION="${DCP_SESSION:-siglip2-86m-dcp}"
WATCH_LOG="${WATCH_LOG:-${REPO}/full_siglip/logs/siglip2_86m_estimate_then_dcp_watch.log}"
AUDIT_JSON="${AUDIT_JSON:-${REPO}/full_siglip/build/siglip2_86m_quant_improvement_status.dcpwatch.json}"
AUDIT_MD="${AUDIT_MD:-${REPO}/full_siglip/build/siglip2_86m_quant_improvement_status.dcpwatch.md}"

mkdir -p "$(dirname -- "${WATCH_LOG}")" "${REPO}/full_siglip/build"
cd "${REPO}"

estimate_ready() {
  "${PYTHON_BIN}" - "${AUDIT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

with open(sys.argv[1], "r", encoding="utf-8") as f:
    report = json.load(f)
estimate = report.get("best_estimate")
if not isinstance(estimate, dict):
    print("ready=0 reason=missing_best_estimate")
    raise SystemExit(1)
if estimate.get("latency_lower_than_baseline") is not True:
    print(f"ready=0 reason=best_estimate_latency_not_lower_than_w6a8 estimate={estimate}")
    raise SystemExit(1)
if estimate.get("cycles_lower_than_baseline") is not True:
    print(f"ready=0 reason=best_estimate_cycles_not_lower_than_w6a8 estimate={estimate}")
    raise SystemExit(1)
build_dir = estimate.get("build_dir")
if not isinstance(build_dir, str):
    print("ready=0 reason=missing_build_dir")
    raise SystemExit(1)
if not (Path(build_dir) / "auto_folding_config.json").is_file():
    print(f"ready=0 reason=missing_auto_folding_config build_dir={build_dir}")
    raise SystemExit(1)
print(f"ready=1 build_dir={build_dir} cycles={estimate.get('total_cycles')} latency_ms={estimate.get('latency_ms')}")
PY
}

{
  echo "$(date -Is) waiting for lower-than-W6A8 86M FINN estimate"
  while true; do
    "${PYTHON_BIN}" -m full_siglip.audit_siglip2_86m_quant_improvement_goal \
      --clock-ns "${CLOCK_NS}" \
      --json-out "${AUDIT_JSON}" \
      --md-out "${AUDIT_MD}" >/dev/null || true
    if ready_line="$(estimate_ready 2>/dev/null)"; then
      echo "$(date -Is) ${ready_line}"
      break
    fi
    echo "$(date -Is) $(estimate_ready || true)"
    sleep "${POLL_SECONDS}"
  done

  if tmux has-session -t "${DCP_SESSION}" 2>/dev/null; then
    echo "$(date -Is) DCP session already exists: ${DCP_SESSION}"
    exit 0
  fi

  echo "$(date -Is) launching guarded DCP session ${DCP_SESSION}"
  tmux new -d -s "${DCP_SESSION}" \
    "cd '${REPO}' && CLOCK_NS='${CLOCK_NS}' full_siglip/run_siglip2_86m_dcp_from_estimate.sh"
} 2>&1 | tee -a "${WATCH_LOG}"
