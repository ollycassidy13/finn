#!/usr/bin/env python3
"""Audit the current SigLIP2-86M W6A7 sub-30ms VCK190 objective."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any


DEFAULT_QAT_REPORT = Path(
    "full_siglip/build/"
    "static_qat_siglip2_86m_patch16_224_w6a7_qvlsq_featurehead_50k_50k_export/"
    "qat_report.json"
)
DEFAULT_ESTIMATE_DIR = Path(
    "full_siglip/build/"
    "static_qat_siglip2_86m_patch16_224_w6a7_qvlsq_featurehead_50k_50k_export"
    "_mlo_fps1200_loopfps120_w72_hlstop_stream_feedback_outershuffle12_bits"
    "_sub30_unpump235pe49_est"
)
DEFAULT_DCP_DIR = Path(
    "full_siglip/build/"
    "static_qat_siglip2_86m_patch16_224_w6a7_qvlsq_featurehead_50k_50k_export"
    "_mlo_fps1200_loopfps120_w72_hlstop_stream_feedback_outershuffle12_bits"
    "_sub30_unpump235pe49_dcp_250p06mhz_sub30_unpump235pe49"
)
DEFAULT_JSON_OUT = Path("full_siglip/build/siglip2_86m_sub30_goal_status.json")
DEFAULT_MD_OUT = Path("full_siglip/build/siglip2_86m_sub30_goal_status.md")

EXPECTED_MODEL_ID = "google/siglip2-base-patch16-224"
EXPECTED_WEIGHT_BITS = 6
EXPECTED_ACT_BITS = 7
EXPECTED_EDGE_BITS = 8
EXPECTED_EVAL_IMAGES = 50_000
EXPECTED_VISION_DEPTH = 12
EXPECTED_IMAGE_SIZE = 224
EXPECTED_PATCH_GRID = [14, 14]
EXPECTED_IMAGE_TOKENS = 196
EXPECTED_SCHEDULER_MODE = "overlapped_loop_body_throughput"
EXPECTED_VCK190_DEVICE = "xcvc1902-vsva2197"


def load_optional_json(path: Path) -> Any:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_percent(data: dict[str, Any], name: str) -> float | None:
    value = data.get(f"{name}_percent")
    if isinstance(value, (int, float)):
        return float(value)
    value = data.get(name)
    if isinstance(value, (int, float)):
        value = float(value)
        return value * 100.0 if value <= 1.0 else value
    return None


def report_eval(data: dict[str, Any]) -> dict[str, Any]:
    for key in ("eval", "best_eval"):
        value = data.get(key)
        if isinstance(value, dict):
            return value
    return data


def candidate_paths(raw_path: Any, anchor: Path) -> list[Path]:
    if not isinstance(raw_path, str) or not raw_path:
        return []
    path = Path(raw_path)
    paths = [path]
    if path.is_absolute():
        paths.append(anchor.parent / path.name)
    else:
        paths.append(anchor.parent / path)
        paths.append(Path.cwd() / path)
    existing: list[Path] = []
    seen: set[str] = set()
    for item in paths:
        key = str(item)
        if key not in seen and item.exists():
            existing.append(item)
            seen.add(key)
    return existing


def first_nonempty_candidate(raw_path: Any, anchor: Path) -> str | None:
    for path in candidate_paths(raw_path, anchor):
        if path.is_file() and path.stat().st_size > 0:
            return str(path)
    return None


def requirement(name: str, ok: bool, evidence: str) -> dict[str, Any]:
    return {
        "requirement": name,
        "status": "pass" if ok else "fail",
        "blocking": not ok,
        "evidence": evidence,
    }


def inspect_qat(report_path: Path, min_top1: float) -> dict[str, Any]:
    data = load_optional_json(report_path)
    if not isinstance(data, dict):
        return {
            "path": str(report_path),
            "exists": report_path.exists(),
            "ok": False,
            "checks": {"report_json_object": False},
        }
    eval_data = report_eval(data)
    top1 = metric_percent(eval_data, "top1")
    top5 = metric_percent(eval_data, "top5")
    qonnx_path = first_nonempty_candidate(data.get("qonnx_path"), report_path)
    checks = {
        "report_exists": report_path.is_file(),
        "model_id": data.get("model_id") == EXPECTED_MODEL_ID,
        "weight_bits_6": data.get("weight_bits") == EXPECTED_WEIGHT_BITS,
        "act_bits_7": data.get("act_bits") == EXPECTED_ACT_BITS,
        "edge_bits_8": data.get("edge_bits") == EXPECTED_EDGE_BITS,
        "eval_50k": data.get("eval_images") == EXPECTED_EVAL_IMAGES
        and eval_data.get("num_images") == EXPECTED_EVAL_IMAGES,
        "top1_gt_target": isinstance(top1, float) and top1 > min_top1,
        "qonnx_nonempty": qonnx_path is not None,
    }
    return {
        "path": str(report_path),
        "model_id": data.get("model_id"),
        "weight_bits": data.get("weight_bits"),
        "act_bits": data.get("act_bits"),
        "edge_bits": data.get("edge_bits"),
        "num_images": eval_data.get("num_images"),
        "top1_percent": top1,
        "top5_percent": top5,
        "qonnx_path": qonnx_path,
        "checks": checks,
        "ok": all(checks.values()),
    }


def scheduler_scope_ok(scope: Any) -> bool:
    return (
        isinstance(scope, dict)
        and scope.get("model") == EXPECTED_MODEL_ID
        and scope.get("weight_bits") == EXPECTED_WEIGHT_BITS
        and scope.get("activation_bits") == EXPECTED_ACT_BITS
        and scope.get("vision_depth") == EXPECTED_VISION_DEPTH
        and scope.get("image_size") == EXPECTED_IMAGE_SIZE
        and scope.get("patch_grid") == EXPECTED_PATCH_GRID
        and scope.get("image_tokens") == EXPECTED_IMAGE_TOKENS
    )


def inspect_estimate(estimate_dir: Path, clock_ns: float, target_ms: float) -> dict[str, Any]:
    cycles_path = estimate_dir / "report" / "estimate_layer_cycles.json"
    cycles = load_optional_json(cycles_path)
    spec_path = estimate_dir / "siglip2_86m_overlapped_scheduler_spec.json"
    spec = load_optional_json(spec_path)
    cycle_values = [int(value) for value in cycles.values()] if isinstance(cycles, dict) else []
    total_cycles = sum(cycle_values) if cycle_values else None
    latency_ms = total_cycles * clock_ns / 1_000_000.0 if isinstance(total_cycles, int) else None
    spec_obj = spec if isinstance(spec, dict) else {}
    schedule = spec_obj.get("schedule_model")
    schedule = schedule if isinstance(schedule, dict) else {}
    checks = {
        "estimate_dir_exists": estimate_dir.is_dir(),
        "cycles_json_object": isinstance(cycles, dict),
        "total_cycles_positive": isinstance(total_cycles, int) and total_cycles > 0,
        "latency_under_target": isinstance(latency_ms, float) and latency_ms < target_ms,
        "scheduler_spec_exists": spec_path.is_file(),
        "scheduler_scope_w6a7": scheduler_scope_ok(spec_obj.get("exact_scope")),
        "scheduler_mode": schedule.get("mode") == EXPECTED_SCHEDULER_MODE,
        "scheduler_total_cycles_match": schedule.get("total_cycles_with_non_loop")
        == total_cycles,
        "scheduler_latency_match": isinstance(schedule.get("latency_ms"), (int, float))
        and isinstance(latency_ms, float)
        and abs(float(schedule["latency_ms"]) - latency_ms) < 1e-9,
        "scheduler_latency_under_target": isinstance(schedule.get("latency_ms"), (int, float))
        and float(schedule["latency_ms"]) < target_ms,
    }
    return {
        "estimate_dir": str(estimate_dir),
        "cycles_path": str(cycles_path),
        "scheduler_spec_path": str(spec_path),
        "total_cycles": total_cycles,
        "clock_ns": clock_ns,
        "latency_ms": latency_ms,
        "scheduler_latency_ms": schedule.get("latency_ms"),
        "schedule_model": schedule,
        "checks": checks,
        "ok": all(checks.values()),
    }


def route_clean(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(errors="ignore")
    match = re.search(r"# of nets with routing errors\.+\s*:\s*(\d+)\s*:", text)
    return bool(match and int(match.group(1)) == 0)


def timing_met(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(errors="ignore")
    return (
        "All user specified timing constraints are met" in text
        and "Timing constraints are not met" not in text
    )


def timing_summary(path: Path, target_clock_ns: float) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "exists": False,
            "primary_clock": None,
            "target_clock_ns": target_clock_ns,
        }
    text = path.read_text(errors="ignore")
    primary = None
    device_match = re.search(r"\|\s*Device\s*:\s*([^\s|]+)", text)
    device = device_match.group(1) if device_match else None
    match = re.search(
        r"^\s*ap_clk\s+\{[^}]+\}\s+([0-9.]+)\s+([0-9.]+)\s*$",
        text,
        re.MULTILINE,
    )
    if match:
        primary = {
            "clock": "ap_clk",
            "period_ns": float(match.group(1)),
            "frequency_mhz": float(match.group(2)),
        }
    else:
        req = re.search(r"Requirement:\s+([0-9.]+)ns\s+\(ap_clk rise@", text)
        if req:
            period = float(req.group(1))
            primary = {
                "clock": "ap_clk",
                "period_ns": period,
                "frequency_mhz": 1000.0 / period if period else None,
            }
    return {
        "path": str(path),
        "exists": True,
        "device": device,
        "device_is_vck190": isinstance(device, str)
        and device.startswith(EXPECTED_VCK190_DEVICE),
        "primary_clock": primary,
        "target_clock_ns": target_clock_ns,
        "primary_clock_at_or_below_target": isinstance(primary, dict)
        and primary.get("period_ns") <= target_clock_ns + 1e-12,
        "timing_met": timing_met(path),
    }


def inspect_dcp(dcp_dir: Path, target_clock_ns: float) -> dict[str, Any]:
    root = dcp_dir / "stitched_ip"
    routed_dcp = root / "finn_design_routed.dcp"
    timing = root / "ooc_timing.rpt"
    route = root / "ooc_route_status.rpt"
    utilization = root / "ooc_utilization.rpt"
    clock = timing_summary(timing, target_clock_ns)
    checks = {
        "stitched_ip_dir_exists": root.is_dir(),
        "routed_dcp_nonempty": routed_dcp.is_file() and routed_dcp.stat().st_size > 0,
        "route_status_nonempty": route.is_file() and route.stat().st_size > 0,
        "route_clean": route_clean(route),
        "timing_report_nonempty": timing.is_file() and timing.stat().st_size > 0,
        "timing_device_vck190": clock.get("device_is_vck190") is True,
        "timing_met": timing_met(timing),
        "timing_primary_clock_at_target": clock.get("primary_clock_at_or_below_target") is True,
    }
    return {
        "dcp_dir": str(dcp_dir),
        "stitched_ip": str(root),
        "routed_dcp": str(routed_dcp),
        "routed_dcp_size_bytes": routed_dcp.stat().st_size if routed_dcp.exists() else None,
        "route_status_report": str(route),
        "timing_report": str(timing),
        "utilization_report": str(utilization),
        "utilization_report_size_bytes": utilization.stat().st_size
        if utilization.exists()
        else None,
        "clock_summary": clock,
        "checks": checks,
        "ok": all(checks.values()),
    }


def summarize(
    qat_report: Path,
    estimate_dir: Path,
    dcp_dir: Path,
    clock_ns: float,
    target_ms: float,
    min_top1: float,
) -> dict[str, Any]:
    qat = inspect_qat(qat_report, min_top1)
    estimate = inspect_estimate(estimate_dir, clock_ns, target_ms)
    dcp = inspect_dcp(dcp_dir, clock_ns)
    requirements = [
        requirement(
            "imagenet_top1_gt_72_percent",
            qat.get("ok") is True,
            (
                f"path={qat.get('path')} top1={qat.get('top1_percent')} "
                f"images={qat.get('num_images')} w={qat.get('weight_bits')} "
                f"a={qat.get('act_bits')}"
            ),
        ),
        requirement(
            "latency_estimate_lt_30ms",
            estimate.get("ok") is True,
            (
                f"estimate_dir={estimate.get('estimate_dir')} "
                f"cycles={estimate.get('total_cycles')} "
                f"clock_ns={estimate.get('clock_ns')} "
                f"latency_ms={estimate.get('latency_ms')}"
            ),
        ),
        requirement(
            "routed_vck190_dcp_exists",
            dcp.get("checks", {}).get("routed_dcp_nonempty") is True,
            f"path={dcp.get('routed_dcp')} size={dcp.get('routed_dcp_size_bytes')}",
        ),
        requirement(
            "route_clean",
            dcp.get("checks", {}).get("route_clean") is True,
            f"path={dcp.get('route_status_report')}",
        ),
        requirement(
            "timing_clean_at_target_clock",
            dcp.get("checks", {}).get("timing_met") is True
            and dcp.get("checks", {}).get("timing_device_vck190") is True
            and dcp.get("checks", {}).get("timing_primary_clock_at_target") is True,
            (
                f"path={dcp.get('timing_report')} "
                f"device={dcp.get('clock_summary', {}).get('device')} "
                f"clock={dcp.get('clock_summary', {}).get('primary_clock')}"
            ),
        ),
    ]
    complete = all(item["status"] == "pass" for item in requirements)
    return {
        "artifact_type": "siglip2_86m_sub30_goal_status",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "get <30ms latency with >72% accuracy",
        "objective_complete": complete,
        "target": {
            "model_id": EXPECTED_MODEL_ID,
            "weight_bits": EXPECTED_WEIGHT_BITS,
            "act_bits": EXPECTED_ACT_BITS,
            "edge_bits": EXPECTED_EDGE_BITS,
            "eval_images": EXPECTED_EVAL_IMAGES,
            "min_top1_percent_exclusive": min_top1,
            "target_latency_ms_exclusive": target_ms,
            "clock_ns": clock_ns,
            "vck190_device_prefix": EXPECTED_VCK190_DEVICE,
        },
        "qat": qat,
        "estimate": estimate,
        "dcp": dcp,
        "requirements": requirements,
        "blockers": [
            f"{item['requirement']}: {item['evidence']}"
            for item in requirements
            if item["blocking"]
        ],
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# SigLIP2 86M Sub-30 Goal Status",
        "",
        f"- generated UTC: `{report['generated_at_utc']}`",
        f"- objective complete: `{str(report['objective_complete']).lower()}`",
        f"- objective: `{report['objective']}`",
        "",
        "| Requirement | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for item in report["requirements"]:
        evidence = str(item["evidence"]).replace("|", "\\|")
        lines.append(f"| `{item['requirement']}` | `{item['status']}` | {evidence} |")
    lines.extend(["", "## Blockers", ""])
    if report["blockers"]:
        lines.extend(f"- {item}" for item in report["blockers"])
    else:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qat-report", type=Path, default=DEFAULT_QAT_REPORT)
    parser.add_argument("--estimate-dir", type=Path, default=DEFAULT_ESTIMATE_DIR)
    parser.add_argument("--dcp-dir", type=Path, default=DEFAULT_DCP_DIR)
    parser.add_argument("--clock-ns", type=float, default=3.999)
    parser.add_argument("--target-ms", type=float, default=30.0)
    parser.add_argument("--min-top1", type=float, default=72.0)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args()

    report = summarize(
        qat_report=args.qat_report,
        estimate_dir=args.estimate_dir,
        dcp_dir=args.dcp_dir,
        clock_ns=args.clock_ns,
        target_ms=args.target_ms,
        min_top1=args.min_top1,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.md_out, report)
    print(json.dumps(report, indent=2))
    return 0 if report["objective_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
