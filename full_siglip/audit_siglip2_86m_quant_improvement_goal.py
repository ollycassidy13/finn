#!/usr/bin/env python3
"""Audit the SigLIP2 86M lower-precision improvement objective."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any


EXPECTED_MODEL_ID = "google/siglip2-base-patch16-224"
EXPECTED_IMAGE_SIZE = 224
EXPECTED_PATCH_SIZE = 16
EXPECTED_VISION_DEPTH = 12
EXPECTED_QUANTIZER = "qv_lsq"
EXPECTED_EVAL_IMAGES = 50_000
EXPECTED_PATCH_GRID = [14, 14]
EXPECTED_IMAGE_TOKENS = 196
BASELINE_WEIGHT_BITS = 6
BASELINE_ACT_BITS = 8
DEFAULT_CLOCK_NS = 3.9
MIN_DCP_CLOCK_MHZ = 250.0
PRIMARY_DCP_CLOCK = "ap_clk"
EXPECTED_VCK190_DEVICE = "xcvc1902-vsva2197"
EXPECTED_SCHEDULER_SPEC = "siglip2_86m_overlapped_scheduler_spec.json"
EXPECTED_SCHEDULER_MATERIALIZATION = "overlapped_scheduler_materialization.json"
EXPECTED_SCHEDULER_KIND = "builtin_stream_feedback"
EXPECTED_SCHEDULER_MODE = "overlapped_loop_body_throughput"
EXPECTED_LOOP_NAME = "FINNLoop_0"

DEFAULT_BUILD_ROOT = Path("full_siglip/build")
DEFAULT_FP32_REPORT = (
    DEFAULT_BUILD_ROOT
    / "static_siglip2_86m_patch16_224_fp32_baseline"
    / "fp32_static_imagenet_report.json"
)
DEFAULT_BASELINE_ESTIMATE = (
    DEFAULT_BUILD_ROOT
    / "static_qat_siglip2_86m_patch16_224_w6a8_qvlsq_featurehead_50k_50k_export"
    "_mlo_fps1200_loopfps120_w72_hlstop_stream_feedback_est"
    / "report"
    / "estimate_layer_cycles.json"
)
DEFAULT_BASELINE_DCP_TIMING = (
    DEFAULT_BUILD_ROOT
    / "static_qat_siglip2_86m_patch16_224_w6a8_qvlsq_featurehead_50k_50k_export"
    "_mlo_fps1200_loopfps120_w72_hlstop_stream_feedback_dcp_250p06mhz_postroute3999_agghold"
    / "stitched_ip"
    / "ooc_timing.rpt"
)
DEFAULT_JSON_OUT = DEFAULT_BUILD_ROOT / "siglip2_86m_quant_improvement_status.json"
DEFAULT_MD_OUT = DEFAULT_BUILD_ROOT / "siglip2_86m_quant_improvement_status.md"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_optional_json(path: Path) -> Any:
    if not path.exists():
        return None
    return load_json(path)


def metric_percent(data: dict[str, Any], name: str) -> float | None:
    value = data.get(f"{name}_percent")
    if isinstance(value, (int, float)):
        return float(value)
    value = data.get(name)
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value * 100.0 if value <= 1.0 else value


def report_eval(data: dict[str, Any]) -> dict[str, Any]:
    for key in ("eval", "best_eval"):
        value = data.get(key)
        if isinstance(value, dict):
            return value
    return data


def fmt_percent(value: Any) -> str:
    return "missing" if not isinstance(value, (int, float)) else f"{float(value):.3f}%"


def existing_candidate_paths(raw_path: Any, anchor_path: Path) -> list[Path]:
    if not isinstance(raw_path, str) or not raw_path:
        return []
    path = Path(raw_path)
    candidates = [path]
    if path.is_absolute():
        candidates.append(anchor_path.parent / path.name)
    else:
        candidates.append(anchor_path.parent / path)
        candidates.append(Path.cwd() / path)
    existing = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen and candidate.exists():
            existing.append(candidate)
            seen.add(key)
    return existing


def candidate_file_size(raw_path: Any, anchor_path: Path) -> int | None:
    sizes = [path.stat().st_size for path in existing_candidate_paths(raw_path, anchor_path)]
    return max(sizes) if sizes else None


def qonnx_available(report_path: Path, data: dict[str, Any]) -> tuple[bool, list[str]]:
    candidates: list[Any] = [data.get("qonnx_path"), data.get("qonnx")]
    cleanup = data.get("qonnx_cleanup")
    if isinstance(cleanup, dict):
        candidates.extend(cleanup.get(key) for key in ("final_path", "inferred", "canonicalized"))
    existing = []
    seen = set()
    for value in candidates:
        for path in existing_candidate_paths(value, report_path):
            if path.stat().st_size <= 0:
                continue
            raw = str(path)
            if raw not in seen:
                existing.append(raw)
                seen.add(raw)
    return bool(existing), existing


def inspect_fp32(path: Path) -> dict[str, Any]:
    data = load_optional_json(path)
    if not isinstance(data, dict):
        return {"path": str(path), "exists": path.exists(), "ok": False}
    top1 = metric_percent(data, "top1")
    top5 = metric_percent(data, "top5")
    prompt_templates = data.get("prompt_templates")
    checks = {
        "model_id": data.get("model_id") == EXPECTED_MODEL_ID,
        "quantized_false": data.get("quantized") is False,
        "num_images_50k": data.get("num_images") == EXPECTED_EVAL_IMAGES,
        "image_size_224": data.get("exported_image_size", data.get("image_size"))
        == EXPECTED_IMAGE_SIZE,
        "patch_size_16": data.get("patch_size") == EXPECTED_PATCH_SIZE,
        "vision_depth_12": data.get("original_vision_depth") == EXPECTED_VISION_DEPTH
        and data.get("exported_vision_depth") == EXPECTED_VISION_DEPTH,
        "top1_available": isinstance(top1, float),
        "prompt_ensemble_recorded": isinstance(prompt_templates, list)
        and len(prompt_templates) >= 8,
    }
    return {
        "path": str(path),
        "exists": True,
        "model_id": data.get("model_id"),
        "num_images": data.get("num_images"),
        "top1_percent": top1,
        "top5_percent": top5,
        "checks": checks,
        "ok": all(checks.values()),
    }


def module_counts_match_bits(data: dict[str, Any], weight_bits: int, act_bits: int) -> bool:
    quantization = data.get("quantization")
    if not isinstance(quantization, dict):
        return False
    module_counts = quantization.get("module_counts")
    if not isinstance(module_counts, dict):
        return False
    quant_linear = module_counts.get(f"QuantLinear_W{weight_bits}A{act_bits}", 0)
    qv_linear = module_counts.get(f"QVActQuantLinear_W{weight_bits}A{act_bits}", 0)
    return isinstance(quant_linear, int) and quant_linear > 0 and isinstance(qv_linear, int) and qv_linear > 0


def qat_report_paths(build_root: Path) -> list[Path]:
    return sorted(
        path
        for path in build_root.glob("**/static_qat_siglip2_86m_patch16_224_*qvlsq*/qat_report.json")
        if path.is_file()
    )


def inspect_qat(path: Path, fp32_top1: float | None) -> dict[str, Any]:
    data = load_optional_json(path)
    if not isinstance(data, dict):
        return {"path": str(path), "exists": path.exists(), "ok": False}
    eval_data = report_eval(data)
    top1 = metric_percent(eval_data, "top1")
    top5 = metric_percent(eval_data, "top5")
    weight_bits = data.get("weight_bits")
    act_bits = data.get("act_bits")
    qonnx_ok, qonnx_files = qonnx_available(path, data)
    checkpoint = data.get("best_checkpoint") or data.get("checkpoint")
    checkpoint_size = candidate_file_size(checkpoint, path)
    prompt_templates = data.get("prompt_templates")
    lower_precision = (
        isinstance(weight_bits, int)
        and isinstance(act_bits, int)
        and (weight_bits < BASELINE_WEIGHT_BITS or act_bits < BASELINE_ACT_BITS)
    )
    accuracy_ge_fp32 = (
        isinstance(fp32_top1, float)
        and isinstance(top1, float)
        and top1 >= fp32_top1
    )
    checks = {
        "model_id": data.get("model_id") == EXPECTED_MODEL_ID,
        "qv_lsq": data.get("quantizer") == EXPECTED_QUANTIZER,
        "lower_precision_than_w6a8": lower_precision,
        "image_size_224": data.get("exported_image_size", data.get("image_size"))
        == EXPECTED_IMAGE_SIZE,
        "patch_size_16": data.get("patch_size") == EXPECTED_PATCH_SIZE,
        "vision_depth_12": data.get("original_vision_depth") == EXPECTED_VISION_DEPTH
        and data.get("exported_vision_depth") == EXPECTED_VISION_DEPTH,
        "eval_50k": data.get("eval_images") == EXPECTED_EVAL_IMAGES
        and eval_data.get("num_images") == EXPECTED_EVAL_IMAGES,
        "top1_ge_fp32": accuracy_ge_fp32,
        "prompt_ensemble_recorded": isinstance(prompt_templates, list)
        and len(prompt_templates) >= 8,
        "checkpoint_nonempty": isinstance(checkpoint_size, int) and checkpoint_size > 0,
        "qonnx_available": qonnx_ok,
        "quantization_metadata_bits": isinstance(weight_bits, int)
        and isinstance(act_bits, int)
        and module_counts_match_bits(data, weight_bits, act_bits),
    }
    return {
        "path": str(path),
        "model_id": data.get("model_id"),
        "quantizer": data.get("quantizer"),
        "weight_bits": weight_bits,
        "act_bits": act_bits,
        "edge_bits": data.get("edge_bits"),
        "image_size": data.get("exported_image_size", data.get("image_size")),
        "patch_size": data.get("patch_size"),
        "original_vision_depth": data.get("original_vision_depth"),
        "exported_vision_depth": data.get("exported_vision_depth"),
        "top_level_eval_images": data.get("eval_images"),
        "num_images": eval_data.get("num_images"),
        "top1_percent": top1,
        "top5_percent": top5,
        "checkpoint": checkpoint,
        "checkpoint_size_bytes": checkpoint_size,
        "qonnx_files": qonnx_files,
        "prompt_template_count": len(prompt_templates) if isinstance(prompt_templates, list) else None,
        "checks": checks,
        "ok": all(checks.values()),
    }


def score_qat(row: dict[str, Any]) -> tuple[int, int, float]:
    return (
        int(row.get("ok") is True),
        int(row.get("checks", {}).get("qonnx_available") is True),
        float(row.get("top1_percent") or -1.0),
    )


def is_dcp_output_dir(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith("_dcp") or "_dcp_" in name or (path / "stitched_ip").is_dir()


def derived_from_qat_dir(path: Path, qat_dir: Path | None) -> bool:
    if qat_dir is None:
        return False
    try:
        same_parent = path.parent.resolve() == qat_dir.parent.resolve()
    except FileNotFoundError:
        same_parent = path.parent == qat_dir.parent
    return same_parent and path.name.startswith(f"{qat_dir.name}_mlo_")


def derived_from_estimate_dir(path: Path, estimate_dir: Path | None) -> bool:
    if estimate_dir is None:
        return False
    try:
        same_parent = path.parent.resolve() == estimate_dir.parent.resolve()
    except FileNotFoundError:
        same_parent = path.parent == estimate_dir.parent
    name = estimate_dir.name
    prefix = f"{name[:-4]}_" if name.endswith("_est") else f"{name}_"
    return same_parent and path.name.startswith(prefix)


def matching_estimate_for_dcp(
    dcp_dir: Path, estimate_candidates: list[dict[str, Any]]
) -> dict[str, Any] | None:
    matches = []
    for estimate in estimate_candidates:
        build_dir = estimate.get("build_dir")
        if not isinstance(build_dir, str) or not build_dir:
            continue
        estimate_dir = Path(build_dir)
        if derived_from_estimate_dir(dcp_dir, estimate_dir):
            matches.append((len(estimate_dir.name), estimate))
    if not matches:
        return None
    return max(matches, key=lambda item: item[0])[1]


def cycles_latency(path: Path, clock_ns: float) -> dict[str, Any]:
    data = load_optional_json(path)
    if not isinstance(data, dict):
        return {"path": str(path), "available": False}
    cycles = [value for value in data.values() if isinstance(value, int)]
    layer_total = sum(cycles) if cycles else None
    network_path = path.with_name("estimate_network_performance.json")
    network = load_optional_json(network_path)
    critical_cycles = None
    estimated_latency_ns = None
    if isinstance(network, dict):
        for key in ("critical_path_cycles", "max_cycles"):
            value = network.get(key)
            if isinstance(value, int):
                critical_cycles = value
                break
        value = network.get("estimated_latency_ns")
        if isinstance(value, (int, float)):
            estimated_latency_ns = float(value)
    if isinstance(critical_cycles, int):
        total_cycles = critical_cycles
        latency_ms = total_cycles * clock_ns / 1_000_000.0
        source = "estimate_network_performance_cycles"
    elif isinstance(estimated_latency_ns, float):
        total_cycles = None
        latency_ms = estimated_latency_ns / 1_000_000.0
        source = "estimate_network_performance_latency_ns"
    else:
        total_cycles = layer_total
        latency_ms = total_cycles * clock_ns / 1_000_000.0 if isinstance(total_cycles, int) else None
        source = "sum_layer_cycles"
    return {
        "path": str(path),
        "available": True,
        "total_cycles": total_cycles,
        "layer_total_cycles": layer_total,
        "critical_path_cycles": critical_cycles,
        "estimated_latency_ns": estimated_latency_ns,
        "latency_ms": latency_ms,
        "latency_source": source,
        "max_layer_cycles": max(cycles) if cycles else None,
    }


def scope_matches_candidate(scope: Any, candidate: dict[str, Any]) -> bool:
    return (
        isinstance(scope, dict)
        and scope.get("model") == EXPECTED_MODEL_ID
        and scope.get("weight_bits") == candidate.get("weight_bits")
        and scope.get("activation_bits") == candidate.get("act_bits")
        and scope.get("vision_depth") == EXPECTED_VISION_DEPTH
        and scope.get("image_size") == EXPECTED_IMAGE_SIZE
        and scope.get("patch_grid") == EXPECTED_PATCH_GRID
        and scope.get("image_tokens") == EXPECTED_IMAGE_TOKENS
    )


def nonempty_candidate_path(raw_path: Any, anchor_path: Path) -> str | None:
    for candidate in existing_candidate_paths(raw_path, anchor_path):
        if candidate.is_file() and candidate.stat().st_size > 0:
            return str(candidate)
    return None


def same_existing_path(left: str | None, right: Path) -> bool:
    if left is None:
        return False
    try:
        return Path(left).resolve() == right.resolve()
    except OSError:
        return False


def scheduler_evidence(build_dir: Path, candidate: dict[str, Any], baseline_latency_ms: float | None) -> dict[str, Any]:
    spec_path = build_dir / EXPECTED_SCHEDULER_SPEC
    materialization_path = build_dir / EXPECTED_SCHEDULER_MATERIALIZATION
    spec = load_optional_json(spec_path)
    materialization = load_optional_json(materialization_path)
    spec_obj = spec if isinstance(spec, dict) else {}
    materialization_obj = materialization if isinstance(materialization, dict) else {}
    spec_schedule = spec_obj.get("schedule_model")
    spec_schedule = spec_schedule if isinstance(spec_schedule, dict) else {}
    materialization_schedule = materialization_obj.get("schedule_model")
    materialization_schedule = materialization_schedule if isinstance(materialization_schedule, dict) else {}
    materialization_checks = materialization_obj.get("checks")
    materialization_checks = materialization_checks if isinstance(materialization_checks, dict) else {}
    graph_annotation = materialization_obj.get("graph_annotation")
    graph_annotation = graph_annotation if isinstance(graph_annotation, dict) else {}
    artifact = materialization_obj.get("artifact")
    artifact = artifact if isinstance(artifact, dict) else {}
    source = materialization_obj.get("source")
    source = source if isinstance(source, dict) else {}
    spec_latency = spec_schedule.get("latency_ms")
    materialization_latency = materialization_schedule.get("latency_ms")
    implementation_artifact_path = nonempty_candidate_path(
        materialization_obj.get("implementation_artifact_abs") or artifact.get("abs_path") or artifact.get("path"),
        materialization_path,
    )
    annotated_graph_path = nonempty_candidate_path(
        materialization_obj.get("annotated_graph_abs") or materialization_obj.get("annotated_graph"),
        materialization_path,
    )
    source_spec_path = nonempty_candidate_path(source.get("scheduler_spec"), materialization_path)
    latency_lower = (
        isinstance(baseline_latency_ms, float)
        and isinstance(spec_latency, (int, float))
        and isinstance(materialization_latency, (int, float))
        and float(spec_latency) < baseline_latency_ms
        and float(materialization_latency) < baseline_latency_ms
    )
    checks = {
        "spec_exists": spec_path.is_file() and spec_path.stat().st_size > 0,
        "spec_json_object": isinstance(spec, dict),
        "spec_exact_scope": scope_matches_candidate(spec_obj.get("exact_scope"), candidate),
        "spec_schedule_mode": spec_schedule.get("mode") == EXPECTED_SCHEDULER_MODE,
        "spec_loop_iterations": spec_schedule.get("loop_iterations") == EXPECTED_VISION_DEPTH,
        "spec_latency_lower_than_baseline": latency_lower,
        "materialization_exists": materialization_path.is_file() and materialization_path.stat().st_size > 0,
        "materialization_json_object": isinstance(materialization, dict),
        "materialization_references_spec": same_existing_path(source_spec_path, spec_path),
        "materialization_exact_scope": scope_matches_candidate(materialization_obj.get("exact_scope"), candidate),
        "materialization_kind": materialization_obj.get("implementation_kind") == EXPECTED_SCHEDULER_KIND,
        "materialization_ready_for_dcp_preflight": materialization_checks.get("ready_for_dcp_preflight") is True,
        "graph_annotation_overlapped_finnloop": graph_annotation.get("attempted") is True
        and graph_annotation.get("attrs_set") is True
        and graph_annotation.get("loop_name") == EXPECTED_LOOP_NAME
        and graph_annotation.get("loop_scheduler_mode") == "overlapped",
        "implementation_artifact_nonempty": implementation_artifact_path is not None,
        "annotated_graph_nonempty": annotated_graph_path is not None,
    }
    return {
        "spec_path": str(spec_path),
        "materialization_path": str(materialization_path),
        "implementation_kind": materialization_obj.get("implementation_kind"),
        "implementation_artifact_path": implementation_artifact_path,
        "annotated_graph_path": annotated_graph_path,
        "source_spec_path": source_spec_path,
        "spec_latency_ms": spec_latency,
        "materialization_latency_ms": materialization_latency,
        "checks": checks,
        "ok": all(checks.values()),
    }


def inspect_estimates(
    build_root: Path,
    clock_ns: float,
    baseline_latency_ms: float | None,
    baseline_cycles: int | None,
    accepted_qat_dir: Path | None,
    accepted_qat: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(build_root.glob("**/*siglip2_86m*/report/estimate_layer_cycles.json")):
        build_dir = path.parents[1]
        if is_dcp_output_dir(build_dir):
            continue
        estimate = cycles_latency(path, clock_ns)
        latency = estimate.get("latency_ms")
        total_cycles = estimate.get("total_cycles")
        derived = derived_from_qat_dir(build_dir, accepted_qat_dir)
        scheduler = scheduler_evidence(build_dir, accepted_qat or {}, baseline_latency_ms) if derived else None
        rows.append(
            {
                "build_dir": str(build_dir),
                **estimate,
                "folding_config": str(build_dir / "auto_folding_config.json"),
                "folding_config_exists": (build_dir / "auto_folding_config.json").is_file(),
                "derived_from_accepted_qat": derived,
                "latency_lower_than_baseline": isinstance(latency, (int, float))
                and isinstance(baseline_latency_ms, float)
                and float(latency) < baseline_latency_ms,
                "cycles_lower_than_baseline": isinstance(total_cycles, int)
                and isinstance(baseline_cycles, int)
                and total_cycles < baseline_cycles,
                "scheduler_evidence": scheduler,
                "scheduler_evidence_ok": isinstance(scheduler, dict) and scheduler.get("ok") is True,
            }
        )
    return rows


def timing_met(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(errors="ignore")
    return "All user specified timing constraints are met" in text and "Timing constraints are not met" not in text


def route_clean(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(errors="ignore")
    match = re.search(r"# of nets with routing errors\.+\s*:\s*(\d+)", text)
    return bool(match and int(match.group(1)) == 0)


def timing_clock_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "clocks": []}
    clocks = []
    device = None
    for line in path.read_text(errors="ignore").splitlines():
        device_match = re.match(r"^\|\s*Device\s*:\s*(\S+)", line)
        if device_match:
            device = device_match.group(1)
        match = re.match(
            r"^\s*(?P<clock>\S+)\s+\{[^}]+\}\s+"
            r"(?P<period>[0-9]+(?:\.[0-9]+)?)\s+"
            r"(?P<frequency>[0-9]+(?:\.[0-9]+)?)\s*$",
            line,
        )
        if match:
            clocks.append(
                {
                    "clock": match.group("clock"),
                    "period_ns": float(match.group("period")),
                    "frequency_mhz": float(match.group("frequency")),
                }
            )
    primary = next((clock for clock in clocks if clock["clock"] == PRIMARY_DCP_CLOCK), None)
    return {"available": bool(clocks), "device": device, "primary_clock": primary, "clocks": clocks}


def inspect_dcps(
    build_root: Path,
    estimate_candidates: list[dict[str, Any]],
    accepted_qat: dict[str, Any] | None,
    baseline_routed_latency_ms: float | None,
) -> list[dict[str, Any]]:
    rows = []
    for root in sorted(build_root.glob("**/*siglip2_86m*/stitched_ip")):
        if not root.is_dir():
            continue
        dcp_dir = root.parent
        routed_dcp = root / "finn_design_routed.dcp"
        timing = root / "ooc_timing.rpt"
        route = root / "ooc_route_status.rpt"
        clock_summary = timing_clock_summary(timing)
        primary = clock_summary.get("primary_clock")
        primary_frequency = primary.get("frequency_mhz") if isinstance(primary, dict) else None
        primary_period = primary.get("period_ns") if isinstance(primary, dict) else None
        matched_estimate = matching_estimate_for_dcp(dcp_dir, estimate_candidates)
        matched_estimate_dir = (
            Path(str(matched_estimate["build_dir"])) if isinstance(matched_estimate, dict) else None
        )
        matched_cycles = (
            matched_estimate.get("total_cycles") if isinstance(matched_estimate, dict) else None
        )
        routed_latency_ms = (
            matched_cycles * primary_period / 1_000_000.0
            if isinstance(matched_cycles, int) and isinstance(primary_period, float)
            else None
        )
        derived = matched_estimate is not None
        scheduler = scheduler_evidence(dcp_dir, accepted_qat or {}, baseline_routed_latency_ms) if derived else None
        checks = {
            "derived_from_accepted_estimate": derived,
            "derived_from_lower_latency_estimate": derived,
            "routed_dcp_nonempty": routed_dcp.is_file() and routed_dcp.stat().st_size > 0,
            "timing_report_nonempty": timing.is_file() and timing.stat().st_size > 0,
            "route_status_nonempty": route.is_file() and route.stat().st_size > 0,
            "timing_met": timing_met(timing),
            "route_clean": route_clean(route),
            "device_vck190": clock_summary.get("device") == EXPECTED_VCK190_DEVICE,
            "primary_clock_gt_250mhz": isinstance(primary_frequency, float) and primary_frequency > MIN_DCP_CLOCK_MHZ,
            "routed_latency_lower_than_baseline": isinstance(routed_latency_ms, float)
            and isinstance(baseline_routed_latency_ms, float)
            and routed_latency_ms < baseline_routed_latency_ms,
            "scheduler_evidence_ok": isinstance(scheduler, dict) and scheduler.get("ok") is True,
        }
        rows.append(
            {
                "root": str(root),
                "routed_dcp": str(routed_dcp),
                "routed_dcp_size_bytes": routed_dcp.stat().st_size if routed_dcp.exists() else None,
                "timing_report": str(timing),
                "route_status_report": str(route),
                "clock_summary": clock_summary,
                "routed_latency_ms": routed_latency_ms,
                "accepted_estimate_dir": str(matched_estimate_dir) if matched_estimate_dir is not None else None,
                "matched_estimate_dir": str(matched_estimate_dir) if matched_estimate_dir is not None else None,
                "matched_estimate_cycles": matched_cycles,
                "matched_estimate_latency_ms": (
                    matched_estimate.get("latency_ms") if isinstance(matched_estimate, dict) else None
                ),
                "derived_from_accepted_estimate": derived,
                "derived_from_lower_latency_estimate": derived,
                "scheduler_evidence": scheduler,
                "scheduler_evidence_ok": isinstance(scheduler, dict) and scheduler.get("ok") is True,
                "checks": checks,
                "ok": all(checks.values()),
            }
        )
    return rows


def requirement(name: str, ok: bool, evidence: str) -> dict[str, Any]:
    return {"requirement": name, "status": "pass" if ok else "fail", "blocking": not ok, "evidence": evidence}


def summarize(
    build_root: Path,
    fp32_report: Path,
    baseline_estimate: Path,
    baseline_dcp_timing: Path,
    clock_ns: float,
) -> dict[str, Any]:
    fp32 = inspect_fp32(fp32_report)
    fp32_top1 = fp32.get("top1_percent") if fp32.get("ok") else None
    fp32_top1 = fp32_top1 if isinstance(fp32_top1, float) else None
    baseline = cycles_latency(baseline_estimate, clock_ns)
    baseline_latency = baseline.get("latency_ms")
    baseline_latency = float(baseline_latency) if isinstance(baseline_latency, (int, float)) else None
    baseline_cycles = baseline.get("total_cycles")
    baseline_cycles = baseline_cycles if isinstance(baseline_cycles, int) else None
    baseline_clock = timing_clock_summary(baseline_dcp_timing)
    baseline_primary = baseline_clock.get("primary_clock")
    baseline_period = baseline_primary.get("period_ns") if isinstance(baseline_primary, dict) else None
    baseline_routed_latency = (
        baseline_cycles * baseline_period / 1_000_000.0
        if isinstance(baseline_cycles, int) and isinstance(baseline_period, float)
        else None
    )
    baseline_ok = (
        isinstance(baseline_cycles, int)
        and isinstance(baseline_latency, float)
        and isinstance(baseline_routed_latency, float)
    )

    qat_reports = [inspect_qat(path, fp32_top1) for path in qat_report_paths(build_root)]
    acceptable_qats = [row for row in qat_reports if row.get("ok") is True]
    best_qat = max(acceptable_qats, key=score_qat, default=None)
    accepted_qat_dir = Path(str(best_qat["path"])).parent if isinstance(best_qat, dict) else None
    estimates = inspect_estimates(
        build_root,
        clock_ns,
        baseline_latency,
        baseline_cycles,
        accepted_qat_dir,
        best_qat,
    )
    estimate_candidates = [
        row
        for row in estimates
        if row.get("derived_from_accepted_qat") is True
        and row.get("folding_config_exists") is True
        and row.get("latency_lower_than_baseline") is True
        and row.get("cycles_lower_than_baseline") is True
        and row.get("scheduler_evidence_ok") is True
    ]
    best_estimate = min(estimate_candidates, key=lambda row: float(row["latency_ms"]), default=None)
    if isinstance(best_qat, dict) and isinstance(best_estimate, dict):
        best_qat = {**best_qat, "accepted_total_cycles": best_estimate.get("total_cycles")}
    dcps = inspect_dcps(build_root, estimate_candidates, best_qat, baseline_routed_latency)
    timing_clean_dcp = next((row for row in dcps if row.get("ok") is True), None)

    requirements = [
        requirement(
            "fp32_50k_baseline",
            fp32.get("ok") is True,
            f"path={fp32.get('path')} top1={fmt_percent(fp32.get('top1_percent'))}",
        ),
        requirement(
            "baseline_w6a8_latency_available",
            baseline_ok,
            (
                f"estimate={baseline_estimate} cycles={baseline_cycles} "
                f"estimate_latency_ms={baseline_latency} routed_latency_ms={baseline_routed_latency}"
            ),
        ),
        requirement(
            "lower_precision_qat_50k_top1_ge_fp32",
            best_qat is not None,
            (
                "missing"
                if best_qat is None
                else (
                    f"path={best_qat.get('path')} w={best_qat.get('weight_bits')} "
                    f"a={best_qat.get('act_bits')} top1={fmt_percent(best_qat.get('top1_percent'))} "
                    f"fp32_top1={fmt_percent(fp32_top1)}"
                )
            ),
        ),
        requirement(
            "qonnx_export_from_lower_precision_qat",
            best_qat is not None and best_qat.get("checks", {}).get("qonnx_available") is True,
            "missing" if best_qat is None else f"qonnx_files={best_qat.get('qonnx_files')}",
        ),
        requirement(
            "finn_estimate_lower_latency_than_w6a8",
            best_estimate is not None,
            (
                "missing"
                if best_estimate is None
                else (
                    f"build={best_estimate.get('build_dir')} cycles={best_estimate.get('total_cycles')} "
                    f"latency_ms={best_estimate.get('latency_ms')} "
                    f"baseline_cycles={baseline_cycles} baseline_latency_ms={baseline_latency}"
                )
            ),
        ),
        requirement(
            "timing_clean_vck190_dcp_gt250mhz_lower_latency",
            timing_clean_dcp is not None,
            (
                "missing"
                if timing_clean_dcp is None
                else (
                    f"root={timing_clean_dcp.get('root')} "
                    f"primary_clock={timing_clean_dcp.get('clock_summary', {}).get('primary_clock')} "
                    f"routed_latency_ms={timing_clean_dcp.get('routed_latency_ms')} "
                    f"baseline_routed_latency_ms={baseline_routed_latency}"
                )
            ),
        ),
    ]
    complete = all(item["status"] == "pass" for item in requirements)
    return {
        "artifact_type": "siglip2_86m_quant_improvement_status",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective_complete": complete,
        "target": {
            "model_id": EXPECTED_MODEL_ID,
            "image_size": EXPECTED_IMAGE_SIZE,
            "patch_size": EXPECTED_PATCH_SIZE,
            "vision_depth": EXPECTED_VISION_DEPTH,
            "baseline_weight_bits": BASELINE_WEIGHT_BITS,
            "baseline_act_bits": BASELINE_ACT_BITS,
            "quantizer": EXPECTED_QUANTIZER,
            "eval_images": EXPECTED_EVAL_IMAGES,
            "accuracy_requirement": "top1_percent >= fp32_top1_percent",
            "latency_requirement": "candidate cycles and estimate latency lower than baseline W6A8",
            "clock_ns_for_estimates": clock_ns,
            "min_dcp_clock_mhz": MIN_DCP_CLOCK_MHZ,
            "primary_dcp_clock": PRIMARY_DCP_CLOCK,
            "vck190_device": EXPECTED_VCK190_DEVICE,
        },
        "fp32": fp32,
        "baseline": {
            "estimate": baseline,
            "estimate_path": str(baseline_estimate),
            "dcp_timing_path": str(baseline_dcp_timing),
            "clock_summary": baseline_clock,
            "routed_latency_ms": baseline_routed_latency,
            "ok": baseline_ok,
        },
        "qat_reports": qat_reports,
        "best_qat_report": best_qat,
        "estimates": estimates,
        "estimate_candidates": estimate_candidates,
        "best_estimate": best_estimate,
        "dcps": dcps,
        "timing_clean_dcp": timing_clean_dcp,
        "requirements": requirements,
        "blockers": [
            f"{item['requirement']}: {item['evidence']}"
            for item in requirements
            if item["blocking"]
        ],
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# SigLIP2 86M Quantization Improvement Status",
        "",
        f"- generated UTC: `{report['generated_at_utc']}`",
        f"- objective complete: `{str(report['objective_complete']).lower()}`",
        f"- FP32 top1: `{fmt_percent(report.get('fp32', {}).get('top1_percent'))}`",
        f"- baseline estimate latency: `{report.get('baseline', {}).get('estimate', {}).get('latency_ms')}` ms",
        f"- baseline routed latency: `{report.get('baseline', {}).get('routed_latency_ms')}` ms",
        "",
        "| Requirement | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for item in report["requirements"]:
        evidence = str(item["evidence"]).replace("|", "\\|")
        lines.append(f"| `{item['requirement']}` | `{item['status']}` | {evidence} |")
    blockers = report.get("blockers") or []
    if blockers:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in blockers)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, default=DEFAULT_BUILD_ROOT)
    parser.add_argument("--fp32-report", type=Path, default=DEFAULT_FP32_REPORT)
    parser.add_argument("--baseline-estimate", type=Path, default=DEFAULT_BASELINE_ESTIMATE)
    parser.add_argument("--baseline-dcp-timing", type=Path, default=DEFAULT_BASELINE_DCP_TIMING)
    parser.add_argument("--clock-ns", type=float, default=DEFAULT_CLOCK_NS)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args()

    report = summarize(
        build_root=args.build_root,
        fp32_report=args.fp32_report,
        baseline_estimate=args.baseline_estimate,
        baseline_dcp_timing=args.baseline_dcp_timing,
        clock_ns=args.clock_ns,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.md_out, report)
    print(json.dumps(report["requirements"], indent=2))
    return 0 if report["objective_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
