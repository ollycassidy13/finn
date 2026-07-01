#!/usr/bin/env python3
"""Materialize a SO400M overlapped FINNLoop scheduler artifact.

This creates an auditable graph/RTL artifact for the valid overlapped
throughput model and can annotate a FINNLoop graph for estimate-only runs. The
default generated RTL contract is deliberately not treated as DCP-ready
scheduler RTL. The guarded ``builtin_stream_feedback`` mode annotates the graph
for the concrete built-in stream-feedback MLO shell.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_SCHEDULER_SPEC_JSON = Path(
    "full_siglip/build/so400m_overlapped_scheduler_spec.json"
)
DEFAULT_INPUT_ONNX = Path(
    "full_siglip/build/"
    "static_qat_so400m_patch14_384_w6a8_qvlsq_step200_featurehead32k_freshbias_lr3e4_ep60_cal0_export_mlo_fps1200_w72_pumped_embedding_overlapped_target10x_est/"
    "intermediate_models/supported_op_partitions/partition_0.onnx"
)
DEFAULT_OUTPUT_DIR = Path(
    "full_siglip/build/so400m_overlapped_scheduler_materialized"
)
DEFAULT_JSON_OUT = Path(
    "full_siglip/build/so400m_overlapped_scheduler_materialization.json"
)
DEFAULT_MD_OUT = Path(
    "full_siglip/build/so400m_overlapped_scheduler_materialization.md"
)
ALLOWED_GRAPH_OR_RTL_SUFFIXES = {".onnx", ".v", ".sv", ".vhd", ".vhdl"}
IMPLEMENTATION_KINDS = {"contract", "builtin_stream_feedback"}
STREAM_FEEDBACK_ARTIFACT = "stream_feedback"
STREAM_FEEDBACK_RTL_SOURCES = (
    Path("finn-rtllib/mlo/overlapped_loop_control.sv"),
    Path("finn-rtllib/mlo/infrastructure/streamed_feedback_frames.sv"),
)


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def display_path(path: Path, source_root: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(source_root.resolve()))
    except ValueError:
        return str(path)


def sv_identifier(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_$]", "_", name)


def graph_or_rtl_artifact(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_GRAPH_OR_RTL_SUFFIXES


def scheduler_contract_text(
    scheduler_spec: dict[str, Any],
    *,
    loop_name: str,
    module_name: str,
) -> str:
    schedule = scheduler_spec.get("schedule_model", {})
    exact = scheduler_spec.get("exact_scope", {})
    floor = scheduler_spec.get("resource_floor", {})
    body_ii = int(schedule.get("body_initiation_interval_cycles") or 0)
    iterations = int(schedule.get("loop_iterations") or 0)
    overhead = int(schedule.get("loop_overhead_per_iter") or 40)
    total_cycles = int(schedule.get("total_cycles_with_non_loop") or 0)
    cycle_budget = int(schedule.get("cycle_budget") or 0)
    required_budgets = int(floor.get("required_total_vck190_budgets") or 0)
    model = exact.get("model", "unknown")
    weight_bits = exact.get("weight_bits", "unknown")
    activation_bits = exact.get("activation_bits", "unknown")
    depth = exact.get("vision_depth", "unknown")
    image_size = exact.get("image_size", "unknown")
    tokens = exact.get("image_tokens", "unknown")

    return "\n".join(
        [
            "`default_nettype none",
            f"module {module_name} #(",
            f"    parameter int BODY_II_CYCLES = {body_ii},",
            f"    parameter int LOOP_ITERATIONS = {iterations},",
            f"    parameter int LOOP_OVERHEAD_PER_ITER = {overhead},",
            f"    parameter int TOTAL_CYCLES_WITH_NON_LOOP = {total_cycles},",
            f"    parameter int CYCLE_BUDGET = {cycle_budget},",
            f"    parameter int REQUIRED_TOTAL_VCK190_BUDGETS = {required_budgets}",
            ") ();",
            "",
            "    // Contract artifact for the SO400M overlapped FINNLoop schedule.",
            "    // This module records the estimate contract only. It does not",
            "    // implement AXI streams, MLO feedback replay, static parameter",
            "    // replay, or a routable overlapped scheduler datapath.",
            f"    // loop: {loop_name}",
            f"    // model: {model}",
            f"    // quantization: W{weight_bits}A{activation_bits}",
            f"    // depth/image/tokens: {depth}/{image_size}/{tokens}",
            "",
            "endmodule",
            "`default_nettype wire",
            "",
        ]
    )


def write_scheduler_contract(
    scheduler_spec: dict[str, Any],
    *,
    output_dir: Path,
    source_root: Path,
    loop_name: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    module_name = sv_identifier(f"{loop_name}_overlapped_scheduler_contract")
    artifact_path = output_dir / f"{module_name}.sv"
    artifact_path.write_text(
        scheduler_contract_text(
            scheduler_spec,
            loop_name=loop_name,
            module_name=module_name,
        )
    )
    return {
        "path": str(artifact_path),
        "display_path": display_path(artifact_path, source_root),
        "module_name": module_name,
        "exists": artifact_path.is_file(),
        "is_graph_or_rtl": graph_or_rtl_artifact(artifact_path),
    }


def stream_feedback_builtin_artifact(*, source_root: Path) -> dict[str, Any]:
    rtl_sources = []
    for rel_path in STREAM_FEEDBACK_RTL_SOURCES:
        path = source_root / rel_path
        rtl_sources.append(
            {
                "path": str(rel_path),
                "abs_path": str(path.resolve()),
                "exists": path.is_file(),
                "is_graph_or_rtl": graph_or_rtl_artifact(path),
            }
        )
    primary = source_root / STREAM_FEEDBACK_RTL_SOURCES[0]
    return {
        "path": str(primary.resolve()),
        "display_path": STREAM_FEEDBACK_ARTIFACT,
        "annotation_artifact": STREAM_FEEDBACK_ARTIFACT,
        "module_name": "overlapped_loop_control",
        "exists": all(item["exists"] for item in rtl_sources),
        "is_graph_or_rtl": all(item["is_graph_or_rtl"] for item in rtl_sources),
        "rtl_sources": rtl_sources,
    }


def annotate_model_with_overlapped_scheduler(
    model: Any,
    *,
    scheduler_spec: dict[str, Any],
    artifact_path: Path,
    source_root: Path,
    loop_name: str,
    dcp_ready: bool = False,
    artifact_ref_override: str | None = None,
) -> dict[str, Any]:
    from finn.util.basic import getHWCustomOp

    loops = list(model.get_nodes_by_op_type("FINNLoop"))
    loop = None
    for candidate in loops:
        if candidate.name == loop_name:
            loop = candidate
            break
    if loop is None and len(loops) == 1:
        loop = loops[0]
    if loop is None:
        return {
            "attempted": True,
            "attrs_set": False,
            "loop_count": len(loops),
            "loop_name": loop_name,
            "reason": f"could not find unique FINNLoop named {loop_name}",
        }

    schedule = scheduler_spec.get("schedule_model", {})
    body_ii_cycles = int(schedule["body_initiation_interval_cycles"])
    artifact_ref = (
        artifact_ref_override
        if artifact_ref_override is not None
        else display_path(artifact_path, source_root)
    )
    inst = getHWCustomOp(loop, model)
    inst.set_nodeattr("loop_scheduler_mode", "overlapped")
    inst.set_nodeattr("overlapped_body_ii_cycles", body_ii_cycles)
    inst.set_nodeattr("overlapped_scheduler_artifact", artifact_ref)
    inst.set_nodeattr("overlapped_scheduler_dcp_ready", int(dcp_ready))
    return {
        "attempted": True,
        "attrs_set": True,
        "loop_count": len(loops),
        "loop_name": loop.name,
        "loop_scheduler_mode": "overlapped",
        "overlapped_body_ii_cycles": body_ii_cycles,
        "overlapped_scheduler_artifact": artifact_ref,
        "overlapped_scheduler_dcp_ready": bool(dcp_ready),
    }


def scheduler_spec_ok(scheduler_spec: dict[str, Any]) -> bool:
    checks = scheduler_spec.get("checks", {})
    schedule = scheduler_spec.get("schedule_model", {})
    return (
        checks.get("spec_valid_for_objective") is True
        and isinstance(schedule.get("body_initiation_interval_cycles"), int)
        and int(schedule.get("body_initiation_interval_cycles")) > 0
    )


def summarize_materialization(
    scheduler_spec: dict[str, Any],
    *,
    output_dir: Path,
    source_root: Path,
    scheduler_spec_path: Path | None = None,
    input_onnx: Path | None = None,
    annotated_onnx: Path | None = None,
    loop_name: str = "FINNLoop_0",
    model: Any | None = None,
    skip_graph_annotation: bool = False,
    allow_missing_onnx_deps: bool = False,
    ready_for_dcp_preflight: bool = False,
    implementation_kind: str = "contract",
) -> tuple[Any | None, dict[str, Any]]:
    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    if implementation_kind not in IMPLEMENTATION_KINDS:
        raise ValueError(f"unknown implementation_kind {implementation_kind}")
    if implementation_kind == "builtin_stream_feedback":
        artifact = stream_feedback_builtin_artifact(source_root=source_root)
    else:
        artifact = write_scheduler_contract(
            scheduler_spec,
            output_dir=output_dir,
            source_root=source_root,
            loop_name=loop_name,
        )
    artifact_path = Path(artifact["path"])
    annotation_artifact = artifact.get("annotation_artifact")
    schedule = scheduler_spec.get("schedule_model", {})
    annotated_onnx = (
        output_dir / "overlapped_scheduler_annotated.onnx"
        if annotated_onnx is None
        else annotated_onnx
    )

    annotation = {
        "attempted": False,
        "attrs_set": False,
        "reason": "graph annotation skipped",
    }
    loaded_model = model
    if not skip_graph_annotation:
        try:
            if loaded_model is None:
                if input_onnx is None:
                    raise RuntimeError("no input ONNX supplied")
                from qonnx.core.modelwrapper import ModelWrapper

                loaded_model = ModelWrapper(str(input_onnx))
            annotation = annotate_model_with_overlapped_scheduler(
                loaded_model,
                scheduler_spec=scheduler_spec,
                artifact_path=artifact_path,
                source_root=source_root,
                loop_name=loop_name,
                dcp_ready=implementation_kind == "builtin_stream_feedback",
                artifact_ref_override=annotation_artifact,
            )
            if annotation["attrs_set"]:
                annotated_onnx.parent.mkdir(parents=True, exist_ok=True)
                loaded_model.save(str(annotated_onnx))
        except (ImportError, ModuleNotFoundError) as err:
            if not allow_missing_onnx_deps:
                raise
            annotation = {
                "attempted": True,
                "attrs_set": False,
                "reason": f"ONNX/QONNX dependency unavailable: {err}",
            }
        except Exception as err:
            annotation = {
                "attempted": True,
                "attrs_set": False,
                "reason": str(err),
            }

    spec_valid = scheduler_spec_ok(scheduler_spec)
    annotated_exists = annotated_onnx.is_file()
    ready = (
        implementation_kind == "builtin_stream_feedback"
        and spec_valid
        and artifact["exists"]
        and artifact["is_graph_or_rtl"]
        and annotation.get("attrs_set") is True
    )
    blockers = []
    if not spec_valid:
        blockers.append("scheduler spec is not valid for the exact objective")
    if not artifact["exists"]:
        blockers.append("scheduler artifact was not written")
    if not artifact["is_graph_or_rtl"]:
        blockers.append("scheduler artifact is not graph or RTL")
    if not annotation.get("attrs_set"):
        blockers.append(
            f"annotated FINNLoop graph not available: {annotation.get('reason', 'unknown')}"
        )
    if implementation_kind == "contract":
        blockers.append(
            "materialized artifact is an estimate contract and is not DCP-ready scheduler RTL"
        )
    if ready_for_dcp_preflight:
        if implementation_kind == "contract":
            blockers.append(
                "--ready-for-dcp-preflight was requested, but generated contract artifacts "
                "cannot be promoted to DCP-ready scheduler RTL"
            )
        elif not ready:
            blockers.append(
                "--ready-for-dcp-preflight was requested, but the built-in stream-feedback "
                "implementation is not ready for DCP preflight"
            )

    report = {
        "artifact_type": "overlapped_scheduler_materialization",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_kind": implementation_kind,
        "implementation_status": (
            "builtin_stream_feedback_dcp_preflight_ready"
            if ready
            else (
                "builtin_stream_feedback_not_dcp_ready"
                if implementation_kind == "builtin_stream_feedback"
                else "materialized_contract_not_dcp_ready"
            )
        ),
        "source": {
            "scheduler_spec": None
            if scheduler_spec_path is None
            else display_path(scheduler_spec_path, source_root),
            "input_onnx": None if input_onnx is None else display_path(input_onnx, source_root),
        },
        "implementation_artifact": artifact["display_path"],
        "implementation_artifact_abs": artifact["path"],
        "annotation_artifact": annotation_artifact or artifact["display_path"],
        "annotated_graph": display_path(annotated_onnx, source_root),
        "annotated_graph_abs": str(annotated_onnx),
        "exact_scope": scheduler_spec.get("exact_scope", {}),
        "schedule_model": {
            "mode": schedule.get("mode"),
            "loop_iterations": schedule.get("loop_iterations"),
            "body_initiation_interval_cycles": schedule.get(
                "body_initiation_interval_cycles"
            ),
            "body_cycle_budget": schedule.get("body_cycle_budget"),
            "total_cycles_with_non_loop": schedule.get("total_cycles_with_non_loop"),
            "cycle_budget": schedule.get("cycle_budget"),
            "latency_ms": schedule.get("latency_ms"),
        },
        "artifact": artifact,
        "graph_annotation": annotation,
        "checks": {
            "scheduler_spec_valid_for_objective": spec_valid,
            "implementation_artifact_exists": artifact["exists"],
            "implementation_artifact_is_graph_or_rtl": artifact["is_graph_or_rtl"],
            "annotated_graph_exists": annotated_exists,
            "annotated_graph_attrs_set": annotation.get("attrs_set") is True,
            "ready_for_dcp_preflight": ready,
            "stream_feedback_rtl_sources_exist": (
                implementation_kind != "builtin_stream_feedback"
                or all(item["exists"] for item in artifact.get("rtl_sources", []))
            ),
        },
        "blockers": blockers,
    }
    return loaded_model, report


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    schedule = report["schedule_model"]
    checks = report["checks"]
    lines = [
        "# SO400M Overlapped Scheduler Materialization",
        "",
        "This report records the concrete scheduler artifact and any FINNLoop graph annotation. Contract artifacts are not DCP-ready scheduler RTL; the guarded built-in stream-feedback mode is a DCP-preflight candidate.",
        "",
        f"- status: `{report['implementation_status']}`",
        f"- implementation kind: `{report['implementation_kind']}`",
        f"- implementation artifact: `{report['implementation_artifact']}`",
        f"- annotation artifact: `{report['annotation_artifact']}`",
        f"- annotated graph: `{report['annotated_graph']}`",
        f"- body initiation interval cycles: {schedule.get('body_initiation_interval_cycles', 'missing')}",
        f"- total cycles with non-loop: {schedule.get('total_cycles_with_non_loop', 'missing')}",
        f"- latency: {schedule.get('latency_ms', 'missing')} ms",
        "",
        "## Graph Annotation",
        "",
    ]
    annotation = report["graph_annotation"]
    for name, value in annotation.items():
        lines.append(f"- {name}: `{value}`")
    lines.extend(["", "## Checks", ""])
    for name, value in checks.items():
        lines.append(f"- {name}: `{str(value).lower()}`")
    lines.extend(["", "## Blockers", ""])
    if report["blockers"]:
        lines.extend(f"- {item}" for item in report["blockers"])
    else:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scheduler-spec-json",
        type=Path,
        default=DEFAULT_SCHEDULER_SPEC_JSON,
    )
    parser.add_argument("--input-onnx", type=Path, default=DEFAULT_INPUT_ONNX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--annotated-onnx", type=Path, default=None)
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--loop-name", default="FINNLoop_0")
    parser.add_argument("--skip-graph-annotation", action="store_true")
    parser.add_argument("--allow-missing-onnx-deps", action="store_true")
    parser.add_argument(
        "--ready-for-dcp-preflight",
        action="store_true",
        help="Only use with a real scheduler implementation; the generated contract stays false by default.",
    )
    parser.add_argument(
        "--implementation-kind",
        choices=sorted(IMPLEMENTATION_KINDS),
        default="contract",
        help=(
            "Use contract for estimate-only auditing, or builtin_stream_feedback "
            "to annotate the FINNLoop for the guarded stream-feedback MLO shell."
        ),
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args()

    _, report = summarize_materialization(
        load_json(args.scheduler_spec_json),
        output_dir=args.output_dir,
        source_root=args.source_root,
        scheduler_spec_path=args.scheduler_spec_json,
        input_onnx=args.input_onnx,
        annotated_onnx=args.annotated_onnx,
        loop_name=args.loop_name,
        skip_graph_annotation=args.skip_graph_annotation,
        allow_missing_onnx_deps=args.allow_missing_onnx_deps,
        ready_for_dcp_preflight=args.ready_for_dcp_preflight,
        implementation_kind=args.implementation_kind,
    )
    write_json(args.json_out, report)
    write_markdown(args.md_out, report)
    print(json.dumps(report, indent=2))
    ok = (
        report["checks"]["scheduler_spec_valid_for_objective"]
        and report["checks"]["implementation_artifact_exists"]
        and report["checks"]["implementation_artifact_is_graph_or_rtl"]
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
