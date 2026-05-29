#!/usr/bin/env python3
"""Run the TinyDeiT FINN MLO build for V80."""

from __future__ import annotations

import argparse
import csv
import json
import numpy as np
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import gen_finn_dt_tensor

import finn.builder.build_dataflow as build
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    VerificationStepType,
)
from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.set_folding import SetFolding
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.util.basic import getHWCustomOp
from finn.util.config import extract_model_config_to_json
from finn.transformation.general import ApplyConfig
from qonnx.transformation.general import GiveUniqueNodeNames
from tinydeit.common import (
    DEFAULT_BOARD,
    DEFAULT_BUILD_CSV,
    DEFAULT_BUILD_DIR,
    DEFAULT_CHECKPOINT,
    DEFAULT_CLOCK_NS,
    DEFAULT_TARGET_FPS,
    repo_path,
)
from tinydeit.prepare_model import prepare


def _rounded_threshold_datatype(dtype: DataType) -> DataType:
    max_val = dtype.max() + 1
    if not dtype.signed():
        return DataType.get_smallest_possible(max_val)
    return DataType.get_smallest_possible(-(max_val) - 1)


def step_round_mlo_threshold_params(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Round FINNLoop threshold parameter tensors after bit-width minimization.

    RoundAndClipThresholds handles regular Thresholding initializers, but MLO
    rolling turns per-layer thresholds into indexed FINNLoop inputs.  The RTL
    threshold generator still expects those parameter tensors and the loop-body
    weightDataType attributes to reflect integer thresholds.
    """

    for loop_node in model.get_nodes_by_op_type("FINNLoop"):
        loop_inst = getHWCustomOp(loop_node)
        loop_body = loop_inst.get_nodeattr("body")
        body_input_names = [value_info.name for value_info in loop_body.graph.input]

        for node in loop_body.graph.node:
            if not node.op_type.startswith("Thresholding"):
                continue
            inst = getHWCustomOp(node)
            dtype = inst.get_input_datatype(0)
            if not dtype.is_integer():
                continue
            try:
                body_input_index = body_input_names.index(node.input[1])
            except ValueError:
                continue

            top_param_name = loop_node.input[body_input_index]
            thresholds = model.get_initializer(top_param_name)
            if thresholds is None:
                raise RuntimeError(f"Missing FINNLoop threshold parameter {top_param_name}")

            new_thresholds = np.clip(np.ceil(thresholds), dtype.min(), dtype.max() + 1)
            new_thresholds = new_thresholds.astype(np.float32)
            tdt = _rounded_threshold_datatype(dtype)
            model.set_initializer(top_param_name, new_thresholds)
            model.set_tensor_datatype(top_param_name, tdt)
            loop_body.set_tensor_datatype(node.input[1], tdt)
            inst.set_nodeattr("weightDataType", tdt.name)

        loop_inst.set_nodeattr("body", loop_body.graph)
    return model


DEFAULT_FOLDING_TARGET_CYCLES = 15000

FOLDING_HW_ATTRS = [
    "PE",
    "SIMD",
    "parallel_window",
    "ram_style",
    "resType",
    "mem_mode",
    "runtime_writeable_weights",
    "depth_trigger_uram",
    "depth_trigger_bram",
]


def _canonicalize_loop_body_names(model: ModelWrapper) -> ModelWrapper:
    model = model.transform(GiveUniqueNodeNames())
    for node in model.get_nodes_by_op_type("FINNLoop"):
        node_inst = getHWCustomOp(node)
        loop_body = node_inst.get_nodeattr("body")
        loop_body = loop_body.transform(GiveUniqueNodeNames(prefix=node.name + "_"))
        node_inst.set_nodeattr("body", loop_body.graph)
    return model


def step_tinydeit_post_transpose_parallelization(
    model: ModelWrapper, cfg: DataflowBuildConfig
) -> ModelWrapper:
    """Re-run folding after Shuffle decomposition creates final shuffle nodes.

    FINN's standard target-fps folding step runs before TinyDeiT's transpose
    decomposition.  The actual stitched design contains InnerShuffle/OuterShuffle
    nodes created later, so this final pass makes their SIMD and the surrounding
    RTL operators match the same aggressive target.
    """

    if not getattr(cfg, "tinydeit_post_transpose_folding", True):
        return model

    target_cycles_per_frame = cfg._resolve_cycles_per_frame()
    if target_cycles_per_frame is None:
        print("No target_fps provided, skipping step_tinydeit_post_transpose_parallelization.")
        return model

    model = model.transform(
        SetFolding(
            target_cycles_per_frame,
            mvau_wwidth_max=cfg.mvau_wwidth_max,
            two_pass_relaxation=cfg.folding_two_pass_relaxation,
        ),
        apply_to_subgraphs=True,
    )
    model = _canonicalize_loop_body_names(model)
    if cfg.folding_config_file is not None:
        model = model.transform(ApplyConfig(cfg.folding_config_file), apply_to_subgraphs=True)
        model = _canonicalize_loop_body_names(model)
    extract_model_config_to_json(
        model, cfg.output_dir + "/auto_folding_config.json", FOLDING_HW_ATTRS
    )
    return model


BUILD_STEPS_ESTIMATE = [
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    step_round_mlo_threshold_params,
    "step_transpose_decomposition",
    step_tinydeit_post_transpose_parallelization,
    "step_generate_estimate_reports",
]

BUILD_STEPS_RTL = BUILD_STEPS_ESTIMATE + [
    "step_hw_codegen",
    "step_hw_ipgen",
    "step_set_fifo_depths",
    "step_create_stitched_ip",
]

BUILD_STEPS_FULL_RTLSIM = BUILD_STEPS_RTL + ["step_measure_rtlsim_performance"]


def build_config(args: argparse.Namespace, output_dir: Path) -> DataflowBuildConfig:
    if args.mode == "estimate":
        steps = BUILD_STEPS_ESTIMATE
        outputs = [DataflowOutputType.ESTIMATE_REPORTS]
        verify_steps = []
    elif args.mode in ["rtl", "dcp"]:
        steps = BUILD_STEPS_RTL
        outputs = [DataflowOutputType.ESTIMATE_REPORTS, DataflowOutputType.STITCHED_IP]
        if args.mode == "dcp":
            outputs.append(DataflowOutputType.OOC_SYNTH)
        verify_steps = []
        if args.stitched_rtlsim:
            verify_steps.append(VerificationStepType.STITCHED_IP_RTLSIM)
    elif args.mode == "full-rtlsim":
        steps = BUILD_STEPS_FULL_RTLSIM
        outputs = [
            DataflowOutputType.ESTIMATE_REPORTS,
            DataflowOutputType.STITCHED_IP,
            DataflowOutputType.RTLSIM_PERFORMANCE,
        ]
        verify_steps = [VerificationStepType.STITCHED_IP_RTLSIM]
    else:
        steps = BUILD_STEPS_RTL + [
            "step_out_of_context_synthesis",
            "step_synthesize_bitfile",
            "step_make_driver",
            "step_deployment_package",
        ]
        outputs = [
            DataflowOutputType.ESTIMATE_REPORTS,
            DataflowOutputType.STITCHED_IP,
            DataflowOutputType.OOC_SYNTH,
            DataflowOutputType.BITFILE,
            DataflowOutputType.CPP_DRIVER,
            DataflowOutputType.DEPLOYMENT_PACKAGE,
        ]
        verify_steps = [VerificationStepType.FOLDED_HLS_CPPSIM]

    cfg = DataflowBuildConfig(
        output_dir=str(output_dir),
        synth_clk_period_ns=args.clock_ns,
        board=args.board,
        target_fps=args.target_fps,
        mvau_wwidth_max=args.mvau_wwidth_max,
        folding_two_pass_relaxation=args.folding_two_pass_relaxation,
        standalone_thresholds=True,
        infer_shuffle_skip_first=False,
        folding_config_file=args.folding_config_file,
        save_intermediate_models=True,
        verify_steps=verify_steps,
        verify_input_npy=str(output_dir / "input.npy"),
        verify_expected_output_npy=str(output_dir / "expected_output.npy"),
        verify_save_full_context=True,
        verification_atol=args.atol,
        generate_outputs=outputs,
        steps=steps,
        # MLO rolling is performed explicitly in prepare_model.py before this
        # build starts. Keep cfg.mlo disabled here so build_dataflow does not
        # require loop-body metadata for another rolling pass.
        mlo=False,
        auto_fifo_depths=False,
        fifosim_n_inferences=args.fifosim_n_inferences,
        rtlsim_batch_size=args.rtlsim_batch_size,
        stitched_rtlsim_liveness_threshold=args.stitched_rtlsim_liveness_threshold,
        stitched_ip_gen_dcp=args.mode == "dcp" or args.stitched_ip_dcp,
        no_stdout_redirect=True,
        enable_build_pdb_debug=False,
    )
    cfg.tinydeit_post_transpose_folding = args.post_transpose_folding
    return cfg


BUILD_CSV_FIELDS = [
    "timestamp_utc",
    "hostname",
    "git_commit",
    "mode",
    "board",
    "clock_ns",
    "target_fps",
    "return_code",
    "timing_status",
    "wns_ns",
    "fmax_mhz",
    "estimated_throughput_fps",
    "resources",
    "folding_pe_simd",
    "build_step_times",
    "output_dir",
    "model_path",
    "report_dir",
    "stitched_ip_dir",
    "dcp_paths",
]

FAILED_VIVADO_ARTIFACT_SUFFIXES = {
    ".csv",
    ".jou",
    ".json",
    ".log",
    ".rpt",
    ".str",
    ".tcl",
    ".txt",
    ".xdc",
    ".xml",
}
FAILED_VIVADO_TOP_DCP_NAMES = {
    "finn_design.dcp",
    "finn_design_routed.dcp",
}

FAILED_VIVADO_CLEAN_PREFIXES = (
    "code_gen_ipgen_",
    "rtlsim_",
    "vivado_stitch_proj_",
    "vivado_zynq_proj_",
    "vitis_floorplan_",
    "vitis_link_proj_",
)

VIVADO_HARD_FAILURE_MARKERS = (
    "opt_design failed",
    "place_design failed",
    "route_design failed",
    "Macro Placer failed",
    "Command failed: Placer could not place all instances",
)


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        with path.open() as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        print(f"WARNING: ignoring invalid JSON in {path}: {exc}")
        return {}


def _json_cell(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_path("."),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _folding_summary(config: dict) -> dict:
    attrs = ["PE", "SIMD", "TH", "MW", "MH", "mem_mode", "resType", "ram_style", "gemm_type"]
    summary = {}
    for node_name, node_cfg in sorted(config.items()):
        if node_name == "Defaults" or not isinstance(node_cfg, dict):
            continue
        node_summary = {attr: node_cfg[attr] for attr in attrs if attr in node_cfg}
        if any(attr in node_summary for attr in ["PE", "SIMD", "TH"]):
            summary[node_name] = node_summary
    return summary


def _parse_int_cell(cell: str) -> int:
    return int(cell.replace(",", "").strip())


def _partition_resource_summary(report_path: Path) -> dict:
    if not report_path.is_file():
        return {}

    key_map = {
        "Total LUTs": "stitched_LUT",
        "Logic LUTs": "stitched_Logic_LUT",
        "LUTRAMs": "stitched_LUTRAM",
        "SRLs": "stitched_SRL",
        "FFs": "stitched_FF",
        "RAMB36": "stitched_BRAM_36K",
        "RAMB18": "stitched_BRAM_18K",
        "URAM": "stitched_URAM",
        "DSP Blocks": "stitched_DSP",
    }
    headers = None
    with report_path.open() as f:
        for line in f:
            if not line.startswith("|"):
                continue
            cells = [cell.strip() for cell in line.split("|")[1:-1]]
            if len(cells) < 2:
                continue
            if cells[0] == "Instance" and cells[1] == "Module":
                headers = cells
                continue
            if cells[0] == "finn_design_wrapper" and headers is not None:
                return {
                    key_map[header]: _parse_int_cell(value)
                    for header, value in zip(headers, cells)
                    if header in key_map
                }
    return {}


def _stitched_ip_dirs(output_dir: Path) -> list[Path]:
    """Return current and retained failed-build stitched-IP locations."""

    candidates = [
        output_dir / "stitched_ip",
        output_dir / "failed_vivado_artifacts" / "output_dir" / "stitched_ip",
    ]
    return [path for path in candidates if path.is_dir()]


def _primary_stitched_ip_dir(output_dir: Path) -> Path:
    stitched_dirs = _stitched_ip_dirs(output_dir)
    return stitched_dirs[0] if stitched_dirs else output_dir / "stitched_ip"


def _resource_summary(ooc: dict, post_synth: dict, output_dir: Path) -> dict:
    resource_keys = [
        "LUT",
        "FF",
        "DSP",
        "BRAM",
        "BRAM_18K",
        "BRAM_36K",
        "URAM",
        "SRL",
        "total_power_W",
    ]
    source = ooc or post_synth or {}
    summary = {key: source[key] for key in resource_keys if key in source}
    for stitched_ip_dir in _stitched_ip_dirs(output_dir):
        partition_report = stitched_ip_dir / "finn_design_partition_util.rpt"
        partition_summary = _partition_resource_summary(partition_report)
        if partition_summary:
            summary.update(partition_summary)
            break
    return summary


def _timing_report_summary(output_dir: Path) -> dict:
    report_path = None
    for stitched_ip_dir in _stitched_ip_dirs(output_dir):
        candidate = stitched_ip_dir / "ooc_timing.rpt"
        if candidate.is_file():
            report_path = candidate
            break
    if report_path is None:
        report_path = output_dir / "stitched_ip" / "ooc_timing.rpt"
    if not report_path.is_file():
        return {}

    patterns = {
        "setup": re.compile(
            r"^\s*Setup\s*:\s*(\d+)\s+Failing Endpoints,\s+"
            r"Worst Slack\s+([-+0-9.]+)ns,\s+Total Violation\s+([-+0-9.]+)ns"
        ),
        "hold": re.compile(
            r"^\s*Hold\s*:\s*(\d+)\s+Failing Endpoints,\s+"
            r"Worst Slack\s+([-+0-9.]+)ns,\s+Total Violation\s+([-+0-9.]+)ns"
        ),
        "pulse_width": re.compile(
            r"^\s*PW\s*:\s*(\d+)\s+Failing Endpoints,\s+"
            r"Worst Slack\s+([-+0-9.]+)ns,\s+Total Violation\s+([-+0-9.]+)ns"
        ),
    }
    summary = {"timing_report_path": str(report_path.resolve())}
    with report_path.open() as f:
        for line in f:
            for name, pattern in patterns.items():
                match = pattern.match(line)
                if match is None:
                    continue
                failing, slack, violation = match.groups()
                summary[f"{name}_failing_endpoints"] = int(failing)
                summary[f"{name}_worst_slack_ns"] = float(slack)
                summary[f"{name}_total_violation_ns"] = float(violation)
    return summary


def _parse_rtlsim_results(results_path: Path) -> dict:
    if not results_path.is_file():
        return {}
    parsed = {}
    with results_path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            key, value = parts
            try:
                parsed[key] = int(value)
            except ValueError:
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
    return parsed


def _latest_rtlsim_results(
    output_dir: Path,
    search_build_dir: bool = True,
    min_mtime: float | None = None,
) -> Path | None:
    candidates = list(output_dir.glob("**/results.txt"))
    build_dir = os.environ.get("FINN_BUILD_DIR")
    if search_build_dir and build_dir:
        build_path = Path(build_dir)
        if build_path.is_dir():
            candidates.extend(build_path.glob("rtlsim*/results.txt"))
    fresh_candidates = []
    for path in candidates:
        if not path.is_file():
            continue
        if min_mtime is not None and path.stat().st_mtime < min_mtime:
            continue
        fresh_candidates.append(path)
    if not fresh_candidates:
        return None
    return max(fresh_candidates, key=lambda path: path.stat().st_mtime)


def _rtlsim_summary(
    output_dir: Path,
    clock_ns: float,
    search_build_dir: bool = True,
    min_mtime: float | None = None,
) -> dict:
    results_path = _latest_rtlsim_results(output_dir, search_build_dir, min_mtime)
    if results_path is None:
        return {}
    return _rtlsim_summary_from_path(results_path, clock_ns)


def _rtlsim_summary_from_path(results_path: Path, clock_ns: float) -> dict:
    raw = _parse_rtlsim_results(results_path)
    summary = {"rtlsim_results_path": str(results_path.resolve())}
    for src_key, dst_key in [
        ("cycles", "rtlsim_cycles"),
        ("latency_cycles", "rtlsim_latency_cycles"),
        ("interval_cycles", "rtlsim_interval_cycles"),
        ("TIMEOUT", "rtlsim_timeout"),
        ("UNFINISHED_INS", "rtlsim_unfinished_ins"),
        ("UNFINISHED_OUTS", "rtlsim_unfinished_outs"),
        ("RUNTIME_S", "rtlsim_runtime_s"),
    ]:
        if src_key in raw:
            summary[dst_key] = raw[src_key]
    interval = raw.get("interval_cycles")
    if interval:
        summary["rtlsim_throughput_fps"] = (10**9 / clock_ns) / float(interval)
    return summary


def _timing_status(ooc: dict, timing_report: dict | None = None) -> str:
    timing_report = timing_report or {}
    setup_slack = timing_report.get("setup_worst_slack_ns", ooc.get("WNS"))
    hold_slack = timing_report.get("hold_worst_slack_ns")
    if setup_slack is None and hold_slack is None:
        return "not_run" if not ooc else "unknown"
    if setup_slack is None:
        return "unknown"
    if float(setup_slack) < 0:
        return "failed"
    if hold_slack is not None and float(hold_slack) < 0:
        return "failed"
    return "met"


def _timing_wns(ooc: dict, timing_report: dict) -> float | str:
    if "WNS" in ooc:
        return ooc["WNS"]
    return timing_report.get("setup_worst_slack_ns", "")


def _timing_fmax_mhz(ooc: dict, timing_report: dict, clock_ns: float) -> float | str:
    if "fmax_mhz" in ooc:
        return ooc["fmax_mhz"]
    setup_slack = timing_report.get("setup_worst_slack_ns")
    if setup_slack is None:
        return ""
    period_ns = float(clock_ns) - float(setup_slack)
    if period_ns <= 0:
        return ""
    return 1000.0 / period_ns


def _output_timing_failed(output_dir: Path) -> bool:
    report_dir = output_dir / "report"
    ooc = _load_json(report_dir / "ooc_synth_and_timing.json")
    timing_report = _timing_report_summary(output_dir)
    return _timing_status(ooc, timing_report) == "failed"


def _vivado_failure_summary(args: argparse.Namespace, output_dir: Path) -> dict:
    current_stitched_ip_dir = output_dir / "stitched_ip"
    stitched_ip_dirs = (
        [current_stitched_ip_dir]
        if current_stitched_ip_dir.is_dir()
        else _stitched_ip_dirs(output_dir)
    )
    for stitched_ip_dir in stitched_ip_dirs:
        log_path = stitched_ip_dir / "vivado.log"
        if not log_path.is_file():
            continue
        with log_path.open(errors="replace") as f:
            for line in f:
                stripped = line.strip()
                if any(marker in stripped for marker in VIVADO_HARD_FAILURE_MARKERS):
                    return {
                        "vivado_failure": stripped,
                        "vivado_log_path": str(log_path.resolve()),
                    }

    expects_dcp_timing = args.mode == "dcp" or bool(getattr(args, "stitched_ip_dcp", False))
    if expects_dcp_timing and _stitched_ip_dirs(output_dir) and not _timing_report_summary(
        output_dir
    ):
        return {
            "vivado_failure": "missing expected ooc_timing.rpt after stitched IP DCP generation",
        }
    return {}


def _output_vivado_failed(args: argparse.Namespace, output_dir: Path) -> bool:
    return bool(_vivado_failure_summary(args, output_dir))


def record_build_result(
    args: argparse.Namespace,
    output_dir: Path,
    model_path: Path,
    return_code: int,
    build_started_at: float | None = None,
) -> Path:
    output_dir = output_dir.resolve()
    report_dir = output_dir / "report"
    final_config = _load_json(output_dir / "final_hw_config.json")
    if not final_config:
        final_config = _load_json(output_dir / "auto_folding_config.json")
    ooc = _load_json(report_dir / "ooc_synth_and_timing.json")
    post_synth = _load_json(report_dir / "post_synth_resources.json")
    step_times = _load_json(output_dir / "time_per_step.json")
    dcp_paths = sorted(
        str(path.resolve())
        for stitched_ip_dir in _stitched_ip_dirs(output_dir)
        for path in stitched_ip_dir.glob("**/*.dcp")
    )
    timing_report = _timing_report_summary(output_dir)
    search_build_rtlsim = args.mode == "full-rtlsim" or bool(
        getattr(args, "stitched_rtlsim", False)
    )
    rtlsim_results_file = getattr(args, "rtlsim_results_file", None)
    if rtlsim_results_file:
        rtlsim_results_path = Path(rtlsim_results_file)
        if not rtlsim_results_path.is_absolute():
            rtlsim_results_path = repo_path(rtlsim_results_file)
        rtlsim_summary = _rtlsim_summary_from_path(rtlsim_results_path, args.clock_ns)
    else:
        rtlsim_summary = _rtlsim_summary(
            output_dir,
            args.clock_ns,
            search_build_dir=search_build_rtlsim,
            min_mtime=build_started_at,
        )
    resources = _resource_summary(ooc, post_synth, output_dir)
    resources.update(timing_report)
    resources.update(rtlsim_summary)
    vivado_failure = _vivado_failure_summary(args, output_dir)
    resources.update(vivado_failure)

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "git_commit": _git_commit(),
        "mode": args.mode,
        "board": args.board,
        "clock_ns": args.clock_ns,
        "target_fps": args.target_fps,
        "return_code": return_code,
        "timing_status": "failed" if vivado_failure else _timing_status(ooc, timing_report),
        "wns_ns": _timing_wns(ooc, timing_report),
        "fmax_mhz": _timing_fmax_mhz(ooc, timing_report, args.clock_ns),
        "estimated_throughput_fps": rtlsim_summary.get(
            "rtlsim_throughput_fps", ooc.get("estimated_throughput_fps", "")
        ),
        "resources": _json_cell(resources),
        "folding_pe_simd": _json_cell(_folding_summary(final_config)),
        "build_step_times": _json_cell(step_times),
        "output_dir": str(output_dir),
        "model_path": str(model_path.resolve()),
        "report_dir": str(report_dir),
        "stitched_ip_dir": str(_primary_stitched_ip_dir(output_dir).resolve()),
        "dcp_paths": _json_cell(dcp_paths),
    }

    csv_path = repo_path(args.build_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.is_file()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BUILD_CSV_FIELDS, lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return csv_path


def _snapshot_dir_entries(path: Path | None) -> set[Path]:
    if path is None or not path.is_dir():
        return set()
    return {entry.resolve() for entry in path.iterdir()}


def _failed_cleanup_snapshot(output_dir: Path) -> dict:
    finn_build_dir_env = os.environ.get("FINN_BUILD_DIR")
    finn_build_dir = Path(finn_build_dir_env).resolve() if finn_build_dir_env else None
    return {
        "finn_build_dir": finn_build_dir,
        "finn_build_entries": _snapshot_dir_entries(finn_build_dir),
        "output_entries": _snapshot_dir_entries(output_dir),
    }


def _looks_like_vivado_cleanup_dir(path: Path) -> bool:
    name = path.name
    return (
        name in {".Xil", "vivado_ip_cache"}
        or name.startswith(FAILED_VIVADO_CLEAN_PREFIXES)
        or ("vivado" in name.lower() and "proj" in name.lower())
    )


def _artifact_file(path: Path, rel_path: Path) -> bool:
    if not path.is_file():
        return False
    suffix = path.suffix.lower()
    if suffix == ".dcp":
        return len(rel_path.parts) == 1 and path.name in FAILED_VIVADO_TOP_DCP_NAMES
    return suffix in FAILED_VIVADO_ARTIFACT_SUFFIXES


def _preserve_failed_vivado_artifacts(src: Path, dst_root: Path) -> int:
    if not src.exists():
        return 0
    preserved = 0
    for artifact in sorted(src.rglob("*")):
        rel = artifact.relative_to(src)
        if not _artifact_file(artifact, rel):
            continue
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            artifact.rename(dst)
        except OSError:
            shutil.copy2(artifact, dst)
            artifact.unlink()
        preserved += 1
    return preserved


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _path_contains(candidate: Path, active_path: Path) -> bool:
    try:
        active_path.relative_to(candidate)
    except ValueError:
        return False
    return True


def _path_has_active_process_handle(path: Path) -> bool:
    try:
        candidate = path.resolve()
    except OSError:
        return False
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return False
    for proc_entry in proc_root.iterdir():
        if not proc_entry.name.isdigit():
            continue
        try:
            cwd = (proc_entry / "cwd").resolve()
        except OSError:
            cwd = None
        if cwd is not None and _path_contains(candidate, cwd):
            return True

        fd_dir = proc_entry / "fd"
        try:
            fd_entries = tuple(fd_dir.iterdir())
        except OSError:
            continue
        for fd_entry in fd_entries:
            try:
                fd_target = os.readlink(fd_entry)
            except OSError:
                continue
            if fd_target.endswith(" (deleted)"):
                fd_target = fd_target[: -len(" (deleted)")]
            if not fd_target.startswith("/"):
                continue
            try:
                fd_path = Path(fd_target).resolve()
            except OSError:
                continue
            if _path_contains(candidate, fd_path):
                return True
    return False


def cleanup_failed_vivado_projects(
    output_dir: Path,
    snapshot: dict,
    keep_artifacts: bool,
) -> None:
    """Remove bulky Vivado/IP project dirs left by an unsuccessful TinyDeiT build."""

    candidates: list[tuple[str, Path, Path]] = []
    finn_build_dir = snapshot.get("finn_build_dir")
    finn_entries = snapshot.get("finn_build_entries", set())
    if finn_build_dir is not None and finn_build_dir.is_dir():
        for entry in finn_build_dir.iterdir():
            entry_resolved = entry.resolve()
            if entry_resolved in finn_entries:
                continue
            if _looks_like_vivado_cleanup_dir(entry):
                candidates.append(("finn_build_dir", entry, finn_build_dir))

    output_entries = snapshot.get("output_entries", set())
    if output_dir.is_dir():
        for entry in output_dir.iterdir():
            entry_resolved = entry.resolve()
            if entry_resolved in output_entries:
                continue
            if entry.name == "stitched_ip" or _looks_like_vivado_cleanup_dir(entry):
                candidates.append(("output_dir", entry, output_dir.resolve()))

    if not candidates:
        print("Failed-build Vivado cleanup: no new Vivado project dirs to remove.")
        return

    has_output_stitched_archive = any(
        source_name == "output_dir" and candidate.name == "stitched_ip"
        for source_name, candidate, _ in candidates
    )
    artifact_root = output_dir / "failed_vivado_artifacts"
    cleaned = 0
    preserved = 0
    skipped_duplicate_artifacts = 0
    for source_name, candidate, expected_parent in candidates:
        try:
            if candidate.resolve().parent != expected_parent:
                print(f"Skipping cleanup candidate outside expected parent: {candidate}")
                continue
            if _path_has_active_process_handle(candidate):
                print(f"Skipping cleanup candidate with active process handle: {candidate}")
                continue
            archive_dir = artifact_root / source_name / candidate.name
            duplicate_stitched_project = (
                has_output_stitched_archive
                and source_name == "finn_build_dir"
                and candidate.name.startswith("vivado_stitch_proj_")
                and (candidate / "finn_design_routed.dcp").is_file()
                and (candidate / "ooc_timing.rpt").is_file()
            )
            if keep_artifacts and not duplicate_stitched_project:
                preserved += _preserve_failed_vivado_artifacts(candidate, archive_dir)
            elif duplicate_stitched_project:
                skipped_duplicate_artifacts += 1
            _remove_path(candidate)
            cleaned += 1
        except Exception as exc:
            print(f"WARNING: failed to clean Vivado candidate {candidate}: {exc}")

    if keep_artifacts:
        print(
            "Failed-build Vivado cleanup: removed "
            f"{cleaned} dirs, preserved {preserved} report/DCP artifacts under {artifact_root}."
        )
        if skipped_duplicate_artifacts:
            print(
                "Failed-build Vivado cleanup: skipped artifact preservation for "
                f"{skipped_duplicate_artifacts} duplicate stitched Vivado project dirs."
            )
    else:
        print(f"Failed-build Vivado cleanup: removed {cleaned} dirs.")


def prepare_cppsim(model: ModelWrapper, num_workers: int | None) -> ModelWrapper:
    model = model.transform(SetExecMode("cppsim"), apply_to_subgraphs=True)
    model = model.transform(PrepareCppSim(num_workers), apply_to_subgraphs=True)
    model = model.transform(CompileCppSim(num_workers), apply_to_subgraphs=True)
    model = model.transform(SetExecMode("cppsim"), apply_to_subgraphs=True)
    return model


def generate_reference_io(
    reference_model_path: Path,
    output_dir: Path,
    seed: int,
    shape_model_path: Path | None = None,
    cppsim_prepare: bool = True,
    cppsim_workers: int | None = None,
) -> None:
    shape_model = ModelWrapper(str(shape_model_path or reference_model_path))
    input_name = shape_model.get_first_global_in()
    input_shape = shape_model.get_tensor_shape(input_name)
    if input_shape is None:
        raise RuntimeError(f"Could not infer input shape for {input_name}")
    np.random.seed(seed)
    input_tensor = gen_finn_dt_tensor(DataType["FLOAT32"], input_shape)
    np.save(output_dir / "input.npy", input_tensor)

    reference_model = ModelWrapper(str(reference_model_path))
    if cppsim_prepare:
        reference_model = prepare_cppsim(reference_model, cppsim_workers)
    reference_input_name = reference_model.get_first_global_in()
    reference_output_name = reference_model.get_first_global_out()
    output_dict = execute_onnx(
        reference_model,
        {reference_input_name: input_tensor},
        return_full_exec_context=True,
    )
    np.save(output_dir / "expected_output.npy", output_dict[reference_output_name])
    np.savez(output_dir / "expected_context.npz", **output_dict)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_CHECKPOINT.relative_to(repo_path("."))))
    parser.add_argument(
        "--output-dir",
        default=str((DEFAULT_BUILD_DIR / "v80_mlo").relative_to(repo_path("."))),
    )
    parser.add_argument(
        "--mode", choices=["estimate", "rtl", "dcp", "full-rtlsim", "bitfile"], default="rtl"
    )
    parser.add_argument("--board", default=DEFAULT_BOARD)
    parser.add_argument("--clock-ns", type=float, default=DEFAULT_CLOCK_NS)
    parser.add_argument("--target-fps", type=int, default=DEFAULT_TARGET_FPS)
    parser.add_argument(
        "--folding-target-cycles",
        type=int,
        default=DEFAULT_FOLDING_TARGET_CYCLES,
        help=(
            "Aggressive folding target in cycles/frame. The build converts this "
            "to an equivalent target FPS; set to 0 to use --target-fps directly."
        ),
    )
    parser.add_argument("--mvau-wwidth-max", type=int, default=10000)
    parser.add_argument("--folding-config-file", default=None)
    parser.add_argument(
        "--folding-two-pass-relaxation",
        dest="folding_two_pass_relaxation",
        action="store_true",
    )
    parser.add_argument(
        "--no-folding-two-pass-relaxation",
        dest="folding_two_pass_relaxation",
        action="store_false",
    )
    parser.add_argument(
        "--no-post-transpose-folding",
        dest="post_transpose_folding",
        action="store_false",
    )
    parser.add_argument(
        "--build-csv", default=str(DEFAULT_BUILD_CSV.relative_to(repo_path(".")))
    )
    parser.add_argument("--atol", type=float, default=1e-1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rtlsim-batch-size", type=int, default=1)
    parser.add_argument(
        "--fifosim-n-inferences",
        type=int,
        default=1,
        help=(
            "Number of inferences used by loop-body FIFO sizing simulation. "
            "TinyDeiT's loop body can stall in the stale tail of the second "
            "dummy inference, so the default keeps sizing to one inference."
        ),
    )
    parser.add_argument("--node-by-node", action="store_true")
    parser.add_argument("--stitched-rtlsim", action="store_true")
    parser.add_argument(
        "--stitched-rtlsim-liveness-threshold",
        type=int,
        default=None,
        help=(
            "Override the stitched-IP rtlsim liveness watchdog in cycles. "
            "Useful for MLO graphs whose top-level loop estimate is much larger "
            "than the measured end-to-end RTL interval."
        ),
    )
    parser.add_argument("--stitched-ip-dcp", action="store_true")
    parser.add_argument(
        "--rtlsim-results-file",
        default=None,
        help=(
            "Use an existing FINN rtlsim results.txt file for CSV throughput "
            "accounting without rerunning stitched-IP rtlsim."
        ),
    )
    parser.add_argument("--prepared-model", default=None)
    parser.add_argument("--reference-model", default=None)
    parser.add_argument(
        "--no-reference-cppsim-prepare", dest="reference_cppsim_prepare", action="store_false"
    )
    parser.add_argument("--reference-cppsim-workers", type=int, default=None)
    parser.add_argument("--skip-reference-io", action="store_true")
    parser.add_argument(
        "--cleanup-failed-vivado",
        dest="cleanup_failed_vivado",
        action="store_true",
        help=(
            "After an unsuccessful build, remove new Vivado/IP project dirs created "
            "under FINN_BUILD_DIR and this output dir."
        ),
    )
    parser.add_argument(
        "--no-cleanup-failed-vivado",
        dest="cleanup_failed_vivado",
        action="store_false",
    )
    parser.add_argument(
        "--keep-failed-vivado-artifacts",
        dest="keep_failed_vivado_artifacts",
        action="store_true",
        help=(
            "Keep diagnostic reports/logs plus top-level design DCPs before "
            "deleting failed Vivado project dirs."
        ),
    )
    parser.add_argument(
        "--discard-failed-vivado-artifacts",
        dest="keep_failed_vivado_artifacts",
        action="store_false",
    )
    parser.set_defaults(reference_cppsim_prepare=True)
    parser.set_defaults(folding_two_pass_relaxation=False, post_transpose_folding=True)
    parser.set_defaults(cleanup_failed_vivado=True, keep_failed_vivado_artifacts=True)
    args = parser.parse_args()

    if args.folding_target_cycles and args.folding_target_cycles > 0:
        target_cycles_per_sec = 10**9 / args.clock_ns
        target_fps_from_cycles = int(round(target_cycles_per_sec / args.folding_target_cycles))
        args.target_fps = max(args.target_fps, target_fps_from_cycles)

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    build_started_at = time.time()
    cleanup_snapshot = _failed_cleanup_snapshot(output_dir)

    ret = -1
    model_path: Path | None = None
    cleanup_done = False

    def cleanup_failed_once() -> None:
        nonlocal cleanup_done
        if args.cleanup_failed_vivado and not cleanup_done:
            cleanup_failed_vivado_projects(
                output_dir, cleanup_snapshot, args.keep_failed_vivado_artifacts
            )
            cleanup_done = True

    try:
        if args.prepared_model is None:
            prep_args = argparse.Namespace(
                input=args.input,
                output_dir=str(output_dir / "prepare"),
                board=args.board,
                clock_ns=args.clock_ns,
                target_fps=args.target_fps,
                depth=12,
                skip_streamline=False,
                collapse_pwpolyf=True,
                mlo=True,
                save_intermediate=True,
            )
            model_path = prepare(prep_args)
        else:
            model_path = repo_path(args.prepared_model)

        cfg = build_config(args, output_dir)
        if not args.skip_reference_io and cfg.verify_steps:
            reference_model_path = (
                repo_path(args.reference_model) if args.reference_model else model_path
            )
            generate_reference_io(
                reference_model_path,
                output_dir,
                args.seed,
                model_path,
                args.reference_cppsim_prepare,
                args.reference_cppsim_workers,
            )
        ret = build.build_dataflow_cfg(str(model_path), cfg)
    except BaseException:
        cleanup_failed_once()
        if model_path is not None:
            csv_path = record_build_result(
                args, output_dir, model_path, ret, build_started_at
            )
            print(f"Build CSV: {csv_path}")
        print(f"Build output: {output_dir}")
        raise

    if ret != 0:
        cleanup_failed_once()
        csv_path = record_build_result(args, output_dir, model_path, ret, build_started_at)
        print(f"Build CSV: {csv_path}")
        print(f"Build output: {output_dir}")
        raise SystemExit(ret)

    else:
        vivado_failed = _output_vivado_failed(args, output_dir)
        timing_failed = _output_timing_failed(output_dir)
        if vivado_failed:
            print("Vivado implementation failed; treating build as unsuccessful for cleanup.")
            cleanup_failed_once()
            csv_path = record_build_result(args, output_dir, model_path, 1, build_started_at)
            print(f"Build CSV: {csv_path}")
            print(f"Build output: {output_dir}")
            raise SystemExit(1)
        if timing_failed:
            print("Timing failed; treating build as unsuccessful for Vivado cleanup.")
            cleanup_failed_once()
        csv_path = record_build_result(args, output_dir, model_path, ret, build_started_at)
        print(f"Build CSV: {csv_path}")
        print(f"Build output: {output_dir}")


if __name__ == "__main__":
    main()
