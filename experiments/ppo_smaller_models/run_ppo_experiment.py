"""Run the resource-aware PPO folding experiments for TFC-W1A1 or CNV-W2A2."""

import argparse
import json
import onnx
from pathlib import Path

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
MODEL_PRESETS = {
    "tfc": {
        "model": REPO_ROOT / "src/finn/qnn-data/build_dataflow/model.onnx",
        "specialize_config": REPO_ROOT
        / "src/finn/qnn-data/build_dataflow/specialize_layers_config.json",
        "board": "Pynq-Z1",
        "target_fps": 100_000,
        "clock_ns": 10.0,
        "rtlsim_batch_size": 100,
    },
    "cnv": {
        "model": EXPERIMENT_ROOT / "models/cnv-w2a2.onnx",
        "specialize_config": REPO_ROOT
        / "src/finn/qnn-data/test_ext_weights/specialize_layers_config_cnv.json",
        "board": "ZCU102",
        "target_fps": 500,
        "clock_ns": 10.0,
        "rtlsim_batch_size": 2,
    },
}


def load_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def check_onnx_model(path):
    model = onnx.load(str(path))
    imported_domains = {opset.domain for opset in model.opset_import}
    used_custom_domains = {node.domain for node in model.graph.node if node.domain}
    for domain in sorted(used_custom_domains - imported_domains):
        opset = model.opset_import.add()
        opset.domain = domain
        opset.version = 1
    onnx.checker.check_model(model)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=MODEL_PRESETS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="Override the preset ONNX model")
    parser.add_argument("--specialize-config", type=Path)
    parser.add_argument("--board")
    parser.add_argument("--target-fps", type=int)
    parser.add_argument("--clock-ns", type=float)
    parser.add_argument("--rtlsim-batch-size", type=int)
    parser.add_argument("--mvau-wwidth-max", type=int, default=10_000)
    parser.add_argument("--accuracy-tolerance-pct", type=float, default=5.0)
    parser.add_argument(
        "--implement",
        action="store_true",
        help="Also run IP generation, stitched RTLSIM, and out-of-context synthesis",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def resolve_setting(args, preset, name):
    value = getattr(args, name)
    return preset[name] if value is None else value


def error_pct(value, reference):
    if value is None or reference is None or reference == 0:
        return None
    return 100.0 * (value - reference) / reference


def summarize_result(args, return_code, model, board, target_fps, clock_ns):
    report_dir = args.output_dir / "report"
    performance = load_json(report_dir / "estimate_network_performance.json")
    resource_report = load_json(report_dir / "estimate_layer_resources.json")
    rtlsim = load_json(report_dir / "rtlsim_performance.json")
    ooc = load_json(report_dir / "ooc_synth_and_timing.json")

    estimated_fps = performance.get("estimated_throughput_fps") if performance is not None else None
    interval_cycles = None
    rtlsim_fps = None
    rtlsim_complete = False
    if rtlsim is not None:
        interval_cycles = int(rtlsim.get("interval_cycles", 0))
        timeout = int(rtlsim.get("TIMEOUT", 1))
        unfinished_inputs = int(rtlsim.get("UNFINISHED_INS", 1))
        unfinished_outputs = int(rtlsim.get("UNFINISHED_OUTS", 1))
        run_complete = timeout == 0 and unfinished_inputs == 0 and unfinished_outputs == 0
        requested_frames = int(rtlsim.get("N", 0))
        completed_frames = int(
            rtlsim.get("completed_output_frames", requested_frames if run_complete else 0)
        )
        stable_fps = rtlsim.get("stable_throughput[images/s]")
        rtlsim_complete = (
            completed_frames >= 2
            and interval_cycles > 0
            and run_complete
            and rtlsim.get("stable_throughput_valid") is True
            and stable_fps is not None
        )
        if rtlsim_complete:
            rtlsim_fps = float(stable_fps)

    ooc_fmax_fps = ooc.get("estimated_throughput_fps") if ooc is not None else None
    ooc_timing_met = ooc is not None and float(ooc.get("WNS", float("-inf"))) >= 0.0
    estimate_vs_rtlsim_error = error_pct(estimated_fps, rtlsim_fps)
    validation_passed = (
        return_code == 0 and estimated_fps is not None and estimated_fps >= target_fps
    )
    if args.implement:
        validation_passed = (
            validation_passed
            and rtlsim_complete
            and ooc_timing_met
            and rtlsim_fps >= target_fps
            and ooc_fmax_fps is not None
            and ooc_fmax_fps >= target_fps
            and abs(estimate_vs_rtlsim_error) <= args.accuracy_tolerance_pct
        )

    result = {
        "return_code": return_code,
        "experiment": args.experiment,
        "source_model": str(model),
        "board": board,
        "clock_ns": clock_ns,
        "target_fps": target_fps,
        "accuracy_tolerance_pct": args.accuracy_tolerance_pct,
        "estimated_fps": estimated_fps,
        "estimated_target_error_pct": error_pct(estimated_fps, target_fps),
        "estimated_max_cycles": performance.get("max_cycles") if performance is not None else None,
        "estimated_resources": (
            resource_report.get("total") if resource_report is not None else None
        ),
        "rtlsim_interval_cycles": interval_cycles,
        "rtlsim_fps": rtlsim_fps,
        "rtlsim_complete": rtlsim_complete,
        "rtlsim_target_error_pct": error_pct(rtlsim_fps, target_fps),
        "estimate_vs_rtlsim_error_pct": estimate_vs_rtlsim_error,
        "rtlsim_report": rtlsim,
        "ooc_timing_met": ooc_timing_met,
        "ooc_fmax_fps": ooc_fmax_fps,
        "ooc_target_error_pct": error_pct(ooc_fmax_fps, target_fps),
        "estimate_vs_ooc_fmax_error_pct": error_pct(estimated_fps, ooc_fmax_fps),
        "ooc_report": ooc,
        "validation_passed": validation_passed,
    }
    (args.output_dir / "experiment_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return validation_passed


def main():
    args = parse_args()
    preset = MODEL_PRESETS[args.experiment]
    model = (args.model or preset["model"]).expanduser().resolve()
    specialize_config = (
        (args.specialize_config or preset["specialize_config"]).expanduser().resolve()
    )
    output_dir = args.output_dir.expanduser().resolve()
    board = resolve_setting(args, preset, "board")
    target_fps = resolve_setting(args, preset, "target_fps")
    clock_ns = resolve_setting(args, preset, "clock_ns")
    rtlsim_batch_size = resolve_setting(args, preset, "rtlsim_batch_size")

    if not model.is_file():
        raise FileNotFoundError(model)
    if not specialize_config.is_file():
        raise FileNotFoundError(specialize_config)
    check_onnx_model(model)
    if target_fps <= 0:
        raise ValueError("target-fps must be greater than zero")
    if clock_ns <= 0:
        raise ValueError("clock-ns must be greater than zero")
    if rtlsim_batch_size < 2:
        raise ValueError("rtlsim-batch-size must be at least two")
    if args.mvau_wwidth_max <= 0:
        raise ValueError("mvau-wwidth-max must be greater than zero")
    if args.accuracy_tolerance_pct < 0:
        raise ValueError("accuracy-tolerance-pct must not be negative")

    outputs = [build_cfg.DataflowOutputType.ESTIMATE_REPORTS]
    steps = build_cfg.estimate_only_dataflow_steps
    if args.implement:
        outputs.extend(
            [
                build_cfg.DataflowOutputType.STITCHED_IP,
                build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
                build_cfg.DataflowOutputType.OOC_SYNTH,
            ]
        )
        steps = None

    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir
    cfg = build_cfg.DataflowBuildConfig(
        output_dir=str(output_dir),
        synth_clk_period_ns=clock_ns,
        board=board,
        target_fps=target_fps,
        mvau_wwidth_max=args.mvau_wwidth_max,
        specialize_layers_config_file=str(specialize_config),
        steps=steps,
        generate_outputs=outputs,
        auto_fifo_depths=args.implement,
        fifosim_n_inferences=2,
        rtlsim_batch_size=rtlsim_batch_size,
        save_intermediate_models=True,
        enable_build_pdb_debug=False,
        verbose=args.verbose,
    )
    return_code = build.build_dataflow_cfg(str(model), cfg)
    validation_passed = summarize_result(args, return_code, model, board, target_fps, clock_ns)
    raise SystemExit(0 if validation_passed else 1)


if __name__ == "__main__":
    main()
