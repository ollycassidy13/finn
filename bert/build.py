#!/usr/bin/env python3
"""Build and verify the BERT safety FINN core for V80."""

from __future__ import annotations

import argparse
import numpy as np
import os
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper

import finn.builder.build_dataflow as build
from bert.common import (
    CORE_PRESETS,
    DEFAULT_BOARD,
    DEFAULT_BUILD_DIR,
    DEFAULT_CLOCK_NS,
    DEFAULT_FPGA_PART,
    get_preset,
    repo_path,
    write_json,
)
from bert.make_core_model import write_models
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    VerificationStepType,
)
from finn.core.onnx_exec import execute_onnx

BUILD_STEPS_ESTIMATE = [
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_transpose_decomposition",
    "step_generate_estimate_reports",
]

BUILD_STEPS_RTL = BUILD_STEPS_ESTIMATE + [
    "step_hw_codegen",
    "step_hw_ipgen",
    "step_set_fifo_depths",
    "step_create_stitched_ip",
]

BUILD_STEPS_DCP = BUILD_STEPS_RTL + ["step_out_of_context_synthesis"]


def make_reference_io(base_model_path: Path, output_dir: Path, seed: int) -> None:
    model = ModelWrapper(str(base_model_path))
    input_name = model.get_first_global_in()
    output_name = model.get_first_global_out()
    input_shape = model.get_tensor_shape(input_name)
    rng = np.random.default_rng(seed)
    inp = rng.integers(0, 16, size=input_shape, dtype=np.uint8).astype(np.float32)
    np.save(output_dir / "input.npy", inp)
    context = execute_onnx(model, {input_name: inp}, return_full_exec_context=True)
    np.save(output_dir / "expected_output.npy", context[output_name])
    np.savez(output_dir / "expected_context.npz", **context)


def build_config(args: argparse.Namespace, output_dir: Path) -> DataflowBuildConfig:
    preset = get_preset(args.preset)
    if args.mode == "estimate":
        steps = BUILD_STEPS_ESTIMATE
        outputs = [DataflowOutputType.ESTIMATE_REPORTS]
        verify_steps = []
        stitched_dcp = False
    elif args.mode == "rtl":
        steps = BUILD_STEPS_RTL
        outputs = [DataflowOutputType.ESTIMATE_REPORTS, DataflowOutputType.STITCHED_IP]
        verify_steps = [VerificationStepType.STITCHED_IP_RTLSIM] if args.verify else []
        stitched_dcp = False
    else:
        steps = BUILD_STEPS_DCP
        outputs = [
            DataflowOutputType.ESTIMATE_REPORTS,
            DataflowOutputType.STITCHED_IP,
            DataflowOutputType.OOC_SYNTH,
        ]
        verify_steps = [VerificationStepType.STITCHED_IP_RTLSIM] if args.verify else []
        stitched_dcp = True

    return DataflowBuildConfig(
        output_dir=str(output_dir),
        synth_clk_period_ns=args.clock_ns,
        board=args.board,
        fpga_part=args.fpga_part,
        target_fps=args.target_fps or preset.target_fps,
        standalone_thresholds=True,
        infer_shuffle_skip_first=False,
        save_intermediate_models=True,
        verify_steps=verify_steps,
        verify_input_npy=str(output_dir / "input.npy"),
        verify_expected_output_npy=str(output_dir / "expected_output.npy"),
        verify_save_full_context=True,
        verification_atol=args.atol,
        generate_outputs=outputs,
        steps=steps,
        auto_fifo_depths=False,
        rtlsim_batch_size=args.rtlsim_batch_size,
        no_stdout_redirect=True,
        enable_build_pdb_debug=False,
        stitched_ip_gen_dcp=stitched_dcp,
        mvau_wwidth_max=preset.mvau_wwidth_max,
        minimize_bit_width=args.minimize_bit_width,
        mlo=args.mlo,
        start_step=args.start_step,
        stop_step=args.stop_step,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(CORE_PRESETS), default="smoke")
    parser.add_argument("--mode", choices=["estimate", "rtl", "dcp"], default="rtl")
    parser.add_argument(
        "--output-dir",
        default=str((DEFAULT_BUILD_DIR / "v80_smoke").relative_to(repo_path("."))),
    )
    parser.add_argument("--board", default=DEFAULT_BOARD)
    parser.add_argument("--fpga-part", default=DEFAULT_FPGA_PART)
    parser.add_argument("--clock-ns", type=float, default=DEFAULT_CLOCK_NS)
    parser.add_argument("--target-fps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtlsim-batch-size", type=int, default=1)
    parser.add_argument(
        "--xelab-mt",
        default="8",
        help=(
            "Thread count passed to xelab through FINN_XELAB_MT for XSI-based "
            "RTLSIM/FIFO sizing. Use an empty string to keep Vivado's default."
        ),
    )
    parser.add_argument(
        "--liveness-threshold-cycles",
        type=int,
        default=None,
        help=(
            "Minimum stitched-IP RTLSIM liveness threshold. Useful for long "
            "rolled BERT MLO graphs where the first output arrives after many cycles."
        ),
    )
    parser.add_argument(
        "--start-step",
        default=None,
        help="Resume from the intermediate model immediately before this build step.",
    )
    parser.add_argument(
        "--stop-step",
        default=None,
        help="Stop after this build step.",
    )
    parser.add_argument(
        "--minimize-bit-width",
        action="store_true",
        help=(
            "Enable FINN value-driven weight/accumulator bit-width minimization. "
            "Leave disabled for deterministic initialized hardware bring-up models."
        ),
    )
    parser.add_argument(
        "--mlo",
        dest="mlo",
        action="store_true",
        help="Roll repeated encoder blocks into a FINNLoop before building.",
    )
    parser.add_argument("--no-mlo", dest="mlo", action="store_false")
    parser.add_argument(
        "--fixed-mlo-fifos",
        action="store_true",
        help=(
            "Use deterministic fixed FIFO insertion for FINNLoop bodies instead "
            "of simulation-based FIFO sizing. This is intended for long max-util "
            "DCP/timing exploration, not final correctness verification."
        ),
    )
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    parser.add_argument("--skip-reference-io", action="store_true")
    parser.add_argument(
        "--weight-state",
        default=None,
        help=(
            "Optional trained student checkpoint directory/file. Matching dense BERT "
            "weights are quantized to INT8 and imported into the generated FINN core."
        ),
    )
    parser.add_argument(
        "--strict-weights",
        action="store_true",
        help="Fail if any FINN core weight cannot be imported from --weight-state.",
    )
    parser.set_defaults(verify=True, mlo=True)
    args = parser.parse_args()
    if args.xelab_mt:
        os.environ["FINN_XELAB_MT"] = args.xelab_mt
    if args.liveness_threshold_cycles is not None:
        os.environ["LIVENESS_THRESHOLD"] = str(args.liveness_threshold_cycles)
    if args.fixed_mlo_fifos:
        os.environ["FINN_MLO_FIXED_FIFOS"] = "1"
    if args.verify and args.preset == "max-util" and args.mode in {"rtl", "dcp"}:
        print(
            "Warning: max-util stitched-IP RTLSIM is a long XSIM run. "
            "Use --no-verify for DCP/timing exploration."
        )

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "model"
    base_model, specialized_model = write_models(
        get_preset(args.preset),
        model_dir,
        args.fpga_part,
        save_specialized=True,
        save_mlo=args.mlo,
        weight_state=args.weight_state,
        strict_weights=args.strict_weights,
    )
    assert specialized_model is not None

    if not args.skip_reference_io:
        make_reference_io(base_model, output_dir, args.seed)

    cfg = build_config(args, output_dir)
    write_json(output_dir / "build_request.json", vars(args))
    write_json(output_dir / "preset.json", get_preset(args.preset).as_dict())
    ret = build.build_dataflow_cfg(str(specialized_model), cfg)
    if ret != 0:
        raise SystemExit(ret)
    print(f"BERT safety FINN build output: {output_dir}")


if __name__ == "__main__":
    main()
