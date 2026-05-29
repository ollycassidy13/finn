#!/usr/bin/env python3
"""Rerun TinyDeiT stitched IP generation from a saved FIFO-sizing checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import finn.builder.build_dataflow as build
from finn.builder.build_dataflow_config import DataflowBuildConfig, DataflowOutputType
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.util.basic import getHWCustomOp
from qonnx.core.modelwrapper import ModelWrapper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinydeit.build import BUILD_STEPS_RTL


def force_loop_codegen(output_dir: Path, cfg: DataflowBuildConfig) -> None:
    """Regenerate top FINNLoop code in the step_hw_codegen checkpoint.

    Checkpointed reruns normally reuse code_gen_dir_ipgen/ipgen_path. This helper
    clears only the top FINNLoop generated-code attributes, reruns PrepareIP on
    the top graph, and leaves loop-body IP/code untouched.
    """
    checkpoint = output_dir / "intermediate_models" / "step_hw_codegen.onnx"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Cannot force loop codegen: {checkpoint} not found")

    model = ModelWrapper(str(checkpoint))
    changed = False
    for loop_node in model.get_nodes_by_op_type("FINNLoop"):
        loop_inst = getHWCustomOp(loop_node, model)
        loop_inst.set_nodeattr("code_gen_dir_ipgen", "")
        loop_inst.set_nodeattr("ipgen_path", "")
        changed = True
    if not changed:
        raise RuntimeError(f"No FINNLoop nodes found in {checkpoint}")

    model = model.transform(PrepareIP(cfg._resolve_fpga_part(), cfg._resolve_hls_clk_period()))
    model.save(str(checkpoint))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clock-ns", type=float, default=3.3333333333333335)
    parser.add_argument("--board", default="V80")
    parser.add_argument("--folding-config-file", default=None)
    parser.add_argument("--start-step", default="step_create_stitched_ip")
    parser.add_argument("--stop-step", default="step_create_stitched_ip")
    parser.add_argument("--dcp", action="store_true")
    parser.add_argument(
        "--force-loop-codegen",
        action="store_true",
        help="Regenerate top FINNLoop code in step_hw_codegen before continuing.",
    )
    args = parser.parse_args()

    cfg = DataflowBuildConfig(
        output_dir=args.output_dir,
        synth_clk_period_ns=args.clock_ns,
        board=args.board,
        generate_outputs=[DataflowOutputType.STITCHED_IP],
        steps=BUILD_STEPS_RTL,
        start_step=args.start_step,
        stop_step=args.stop_step,
        folding_config_file=args.folding_config_file,
        save_intermediate_models=True,
        no_stdout_redirect=True,
        enable_build_pdb_debug=False,
        mlo=False,
        auto_fifo_depths=False,
        stitched_ip_gen_dcp=args.dcp,
    )
    if args.force_loop_codegen:
        force_loop_codegen(Path(args.output_dir), cfg)
    ret = build.build_dataflow_cfg("unused.onnx", cfg)
    print(f"RET {ret}")
    raise SystemExit(ret)


if __name__ == "__main__":
    main()
