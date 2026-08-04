"""Run resource-aware PPO folding on a prepared TinyDeiT FINN model."""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes

from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.transpose_decomposition import (
    InferInnerOuterShuffles,
    ShuffleDecomposition,
)

DEFAULT_PART = "xcvc1902-vsva2197-2MP-e-S"
DEFAULT_PPO_ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_ppo_modules(ppo_root):
    """Load this branch's PPO implementation in a TinyDeiT FINN environment."""

    source_root = ppo_root / "src/finn"
    dataflow_performance = load_module(
        "finn.analysis.fpgadataflow.dataflow_performance",
        source_root / "analysis/fpgadataflow/dataflow_performance.py",
    )
    res_estimation = load_module(
        "finn.analysis.fpgadataflow.res_estimation",
        source_root / "analysis/fpgadataflow/res_estimation.py",
    )
    set_folding = load_module(
        "tinydeit_ppo_set_folding",
        source_root / "transformation/fpgadataflow/set_folding.py",
    )
    return dataflow_performance, res_estimation, set_folding


def prepare_model(model, fpga_part):
    """Apply the same pre-folding transformations as the original experiment."""

    model = model.transform(ShuffleDecomposition(), apply_to_subgraphs=True)
    model = model.transform(InferInnerOuterShuffles(), apply_to_subgraphs=True)
    model = model.transform(SpecializeLayers(fpga_part), apply_to_subgraphs=True)
    model = model.transform(InferShapes(), apply_to_subgraphs=True)
    return model.transform(InferDataTypes(), apply_to_subgraphs=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run resource-aware PPO folding on a prepared TinyDeiT FINN/MLO ONNX model."
    )
    parser.add_argument("--model", type=Path, required=True, help="Input FINN/MLO ONNX model")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for results")
    parser.add_argument(
        "--ppo-root",
        type=Path,
        default=DEFAULT_PPO_ROOT,
        help="Root of the ppo-tinydeit FINN worktree",
    )
    parser.add_argument("--target-fps", type=float, default=200.0)
    parser.add_argument("--clock-ns", type=float, default=5.0)
    parser.add_argument("--board", default="VCK190")
    parser.add_argument("--fpga-part", default=DEFAULT_PART)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mvau-wwidth-max", type=int, default=36)
    parser.add_argument("--resource-limit", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    ppo_root = args.ppo_root.expanduser().resolve()

    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if args.target_fps <= 0:
        raise ValueError("target-fps must be greater than zero")
    if args.clock_ns <= 0:
        raise ValueError("clock-ns must be greater than zero")

    dataflow_performance, res_estimation, set_folding = load_ppo_modules(ppo_root)
    target_cycles = max(1, int(1.0e9 / (args.clock_ns * args.target_fps)))
    model = prepare_model(ModelWrapper(str(model_path)), args.fpga_part)
    model = model.transform(GiveUniqueNodeNames())
    if len(model.get_nodes_by_op_type("FINNLoop")) != 1:
        raise ValueError(
            "Expected one FINNLoop. Prepare and roll the TinyDeiT model before running PPO."
        )
    optimizer = set_folding.ResourceAwareFoldingPPO(
        model,
        target_cycles_per_frame=target_cycles,
        fpgapart=args.fpga_part,
        board=args.board,
        mvau_wwidth_max=args.mvau_wwidth_max,
        resource_limit=args.resource_limit,
        episodes=args.episodes,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        seed=args.seed,
    )
    if not optimizer.knobs:
        raise ValueError(
            "No folding knobs found. Run with the TinyDeiT FINN environment as documented."
        )

    started = time.monotonic()
    model = optimizer.optimize()
    search_seconds = time.monotonic() - started

    performance = dataflow_performance.folding_performance(model)
    resources_by_node = res_estimation.res_estimation_recursive(model, args.fpga_part)
    resources = set_folding.aggregate_resources(resources_by_node)
    max_cycles = performance["max_cycles"]

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir / "folded.onnx"))
    result = {
        "source_model": str(model_path),
        "ppo_root": str(ppo_root),
        "board": args.board,
        "fpga_part": args.fpga_part,
        "clock_ns": args.clock_ns,
        "target_fps": args.target_fps,
        "target_cycles": target_cycles,
        "estimated_fps": 1.0e9 / (args.clock_ns * max_cycles),
        "max_cycles": max_cycles,
        "max_cycles_node": performance["max_cycles_node_name"],
        "knobs": len(optimizer.knobs),
        "search_seconds": search_seconds,
        "resources": resources,
        "capacity": optimizer.capacity,
        "analytical_target_feasibility": optimizer.target_feasibility,
        "ppo_seed": args.seed,
        "ppo_episodes": args.episodes,
        "ppo_rollout_steps": args.rollout_steps,
        "ppo_update_epochs": args.update_epochs,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
