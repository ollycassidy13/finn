#!/usr/bin/env python3
"""Build the static ImageNet SigLIP dataflow graph after HW conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from full_siglip.so400m_objective_guard import (
    evaluate_so400m_dcp_preflight,
    guard_so400m_w6a8_dcp,
    is_so400m_w6a8_objective_path,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BOARD = "VCK190"
DEFAULT_CLOCK_NS = 1000.0 / 300.0
DEFAULT_TARGET_FPS = 1000
SIGLIP_DEPTH = 12
SIGLIP2_86M_TARGET_MS = 50.0


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def write_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def cycles_per_frame(clock_ns: float, target_fps: Optional[float]) -> Optional[int]:
    if target_fps is None:
        return None
    return int((10**9 / clock_ns) / target_fps)


def siglip2_86m_overlapped_scheduler_spec_from_cycles(
    *,
    cycle_dict: dict[str, int],
    loop_summaries: list[dict[str, Any]],
    clock_ns: float,
    depth: int,
    weight_bits: int = 6,
    act_bits: int = 8,
    target_ms: float = SIGLIP2_86M_TARGET_MS,
) -> dict[str, Any]:
    """Build the SigLIP2 86M stream-feedback scheduler spec from folded cycles."""

    if len(loop_summaries) != 1:
        raise RuntimeError(f"Expected one FINNLoop summary, found {len(loop_summaries)}")
    summary = loop_summaries[0]
    loop_name = str(summary["loop"])
    if loop_name not in cycle_dict:
        raise RuntimeError(f"Missing {loop_name} in top-level cycle estimates")
    iteration = int(summary["iteration"])
    body_ii_cycles = int(summary["body_max_cycles"])
    loop_overhead = 40
    non_loop_cycles = sum(int(value) for value in cycle_dict.values()) - int(
        cycle_dict[loop_name]
    )
    overlapped_loop_cycles = (body_ii_cycles + loop_overhead) * iteration
    total_cycles = non_loop_cycles + overlapped_loop_cycles
    cycle_budget = int((target_ms * 1_000_000.0) / clock_ns)
    checks = {
        "single_finnloop": len(loop_summaries) == 1,
        "siglip2_86m_depth_12": depth == SIGLIP_DEPTH,
        "loop_iterations_match_depth": iteration == depth,
        "body_ii_positive": body_ii_cycles > 0,
        "non_loop_cycles_nonnegative": non_loop_cycles >= 0,
        "latency_model_under_50ms": total_cycles < cycle_budget,
    }
    checks["spec_valid_for_objective"] = all(
        checks[name]
        for name in (
            "single_finnloop",
            "siglip2_86m_depth_12",
            "loop_iterations_match_depth",
            "body_ii_positive",
            "non_loop_cycles_nonnegative",
        )
    )
    return {
        "artifact_type": "overlapped_scheduler_spec",
        "spec_version": 1,
        "implementation_status": "spec_only",
        "implementation_artifact": None,
        "exact_scope": {
            "model": "google/siglip2-base-patch16-224",
            "weight_bits": weight_bits,
            "activation_bits": act_bits,
            "vision_depth": depth,
            "image_size": 224,
            "patch_grid": [14, 14],
            "image_tokens": 196,
        },
        "schedule_model": {
            "mode": "overlapped_loop_body_throughput",
            "loop_iterations": iteration,
            "loop_overhead_per_iter": loop_overhead,
            "body_initiation_interval_cycles": body_ii_cycles,
            "body_cycle_budget": int(summary.get("body_target_cycles", body_ii_cycles)),
            "non_loop_cycles": non_loop_cycles,
            "sequential_loop_cycles": int(cycle_dict[loop_name]),
            "overlapped_loop_cycles": overlapped_loop_cycles,
            "total_cycles_with_non_loop": total_cycles,
            "cycle_budget": cycle_budget,
            "latency_ms": (total_cycles * clock_ns) / 1_000_000.0,
            "max_stage": {
                "name": summary.get("body_max_cycles_node"),
                "cycles": body_ii_cycles,
            },
        },
        "checks": checks,
    }


def so400m_preflight_payload(input_model: Path, output_dir: Path) -> dict:
    is_objective = is_so400m_w6a8_objective_path(input_model, output_dir)
    if not is_objective:
        return {
            "input_model": str(input_model),
            "output_dir": str(output_dir),
            "is_so400m_w6a8_objective": False,
            "normal_dcp_allowed": True,
            "reason": "not the exact SO400M W6A8 objective path",
            "blockers": [],
        }

    preflight = evaluate_so400m_dcp_preflight()
    return {
        "input_model": str(input_model),
        "output_dir": str(output_dir),
        "is_so400m_w6a8_objective": True,
        **preflight,
    }


def _get_attr(node, name, default=None):
    from onnx import helper as oh

    for attr in node.attribute:
        if attr.name == name:
            return oh.get_attribute_value(attr)
    return default


def _static_tensor_value(model, tensor_name, seen=None):
    import numpy as np

    if seen is None:
        seen = set()
    if tensor_name in seen:
        return None
    seen.add(tensor_name)

    initializer = model.get_initializer(tensor_name)
    if initializer is not None:
        return initializer

    producer = model.find_producer(tensor_name)
    if producer is None:
        return None

    if producer.op_type == "Constant":
        value = _get_attr(producer, "value")
        if value is not None:
            from onnx import numpy_helper

            return numpy_helper.to_array(value)
        value_float = _get_attr(producer, "value_float")
        if value_float is not None:
            return np.asarray(value_float, dtype=np.float32)
        value_floats = _get_attr(producer, "value_floats")
        if value_floats is not None:
            return np.asarray(value_floats, dtype=np.float32)
        value_int = _get_attr(producer, "value_int")
        if value_int is not None:
            return np.asarray(value_int, dtype=np.int64)
        value_ints = _get_attr(producer, "value_ints")
        if value_ints is not None:
            return np.asarray(value_ints, dtype=np.int64)
        return None

    if producer.op_type == "Identity":
        return _static_tensor_value(model, producer.input[0], seen)

    if producer.op_type == "Concat":
        values = [_static_tensor_value(model, inp, seen.copy()) for inp in producer.input]
        if any(value is None for value in values):
            return None
        axis = int(_get_attr(producer, "axis", 0))
        return np.concatenate(values, axis=axis)

    if producer.op_type == "Reshape":
        data = _static_tensor_value(model, producer.input[0], seen.copy())
        shape = _static_tensor_value(model, producer.input[1], seen.copy())
        if data is None or shape is None:
            return None
        new_shape = [int(dim) for dim in np.asarray(shape).flatten()]
        allowzero = int(_get_attr(producer, "allowzero", 0))
        if not allowzero:
            input_shape = list(data.shape)
            new_shape = [
                input_shape[index] if dim == 0 else dim for index, dim in enumerate(new_shape)
            ]
        return np.reshape(data, new_shape)

    if producer.op_type == "Transpose":
        data = _static_tensor_value(model, producer.input[0], seen.copy())
        if data is None:
            return None
        return np.transpose(data, axes=_get_attr(producer, "perm"))

    if producer.op_type == "Squeeze":
        data = _static_tensor_value(model, producer.input[0], seen.copy())
        if data is None:
            return None
        axes = _get_attr(producer, "axes")
        if axes is None and len(producer.input) > 1:
            axes = _static_tensor_value(model, producer.input[1], seen.copy())
        if axes is None:
            return np.squeeze(data)
        return np.squeeze(data, axis=tuple(int(axis) for axis in np.asarray(axes).flatten()))

    if producer.op_type == "Unsqueeze":
        data = _static_tensor_value(model, producer.input[0], seen.copy())
        if data is None:
            return None
        axes = _get_attr(producer, "axes")
        if axes is None and len(producer.input) > 1:
            axes = _static_tensor_value(model, producer.input[1], seen.copy())
        if axes is None:
            return None
        for axis in sorted(int(axis) for axis in np.asarray(axes).flatten()):
            data = np.expand_dims(data, axis)
        return data

    return None


def _fresh_initializer_name(model, base, suffix):
    existing = {initializer.name for initializer in model.graph.initializer}
    existing.update(value_info.name for value_info in model.graph.input)
    existing.update(value_info.name for value_info in model.graph.output)
    existing.update(value_info.name for value_info in model.graph.value_info)
    for node in model.graph.node:
        existing.update(node.input)
        existing.update(node.output)

    candidate = f"{base}_{suffix}"
    index = 0
    while candidate in existing:
        index += 1
        candidate = f"{base}_{suffix}_{index}"
    return candidate


def fold_mlo_static_loop_params(model):
    """Fold static parent-side FINNLoop parameter stacks into initializers."""

    from finn.util.basic import getHWCustomOp

    for loop_node in model.get_nodes_by_op_type("FINNLoop"):
        loop_inst = getHWCustomOp(loop_node, model)
        loop_body = loop_inst.get_nodeattr("body")
        activation_inputs = max(1, len(loop_node.output))
        for input_index, param_name in enumerate(
            loop_node.input[activation_inputs:], start=activation_inputs
        ):
            if model.get_initializer(param_name) is not None:
                continue

            params = _static_tensor_value(model, param_name)
            if params is None:
                continue

            folded_name = _fresh_initializer_name(model, param_name, "mlo_static")
            dtype = model.get_tensor_datatype(param_name)
            model.set_initializer(folded_name, params)
            model.set_tensor_shape(folded_name, list(params.shape))
            if dtype is not None:
                model.set_tensor_datatype(folded_name, dtype)
                loop_body.set_tensor_datatype(loop_body.graph.input[input_index].name, dtype)
            loop_node.input[input_index] = folded_name
        loop_inst.set_nodeattr("body", loop_body.graph)
    return model


def round_and_clip_mlo_threshold_params(model):
    """Repair MLO threshold parameter dtypes after loop rolling.

    After LoopRolling, threshold tables are FINNLoop inputs instead of
    loop-body initializers, so FINN's normal RoundAndClipThresholds pass cannot
    see them. RTL threshold codegen still requires integer threshold datatypes
    when the activation stream is integer.
    """

    import numpy as np
    from finn.util.basic import getHWCustomOp
    from onnx import helper as oh
    from qonnx.core.datatype import DataType

    def get_attr(node, name, default=None):
        for attr in node.attribute:
            if attr.name == name:
                return oh.get_attribute_value(attr)
        return default

    def static_tensor_value(tensor_name, seen=None):
        if seen is None:
            seen = set()
        if tensor_name in seen:
            return None
        seen.add(tensor_name)

        initializer = model.get_initializer(tensor_name)
        if initializer is not None:
            return initializer

        producer = model.find_producer(tensor_name)
        if producer is None:
            return None

        if producer.op_type == "Constant":
            value = get_attr(producer, "value")
            if value is not None:
                from onnx import numpy_helper

                return numpy_helper.to_array(value)
            value_float = get_attr(producer, "value_float")
            if value_float is not None:
                return np.asarray(value_float, dtype=np.float32)
            value_floats = get_attr(producer, "value_floats")
            if value_floats is not None:
                return np.asarray(value_floats, dtype=np.float32)
            value_int = get_attr(producer, "value_int")
            if value_int is not None:
                return np.asarray(value_int, dtype=np.int64)
            value_ints = get_attr(producer, "value_ints")
            if value_ints is not None:
                return np.asarray(value_ints, dtype=np.int64)
            return None

        if producer.op_type == "Identity":
            return static_tensor_value(producer.input[0], seen)

        if producer.op_type == "Concat":
            values = [static_tensor_value(inp, seen.copy()) for inp in producer.input]
            if any(value is None for value in values):
                return None
            axis = int(get_attr(producer, "axis", 0))
            return np.concatenate(values, axis=axis)

        if producer.op_type == "Reshape":
            data = static_tensor_value(producer.input[0], seen.copy())
            shape = static_tensor_value(producer.input[1], seen.copy())
            if data is None or shape is None:
                return None
            new_shape = [int(dim) for dim in np.asarray(shape).flatten()]
            allowzero = int(get_attr(producer, "allowzero", 0))
            if not allowzero:
                input_shape = list(data.shape)
                new_shape = [
                    input_shape[index] if dim == 0 else dim for index, dim in enumerate(new_shape)
                ]
            return np.reshape(data, new_shape)

        if producer.op_type == "Transpose":
            data = static_tensor_value(producer.input[0], seen.copy())
            if data is None:
                return None
            return np.transpose(data, axes=get_attr(producer, "perm"))

        if producer.op_type == "Squeeze":
            data = static_tensor_value(producer.input[0], seen.copy())
            if data is None:
                return None
            axes = get_attr(producer, "axes")
            if axes is None and len(producer.input) > 1:
                axes = static_tensor_value(producer.input[1], seen.copy())
            if axes is None:
                return np.squeeze(data)
            return np.squeeze(data, axis=tuple(int(axis) for axis in np.asarray(axes).flatten()))

        if producer.op_type == "Unsqueeze":
            data = static_tensor_value(producer.input[0], seen.copy())
            if data is None:
                return None
            axes = get_attr(producer, "axes")
            if axes is None and len(producer.input) > 1:
                axes = static_tensor_value(producer.input[1], seen.copy())
            if axes is None:
                return None
            for axis in sorted(int(axis) for axis in np.asarray(axes).flatten()):
                data = np.expand_dims(data, axis)
            return data

        return None

    def fresh_initializer_name(base):
        existing = {initializer.name for initializer in model.graph.initializer}
        existing.update(value_info.name for value_info in model.graph.input)
        existing.update(value_info.name for value_info in model.graph.output)
        existing.update(value_info.name for value_info in model.graph.value_info)
        for node in model.graph.node:
            existing.update(node.input)
            existing.update(node.output)

        candidate = f"{base}_mlo_rounded"
        index = 0
        while candidate in existing:
            index += 1
            candidate = f"{base}_mlo_rounded_{index}"
        return candidate

    for loop_node in model.get_nodes_by_op_type("FINNLoop"):
        loop_inst = getHWCustomOp(loop_node, model)
        loop_body = loop_inst.get_nodeattr("body")
        activation_inputs = max(1, len(loop_node.output))
        for input_index, param_name in enumerate(
            loop_node.input[activation_inputs:], start=activation_inputs
        ):
            params = model.get_initializer(param_name)
            loop_tensor = loop_body.graph.input[input_index].name
            param_node = loop_body.find_consumer(loop_tensor)
            if param_node is None or not param_node.op_type.startswith("Thresholding"):
                continue
            param_inst = getHWCustomOp(param_node, loop_body)
            input_dtype = DataType[param_inst.get_nodeattr("inputDataType")]
            if not input_dtype.is_integer():
                continue

            if params is None:
                params = static_tensor_value(param_name)
                if params is None:
                    continue
                param_name = fresh_initializer_name(param_name)
                loop_node.input[input_index] = param_name

            rounded = np.clip(
                np.ceil(params),
                input_dtype.min(),
                input_dtype.max() + 1,
            ).astype(np.float32)
            model.set_initializer(param_name, rounded)
            model.set_tensor_shape(param_name, list(rounded.shape))

            max_val = input_dtype.max() + 1
            if input_dtype.signed():
                threshold_dtype = DataType.get_smallest_possible(-max_val - 1)
            else:
                threshold_dtype = DataType.get_smallest_possible(max_val)
            model.set_tensor_datatype(param_name, threshold_dtype)
            loop_body.set_tensor_datatype(loop_tensor, threshold_dtype)
            param_inst.set_nodeattr("weightDataType", threshold_dtype.name)
        loop_inst.set_nodeattr("body", loop_body.graph)
    model = fold_mlo_static_loop_params(model)
    return fold_mlo_body_constant_nodes(model)


def remove_unused_mlo_static_scaffolding(model, output_dir: Path):
    """Drop stale parent-side ONNX scaffolding after MLO params are folded."""

    from collections import Counter
    from qonnx.transformation.general import RemoveUnusedTensors
    from qonnx.transformation.remove import RemoveUnusedNodes

    tracked_ops = {"Concat", "Constant", "Reshape"}
    before_ops = Counter(node.op_type for node in model.graph.node)
    before_nodes = len(model.graph.node)
    before_initializers = len(model.graph.initializer)
    model = model.transform(RemoveUnusedNodes(), cleanup=False)
    model = model.transform(RemoveUnusedTensors(), cleanup=False)
    after_ops = Counter(node.op_type for node in model.graph.node)
    removed = before_nodes - len(model.graph.node)
    write_json(
        output_dir / "mlo_static_scaffold_cleanup.json",
        {
            "removed_nodes": removed,
            "removed_initializers": before_initializers - len(model.graph.initializer),
            "tracked_ops": {
                op_type: {
                    "before": before_ops.get(op_type, 0),
                    "after": after_ops.get(op_type, 0),
                    "removed": before_ops.get(op_type, 0) - after_ops.get(op_type, 0),
                }
                for op_type in sorted(tracked_ops)
            },
            "remaining_nodes": len(model.graph.node),
        },
    )
    return model


def fold_mlo_body_constant_nodes(model):
    """Fold ONNX Constant nodes inside FINNLoop bodies into initializers.

    FINN's loop-body FIFO sizing expects loop bodies to contain only HLS/RTL
    dataflow nodes. Loop rolling can leave small scalar Constant nodes inside a
    body when constants are intentionally not folded at extraction time; these
    are still compile-time parameters and can be represented as initializers.
    """

    from onnx import helper as oh
    from onnx import numpy_helper
    from finn.util.basic import getHWCustomOp

    def get_attr(node, name, default=None):
        for attr in node.attribute:
            if attr.name == name:
                return oh.get_attribute_value(attr)
        return default

    for loop_node in model.get_nodes_by_op_type("FINNLoop"):
        loop_inst = getHWCustomOp(loop_node, model)
        loop_body = loop_inst.get_nodeattr("body")
        for node in list(loop_body.graph.node):
            if node.op_type != "Constant" or len(node.output) != 1:
                continue
            value = get_attr(node, "value")
            if value is None:
                continue
            tensor_name = node.output[0]
            tensor_value = numpy_helper.to_array(value)
            try:
                tensor_dtype = loop_body.get_tensor_datatype(tensor_name)
            except Exception:
                tensor_dtype = None

            loop_body.set_initializer(tensor_name, tensor_value)
            loop_body.set_tensor_shape(tensor_name, list(tensor_value.shape))
            if tensor_dtype is not None:
                loop_body.set_tensor_datatype(tensor_name, tensor_dtype)
            loop_body.graph.node.remove(node)

        loop_inst.set_nodeattr("body", loop_body.graph)
    return model


def set_large_outer_shuffle_uram(model, output_dir: Path, min_bits: int = 1 << 20):
    """Use URAM for large HLS outer-shuffle reorder buffers in the static graph."""

    from finn.util.basic import getHWCustomOp

    assignments = []

    def visit_graph(graph_model, scope: str = ""):
        for node in graph_model.graph.node:
            if node.op_type == "OuterShuffle_hls":
                inst = getHWCustomOp(node)
                buf_bits = int(inst.get_input_gen_buf_bits())
                if buf_bits >= min_bits:
                    inst.set_nodeattr("ram_style", "ultra")
                assignments.append(
                    {
                        "name": f"{scope}{node.name}",
                        "buf_bits": buf_bits,
                        "buf_size": int(inst.get_input_gen_buf_size()),
                        "stream_width": int(inst.get_instream_width()),
                        "ram_style": inst.get_nodeattr("ram_style"),
                    }
                )
            elif node.op_type == "FINNLoop":
                inst = getHWCustomOp(node)
                loop_body = inst.get_nodeattr("body")
                visit_graph(loop_body, scope=f"{scope}{node.name}/")
                inst.set_nodeattr("body", loop_body.graph)

    visit_graph(model)
    write_json(output_dir / "outer_shuffle_ram_style.json", assignments)
    return model


def fold_static_siglip_loop_bodies(
    model,
    *,
    target_cycles_per_frame: int | None,
    mvau_wwidth_max: int,
    output_dir: Path,
):
    """Apply target-FPS folding inside FINNLoop bodies with per-iteration targets."""

    if target_cycles_per_frame is None:
        return model

    from finn.transformation.fpgadataflow.set_folding import SetFolding
    from finn.util.basic import getHWCustomOp
    from qonnx.transformation.general import GiveUniqueNodeNames
    from qonnx.transformation.infer_datatypes import InferDataTypes
    from qonnx.transformation.infer_shapes import InferShapes

    for loop_node in model.get_nodes_by_op_type("FINNLoop"):
        loop_inst = getHWCustomOp(loop_node, model)
        iteration = max(1, int(loop_inst.get_nodeattr("iteration")))
        body_target_cycles = max(1, int(target_cycles_per_frame // iteration) - 40)
        loop_body = loop_inst.get_nodeattr("body")

        loop_body = loop_body.transform(
            SetFolding(
                body_target_cycles,
                mvau_wwidth_max=mvau_wwidth_max,
                two_pass_relaxation=True,
            )
        )
        loop_body = loop_body.transform(InferShapes())
        loop_body = loop_body.transform(InferDataTypes())
        loop_body = loop_body.transform(GiveUniqueNodeNames(prefix=loop_node.name + "_"))
        loop_inst.set_nodeattr("body", loop_body.graph)

    return refresh_static_siglip_loop_body_summary(
        model,
        target_cycles_per_frame=target_cycles_per_frame,
        output_dir=output_dir,
    )


def refresh_static_siglip_loop_body_summary(
    model,
    *,
    target_cycles_per_frame: int | None,
    output_dir: Path,
):
    """Write loop-body folding metadata from the current FINNLoop bodies."""

    if target_cycles_per_frame is None:
        return model

    from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
    from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
    from finn.util.basic import getHWCustomOp
    from finn.util.config import extract_model_config_to_json

    summaries = []
    for loop_node in model.get_nodes_by_op_type("FINNLoop"):
        loop_inst = getHWCustomOp(loop_node, model)
        iteration = max(1, int(loop_inst.get_nodeattr("iteration")))
        body_target_cycles = max(1, int(target_cycles_per_frame // iteration) - 40)
        loop_body = loop_inst.get_nodeattr("body")
        loop_body = loop_body.transform(AnnotateCycles())
        perf = loop_body.analysis(dataflow_performance)
        loop_inst.set_nodeattr("body", loop_body.graph)

        summaries.append(
            {
                "loop": loop_node.name,
                "iteration": iteration,
                "frame_target_cycles": int(target_cycles_per_frame),
                "body_target_cycles": body_target_cycles,
                "body_max_cycles": int(perf["max_cycles"]),
                "body_max_cycles_node": perf["max_cycles_node_name"],
                "body_critical_path_cycles": int(perf["critical_path_cycles"]),
                "estimated_loop_cycles": int(
                    (int(perf["critical_path_cycles"]) + 40) * iteration
                ),
            }
        )

    if summaries:
        model = model.transform(AnnotateCycles())
        write_json(output_dir / "static_loop_body_folding_summary.json", summaries)
        hw_attrs = [
            "PE",
            "SIMD",
            "parallel_window",
            "ram_style",
            "resType",
            "mem_mode",
            "runtime_writeable_weights",
            "pumpedCompute",
            "depth_trigger_uram",
            "depth_trigger_bram",
        ]
        extract_model_config_to_json(
            model,
            str(output_dir / "auto_folding_config.json"),
            hw_attrs,
        )
    return model


def set_static_mvau_external_mem(model, output_dir: Path):
    """Use external/block memory for static-weight MVAUs and dynamic mode otherwise."""

    from finn.util.basic import getHWCustomOp
    from finn.util.config import extract_model_config_to_json

    assignments = []

    def visit_graph(
        graph_model,
        scope: str = "",
        static_graph_inputs: set[str] | None = None,
    ):
        static_graph_inputs = static_graph_inputs or set()
        for node in graph_model.graph.node:
            if node.op_type == "FINNLoop":
                loop_inst = getHWCustomOp(node, graph_model)
                loop_body = loop_inst.get_nodeattr("body")
                activation_inputs = max(1, len(node.output))
                dynamic_loop_inputs = set(
                    int(x) for x in loop_inst.get_nodeattr("dynamic_loop_inputs")
                )
                static_body_inputs = set()
                for input_index in range(activation_inputs, len(node.input)):
                    if input_index in dynamic_loop_inputs:
                        continue
                    parent_input = node.input[input_index]
                    if graph_model.get_initializer(parent_input) is not None:
                        static_body_inputs.add(loop_body.graph.input[input_index].name)
                visit_graph(
                    loop_body,
                    scope=f"{scope}{node.name}/",
                    static_graph_inputs=static_body_inputs,
                )
                loop_inst.set_nodeattr("body", loop_body.graph)
                continue

            if not node.op_type.startswith("MVAU") or len(node.input) < 2:
                continue

            inst = getHWCustomOp(node, graph_model)
            before = {
                "mem_mode": inst.get_nodeattr("mem_mode"),
                "ram_style": inst.get_nodeattr("ram_style"),
            }
            rhs = node.input[1]
            has_static_weights = (
                graph_model.get_initializer(rhs) is not None or rhs in static_graph_inputs
            )
            if has_static_weights:
                if inst.get_nodeattr("mem_mode") != "external_mem":
                    inst.set_nodeattr("mem_mode", "external_mem")
                    inst.set_nodeattr("ipgen_path", "")
                    inst.set_nodeattr("ip_path", "")
                    inst.set_nodeattr("code_gen_dir_ipgen", "")
                inst.set_nodeattr("ram_style", "block")
            elif inst.get_nodeattr("mem_mode") != "dynamic":
                inst.set_nodeattr("mem_mode", "dynamic")
                inst.set_nodeattr("ipgen_path", "")
                inst.set_nodeattr("ip_path", "")
                inst.set_nodeattr("code_gen_dir_ipgen", "")

            assignments.append(
                {
                    "name": f"{scope}{node.name}",
                    "static_weights": has_static_weights,
                    "before": before,
                    "after": {
                        "mem_mode": inst.get_nodeattr("mem_mode"),
                        "ram_style": inst.get_nodeattr("ram_style"),
                    },
                }
            )

    visit_graph(model)
    write_json(output_dir / "static_mvau_mem_mode.json", assignments)
    hw_attrs = [
        "PE",
        "SIMD",
        "parallel_window",
        "ram_style",
        "resType",
            "mem_mode",
            "runtime_writeable_weights",
            "pumpedCompute",
            "depth_trigger_uram",
            "depth_trigger_bram",
        ]
    extract_model_config_to_json(
        model,
        str(output_dir / "auto_folding_config.json"),
        hw_attrs,
    )
    return model


def set_mvau_res_type(model, output_dir: Path, res_type: str | None):
    """Set MVAU multiplier resource preference for top-level and loop-body nodes."""

    if res_type is None:
        return model

    from finn.util.basic import getHWCustomOp

    assignments = []

    def visit_graph(graph_model, scope: str = ""):
        for node in graph_model.graph.node:
            if node.op_type == "FINNLoop":
                loop_inst = getHWCustomOp(node, graph_model)
                loop_body = loop_inst.get_nodeattr("body")
                visit_graph(loop_body, scope=f"{scope}{node.name}/")
                loop_inst.set_nodeattr("body", loop_body.graph)
                continue

            if not node.op_type.startswith("MVAU"):
                continue

            inst = getHWCustomOp(node, graph_model)
            before = inst.get_nodeattr("resType")
            inst.set_nodeattr("resType", res_type)
            assignments.append(
                {
                    "name": f"{scope}{node.name}",
                    "op_type": node.op_type,
                    "before": before,
                    "after": inst.get_nodeattr("resType"),
                }
            )

    visit_graph(model)
    write_json(output_dir / "mvau_res_type.json", assignments)
    return model


def _matches_scoped_node(name: str, scoped_name: str, match_names: set[str]) -> bool:
    """Match leaf, slash-scoped, or folding-config loop-body node names."""

    return (
        name in match_names
        or scoped_name in match_names
        or scoped_name.replace("/", "_body_") in match_names
    )


def set_mvau_pumped_compute(
    model, output_dir: Path, enable: bool, exclude_nodes: set[str] | None = None
):
    """Enable double-pumped compute for RTL MVAUs."""

    if not enable:
        return model

    from finn.util.basic import getHWCustomOp

    exclude_nodes = set() if exclude_nodes is None else set(exclude_nodes)
    assignments = []

    def visit_graph(graph_model, scope: str = ""):
        for node in graph_model.graph.node:
            if node.op_type == "FINNLoop":
                loop_inst = getHWCustomOp(node, graph_model)
                loop_body = loop_inst.get_nodeattr("body")
                visit_graph(loop_body, scope=f"{scope}{node.name}/")
                loop_inst.set_nodeattr("body", loop_body.graph)
                continue

            if node.op_type != "MVAU_rtl":
                continue

            inst = getHWCustomOp(node, graph_model)
            before = inst.get_nodeattr("pumpedCompute")
            scoped_name = f"{scope}{node.name}"
            excluded = _matches_scoped_node(node.name, scoped_name, exclude_nodes)
            if inst.get_nodeattr("SIMD") > 1 and not excluded:
                inst.set_nodeattr("pumpedCompute", 1)
            elif excluded:
                inst.set_nodeattr("pumpedCompute", 0)
            assignments.append(
                {
                    "name": scoped_name,
                    "simd": inst.get_nodeattr("SIMD"),
                    "excluded": excluded,
                    "before": before,
                    "after": inst.get_nodeattr("pumpedCompute"),
                }
            )

    visit_graph(model)
    write_json(output_dir / "mvau_pumped_compute.json", assignments)
    return model


def clear_layernorm_ipgen_artifacts(model, output_dir: Path):
    """Force LayerNorm RTL regeneration so rtllib changes are picked up."""

    from finn.util.basic import getHWCustomOp

    cleared = []

    def visit_graph(graph_model, scope: str = ""):
        for node in graph_model.graph.node:
            if node.op_type == "FINNLoop":
                loop_inst = getHWCustomOp(node, graph_model)
                loop_body = loop_inst.get_nodeattr("body")
                visit_graph(loop_body, scope=f"{scope}{node.name}/")
                loop_inst.set_nodeattr("body", loop_body.graph)
                continue

            if node.op_type != "LayerNorm_rtl":
                continue

            inst = getHWCustomOp(node, graph_model)
            before = {
                "code_gen_dir_ipgen": inst.get_nodeattr("code_gen_dir_ipgen"),
                "ipgen_path": inst.get_nodeattr("ipgen_path"),
                "ip_path": inst.get_nodeattr("ip_path"),
            }
            inst.set_nodeattr("code_gen_dir_ipgen", "")
            inst.set_nodeattr("ipgen_path", "")
            inst.set_nodeattr("ip_path", "")
            cleared.append({"name": f"{scope}{node.name}", "before": before})

    visit_graph(model)
    write_json(output_dir / "layernorm_codegen_reset.json", cleared)
    return model


def keep_embedding_output_only(model, output_dir: Path):
    """Keep only the SigLIP image embedding graph output.

    The static ImageNet head is useful when PL should return class scores, but
    an embedding-only build is a cleaner option when label comparison is done
    outside PL. If the embedding output is currently only a branch of a
    DuplicateStreams node, expose the stream before the duplicate so cleanup can
    remove both the duplicate and the class-score branch.
    """

    import copy
    from math import prod
    from qonnx.transformation.remove import RemoveUnusedNodes

    outputs = list(model.graph.output)
    output_shapes = {
        value_info.name: model.get_tensor_shape(value_info.name) for value_info in outputs
    }

    embedding_output_widths = {768, 1152}

    def has_embedding_shape(value_info):
        shape = output_shapes.get(value_info.name)
        if shape is None:
            return False
        if any(not isinstance(dim, int) or dim <= 0 for dim in shape):
            return False
        return prod(shape) in embedding_output_widths

    candidates = [
        value_info
        for value_info in outputs
        if has_embedding_shape(value_info)
    ]
    if not candidates:
        candidates = [
            value_info
            for value_info in outputs
            if "embed" in value_info.name.lower()
        ]
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one embedding output; found "
            f"{[(value_info.name, output_shapes.get(value_info.name)) for value_info in candidates]}; "
            "all outputs: "
            f"{[(value_info.name, output_shapes.get(value_info.name)) for value_info in outputs]}"
        )

    selected = candidates[0]
    exposed_tensor = selected.name
    producer = model.find_producer(selected.name)
    bypassed_duplicate = False
    if producer is not None and producer.op_type == "DuplicateStreams":
        exposed_tensor = producer.input[0]
        bypassed_duplicate = True

    new_output = copy.deepcopy(selected)
    new_output.name = exposed_tensor
    for index in reversed(range(len(model.graph.value_info))):
        if model.graph.value_info[index].name == exposed_tensor:
            del model.graph.value_info[index]
    del model.graph.output[:]
    model.graph.output.append(new_output)
    model = model.transform(RemoveUnusedNodes())
    model = model.cleanup()

    write_json(
        output_dir / "embedding_only_output.json",
        {
            "original_outputs": [
                {"name": value_info.name, "shape": output_shapes.get(value_info.name)}
                for value_info in outputs
            ],
            "embedding_output_widths": sorted(embedding_output_widths),
            "selected_output": selected.name,
            "exposed_output": exposed_tensor,
            "bypassed_duplicate_streams": bypassed_duplicate,
            "remaining_outputs": [
                {"name": value_info.name, "shape": model.get_tensor_shape(value_info.name)}
                for value_info in model.graph.output
            ],
            "remaining_nodes": len(model.graph.node),
        },
    )
    return model


def roll_static_vision_mlo(model, output_dir: Path, depth: int):
    from full_siglip.common import (
        find_static_vision_loop_body_ranges,
        first_static_vision_loop_body_node_range,
    )
    from finn.transformation.fpgadataflow.loop_rolling import LoopExtraction, LoopRolling
    from finn.transformation.fpgadataflow.set_loop_boundary import SetLoopBoundary
    from finn.util.basic import getHWCustomOp
    from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
    from qonnx.transformation.infer_datatypes import InferDataTypes
    from qonnx.transformation.infer_shapes import InferShapes

    loop_ranges = find_static_vision_loop_body_ranges(model.model, depth)
    if len(loop_ranges) != depth:
        raise RuntimeError(f"Expected {depth} static SigLIP vision blocks, found {len(loop_ranges)}")

    loop_op_types = loop_ranges[0]["loop_op_types"]
    mismatched = [item["block"] for item in loop_ranges if item["loop_op_types"] != loop_op_types]
    if mismatched:
        raise RuntimeError(f"Static SigLIP loop-body topology mismatch in blocks {mismatched}")

    write_json(output_dir / "static_vision_mlo_ranges.json", loop_ranges)
    start_node, end_node = first_static_vision_loop_body_node_range(model, depth)
    node_metadata = {
        "pkg.torch.onnx.name_scopes": "['', 'siglip_vision_encoder_layer_0']",
        "pkg.torch.onnx.class_hierarchy": "['SigLIPVisionEncoder', 'SigLIPEncoderLayer']",
    }
    model = model.transform(SetLoopBoundary(node_metadata, (start_node, end_node)))
    loop_extraction = LoopExtraction(hierarchy_list=[["", "siglip_vision_encoder_layer_0"]])
    model = model.transform(loop_extraction)
    fn_count = len(model.get_nodes_by_op_type("fn_loop-body"))
    if fn_count != depth:
        raise RuntimeError(f"Loop extraction found {fn_count} function calls, expected {depth}")

    model = model.transform(LoopRolling(loop_extraction.loop_body_template, fold_constants=False))
    model = model.transform(InferShapes(), apply_to_subgraphs=True)
    model = model.transform(InferDataTypes(), apply_to_subgraphs=True)
    model = model.transform(GiveUniqueNodeNames(), apply_to_subgraphs=True)
    model = model.transform(GiveReadableTensorNames())

    loop_template = Path("loop-body-template.onnx")
    if loop_template.is_file():
        loop_template.replace(output_dir / "loop-body-template.onnx")

    for node in model.get_nodes_by_op_type("FINNLoop"):
        node_inst = getHWCustomOp(node)
        loop_body = node_inst.get_nodeattr("body")
        loop_body = loop_body.transform(GiveUniqueNodeNames(prefix=node.name + "_"))
        node_inst.set_nodeattr("body", loop_body.graph)
    return model


def build_static(
    *,
    input_model: Path,
    output_dir: Path,
    mode: str,
    board: str,
    clock_ns: float,
    target_fps: Optional[float],
    loop_target_fps: Optional[float],
    mvau_wwidth_max: int,
    mvau_impl_style: str,
    mvau_hls_nodes: list[str],
    mvau_res_type: str | None,
    mvau_pumped_compute: bool,
    mvau_pumped_exclude_nodes: list[str],
    folding_config_file: Optional[Path],
    mlo: bool,
    depth: int,
    auto_fifo_depths: bool,
    split_large_fifos: bool,
    save_intermediate_models: bool,
    output_mode: str,
    overlapped_scheduler_spec_json: Optional[Path] = None,
    overlapped_scheduler_materialization_json: Optional[Path] = None,
    overlapped_scheduler_implementation_kind: str = "contract",
    generate_siglip2_86m_overlapped_scheduler_spec: bool = False,
    weight_bits: int = 6,
    act_bits: int = 8,
    verify_stitched_ip_rtlsim: bool = False,
    verify_input_npy: Optional[Path] = None,
    verify_expected_output_npy: Optional[Path] = None,
    verification_atol: float = 1e-1,
    verify_save_rtlsim_waveforms: bool = False,
    rtlsim_use_vivado_comps: bool = True,
) -> None:
    import finn.builder.build_dataflow as build
    from finn.builder.build_dataflow_config import (
        DataflowBuildConfig,
        DataflowOutputType,
        VerificationStepType,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    def step_keep_embedding_output_only(model, cfg):
        if output_mode == "both":
            return model
        return keep_embedding_output_only(model, output_dir)

    def step_specialize_siglip_static(model, cfg):
        from full_siglip.prepare_model import stage_specialize_layers

        return stage_specialize_layers(
            model,
            cfg,
            output_dir,
            mvau_impl_style=mvau_impl_style,
            mvau_hls_nodes=mvau_hls_nodes,
        )

    def step_roll_static_siglip_mlo(model, cfg):
        if not mlo:
            return model
        return roll_static_vision_mlo(model, output_dir, depth)

    def step_round_static_siglip_mlo_threshold_params(model, cfg):
        if not mlo:
            return model
        model = round_and_clip_mlo_threshold_params(model)
        return remove_unused_mlo_static_scaffolding(model, output_dir)

    def step_fold_static_siglip_loop_bodies(model, cfg):
        if not mlo:
            return model
        loop_target_cycles = (
            cycles_per_frame(clock_ns, loop_target_fps)
            if loop_target_fps is not None
            else cfg._resolve_cycles_per_frame()
        )
        return fold_static_siglip_loop_bodies(
            model,
            target_cycles_per_frame=loop_target_cycles,
            mvau_wwidth_max=cfg.mvau_wwidth_max,
            output_dir=output_dir,
        )

    def step_refresh_static_siglip_loop_body_summary(model, cfg):
        if not mlo:
            return model
        loop_target_cycles = (
            cycles_per_frame(clock_ns, loop_target_fps)
            if loop_target_fps is not None
            else cfg._resolve_cycles_per_frame()
        )
        return refresh_static_siglip_loop_body_summary(
            model,
            target_cycles_per_frame=loop_target_cycles,
            output_dir=output_dir,
        )

    def step_materialize_siglip_overlapped_scheduler(model, cfg):
        spec_json = overlapped_scheduler_spec_json
        if generate_siglip2_86m_overlapped_scheduler_spec:
            from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer

            loop_summaries = json.loads(
                (output_dir / "static_loop_body_folding_summary.json").read_text()
            )
            spec = siglip2_86m_overlapped_scheduler_spec_from_cycles(
                cycle_dict=model.analysis(exp_cycles_per_layer),
                loop_summaries=loop_summaries,
                clock_ns=clock_ns,
                depth=depth,
                weight_bits=weight_bits,
                act_bits=act_bits,
            )
            spec_json = output_dir / "siglip2_86m_overlapped_scheduler_spec.json"
            write_json(spec_json, spec)
        if spec_json is None:
            return model
        from full_siglip.materialize_so400m_overlapped_scheduler import (
            summarize_materialization,
            write_json as write_materialization_json,
            write_markdown as write_materialization_markdown,
        )

        local_json = output_dir / "overlapped_scheduler_materialization.json"
        local_md = output_dir / "overlapped_scheduler_materialization.md"
        extra_json = overlapped_scheduler_materialization_json
        extra_md = None if extra_json is None else extra_json.with_suffix(".md")
        annotated_onnx = output_dir / "overlapped_scheduler_annotated.onnx"
        model, report = summarize_materialization(
            json.loads(spec_json.read_text()),
            output_dir=output_dir / "overlapped_scheduler",
            source_root=REPO_ROOT,
            scheduler_spec_path=spec_json,
            input_onnx=None,
            annotated_onnx=annotated_onnx,
            model=model,
            implementation_kind=overlapped_scheduler_implementation_kind,
        )
        write_materialization_json(local_json, report)
        write_materialization_markdown(local_md, report)
        if extra_json is not None and extra_json != local_json:
            write_materialization_json(extra_json, report)
            write_materialization_markdown(extra_md, report)
        return model

    def step_set_static_siglip_mvau_mem(model, cfg):
        return set_static_mvau_external_mem(model, output_dir)

    def step_set_siglip_mvau_res_type(model, cfg):
        return set_mvau_res_type(model, output_dir, mvau_res_type)

    def step_set_siglip_mvau_pumped_compute(model, cfg):
        return set_mvau_pumped_compute(
            model,
            output_dir,
            mvau_pumped_compute,
            exclude_nodes=set(mvau_pumped_exclude_nodes),
        )

    def step_set_large_outer_shuffle_uram(model, cfg):
        return set_large_outer_shuffle_uram(model, output_dir)

    def step_clear_layernorm_ipgen_artifacts(model, cfg):
        return clear_layernorm_ipgen_artifacts(model, output_dir)

    steps = [
        step_keep_embedding_output_only,
        "step_create_dataflow_partition",
        step_specialize_siglip_static,
        step_roll_static_siglip_mlo,
        step_set_siglip_mvau_res_type,
        "step_transpose_decomposition",
        "step_target_fps_parallelization",
        step_fold_static_siglip_loop_bodies,
        "step_apply_folding_config",
        step_set_siglip_mvau_pumped_compute,
        step_set_large_outer_shuffle_uram,
        "step_minimize_bit_width",
        step_round_static_siglip_mlo_threshold_params,
        step_set_static_siglip_mvau_mem,
        step_refresh_static_siglip_loop_body_summary,
        step_materialize_siglip_overlapped_scheduler,
        "step_generate_estimate_reports",
        step_clear_layernorm_ipgen_artifacts,
    ]
    outputs = [DataflowOutputType.ESTIMATE_REPORTS]
    if mode in ("rtl", "dcp"):
        steps.extend(
            [
                "step_hw_codegen",
                "step_hw_ipgen",
                "step_set_fifo_depths",
                "step_create_stitched_ip",
            ]
        )
        outputs.append(DataflowOutputType.STITCHED_IP)
    if mode == "dcp":
        outputs.append(DataflowOutputType.OOC_SYNTH)

    cfg = DataflowBuildConfig(
        output_dir=str(output_dir),
        steps=steps,
        synth_clk_period_ns=clock_ns,
        board=board,
        target_fps=target_fps,
        mvau_wwidth_max=mvau_wwidth_max,
        folding_config_file=str(folding_config_file) if folding_config_file else None,
        standalone_thresholds=True,
        infer_shuffle_skip_first=False,
        save_intermediate_models=save_intermediate_models,
        auto_fifo_depths=auto_fifo_depths,
        split_large_fifos=split_large_fifos,
        folding_two_pass_relaxation=not mlo,
        generate_outputs=outputs,
        # MLO rolling is performed explicitly in this wrapper step when
        # requested. Keep cfg.mlo disabled so FINN does not expect a second
        # loop-body range to be supplied through DataflowBuildConfig.
        mlo=False,
        stitched_ip_gen_dcp=mode == "dcp",
        verify_steps=(
            [VerificationStepType.STITCHED_IP_RTLSIM] if verify_stitched_ip_rtlsim else None
        ),
        verify_input_npy=str(verify_input_npy) if verify_input_npy else "input.npy",
        verify_expected_output_npy=(
            str(verify_expected_output_npy) if verify_expected_output_npy else "expected_output.npy"
        ),
        verification_atol=verification_atol,
        verify_save_rtlsim_waveforms=verify_save_rtlsim_waveforms,
        rtlsim_use_vivado_comps=rtlsim_use_vivado_comps,
        no_stdout_redirect=True,
        enable_build_pdb_debug=False,
    )
    build_result = build.build_dataflow_cfg(str(input_model), cfg)
    if build_result != 0:
        raise SystemExit(build_result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="full_siglip/build/static_imagenet_w8a8_smoke_finn_probe/convert_to_hw_shuffle.onnx",
    )
    parser.add_argument(
        "--output-dir",
        default="full_siglip/build/static_imagenet_w8a8_smoke_estimate",
    )
    parser.add_argument("--mode", choices=("estimate", "rtl", "dcp"), default="estimate")
    parser.add_argument(
        "--allow-infeasible-so400m-dcp",
        action="store_true",
        help=(
            "Allow a diagnostic DCP run for the exact SO400M patch14/384 W6A8 graph even "
            "when the local objective/lower-bound gates reject it."
        ),
    )
    parser.add_argument(
        "--so400m-preflight",
        action="store_true",
        help=(
            "Print exact SO400M W6A8 DCP preflight JSON and exit before starting "
            "any FINN build work."
        ),
    )
    parser.add_argument("--board", default=DEFAULT_BOARD)
    parser.add_argument("--clock-ns", type=float, default=DEFAULT_CLOCK_NS)
    parser.add_argument("--target-fps", type=float, default=DEFAULT_TARGET_FPS)
    parser.add_argument(
        "--loop-target-fps",
        type=float,
        default=0.0,
        help=(
            "Optional target FPS for FINNLoop bodies. When unset, loop bodies use "
            "--target-fps; use this for mixed top-level/loop folding."
        ),
    )
    parser.add_argument("--mvau-wwidth-max", type=int, default=36)
    parser.add_argument(
        "--mvau-impl-style",
        choices=("rtl", "hls"),
        default="rtl",
        help="Preferred implementation style for MVAU nodes before specialization.",
    )
    parser.add_argument(
        "--mvau-hls-nodes",
        default="",
        help="Comma-separated unspecialized MVAU node names to force to HLS.",
    )
    parser.add_argument(
        "--mvau-res-type",
        choices=("auto", "dsp", "lut"),
        default=None,
        help="Optional resType override for MVAU nodes after specialization.",
    )
    parser.add_argument(
        "--mvau-pumped-compute",
        action="store_true",
        help="Enable double-pumped compute for RTL MVAUs with SIMD > 1.",
    )
    parser.add_argument(
        "--mvau-pumped-exclude-nodes",
        default="",
        help="Comma-separated RTL MVAU names to keep unpumped even when --mvau-pumped-compute is set.",
    )
    parser.add_argument(
        "--folding-config-file",
        default=None,
        help="Optional JSON folding config to apply after automatic target-FPS folding.",
    )
    parser.add_argument("--mlo", action="store_true", help="Roll the 12 static vision blocks")
    parser.add_argument("--depth", type=int, default=SIGLIP_DEPTH)
    parser.add_argument(
        "--no-auto-fifo-depths",
        dest="auto_fifo_depths",
        action="store_false",
        help="Skip rtlsim FIFO characterization and use inserted/configured FIFO depths.",
    )
    parser.add_argument(
        "--split-large-fifos",
        action="store_true",
        help="Split large FIFOs after FIFO insertion/sizing.",
    )
    parser.add_argument(
        "--no-save-intermediate-models",
        dest="save_intermediate_models",
        action="store_false",
        help="Do not save intermediate ONNX checkpoints, useful for folding sweeps.",
    )
    parser.add_argument(
        "--output-mode",
        choices=("both", "embedding"),
        default="both",
        help="Use 'embedding' to expose only the image embedding and prune the static ImageNet head.",
    )
    parser.add_argument(
        "--overlapped-scheduler-spec-json",
        default=None,
        help=(
            "Estimate-only path: materialize and annotate the valid overlapped "
            "FINNLoop scheduler contract before estimate reports."
        ),
    )
    parser.add_argument(
        "--overlapped-scheduler-materialization-json",
        default=None,
        help="Optional extra path for the overlapped scheduler materialization report.",
    )
    parser.add_argument(
        "--overlapped-scheduler-implementation-kind",
        choices=("contract", "builtin_stream_feedback"),
        default="contract",
        help=(
            "Use contract for estimate-only audits, or builtin_stream_feedback "
            "for the guarded DCP-preflight stream-feedback MLO shell."
        ),
    )
    parser.add_argument(
        "--generate-siglip2-86m-overlapped-scheduler-spec",
        action="store_true",
        help=(
            "Generate the exact SigLIP2 86M overlapped FINNLoop scheduler spec "
            "from the current folded graph and loop-body summary."
        ),
    )
    parser.add_argument(
        "--weight-bits",
        type=int,
        default=6,
        help="Weight precision to record in generated scheduler metadata.",
    )
    parser.add_argument(
        "--act-bits",
        type=int,
        default=8,
        help="Activation precision to record in generated scheduler metadata.",
    )
    parser.add_argument(
        "--verify-stitched-ip-rtlsim",
        action="store_true",
        help="Run FINN stitched-IP RTL simulation after step_create_stitched_ip.",
    )
    parser.add_argument(
        "--verify-input-npy",
        default=None,
        help="Input .npy file for FINN verification steps.",
    )
    parser.add_argument(
        "--verify-expected-output-npy",
        default=None,
        help="Expected-output .npy file for FINN verification steps.",
    )
    parser.add_argument(
        "--verification-atol",
        type=float,
        default=1e-1,
        help="Absolute tolerance for FINN verification output comparison.",
    )
    parser.add_argument(
        "--verify-save-rtlsim-waveforms",
        action="store_true",
        help="Save stitched-IP RTL simulation waveforms during verification.",
    )
    parser.add_argument(
        "--rtlsim-no-vivado-comps",
        dest="rtlsim_use_vivado_comps",
        action="store_false",
        help="Replace Vivado FIFO components with RTL implementations before RTL sim.",
    )
    args = parser.parse_args()
    target_fps = None if args.target_fps <= 0 else args.target_fps
    loop_target_fps = None if args.loop_target_fps <= 0 else args.loop_target_fps
    mvau_hls_nodes = [name.strip() for name in args.mvau_hls_nodes.split(",") if name.strip()]
    mvau_pumped_exclude_nodes = [
        name.strip() for name in args.mvau_pumped_exclude_nodes.split(",") if name.strip()
    ]
    input_model = repo_path(args.input)
    output_dir = repo_path(args.output_dir)
    if args.so400m_preflight:
        payload = so400m_preflight_payload(input_model, output_dir)
        print(json.dumps(payload, indent=2))
        raise SystemExit(0 if payload["normal_dcp_allowed"] else 1)
    if args.mode == "dcp":
        guard_so400m_w6a8_dcp(input_model, output_dir, args.allow_infeasible_so400m_dcp)
    if args.verify_stitched_ip_rtlsim and args.mode == "estimate":
        raise SystemExit("--verify-stitched-ip-rtlsim requires --mode rtl or --mode dcp")
    if (
        args.overlapped_scheduler_spec_json is not None
        and args.generate_siglip2_86m_overlapped_scheduler_spec
    ):
        raise SystemExit(
            "Use either --overlapped-scheduler-spec-json or "
            "--generate-siglip2-86m-overlapped-scheduler-spec, not both"
        )
    if (
        args.overlapped_scheduler_spec_json is not None
        and args.mode != "estimate"
        and args.overlapped_scheduler_implementation_kind != "builtin_stream_feedback"
    ):
        raise SystemExit(
            "--overlapped-scheduler-spec-json is estimate-only until real "
            "DCP-ready overlapped scheduler RTL is available"
        )
    if (
        args.generate_siglip2_86m_overlapped_scheduler_spec
        and args.mode != "estimate"
        and args.overlapped_scheduler_implementation_kind != "builtin_stream_feedback"
    ):
        raise SystemExit(
            "--generate-siglip2-86m-overlapped-scheduler-spec requires "
            "--overlapped-scheduler-implementation-kind builtin_stream_feedback "
            "outside estimate mode"
        )

    build_static(
        input_model=input_model,
        output_dir=output_dir,
        mode=args.mode,
        board=args.board,
        clock_ns=args.clock_ns,
        target_fps=target_fps,
        loop_target_fps=loop_target_fps,
        mvau_wwidth_max=args.mvau_wwidth_max,
        mvau_impl_style=args.mvau_impl_style,
        mvau_hls_nodes=mvau_hls_nodes,
        mvau_res_type=args.mvau_res_type,
        mvau_pumped_compute=args.mvau_pumped_compute,
        mvau_pumped_exclude_nodes=mvau_pumped_exclude_nodes,
        folding_config_file=(
            repo_path(args.folding_config_file) if args.folding_config_file else None
        ),
        mlo=args.mlo,
        depth=args.depth,
        auto_fifo_depths=args.auto_fifo_depths,
        split_large_fifos=args.split_large_fifos,
        save_intermediate_models=args.save_intermediate_models,
        output_mode=args.output_mode,
        overlapped_scheduler_spec_json=(
            repo_path(args.overlapped_scheduler_spec_json)
            if args.overlapped_scheduler_spec_json
            else None
        ),
        overlapped_scheduler_materialization_json=(
            repo_path(args.overlapped_scheduler_materialization_json)
            if args.overlapped_scheduler_materialization_json
            else None
        ),
        overlapped_scheduler_implementation_kind=args.overlapped_scheduler_implementation_kind,
        generate_siglip2_86m_overlapped_scheduler_spec=(
            args.generate_siglip2_86m_overlapped_scheduler_spec
        ),
        weight_bits=args.weight_bits,
        act_bits=args.act_bits,
        verify_stitched_ip_rtlsim=args.verify_stitched_ip_rtlsim,
        verify_input_npy=repo_path(args.verify_input_npy) if args.verify_input_npy else None,
        verify_expected_output_npy=(
            repo_path(args.verify_expected_output_npy)
            if args.verify_expected_output_npy
            else None
        ),
        verification_atol=args.verification_atol,
        verify_save_rtlsim_waveforms=args.verify_save_rtlsim_waveforms,
        rtlsim_use_vivado_comps=args.rtlsim_use_vivado_comps,
    )
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
