#!/usr/bin/env python3
"""Probe full SigLIP QONNX through the early FINN dataflow transforms."""

from __future__ import annotations

import argparse
import signal
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from full_siglip.common import (
    DEFAULT_BOARD,
    DEFAULT_CLOCK_NS,
    DEFAULT_QONNX,
    DEFAULT_TARGET_FPS,
    repo_path,
    summarize_model,
    write_json,
    write_rtl_specialization_config,
)

CONVERT_TO_HW_STAGE_NAMES = (
    "convert_to_hw_requant_0",
    "convert_to_hw_threshold_0",
    "convert_to_hw_absorb_pre_matmul_dequant",
    "convert_to_hw_integerize_static_lhs_matmul",
    "convert_to_hw_binary_matmul",
    "convert_to_hw_quantized_matmul",
    "convert_to_hw_vector_vector",
    "convert_to_hw_label_select",
    "convert_to_hw_requant_1",
    "convert_to_hw_threshold_1",
    "convert_to_hw_pool",
    "convert_to_hw_conv_input_gen",
    "convert_to_hw_flatten",
    "convert_to_hw_cls_token",
    "convert_to_hw_concat",
    "convert_to_hw_split",
    "convert_to_hw_repair_static_shapes",
    "convert_to_hw_fold_constants",
    "convert_to_hw_elementwise_binary",
    "convert_to_hw_where",
    "convert_to_hw_relu_max",
    "convert_to_hw_upsample",
    "convert_to_hw_global_pooling",
    "convert_to_hw_select_token",
    "convert_to_hw_lookup",
    "convert_to_hw_softmax",
    "convert_to_hw_pwpolyf",
    "convert_to_hw_layernorm",
    "convert_to_hw_crop",
    "convert_to_hw_duplicate_streams",
    "convert_to_hw_absorb_elementwise_into_requant",
    "convert_to_hw_absorb_consecutive_transposes",
    "convert_to_hw_give_unique_node_names",
    "convert_to_hw_infer_data_layouts",
    "convert_to_hw_shuffle",
)

STAGE_NAMES = (
    "qonnx_domain_fix",
    "infer_shapes",
    "give_unique_parameter_tensors",
    "fold_constants",
    "fold_transpose_into_quant_init",
    "remove_unused_tensors",
    "remove_static_graph_inputs",
    "give_unique_node_names",
    "give_readable_tensor_names",
    "convert_qonnx_to_finn",
    "gemm_to_matmul",
    "extract_conv_bias_safe",
    "infer_datatypes_before_quant_weight_fold",
    "fold_quant_weights",
    "infer_shapes_after_quant_weight_fold",
    "convert_quant_act_to_multithreshold",
    "infer_datatypes_after_quant_act",
    "avgpool_trunc_to_quantavgpool",
    "remove_identity_ops_after_qonnx",
    "tidy_up_no_fold_constants",
    "tidy_up",
    "streamline",
    "streamline_absorb_sign_bias_pre",
    "streamline_lower_convs_to_matmul",
    "streamline_make_maxpool_nhwc_0",
    "streamline_absorb_transpose_into_multithreshold",
    "streamline_make_maxpool_nhwc_1",
    "streamline_absorb_consecutive_transposes",
    "streamline_convert_bipolar_matmul",
    "streamline_absorb_scalar_mul_add_topk",
    "streamline_infer_data_layouts",
    "streamline_remove_unused_tensors",
    "extract_norm_scale_bias",
    "repair_static_shapes",
    "convert_to_hw",
    *CONVERT_TO_HW_STAGE_NAMES,
    "dehw_siglip_io_postprocess",
    "dataflow_partition",
    "specialize_layers",
)


@contextmanager
def stage_timeout(seconds: int | None, stage_name: str):
    if seconds is None or seconds <= 0:
        yield
        return

    def _handle_timeout(_signum, _frame):
        raise TimeoutError(f"{stage_name} exceeded {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def make_cfg(args: argparse.Namespace, output_dir: Path):
    from finn.builder.build_dataflow_config import DataflowBuildConfig, DataflowOutputType

    return DataflowBuildConfig(
        output_dir=str(output_dir),
        synth_clk_period_ns=args.clock_ns,
        board=args.board,
        target_fps=args.target_fps,
        standalone_thresholds=True,
        infer_shuffle_skip_first=False,
        save_intermediate_models=True,
        generate_outputs=[DataflowOutputType.ESTIMATE_REPORTS],
        mlo=False,
        no_stdout_redirect=True,
    )


def save_checkpoint(model, output_dir: Path, stage: str, save: bool) -> Path | None:
    if not save:
        return None
    path = output_dir / f"{stage}.onnx"
    model.save(str(path))
    return path


def stage_summary(model) -> dict:
    return {
        "nodes": len(model.graph.node),
        "op_counts": dict(Counter(node.op_type for node in model.graph.node).most_common()),
    }


def stage_extract_norm_scale_bias(model, _cfg):
    from finn.transformation.streamline.extract_norm_scale_bias import ExtractNormScaleBias
    from qonnx.transformation.general import GiveUniqueParameterTensors, SortGraph
    from qonnx.transformation.infer_datatypes import InferDataTypes
    from qonnx.transformation.infer_shapes import InferShapes
    from qonnx.transformation.remove import RemoveIdentityOps

    model = model.transform(ExtractNormScaleBias(cleanup=False), cleanup=False)
    model = model.transform(SortGraph())
    model = model.transform(RemoveIdentityOps())
    model = model.transform(GiveUniqueParameterTensors())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    return model


def _attr_value(node, name: str, default=None):
    import onnx

    for attr in node.attribute:
        if attr.name == name:
            return onnx.helper.get_attribute_value(attr)
    return default


def _known_dim(dim) -> int | None:
    return dim if isinstance(dim, int) and dim > 0 else None


def _static_shape(shape) -> list[int] | None:
    if shape is None:
        return None
    if not all(_known_dim(dim) is not None for dim in shape):
        return None
    return [int(dim) for dim in shape]


def _prod(values: list[int]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


def _broadcast_shapes(*shapes) -> list[int] | None:
    from itertools import zip_longest

    if any(shape is None for shape in shapes):
        return None
    out_rev = []
    for dims in zip_longest(*(reversed(shape) for shape in shapes), fillvalue=1):
        known = [_known_dim(dim) for dim in dims]
        known = [dim for dim in known if dim is not None]
        non_one = {dim for dim in known if dim != 1}
        if len(non_one) > 1:
            return None
        if non_one:
            out_rev.append(non_one.pop())
        elif known:
            out_rev.append(max(known))
        else:
            return None
    return list(reversed(out_rev))


def _reshape_output_shape(input_shape: list[int], target_shape, allowzero: int) -> list[int] | None:
    target = [int(dim) for dim in target_shape]
    out = []
    infer_index = None
    for index, dim in enumerate(target):
        if dim == 0 and allowzero == 0:
            if index >= len(input_shape):
                return None
            out.append(int(input_shape[index]))
        elif dim == -1:
            if infer_index is not None:
                return None
            infer_index = index
            out.append(-1)
        elif dim > 0:
            out.append(dim)
        else:
            return None
    if infer_index is not None:
        known = _prod([dim for dim in out if dim != -1])
        total = _prod(input_shape)
        if known == 0 or total % known != 0:
            return None
        out[infer_index] = total // known
    if _prod(out) != _prod(input_shape):
        return None
    return out


def _matmul_output_shape(lhs_shape, rhs_shape) -> list[int] | None:
    lhs = _static_shape(lhs_shape)
    rhs = _static_shape(rhs_shape)
    if lhs is None or rhs is None:
        return None
    if len(lhs) == 1 and len(rhs) == 1:
        if lhs[0] != rhs[0]:
            return None
        return []
    if len(lhs) == 1:
        if lhs[0] != rhs[-2]:
            return None
        return rhs[:-2] + [rhs[-1]]
    if len(rhs) == 1:
        if lhs[-1] != rhs[0]:
            return None
        return lhs[:-1]
    if lhs[-1] != rhs[-2]:
        return None
    batch_shape = _broadcast_shapes(lhs[:-2], rhs[:-2])
    if batch_shape is None:
        return None
    return batch_shape + [lhs[-2], rhs[-1]]


def stage_repair_static_shapes(model, _cfg):
    """Repair static shapes lost around SigLIP reshape/norm glue.

    The full SigLIP export has fixed public input shapes, but skipping full
    constant folding leaves small shape subgraphs in the vision embedding path.
    ONNX shape inference then emits symbolic or zero dimensions, which prevents
    FINN's existing elementwise/normalization conversions from firing.
    """

    import numpy as np
    from onnx import numpy_helper

    graph = model.graph
    small_const_values = {}
    graph_modified = False
    any_modified = False
    const_fold_limit = 4096

    def get_shape(tensor_name: str):
        initializer = model.get_initializer(tensor_name)
        if initializer is not None:
            return list(initializer.shape)
        try:
            return model.get_tensor_shape(tensor_name)
        except Exception:
            return None

    def set_shape(tensor_name: str, shape) -> None:
        nonlocal any_modified, graph_modified
        shape = _static_shape(shape)
        if shape is None:
            return
        if get_shape(tensor_name) != shape:
            model.set_tensor_shape(tensor_name, shape)
            graph_modified = True
            any_modified = True

    def get_const(tensor_name: str):
        if tensor_name in small_const_values:
            return small_const_values[tensor_name]
        initializer = model.get_initializer(tensor_name)
        if initializer is not None and initializer.size <= const_fold_limit:
            small_const_values[tensor_name] = initializer
            return initializer
        return None

    def set_const(tensor_name: str, value) -> None:
        value = np.asarray(value)
        if value.size <= const_fold_limit:
            small_const_values[tensor_name] = value

    graph_inputs = {value_info.name for value_info in graph.input}

    for initializer in graph.initializer:
        initializer_shape = list(initializer.dims)
        try:
            current_shape = model.get_tensor_shape(initializer.name)
        except Exception:
            current_shape = None
        if current_shape != initializer_shape:
            model.set_tensor_shape(initializer.name, initializer_shape)
            graph_modified = True
            any_modified = True

    def has_source(tensor_name: str) -> bool:
        if tensor_name in graph_inputs:
            return True
        if model.get_initializer(tensor_name) is not None:
            return True
        return model.find_producer(tensor_name) is not None

    for node in graph.node:
        if node.op_type != "Mul" or len(node.input) != 2:
            continue
        missing_indices = [index for index, name in enumerate(node.input) if not has_source(name)]
        if len(missing_indices) != 1:
            continue
        missing_index = missing_indices[0]
        other_index = 1 - missing_index
        missing_name = node.input[missing_index]
        if "/mlp/fc1/Add_output_0" in missing_name and has_source(node.input[other_index]):
            node.input[missing_index] = node.input[other_index]
            graph_modified = True
            any_modified = True

    for _ in range(4):
        pass_modified = False
        for node in graph.node:
            if node.op_type == "Shape":
                in_shape = _static_shape(get_shape(node.input[0]))
                if in_shape is not None:
                    set_shape(node.output[0], [len(in_shape)])
                    set_const(node.output[0], np.asarray(in_shape, dtype=np.int64))

            elif node.op_type == "ConstantOfShape":
                shape_value = get_const(node.input[0])
                if shape_value is not None:
                    out_shape = [int(dim) for dim in np.asarray(shape_value).reshape(-1)]
                    if all(dim >= 0 for dim in out_shape):
                        set_shape(node.output[0], out_shape)
                        fill_value = _attr_value(node, "value", None)
                        if fill_value is not None:
                            fill = numpy_helper.to_array(fill_value)
                            fill_scalar = fill.reshape(-1)[0]
                        else:
                            fill_scalar = 0
                        if _prod(out_shape) <= 1024:
                            set_const(node.output[0], np.full(out_shape, fill_scalar))

            elif node.op_type in {"Cast", "Exp"}:
                set_shape(node.output[0], get_shape(node.input[0]))
                value = get_const(node.input[0])
                if value is not None:
                    if node.op_type == "Exp":
                        value = np.exp(value)
                    set_const(node.output[0], value)

            elif node.op_type in {"Add", "Mul", "Sub", "Div", "Mod"}:
                out_shape = _broadcast_shapes(get_shape(node.input[0]), get_shape(node.input[1]))
                set_shape(node.output[0], out_shape)
                lhs = get_const(node.input[0])
                rhs = get_const(node.input[1])
                if lhs is not None and rhs is not None:
                    if node.op_type == "Add":
                        value = lhs + rhs
                    elif node.op_type == "Mul":
                        value = lhs * rhs
                    elif node.op_type == "Sub":
                        value = lhs - rhs
                    elif node.op_type == "Div":
                        value = lhs / rhs
                    else:
                        value = np.mod(lhs, rhs)
                    set_const(node.output[0], value)

            elif node.op_type in {
                "Equal",
                "Less",
                "LessOrEqual",
                "Greater",
                "GreaterOrEqual",
            }:
                out_shape = _broadcast_shapes(get_shape(node.input[0]), get_shape(node.input[1]))
                set_shape(node.output[0], out_shape)
                lhs = get_const(node.input[0])
                rhs = get_const(node.input[1])
                if lhs is not None and rhs is not None:
                    if node.op_type == "Equal":
                        value = lhs == rhs
                    elif node.op_type == "Less":
                        value = lhs < rhs
                    elif node.op_type == "LessOrEqual":
                        value = lhs <= rhs
                    elif node.op_type == "Greater":
                        value = lhs > rhs
                    else:
                        value = lhs >= rhs
                    set_const(node.output[0], value)

            elif node.op_type == "Unsqueeze":
                in_shape = _static_shape(get_shape(node.input[0]))
                axes = get_const(node.input[1]) if len(node.input) > 1 else _attr_value(node, "axes")
                value = get_const(node.input[0])
                if axes is not None and in_shape is not None:
                    axes = [int(axis) for axis in np.asarray(axes).reshape(-1)]
                    out_shape = list(in_shape)
                    rank = len(out_shape) + len(axes)
                    for axis in sorted(axis if axis >= 0 else axis + rank for axis in axes):
                        out_shape.insert(axis, 1)
                    set_shape(node.output[0], out_shape)
                if axes is not None and value is not None:
                    set_const(node.output[0], np.expand_dims(value, tuple(axes)))

            elif node.op_type == "Squeeze":
                in_shape = _static_shape(get_shape(node.input[0]))
                axes = get_const(node.input[1]) if len(node.input) > 1 else _attr_value(node, "axes")
                value = get_const(node.input[0])
                if axes is not None and in_shape is not None:
                    axes = sorted((int(axis) for axis in np.asarray(axes).reshape(-1)), reverse=True)
                    out_shape = list(in_shape)
                    for axis in axes:
                        axis = axis if axis >= 0 else axis + len(out_shape)
                        if axis < 0 or axis >= len(out_shape) or out_shape[axis] != 1:
                            out_shape = None
                            break
                        out_shape.pop(axis)
                    set_shape(node.output[0], out_shape)
                if axes is not None and value is not None:
                    set_const(node.output[0], np.squeeze(value, tuple(axes)))

            elif node.op_type == "Slice":
                value = get_const(node.input[0])
                if value is not None and len(node.input) >= 3:
                    starts = get_const(node.input[1])
                    ends = get_const(node.input[2])
                    axes = get_const(node.input[3]) if len(node.input) > 3 else None
                    steps = get_const(node.input[4]) if len(node.input) > 4 else None
                    if starts is not None and ends is not None:
                        axes = range(value.ndim) if axes is None else np.asarray(axes).reshape(-1)
                        steps = np.ones_like(starts) if steps is None else np.asarray(steps).reshape(-1)
                        slices = [slice(None)] * value.ndim
                        for start, end, axis, step in zip(starts, ends, axes, steps):
                            slices[int(axis)] = slice(int(start), int(end), int(step))
                        sliced = value[tuple(slices)]
                        set_shape(node.output[0], list(sliced.shape))
                        set_const(node.output[0], sliced)

            elif node.op_type == "Concat":
                input_shapes = [_static_shape(get_shape(inp)) for inp in node.input]
                axis = int(_attr_value(node, "axis", 0))
                if all(shape is not None for shape in input_shapes):
                    out_shape = list(input_shapes[0])
                    out_shape[axis] = sum(shape[axis] for shape in input_shapes)
                    set_shape(node.output[0], out_shape)
                values = [get_const(inp) for inp in node.input]
                if all(value is not None for value in values):
                    set_const(node.output[0], np.concatenate(values, axis=axis))

            elif node.op_type == "Gather":
                data_shape = _static_shape(get_shape(node.input[0]))
                indices_shape = _static_shape(get_shape(node.input[1]))
                axis = int(_attr_value(node, "axis", 0))
                if data_shape is not None and indices_shape is not None:
                    if axis < 0:
                        axis += len(data_shape)
                    set_shape(
                        node.output[0],
                        data_shape[:axis] + indices_shape + data_shape[axis + 1 :],
                    )

            elif node.op_type == "Reshape":
                in_shape = _static_shape(get_shape(node.input[0]))
                target = get_const(node.input[1])
                if in_shape is not None and target is not None:
                    out_shape = _reshape_output_shape(
                        in_shape,
                        np.asarray(target).reshape(-1),
                        int(_attr_value(node, "allowzero", 0)),
                    )
                    set_shape(node.output[0], out_shape)

            elif node.op_type == "Transpose":
                in_shape = _static_shape(get_shape(node.input[0]))
                if in_shape is not None:
                    perm = _attr_value(node, "perm", None)
                    if perm is None:
                        perm = list(reversed(range(len(in_shape))))
                    if len(perm) == len(in_shape) and all(
                        0 <= int(axis) < len(in_shape) for axis in perm
                    ):
                        set_shape(node.output[0], [in_shape[int(axis)] for axis in perm])

            elif node.op_type == "Expand":
                in_shape = _static_shape(get_shape(node.input[0]))
                target = get_const(node.input[1])
                if in_shape is not None and target is not None:
                    target_shape = [int(dim) for dim in np.asarray(target).reshape(-1)]
                    set_shape(node.output[0], _broadcast_shapes(in_shape, target_shape))

            elif node.op_type == "Tile":
                in_shape = _static_shape(get_shape(node.input[0]))
                repeats = get_const(node.input[1])
                if in_shape is not None and repeats is not None:
                    repeats = [int(dim) for dim in np.asarray(repeats).reshape(-1)]
                    if len(in_shape) == len(repeats):
                        set_shape(
                            node.output[0],
                            [dim * repeat for dim, repeat in zip(in_shape, repeats)],
                        )

            elif node.op_type == "MatMul":
                out_shape = _matmul_output_shape(
                    get_shape(node.input[0]), get_shape(node.input[1])
                )
                set_shape(node.output[0], out_shape)
                lhs = get_const(node.input[0])
                rhs = get_const(node.input[1])
                if lhs is None:
                    lhs = model.get_initializer(node.input[0])
                if rhs is None:
                    rhs = model.get_initializer(node.input[1])
                if lhs is not None and rhs is not None:
                    try:
                        value = np.matmul(lhs, rhs)
                    except ValueError:
                        value = None
                    if value is not None:
                        set_const(node.output[0], value)

            elif node.op_type == "LayerNormalization":
                set_shape(node.output[0], get_shape(node.input[0]))

        pass_modified = graph_modified
        if not pass_modified:
            break
        graph_modified = False

    foldable_ops = {
        "Add",
        "Cast",
        "Concat",
        "ConstantOfShape",
        "Div",
        "Equal",
        "Exp",
        "Expand",
        "Gather",
        "Greater",
        "GreaterOrEqual",
        "Less",
        "LessOrEqual",
        "MatMul",
        "Mod",
        "Mul",
        "Reshape",
        "Shape",
        "Slice",
        "Squeeze",
        "Sub",
        "Tile",
        "Transpose",
        "Unsqueeze",
    }
    graph_outputs = {value_info.name for value_info in graph.output}
    removed_const_nodes = True
    while removed_const_nodes:
        removed_const_nodes = False
        for node in list(graph.node):
            if node.domain not in ("", "ai.onnx") or node.op_type not in foldable_ops:
                continue
            if any(output in graph_outputs for output in node.output):
                continue
            output_values = [small_const_values.get(output) for output in node.output]
            if not output_values or any(value is None for value in output_values):
                continue
            graph.node.remove(node)
            for output, value in zip(node.output, output_values):
                model.set_initializer(output, value)
            any_modified = True
            removed_const_nodes = True
            break

    if any_modified:
        from qonnx.transformation.general import RemoveUnusedTensors
        from qonnx.transformation.infer_datatypes import InferDataTypes
        from qonnx.transformation.infer_shapes import InferShapes
        from qonnx.transformation.remove import RemoveUnusedNodes

        model = model.transform(RemoveUnusedNodes())
        model = model.transform(RemoveUnusedTensors())
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
    return model


def has_qonnx_nodes(model) -> bool:
    return any(
        model.get_nodes_by_op_type(op_type)
        for op_type in ("BinaryQuant", "Quant", "Trunc")
    )


def stage_qonnx_domain_fix(model, _cfg):
    for op_type in ("Quant", "Trunc", "BipolarQuant"):
        for node in model.get_nodes_by_op_type(op_type):
            node.domain = "qonnx.custom_op.general"
    return model


def stage_infer_shapes(model, _cfg):
    from qonnx.transformation.infer_shapes import InferShapes

    return model.transform(InferShapes())


def stage_give_unique_parameter_tensors(model, _cfg):
    from qonnx.transformation.general import GiveUniqueParameterTensors

    return model.transform(GiveUniqueParameterTensors())


def stage_fold_constants(model, _cfg):
    from qonnx.transformation.fold_constants import FoldConstants

    preserve_qnt_optypes = ["Quant", "BipolarQuant", "QuantizeLinear", "DequantizeLinear"]
    return model.transform(FoldConstants(exclude_op_types=preserve_qnt_optypes))


def stage_fold_constants_repair_static_shapes(model, cfg):
    model = stage_fold_constants(model, cfg)
    return stage_repair_static_shapes(model, cfg)


def stage_absorb_pre_matmul_dequant(model, _cfg):
    """Move scalar activation dequantization after static-weight MatMul.

    Brevitas exports some QV attention projections as:

        MultiThreshold -> Mul(scale_a) -> MatMul(int_weight) -> Mul(scale_w)

    The pre-MatMul scale makes the MatMul input FLOAT32, so FINN cannot convert
    the static-weight projection to MVAU. Since scale_a is scalar, rewrite it to:

        MultiThreshold -> MatMul(int_weight) -> Mul(scale_a * scale_w)

    This preserves the dequantized value while exposing an integer MatMul input.
    """

    import numpy as np
    from qonnx.core.datatype import DataType
    from qonnx.transformation.general import RemoveUnusedTensors
    from qonnx.transformation.infer_datatypes import InferDataTypes
    from qonnx.transformation.infer_shapes import InferShapes
    from qonnx.transformation.remove import RemoveUnusedNodes

    def producer_map():
        return {output: node for node in model.graph.node for output in node.output}

    def consumer_map():
        consumers = {}
        for node in model.graph.node:
            for tensor_name in node.input:
                consumers.setdefault(tensor_name, []).append(node)
        return consumers

    def datatype(tensor_name: str):
        try:
            return model.get_tensor_datatype(tensor_name)
        except Exception:
            return None

    def scalar_initializer(tensor_name: str):
        value = model.get_initializer(tensor_name)
        if value is None:
            return None
        value = np.asarray(value)
        if value.size != 1:
            return None
        return value.reshape(-1)[0]

    def constant_input(node, *, exclude: str | None = None):
        for tensor_name in node.input:
            if tensor_name == exclude:
                continue
            value = model.get_initializer(tensor_name)
            if value is not None:
                return tensor_name, np.asarray(value)
        return None, None

    modified = False
    for _ in range(4):
        changed_this_pass = False
        producers = producer_map()
        consumers = consumer_map()
        for matmul in list(model.graph.node):
            if matmul.op_type != "MatMul" or len(matmul.input) != 2:
                continue
            weight_name = matmul.input[1]
            weight_dtype = datatype(weight_name)
            if weight_dtype is None or not weight_dtype.is_integer():
                continue
            if model.get_initializer(weight_name) is None:
                continue

            pre_mul = producers.get(matmul.input[0])
            if pre_mul is None or pre_mul.op_type != "Mul" or len(pre_mul.input) != 2:
                continue

            int_input = None
            pre_scale_name = None
            pre_scale_value = None
            for input_name in pre_mul.input:
                input_dtype = datatype(input_name)
                other_name = pre_mul.input[1] if input_name == pre_mul.input[0] else pre_mul.input[0]
                other_scale = scalar_initializer(other_name)
                if (
                    input_dtype is not None
                    and input_dtype.is_integer()
                    and other_scale is not None
                ):
                    int_input = input_name
                    pre_scale_name = other_name
                    pre_scale_value = other_scale
                    break
            if int_input is None:
                continue

            matmul_consumers = consumers.get(matmul.output[0], [])
            if len(matmul_consumers) != 1:
                continue
            post_mul = matmul_consumers[0]
            if post_mul.op_type != "Mul":
                continue
            post_scale_name, post_scale_value = constant_input(
                post_mul, exclude=matmul.output[0]
            )
            if post_scale_name is None:
                continue

            model.set_initializer(post_scale_name, post_scale_value * pre_scale_value)
            matmul.input[0] = int_input
            model.set_tensor_datatype(matmul.output[0], DataType["INT32"])
            changed_this_pass = True
            modified = True

            if pre_scale_name is not None and not consumers.get(pre_mul.output[0], []):
                model.graph.node.remove(pre_mul)

        if not changed_this_pass:
            break

    if modified:
        model = model.transform(RemoveUnusedNodes())
        model = model.transform(RemoveUnusedTensors())
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
    return model


def stage_integerize_static_lhs_matmul(model, _cfg):
    """Expose static-lhs attention-pool MatMuls to existing MVAU conversion.

    SigLIP's vision attention-pool query can be folded into a small static
    float tensor on the left side of a MatMul:

        MatMul(static_query, quantized_keys) -> Mul(scale)

    The existing FINN MVAU conversion can already handle dynamic right-hand
    weights, but only when both MatMul inputs are integer typed. Integerize the
    folded query and absorb its scalar quantum into the following Mul.
    """

    import numpy as np
    from qonnx.core.datatype import DataType
    from qonnx.transformation.infer_datatypes import InferDataTypes
    from qonnx.transformation.infer_shapes import InferShapes

    def consumer_map():
        consumers = {}
        for node in model.graph.node:
            for tensor_name in node.input:
                consumers.setdefault(tensor_name, []).append(node)
        return consumers

    def datatype(tensor_name: str):
        try:
            return model.get_tensor_datatype(tensor_name)
        except Exception:
            return None

    def scalar_initializer(tensor_name: str):
        value = model.get_initializer(tensor_name)
        if value is None:
            return None
        value = np.asarray(value)
        if value.size != 1:
            return None
        return value.reshape(-1)[0]

    def quantize_static_tensor(value):
        value = np.asarray(value, dtype=np.float32)
        max_abs = float(np.max(np.abs(value)))
        if max_abs == 0.0:
            return value.astype(np.float32), 1.0

        flat = np.sort(np.unique(value.reshape(-1)))
        candidates = []
        diffs = np.diff(flat)
        candidates.extend(float(diff) for diff in diffs if abs(float(diff)) > 1e-12)
        abs_vals = np.unique(np.abs(flat))
        candidates.extend(float(val) for val in abs_vals if float(val) > 1e-12)

        quantum = min(abs(candidate) for candidate in candidates) if candidates else max_abs / 127.0
        q_value = np.round(value / quantum)
        reconstructed = q_value * quantum
        max_error = float(np.max(np.abs(value - reconstructed)))
        if max_error > max(1e-6, 1e-3 * max_abs) or np.max(np.abs(q_value)) > 127:
            quantum = max_abs / 127.0
            q_value = np.round(value / quantum)

        q_value = np.clip(q_value, -128, 127).astype(np.float32)
        return q_value, float(quantum)

    modified = False
    consumers = consumer_map()
    for matmul in list(model.graph.node):
        if matmul.op_type != "MatMul" or len(matmul.input) != 2:
            continue

        lhs_name, rhs_name = matmul.input
        lhs_value = model.get_initializer(lhs_name)
        if lhs_value is None or model.get_initializer(rhs_name) is not None:
            continue

        lhs_dtype = datatype(lhs_name)
        rhs_dtype = datatype(rhs_name)
        if rhs_dtype is None or not rhs_dtype.is_integer():
            continue
        if lhs_dtype is not None and lhs_dtype.is_integer():
            continue

        post_consumers = consumers.get(matmul.output[0], [])
        if len(post_consumers) != 1:
            continue
        post_mul = post_consumers[0]
        if post_mul.op_type != "Mul" or len(post_mul.input) != 2:
            continue

        scale_name = None
        scale_value = None
        for input_name in post_mul.input:
            if input_name == matmul.output[0]:
                continue
            candidate = scalar_initializer(input_name)
            if candidate is not None:
                scale_name = input_name
                scale_value = candidate
                break
        if scale_name is None:
            continue

        q_value, quantum = quantize_static_tensor(lhs_value)
        model.set_initializer(lhs_name, q_value)
        model.set_tensor_shape(lhs_name, list(q_value.shape))
        model.set_tensor_datatype(lhs_name, DataType["INT8"])
        model.set_tensor_datatype(matmul.output[0], DataType["INT32"])

        scale_array = np.asarray(model.get_initializer(scale_name), dtype=np.float32)
        model.set_initializer(scale_name, scale_array * np.float32(quantum))

        modified = True

    if modified:
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
    return model


def stage_dehw_siglip_io_postprocess(model, _cfg):
    """Keep SigLIP mask prep and output postprocess outside dataflow partitioning.

    Full SigLIP still has ONNX-only islands around attention-mask ``Where``, L2
    embedding normalization, and the final image/text similarity ``MatMul``. If
    FINN fpgadataflow elementwise/duplicate/shuffle nodes remain on both sides
    of those islands, generic partitioning sees a dataflow -> ONNX -> dataflow
    dependency and rejects the graph as a cycle. Convert only those small I/O
    islands back to standard ONNX ops so the single dataflow partition covers
    the quantized towers and attention body.
    """

    from onnx import helper
    from finn.transformation.fpgadataflow.convert_to_hw_layers import (
        InferElementwiseBinaryOperation,
    )
    from qonnx.transformation.general import RemoveUnusedTensors
    from qonnx.transformation.infer_datatypes import InferDataTypes
    from qonnx.transformation.infer_shapes import InferShapes
    from qonnx.transformation.remove import RemoveUnusedNodes

    binary_ops = {
        "ElementwiseAdd": "Add",
        "ElementwiseDiv": "Div",
        "ElementwiseMul": "Mul",
        "ElementwiseSub": "Sub",
    }
    fpgadataflow_passthrough = set(binary_ops) | {"DuplicateStreams", "Shuffle"}
    nondataflow_tail_passthrough = {
        "Cast",
        "Identity",
        "MatMul",
        "Reshape",
        "Transpose",
    }
    nondataflow_mask_passthrough = {
        "Cast",
        "Expand",
        "Identity",
        "Reshape",
        "Sub",
        "Unsqueeze",
    }

    def producer_map():
        return {output: node for node in model.graph.node for output in node.output}

    def consumer_map():
        consumers = {}
        for node in model.graph.node:
            for tensor_name in node.input:
                consumers.setdefault(tensor_name, []).append(node)
        return consumers

    def repair_late_elementwise_shapes() -> None:
        for node in model.graph.node:
            if node.op_type not in {"Add", "Div", "Mul", "Sub"} or len(node.output) != 1:
                continue
            out_shape = _static_shape(model.get_tensor_shape(node.output[0]))
            if out_shape is None:
                continue
            for tensor_name in node.input:
                if model.get_initializer(tensor_name) is not None:
                    continue
                in_shape = model.get_tensor_shape(tensor_name)
                if in_shape is None or len(in_shape) != len(out_shape):
                    continue
                repaired = [
                    out_dim if dim in (0, None, "") else dim
                    for dim, out_dim in zip(in_shape, out_shape)
                ]
                if repaired != in_shape and _static_shape(repaired) is not None:
                    model.set_tensor_shape(tensor_name, repaired)

    repair_late_elementwise_shapes()
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(InferElementwiseBinaryOperation(batch=True))

    def mark_downstream_from(tensor_names: list[str], marked: set[str]) -> None:
        consumers = consumer_map()
        queue = list(tensor_names)
        seen_tensors = set()
        while queue:
            tensor_name = queue.pop(0)
            if tensor_name in seen_tensors:
                continue
            seen_tensors.add(tensor_name)
            for node in consumers.get(tensor_name, []):
                if node.op_type in fpgadataflow_passthrough:
                    marked.add(node.name)
                    queue.extend(node.output)
                elif node.op_type in nondataflow_tail_passthrough:
                    queue.extend(node.output)

    def mark_mask_prep_upstream(node, marked: set[str]) -> None:
        producers = producer_map()
        queue = list(node.input)
        seen_tensors = set()
        while queue:
            tensor_name = queue.pop(0)
            if tensor_name in seen_tensors:
                continue
            seen_tensors.add(tensor_name)
            producer = producers.get(tensor_name)
            if producer is None:
                continue
            if producer.op_type in fpgadataflow_passthrough:
                marked.add(producer.name)
                queue.extend(producer.input)
            elif producer.op_type in nondataflow_mask_passthrough:
                queue.extend(producer.input)

    def clear_attributes(node) -> None:
        del node.attribute[:]

    def replace_tensor_uses(old_name: str, new_name: str, *, skip_node_name: str) -> None:
        for other in model.graph.node:
            if other.name == skip_node_name:
                continue
            for index, input_name in enumerate(other.input):
                if input_name == old_name:
                    other.input[index] = new_name

    def dehw_duplicate_streams(node) -> None:
        graph_outputs = {output.name for output in model.graph.output}
        input_name = node.input[0]
        insert_index = list(model.graph.node).index(node)
        identities = []
        for output_name in node.output:
            replace_tensor_uses(output_name, input_name, skip_node_name=node.name)
            if output_name in graph_outputs:
                identities.append(
                    helper.make_node(
                        "Identity",
                        [input_name],
                        [output_name],
                        name=f"{node.name}_{len(identities)}_identity",
                    )
                )
        for identity in identities:
            model.graph.node.insert(insert_index, identity)
            insert_index += 1
        model.graph.node.remove(node)

    def dehw_node(node) -> None:
        if node.op_type in binary_ops:
            node.op_type = binary_ops[node.op_type]
            node.domain = ""
            clear_attributes(node)
        elif node.op_type == "Shuffle":
            perm = _attr_value(node, "perm")
            node.op_type = "Transpose"
            node.domain = ""
            clear_attributes(node)
            if perm is not None:
                node.attribute.append(helper.make_attribute("perm", list(perm)))
        elif node.op_type == "DuplicateStreams":
            dehw_duplicate_streams(node)

    marked: set[str] = set()
    for node in list(model.graph.node):
        if node.op_type == "Where":
            mark_mask_prep_upstream(node, marked)
        if node.op_type in {"ReduceL2", "MatMul"}:
            mark_downstream_from(list(node.output), marked)

    if not marked:
        return model

    for node in list(model.graph.node):
        if node.name in marked:
            dehw_node(node)

    model = model.transform(RemoveUnusedNodes())
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    return model


def stage_fold_transpose_into_quant_init(model, _cfg):
    from qonnx.transformation.quant_constant_folding import FoldTransposeIntoQuantInit

    return model.transform(FoldTransposeIntoQuantInit())


def stage_remove_unused_tensors(model, _cfg):
    from qonnx.transformation.general import RemoveUnusedTensors

    return model.transform(RemoveUnusedTensors())


def stage_remove_static_graph_inputs(model, _cfg):
    from qonnx.transformation.general import RemoveStaticGraphInputs

    return model.transform(RemoveStaticGraphInputs())


def stage_give_unique_node_names(model, _cfg):
    from qonnx.transformation.general import GiveUniqueNodeNames

    return model.transform(GiveUniqueNodeNames())


def stage_give_readable_tensor_names(model, _cfg):
    from qonnx.transformation.general import GiveReadableTensorNames

    return model.transform(GiveReadableTensorNames())


def stage_convert_qonnx_to_finn(model, cfg):
    from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
    from finn.transformation.qonnx.quant_act_to_multithreshold import (
        default_filter_function_generator,
    )

    if not has_qonnx_nodes(model):
        return model
    return model.transform(
        ConvertQONNXtoFINN(
            filter_function=default_filter_function_generator(
                max_multithreshold_bit_width=cfg.max_multithreshold_bit_width
            )
        )
    )


def stage_convert_qonnx_to_finn_skip_conv_bias(model, cfg):
    for stage_fn in (
        stage_gemm_to_matmul,
        stage_fold_transpose_into_quant_init,
        stage_extract_conv_bias_safe,
        stage_infer_datatypes,
        stage_fold_quant_weights,
        stage_infer_shapes,
        stage_convert_quant_act_to_multithreshold,
        stage_infer_datatypes,
        stage_avgpool_trunc_to_quantavgpool,
        stage_remove_identity_ops,
    ):
        model = stage_fn(model, cfg)
    return model


def _float_attr(node, name: str, default: float) -> float:
    import onnx

    for attr in node.attribute:
        if attr.name == name:
            return float(onnx.helper.get_attribute_value(attr))
    return default


def _int_attr(node, name: str, default: int) -> int:
    import onnx

    for attr in node.attribute:
        if attr.name == name:
            return int(onnx.helper.get_attribute_value(attr))
    return default


def _rank2_matmul_shape(model, left_name: str, right_name: str) -> list[int] | None:
    left_shape = model.get_tensor_shape(left_name)
    right_shape = model.get_tensor_shape(right_name)
    if left_shape is None or right_shape is None:
        return None
    if len(left_shape) != 2 or len(right_shape) != 2:
        return None
    return [left_shape[0], right_shape[1]]


def stage_gemm_to_matmul(model, _cfg):
    from onnx import TensorProto, helper
    from qonnx.core.datatype import DataType
    from qonnx.util.basic import copy_metadata_props

    graph = model.graph
    rewritten_nodes = []
    changed = False
    for node in graph.node:
        if node.op_type != "Gemm":
            rewritten_nodes.append(node)
            continue

        if len(node.input) != 3:
            raise ValueError(f"Unsupported Gemm without explicit bias input: {node.name}")
        if _int_attr(node, "transA", 0):
            raise ValueError(f"Unsupported Gemm transA=1 in {node.name}")
        if _float_attr(node, "alpha", 1.0) != 1.0 or _float_attr(node, "beta", 1.0) != 1.0:
            raise ValueError(f"Unsupported non-unity Gemm alpha/beta in {node.name}")

        lhs, rhs, bias = node.input
        if _int_attr(node, "transB", 0):
            transpose_out = model.make_new_valueinfo_name()
            rhs_shape = model.get_tensor_shape(rhs)
            transposed_shape = list(reversed(rhs_shape)) if rhs_shape is not None else None
            graph.value_info.append(
                helper.make_tensor_value_info(
                    transpose_out,
                    TensorProto.FLOAT,
                    transposed_shape,
                )
            )
            transpose_node = helper.make_node(
                "Transpose",
                [rhs],
                [transpose_out],
                name=f"{node.name}_TransposeB",
            )
            copy_metadata_props(node, transpose_node)
            rewritten_nodes.append(transpose_node)
            rhs_dtype = model.get_tensor_datatype(rhs)
            if rhs_dtype != DataType["FLOAT32"]:
                model.set_tensor_datatype(transpose_out, rhs_dtype)
            rhs = transpose_out

        matmul_out = model.make_new_valueinfo_name()
        graph.value_info.append(
            helper.make_tensor_value_info(
                matmul_out,
                TensorProto.FLOAT,
                _rank2_matmul_shape(model, lhs, rhs),
            )
        )
        matmul_node = helper.make_node(
            "MatMul",
            [lhs, rhs],
            [matmul_out],
            name=f"{node.name}_MatMul",
        )
        add_node = helper.make_node(
            "Add",
            [matmul_out, bias],
            list(node.output),
            name=f"{node.name}_Add",
        )
        copy_metadata_props(node, matmul_node)
        copy_metadata_props(node, add_node)
        rewritten_nodes.extend([matmul_node, add_node])
        changed = True

    if changed:
        graph.ClearField("node")
        graph.node.extend(rewritten_nodes)
    return model


def stage_infer_datatypes(model, _cfg):
    from qonnx.transformation.infer_datatypes import InferDataTypes

    return model.transform(InferDataTypes())


def stage_extract_conv_bias_safe(model, _cfg):
    import warnings

    from onnx import helper
    from qonnx.util.basic import copy_metadata_props

    graph = model.graph
    node_ind = 0
    for node in graph.node:
        node_ind += 1
        if node.op_type not in ("Conv", "ConvTranspose") or len(node.input) <= 2:
            continue

        bias_name = node.input[2]
        bias = model.get_initializer(bias_name)
        producer = None
        if bias is None:
            producer = model.find_producer(bias_name)
            if not (
                producer is not None
                and producer.op_type in ("Quant", "IntQuant", "BipolarQuant")
                and not model.find_direct_predecessors(producer)
            ):
                warnings.warn(f"Could not extract bias from node {node.name}")
                continue

        out_shape = model.get_tensor_shape(node.output[0])
        bias_shape = model.get_tensor_shape(bias_name)
        if out_shape is None or bias_shape is None:
            raise ValueError(f"Missing Conv output or bias shape for {node.name}")

        add_shape = [1] * len(out_shape)
        add_shape[1] = bias_shape[0]
        if bias is not None:
            model.set_initializer(bias_name, bias.reshape(add_shape))
        else:
            quant_param = model.get_initializer(producer.input[0])
            quant_scale = model.get_initializer(producer.input[1])
            quant_zpt = model.get_initializer(producer.input[2])
            model.set_initializer(producer.input[0], quant_param.reshape(add_shape))
            if quant_scale.shape not in ((), (1,)):
                model.set_initializer(producer.input[1], quant_scale.reshape(add_shape))
            if quant_zpt.shape not in ((), (1,)):
                model.set_initializer(producer.input[2], quant_zpt.reshape(add_shape))
            model.set_tensor_shape(producer.output[0], add_shape)

        conv_out = node.output[0]
        conv_out_value_info = model.get_tensor_valueinfo(conv_out)
        conv_out_elem_type = conv_out_value_info.type.tensor_type.elem_type
        act_add_tensor = helper.make_tensor_value_info(
            model.make_new_valueinfo_name(),
            conv_out_elem_type,
            out_shape,
        )
        graph.value_info.append(act_add_tensor)
        add_node = helper.make_node(
            "Add",
            [act_add_tensor.name, bias_name],
            [conv_out],
            name=f"{node.name}_BiasAdd",
        )
        copy_metadata_props(node, add_node)
        graph.node.insert(node_ind, add_node)
        node.output[0] = act_add_tensor.name
        node.input.remove(bias_name)
        return model

    return model


def stage_fold_quant_weights(model, _cfg):
    from finn.transformation.qonnx.fold_quant_weights import FoldQuantWeights

    return model.transform(FoldQuantWeights(infer_shapes=False))


def stage_convert_quant_act_to_multithreshold(model, cfg):
    from finn.transformation.qonnx.quant_act_to_multithreshold import (
        ConvertQuantActToMultiThreshold,
        default_filter_function_generator,
    )

    return model.transform(
        ConvertQuantActToMultiThreshold(
            filter_function=default_filter_function_generator(
                max_multithreshold_bit_width=cfg.max_multithreshold_bit_width
            )
        )
    )


def stage_avgpool_trunc_to_quantavgpool(model, _cfg):
    from finn.transformation.qonnx.infer_quant_avg_pool_2d import AvgPoolAndTruncToQuantAvgPool

    return model.transform(AvgPoolAndTruncToQuantAvgPool())


def stage_remove_identity_ops(model, _cfg):
    from qonnx.transformation.remove import RemoveIdentityOps

    return model.transform(RemoveIdentityOps())


def stage_tidy_up_no_fold_constants(model, _cfg):
    from qonnx.transformation.general import (
        GiveReadableTensorNames,
        GiveUniqueNodeNames,
        RemoveStaticGraphInputs,
    )
    from qonnx.transformation.infer_datatypes import InferDataTypes
    from qonnx.transformation.infer_shapes import InferShapes

    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())
    return model


def _apply_streamline_transform(model, transform):
    from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
    from qonnx.transformation.infer_datatypes import InferDataTypes
    from qonnx.transformation.remove import RemoveIdentityOps

    model = model.transform(transform)
    model = model.transform(RemoveIdentityOps())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    return model


def _make_stage(transform_factory):
    def _stage(model, _cfg):
        return model.transform(transform_factory())

    return _stage


def _make_streamline_stage(transform_factory):
    def _stage(model, _cfg):
        return _apply_streamline_transform(model, transform_factory())

    return _stage


def split_streamline_stages() -> list[tuple[str, Callable]]:
    from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
    from qonnx.transformation.general import ConvertDivToMul, ConvertSubToAdd, RemoveUnusedTensors
    from qonnx.transformation.infer_data_layouts import InferDataLayouts

    from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
    from finn.transformation.streamline.absorb import (
        Absorb1BitMulIntoConv,
        Absorb1BitMulIntoMatMul,
        AbsorbAddIntoMultiThreshold,
        AbsorbConsecutiveTransposes,
        AbsorbMulIntoMultiThreshold,
        AbsorbScalarMulAddIntoTopK,
        AbsorbSignBiasIntoMultiThreshold,
        AbsorbTransposeIntoMultiThreshold,
        FactorOutMulSignMagnitude,
    )
    from finn.transformation.streamline.collapse_repeated import (
        CollapseRepeatedAdd,
        CollapseRepeatedMul,
    )
    from finn.transformation.streamline.reorder import (
        MoveAddPastConv,
        MoveAddPastMul,
        MoveMulPastMaxPool,
        MoveScalarAddPastMatMul,
        MoveScalarLinearPastInvariants,
        MoveScalarMulPastConv,
        MoveScalarMulPastMatMul,
    )
    from finn.transformation.streamline.sign_to_thres import ConvertSignToThres
    from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
    from finn.transformation.streamline.reorder import MakeMaxPoolNHWC

    streamline_transforms = [
        ("convert_sub_to_add", ConvertSubToAdd),
        ("convert_div_to_mul", ConvertDivToMul),
        ("batchnorm_to_affine", BatchNormToAffine),
        ("convert_sign_to_thres", ConvertSignToThres),
        ("move_mul_past_maxpool", MoveMulPastMaxPool),
        ("absorb_sign_bias_into_multithreshold", AbsorbSignBiasIntoMultiThreshold),
        ("move_scalar_linear_past_invariants", MoveScalarLinearPastInvariants),
        ("move_add_past_mul_0", MoveAddPastMul),
        ("move_scalar_add_past_matmul", MoveScalarAddPastMatMul),
        ("move_add_past_conv", MoveAddPastConv),
        ("move_scalar_mul_past_matmul", MoveScalarMulPastMatMul),
        ("move_scalar_mul_past_conv", MoveScalarMulPastConv),
        ("move_add_past_mul_1", MoveAddPastMul),
        ("collapse_repeated_add", CollapseRepeatedAdd),
        ("collapse_repeated_mul", CollapseRepeatedMul),
        ("move_mul_past_maxpool_1", MoveMulPastMaxPool),
        ("absorb_add_into_multithreshold", AbsorbAddIntoMultiThreshold),
        ("factor_out_mul_sign_magnitude", FactorOutMulSignMagnitude),
        ("absorb_mul_into_multithreshold", AbsorbMulIntoMultiThreshold),
        ("absorb_1bit_mul_into_matmul", Absorb1BitMulIntoMatMul),
        ("absorb_1bit_mul_into_conv", Absorb1BitMulIntoConv),
    ]

    stages: list[tuple[str, Callable]] = [
        ("streamline_absorb_sign_bias_pre", _make_stage(AbsorbSignBiasIntoMultiThreshold))
    ]
    for prefix in ("streamline0", "streamline1"):
        for name, transform in streamline_transforms:
            stages.append((f"{prefix}_{name}", _make_streamline_stage(transform)))
        if prefix == "streamline0":
            stages.extend(
                [
                    ("streamline_lower_convs_to_matmul", _make_stage(LowerConvsToMatMul)),
                    ("streamline_make_maxpool_nhwc_0", _make_stage(MakeMaxPoolNHWC)),
                    (
                        "streamline_absorb_transpose_into_multithreshold",
                        _make_stage(AbsorbTransposeIntoMultiThreshold),
                    ),
                    ("streamline_make_maxpool_nhwc_1", _make_stage(MakeMaxPoolNHWC)),
                    (
                        "streamline_absorb_consecutive_transposes",
                        _make_stage(AbsorbConsecutiveTransposes),
                    ),
                    (
                        "streamline_convert_bipolar_matmul",
                        _make_stage(ConvertBipolarMatMulToXnorPopcount),
                    ),
                ]
            )
    stages.extend(
        [
            ("streamline_absorb_scalar_mul_add_topk", _make_stage(AbsorbScalarMulAddIntoTopK)),
            ("streamline_infer_data_layouts", _make_stage(InferDataLayouts)),
            ("streamline_remove_unused_tensors", _make_stage(RemoveUnusedTensors)),
        ]
    )
    return stages


def _has_any_op(model, op_types: list[str]) -> bool:
    return any(len(model.get_nodes_by_op_type(op_type)) > 0 for op_type in op_types)


def _make_hw_stage(op_types: list[str], transform_factory: Callable):
    def _stage(model, cfg):
        if not _has_any_op(model, op_types):
            return model
        return model.transform(transform_factory(cfg))

    return _stage


def _make_hw_always_stage(transform_factory: Callable):
    def _stage(model, cfg):
        return model.transform(transform_factory(cfg))

    return _stage


def split_convert_to_hw_stages() -> list[tuple[str, Callable]]:
    from finn.transformation.fpgadataflow import convert_to_hw_layers as to_hw
    from finn.transformation.fpgadataflow.absorb_into_requant import (
        AbsorbElementwiseOpsIntoRequant,
    )
    from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
    from finn.transformation.streamline import absorb
    from qonnx.transformation.general import GiveUniqueNodeNames
    from qonnx.transformation.infer_data_layouts import InferDataLayouts

    return [
        (
            "convert_to_hw_requant_0",
            _make_hw_stage(
                ["MultiThreshold", "Quant"],
                lambda cfg: to_hw.InferRequantLayer(
                    bitwidth_threshold=cfg.requant_bitwidth_threshold
                ),
            ),
        ),
        (
            "convert_to_hw_threshold_0",
            _make_hw_stage(["MultiThreshold"], lambda _cfg: to_hw.InferThresholdingLayer()),
        ),
        ("convert_to_hw_absorb_pre_matmul_dequant", stage_absorb_pre_matmul_dequant),
        ("convert_to_hw_integerize_static_lhs_matmul", stage_integerize_static_lhs_matmul),
        (
            "convert_to_hw_binary_matmul",
            _make_hw_stage(
                ["XnorPopcountMatMul"], lambda _cfg: to_hw.InferBinaryMatrixVectorActivation()
            ),
        ),
        (
            "convert_to_hw_quantized_matmul",
            _make_hw_stage(["MatMul"], lambda _cfg: to_hw.InferQuantizedMatrixVectorActivation()),
        ),
        (
            "convert_to_hw_vector_vector",
            _make_hw_stage(["MatMul"], lambda _cfg: to_hw.InferVectorVectorActivation()),
        ),
        (
            "convert_to_hw_label_select",
            _make_hw_stage(["TopK"], lambda _cfg: to_hw.InferLabelSelectLayer()),
        ),
        (
            "convert_to_hw_requant_1",
            _make_hw_stage(
                ["MultiThreshold", "Quant"],
                lambda cfg: to_hw.InferRequantLayer(
                    bitwidth_threshold=cfg.requant_bitwidth_threshold
                ),
            ),
        ),
        (
            "convert_to_hw_threshold_1",
            _make_hw_stage(["MultiThreshold"], lambda _cfg: to_hw.InferThresholdingLayer()),
        ),
        (
            "convert_to_hw_pool",
            _make_hw_stage(
                ["MaxPool", "AveragePool", "MaxPoolNHWC", "QuantAvgPool2d"],
                lambda _cfg: to_hw.InferPool(),
            ),
        ),
        (
            "convert_to_hw_conv_input_gen",
            _make_hw_stage(["Im2Col"], lambda _cfg: to_hw.InferConvInpGen()),
        ),
        (
            "convert_to_hw_flatten",
            _make_hw_stage(["ConvolutionInputGenerator"], lambda _cfg: RemoveCNVtoFCFlatten()),
        ),
        (
            "convert_to_hw_cls_token",
            _make_hw_stage(["Concat"], lambda _cfg: to_hw.InferAddCLSTokenLayer()),
        ),
        (
            "convert_to_hw_concat",
            _make_hw_stage(["Concat"], lambda _cfg: to_hw.InferConcatLayer()),
        ),
        (
            "convert_to_hw_split",
            _make_hw_stage(["Split"], lambda _cfg: to_hw.InferSplitLayer()),
        ),
        ("convert_to_hw_repair_static_shapes", stage_repair_static_shapes),
        ("convert_to_hw_fold_constants", stage_fold_constants_repair_static_shapes),
        (
            "convert_to_hw_elementwise_binary",
            _make_hw_stage(
                [
                    "Mul",
                    "Div",
                    "Sub",
                    "Add",
                    "And",
                    "Or",
                    "Xor",
                    "Equal",
                    "Less",
                    "LessOrEqual",
                    "Greater",
                    "GreaterOrEqual",
                ],
                lambda _cfg: to_hw.InferElementwiseBinaryOperation(batch=True),
            ),
        ),
        (
            "convert_to_hw_where",
            _make_hw_stage(["Where"], lambda _cfg: to_hw.InferWhereLayer()),
        ),
        (
            "convert_to_hw_relu_max",
            _make_hw_stage(["Relu"], lambda _cfg: to_hw.InferReLUAsElementwiseMax()),
        ),
        (
            "convert_to_hw_upsample",
            _make_hw_stage(["Upsample"], lambda _cfg: to_hw.InferUpsample()),
        ),
        (
            "convert_to_hw_global_pooling",
            _make_hw_stage(["GlobalAveragePool"], lambda _cfg: to_hw.InferGlobalAccPoolLayer()),
        ),
        (
            "convert_to_hw_select_token",
            _make_hw_stage(["Gather"], lambda _cfg: to_hw.InferSelectTokenLayer()),
        ),
        (
            "convert_to_hw_lookup",
            _make_hw_stage(["Gather"], lambda _cfg: to_hw.InferLookupLayer()),
        ),
        (
            "convert_to_hw_softmax",
            _make_hw_stage(["Softmax"], lambda _cfg: to_hw.InferHWSoftmax()),
        ),
        (
            "convert_to_hw_pwpolyf",
            _make_hw_stage(
                ["PWPolyF", "Gelu", "Sigmoid", "Tanh", "Erf"],
                lambda _cfg: to_hw.InferPWPolyFLayer(),
            ),
        ),
        (
            "convert_to_hw_layernorm",
            _make_hw_stage(["LayerNormalization"], lambda _cfg: to_hw.InferLayerNorm()),
        ),
        (
            "convert_to_hw_crop",
            _make_hw_stage(["Crop"], lambda _cfg: to_hw.InferCrop()),
        ),
        (
            "convert_to_hw_duplicate_streams",
            _make_hw_always_stage(lambda _cfg: to_hw.InferDuplicateStreamsLayer()),
        ),
        (
            "convert_to_hw_absorb_elementwise_into_requant",
            _make_hw_stage(["Requant"], lambda _cfg: AbsorbElementwiseOpsIntoRequant()),
        ),
        (
            "convert_to_hw_absorb_consecutive_transposes",
            _make_hw_always_stage(lambda _cfg: absorb.AbsorbConsecutiveTransposes()),
        ),
        (
            "convert_to_hw_give_unique_node_names",
            _make_hw_always_stage(lambda _cfg: GiveUniqueNodeNames()),
        ),
        (
            "convert_to_hw_infer_data_layouts",
            _make_hw_always_stage(lambda _cfg: InferDataLayouts()),
        ),
        (
            "convert_to_hw_shuffle",
            _make_hw_stage(
                ["Transpose"],
                lambda cfg: (
                    to_hw.InferShuffle()
                    if cfg.infer_shuffle_skip_first
                    else to_hw.InferShuffle(_filter=lambda *_: True)
                ),
            ),
        ),
    ]


def stage_specialize_layers(
    model,
    cfg,
    output_dir: Path,
    *,
    mvau_impl_style: str = "rtl",
    mvau_hls_nodes: list[str] | None = None,
):
    import finn.builder.build_dataflow_steps as steps
    from finn.transformation.general import ApplyConfig

    config_path = output_dir / "specialize_layers_config.json"
    write_rtl_specialization_config(
        config_path,
        mvau_impl_style=mvau_impl_style,
        mvau_hls_nodes=mvau_hls_nodes,
    )
    model = model.transform(ApplyConfig(str(config_path)))
    return steps.step_specialize_layers(model, cfg)


def make_stages(
    output_dir: Path,
    *,
    skip_conv_bias_extract: bool,
    skip_fold_constants: bool,
    split_streamline: bool,
    split_convert_to_hw: bool,
) -> list[tuple[str, Callable]]:
    import finn.builder.build_dataflow_steps as steps

    stages = [
        ("qonnx_domain_fix", stage_qonnx_domain_fix),
        ("infer_shapes", stage_infer_shapes),
        ("give_unique_parameter_tensors", stage_give_unique_parameter_tensors),
        ("fold_constants", stage_fold_constants),
        ("fold_transpose_into_quant_init", stage_fold_transpose_into_quant_init),
        ("remove_unused_tensors", stage_remove_unused_tensors),
        ("remove_static_graph_inputs", stage_remove_static_graph_inputs),
        ("give_unique_node_names", stage_give_unique_node_names),
        ("give_readable_tensor_names", stage_give_readable_tensor_names),
    ]
    if skip_conv_bias_extract:
        stages.extend(
            [
                ("gemm_to_matmul", stage_gemm_to_matmul),
                ("fold_transpose_into_quant_init", stage_fold_transpose_into_quant_init),
                ("extract_conv_bias_safe", stage_extract_conv_bias_safe),
                ("infer_datatypes_before_quant_weight_fold", stage_infer_datatypes),
                ("fold_quant_weights", stage_fold_quant_weights),
                ("infer_shapes_after_quant_weight_fold", stage_infer_shapes),
                (
                    "convert_quant_act_to_multithreshold",
                    stage_convert_quant_act_to_multithreshold,
                ),
                ("infer_datatypes_after_quant_act", stage_infer_datatypes),
                ("avgpool_trunc_to_quantavgpool", stage_avgpool_trunc_to_quantavgpool),
                ("remove_identity_ops_after_qonnx", stage_remove_identity_ops),
            ]
        )
    else:
        stages.append(("convert_qonnx_to_finn", stage_convert_qonnx_to_finn))
    tidy_stage = (
        ("tidy_up_no_fold_constants", stage_tidy_up_no_fold_constants)
        if skip_fold_constants
        else ("tidy_up", steps.step_tidy_up)
    )
    stages.append(tidy_stage)
    if split_streamline:
        stages.extend(split_streamline_stages())
    else:
        stages.append(("streamline", steps.step_streamline))
    stages.extend(
        [
            ("extract_norm_scale_bias", stage_extract_norm_scale_bias),
            ("repair_static_shapes", stage_repair_static_shapes),
        ]
    )
    if split_convert_to_hw:
        stages.extend(split_convert_to_hw_stages())
    else:
        stages.append(("convert_to_hw", steps.step_convert_to_hw))
    stages.extend(
        [
            ("dehw_siglip_io_postprocess", stage_dehw_siglip_io_postprocess),
            ("dataflow_partition", steps.step_create_dataflow_partition),
            (
                "specialize_layers",
                lambda model, cfg: stage_specialize_layers(model, cfg, output_dir),
            ),
        ]
    )
    return stages


def run_probe(args: argparse.Namespace) -> dict:
    from qonnx.core.modelwrapper import ModelWrapper

    input_path = repo_path(args.input)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_cfg(args, output_dir)

    model = ModelWrapper(str(input_path))
    report = {
        "input": str(input_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "board": args.board,
        "clock_ns": args.clock_ns,
        "target_fps": args.target_fps,
        "stages": [],
    }
    write_json(output_dir / "00_input_summary.json", summarize_model(model.model))
    checkpoint_stages = set(args.checkpoint_stage or [])
    save_checkpoint(model, output_dir, "00_input", args.save_checkpoints)

    stop_after = args.stop_after
    start_at = args.start_at
    for stage_name, stage_fn in make_stages(
        output_dir,
        skip_conv_bias_extract=args.skip_conv_bias_extract,
        skip_fold_constants=args.skip_fold_constants,
        split_streamline=args.split_streamline,
        split_convert_to_hw=args.split_convert_to_hw,
    ):
        if start_at is not None and stage_name != start_at:
            report["stages"].append({"stage": stage_name, "status": "skipped_start_at"})
            continue
        start_at = None

        if args.skip_streamline and stage_name == "streamline":
            report["stages"].append({"stage": stage_name, "status": "skipped"})
            continue
        if args.skip_fold_constants and stage_name == "fold_constants":
            report["stages"].append({"stage": stage_name, "status": "skipped"})
            continue

        started = time.time()
        item = {"stage": stage_name}
        try:
            with stage_timeout(args.stage_timeout_seconds, stage_name):
                model = stage_fn(model, cfg)
            checkpoint = save_checkpoint(
                model,
                output_dir,
                stage_name,
                args.save_checkpoints or stage_name in checkpoint_stages,
            )
            item.update(
                {
                    "status": "ok",
                    "seconds": time.time() - started,
                    "summary": stage_summary(model),
                }
            )
            if checkpoint is not None:
                item["checkpoint"] = str(checkpoint.resolve())
        except Exception as exc:
            item.update(
                {
                    "status": "failed",
                    "seconds": time.time() - started,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            report["stages"].append(item)
            report["final_status"] = "failed"
            write_json(output_dir / "finn_probe_report.json", report)
            if args.allow_failure:
                return report
            raise

        report["stages"].append(item)
        write_json(output_dir / "finn_probe_report.json", report)
        if stop_after == stage_name:
            report["final_status"] = "stopped"
            write_json(output_dir / "finn_probe_report.json", report)
            return report

    if start_at is not None:
        report["final_status"] = "failed"
        report["error"] = f"start stage {start_at} was not present in this stage plan"
        write_json(output_dir / "finn_probe_report.json", report)
        if args.allow_failure:
            return report
        raise ValueError(report["error"])

    report["final_status"] = "ok"
    write_json(output_dir / "finn_probe_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_QONNX.relative_to(repo_path("."))))
    parser.add_argument("--output-dir", default="full_siglip/build/finn_probe")
    parser.add_argument("--board", default=DEFAULT_BOARD)
    parser.add_argument("--clock-ns", type=float, default=DEFAULT_CLOCK_NS)
    parser.add_argument("--target-fps", type=int, default=DEFAULT_TARGET_FPS)
    parser.add_argument("--stop-after", choices=STAGE_NAMES, default=None)
    parser.add_argument("--start-at", choices=STAGE_NAMES, default=None)
    parser.add_argument("--checkpoint-stage", action="append", choices=STAGE_NAMES)
    parser.add_argument("--skip-streamline", action="store_true")
    parser.add_argument(
        "--split-streamline",
        action="store_true",
        help="Run FINN streamlining as named substages to isolate full-graph bottlenecks.",
    )
    parser.add_argument(
        "--split-convert-to-hw",
        action="store_true",
        help="Run FINN hardware conversion as named substages to isolate full-graph bottlenecks.",
    )
    parser.add_argument("--skip-fold-constants", action="store_true")
    parser.add_argument(
        "--skip-conv-bias-extract",
        action="store_true",
        help="Probe past QONNX ExtractBiasFromConv failures without editing dependencies.",
    )
    parser.add_argument(
        "--no-save-checkpoints",
        dest="save_checkpoints",
        action="store_false",
        help="Write reports only; useful for fine-grained timing probes on large graphs.",
    )
    parser.add_argument(
        "--stage-timeout-seconds",
        type=int,
        default=0,
        help="Timeout for each FINN transform stage; 0 disables the timeout.",
    )
    parser.add_argument(
        "--allow-failure",
        action="store_true",
        help="Write the failure report and exit zero for exploratory probes.",
    )
    parser.set_defaults(save_checkpoints=True)
    args = parser.parse_args()

    report = run_probe(args)
    report_path = repo_path(args.output_dir) / "finn_probe_report.json"
    print(f"Wrote {report_path}")
    print(f"final_status={report['final_status']}")
    for item in report["stages"]:
        print(f"{item['stage']}: {item['status']}")
        if item["status"] == "failed":
            print(f"  {item['error_type']}: {item['error']}")


if __name__ == "__main__":
    main()
