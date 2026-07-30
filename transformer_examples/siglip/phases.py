# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""SigLIP-specific overrides for the phase-based FINN pipeline."""

from __future__ import annotations

import numpy as np
import warnings
from copy import deepcopy
from json import load
from onnx import AttributeProto
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import (
    GiveUniqueParameterTensors,
    RemoveUnusedTensors,
    SortGraph,
)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.remove import RemoveIdentityOps, RemoveUnusedNodes
from qonnx.util.cleanup import cleanup_model

from finn.builder.build_dataflow_config import VerificationStepType
from finn.builder.build_dataflow_phases import phase_optimize_model
from finn.builder.build_dataflow_steps import step_tidy_up, verify_step
from finn.transformation.general import ApplyConfig
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.streamline.extract_norm_scale_bias import ExtractNormScaleBias


class _DuplicateSafeModelWrapper(ModelWrapper):
    """Rename every occurrence when a tensor is used twice by one node.

    QONNX ``ModelWrapper.rename_tensor`` currently updates only the first
    matching input and output on each node. SigLIP's exact GELU contains
    ``Mul(x, x)`` nodes, so ordinary readable-name cleanup otherwise leaves a
    dangling copy of the old tensor name.
    """

    def rename_tensor(self, old_name: str, new_name: str) -> None:
        super().rename_tensor(old_name, new_name)
        for node in self.graph.node:
            for index, input_name in enumerate(node.input):
                if input_name == old_name:
                    node.input[index] = new_name
            for index, output_name in enumerate(node.output):
                if output_name == old_name:
                    node.output[index] = new_name

    def transform(self, transformation, *args, **kwargs):
        model = super().transform(transformation, *args, **kwargs)
        if not isinstance(model, _DuplicateSafeModelWrapper):
            model = _DuplicateSafeModelWrapper(model.model, fix_missing_initializer_valueinfo=False)
        return model


def _select_graph_output(model: ModelWrapper, output_name: str) -> ModelWrapper:
    matches = [value_info for value_info in model.graph.output if value_info.name == output_name]
    if len(matches) != 1:
        available = [value_info.name for value_info in model.graph.output]
        raise ValueError(f"Expected output {output_name!r}; available outputs are {available}")
    selected = deepcopy(matches[0])
    model.graph.ClearField("output")
    model.graph.output.append(selected)
    return model.transform(RemoveUnusedNodes())


def _siglip_quant_filter(max_bit_width: int):
    """Convert static QV activation quantizers within the configured width."""

    def filter_function(model, quant_node):
        if quant_node.op_type == "BipolarQuant":
            return True
        bit_width = model.get_initializer(quant_node.input[3])
        if bit_width is None:
            raise ValueError("Quant nodes must have a static bit width")
        if float(np.asarray(bit_width).reshape(-1)[0]) > max_bit_width:
            return False
        return model.get_initializer(quant_node.input[2]) is not None

    return filter_function


def phase_prepare_siglip(model, cfg):
    """Prepare QV-LSQ QONNX and convert eligible activation quantizers."""

    if not isinstance(model, _DuplicateSafeModelWrapper):
        model = _DuplicateSafeModelWrapper(model.model, fix_missing_initializer_valueinfo=False)
    model = _select_graph_output(model, "image_embeds")

    qonnx_count = sum(
        len(model.get_nodes_by_op_type(op_type)) for op_type in ("BinaryQuant", "Quant", "Trunc")
    )
    if qonnx_count:
        model = cleanup_model(model)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The Quant node with name: .* was not converted to a MultiThreshold.*",
            )
            model = model.transform(
                ConvertQONNXtoFINN(
                    filter_function=_siglip_quant_filter(cfg.max_multithreshold_bit_width)
                )
            )
        if VerificationStepType.QONNX_TO_FINN_PYTHON in cfg._resolve_verification_steps():
            verify_step(model, cfg, "finn_onnx_python", need_parent=False)
    return step_tidy_up(model, cfg)


def _extract_layernorm_affine(model: ModelWrapper) -> ModelWrapper:
    model = model.transform(ExtractNormScaleBias(), cleanup=False)
    model = model.transform(SortGraph())
    model = model.transform(RemoveIdentityOps())
    model = model.transform(GiveUniqueParameterTensors())
    model = model.transform(InferShapes())
    return model.transform(InferDataTypes())


def _tensor_datatype(model: ModelWrapper, tensor_name: str):
    try:
        return model.get_tensor_datatype(tensor_name)
    except Exception:
        return None


def _scalar_initializer(model: ModelWrapper, tensor_name: str):
    value = model.get_initializer(tensor_name)
    if value is None or value.size != 1:
        return None
    return value.reshape(-1)[0]


def _absorb_pre_matmul_dequant(model: ModelWrapper) -> ModelWrapper:
    """Move scalar activation dequantization after static-weight MatMul."""

    producers = {output: node for node in model.graph.node for output in node.output}
    consumers = {}
    for node in model.graph.node:
        for tensor_name in node.input:
            consumers.setdefault(tensor_name, []).append(node)

    modified = False
    for matmul in model.get_nodes_by_op_type("MatMul"):
        weight_name = matmul.input[1]
        weight_dtype = _tensor_datatype(model, weight_name)
        if model.get_initializer(weight_name) is None:
            continue
        if weight_dtype is None or not weight_dtype.is_integer():
            continue

        pre_mul = producers.get(matmul.input[0])
        if pre_mul is None or pre_mul.op_type != "Mul":
            continue
        int_input = None
        pre_scale = None
        for index, input_name in enumerate(pre_mul.input):
            input_dtype = _tensor_datatype(model, input_name)
            candidate_scale = _scalar_initializer(model, pre_mul.input[1 - index])
            if input_dtype is not None and input_dtype.is_integer() and candidate_scale is not None:
                int_input = input_name
                pre_scale = candidate_scale
                break
        if int_input is None:
            continue

        post_consumers = consumers.get(matmul.output[0], [])
        if len(post_consumers) != 1 or post_consumers[0].op_type != "Mul":
            continue
        post_mul = post_consumers[0]
        post_scale_name = next(
            (
                input_name
                for input_name in post_mul.input
                if input_name != matmul.output[0] and model.get_initializer(input_name) is not None
            ),
            None,
        )
        if post_scale_name is None:
            continue

        post_scale = model.get_initializer(post_scale_name)
        model.set_initializer(post_scale_name, post_scale * pre_scale)
        matmul.input[0] = int_input
        model.set_tensor_datatype(matmul.output[0], DataType["INT32"])
        modified = True

    if modified:
        model = model.transform(RemoveUnusedNodes())
        model = model.transform(RemoveUnusedTensors())
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
    return model


def _integerize_static_lhs_matmul(model: ModelWrapper) -> ModelWrapper:
    """Integerize exactly quantized static attention-pool queries."""

    consumers = {}
    for node in model.graph.node:
        for tensor_name in node.input:
            consumers.setdefault(tensor_name, []).append(node)

    modified = False
    for matmul in model.get_nodes_by_op_type("MatMul"):
        lhs_name, rhs_name = matmul.input
        lhs_value = model.get_initializer(lhs_name)
        lhs_dtype = _tensor_datatype(model, lhs_name)
        rhs_dtype = _tensor_datatype(model, rhs_name)
        if lhs_value is None or model.get_initializer(rhs_name) is not None:
            continue
        if rhs_dtype is None or not rhs_dtype.is_integer():
            continue
        if lhs_dtype is not None and lhs_dtype.is_integer():
            continue

        post_consumers = consumers.get(matmul.output[0], [])
        if len(post_consumers) != 1 or post_consumers[0].op_type != "Mul":
            continue
        post_mul = post_consumers[0]
        scale_name = next(
            (
                input_name
                for input_name in post_mul.input
                if input_name != matmul.output[0]
                and _scalar_initializer(model, input_name) is not None
            ),
            None,
        )
        if scale_name is None:
            continue

        unique_values = np.sort(np.unique(lhs_value))
        max_abs = np.float32(np.max(np.abs(unique_values)))
        level_candidates = max_abs / np.arange(1, 128, dtype=np.float32)
        nonzero_candidates = np.concatenate(
            [np.abs(unique_values), np.diff(unique_values), level_candidates]
        )
        nonzero_candidates = nonzero_candidates[nonzero_candidates > 1e-12]
        if nonzero_candidates.size == 0:
            quantum = np.float32(1.0)
            integer_value = np.zeros_like(lhs_value, dtype=np.float32)
        else:
            quantum = None
            integer_value = None
            for candidate in np.sort(np.unique(nonzero_candidates)):
                candidate = np.float32(candidate)
                candidate_integer = np.round(lhs_value / candidate).astype(np.float32)
                if np.min(candidate_integer) < -128 or np.max(candidate_integer) > 127:
                    continue
                if np.array_equal(candidate_integer * candidate, lhs_value):
                    quantum = candidate
                    integer_value = candidate_integer
                    break
        if quantum is None:
            continue

        model.set_initializer(lhs_name, integer_value)
        model.set_tensor_datatype(lhs_name, DataType["INT8"])
        model.set_tensor_datatype(matmul.output[0], DataType["INT32"])
        scale = model.get_initializer(scale_name)
        model.set_initializer(scale_name, scale * quantum)
        modified = True

    if modified:
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
    return model


def _folding_config_nodes(graph, hierarchy=None):
    """Yield the config key and node pairs used by ``ApplyConfig``."""

    for node in graph.node:
        config_key = node.name if hierarchy is None else f"{hierarchy}_{node.name}"
        if node.domain.startswith("finn.custom_op.fpgadataflow"):
            yield config_key, node
        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                nested_hierarchy = f"{config_key}_{attribute.name}"
                yield from _folding_config_nodes(attribute.g, nested_hierarchy)


def make_siglip_folding_step(config_path, require_complete: bool):
    """Apply the entries which exist at the current shuffle-decomposition stage."""

    def step_apply_siglip_folding(model, cfg):
        with open(config_path, encoding="utf-8") as config_file:
            full_config = load(config_file)

        configurable_nodes = dict(_folding_config_nodes(model.graph))
        configured_keys = set(full_config) - {"Defaults"}
        current_keys = set(configurable_nodes)
        if require_complete:
            missing = {
                key
                for key in current_keys - configured_keys
                if configurable_nodes[key].op_type != "FINNLoop"
            }
            unused = configured_keys - current_keys
            if missing or unused:
                raise RuntimeError(
                    "SigLIP folding profile does not match the decomposed graph: "
                    f"missing={sorted(missing)}, unused={sorted(unused)}"
                )

        stage_config = {"Defaults": full_config.get("Defaults", {})}
        for key, node in configurable_nodes.items():
            # Shuffles do not have their final names before decomposition, and
            # FINNLoop itself has no folding parameters. A no-op backend entry
            # lets ApplyConfig verify every current custom node without warnings.
            stage_config[key] = full_config.get(key, {"backend": "fpgadataflow"})

        return model.transform(
            ApplyConfig(
                stage_config,
                node_filter=lambda node: node.domain.startswith("finn.custom_op.fpgadataflow"),
            )
        )

    suffix = "final" if require_complete else "pre_decomposition"
    step_apply_siglip_folding.__name__ = f"step_apply_siglip_folding_{suffix}"
    return step_apply_siglip_folding


def phase_optimize_siglip(model, cfg):
    """Streamline SigLIP and expose its quantized MatMuls and LayerNorms."""

    model = phase_optimize_model(model, cfg)
    model = _extract_layernorm_affine(model)
    model = _absorb_pre_matmul_dequant(model)
    model = _integerize_static_lhs_matmul(model)
    if VerificationStepType.STREAMLINED_PYTHON in cfg._resolve_verification_steps():
        verify_step(model, cfg, "siglip_optimized_python", need_parent=False)
    return model
