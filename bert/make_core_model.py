#!/usr/bin/env python3
"""Create an initialized FINN-ready BERT safety accelerator core.

The generated graph is intentionally initialized, not trained. It is a repeatable
hardware bring-up target with BERT-like repeated encoder blocks:

  AddCLSToken -> repeated quantized projection/FFN residual blocks -> SelectToken -> head

The host-side training/export script produces the eventual safety weights. This
core keeps the FINN deployment path live while those weights are trained.
"""

from __future__ import annotations

import argparse
import hashlib
import numpy as np
from collections import Counter
from onnx import StringStringEntryProto, TensorProto, helper, numpy_helper
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes

from bert.common import (
    CORE_PRESETS,
    DEFAULT_BUILD_DIR,
    DEFAULT_FPGA_PART,
    CorePreset,
    get_preset,
    repo_path,
    write_json,
)
from bert.weight_import import apply_imported_weights
from finn.transformation.fpgadataflow.loop_rolling import LoopExtraction, LoopRolling
from finn.transformation.fpgadataflow.set_loop_boundary import SetLoopBoundary
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

ACT_DTYPE = DataType["UINT8"]
WEIGHT_DTYPE = DataType["INT8"]
ACC_DTYPE = DataType["INT32"]
RESIDUAL_SUM_DTYPE = DataType["UINT9"]


def _vi(name: str, shape: list[int]):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _initial_weight(name: str, in_ch: int, out_ch: int):
    seed = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    values = rng.integers(-2, 3, size=(in_ch, out_ch), endpoint=False)
    return numpy_helper.from_array(values.astype(np.float32), name=name)


def _requant_params(prefix: str, channels: int, scale_value: float = 1.0, bias_value: float = 0.0):
    scale = numpy_helper.from_array(
        np.full((channels,), scale_value, dtype=np.float32), name=f"{prefix}_scale"
    )
    bias = numpy_helper.from_array(
        np.full((channels,), bias_value, dtype=np.float32), name=f"{prefix}_bias"
    )
    return scale, bias


def _add_metadata(node, block_idx: int | None) -> None:
    if block_idx is None:
        scope = "bert.boundary"
        klass = "Boundary"
    else:
        scope = f"bert.encoder.layer.{block_idx}"
        klass = "BertLayer"
    node.metadata_props.append(
        StringStringEntryProto(key="pkg.torch.onnx.name_scopes", value=f"['', '{scope}']")
    )
    node.metadata_props.append(
        StringStringEntryProto(
            key="pkg.torch.onnx.class_hierarchy",
            value=f"['BERTSafetyStudent', '{klass}']",
        )
    )


def _mvau_node(
    name: str,
    inp: str,
    weight: str,
    out: str,
    in_shape: list[int],
    in_ch: int,
    out_ch: int,
    preset: CorePreset,
    block_idx: int | None,
):
    node = helper.make_node(
        "MVAU",
        [inp, weight],
        [out],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        MW=in_ch,
        MH=out_ch,
        SIMD=preset.simd,
        PE=min(preset.pe, out_ch),
        inputDataType=ACT_DTYPE.name,
        weightDataType=WEIGHT_DTYPE.name,
        outputDataType=ACC_DTYPE.name,
        ActVal=0,
        binaryXnorMode=0,
        noActivation=1,
        numInputVectors=list(in_shape[:-1]),
        preferred_impl_style="rtl",
        mem_mode="internal_decoupled",
        ram_style=preset.ram_style,
        name=name,
    )
    _add_metadata(node, block_idx)
    return node


def _requant_node(
    name: str,
    inp: str,
    out: str,
    shape: list[int],
    preset: CorePreset,
    block_idx: int | None,
    input_dtype=ACC_DTYPE,
    output_dtype=ACT_DTYPE,
):
    channels = shape[-1]
    node = helper.make_node(
        "Requant",
        [inp, f"{name}_scale", f"{name}_bias"],
        [out],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumChannels=channels,
        PE=min(preset.pe, channels),
        inputDataType=input_dtype.name,
        outputDataType=output_dtype.name,
        numInputVectors=list(shape[:-1]),
        narrow=0,
        preferred_impl_style="rtl",
        name=name,
    )
    _add_metadata(node, block_idx)
    return node


def _add_node(
    name: str,
    lhs: str,
    rhs: str,
    out: str,
    shape: list[int],
    preset: CorePreset,
    block_idx: int | None,
):
    node = helper.make_node(
        "ElementwiseAdd",
        [lhs, rhs],
        [out],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        lhs_dtype=ACT_DTYPE.name,
        rhs_dtype=ACT_DTYPE.name,
        out_dtype=RESIDUAL_SUM_DTYPE.name,
        lhs_shape=shape,
        rhs_shape=shape,
        out_shape=shape,
        lhs_style="input",
        rhs_style="input",
        PE=min(preset.pe, shape[-1]),
        preferred_impl_style="rtl",
        name=name,
    )
    _add_metadata(node, block_idx)
    return node


def _duplicate_node(
    name: str,
    inp: str,
    out0: str,
    out1: str,
    shape: list[int],
    preset: CorePreset,
    block_idx: int | None,
):
    node = helper.make_node(
        "DuplicateStreams",
        [inp],
        [out0, out1],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumChannels=shape[-1],
        PE=min(preset.pe, shape[-1]),
        NumOutputStreams=2,
        inputDataType=ACT_DTYPE.name,
        numInputVectors=list(shape[:-1]),
        outFIFODepths=[2, 2],
        cpp_interface="hls_vector",
        hls_style="freerunning",
        preferred_impl_style="hls",
        name=name,
    )
    _add_metadata(node, block_idx)
    return node


def _linear_requant(
    nodes: list,
    initializers: list,
    value_info: list,
    name: str,
    inp: str,
    in_shape: list[int],
    out_ch: int,
    preset: CorePreset,
    block_idx: int | None,
) -> tuple[str, list[int]]:
    in_ch = in_shape[-1]
    acc = f"{name}_acc"
    out = f"{name}_q"
    out_shape = list(in_shape[:-1]) + [out_ch]
    weight_name = f"{name}_weight"
    initializers.append(_initial_weight(weight_name, in_ch, out_ch))
    scale, bias = _requant_params(name, out_ch)
    initializers.extend([scale, bias])
    value_info.extend([_vi(acc, out_shape), _vi(out, out_shape)])
    nodes.append(
        _mvau_node(
            f"{name}_mvau",
            inp,
            weight_name,
            acc,
            in_shape,
            in_ch,
            out_ch,
            preset,
            block_idx,
        )
    )
    nodes.append(_requant_node(name, acc, out, out_shape, preset, block_idx))
    return out, out_shape


def create_core_model(preset: CorePreset) -> ModelWrapper:
    if preset.seq_len < 2:
        raise ValueError("seq_len must include CLS plus at least one payload token")
    for value, field in [
        (preset.hidden, "hidden"),
        (preset.intermediate, "intermediate"),
    ]:
        if value % preset.pe != 0:
            raise ValueError(f"{field}={value} must be divisible by pe={preset.pe}")
    for value, field in [
        (preset.hidden, "hidden"),
        (preset.intermediate, "intermediate"),
    ]:
        if value % preset.simd != 0:
            raise ValueError(f"{field}={value} must be divisible by simd={preset.simd}")

    nodes = []
    initializers = []
    value_info = []

    patch_shape = [1, preset.seq_len - 1, preset.hidden]
    token_shape = [1, preset.seq_len, preset.hidden]
    patches = _vi("patches", patch_shape)
    logits = _vi("logits", [1, preset.num_classes])
    cls_token = np.zeros((1, 1, preset.hidden), dtype=np.float32)
    initializers.append(numpy_helper.from_array(cls_token, name="cls_token"))
    value_info.append(_vi("tokens_0", token_shape))

    add_cls = helper.make_node(
        "AddCLSToken",
        ["patches", "cls_token"],
        ["tokens_0"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumTokens=preset.seq_len - 1,
        NumChannels=preset.hidden,
        PadTokens=0,
        SIMD=preset.simd,
        inputDataType=ACT_DTYPE.name,
        outputDataType=ACT_DTYPE.name,
        preferred_impl_style="rtl",
        name="add_cls_token",
    )
    _add_metadata(add_cls, None)
    nodes.append(add_cls)

    current = "tokens_0"
    current_shape = token_shape
    for layer in range(preset.layers):
        attn_in = f"layer_{layer}_attention_input"
        attn_skip = f"layer_{layer}_attention_skip"
        value_info.extend([_vi(attn_in, current_shape), _vi(attn_skip, current_shape)])
        nodes.append(
            _duplicate_node(
                f"layer_{layer}_attention_fork",
                current,
                attn_in,
                attn_skip,
                current_shape,
                preset,
                layer,
            )
        )
        attn, _ = _linear_requant(
            nodes,
            initializers,
            value_info,
            f"layer_{layer}_attention_projection",
            attn_in,
            current_shape,
            preset.hidden,
            preset,
            layer,
        )
        attn_sum = f"layer_{layer}_attention_residual_sum"
        attn_res = f"layer_{layer}_attention_residual"
        value_info.extend([_vi(attn_sum, current_shape), _vi(attn_res, current_shape)])
        nodes.append(
            _add_node(
                f"{attn_sum}_add",
                attn_skip,
                attn,
                attn_sum,
                current_shape,
                preset,
                layer,
            )
        )
        scale, bias = _requant_params(attn_res, current_shape[-1])
        initializers.extend([scale, bias])
        nodes.append(
            _requant_node(
                attn_res,
                attn_sum,
                attn_res,
                current_shape,
                preset,
                layer,
                input_dtype=RESIDUAL_SUM_DTYPE,
            )
        )

        ffn_in = f"layer_{layer}_ffn_input"
        ffn_skip = f"layer_{layer}_ffn_skip"
        value_info.extend([_vi(ffn_in, current_shape), _vi(ffn_skip, current_shape)])
        nodes.append(
            _duplicate_node(
                f"layer_{layer}_ffn_fork",
                attn_res,
                ffn_in,
                ffn_skip,
                current_shape,
                preset,
                layer,
            )
        )
        ff1, ff1_shape = _linear_requant(
            nodes,
            initializers,
            value_info,
            f"layer_{layer}_ffn_expand",
            ffn_in,
            current_shape,
            preset.intermediate,
            preset,
            layer,
        )
        ff2, _ = _linear_requant(
            nodes,
            initializers,
            value_info,
            f"layer_{layer}_ffn_project",
            ff1,
            ff1_shape,
            preset.hidden,
            preset,
            layer,
        )
        ffn_sum = f"layer_{layer}_ffn_residual_sum"
        current = f"layer_{layer}_ffn_residual"
        value_info.extend([_vi(ffn_sum, current_shape), _vi(current, current_shape)])
        nodes.append(
            _add_node(f"{ffn_sum}_add", ffn_skip, ff2, ffn_sum, current_shape, preset, layer)
        )
        scale, bias = _requant_params(current, current_shape[-1])
        initializers.extend([scale, bias])
        nodes.append(
            _requant_node(
                current,
                ffn_sum,
                current,
                current_shape,
                preset,
                layer,
                input_dtype=RESIDUAL_SUM_DTYPE,
            )
        )

    cls = "selected_cls"
    value_info.append(_vi(cls, [1, preset.hidden]))
    select_cls = helper.make_node(
        "SelectToken",
        [current],
        [cls],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumTokens=preset.seq_len,
        NumChannels=preset.hidden,
        TokenIndex=0,
        SIMD=preset.simd,
        inputDataType=ACT_DTYPE.name,
        outputDataType=ACT_DTYPE.name,
        preferred_impl_style="rtl",
        name="select_cls_token",
    )
    _add_metadata(select_cls, None)
    nodes.append(select_cls)
    head_acc = "head_acc"
    value_info.append(_vi(head_acc, [1, preset.num_classes]))
    head_weight = "head_weight"
    initializers.append(_initial_weight(head_weight, preset.hidden, preset.num_classes))
    scale, bias = _requant_params("head", preset.num_classes)
    initializers.extend([scale, bias])
    nodes.append(
        _mvau_node(
            "head_mvau",
            cls,
            head_weight,
            head_acc,
            [1, preset.hidden],
            preset.hidden,
            preset.num_classes,
            preset,
            None,
        )
    )
    head_requant = helper.make_node(
        "Requant",
        [head_acc, "head_scale", "head_bias"],
        ["logits"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumChannels=preset.num_classes,
        PE=min(preset.pe, preset.num_classes),
        inputDataType=ACC_DTYPE.name,
        outputDataType=ACT_DTYPE.name,
        numInputVectors=[1],
        narrow=0,
        preferred_impl_style="rtl",
        name="head_requant",
    )
    _add_metadata(head_requant, None)
    nodes.append(head_requant)

    graph = helper.make_graph(
        nodes,
        f"bert_safety_core_{preset.name}",
        [patches],
        [logits],
        initializers,
        value_info=value_info,
    )
    model = helper.make_model(
        graph,
        producer_name="bert_safety_finn_flow",
        opset_imports=[
            helper.make_opsetid("", 11),
            helper.make_opsetid("finn.custom_op.fpgadataflow", 1),
        ],
    )
    mw = ModelWrapper(model)
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        if vi.name == head_acc or vi.name.endswith("_acc"):
            dtype = ACC_DTYPE
        elif vi.name.endswith("_sum"):
            dtype = RESIDUAL_SUM_DTYPE
        else:
            dtype = ACT_DTYPE
        mw.set_tensor_datatype(vi.name, dtype)
    for init in graph.initializer:
        if init.name.endswith("_weight"):
            mw.set_tensor_datatype(init.name, WEIGHT_DTYPE)
        elif init.name.endswith("_scale") or init.name.endswith("_bias"):
            mw.set_tensor_datatype(init.name, DataType["FLOAT32"])
        elif init.name == "cls_token":
            mw.set_tensor_datatype(init.name, ACT_DTYPE)
    for name in [x.name for x in graph.value_info if x.name.endswith("_acc")] + [head_acc]:
        mw.set_tensor_datatype(name, ACC_DTYPE)
    mw = mw.transform(InferShapes())
    mw = mw.transform(InferDataTypes())
    mw = mw.transform(GiveUniqueNodeNames())
    mw = mw.transform(GiveReadableTensorNames())
    return mw


def _node_has_scope(node, scope: str) -> bool:
    for prop in node.metadata_props:
        if prop.key == "pkg.torch.onnx.name_scopes" and scope in prop.value:
            return True
    return False


def first_layer_node_range(model: ModelWrapper) -> tuple:
    layer_nodes = [
        node for node in model.graph.node if _node_has_scope(node, "bert.encoder.layer.0")
    ]
    if not layer_nodes:
        raise RuntimeError("Could not find BERT encoder layer 0 for MLO rolling")
    return layer_nodes[0], layer_nodes[-1]


def roll_mlo(
    model: ModelWrapper, preset: CorePreset, output_dir: Path | None = None
) -> ModelWrapper:
    if preset.layers < 2:
        return model

    start_node, end_node = first_layer_node_range(model)
    node_metadata = {
        "pkg.torch.onnx.name_scopes": "['', 'bert.encoder.layer.0']",
        "pkg.torch.onnx.class_hierarchy": "['BERTSafetyStudent', 'BertLayer']",
    }
    model = model.transform(SetLoopBoundary(node_metadata, (start_node, end_node)))
    loop_extraction = LoopExtraction(hierarchy_list=[["", "bert.encoder.layer.0"]])
    model = model.transform(loop_extraction)
    fn_count = len(model.get_nodes_by_op_type("fn_loop-body"))
    if fn_count != preset.layers:
        raise RuntimeError(f"Loop extraction found {fn_count} bodies, expected {preset.layers}")
    model = model.transform(LoopRolling(loop_extraction.loop_body_template))
    model = model.transform(FoldConstants(), apply_to_subgraphs=True)
    model = model.transform(InferShapes(), apply_to_subgraphs=True)
    model = model.transform(InferDataTypes(), apply_to_subgraphs=True)
    model = model.transform(GiveUniqueNodeNames(), apply_to_subgraphs=True)
    model = model.transform(GiveReadableTensorNames())

    loop_template = Path("loop-body-template.onnx")
    if output_dir is not None and loop_template.is_file():
        loop_template.replace(output_dir / "loop-body-template.onnx")
    return model


def specialize_core_model(model: ModelWrapper, fpga_part: str = DEFAULT_FPGA_PART) -> ModelWrapper:
    model = model.transform(SpecializeLayers(fpga_part))
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(GiveUniqueNodeNames())
    return model


def summarize(model: ModelWrapper, preset: CorePreset) -> dict:
    return {
        "preset": preset.as_dict(),
        "nodes": len(model.graph.node),
        "op_counts": dict(Counter(node.op_type for node in model.graph.node).most_common()),
        "input": {
            "name": model.get_first_global_in(),
            "shape": model.get_tensor_shape(model.get_first_global_in()),
            "datatype": model.get_tensor_datatype(model.get_first_global_in()).name,
        },
        "output": {
            "name": model.get_first_global_out(),
            "shape": model.get_tensor_shape(model.get_first_global_out()),
            "datatype": model.get_tensor_datatype(model.get_first_global_out()).name,
        },
    }


def write_models(
    preset: CorePreset,
    output_dir: Path,
    fpga_part: str = DEFAULT_FPGA_PART,
    save_specialized: bool = True,
    save_mlo: bool = False,
    weight_state: str | Path | None = None,
    strict_weights: bool = False,
) -> tuple[Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = create_core_model(preset)
    if weight_state is not None:
        apply_imported_weights(
            model,
            preset,
            weight_state,
            output_dir / "quantized_weight_manifest.json",
            strict=strict_weights,
        )
    base_path = output_dir / "bert_safety_core.onnx"
    model.save(str(base_path))
    write_json(output_dir / "bert_safety_core_summary.json", summarize(model, preset))

    specialized_path = None
    if save_specialized:
        specialized = specialize_core_model(model, fpga_part)
        specialized_path = output_dir / "bert_safety_core_v80.onnx"
        specialized.save(str(specialized_path))
        write_json(
            output_dir / "bert_safety_core_v80_summary.json",
            summarize(specialized, preset),
        )
        if save_mlo:
            mlo = roll_mlo(specialized, preset, output_dir)
            specialized_path = output_dir / "bert_safety_core_mlo_v80.onnx"
            mlo.save(str(specialized_path))
            write_json(
                output_dir / "bert_safety_core_mlo_v80_summary.json",
                summarize(mlo, preset),
            )
    return base_path, specialized_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(CORE_PRESETS), default="smoke")
    parser.add_argument(
        "--output-dir",
        default=str((DEFAULT_BUILD_DIR / "model").relative_to(repo_path("."))),
    )
    parser.add_argument("--fpga-part", default=DEFAULT_FPGA_PART)
    parser.add_argument("--no-specialized", dest="save_specialized", action="store_false")
    parser.add_argument("--mlo", action="store_true")
    parser.add_argument(
        "--weight-state",
        default=None,
        help=(
            "Optional trained student checkpoint directory/file. Matching dense BERT "
            "weights are quantized to INT8 and imported into the FINN core."
        ),
    )
    parser.add_argument(
        "--strict-weights",
        action="store_true",
        help="Fail if any FINN core weight cannot be imported from --weight-state.",
    )
    parser.set_defaults(save_specialized=True)
    args = parser.parse_args()

    output_dir = repo_path(args.output_dir)
    base, specialized = write_models(
        get_preset(args.preset),
        output_dir,
        args.fpga_part,
        args.save_specialized,
        args.mlo,
        args.weight_state,
        args.strict_weights,
    )
    print(f"Wrote base model: {base}")
    if specialized is not None:
        print(f"Wrote V80-specialized model: {specialized}")


if __name__ == "__main__":
    main()
