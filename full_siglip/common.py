"""Shared helpers for the full SigLIP FINN flow."""

from __future__ import annotations

import json
import re
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import onnx

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_DIR = REPO_ROOT / "full_siglip" / "build"
DEFAULT_BUILD_CSV = REPO_ROOT / "full_siglip" / "builds.csv"
DEFAULT_QONNX = DEFAULT_BUILD_DIR / "ptq_w8a8" / "full_siglip_w8a8_qonnx.onnx"
DEFAULT_FP32_ONNX = DEFAULT_BUILD_DIR / "fp32_export" / "full_siglip_fp32.onnx"
DEFAULT_BOARD = "VCK190"
DEFAULT_CLOCK_NS = 1000.0 / 300.0
DEFAULT_TARGET_FPS = 1000
SIGLIP_DEPTH = 12
TOWERS = ("vision_model", "text_model")

ENCODER_LAYER_RE = re.compile(r"/wrapped/(vision_model|text_model)/encoder/layers\.(\d+)/")

RTL_PREFERRED_OP_TYPES = [
    "MVAU",
    "HWSoftmax",
    "LayerNorm",
    "PWPolyF",
    "ElementwiseAdd",
    "ElementwiseSub",
    "ElementwiseMul",
    "Requant",
]


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def value_info_summary(value_info: onnx.ValueInfoProto) -> dict[str, Any]:
    tensor_type = value_info.type.tensor_type
    dims = []
    for dim in tensor_type.shape.dim:
        if dim.dim_param:
            dims.append(dim.dim_param)
        elif dim.dim_value:
            dims.append(dim.dim_value)
        else:
            dims.append(None)
    return {"name": value_info.name, "shape": dims}


def op_counts(model: onnx.ModelProto) -> dict[str, int]:
    return dict(Counter(node.op_type for node in model.graph.node).most_common())


def domain_op_counts(model: onnx.ModelProto) -> dict[str, int]:
    counts = Counter(
        f"{node.domain or 'ai.onnx'}::{node.op_type}" for node in model.graph.node
    )
    return dict(counts.most_common())


def _layer_key(node: onnx.NodeProto) -> tuple[str, int] | None:
    match = ENCODER_LAYER_RE.search(node.name)
    if match is None:
        return None
    tower, layer = match.groups()
    return tower, int(layer)


def find_encoder_blocks(
    model: onnx.ModelProto, depth: int = SIGLIP_DEPTH
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[tuple[int, onnx.NodeProto]]] = defaultdict(list)
    for idx, node in enumerate(model.graph.node):
        key = _layer_key(node)
        if key is not None:
            grouped[key].append((idx, node))

    blocks: dict[str, list[dict[str, Any]]] = {tower: [] for tower in TOWERS}
    for tower in TOWERS:
        for layer in range(depth):
            items = grouped.get((tower, layer), [])
            if not items:
                continue
            indices = [idx for idx, _ in items]
            nodes = [node for _, node in items]
            signature = [node.op_type for node in nodes]
            counts = Counter(signature)
            blocks[tower].append(
                {
                    "layer": layer,
                    "start_index": min(indices),
                    "end_index": max(indices),
                    "first_node": nodes[0].name,
                    "last_node": nodes[-1].name,
                    "node_count": len(nodes),
                    "op_counts": dict(counts.most_common()),
                    "softmax_nodes": [node.name for node in nodes if node.op_type == "Softmax"],
                    "quant_nodes": [
                        node.name
                        for node in nodes
                        if node.domain == "qonnx.custom_op.general"
                        and node.op_type == "Quant"
                    ],
                    "topology_signature": signature,
                }
            )
    return blocks


def block_topology_summary(blocks: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary = {}
    for tower, tower_blocks in blocks.items():
        signatures = Counter(tuple(block["topology_signature"]) for block in tower_blocks)
        summary[tower] = {
            "blocks": len(tower_blocks),
            "unique_topologies": len(signatures),
            "topology_counts": sorted(signatures.values(), reverse=True),
        }
    return summary


def conversion_pressure(model: onnx.ModelProto) -> dict[str, Any]:
    counts = Counter(node.op_type for node in model.graph.node)
    pressure_ops = [
        "Conv",
        "MatMul",
        "Gemm",
        "LayerNormalization",
        "Softmax",
        "Tanh",
        "ReduceL2",
        "Div",
        "Gather",
        "Shape",
        "Slice",
        "Transpose",
        "Reshape",
    ]
    return {op: counts[op] for op in pressure_ops if counts[op]}


def summarize_model(model: onnx.ModelProto) -> dict[str, Any]:
    blocks = find_encoder_blocks(model)
    return {
        "ir_version": model.ir_version,
        "opsets": {op.domain or "ai.onnx": op.version for op in model.opset_import},
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "inputs": [value_info_summary(value_info) for value_info in model.graph.input],
        "outputs": [value_info_summary(value_info) for value_info in model.graph.output],
        "op_counts": op_counts(model),
        "domain_op_counts": domain_op_counts(model),
        "encoder_blocks": blocks,
        "encoder_topology": block_topology_summary(blocks),
        "conversion_pressure": conversion_pressure(model),
    }


def write_rtl_specialization_config(
    path: str | Path,
    *,
    mvau_impl_style: str = "rtl",
    mvau_hls_nodes: list[str] | None = None,
) -> dict[str, Any]:
    if mvau_impl_style == "rtl":
        impl_defaults = ["rtl", RTL_PREFERRED_OP_TYPES]
    elif mvau_impl_style == "hls":
        rtl_ops = [op for op in RTL_PREFERRED_OP_TYPES if op != "MVAU"]
        impl_defaults = ["hls", ["MVAU"], "rtl", rtl_ops]
    else:
        raise ValueError(f"Unsupported MVAU implementation style: {mvau_impl_style}")

    config = {"Defaults": {"preferred_impl_style": impl_defaults}}
    for node_name in mvau_hls_nodes or []:
        config[node_name] = {"preferred_impl_style": "hls"}
    write_json(path, config)
    return config


def is_layer_norm_op(node: onnx.NodeProto) -> bool:
    return node.op_type == "LayerNormalization" or node.op_type.startswith("LayerNorm")


def _tensor_shape_map(model: onnx.ModelProto) -> dict[str, tuple[Any, ...]]:
    shape_map = {}
    for value_info in list(model.graph.input) + list(model.graph.value_info) + list(
        model.graph.output
    ):
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims = []
        for dim in tensor_type.shape.dim:
            if dim.dim_param:
                dims.append(dim.dim_param)
            elif dim.dim_value:
                dims.append(dim.dim_value)
            else:
                dims.append(None)
        shape_map[value_info.name] = tuple(dims)
    return shape_map


def _full_siglip_shape_tower(shape: tuple[Any, ...] | None) -> str | None:
    if shape is None:
        return None
    int_dims = {dim for dim in shape if isinstance(dim, int)}
    if 196 in int_dims:
        return "vision"
    if len(shape) >= 3 and shape[0] == 1 and shape[1] == 14 and shape[2] == 14:
        return "vision"
    if 64 in int_dims and 196 not in int_dims:
        return "text"
    return None


def full_siglip_node_tower(
    node: onnx.NodeProto,
    shape_map: dict[str, tuple[Any, ...]],
    initializer_names: set[str],
) -> str | None:
    """Classify a flat full-SigLIP dataflow node as text or vision tower.

    The full two-tower partition has generic node names after cleanup, but the
    activation shapes still distinguish the text sequence (`64`) from the
    vision patch sequence (`196`) and image patch grid (`14x14`). Initializers
    are ignored so shared parameter shapes do not vote.
    """

    votes = []
    for tensor_name in list(node.input) + list(node.output):
        if tensor_name in initializer_names:
            continue
        tower = _full_siglip_shape_tower(shape_map.get(tensor_name))
        if tower is not None:
            votes.append(tower)
    if not votes:
        return None
    counts = Counter(votes)
    if len(counts) != 1:
        return None
    return votes[0]


def find_full_siglip_tower_loop_body_ranges(
    model: onnx.ModelProto,
    depth: int = SIGLIP_DEPTH,
) -> dict[str, Any]:
    """Detect repeated full-model text/vision block ranges in a flat partition.

    The full dynamic SigLIP partition interleaves text and vision tower nodes in
    topological order. Plain LayerNorm-pair ranges therefore mix towers and do
    not expose a repeated MLO body. This detector first classifies nodes by
    activation shape, then builds per-tower block candidates between every
    second same-tower LayerNorm and the following same-tower LayerNorm.
    """

    nodes = list(model.graph.node)
    shape_map = _tensor_shape_map(model)
    initializer_names = {init.name for init in model.graph.initializer}
    node_towers = [
        full_siglip_node_tower(node, shape_map, initializer_names) for node in nodes
    ]
    result: dict[str, Any] = {
        "node_count": len(nodes),
        "tower_counts": dict(Counter(tower for tower in node_towers if tower).most_common()),
        "towers": {},
    }

    for tower in ("text", "vision"):
        layernorm_indices = [
            idx
            for idx, node in enumerate(nodes)
            if is_layer_norm_op(node) and node_towers[idx] == tower
        ]
        tower_result: dict[str, Any] = {
            "layernorm_count": len(layernorm_indices),
            "layernorm_indices": layernorm_indices,
            "blocks": [],
            "signature_counts": {},
            "roll_depth": 0,
        }
        if len(layernorm_indices) < (2 * depth + 1):
            result["towers"][tower] = tower_result
            continue

        for block_idx in range(depth):
            start_idx = layernorm_indices[2 * block_idx]
            end_idx = layernorm_indices[2 * (block_idx + 1)] - 1
            block_items = [
                (idx, nodes[idx])
                for idx in range(start_idx, end_idx + 1)
                if node_towers[idx] == tower
            ]
            signature = [node.op_type for _, node in block_items]
            signature_sha1 = hashlib.sha1("\n".join(signature).encode()).hexdigest()
            tower_result["blocks"].append(
                {
                    "block": block_idx,
                    "loop_start_index": block_items[0][0] if block_items else None,
                    "loop_end_index": block_items[-1][0] if block_items else None,
                    "loop_start_node": block_items[0][1].name if block_items else "",
                    "loop_end_node": block_items[-1][1].name if block_items else "",
                    "loop_node_indices": [idx for idx, _ in block_items],
                    "loop_node_count": len(block_items),
                    "signature_sha1": signature_sha1,
                    "loop_op_types": signature,
                    "op_counts": dict(Counter(signature).most_common()),
                }
            )

        signatures = Counter(block["signature_sha1"] for block in tower_result["blocks"])
        tower_result["signature_counts"] = dict(signatures.most_common())
        first_signature = (
            tower_result["blocks"][0]["signature_sha1"] if tower_result["blocks"] else None
        )
        roll_depth = 0
        for block in tower_result["blocks"]:
            if block["signature_sha1"] != first_signature:
                break
            roll_depth += 1
        tower_result["roll_depth"] = roll_depth
        result["towers"][tower] = tower_result

    return result


def find_static_vision_loop_body_ranges(
    model: onnx.ModelProto, depth: int = SIGLIP_DEPTH
) -> list[dict[str, Any]]:
    """Detect repeated vision encoder loop bodies in the static ImageNet graph.

    The static ImageNet export contains the SigLIP vision tower followed by the
    attention-pool head and frozen text-embedding comparison head. After FINN HW
    conversion the vision encoder blocks are still clean repeated regions:
    each block has two layer norms and the post-stack layer norm is immediately
    after the twelfth block. This detector intentionally targets that static
    image-only graph, not the full dynamic two-tower graph.
    """

    nodes = list(model.graph.node)
    layer_norm_indices = [idx for idx, node in enumerate(nodes) if is_layer_norm_op(node)]
    if len(layer_norm_indices) < 2 * depth + 1:
        return []

    ranges = []
    for block_idx in range(depth):
        start_idx = layer_norm_indices[2 * block_idx]
        start_node = nodes[start_idx]
        if start_idx > 0:
            prev_node = nodes[start_idx - 1]
            if (
                prev_node.op_type.startswith("DuplicateStreams")
                and start_node.input
                and start_node.input[0] in prev_node.output
            ):
                start_idx -= 1

        if block_idx + 1 < depth:
            next_start_idx = layer_norm_indices[2 * (block_idx + 1)]
            end_idx = next_start_idx - 1
            end_node = nodes[end_idx]
            next_start_node = nodes[next_start_idx]
            if (
                end_node.op_type.startswith("DuplicateStreams")
                and next_start_node.input
                and next_start_node.input[0] in end_node.output
            ):
                end_idx -= 1
        else:
            post_stack_norm_idx = layer_norm_indices[2 * depth]
            end_idx = post_stack_norm_idx - 1

        if end_idx < start_idx:
            raise RuntimeError(f"Invalid static SigLIP loop-body range for block {block_idx}")

        loop_nodes = nodes[start_idx : end_idx + 1]
        ranges.append(
            {
                "block": block_idx,
                "loop_start_index": start_idx,
                "loop_end_index": end_idx,
                "loop_start_node": nodes[start_idx].name,
                "loop_end_node": nodes[end_idx].name,
                "loop_node_count": len(loop_nodes),
                "loop_op_types": [node.op_type for node in loop_nodes],
                "op_counts": dict(Counter(node.op_type for node in loop_nodes).most_common()),
            }
        )
    return ranges


def first_static_vision_loop_body_node_range(
    model: Any, depth: int = SIGLIP_DEPTH
) -> tuple[Any, Any]:
    proto = model.model if hasattr(model, "model") else model
    ranges = find_static_vision_loop_body_ranges(proto, depth)
    if not ranges:
        raise RuntimeError("Could not detect static SigLIP vision loop-body ranges")
    loop_range = ranges[0]
    return (
        model.graph.node[loop_range["loop_start_index"]],
        model.graph.node[loop_range["loop_end_index"]],
    )
