# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""SigLIP vision-encoder loop detection for multi-level offload (MLO)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def _is_layer_norm(node: Any) -> bool:
    return node.op_type == "LayerNormalization" or node.op_type.startswith("LayerNorm")


def find_vision_loop_body_ranges(model: Any, depth: int) -> list[dict[str, Any]]:
    """Find the repeated encoder blocks in a converted static SigLIP graph.

    Each vision block begins at every second LayerNorm. The LayerNorm following
    block ``depth - 1`` marks the end of the encoder stack. DuplicateStreams at
    block boundaries are kept on the producing side of an activation edge.
    """

    proto = model.model if hasattr(model, "model") else model
    nodes = list(proto.graph.node)
    layer_norm_indices = [index for index, node in enumerate(nodes) if _is_layer_norm(node)]
    if len(layer_norm_indices) < 2 * depth + 1:
        return []

    ranges = []
    for block_index in range(depth):
        start_index = layer_norm_indices[2 * block_index]
        start_node = nodes[start_index]
        if start_index > 0:
            previous = nodes[start_index - 1]
            if (
                previous.op_type.startswith("DuplicateStreams")
                and start_node.input
                and start_node.input[0] in previous.output
            ):
                start_index -= 1

        if block_index + 1 < depth:
            next_start_index = layer_norm_indices[2 * (block_index + 1)]
            end_index = next_start_index - 1
            end_node = nodes[end_index]
            next_start = nodes[next_start_index]
            if (
                end_node.op_type.startswith("DuplicateStreams")
                and next_start.input
                and next_start.input[0] in end_node.output
            ):
                end_index -= 1
        else:
            end_index = layer_norm_indices[2 * depth] - 1

        if end_index < start_index:
            raise RuntimeError(f"Invalid SigLIP loop-body range for block {block_index}")
        block_nodes = nodes[start_index : end_index + 1]
        op_types = [node.op_type for node in block_nodes]
        ranges.append(
            {
                "block": block_index,
                "start_index": start_index,
                "end_index": end_index,
                "start_node": nodes[start_index].name,
                "end_node": nodes[end_index].name,
                "node_count": len(block_nodes),
                "op_types": op_types,
                "op_counts": dict(Counter(op_types).most_common()),
            }
        )
    return ranges


def make_mlo_boundary_step(depth: int):
    """Create a builder injection which marks the first repeated vision block."""

    def step_mark_siglip_mlo_boundary(model, cfg):
        ranges = find_vision_loop_body_ranges(model, depth)
        if len(ranges) != depth:
            raise RuntimeError(f"Expected {depth} SigLIP vision blocks, found {len(ranges)}")
        first_signature = ranges[0]["op_types"]
        mismatched = [item["block"] for item in ranges if item["op_types"] != first_signature]
        if mismatched:
            raise RuntimeError(f"SigLIP loop-body topology differs in blocks {mismatched}")

        nodes = model.graph.node
        cfg.loop_body_range = (
            nodes[ranges[0]["start_index"]],
            nodes[ranges[0]["end_index"]],
        )
        cfg.loop_body_hierarchy = [["", "layers.0"]]
        output_path = Path(cfg.output_dir) / "siglip_mlo_ranges.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(ranges, output_file, indent=2)
        return model

    step_mark_siglip_mlo_boundary.__name__ = "step_mark_siglip_mlo_boundary"
    return step_mark_siglip_mlo_boundary
