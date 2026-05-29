#!/usr/bin/env python3
"""Print FINNLoop and loop-body stitched interface metadata."""

from __future__ import annotations

import argparse

from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.util.basic import getHWCustomOp
from qonnx.core.modelwrapper import ModelWrapper


def _attr_value(attr):
    if attr.type == attr.INT:
        return attr.i
    if attr.type == attr.FLOAT:
        return attr.f
    if attr.type == attr.STRING:
        return attr.s.decode()
    if attr.type == attr.INTS:
        return list(attr.ints)
    if attr.type == attr.FLOATS:
        return list(attr.floats)
    if attr.type == attr.STRINGS:
        return [x.decode() for x in attr.strings]
    return "<unsupported>"


def _print_node_detail(body, body_node) -> None:
    body_inst = getHWCustomOp(body_node)
    print(f" detail {body_node.name} {body_node.op_type}:")
    for idx, inp in enumerate(body_node.input):
        shape = body.get_tensor_shape(inp)
        folded = None
        if hasattr(body_inst, "get_folded_input_shape"):
            try:
                folded = body_inst.get_folded_input_shape(idx)
            except Exception:
                folded = None
        print(f"  input {idx}: shape={shape} folded={folded}")
    if hasattr(body_inst, "get_folded_output_shape"):
        try:
            folded_out = body_inst.get_folded_output_shape(0)
        except Exception:
            folded_out = None
        print(f"  output 0: shape={body.get_tensor_shape(body_node.output[0])} folded={folded_out}")
    interesting_attrs = {
        "PE",
        "SIMD",
        "MW",
        "MH",
        "NumChannels",
        "mem_mode",
        "mlo_max_iter",
        "backend",
        "resType",
        "ram_style",
    }
    attrs = {
        attr.name: _attr_value(attr)
        for attr in body_node.attribute
        if attr.name in interesting_attrs
    }
    if attrs:
        print(f"  attrs: {attrs}")


def _node_ref(node) -> str:
    if node is None:
        return "-"
    return f"{node.name} ({node.op_type})"


def _consumers_ref(body, tensor_name: str) -> str:
    consumers = body.find_consumers(tensor_name)
    if not consumers:
        return "-"
    return ", ".join(_node_ref(node) for node in consumers)


def _print_tensor_edges(body, tensor_name: str) -> None:
    producer = body.find_producer(tensor_name)
    print(
        f" edge {tensor_name}: shape={body.get_tensor_shape(tensor_name)} "
        f"producer={_node_ref(producer)} consumers={_consumers_ref(body, tensor_name)}"
    )


def _print_node_edges(body, body_node) -> None:
    print(f" edges {body_node.name} {body_node.op_type}:")
    for idx, inp in enumerate(body_node.input):
        producer = body.find_producer(inp)
        print(
            f"  input {idx}: {inp} shape={body.get_tensor_shape(inp)} "
            f"producer={_node_ref(producer)}"
        )
    for idx, outp in enumerate(body_node.output):
        print(
            f"  output {idx}: {outp} shape={body.get_tensor_shape(outp)} "
            f"consumers={_consumers_ref(body, outp)}"
        )


def _print_top_cycles(model, limit: int, prefix: str = " node") -> None:
    nodes = []
    for node in model.graph.node:
        inst = getHWCustomOp(node, model)
        nodes.append((inst.get_nodeattr("cycles_estimate"), node.name, node.op_type))
    for cycles, name, op_type in sorted(nodes, reverse=True)[:limit]:
        print(f"{prefix} {name} {op_type}: cycles_estimate={cycles}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument(
        "--detail",
        nargs="*",
        default=None,
        help="Optional loop-body node names to print folded shapes and key attrs for.",
    )
    parser.add_argument(
        "--edges",
        nargs="*",
        default=None,
        help="Optional loop-body node or tensor names to print producer/consumer edges for.",
    )
    parser.add_argument("--cycles", action="store_true", help="Print annotated cycle estimates.")
    parser.add_argument("--top-cycles", type=int, default=12)
    args = parser.parse_args()

    model = ModelWrapper(args.model)
    if args.cycles:
        model = model.transform(AnnotateCycles())
        print("network performance:", model.analysis(dataflow_performance))
        _print_top_cycles(model, args.top_cycles, prefix=" top")
    for node in model.graph.node:
        if node.op_type != "FINNLoop":
            continue
        inst = getHWCustomOp(node)
        body = inst.get_nodeattr("body")
        print(f"FINNLoop {node.name}")
        print(" top ifnames:", inst.get_verilog_top_module_intf_names())
        print(" iteration:", inst.get_nodeattr("iteration"))
        print(" input shape:", inst.get_normal_input_shape(0))
        print(" folded input shape:", inst.get_folded_input_shape(0))
        print(" output shape:", inst.get_normal_output_shape(0))
        print(" folded output shape:", inst.get_folded_output_shape(0))
        print(" input stream width:", inst.get_instream_width(0))
        print(" output stream width:", inst.get_outstream_width(0))
        if args.cycles:
            body = body.transform(AnnotateCycles())
            print(" cycles estimate:", inst.get_nodeattr("cycles_estimate"))
            print(" exp cycles:", inst.get_exp_cycles())
            print(" body performance:", body.analysis(dataflow_performance))
            _print_top_cycles(body, args.top_cycles, prefix=" body")
        print(" body vivado ifnames:", body.get_metadata_prop("vivado_stitch_ifnames"))
        print(" body inputs:")
        for idx, inp in enumerate(body.graph.input):
            consumer = body.find_consumer(inp.name)
            print(f"  {idx}: {inp.name} -> {consumer.name} ({consumer.op_type})")
        print(" body node s_axis:")
        for body_node in body.graph.node:
            body_inst = getHWCustomOp(body_node)
            if hasattr(body_inst, "get_verilog_top_module_intf_names"):
                ifnames = body_inst.get_verilog_top_module_intf_names()
                s_axis = ifnames.get("s_axis", [])
                if s_axis:
                    print(f"  {body_node.name} {body_node.op_type}: {s_axis}")
        if args.detail:
            for detail_name in args.detail:
                body_node = body.get_node_from_name(detail_name)
                if body_node is None:
                    print(f" detail {detail_name}: not found")
                    continue
                _print_node_detail(body, body_node)
        if args.edges:
            for edge_name in args.edges:
                body_node = body.get_node_from_name(edge_name)
                if body_node is not None:
                    _print_node_edges(body, body_node)
                else:
                    _print_tensor_edges(body, edge_name)


if __name__ == "__main__":
    main()
