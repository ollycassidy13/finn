# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from finn.transformer_examples.siglip.build import _verification_steps
from finn.transformer_examples.siglip.config import DEFAULT_PROFILE, load_profile
from finn.transformer_examples.siglip.mlo import (
    find_vision_loop_body_ranges,
    make_mlo_boundary_step,
)
from finn.transformer_examples.siglip.phases import (
    _DuplicateSafeModelWrapper,
    _absorb_pre_matmul_dequant,
    _integerize_static_lhs_matmul,
    _select_graph_output,
    make_siglip_folding_step,
)
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_shapes import InferShapes


def _node(op_type, index):
    return SimpleNamespace(op_type=op_type, name=f"{op_type}_{index}", input=[], output=[])


def _repeated_encoder(depth=12, mismatch=None):
    nodes = []
    for block in range(depth):
        nodes.extend(
            [
                _node("LayerNorm_rtl", 4 * block),
                _node("MVAU_rtl", 4 * block + 1),
                _node("LayerNorm_rtl", 4 * block + 2),
                _node("PWPolyF_rtl" if block == mismatch else "MVAU_rtl", 4 * block + 3),
            ]
        )
    nodes.append(_node("LayerNorm_rtl", 4 * depth))
    return SimpleNamespace(graph=SimpleNamespace(node=nodes))


def test_default_profile_records_verified_w6a7_contract():
    profile = load_profile(DEFAULT_PROFILE)
    assert profile.model["model_id"] == "google/siglip2-base-patch16-224"
    assert profile.model["vision_depth"] == 12
    assert profile.quantization == {
        "scheme": "qv_lsq",
        "weight_bits": 6,
        "activation_bits": 7,
        "edge_bits": 8,
    }
    assert profile.model["output_name"] == "image_embeds"
    assert profile.reference_metrics["accuracy"]["images"] == 50_000
    assert profile.reference_metrics["board_runtime_throughput_fps"] is None
    assert profile.build["verification_atol"] == 0.25
    assert profile.resolve_file(profile.build["folding_config"]).is_file()


def test_profile_rejects_unvalidated_board(tmp_path):
    text = DEFAULT_PROFILE.read_text().replace('"board": "VCK190"', '"board": "U250"')
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(text)
    with pytest.raises(ValueError, match="board=VCK190"):
        load_profile(profile_path)


def test_final_folding_stage_rejects_stale_node_names(tmp_path):
    node = SimpleNamespace(
        name="MVAU_rtl_0",
        op_type="MVAU_rtl",
        domain="finn.custom_op.fpgadataflow.rtl",
        attribute=[],
    )
    model = SimpleNamespace(graph=SimpleNamespace(node=[node]))
    config_path = tmp_path / "folding.json"
    config_path.write_text('{"Defaults": {}, "old_MVAU_rtl_0": {"PE": 1}}')

    folding_step = make_siglip_folding_step(config_path, require_complete=True)
    with pytest.raises(RuntimeError, match="folding profile does not match"):
        folding_step(model, SimpleNamespace())


def test_readable_names_preserve_repeated_siglip_gelu_input():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    nodes = [
        helper.make_node("Relu", ["x"], ["hidden"]),
        helper.make_node("Mul", ["hidden", "hidden"], ["y"]),
    ]
    model = _DuplicateSafeModelWrapper(
        helper.make_model(helper.make_graph(nodes, "gelu", [x], [y]))
    )
    model = (
        model.transform(InferShapes())
        .transform(GiveUniqueNodeNames())
        .transform(GiveReadableTensorNames())
    )

    mul = model.get_nodes_by_op_type("Mul")[0]
    assert mul.input[0] == mul.input[1]
    assert model.find_producer(mul.input[0]).op_type == "Relu"
    onnx.checker.check_model(model.model)


def test_estimate_mode_rejects_simulators_it_does_not_build():
    for level in ("cppsim", "rtlsim"):
        with pytest.raises(ValueError, match="requires stitched_ip or ooc_synth"):
            _verification_steps(level, "estimate")


def test_selects_embedding_output_and_removes_comparison_branch():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    class_scores = helper.make_tensor_value_info("class_scores", TensorProto.FLOAT, [1, 4])
    image_embeds = helper.make_tensor_value_info("image_embeds", TensorProto.FLOAT, [1, 4])
    nodes = [
        helper.make_node("Identity", ["x"], ["image_embeds"]),
        helper.make_node("Relu", ["image_embeds"], ["class_scores"]),
    ]
    model = _DuplicateSafeModelWrapper(
        helper.make_model(helper.make_graph(nodes, "outputs", [x], [class_scores, image_embeds]))
    )

    model = _select_graph_output(model, "image_embeds")
    assert [value_info.name for value_info in model.graph.output] == ["image_embeds"]
    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == "Identity"


def test_moves_scalar_dequant_after_static_weight_matmul():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    x_scaled = helper.make_tensor_value_info("x_scaled", TensorProto.FLOAT, [1, 4])
    mm_out = helper.make_tensor_value_info("mm_out", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    graph = helper.make_graph(
        [
            helper.make_node("Mul", ["x", "act_scale"], ["x_scaled"], name="dequant"),
            helper.make_node("MatMul", ["x_scaled", "weight"], ["mm_out"], name="proj"),
            helper.make_node("Mul", ["mm_out", "weight_scale"], ["y"]),
        ],
        "pre-matmul-dequant",
        [x],
        [y],
        value_info=[x_scaled, mm_out],
        initializer=[
            numpy_helper.from_array(np.asarray([0.25], dtype=np.float32), "act_scale"),
            numpy_helper.from_array(
                np.asarray(
                    [[1, 0, -1], [0, 1, 1], [1, -1, 0], [-1, 0, 1]],
                    dtype=np.float32,
                ),
                "weight",
            ),
            numpy_helper.from_array(np.asarray([0.5, 0.75, 1.0], dtype=np.float32), "weight_scale"),
        ],
    )
    model = ModelWrapper(helper.make_model(graph))
    model.set_tensor_datatype("x", DataType["INT4"])
    model.set_tensor_datatype("x_scaled", DataType["FLOAT32"])
    model.set_tensor_datatype("weight", DataType["INT4"])
    model.set_tensor_datatype("mm_out", DataType["FLOAT32"])

    model = _absorb_pre_matmul_dequant(model)

    matmul = [node for node in model.graph.node if node.name == "proj"][0]
    assert list(matmul.input) == ["x", "weight"]
    assert model.get_tensor_datatype("mm_out") == DataType["INT32"]
    np.testing.assert_array_equal(
        model.get_initializer("weight_scale"),
        np.asarray([0.125, 0.1875, 0.25], dtype=np.float32),
    )
    assert "dequant" not in [node.name for node in model.graph.node]


def test_integerizes_exact_static_attention_query_once():
    rhs = helper.make_tensor_value_info("rhs", TensorProto.FLOAT, [1, 2, 4, 3])
    mm_out = helper.make_tensor_value_info("mm_out", TensorProto.FLOAT, [1, 2, 1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 1, 3])
    lhs = np.asarray(
        [[[[0.5, -0.5, 0.0, 0.75]], [[-0.5, 0.5, 0.75, 0.0]]]],
        dtype=np.float32,
    )
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["lhs", "rhs"], ["mm_out"], name="pool_scores"),
            helper.make_node("Mul", ["mm_out", "score_scale"], ["y"]),
        ],
        "static-lhs-matmul",
        [rhs],
        [y],
        value_info=[mm_out],
        initializer=[
            numpy_helper.from_array(lhs, "lhs"),
            numpy_helper.from_array(np.asarray([0.5], dtype=np.float32), "score_scale"),
        ],
    )
    model = ModelWrapper(helper.make_model(graph))
    model.set_tensor_datatype("lhs", DataType["FLOAT32"])
    model.set_tensor_datatype("rhs", DataType["INT4"])
    model.set_tensor_datatype("mm_out", DataType["FLOAT32"])

    model = _integerize_static_lhs_matmul(model)
    integer_lhs = model.get_initializer("lhs").copy()
    integer_scale = model.get_initializer("score_scale").copy()

    np.testing.assert_array_equal(integer_lhs * integer_scale, lhs * 0.5)
    assert np.min(integer_lhs) >= -128
    assert np.max(integer_lhs) <= 127
    assert model.get_tensor_datatype("lhs") == DataType["INT8"]
    assert model.get_tensor_datatype("mm_out") == DataType["INT32"]

    model = _integerize_static_lhs_matmul(model)
    np.testing.assert_array_equal(model.get_initializer("lhs"), integer_lhs)
    np.testing.assert_array_equal(model.get_initializer("score_scale"), integer_scale)


def test_detects_all_repeated_vision_blocks():
    ranges = find_vision_loop_body_ranges(_repeated_encoder(), depth=12)
    assert len(ranges) == 12
    assert ranges[0]["op_types"] == ranges[-1]["op_types"]
    assert ranges[0]["start_node"] == "LayerNorm_rtl_0"
    assert ranges[-1]["end_node"] == "MVAU_rtl_47"


def test_exposes_topology_mismatch_to_mlo_step(tmp_path):
    model = _repeated_encoder(mismatch=7)
    cfg = SimpleNamespace(output_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match=r"blocks \[7\]"):
        make_mlo_boundary_step(12)(model, cfg)
