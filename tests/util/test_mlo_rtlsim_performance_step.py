# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import json
import numpy as np
from types import SimpleNamespace

import finn.builder.build_dataflow_steps as steps
import finn.core.throughput_test as throughput
import finn.util.rtlsim as rtlsim
from finn.builder.build_dataflow_config import DataflowBuildConfig, DataflowOutputType


class _FakeMLOModel:
    def transform(self, _transformation):
        return self

    def analysis(self, _analysis):
        return {"critical_path_cycles": 100}

    def get_nodes_by_op_type(self, op_type):
        assert op_type == "FINNLoop"
        return [object()]


class _FakeFIFO:
    def __init__(self, impl_style):
        self.impl_style = impl_style

    def get_nodeattr(self, name):
        assert name == "impl_style"
        return self.impl_style

    def set_nodeattr(self, name, value):
        assert name == "impl_style"
        self.impl_style = value


class _FakeFIFOModel:
    def __init__(self):
        self.fifos = [_FakeFIFO("vivado"), _FakeFIFO("rtl")]

    def get_nodes_by_op_type(self, op_type):
        assert op_type == "StreamingFIFO_rtl"
        return self.fifos


def test_node_by_node_rtlsim_uses_rtl_fifo_copy(monkeypatch):
    model = _FakeFIFOModel()
    monkeypatch.setattr(steps, "getCustomOp", lambda node: node)

    verify_model = steps.prepare_for_node_by_node_rtlsim(model)

    assert verify_model is not model
    assert [fifo.impl_style for fifo in model.fifos] == ["vivado", "rtl"]
    assert [fifo.impl_style for fifo in verify_model.fifos] == ["rtl", "rtl"]


def test_mlo_performance_step_uses_two_frames_and_ideal_memory(tmp_path, monkeypatch):
    model = _FakeMLOModel()
    prehook = object()
    call = {}

    monkeypatch.setattr(steps, "is_mlo", lambda _model: True)
    monkeypatch.setattr(steps, "deepcopy", lambda original: original)
    monkeypatch.setattr(steps, "prepare_for_stitched_ip_rtlsim", lambda original, _cfg: original)

    def fake_prehook_factory(_node, **kwargs):
        call["prehook_kwargs"] = kwargs
        return prehook

    monkeypatch.setattr(steps, "mlo_prehook_func_factory", fake_prehook_factory)
    monkeypatch.setattr(steps, "get_liveness_threshold_cycles", lambda: 123)

    def fake_throughput_test(model_arg, clk_ns, **kwargs):
        call.update(model=model_arg, clk_ns=clk_ns, **kwargs)
        return {
            "N": kwargs["batchsize"],
            "completed_output_frames": 2,
            "interval_valid": 1,
            "steady_state_frames": 1,
            "steady_state_cycles": 50,
            "stable_throughput_valid": True,
        }

    monkeypatch.setattr(steps, "throughput_test_rtlsim", fake_throughput_test)

    cfg = DataflowBuildConfig(
        output_dir=str(tmp_path),
        synth_clk_period_ns=5.0,
        rtlsim_batch_size=1,
        mlo=True,
        generate_outputs=[
            DataflowOutputType.STITCHED_IP,
            DataflowOutputType.RTLSIM_PERFORMANCE,
        ],
    )

    returned = steps.step_measure_rtlsim_performance(model, cfg)

    assert returned is model
    assert call["model"] is model
    assert call["clk_ns"] == 5.0
    assert call["batchsize"] == 2
    assert call["pre_hook"] is prehook
    assert call["collect_performance"] is True
    assert call["input_data_pattern"] == "all_zero"
    assert call["prehook_kwargs"] == {"external_weight_data_pattern": "all_zero"}

    with open(tmp_path / "report" / "rtlsim_performance.json") as report_file:
        report = json.load(report_file)
    assert report["measurement_scope"] == "stitched_mlo"
    assert report["external_memory_model"] == "ideal_axi_mm"
    assert report["external_memory_model_is_ideal"] is True
    assert report["performance_interpretation"] == "ideal_memory_upper_bound"
    assert report["input_data_pattern"] == "all_zero"
    assert report["external_weight_data_pattern"] == "all_zero"
    assert report["timing_schedule_is_data_independent"] is True
    assert report["io_bandwidth_scope"] == "top_level_axi_stream_only"
    assert report["N"] == 2


def test_throughput_test_can_use_all_zero_inputs(monkeypatch):
    captured = {}

    class FakeModel:
        graph = SimpleNamespace(
            input=[SimpleNamespace(name="inp")],
            output=[],
        )

        def get_metadata_prop(self, name):
            return {"exec_mode": "rtlsim", "cycles_rtlsim": "10"}[name]

        def make_empty_exec_context(self):
            return {}

        def get_tensor_shape(self, _name):
            return [1, 3]

        def get_tensor_datatype(self, _name):
            return SimpleNamespace(bitwidth=lambda: 8)

    def fake_rtlsim_exec(_model, context, **_kwargs):
        captured.update(context)
        return {}

    monkeypatch.setattr(throughput, "rtlsim_exec", fake_rtlsim_exec)

    throughput.throughput_test_rtlsim(
        FakeModel(),
        clk_ns=5.0,
        batchsize=2,
        input_data_pattern="all_zero",
    )

    np.testing.assert_array_equal(captured["inp"], np.zeros((2, 3), dtype=np.float32))


def test_mlo_prehook_can_zero_external_weights(tmp_path, monkeypatch):
    body_input = SimpleNamespace(name="weights")
    downstream = SimpleNamespace(op_type="MVAU_rtl")
    body = SimpleNamespace(
        graph=SimpleNamespace(input=[body_input]),
        find_consumer=lambda _name: downstream,
    )
    attrs = {
        "body": body,
        "code_gen_dir_ipgen": str(tmp_path),
        "iteration": 2,
    }
    loop = SimpleNamespace(get_nodeattr=lambda name: attrs[name])
    mvau = SimpleNamespace(get_nodeattr=lambda name: {"address_offset": 0}[name])
    monkeypatch.setattr(rtlsim, "getCustomOp", lambda node: mvau if node is downstream else loop)
    (tmp_path / "memblock_MVAU_rtl_id_0.dat").write_text("010203\n040506\n")

    captured = {}

    class FakeSim:
        def aximm_queue(self, name):
            captured["queue"] = name

        def aximm_ro_image(self, name, address, image):
            captured.update(name=name, address=address, image=image)

    prehook = rtlsim.mlo_prehook_func_factory(object(), external_weight_data_pattern="all_zero")
    prehook(FakeSim())

    assert captured["queue"] == "m_axi_intermediate_frame"
    assert captured["name"] == "m_axi_MVAU_id_0"
    assert captured["address"] == 0
    assert captured["image"].shape == (6,)
    assert np.all(captured["image"] == 0)
