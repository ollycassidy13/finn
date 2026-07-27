# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import json

import finn.builder.build_dataflow_steps as steps
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
    monkeypatch.setattr(steps, "mlo_prehook_func_factory", lambda _node: prehook)
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

    with open(tmp_path / "report" / "rtlsim_performance.json") as report_file:
        report = json.load(report_file)
    assert report["measurement_scope"] == "stitched_mlo"
    assert report["external_memory_model"] == "ideal_axi_mm"
    assert report["external_memory_model_is_ideal"] is True
    assert report["performance_interpretation"] == "ideal_memory_upper_bound"
    assert report["io_bandwidth_scope"] == "top_level_axi_stream_only"
    assert report["N"] == 2
