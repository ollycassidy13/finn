# Copyright (c) 2020, Xilinx, Inc.
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import pytest

import numpy as np
import os
from types import SimpleNamespace
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.alveo_build import PrepareForLinking, VitisLink
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.create_stitched_ip import (
    CreateStitchedIP,
    _append_ip_repo_path_cmds,
    _batch_simple_add_files,
    _check_vivado_stitch_outputs,
    _collapse_ip_repo_paths,
    _drop_intermediate_save_bd_design,
    _pnr_clock_constraint_cmds,
)
from finn.custom_op.fpgadataflow.rtl.finn_loop import (
    FINNLoop,
    _add_files_copy_to_cmd,
    _collapse_ip_repo_paths as _collapse_loop_ip_repo_paths,
    _ip_repo_paths_append_cmd,
    _ip_repo_paths_set_cmd,
)
from finn.transformation.fpgadataflow.floorplan import Floorplan
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_iodma import InsertIODMA
from finn.transformation.fpgadataflow.insert_tlastmarker import InsertTLastMarker
from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.util.basic import (
    getHWCustomOp,
    pynq_part_map,
    vitis_default_platform,
    vitis_part_map,
)
from finn.util.test import load_test_checkpoint_or_skip
from finn.util.vivado import parse_ooc_synth_results

test_pynq_board = "Pynq-Z1"
test_fpga_part = pynq_part_map[test_pynq_board]

ip_stitch_model_dir = os.environ["FINN_BUILD_DIR"]


def test_fpgadataflow_ipstitch_tcl_cleanup_helpers():
    tcl = [
        "add_files -norecurse /tmp/a.sv",
        "add_files -norecurse /tmp/b.sv",
        "create_bd_cell -type module -reference A A",
        "add_files -copy_to /tmp/out -norecurse /tmp/c.sv",
        "save_bd_design",
        "save_bd_design",
        "validate_bd_design",
        "save_bd_design",
    ]

    batched = _batch_simple_add_files(tcl)
    cleaned = _drop_intermediate_save_bd_design(batched)

    assert "add_files -norecurse [list /tmp/a.sv /tmp/b.sv]" in cleaned
    assert "add_files -copy_to /tmp/out -norecurse /tmp/c.sv" in cleaned
    assert "create_bd_cell -type module -reference A A" in cleaned
    assert cleaned.count("save_bd_design") == 2
    assert cleaned.count("validate_bd_design") == 1


def test_fpgadataflow_ipstitch_fails_on_child_vivado_error_marker(tmp_path):
    run_dir = tmp_path / "finn_vivado_stitch_proj.runs" / "bad_synth_1"
    run_dir.mkdir(parents=True)
    (run_dir / ".vivado.error.rst").write_text("")
    (run_dir / "runme.log").write_text("INFO: starting\nERROR: child synth failed\n")
    (tmp_path / "finn_design.dcp").write_text("dcp")
    (tmp_path / "finn_design.xdc").write_text("xdc")

    with pytest.raises(Exception, match="child synth failed"):
        _check_vivado_stitch_outputs(str(tmp_path), "finn_design", True, False, 0)


def test_fpgadataflow_ipstitch_fails_when_expected_pnr_artifacts_missing(tmp_path):
    (tmp_path / "finn_design.dcp").write_text("dcp")
    (tmp_path / "finn_design.xdc").write_text("xdc")
    (tmp_path / "ooc_utilization.rpt").write_text("util")

    with pytest.raises(Exception, match="finn_design_routed.dcp"):
        _check_vivado_stitch_outputs(str(tmp_path), "finn_design", True, True, 0)


def test_fpgadataflow_ipstitch_chunks_large_ip_repo_path_lists():
    tcl = []
    ip_dirs = ["list"] + ["/tmp/ip_repo_%03d" % i for i in range(20)]

    _append_ip_repo_path_cmds(tcl, ip_dirs, max_cmd_chars=80)

    assert tcl[0] == "set_property ip_repo_paths [list] [current_project]"
    assert tcl[-1] == "update_ip_catalog -rebuild -scan_changes"
    assert len([cmd for cmd in tcl if "concat [get_property ip_repo_paths" in cmd]) > 1
    assert all("/tmp/ip_repo_%03d" % i in "\n".join(tcl) for i in range(20))
    assert all(len(cmd) < 260 for cmd in tcl)


def test_fpgadataflow_ipstitch_preserves_exact_generated_ip_repo_paths_by_default():
    hw_root = "/scratch/work/finn_temp_file_siglip_hwipgen"
    fifo_root = "/scratch/work/finn_temp_file_siglip_postfifo"
    ip_dirs = [
        "list",
        "$::env(FINN_ROOT)/finn-rtllib/memstream",
        fifo_root + "/code_gen_ipgen_StreamingFIFO_rtl_0_abcd",
        fifo_root + "/code_gen_ipgen_StreamingFIFO_rtl_1_efgh",
        hw_root + "/code_gen_ipgen_DuplicateStreams_hls_18_abcd/project_DuplicateStreams_hls_18/sol1/impl/ip",
        hw_root + "/code_gen_ipgen_OuterShuffle_hls_56_efgh/project_OuterShuffle_hls_56/sol1/impl/ip",
        hw_root + "/code_gen_ipgen_FINNLoop_0_pzwjxa2b/ip",
        hw_root + "/vivado_stitch_proj_loop/ip",
    ]

    collapsed = _collapse_ip_repo_paths(ip_dirs)

    assert collapsed == ip_dirs


def test_fpgadataflow_ipstitch_can_collapse_generated_ip_repo_roots():
    hw_root = "/scratch/work/finn_temp_file_siglip_hwipgen"
    fifo_root = "/scratch/work/finn_temp_file_siglip_postfifo"
    ip_dirs = [
        "list",
        "$::env(FINN_ROOT)/finn-rtllib/memstream",
        fifo_root + "/code_gen_ipgen_StreamingFIFO_rtl_0_abcd",
        fifo_root + "/code_gen_ipgen_StreamingFIFO_rtl_1_efgh",
        hw_root + "/code_gen_ipgen_DuplicateStreams_hls_18_abcd/project_DuplicateStreams_hls_18/sol1/impl/ip",
        hw_root + "/code_gen_ipgen_OuterShuffle_hls_56_efgh/project_OuterShuffle_hls_56/sol1/impl/ip",
        hw_root + "/code_gen_ipgen_FINNLoop_0_pzwjxa2b/ip",
        hw_root + "/vivado_stitch_proj_loop/ip",
    ]

    collapsed = _collapse_ip_repo_paths(ip_dirs, collapse_generated=True)

    assert collapsed == [
        "list",
        "$::env(FINN_ROOT)/finn-rtllib/memstream",
        fifo_root,
        hw_root,
    ]


def test_finnloop_ip_repo_paths_collapse_generated_roots():
    loop_root = "/scratch/work/finn_temp_file_siglip_loop"
    ip_dirs = [
        "list",
        "$::env(FINN_ROOT)/finn-rtllib/memstream",
        loop_root + "/code_gen_ipgen_FINNLoop_0_MVAU_rtl_0_abcd",
        loop_root + "/code_gen_ipgen_FINNLoop_0_DuplicateStreams_hls_0/proj/sol1/impl/ip",
        loop_root + "/vivado_stitch_proj_loop/ip",
    ]

    collapsed = _collapse_loop_ip_repo_paths(ip_dirs)
    cmd = _ip_repo_paths_append_cmd(ip_dirs)

    assert collapsed == [
        "list",
        "$::env(FINN_ROOT)/finn-rtllib/memstream",
        loop_root,
    ]
    assert "[list $::env(FINN_ROOT)/finn-rtllib/memstream %s]" % loop_root in cmd
    assert cmd.count(loop_root) == 1
    assert "code_gen_ipgen" not in cmd
    assert "vivado_stitch_proj_loop" not in cmd


def test_finnloop_ip_repo_paths_can_preserve_exact_loop_body_repos():
    loop_root = "/scratch/work/finn_temp_file_siglip_loop"
    ip_dirs = [
        "list",
        "$::env(FINN_ROOT)/finn-rtllib/memstream",
        loop_root + "/code_gen_ipgen_FINNLoop_0_MVAU_rtl_0_abcd",
        loop_root + "/code_gen_ipgen_FINNLoop_0_DuplicateStreams_hls_0/proj/sol1/impl/ip",
        loop_root + "/vivado_stitch_proj_loop/ip",
    ]

    exact = _collapse_loop_ip_repo_paths(ip_dirs, collapse_generated=False)
    cmd = _ip_repo_paths_set_cmd(ip_dirs, collapse_generated=False)

    assert exact == ip_dirs
    assert loop_root + "/code_gen_ipgen_FINNLoop_0_MVAU_rtl_0_abcd" in cmd
    assert loop_root + "/vivado_stitch_proj_loop/ip" in cmd
    assert "set_property ip_repo_paths [list" in cmd


def test_finnloop_copy_to_add_files_is_retry_idempotent():
    cmd = _add_files_copy_to_cmd("./ip/verilog/rtl_ops/FINNLoop_0", "/tmp/IN_3_stream_tap_wrapper.v")

    assert cmd == (
        "add_files -force -copy_to ./ip/verilog/rtl_ops/FINNLoop_0 "
        "-norecurse /tmp/IN_3_stream_tap_wrapper.v"
    )


def test_finnloop_exposes_clk2x_when_loop_body_uses_it():
    class FakeLoopBody:
        def get_metadata_prop(self, name):
            assert name == "vivado_stitch_ifnames"
            return repr(
                {
                    "clk": ["ap_clk"],
                    "clk2x": ["ap_clk2x"],
                    "rst": ["ap_rst_n"],
                    "s_axis": [],
                    "m_axis": [],
                    "aximm": [["m_axi_MVAU_id_0", "64"]],
                    "axilite": [],
                    "ap_none": [],
                }
            )

    class FakeFINNLoop:
        def get_instream_width_padded(self, _index):
            return 8

        def get_outstream_width_padded(self, _index):
            return 8

        def _dynamic_loop_inputs(self):
            return []

        def get_nodeattr(self, name):
            assert name == "body"
            return FakeLoopBody()

    ifnames = FINNLoop.get_verilog_top_module_intf_names(FakeFINNLoop())

    assert ifnames["clk2x"] == ["ap_clk2x"]
    assert ["m_axi_MVAU_id_0", "64"] in ifnames["aximm"]


def test_ipstitch_treats_clk2x_interface_as_double_pumped(monkeypatch):
    class FakeNodeInst:
        def get_verilog_top_module_intf_names(self):
            return {"clk": ["ap_clk"], "rst": ["ap_rst_n"], "clk2x": ["ap_clk2x"]}

    import finn.transformation.fpgadataflow.create_stitched_ip as stitch_mod

    monkeypatch.setattr(stitch_mod, "getHWCustomOp", lambda _node, _model: FakeNodeInst())

    stitcher = CreateStitchedIP(test_fpga_part, 5)
    node = SimpleNamespace(name="FINNLoop_0", op_type="FINNLoop")

    assert stitcher.is_double_pumped(node, model=None)


def test_ipstitch_pnr_clk2x_is_external_primary_clock():
    cmds = _pnr_clock_constraint_cmds(3.3333333333333335, has_clk2x=True)

    assert "create_clock -name ap_clk -period 3.333333 $ap_clk_port" in cmds
    assert "create_clock -name ap_clk2x -period 1.666667 $ap_clk2x_port" in cmds
    assert not any("create_generated_clock -name ap_clk2x" in cmd for cmd in cmds)
    assert any("delete_clocks $ap_clk_existing" in cmd for cmd in cmds)
    assert any("delete_clocks $ap_clk2x_existing" in cmd for cmd in cmds)


def create_one_fc_model(mem_mode="internal_embedded"):
    # create a model with a MatrixVectorActivation instance with no activation
    # the wider range of the full accumulator makes debugging a bit easier
    wdt = DataType["INT2"]
    idt = DataType["INT32"]
    odt = DataType["INT32"]
    m = 4
    no_act = 1
    binary_xnor_mode = 0
    actval = 0
    simd = 4
    pe = 4

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, m])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, m])

    fc0 = helper.make_node(
        "MVAU_hls",
        ["inp", "w0"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        MW=m,
        MH=m,
        SIMD=simd,
        PE=pe,
        inputDataType=idt.name,
        weightDataType=wdt.name,
        outputDataType=odt.name,
        ActVal=actval,
        binaryXnorMode=binary_xnor_mode,
        noActivation=no_act,
        mem_mode=mem_mode,
    )

    graph = helper.make_graph(nodes=[fc0], name="fclayer_graph", inputs=[inp], outputs=[outp])

    model = qonnx_make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("w0", wdt)

    # generate weights
    w0 = np.eye(m, dtype=np.float32)
    model.set_initializer("w0", w0)

    model = model.transform(CreateDataflowPartition())
    return model


def create_two_fc_model(mem_mode="internal_decoupled"):
    # create a model with two MatrixVectorActivation instances
    wdt = DataType["INT2"]
    idt = DataType["INT32"]
    odt = DataType["INT32"]
    m = 4
    actval = 0
    no_act = 1
    binary_xnor_mode = 0
    pe = 2
    simd = 2

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, m])
    mid = helper.make_tensor_value_info("mid", TensorProto.FLOAT, [1, m])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, m])

    fc0 = helper.make_node(
        "MVAU_hls",
        ["inp", "w0"],
        ["mid"],
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        MW=m,
        MH=m,
        SIMD=simd,
        PE=pe,
        inputDataType=idt.name,
        weightDataType=wdt.name,
        outputDataType=odt.name,
        ActVal=actval,
        binaryXnorMode=binary_xnor_mode,
        noActivation=no_act,
        mem_mode=mem_mode,
    )

    fc1 = helper.make_node(
        "MVAU_hls",
        ["mid", "w1"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        MW=m,
        MH=m,
        SIMD=simd,
        PE=pe,
        inputDataType=idt.name,
        weightDataType=wdt.name,
        outputDataType=odt.name,
        ActVal=actval,
        binaryXnorMode=binary_xnor_mode,
        noActivation=no_act,
        mem_mode=mem_mode,
    )

    graph = helper.make_graph(
        nodes=[fc0, fc1],
        name="fclayer_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[mid],
    )

    model = qonnx_make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("mid", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("w0", wdt)
    model.set_tensor_datatype("w1", wdt)

    # generate weights
    w0 = np.eye(m, dtype=np.float32)
    w1 = np.eye(m, dtype=np.float32)
    model.set_initializer("w0", w0)
    model.set_initializer("w1", w1)

    model = model.transform(CreateDataflowPartition())
    return model


@pytest.mark.parametrize("mem_mode", ["internal_embedded", "internal_decoupled"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_fpgadataflow_ipstitch_gen_model(mem_mode):
    model = create_one_fc_model(mem_mode)
    if model.graph.node[0].op_type == "StreamingDataflowPartition":
        sdp_node = getHWCustomOp(model.graph.node[0])
        assert sdp_node.__class__.__name__ == "StreamingDataflowPartition"
        assert os.path.isfile(sdp_node.get_nodeattr("model"))
        model = load_test_checkpoint_or_skip(sdp_node.get_nodeattr("model"))
    model = model.transform(InsertTLastMarker())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(test_fpga_part, 5))
    model = model.transform(HLSSynthIP())
    assert model.graph.node[0].op_type == "MVAU_hls"
    assert model.graph.node[-1].op_type == "TLastMarker_hls"
    model.save(ip_stitch_model_dir + "/test_fpgadataflow_ipstitch_gen_model_%s.onnx" % mem_mode)


@pytest.mark.parametrize("mem_mode", ["internal_embedded", "internal_decoupled"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_ipstitch_do_stitch(mem_mode):
    model = load_test_checkpoint_or_skip(
        ip_stitch_model_dir + "/test_fpgadataflow_ipstitch_gen_model_%s.onnx" % mem_mode
    )
    # Run CreateStitchedIP with run_pnr=True to also get OOC synthesis results
    model = model.transform(CreateStitchedIP(test_fpga_part, 5, run_pnr=True))

    # Check IP stitching outputs
    vivado_stitch_proj_dir = model.get_metadata_prop("vivado_stitch_proj")
    assert vivado_stitch_proj_dir is not None
    assert os.path.isdir(vivado_stitch_proj_dir)
    assert os.path.isfile(vivado_stitch_proj_dir + "/ip/component.xml")
    vivado_stitch_vlnv = model.get_metadata_prop("vivado_stitch_vlnv")
    assert vivado_stitch_vlnv is not None
    assert vivado_stitch_vlnv == "xilinx_finn:finn:finn_design:1.0"

    # Check OOC synthesis results
    ret = parse_ooc_synth_results(vivado_stitch_proj_dir)
    assert ret is not None
    # example expected output: (details may differ based on Vivado version etc)
    # {'LUT': 708, 'FF': 1516, 'DSP': 0, 'BRAM_18K': 0, 'BRAM_36K': 0,
    # 'WNS': 0.152, 'fmax_mhz': 206.27}
    assert ret["LUT"] > 0
    assert ret["FF"] > 0
    assert ret["DSP"] == 0
    assert ret.get("BRAM_18K", 0) == 0
    assert ret.get("BRAM_36K", 0) == 0
    assert ret["fmax_mhz"] > 100

    model.save(ip_stitch_model_dir + "/test_fpgadataflow_ip_stitch_%s.onnx" % mem_mode)


@pytest.mark.parametrize("mem_mode", ["internal_embedded", "internal_decoupled"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_fpgadataflow_ipstitch_rtlsim(mem_mode):
    model = load_test_checkpoint_or_skip(
        ip_stitch_model_dir + "/test_fpgadataflow_ip_stitch_%s.onnx" % mem_mode
    )
    model.set_metadata_prop("rtlsim_trace", "whole_trace.wdb")
    model.set_metadata_prop("exec_mode", "rtlsim")
    idt = model.get_tensor_datatype("inp")
    ishape = model.get_tensor_shape("inp")
    x = gen_finn_dt_tensor(idt, ishape)
    # x = np.zeros(ishape, dtype=np.float32)
    # x = np.asarray([[-2, -1, 0, 1]], dtype=np.float32)
    rtlsim_res = execute_onnx(model, {"inp": x})["outp"]
    assert (rtlsim_res == x).all()


@pytest.mark.fpgadataflow
def test_fpgadataflow_ipstitch_iodma_floorplan():
    model = create_one_fc_model()
    if model.graph.node[0].op_type == "StreamingDataflowPartition":
        sdp_node = getHWCustomOp(model.graph.node[0])
        assert sdp_node.__class__.__name__ == "StreamingDataflowPartition"
        assert os.path.isfile(sdp_node.get_nodeattr("model"))
        model = load_test_checkpoint_or_skip(sdp_node.get_nodeattr("model"))
    model = model.transform(InferDataLayouts())
    model = model.transform(InsertIODMA())
    model = model.transform(Floorplan())
    assert getHWCustomOp(model.graph.node[0]).get_nodeattr("partition_id") == 0
    assert getHWCustomOp(model.graph.node[1]).get_nodeattr("partition_id") == 2
    assert getHWCustomOp(model.graph.node[2]).get_nodeattr("partition_id") == 1
    model.save(ip_stitch_model_dir + "/test_fpgadataflow_ipstitch_iodma_floorplan.onnx")


# board
@pytest.mark.parametrize("board", ["U250"])
# clock period
@pytest.mark.parametrize("period_ns", [5])
# override mem_mode to external
@pytest.mark.parametrize("extw", [True, False])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.vitis
def test_fpgadataflow_ipstitch_vitis_end2end(board, period_ns, extw):
    if "VITIS_PATH" not in os.environ:
        pytest.skip("VITIS_PATH not set")
    platform = vitis_default_platform[board]
    fpga_part = vitis_part_map[board]
    model = create_two_fc_model("external" if extw else "internal_decoupled")
    if model.graph.node[0].op_type == "StreamingDataflowPartition":
        sdp_node = getHWCustomOp(model.graph.node[0])
        assert sdp_node.__class__.__name__ == "StreamingDataflowPartition"
        assert os.path.isfile(sdp_node.get_nodeattr("model"))
        model = load_test_checkpoint_or_skip(sdp_node.get_nodeattr("model"))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(fpga_part, period_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareForLinking(fpga_part, period_ns, "vitis-xrt"))
    model = model.transform(VitisLink(platform, period_ns))
    model.save(ip_stitch_model_dir + "/test_fpgadataflow_ipstitch_vitis.onnx")
    assert model.get_metadata_prop("platform") == "vitis-xrt"
    assert os.path.isdir(model.get_metadata_prop("vitis_link_proj"))
    assert os.path.isfile(model.get_metadata_prop("bitfile"))


# board
@pytest.mark.parametrize("board", ["Pynq-Z1"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_ipstitch_zynqbuild_end2end(board):
    model = create_two_fc_model()
    if model.graph.node[0].op_type == "StreamingDataflowPartition":
        sdp_node = getHWCustomOp(model.graph.node[0])
        assert sdp_node.__class__.__name__ == "StreamingDataflowPartition"
        assert os.path.isfile(sdp_node.get_nodeattr("model"))
        model = load_test_checkpoint_or_skip(sdp_node.get_nodeattr("model"))
    # bitfile using ZynqBuild
    model = model.transform(ZynqBuild(board, 10))
    model.save(ip_stitch_model_dir + "/test_fpgadataflow_ipstitch_customzynq.onnx")

    bitfile_name = model.get_metadata_prop("bitfile")
    assert bitfile_name is not None
    assert os.path.isfile(bitfile_name)
