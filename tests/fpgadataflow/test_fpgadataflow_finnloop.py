import pytest

import glob
import numpy as np
import os
import re
import shutil
import subprocess
from dataclasses import replace
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveUnusedTensors,
)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.util.basic import gen_finn_dt_tensor, get_by_name, qonnx_make_model

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import finn.core.onnx_exec as oxe
from finn.custom_op.fpgadataflow.rtl.finn_loop import _bd_clock_freq_hz_cmds
from finn.core.rtlsim_exec import rtlsim_exec
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.derive_characteristic import (
    DeriveCharacteristic,
    DeriveFIFOSizes,
)
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.loop_rolling import LoopExtraction, LoopRolling
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.set_fifo_depths import (
    InsertAndSetFIFODepths,
    RemoveShallowFIFOs,
    SplitLargeFIFOs,
)
from finn.transformation.fpgadataflow.set_loop_boundary import SetLoopBoundary
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import make_build_dir
from finn.util.mlo_sim import mlo_prehook_func_factory

verif_steps = [
    "folded_hls_cppsim",
    "node_by_node_rtlsim",
    "stitched_ip_rtlsim",
]

fpga_part = "xcvc1902-vsva2197-2MP-e-S"
clk_ns = 5


def test_finnloop_bd_clock_freq_hz_cmds():
    cmds = _bd_clock_freq_hz_cmds(10.0 / 3.0, has_clk2x=True)
    assert "set_property CONFIG.FREQ_HZ 300000000 [get_bd_ports /ap_clk]" in cmds
    assert "set_property CONFIG.FREQ_HZ 600000000 [get_bd_ports /ap_clk2x]" in cmds

    cmds = _bd_clock_freq_hz_cmds(5.0, has_clk2x=False)
    assert cmds == ["set_property CONFIG.FREQ_HZ 200000000 [get_bd_ports /ap_clk]"]


def test_finnloop_overlapped_scheduler_estimate_is_artifact_guarded(tmp_path):
    body = helper.make_graph([], "empty_loop_body", [], [])
    loop_node = helper.make_node(
        "FINNLoop",
        ["in0"],
        ["out0"],
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        body=body,
        iteration=3,
        inputDataType="INT8",
        outputDataType="INT8",
        name="FINNLoop_0",
    )
    loop_inst = getCustomOp(loop_node)

    assert loop_inst.get_nodeattr("loop_scheduler_mode") == "sequential"
    assert loop_inst.get_nodeattr("overlapped_scheduler_dcp_ready") == 0
    assert loop_inst._overlapped_exp_cycles() is None

    loop_inst.set_nodeattr("loop_scheduler_mode", "overlapped")
    loop_inst.set_nodeattr("overlapped_body_ii_cycles", 7)
    with pytest.raises(RuntimeError, match="existing graph/RTL artifact"):
        loop_inst.get_exp_cycles()

    non_rtl_artifact = tmp_path / "scheduler.json"
    non_rtl_artifact.write_text("{}\n")
    loop_inst.set_nodeattr("overlapped_scheduler_artifact", str(non_rtl_artifact))
    with pytest.raises(RuntimeError, match="existing graph/RTL artifact"):
        loop_inst.get_exp_cycles()

    rtl_artifact = tmp_path / "scheduler.sv"
    rtl_artifact.write_text("module scheduler; endmodule\n")
    loop_inst.set_nodeattr("overlapped_scheduler_artifact", str(rtl_artifact))

    assert loop_inst.get_exp_cycles() == (7 + 40) * 3

    with pytest.raises(RuntimeError, match="overlapped_scheduler_dcp_ready=1"):
        loop_inst._overlapped_scheduler_wrapper_source()


def test_finnloop_overlapped_hdl_codegen_requires_real_wrapper(tmp_path):
    body = helper.make_graph([], "empty_loop_body", [], [])
    loop_node = helper.make_node(
        "FINNLoop",
        ["in0"],
        ["out0"],
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        body=body,
        iteration=3,
        inputDataType="INT8",
        outputDataType="INT8",
        name="FINNLoop_0",
    )
    loop_inst = getCustomOp(loop_node)
    loop_inst.set_nodeattr("code_gen_dir_ipgen", str(tmp_path / "codegen"))
    loop_inst.set_nodeattr("loop_scheduler_mode", "overlapped")
    loop_inst.set_nodeattr("overlapped_body_ii_cycles", 7)

    contract = tmp_path / "contract.sv"
    contract.write_text("module FINNLoop_0_overlapped_scheduler_contract; endmodule\n")
    loop_inst.set_nodeattr("overlapped_scheduler_artifact", str(contract))
    loop_inst.set_nodeattr("overlapped_scheduler_dcp_ready", 1)
    with pytest.raises(RuntimeError, match="must define module FINNLoop_0_loop_cont_wrapper"):
        loop_inst._overlapped_scheduler_wrapper_source()

    wrapper = tmp_path / "wrapper.sv"
    wrapper.write_text("module FINNLoop_0_loop_cont_wrapper; endmodule\n")
    loop_inst.set_nodeattr("overlapped_scheduler_artifact", str(wrapper))

    assert loop_inst._overlapped_scheduler_wrapper_source() == wrapper
    assert loop_inst._write_overlapped_scheduler_wrapper() is True
    generated = tmp_path / "codegen" / "FINNLoop_0_wrapper.sv"
    assert generated.read_text() == wrapper.read_text()


def test_finnloop_builtin_stream_feedback_codegen(tmp_path, monkeypatch):
    monkeypatch.setenv("FINN_ROOT", os.getcwd())

    inp = helper.make_tensor_value_info("loop_in", TensorProto.FLOAT, [1, 4])
    outp = helper.make_tensor_value_info("loop_out", TensorProto.FLOAT, [1, 4])
    body = helper.make_graph([], "empty_loop_body", [inp], [outp])
    body_model = ModelWrapper(qonnx_make_model(body))
    body_model.set_tensor_datatype("loop_in", DataType["INT8"])
    body_model.set_tensor_datatype("loop_out", DataType["INT8"])
    body_model.set_metadata_prop(
        "vivado_stitch_ifnames",
        str({"s_axis": [("s_axis_0", 8)], "aximm": [], "clk2x": ["ap_clk2x"]}),
    )
    body_model.set_metadata_prop("vivado_stitch_proj", str(tmp_path / "body_proj"))
    body_model.set_metadata_prop("vivado_stitch_vlnv", "xilinx.com:finn:body:1.0")

    loop_node = helper.make_node(
        "FINNLoop",
        ["in0"],
        ["out0"],
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        body=body_model.graph,
        iteration=3,
        inputDataType="INT8",
        outputDataType="INT8",
        name="FINNLoop_0",
    )
    loop_inst = getCustomOp(loop_node)
    codegen_dir = tmp_path / "codegen"
    codegen_dir.mkdir()
    loop_inst.set_nodeattr("code_gen_dir_ipgen", str(codegen_dir))
    loop_inst.set_nodeattr("loop_scheduler_mode", "overlapped")
    loop_inst.set_nodeattr("overlapped_body_ii_cycles", 7)
    loop_inst.set_nodeattr("overlapped_scheduler_artifact", "stream_feedback")

    with pytest.raises(RuntimeError, match="existing graph/RTL artifact"):
        loop_inst.get_exp_cycles()
    with pytest.raises(RuntimeError, match="overlapped_scheduler_dcp_ready=1"):
        loop_inst._overlapped_scheduler_wrapper_source()

    loop_inst.set_nodeattr("overlapped_scheduler_dcp_ready", 1)
    assert loop_inst.get_exp_cycles() == (7 + 40) * 3
    assert loop_inst._overlapped_scheduler_wrapper_source() is None

    loop_inst.generate_hdl(body_model, fpga_part, clk_ns)
    generated = (codegen_dir / "FINNLoop_0_wrapper.v").read_text()
    assert "overlapped_loop_control #(" in generated
    assert re.search(r"\n\s+loop_control #\(", generated) is None

    ipgen_cmd = "\n".join(
        loop_inst._intermediate_frame_dwc_ipgen_cmds()
        + loop_inst._loop_shell_source_ipgen_cmds()
    )
    assert "-module_name if_dwc_feedback" in ipgen_cmd
    assert "finn-rtllib/mlo/overlapped_loop_control.sv" in ipgen_cmd
    assert "finn-rtllib/mlo/infrastructure/streamed_feedback_frames.sv" in ipgen_cmd
    assert loop_inst.get_verilog_top_module_intf_names()["clk2x"] == ["ap_clk2x"]


@pytest.mark.vivado
def test_finnloop_stream_feedback_rtl_xsim(tmp_path):
    if shutil.which("xvlog") is None or shutil.which("xelab") is None or shutil.which("xsim") is None:
        pytest.skip("Vivado xsim tools are not on PATH")

    repo_root = os.getcwd()
    dwc_stub = tmp_path / "if_dwc_feedback_stub.sv"
    dwc_stub.write_text(
        """\
module if_dwc_feedback (
    input  logic       aclk,
    input  logic       aresetn,
    input  logic       s_axis_tvalid,
    output logic       s_axis_tready,
    input  logic [7:0] s_axis_tdata,
    input  logic [0:0] s_axis_tkeep,
    input  logic       s_axis_tlast,
    output logic       m_axis_tvalid,
    input  logic       m_axis_tready,
    output logic [7:0] m_axis_tdata,
    output logic [0:0] m_axis_tkeep,
    output logic       m_axis_tlast
);
    assign s_axis_tready = m_axis_tready;
    assign m_axis_tvalid = s_axis_tvalid;
    assign m_axis_tdata = s_axis_tdata;
    assign m_axis_tkeep = s_axis_tkeep;
    assign m_axis_tlast = s_axis_tlast;
endmodule
"""
    )
    tb = tmp_path / "tb_streamed_feedback_frames.sv"
    tb.write_text(
        """\
module tb_streamed_feedback_frames;
    logic clk = 1'b0;
    logic rstn = 1'b0;
    always #5 clk = ~clk;

    logic [7:0] s_idx_tdata;
    logic s_idx_tvalid;
    logic s_idx_tready;
    logic [7:0] m_idx_tdata;
    logic m_idx_tvalid;
    logic m_idx_tready;

    logic [7:0] s_axis_tdata;
    logic s_axis_tvalid;
    logic s_axis_tready;
    logic [7:0] m_axis_tdata;
    logic m_axis_tvalid;
    logic m_axis_tready;

    streamed_feedback_frames #(
        .ILEN_BITS(8),
        .OLEN_BITS(8),
        .IDX_BITS(8),
        .FM_SIZE(4),
        .N_DCPL_STGS(0)
    ) dut (
        .aclk(clk),
        .aresetn(rstn),
        .s_idx_tdata(s_idx_tdata),
        .s_idx_tvalid(s_idx_tvalid),
        .s_idx_tready(s_idx_tready),
        .m_idx_tdata(m_idx_tdata),
        .m_idx_tvalid(m_idx_tvalid),
        .m_idx_tready(m_idx_tready),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready)
    );

    logic [31:0] unused_axi;
    logic [7:0] unused_stream;
    overlapped_loop_control #(
        .FM_SIZE(4),
        .N_LAYERS(3),
        .ILEN_BITS(8),
        .OLEN_BITS(8),
        .IDX_BITS(8),
        .ADDR_BITS(32),
        .DATA_BITS(32),
        .LEN_BITS(16)
    ) shell_elab (
        .aclk(clk),
        .aresetn(rstn),
        .m_axi_hbm_araddr(),
        .m_axi_hbm_arburst(),
        .m_axi_hbm_arcache(),
        .m_axi_hbm_arid(),
        .m_axi_hbm_arlen(),
        .m_axi_hbm_arlock(),
        .m_axi_hbm_arprot(),
        .m_axi_hbm_arsize(),
        .m_axi_hbm_arready(1'b1),
        .m_axi_hbm_arvalid(),
        .m_axi_hbm_awaddr(),
        .m_axi_hbm_awburst(),
        .m_axi_hbm_awcache(),
        .m_axi_hbm_awid(),
        .m_axi_hbm_awlen(),
        .m_axi_hbm_awlock(),
        .m_axi_hbm_awprot(),
        .m_axi_hbm_awsize(),
        .m_axi_hbm_awready(1'b1),
        .m_axi_hbm_awvalid(),
        .m_axi_hbm_rdata(32'h0),
        .m_axi_hbm_rid(2'b0),
        .m_axi_hbm_rlast(1'b0),
        .m_axi_hbm_rresp(2'b0),
        .m_axi_hbm_rready(),
        .m_axi_hbm_rvalid(1'b0),
        .m_axi_hbm_wdata(),
        .m_axi_hbm_wlast(),
        .m_axi_hbm_wstrb(),
        .m_axi_hbm_wready(1'b1),
        .m_axi_hbm_wvalid(),
        .m_axi_hbm_bid(2'b0),
        .m_axi_hbm_bresp(2'b0),
        .m_axi_hbm_bready(),
        .m_axi_hbm_bvalid(1'b0),
        .m_axis_core_tdata(),
        .m_axis_core_tvalid(),
        .m_axis_core_tready(1'b1),
        .s_axis_core_tdata(8'h0),
        .s_axis_core_tvalid(1'b0),
        .s_axis_core_tready(),
        .m_idx_tdata(),
        .m_idx_tvalid(),
        .m_idx_tready(1'b1),
        .s_idx_tdata(8'h0),
        .s_idx_tvalid(1'b0),
        .s_idx_tready(),
        .s_axis_fs_tdata(8'h0),
        .s_axis_fs_tvalid(1'b0),
        .s_axis_fs_tready(),
        .m_axis_se_tdata(),
        .m_axis_se_tvalid(),
        .m_axis_se_tready(1'b1)
    );

    task automatic send_idx(input logic [7:0] value);
        s_idx_tdata = value;
        s_idx_tvalid = 1'b1;
        @(posedge clk);
        while (!s_idx_tready) @(posedge clk);
        s_idx_tvalid = 1'b0;
    endtask

    task automatic send_word(input logic [7:0] value);
        s_axis_tdata = value;
        s_axis_tvalid = 1'b1;
        @(posedge clk);
        while (!s_axis_tready) @(posedge clk);
        s_axis_tvalid = 1'b0;
    endtask

    task automatic expect_idx(input logic [7:0] value);
        @(posedge clk);
        while (!m_idx_tvalid) @(posedge clk);
        if (m_idx_tdata !== value) begin
            $error("expected idx %0d, got %0d", value, m_idx_tdata);
            $fatal;
        end
    endtask

    task automatic expect_word(input logic [7:0] value);
        @(posedge clk);
        while (!m_axis_tvalid) @(posedge clk);
        if (m_axis_tdata !== value) begin
            $error("expected word %0d, got %0d", value, m_axis_tdata);
            $fatal;
        end
    endtask

    initial begin
        s_idx_tdata = '0;
        s_idx_tvalid = 1'b0;
        m_idx_tready = 1'b1;
        s_axis_tdata = '0;
        s_axis_tvalid = 1'b0;
        m_axis_tready = 1'b1;

        repeat (4) @(posedge clk);
        rstn = 1'b1;
        repeat (2) @(posedge clk);

        fork
            send_idx(8'd4);
            expect_idx(8'd5);
        join

        fork
            begin
                send_word(8'h10);
                send_word(8'h11);
                send_word(8'h12);
                send_word(8'h13);
            end
            begin
                expect_word(8'h10);
                expect_word(8'h11);
                expect_word(8'h12);
                expect_word(8'h13);
            end
        join

        repeat (4) @(posedge clk);
        $display("PASS streamed_feedback_frames");
        $finish;
    end
endmodule
"""
    )

    sources = [
        os.path.join(repo_root, "finn-rtllib/fifo/hdl/Q_srl.v"),
        os.path.join(repo_root, "finn-rtllib/skid/skid.sv"),
        os.path.join(repo_root, "finn-rtllib/mlo/infrastructure/mux.sv"),
        os.path.join(repo_root, "finn-rtllib/mlo/infrastructure/demux.sv"),
        str(dwc_stub),
        os.path.join(repo_root, "finn-rtllib/mlo/infrastructure/streamed_feedback_frames.sv"),
        os.path.join(repo_root, "finn-rtllib/mlo/overlapped_loop_control.sv"),
        str(tb),
    ]
    run_env = os.environ.copy()
    run_env.setdefault("XILINX_LOCAL_USER_DATA", "no")
    for cmd in (
        ["xvlog", "-sv", "-nolog", "-work", "work"] + sources,
        ["xelab", "-nolog", "tb_streamed_feedback_frames", "-s", "tb_streamed_feedback_frames_sim"],
        ["xsim", "-nolog", "tb_streamed_feedback_frames_sim", "-runall"],
    ):
        completed = subprocess.run(
            cmd,
            cwd=tmp_path,
            env=run_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout
        if cmd[0] == "xsim":
            assert "PASS streamed_feedback_frames" in completed.stdout


def generate_random_threshold_values(data_type, num_input_channels, num_steps):
    if data_type.is_integer():
        return np.random.randint(
            data_type.min(),
            data_type.max() + 1,
            (num_input_channels, num_steps),
        ).astype(np.float32)
    else:
        return (np.random.randn(num_input_channels, num_steps) * 1000).astype(
            data_type.to_numpy_dt()
        )


def create_tensor_info(name, shape, proto=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, proto, shape)


def create_threshold(name, shape):
    return create_tensor_info(name, shape)


def create_node(node_type, inputs, outputs, name, extra_params={}):
    base_params = {
        "domain": "finn.custom_op.fpgadataflow.rtl"
        if "rtl" in node_type
        else "finn.custom_op.fpgadataflow.hls",
        "backend": "fpgadataflow",
        "numInputVectors": list((1, 3, 3)),
        "name": name,
    }
    return helper.make_node(node_type, inputs, outputs, **{**base_params, **extra_params})


def make_loop_modelwrapper(
    mw,
    mh,
    dtype=DataType["INT8"],
    elemwise_optype="ElementwiseMul_hls",
    rhs_shape=[1],
    eltw_param_dtype="INT8",
    name_suffix="",
    mvau_pe=2,
    mvau_simd=2,
    mvau_th=1,
    helper_pe=2,
):
    is_float = eltw_param_dtype == "FLOAT32"

    # Determine elementwise output dtype
    # HLS elementwise outputs FLOAT32 if parameter is FLOAT32, otherwise INT32
    if is_float:
        elemwise_output_dtype = DataType["FLOAT32"]
        thresholding_input_dtype = DataType["FLOAT32"]
    else:
        elemwise_output_dtype = DataType["INT32"]
        thresholding_input_dtype = DataType["INT32"]

    W0 = gen_finn_dt_tensor(dtype, (mw, mh))
    W1 = gen_finn_dt_tensor(dtype, (mw, mh))
    W2 = gen_finn_dt_tensor(dtype, (mh, mh))
    T0 = np.sort(
        generate_random_threshold_values(dtype, 1, dtype.get_num_possible_values() - 1), axis=1
    )
    T1 = np.sort(
        generate_random_threshold_values(dtype, 1, dtype.get_num_possible_values() - 1), axis=1
    )
    T2 = np.sort(
        generate_random_threshold_values(dtype, 1, dtype.get_num_possible_values() - 1), axis=1
    )
    T3_dtype = dtype
    T3 = np.sort(
        generate_random_threshold_values(T3_dtype, 1, dtype.get_num_possible_values() - 1), axis=1
    )
    EltwParam = gen_finn_dt_tensor(DataType[eltw_param_dtype], rhs_shape)

    tensor_shapes = {
        f"ifm{name_suffix}": [1, 3, 3, mw],
        f"weights{name_suffix}": [mw, mh],
        f"weights2{name_suffix}": [mh, mh],
    }
    output_shapes = {f"mm{name_suffix}": [1, 3, 3, mh], f"ofm{name_suffix}": (1, 3, 3, mh)}

    tensor_infos = {k: create_tensor_info(k, v) for k, v in tensor_shapes.items()}
    thresholds = [
        create_threshold(f"thresh{i}{name_suffix}", (1, dtype.get_num_possible_values() - 1))
        for i in range(4)
    ]

    nodes = [
        create_node(
            "DuplicateStreams_hls",
            [f"ifm{name_suffix}"],
            [f"ifm_1{name_suffix}", f"ifm_2{name_suffix}"],
            f"DuplicateStreams_hls_0{name_suffix}",
            {
                "NumChannels": mh,
                "NumOutputStreams": 2,
                "PE": helper_pe,
                "inputDataType": dtype.name,
                "outFIFODepths": [2, 2],
                "cpp_interface": "hls_vector",
                "hls_style": "freerunning",
            },
        ),
        create_node(
            "MVAU_rtl",
            [f"ifm_1{name_suffix}", f"weights0{name_suffix}"],
            [f"mm0_out{name_suffix}"],
            f"MVAU_rtl_0{name_suffix}",
            {
                "MW": mw,
                "MH": mh,
                "SIMD": mvau_simd,
                "PE": mvau_pe,
                "TH": mvau_th,
                "inputDataType": "INT8",
                "weightDataType": "INT8",
                "outputDataType": "INT32",
                "ActVal": 0,
                "binaryXnorMode": 0,
                "noActivation": 1,
                "mem_mode": "external_mem",
            },
        ),
        create_node(
            "Thresholding_rtl",
            [f"mm0_out{name_suffix}", f"thresh0{name_suffix}"],
            [f"mt0_out{name_suffix}"],
            f"Thresholding_rtl_0{name_suffix}",
            {
                "NumChannels": mh,
                "PE": helper_pe,
                "inputDataType": "INT32",
                "weightDataType": "INT33",
                "outputDataType": dtype.name,
                "ActVal": int(dtype.min()),
                "numSteps": dtype.get_num_possible_values() - 1,
            },
        ),
        create_node(
            "MVAU_rtl",
            [f"mt0_out{name_suffix}", f"weights1{name_suffix}"],
            [f"mm1_out{name_suffix}"],
            f"MVAU_rtl_1{name_suffix}",
            {
                "MW": mw,
                "MH": mh,
                "SIMD": mvau_simd,
                "PE": mvau_pe,
                "TH": mvau_th,
                "inputDataType": "INT8",
                "weightDataType": "INT8",
                "outputDataType": "INT32",
                "ActVal": 0,
                "binaryXnorMode": 0,
                "noActivation": 1,
                "mem_mode": "external_mem",
            },
        ),
        create_node(
            "Thresholding_rtl",
            [f"mm1_out{name_suffix}", f"thresh1{name_suffix}"],
            [f"mt1_out{name_suffix}"],
            f"Thresholding_rtl_1{name_suffix}",
            {
                "NumChannels": mh,
                "PE": helper_pe,
                "inputDataType": "INT32",
                "weightDataType": "INT33",
                "outputDataType": dtype.name,
                "ActVal": int(dtype.min()),
                "numSteps": dtype.get_num_possible_values() - 1,
            },
        ),
        create_node(
            "MVAU_rtl",
            [f"ifm_2{name_suffix}", f"weights2{name_suffix}"],
            [f"mm2_out{name_suffix}"],
            f"MVAU_rtl_2{name_suffix}",
            {
                "MW": mw,
                "MH": mh,
                "SIMD": mvau_simd,
                "PE": mvau_pe,
                "TH": mvau_th,
                "inputDataType": "INT8",
                "weightDataType": "INT8",
                "outputDataType": "INT32",
                "ActVal": 0,
                "binaryXnorMode": 0,
                "noActivation": 1,
                "mem_mode": "external_mem",
            },
        ),
        create_node(
            "Thresholding_rtl",
            [f"mm2_out{name_suffix}", f"thresh2{name_suffix}"],
            [f"mt2_out{name_suffix}"],
            f"Thresholding_rtl_2{name_suffix}",
            {
                "NumChannels": mh,
                "PE": helper_pe,
                "inputDataType": "INT32",
                "weightDataType": "INT33",
                "outputDataType": dtype.name,
                "ActVal": int(dtype.min()),
                "numSteps": dtype.get_num_possible_values() - 1,
            },
        ),
        create_node(
            "ElementwiseAdd_hls",
            [f"mt2_out{name_suffix}", f"mt1_out{name_suffix}"],
            [f"ofm{name_suffix}"],
            f"ElementwiseAdd_hls_0{name_suffix}",
            {
                "lhs_shape": [1, 3, 3, mh],
                "rhs_shape": [1, 3, 3, mh],
                "out_shape": [1, 3, 3, mh],
                "lhs_dtype": dtype.name,
                "rhs_dtype": dtype.name,
                "out_dtype": "INT9",
                "lhs_style": "input",
                "rhs_style": "input",
                "PE": helper_pe,
            },
        ),
        create_node(
            elemwise_optype,
            [f"ofm{name_suffix}", f"mul_param{name_suffix}"],
            [f"ofm_ew{name_suffix}"],
            f"ElementwiseOp{'_rtl' if 'rtl' in elemwise_optype else '_hls'}_0{name_suffix}",
            {
                "lhs_shape": [1, 3, 3, mh],
                "rhs_shape": rhs_shape,
                "out_shape": [1, 3, 3, mh],
                "lhs_dtype": "INT9",
                "rhs_dtype": eltw_param_dtype,
                "out_dtype": elemwise_output_dtype.name,
            },
        ),
    ]

    # Add RTL elementwise node after HLS elementwise when parameter is FLOAT32
    if is_float:
        # Use the same operation type as HLS but with _rtl suffix
        rtl_optype = elemwise_optype.replace("_hls", "_rtl")
        nodes.append(
            create_node(
                rtl_optype,
                [f"ofm_ew{name_suffix}", f"mul_param_rtl{name_suffix}"],
                [f"ofm_ew_rtl{name_suffix}"],
                f"ElementwiseOp_rtl_1{name_suffix}",
                {
                    "lhs_shape": [1, 3, 3, mh],
                    "rhs_shape": rhs_shape,
                    "out_shape": [1, 3, 3, mh],
                    "lhs_dtype": "FLOAT32",
                    "rhs_dtype": "FLOAT32",
                    "out_dtype": "FLOAT32",
                },
            )
        )
        thresholding_input_tensor = f"ofm_ew_rtl{name_suffix}"
    else:
        thresholding_input_tensor = f"ofm_ew{name_suffix}"

    nodes.append(
        create_node(
            "Thresholding_rtl",
            [thresholding_input_tensor, f"thresh3{name_suffix}"],
            [f"ofm_final{name_suffix}"],
            f"Thresholding_rtl4{name_suffix}",
            {
                "NumChannels": mh,
                "PE": helper_pe,
                "numSteps": dtype.get_num_possible_values() - 1,
                "inputDataType": thresholding_input_dtype.name,
                "weightDataType": thresholding_input_dtype.name,
                "outputDataType": dtype.name,
                "ActVal": int(dtype.min()),
            },
        ),
    )

    # Build value_info list
    value_info_list = [
        create_tensor_info(name, output_shapes[f"mm{name_suffix}"])
        for name in [
            f"mm0_out{name_suffix}",
            f"mm1_out{name_suffix}",
            f"mm2_out{name_suffix}",
            f"ifm_1{name_suffix}",
            f"ifm_2{name_suffix}",
        ]
    ] + [
        create_tensor_info(name, output_shapes[f"ofm{name_suffix}"])
        for name in [
            f"mt0_out{name_suffix}",
            f"mt1_out{name_suffix}",
            f"mt2_out{name_suffix}",
            f"ofm{name_suffix}",
            f"ofm_ew{name_suffix}",
        ]
    ]

    # Add RTL elementwise output tensor to value_info if FLOAT32
    if is_float:
        value_info_list.append(
            create_tensor_info(f"ofm_ew_rtl{name_suffix}", output_shapes[f"ofm{name_suffix}"])
        )

    loop_body = helper.make_graph(
        nodes=nodes,
        name=f"matmul_graph{name_suffix}",
        inputs=[tensor_infos[f"ifm{name_suffix}"]] + thresholds,
        outputs=[create_tensor_info(f"ofm_final{name_suffix}", output_shapes[f"ofm{name_suffix}"])],
        value_info=value_info_list,
    )

    loop_body_model = qonnx_make_model(loop_body, producer_name=f"loop-body-model{name_suffix}")
    loop_body_model = ModelWrapper(loop_body_model)

    # Set initializers using generated values
    loop_body_model.set_initializer(f"weights0{name_suffix}", W0)
    loop_body_model.set_initializer(f"weights1{name_suffix}", W1)
    loop_body_model.set_initializer(f"weights2{name_suffix}", W2)
    loop_body_model.set_initializer(f"thresh0{name_suffix}", T0)
    loop_body_model.set_initializer(f"thresh1{name_suffix}", T1)
    loop_body_model.set_initializer(f"thresh2{name_suffix}", T2)
    loop_body_model.set_initializer(f"thresh3{name_suffix}", T3)
    loop_body_model.set_initializer(f"mul_param{name_suffix}", EltwParam)

    # Add RTL elementwise parameter when FLOAT32
    if is_float:
        EltwParamRtl = gen_finn_dt_tensor(DataType["FLOAT32"], rhs_shape)
        loop_body_model.set_initializer(f"mul_param_rtl{name_suffix}", EltwParamRtl)

    # Set tensor datatypes
    tensors = [
        f"weights0{name_suffix}",
        f"weights1{name_suffix}",
        f"weights2{name_suffix}",
        f"thresh0{name_suffix}",
        f"thresh1{name_suffix}",
        f"thresh2{name_suffix}",
        f"ifm{name_suffix}",
        f"ofm_final{name_suffix}",
    ]
    for tensor in tensors:
        loop_body_model.set_tensor_datatype(tensor, dtype)

    loop_body_model.set_tensor_datatype(f"thresh3{name_suffix}", T3_dtype)
    loop_body_model.set_tensor_datatype(f"mul_param{name_suffix}", DataType[eltw_param_dtype])

    # Set RTL elementwise parameter datatype when FLOAT32
    if is_float:
        loop_body_model.set_tensor_datatype(f"mul_param_rtl{name_suffix}", DataType["FLOAT32"])

    return loop_body_model


def make_single_mvau_loop_body(
    mw,
    mh,
    dtype=DataType["INT8"],
    name_suffix="",
    mvau_pe=2,
    mvau_simd=2,
    mvau_th=1,
    helper_pe=2,
):
    """Create a minimal loop body with just MVAU_rtl -> Thresholding_rtl."""

    W0 = gen_finn_dt_tensor(dtype, (mw, mh))
    T0 = np.sort(
        generate_random_threshold_values(dtype, 1, dtype.get_num_possible_values() - 1), axis=1
    )

    nodes = [
        create_node(
            "MVAU_rtl",
            [f"ifm{name_suffix}", f"weights0{name_suffix}"],
            [f"mm0_out{name_suffix}"],
            f"MVAU_rtl_0{name_suffix}",
            {
                "MW": mw,
                "MH": mh,
                "SIMD": mvau_simd,
                "PE": mvau_pe,
                "TH": mvau_th,
                "inputDataType": "INT8",
                "weightDataType": "INT8",
                "outputDataType": "INT32",
                "ActVal": 0,
                "binaryXnorMode": 0,
                "noActivation": 1,
                "mem_mode": "external_mem",
            },
        ),
        create_node(
            "Thresholding_rtl",
            [f"mm0_out{name_suffix}", f"thresh0{name_suffix}"],
            [f"ofm{name_suffix}"],
            f"Thresholding_rtl_0{name_suffix}",
            {
                "NumChannels": mh,
                "PE": helper_pe,
                "inputDataType": "INT32",
                "weightDataType": "INT33",
                "outputDataType": dtype.name,
                "ActVal": int(dtype.min()),
                "numSteps": dtype.get_num_possible_values() - 1,
            },
        ),
    ]

    loop_body = helper.make_graph(
        nodes=nodes,
        name=f"single_mvau_graph{name_suffix}",
        inputs=[
            create_tensor_info(f"ifm{name_suffix}", [1, 3, 3, mw]),
            create_threshold(f"thresh0{name_suffix}", (1, dtype.get_num_possible_values() - 1)),
        ],
        outputs=[create_tensor_info(f"ofm{name_suffix}", (1, 3, 3, mh))],
        value_info=[
            create_tensor_info(f"mm0_out{name_suffix}", [1, 3, 3, mh]),
        ],
    )

    loop_body_model = qonnx_make_model(loop_body, producer_name=f"single-mvau-body{name_suffix}")
    loop_body_model = ModelWrapper(loop_body_model)

    loop_body_model.set_initializer(f"weights0{name_suffix}", W0)
    loop_body_model.set_initializer(f"thresh0{name_suffix}", T0)

    for tensor in [
        f"weights0{name_suffix}",
        f"thresh0{name_suffix}",
        f"ifm{name_suffix}",
        f"ofm{name_suffix}",
    ]:
        loop_body_model.set_tensor_datatype(tensor, dtype)

    return loop_body_model


def create_chained_loop_bodies(
    mw,
    mh,
    num_copies,
    elemwise_optype="ElementwiseMul_hls",
    rhs_shape=[1],
    eltw_param_dtype="INT8",
    mvau_pe=2,
    mvau_simd=2,
    mvau_th=1,
    helper_pe=2,
):
    loop_body_models = []

    # Create multiple instances of the loop body with unique name_suffix
    for i in range(num_copies):
        name_suffix = f"_{i}"
        loop_body_model = make_loop_modelwrapper(
            mw=mw,
            mh=mh,
            dtype=DataType["INT8"],
            elemwise_optype=elemwise_optype,
            rhs_shape=rhs_shape,
            eltw_param_dtype=eltw_param_dtype,
            name_suffix=name_suffix,
            mvau_pe=mvau_pe,
            mvau_simd=mvau_simd,
            mvau_th=mvau_th,
            helper_pe=helper_pe,
        )
        loop_body_models.append(loop_body_model)

    return loop_body_models


# dimensions
@pytest.mark.parametrize("dim", [16])
# iteration count, number of models chained together
@pytest.mark.parametrize("iteration", [3])
# elementwise operation
@pytest.mark.parametrize("elemwise_optype", ["ElementwiseMul_hls", "ElementwiseAdd_hls"])
# elementwise shape
@pytest.mark.parametrize("rhs_shape", [[1], [16]])
# eltwise param dtype
@pytest.mark.parametrize("eltw_param_dtype", ["INT8", "FLOAT32"])
# tail node
@pytest.mark.parametrize("tail_node", [False, True])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_finnloop_end2end_mlo(
    dim, iteration, elemwise_optype, rhs_shape, eltw_param_dtype, tail_node
):
    # Check vivado version
    vivado_path = os.environ.get("XILINX_VIVADO")
    match = re.search(r"\b(20\d{2})\.(1|2)\b", vivado_path)
    year, minor = int(match.group(1)), int(match.group(2))
    if (year, minor) < (2024, 2):
        pytest.skip("""At least Vivado version 2024.2 needed for MLO.""")
    loop_body_models = create_chained_loop_bodies(
        dim, dim, iteration, elemwise_optype, rhs_shape, eltw_param_dtype
    )
    nodes_per_body = len(loop_body_models[0].graph.node)
    model = loop_body_models[0]
    for m in loop_body_models[1:]:
        model = model.transform(MergeONNXModels(m))

    if tail_node:
        tail_outp = create_tensor_info("tail_outp", [1, 3, 3, dim])
        tr_node = create_node(
            "ElementwiseAdd_hls",
            [model.graph.output[0].name, "tail_add"],
            ["tail_outp"],
            "Add_tail",
            {
                "lhs_shape": [1, 3, 3, dim],
                "rhs_shape": [1],
                "out_shape": [1, 3, 3, dim],
                "lhs_dtype": "INT8",
                "rhs_dtype": "INT8",
                "out_dtype": "INT9",
            },
        )
        model.graph.node.insert(len(model.graph.node), tr_node)
        model.graph.value_info.append(model.graph.output[0])
        model.graph.output.pop(0)
        model.graph.output.append(tail_outp)
        AddtailParam = gen_finn_dt_tensor(DataType["INT8"], [1])
        model.set_initializer("tail_add", AddtailParam)
        model.set_tensor_datatype("tail_add", DataType["INT8"])

    # cleanup
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # Generate reference output
    input_dtype = DataType["INT8"]
    x = gen_finn_dt_tensor(input_dtype, (1, 3, 3, dim))
    model_ref = model.transform(PrepareCppSim())
    model_ref = model_ref.transform(CompileCppSim())
    model_ref = model_ref.transform(SetExecMode("cppsim"))
    io_dict = {model_ref.graph.input[0].name: x}
    y_dict = oxe.execute_onnx(model_ref, io_dict)
    y_ref = y_dict[model_ref.graph.output[0].name]

    tmp_output_dir = make_build_dir("build_mlo")

    np.save(tmp_output_dir + "/input.npy", x)
    np.save(tmp_output_dir + "/expected_output.npy", y_ref)

    model.save(tmp_output_dir + "/mlo_model.onnx")

    # steps are skipped because test model created with HLS and RTL layers
    steps = [
        # "step_qonnx_to_finn",
        # "step_tidy_up",
        # "step_streamline",
        # "step_convert_to_hw",
        "step_create_dataflow_partition",
        # "step_specialize_layers",
        "step_loop_rolling",
        # "step_target_fps_parallelization",
        "step_apply_folding_config",
        "step_minimize_bit_width",
        "step_generate_estimate_reports",
        "step_hw_codegen",
        "step_hw_ipgen",
        "step_set_fifo_depths",
        "step_create_stitched_ip",
    ]

    cfg = build_cfg.DataflowBuildConfig(
        output_dir=tmp_output_dir,
        steps=steps,
        synth_clk_period_ns=10.0,
        board="V80",
        rtlsim_batch_size=100,
        standalone_thresholds=True,
        mlo=True,
        loop_body_hierarchy=[["", "layers.0"]],
        loop_body_range=(model.graph.node[0], model.graph.node[nodes_per_body - 1]),
        verify_steps=verif_steps,
        verify_input_npy=tmp_output_dir + "/input.npy",
        verify_expected_output_npy=tmp_output_dir + "/expected_output.npy",
        verify_save_full_context=True,  # Enable per-iteration context saving
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
        ],
    )
    build.build_dataflow_cfg(tmp_output_dir + "/mlo_model.onnx", cfg)

    # check if expected files are there
    assert os.path.isfile(tmp_output_dir + "/loop-body-template.onnx")
    report_dir = tmp_output_dir + "/report"
    assert os.path.isfile(report_dir + "/estimate_layer_config_alternatives_FINNLoop_0.json")
    assert os.path.isfile(report_dir + "/estimate_layer_config_alternatives.json")
    assert os.path.isfile(report_dir + "/estimate_layer_cycles_FINNLoop_0.json")
    assert os.path.isfile(report_dir + "/estimate_layer_cycles.json")
    assert os.path.isfile(report_dir + "/estimate_layer_resources_FINNLoop_0.json")
    assert os.path.isfile(report_dir + "/estimate_layer_resources.json")
    assert os.path.isfile(report_dir + "/op_and_param_counts_FINNLoop_0.json")
    assert os.path.isfile(report_dir + "/op_and_param_counts.json")
    assert os.path.isfile(tmp_output_dir + "/stitched_ip/ip/component.xml")

    verif_dir = tmp_output_dir + "/verification_output"
    # With verify_save_full_context=True, cppsim and node_by_node_rtlsim save as .npz
    # stitched_ip_rtlsim with MLO uses rtlsim_pre_hook so it saves as .npy
    assert os.path.isfile(
        verif_dir + "/verify_folded_hls_cppsim_0_SUCCESS.npz"
    ), f"Check npz files in {verif_dir}"
    assert os.path.isfile(
        verif_dir + "/verify_node_by_node_rtlsim_0_SUCCESS.npz"
    ), f"Check npz files in {verif_dir}"
    assert os.path.isfile(
        verif_dir + "/verify_stitched_ip_rtlsim_0_SUCCESS.npy"
    ), f"Check npy files in {verif_dir}"

    # Verify that the per-iteration context file was created for FINNLoop
    iteration_context_files = [
        f for f in os.listdir(verif_dir) if f.startswith("iteration_context_")
    ]
    assert len(iteration_context_files) > 0, f"No iteration context files found in {verif_dir}"

    # Load and verify the iteration context file has expected structure
    ctx_file = os.path.join(verif_dir, iteration_context_files[0])
    ctx_data = np.load(ctx_file)
    iter_keys = [k for k in ctx_data.files if k.startswith("iter_")]
    assert len(iter_keys) > 0, "No iteration keys found in context file"

    # Verify we have contexts for all iterations
    iter_indices = set()
    for key in iter_keys:
        parts = key.split("_", 2)
        if len(parts) >= 2:
            iter_indices.add(int(parts[1]))
    assert (
        len(iter_indices) == iteration
    ), f"Expected {iteration} iterations in context, found {len(iter_indices)}"

    # also run dcp generation for a subset of the test parameters
    # this extends the test run time quite a lot
    # so only do for 2 of the scenarios

    if (
        elemwise_optype == "ElementwiseMul_hls"
        and rhs_shape == [1]
        and eltw_param_dtype == "FLOAT32"
    ):
        # launch another build just to test dcp generation
        cfg = replace(
            cfg,
            start_step="step_create_stitched_ip",
            stitched_ip_gen_dcp=True,
            verify_steps=[],
        )
        build.build_dataflow_cfg(tmp_output_dir + "/mlo_model.onnx", cfg)

        # check if stitched IP dcp is there
        assert os.path.isfile(
            tmp_output_dir + "/stitched_ip/finn_design.dcp"
        ), f"Check vivado.log in {tmp_output_dir}/stitched_ip"


# iteration count, number of models chained together
@pytest.mark.parametrize("iteration", [3])
# elementwise operation
@pytest.mark.parametrize("elemwise_optype", ["ElementwiseMul_hls"])
# elementwise shape
@pytest.mark.parametrize("rhs_shape", [[1]])
# eltwise param dtype
@pytest.mark.parametrize("eltw_param_dtype", ["INT8"])
# tail node
@pytest.mark.parametrize("tail_node", [False])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_finnloop_end2end_mlo_tiled(
    iteration, elemwise_optype, rhs_shape, eltw_param_dtype, tail_node
):
    """End-to-end MLO test with tiled MVAUs (TH>1)."""
    dim = 12
    mvau_pe = 6
    mvau_simd = 3
    mvau_th = 3
    helper_pe = 6

    # Check vivado version
    vivado_path = os.environ.get("XILINX_VIVADO")
    match = re.search(r"\b(20\d{2})\.(1|2)\b", vivado_path)
    year, minor = int(match.group(1)), int(match.group(2))
    if (year, minor) < (2024, 2):
        pytest.skip("""At least Vivado version 2024.2 needed for MLO.""")
    loop_body_models = create_chained_loop_bodies(
        dim,
        dim,
        iteration,
        elemwise_optype,
        rhs_shape,
        eltw_param_dtype,
        mvau_pe=mvau_pe,
        mvau_simd=mvau_simd,
        mvau_th=mvau_th,
        helper_pe=helper_pe,
    )
    model = loop_body_models[0]
    for m in loop_body_models[1:]:
        model = model.transform(MergeONNXModels(m))

    if tail_node:
        tail_outp = create_tensor_info("tail_outp", [1, 3, 3, dim])
        tr_node = create_node(
            "ElementwiseAdd_hls",
            [model.graph.output[0].name, "tail_add"],
            ["tail_outp"],
            "Add_tail",
            {
                "lhs_shape": [1, 3, 3, dim],
                "rhs_shape": [1],
                "out_shape": [1, 3, 3, dim],
                "lhs_dtype": "INT8",
                "rhs_dtype": "INT8",
                "out_dtype": "INT9",
            },
        )
        model.graph.node.insert(len(model.graph.node), tr_node)
        model.graph.value_info.append(model.graph.output[0])
        model.graph.output.pop(0)
        model.graph.output.append(tail_outp)
        AddtailParam = gen_finn_dt_tensor(DataType["INT8"], [1])
        model.set_initializer("tail_add", AddtailParam)
        model.set_tensor_datatype("tail_add", DataType["INT8"])

    # cleanup
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # Generate reference by first copying the model and running cppsim
    model_ref = model.transform(PrepareCppSim())
    model_ref = model_ref.transform(CompileCppSim())
    model_ref = model_ref.transform(SetExecMode("cppsim"))

    # generate reference io pair
    x = gen_finn_dt_tensor(DataType["INT8"], (1, 3, 3, dim))
    io_dict = {model_ref.graph.input[0].name: x}
    y_dict = oxe.execute_onnx(model_ref, io_dict)
    y_ref = y_dict[model_ref.graph.output[0].name]

    tmp_output_dir = make_build_dir("build_mlo_tiled")

    np.save(tmp_output_dir + "/input.npy", x)
    np.save(tmp_output_dir + "/expected_output.npy", y_ref)

    model.save(tmp_output_dir + "/mlo_model.onnx")

    # steps - skip step_target_fps_parallelization since PE/SIMD/TH already set
    steps = [
        "step_create_dataflow_partition",
        "step_loop_rolling",
        "step_apply_folding_config",
        "step_minimize_bit_width",
        "step_generate_estimate_reports",
        "step_hw_codegen",
        "step_hw_ipgen",
        "step_set_fifo_depths",
        "step_create_stitched_ip",
    ]

    cfg = build_cfg.DataflowBuildConfig(
        output_dir=tmp_output_dir,
        steps=steps,
        target_fps=1000,
        synth_clk_period_ns=10.0,
        board="V80",
        rtlsim_batch_size=100,
        standalone_thresholds=True,
        mlo=True,
        loop_body_hierarchy=[["", "layers.0"]],
        loop_body_range=(model.graph.node[0], model.graph.node[9]),
        verify_steps=verif_steps,
        verify_input_npy=tmp_output_dir + "/input.npy",
        verify_expected_output_npy=tmp_output_dir + "/expected_output.npy",
        verify_save_rtlsim_waveforms=True,
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
        ],
    )
    build.build_dataflow_cfg(tmp_output_dir + "/mlo_model.onnx", cfg)

    # Dump weight files for hardware debug
    built_model = ModelWrapper(tmp_output_dir + "/mlo_model.onnx")
    for node in built_model.graph.node:
        if node.op_type == "FINNLoop":
            fl_op = getCustomOp(node)
            code_gen_dir = fl_op.get_nodeattr("code_gen_dir_ipgen")
            if code_gen_dir and os.path.isdir(code_gen_dir):
                for f in sorted(glob.glob(code_gen_dir + "/input1_*.npy")):
                    arr = np.load(f)
                    base = os.path.basename(f).replace(".npy", "")
                    txt_path = tmp_output_dir + f"/weights_{base}.txt"
                    with open(txt_path, "w") as tf:
                        tf.write(f"# {f}\n# shape: {arr.shape}, dtype: {arr.dtype}\n")
                        tf.write("# row | decimal_values | hex_bytes | bus_word\n\n")
                        flat = arr.reshape(-1, arr.shape[-1])
                        for i, row in enumerate(flat):
                            dec = " ".join(f"{int(v):5d}" for v in row)
                            hx = " ".join(f"{int(v) & 0xFF:02x}" for v in row)
                            bus = 0
                            for j, v in enumerate(row):
                                bus |= (int(v) & 0xFF) << (j * 8)
                            tf.write(f"[{i:3d}] {dec}  | {hx}  | 0x{bus:0{arr.shape[-1]*2}x}\n")
                    print(f"DEBUG: weight dump -> {txt_path}")
                for f in sorted(glob.glob(code_gen_dir + "/memblock_*.dat")):
                    base = os.path.basename(f).replace(".dat", "")
                    txt_path = tmp_output_dir + f"/weights_{base}.txt"
                    with open(txt_path, "w") as tf:
                        tf.write(f"# {f}\n")
                        with open(f) as df:
                            for i, line in enumerate(df):
                                tf.write(f"[{i:3d}] {line.strip()}\n")
                    print(f"DEBUG: dat dump -> {txt_path}")

    # check if expected files are there
    assert os.path.isfile(tmp_output_dir + "/loop-body-template.onnx")
    assert os.path.isfile(tmp_output_dir + "/stitched_ip/ip/component.xml")

    verif_dir = tmp_output_dir + "/verification_output"
    assert os.path.isfile(
        verif_dir + "/verify_folded_hls_cppsim_0_SUCCESS.npy"
    ), f"Check npy files in {verif_dir}"
    assert os.path.isfile(
        verif_dir + "/verify_node_by_node_rtlsim_0_SUCCESS.npy"
    ), f"Check npy files in {verif_dir}"
    assert os.path.isfile(
        verif_dir + "/verify_stitched_ip_rtlsim_0_SUCCESS.npy"
    ), f"Check npy files in {verif_dir}"


# Debug test for manual loop transformation steps below
# This test is intentionally not marked for CI
# Use test_finnloop_end2end_mlo instead
# If required, to run manually:
# pytest tests/fpgadataflow/test_fpgadataflow_finnloop.py::test_fpgadataflow_finnloop


# helper functions
def prepare_loop_ops_for_ipgen_step1(node, fpga_part, clk_ns):
    node_inst = getCustomOp(node)
    loop_model = node_inst.get_nodeattr("body")
    # go first into subgraph to check if there are other loop ops
    loop_nodes = loop_model.get_nodes_by_op_type("FINNLoop")
    for loop_node in loop_nodes:
        prepare_loop_ops_for_ipgen_step1(loop_node, fpga_part, clk_ns)
    loop_model = loop_model.transform(PrepareIP(fpga_part, clk_ns))
    loop_model = loop_model.transform(HLSSynthIP(fpgapart=fpga_part))
    loop_model = loop_model.transform(ReplaceVerilogRelPaths())
    loop_model = loop_model.transform(GiveUniqueNodeNames())
    loop_model = loop_model.transform(GiveReadableTensorNames())
    if node_inst.get_nodeattr("rtlsim_trace"):
        loop_model.set_metadata_prop("rtlsim_trace", f"{node.name}_fifosim_trace.wdb")
    loop_model = loop_model.transform(
        InsertAndSetFIFODepths(
            fpga_part,
            clk_ns,
        )
    )
    loop_model = loop_model.transform(SplitLargeFIFOs())
    loop_model = loop_model.transform(RemoveShallowFIFOs())
    node_inst.set_nodeattr("body", loop_model.graph)


def prepare_loop_ops_for_ipgen_step2(node, fpga_part, clk_ns):
    node_inst = getCustomOp(node)
    loop_model = node_inst.get_nodeattr("body")
    # go first into subgraph to check if there are other loop ops
    loop_nodes = loop_model.get_nodes_by_op_type("FINNLoop")
    for loop_node in loop_nodes:
        prepare_loop_ops_for_ipgen_step2(loop_node, fpga_part, clk_ns)
    loop_model = loop_model.transform(HLSSynthIP(fpgapart=fpga_part))
    loop_model = loop_model.transform(
        CreateStitchedIP(
            fpga_part,
            clk_ns,
        )
    )
    node_inst.set_nodeattr("body", loop_model.graph)


# dimensions
@pytest.mark.parametrize("dim", [16])
# iteration count, number of models chained together
@pytest.mark.parametrize("iteration", [3])
# elementwise operation
@pytest.mark.parametrize("elemwise_optype", ["ElementwiseMul_hls", "ElementwiseAdd_hls"])
# elementwise shape
@pytest.mark.parametrize("rhs_shape", [[1], [16]])
# eltwise param dtype
@pytest.mark.parametrize("eltw_param_dtype", ["INT8", "FLOAT32"])
# tail node
@pytest.mark.parametrize("tail_node", [False, True])
@pytest.mark.vivado
@pytest.mark.slow
# Note: fpgadataflow marker removed to prevent CI auto-discovery
def test_fpgadataflow_finnloop_manual(
    dim, iteration, elemwise_optype, rhs_shape, eltw_param_dtype, tail_node
):
    """Manual step-by-step test for FINNLoop transformations.

    This test manually applies each transformation step for debugging purposes.
    For automated CI testing, use test_finnloop_end2end_mlo instead, which uses
    the build system and represents the actual end-to-end workflow.
    """
    # Check vivado version
    vivado_path = os.environ.get("XILINX_VIVADO")
    match = re.search(r"\b(20\d{2})\.(1|2)\b", vivado_path)
    year, minor = int(match.group(1)), int(match.group(2))
    if (year, minor) < (2024, 2):
        pytest.skip("""At least Vivado version 2024.2 needed for MLO.""")
    loop_body_models = create_chained_loop_bodies(
        dim, dim, iteration, elemwise_optype, rhs_shape, eltw_param_dtype
    )
    nodes_per_body = len(loop_body_models[0].graph.node)
    model = loop_body_models[0]
    for m in loop_body_models[1:]:
        model = model.transform(MergeONNXModels(m))

    if tail_node:
        tail_outp = create_tensor_info("tail_outp", [1, 3, 3, dim])
        tr_node = create_node(
            "ElementwiseAdd_hls",
            [model.graph.output[0].name, "tail_add"],
            ["tail_outp"],
            "Add_tail",
            {
                "lhs_shape": [1, 3, 3, dim],
                "rhs_shape": [1],
                "out_shape": [1, 3, 3, dim],
                "lhs_dtype": "INT8",
                "rhs_dtype": "INT8",
                "out_dtype": "INT9",
            },
        )
        model.graph.node.insert(len(model.graph.node), tr_node)
        model.graph.value_info.append(model.graph.output[0])
        model.graph.output.pop(0)
        model.graph.output.append(tail_outp)
        AddtailParam = gen_finn_dt_tensor(DataType["INT8"], [1])
        model.set_initializer("tail_add", AddtailParam)
        model.set_tensor_datatype("tail_add", DataType["INT8"])

    # cleanup
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # Generate reference output
    x = gen_finn_dt_tensor(DataType["INT8"], (1, 3, 3, dim))
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())
    model = model.transform(SetExecMode("cppsim"))
    io_dict = {model.graph.input[0].name: x}
    y_dict = oxe.execute_onnx(model, io_dict)
    y_ref = y_dict[model.graph.output[0].name]

    # set loop boundary
    node_metadata = {
        "pkg.torch.onnx.name_scopes": "['', 'layers.0']",
        "pkg.torch.onnx.class_hierarchy": "['TestModule', 'test']",
    }
    node_range = (model.graph.node[0], model.graph.node[nodes_per_body - 1])
    model = model.transform(SetLoopBoundary(node_metadata, node_range))

    # loop extraction and rolling
    loop_extraction = LoopExtraction(hierarchy_list=[["", "layers.0"]])
    model = model.transform(loop_extraction)

    assert (
        len(model.get_nodes_by_op_type("fn_loop-body")) == iteration
    ), "Loop extraction did not find expected number of loop bodies"

    model = model.transform(LoopRolling(loop_extraction.loop_body_template))

    # LoopRolling automatically adapts operator attributes for loop context
    # (e.g., rhs_style changes from "const" to "input" for streamed parameters)
    # This requires recompilation of the elementwise node for cppsim
    loop_node = model.get_nodes_by_op_type("FINNLoop")[0]
    loop_body_graph = get_by_name(loop_node.attribute, "body").g
    elementwise_node = get_by_name(loop_body_graph.node, elemwise_optype, "op_type")
    code_gen_dir_cppsim_attr = get_by_name(elementwise_node.attribute, "code_gen_dir_cppsim")
    code_gen_dir_cppsim_attr.s = b""  # reset cpp gen directory to force recompilation
    executable_path_attr = get_by_name(elementwise_node.attribute, "executable_path")
    executable_path_attr.s = b""  # reset cpp exec directory to force recompilation

    # recompile elementwise node for cppsim
    model = model.transform(PrepareCppSim(), apply_to_subgraphs=True)
    model = model.transform(CompileCppSim(), apply_to_subgraphs=True)

    y_dict = oxe.execute_onnx(model, io_dict)
    y_prod = y_dict[model.graph.output[0].name]
    assert (y_prod == y_ref).all()

    # node-by-node rtlsim
    model = model.transform(GiveUniqueNodeNames(), apply_to_subgraphs=True)
    # TODO: allow for node-by-node rtlsim of a finn loop op
    model = model.transform(SetExecMode("rtlsim"))
    loop_nodes = model.get_nodes_by_op_type("FINNLoop")
    for node in loop_nodes:
        prepare_loop_ops_for_ipgen_step1(node, fpga_part, clk_ns)
    model = model.transform(GiveUniqueNodeNames())
    loop_nodes = model.get_nodes_by_op_type("FINNLoop")
    for loop_node in loop_nodes:
        loop_inst = getCustomOp(loop_node)
        loop_body = loop_inst.get_nodeattr("body")
        loop_body = loop_body.transform(GiveUniqueNodeNames(prefix=loop_node.name + "_"))
        loop_inst.set_nodeattr("body", loop_body.graph)
    model = model.transform(
        PrepareIP(fpga_part, clk_ns), apply_to_subgraphs=True, use_preorder_traversal=False
    )
    loop_nodes = model.get_nodes_by_op_type("FINNLoop")
    for node in loop_nodes:
        prepare_loop_ops_for_ipgen_step2(node, fpga_part, clk_ns)

    model = model.transform(HLSSynthIP(fpgapart=fpga_part))
    model = model.transform(PrepareRTLSim())

    io_dict = {model.graph.input[0].name: x}
    y_dict = oxe.execute_onnx(model, io_dict)
    y_prod = y_dict[model.graph.output[0].name]
    assert (y_prod == y_ref).all()

    # FIFO sizing
    model = model.transform(InsertDWC())
    model = model.transform(SpecializeLayers(fpga_part))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(fpga_part, clk_ns))
    model = model.transform(HLSSynthIP(fpgapart=fpga_part))
    model = model.transform(PrepareRTLSim())
    model = model.transform(DeriveCharacteristic(6000))
    model = model.transform(DeriveFIFOSizes())
    model = model.transform(InsertFIFO(True))
    model = model.transform(SpecializeLayers(fpga_part))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    # stitched IP rtlsim
    model = model.transform(PrepareIP(fpga_part, clk_ns))
    model = model.transform(HLSSynthIP(fpgapart=fpga_part))
    model = model.transform(CreateStitchedIP(fpga_part, clk_ns))

    loop_node = model.get_nodes_by_op_type("FINNLoop")[0]
    mlo_prehook = mlo_prehook_func_factory(loop_node)
    rtlsim_exec(model, io_dict, pre_hook=mlo_prehook)
    y_prod = io_dict[model.graph.output[0].name]
    assert (y_prod == y_ref).all()
