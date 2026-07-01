/******************************************************************************
 * Copyright (C) 2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and this disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *****************************************************************************/

module overlapped_loop_control #(
    int unsigned FM_SIZE,
    int unsigned N_LAYERS,
    int unsigned ILEN_BITS,
    int unsigned OLEN_BITS,
    int unsigned IDX_BITS,
    int unsigned ADDR_BITS,
    int unsigned DATA_BITS,
    int unsigned LEN_BITS
) (
    input  logic                aclk,
    input  logic                aresetn,

    output [ADDR_BITS-1:0]      m_axi_hbm_araddr,
    output [1:0]                m_axi_hbm_arburst,
    output [3:0]                m_axi_hbm_arcache,
    output [1:0]                m_axi_hbm_arid,
    output [7:0]                m_axi_hbm_arlen,
    output                      m_axi_hbm_arlock,
    output [2:0]                m_axi_hbm_arprot,
    output [2:0]                m_axi_hbm_arsize,
    input                       m_axi_hbm_arready,
    output                      m_axi_hbm_arvalid,
    output [ADDR_BITS-1:0]      m_axi_hbm_awaddr,
    output [1:0]                m_axi_hbm_awburst,
    output [3:0]                m_axi_hbm_awcache,
    output [1:0]                m_axi_hbm_awid,
    output [7:0]                m_axi_hbm_awlen,
    output                      m_axi_hbm_awlock,
    output [2:0]                m_axi_hbm_awprot,
    output [2:0]                m_axi_hbm_awsize,
    input                       m_axi_hbm_awready,
    output                      m_axi_hbm_awvalid,
    input  [DATA_BITS-1:0]      m_axi_hbm_rdata,
    input  [1:0]                m_axi_hbm_rid,
    input                       m_axi_hbm_rlast,
    input  [1:0]                m_axi_hbm_rresp,
    output                      m_axi_hbm_rready,
    input                       m_axi_hbm_rvalid,
    output [DATA_BITS-1:0]      m_axi_hbm_wdata,
    output                      m_axi_hbm_wlast,
    output [DATA_BITS/8-1:0]    m_axi_hbm_wstrb,
    input                       m_axi_hbm_wready,
    output                      m_axi_hbm_wvalid,
    input  [1:0]                m_axi_hbm_bid,
    input  [1:0]                m_axi_hbm_bresp,
    output                      m_axi_hbm_bready,
    input                       m_axi_hbm_bvalid,

    output [ILEN_BITS-1:0]      m_axis_core_tdata,
    output                      m_axis_core_tvalid,
    input                       m_axis_core_tready,

    input  [OLEN_BITS-1:0]      s_axis_core_tdata,
    input                       s_axis_core_tvalid,
    output                      s_axis_core_tready,

    output [IDX_BITS-1:0]       m_idx_tdata,
    output                      m_idx_tvalid,
    input                       m_idx_tready,

    input  [IDX_BITS-1:0]       s_idx_tdata,
    input                       s_idx_tvalid,
    output                      s_idx_tready,

    input  [ILEN_BITS-1:0]      s_axis_fs_tdata,
    input                       s_axis_fs_tvalid,
    output                      s_axis_fs_tready,

    output [OLEN_BITS-1:0]      m_axis_se_tdata,
    output                      m_axis_se_tvalid,
    input                       m_axis_se_tready
);

logic idx_if_in_tvalid, idx_if_in_tready;
logic [IDX_BITS-1:0] idx_if_in_tdata;
logic idx_if_out_tvalid, idx_if_out_tready;
logic [IDX_BITS-1:0] idx_if_out_tdata;

logic axis_if_in_tvalid, axis_if_in_tready;
logic [OLEN_BITS-1:0] axis_if_in_tdata;
logic axis_if_out_tvalid, axis_if_out_tready;
logic [ILEN_BITS-1:0] axis_if_out_tdata;

assign m_axi_hbm_araddr = '0;
assign m_axi_hbm_arburst = '0;
assign m_axi_hbm_arcache = '0;
assign m_axi_hbm_arid = '0;
assign m_axi_hbm_arlen = '0;
assign m_axi_hbm_arlock = 1'b0;
assign m_axi_hbm_arprot = '0;
assign m_axi_hbm_arsize = '0;
assign m_axi_hbm_arvalid = 1'b0;
assign m_axi_hbm_awaddr = '0;
assign m_axi_hbm_awburst = '0;
assign m_axi_hbm_awcache = '0;
assign m_axi_hbm_awid = '0;
assign m_axi_hbm_awlen = '0;
assign m_axi_hbm_awlock = 1'b0;
assign m_axi_hbm_awprot = '0;
assign m_axi_hbm_awsize = '0;
assign m_axi_hbm_awvalid = 1'b0;
assign m_axi_hbm_rready = 1'b1;
assign m_axi_hbm_wdata = '0;
assign m_axi_hbm_wlast = 1'b0;
assign m_axi_hbm_wstrb = '0;
assign m_axi_hbm_wvalid = 1'b0;
assign m_axi_hbm_bready = 1'b1;

mux #(
    .IDX_BITS(IDX_BITS),
    .FM_SIZE(FM_SIZE),
    .ILEN_BITS(ILEN_BITS)
) inst_mux_in (
    .aclk(aclk),
    .aresetn(aresetn),
    .s_idx_tvalid(idx_if_out_tvalid),
    .s_idx_tready(idx_if_out_tready),
    .s_idx_tdata(idx_if_out_tdata),
    .m_idx_tvalid(m_idx_tvalid),
    .m_idx_tready(m_idx_tready),
    .m_idx_tdata(m_idx_tdata),
    .s_axis_fs_tvalid(s_axis_fs_tvalid),
    .s_axis_fs_tready(s_axis_fs_tready),
    .s_axis_fs_tdata(s_axis_fs_tdata),
    .s_axis_if_tvalid(axis_if_out_tvalid),
    .s_axis_if_tready(axis_if_out_tready),
    .s_axis_if_tdata(axis_if_out_tdata),
    .m_axis_tvalid(m_axis_core_tvalid),
    .m_axis_tready(m_axis_core_tready),
    .m_axis_tdata(m_axis_core_tdata)
);

demux #(
    .N_LAYERS(N_LAYERS),
    .IDX_BITS(IDX_BITS),
    .FM_SIZE(FM_SIZE),
    .OLEN_BITS(OLEN_BITS)
) inst_mux_out (
    .aclk(aclk),
    .aresetn(aresetn),
    .s_idx_tvalid(s_idx_tvalid),
    .s_idx_tready(s_idx_tready),
    .s_idx_tdata(s_idx_tdata),
    .m_idx_tvalid(idx_if_in_tvalid),
    .m_idx_tready(idx_if_in_tready),
    .m_idx_tdata(idx_if_in_tdata),
    .s_axis_tvalid(s_axis_core_tvalid),
    .s_axis_tready(s_axis_core_tready),
    .s_axis_tdata(s_axis_core_tdata),
    .m_axis_if_tvalid(axis_if_in_tvalid),
    .m_axis_if_tready(axis_if_in_tready),
    .m_axis_if_tdata(axis_if_in_tdata),
    .m_axis_se_tvalid(m_axis_se_tvalid),
    .m_axis_se_tready(m_axis_se_tready),
    .m_axis_se_tdata(m_axis_se_tdata)
);

streamed_feedback_frames #(
    .FM_SIZE(FM_SIZE),
    .ILEN_BITS(ILEN_BITS),
    .OLEN_BITS(OLEN_BITS),
    .IDX_BITS(IDX_BITS)
) inst_streamed_feedback_frames (
    .aclk(aclk),
    .aresetn(aresetn),
    .s_idx_tvalid(idx_if_in_tvalid),
    .s_idx_tready(idx_if_in_tready),
    .s_idx_tdata(idx_if_in_tdata),
    .m_idx_tvalid(idx_if_out_tvalid),
    .m_idx_tready(idx_if_out_tready),
    .m_idx_tdata(idx_if_out_tdata),
    .s_axis_tvalid(axis_if_in_tvalid),
    .s_axis_tready(axis_if_in_tready),
    .s_axis_tdata(axis_if_in_tdata),
    .m_axis_tvalid(axis_if_out_tvalid),
    .m_axis_tready(axis_if_out_tready),
    .m_axis_tdata(axis_if_out_tdata)
);

endmodule
