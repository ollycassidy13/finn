/******************************************************************************
 * Copyright (C) 2024, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
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
 * @brief	Verilog AXI-lite wrapper for MVU & VVU.
 *****************************************************************************/

`define $EN_MLO$

module $MODULE_NAME_AXI_WRAPPER$ #(
	parameter	MW = $MW$,
	parameter	MH = $MH$,
	parameter	PE = $PE$,
	parameter	SIMD = $SIMD$,
    parameter   TH = $TH$,
    parameter   N_REPS = $N_REPS$,
	parameter	WEIGHT_WIDTH = $WEIGHT_WIDTH$,
    parameter   N_LAYERS = $N_LAYERS$,
    parameter   RAM_STYLE = $RAM_STYLE$,

    parameter   ADDR_BITS = 64,
    parameter   DATA_BITS = 256,
    parameter   LEN_BITS = 32,
    parameter   IDX_BITS = 16,

	// Safely deducible parameters
    parameter   IWSIMD = $IWSIMD$,
    parameter   WSIMD = $WSIMD$,
    parameter   DS_BITS_BA = (IWSIMD*WEIGHT_WIDTH+7)/8 * 8,
	parameter	WS_BITS_BA = (WSIMD*WEIGHT_WIDTH+7)/8 * 8
)(
	// Global Control
	(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF axi_mm:in_idx0_V:out0_V, ASSOCIATED_RESET ap_rst_n" *)
	(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
	input	ap_clk,
	(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
	input	ap_rst_n,

    // Completion
    output wire                                out_done,

    // AXI
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm ARADDR" *)
    output wire[ADDR_BITS-1:0]                 axi_mm_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm ARBURST" *)
    output wire[1:0]		                    axi_mm_arburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm ARCACHE" *)
    output wire[3:0]		                    axi_mm_arcache,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm ARID" *)
    output wire[1:0]      		                axi_mm_arid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm ARLEN" *)
    output wire[7:0]		                    axi_mm_arlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm ARLOCK" *)
    output wire[0:0]		                    axi_mm_arlock,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm ARPROT" *)
    output wire[2:0]		                    axi_mm_arprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm ARSIZE" *)
    output wire[2:0]		                    axi_mm_arsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm ARREADY" *)
    input  wire			                    axi_mm_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm ARVALID" *)
    output wire			                    axi_mm_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm AWADDR" *)
    output wire[ADDR_BITS-1:0] 	            axi_mm_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm AWBURST" *)
    output wire[1:0]		                    axi_mm_awburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm AWCACHE" *)
    output wire[3:0]		                    axi_mm_awcache,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm AWID" *)
    output wire[1:0]		                    axi_mm_awid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm AWLEN" *)
    output wire[7:0]		                    axi_mm_awlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm AWLOCK" *)
    output wire[0:0]		                    axi_mm_awlock,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm AWPROT" *)
    output wire[2:0]		                    axi_mm_awprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm AWSIZE" *)
    output wire[2:0]		                    axi_mm_awsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm AWREADY" *)
    input  wire			                    axi_mm_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm AWVALID" *)
    output wire			                    axi_mm_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm RDATA" *)
    input  wire[DATA_BITS-1:0] 	            axi_mm_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm RID" *)
    input  wire[1:0]      		                axi_mm_rid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm RLAST" *)
    input  wire			                    axi_mm_rlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm RRESP" *)
    input  wire[1:0]		                    axi_mm_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm RREADY" *)
    output wire 			                    axi_mm_rready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm RVALID" *)
    input  wire			                    axi_mm_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm WDATA" *)
    output wire[DATA_BITS-1:0] 	            axi_mm_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm WLAST" *)
    output wire			                    axi_mm_wlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm WSTRB" *)
    output wire[DATA_BITS/8-1:0] 	            axi_mm_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm WREADY" *)
    input  wire			                    axi_mm_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm WVALID" *)
    output wire			                    axi_mm_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm BID" *)
    input  wire[1:0]      		                axi_mm_bid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm BRESP" *)
    input  wire[1:0]		                    axi_mm_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm BREADY" *)
    output wire			                    axi_mm_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 axi_mm BVALID" *)
    input  wire			                    axi_mm_bvalid,

`ifdef EN_MLO
    // Index
    input  wire                                in_idx0_V_tvalid,
    output wire                                in_idx0_V_tready,
    input  wire[IDX_BITS-1:0]                  in_idx0_V_tdata,
`endif

    // Stream
    output wire                                out0_V_tvalid,
    input  wire                                out0_V_tready,
    output wire[WS_BITS_BA-1:0]                out0_V_tdata
);

`ifndef EN_MLO
    wire in_idx0_V_tvalid;
    wire in_idx0_V_tready;
    wire [IDX_BITS-1:0] in_idx0_V_tdata;

    assign in_idx0_V_tvalid = 1'b1;
    assign in_idx0_V_tdata = 0;
`endif

// DMA <-> DWC internal wires
wire axis_dma_tvalid;
wire axis_dma_tready;
wire [DATA_BITS-1:0] axis_dma_tdata;
wire [DATA_BITS/8-1:0] axis_dma_tkeep;
wire axis_dma_tlast;

wire axis_dwc_tvalid;
wire axis_dwc_tready;
wire [DS_BITS_BA-1:0] axis_dwc_tdata;
wire [(DS_BITS_BA)/8-1:0] axis_dwc_tkeep;
wire axis_dwc_tlast;

// Width converter
$DWC_INST$

fetch_weights #(
    .PE(PE), .SIMD(SIMD), .TH(TH),
    .MH(MH), .MW(MW), .N_REPS(N_REPS),
    .WEIGHT_WIDTH(WEIGHT_WIDTH),
    .IWSIMD(IWSIMD), .OWSIMD(WSIMD),
    .ADDR_BITS(ADDR_BITS), .DATA_BITS(DATA_BITS), .LEN_BITS(LEN_BITS), .IDX_BITS(IDX_BITS),
    .N_LAYERS(N_LAYERS),
    .RAM_STYLE(RAM_STYLE),
`ifdef EN_MLO
    .EN_MLO(1)
`else
    .EN_MLO(0)
`endif
) inst (
    .aclk               (ap_clk),
    .aresetn            (ap_rst_n),

    .m_axi_ddr_araddr   (axi_mm_araddr),
    .m_axi_ddr_arburst  (axi_mm_arburst),
    .m_axi_ddr_arcache  (axi_mm_arcache),
    .m_axi_ddr_arid     (axi_mm_arid),
    .m_axi_ddr_arlen    (axi_mm_arlen),
    .m_axi_ddr_arlock   (axi_mm_arlock),
    .m_axi_ddr_arprot   (axi_mm_arprot),
    .m_axi_ddr_arsize   (axi_mm_arsize),
    .m_axi_ddr_arready  (axi_mm_arready),
    .m_axi_ddr_arvalid  (axi_mm_arvalid),
    .m_axi_ddr_awaddr   (axi_mm_awaddr),
    .m_axi_ddr_awburst  (axi_mm_awburst),
    .m_axi_ddr_awcache  (axi_mm_awcache),
    .m_axi_ddr_awid     (axi_mm_awid),
    .m_axi_ddr_awlen    (axi_mm_awlen),
    .m_axi_ddr_awlock   (axi_mm_awlock),
    .m_axi_ddr_awprot   (axi_mm_awprot),
    .m_axi_ddr_awsize   (axi_mm_awsize),
    .m_axi_ddr_awready  (axi_mm_awready),
    .m_axi_ddr_awvalid  (axi_mm_awvalid),
    .m_axi_ddr_rdata    (axi_mm_rdata),
    .m_axi_ddr_rid      (axi_mm_rid),
    .m_axi_ddr_rlast    (axi_mm_rlast),
    .m_axi_ddr_rresp    (axi_mm_rresp),
    .m_axi_ddr_rready   (axi_mm_rready),
    .m_axi_ddr_rvalid   (axi_mm_rvalid),
    .m_axi_ddr_wdata    (axi_mm_wdata),
    .m_axi_ddr_wlast    (axi_mm_wlast),
    .m_axi_ddr_wstrb    (axi_mm_wstrb),
    .m_axi_ddr_wready   (axi_mm_wready),
    .m_axi_ddr_wvalid   (axi_mm_wvalid),
    .m_axi_ddr_bid      (axi_mm_bid),
    .m_axi_ddr_bresp    (axi_mm_bresp),
    .m_axi_ddr_bready   (axi_mm_bready),
    .m_axi_ddr_bvalid   (axi_mm_bvalid),

    .s_idx_tvalid       (in_idx0_V_tvalid),
    .s_idx_tready       (in_idx0_V_tready),
    .s_idx_tdata        (in_idx0_V_tdata),

    .axis_dma_tvalid    (axis_dma_tvalid),
    .axis_dma_tready    (axis_dma_tready),
    .axis_dma_tdata     (axis_dma_tdata),
    .axis_dma_tkeep     (axis_dma_tkeep),
    .axis_dma_tlast     (axis_dma_tlast),

    .axis_dwc_tvalid    (axis_dwc_tvalid),
    .axis_dwc_tready    (axis_dwc_tready),
    .axis_dwc_tdata     (axis_dwc_tdata),
    .axis_dwc_tkeep     (axis_dwc_tkeep),
    .axis_dwc_tlast     (axis_dwc_tlast),

    .m_axis_tvalid      (out0_V_tvalid),
    .m_axis_tready      (out0_V_tready),
    .m_axis_tdata       (out0_V_tdata)
);

endmodule // $MODULE_NAME_AXI_WRAPPER$
