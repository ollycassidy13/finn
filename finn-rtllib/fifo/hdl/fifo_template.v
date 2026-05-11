/******************************************************************************
 * Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
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
 *****************************************************************************/

module $TOP_MODULE_NAME$(
//- Global Control ------------------
(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in0_V:out0_V, ASSOCIATED_RESET ap_rst_n" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
input   ap_clk,
(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
input   ap_rst_n,

output $COUNT_RANGE$ count,
output $COUNT_RANGE$ maxcount,

//- AXI Stream - Input --------------
output   in0_V_TREADY,
input   in0_V_TVALID,
input  $IN_RANGE$ in0_V_TDATA,

//- AXI Stream - Output --------------
input   out0_V_TREADY,
output   out0_V_TVALID,
output  $OUT_RANGE$ out0_V_TDATA
);

localparam integer FIFO_DATA_WIDTH = $WIDTH$;
localparam integer FIFO_DEPTH = $DEPTH$;
localparam integer FIFO_SPLIT_WIDTH = 512;
localparam integer FIFO_SPLIT_COUNT =
	(FIFO_DATA_WIDTH + FIFO_SPLIT_WIDTH - 1) / FIFO_SPLIT_WIDTH;

`ifdef FINN_SIMULATION
	fifo_gauge #(.WIDTH($WIDTH$), .COUNT_WIDTH($COUNT_WIDTH$)) fifo (
		.clk(ap_clk), .rst(!ap_rst_n),
		.idat(in0_V_TDATA), .ivld(in0_V_TVALID), .irdy(in0_V_TREADY),
		.odat(out0_V_TDATA), .ovld(out0_V_TVALID), .ordy(out0_V_TREADY),
		.count(count), .maxcount(maxcount)
	);
`else
	generate
		if((FIFO_DATA_WIDTH > FIFO_SPLIT_WIDTH) && (FIFO_DEPTH <= 256)) begin : g_wide_fifo
			genvar chunk;
			for(chunk = 0; chunk < FIFO_SPLIT_COUNT; chunk = chunk + 1) begin : g_chunk
				localparam integer LO = chunk * FIFO_SPLIT_WIDTH;
				localparam integer REM = FIFO_DATA_WIDTH - LO;
				localparam integer CW = (REM > FIFO_SPLIT_WIDTH) ? FIFO_SPLIT_WIDTH : REM;
				if(chunk == 0) begin : g_first
					Q_srl #(.depth(FIFO_DEPTH), .width(CW)) fifo (
						.clock(ap_clk), .reset(!ap_rst_n),
						.i_d(in0_V_TDATA[LO +: CW]), .i_v(in0_V_TVALID), .i_r(in0_V_TREADY),
						.o_d(out0_V_TDATA[LO +: CW]), .o_v(out0_V_TVALID), .o_r(out0_V_TREADY),
						.count(count), .maxcount(maxcount)
					);
				end
				else begin : g_rest
					Q_srl #(.depth(FIFO_DEPTH), .width(CW)) fifo (
						.clock(ap_clk), .reset(!ap_rst_n),
						.i_d(in0_V_TDATA[LO +: CW]), .i_v(in0_V_TVALID), .i_r(),
						.o_d(out0_V_TDATA[LO +: CW]), .o_v(), .o_r(out0_V_TREADY),
						.count(), .maxcount()
					);
				end
			end
		end
		else begin : g_narrow_fifo
			Q_srl #(.depth(FIFO_DEPTH), .width(FIFO_DATA_WIDTH)) fifo (
				.clock(ap_clk), .reset(!ap_rst_n),
				.i_d(in0_V_TDATA), .i_v(in0_V_TVALID), .i_r(in0_V_TREADY),
				.o_d(out0_V_TDATA), .o_v(out0_V_TVALID), .o_r(out0_V_TREADY),
				.count(count), .maxcount(maxcount)
			);
		end
	endgenerate
`endif

endmodule
