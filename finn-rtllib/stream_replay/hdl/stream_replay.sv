/******************************************************************************
 * Copyright (C) 2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief
 *   Buffer one AXI-Stream frame and replay it a fixed number of times.
 *****************************************************************************/

module stream_replay #(
	int unsigned  DATA_WIDTH = 32,
	int unsigned  FRAME_WORDS = 1,
	int unsigned  REPS = 1
)(
	input	logic  clk,
	input	logic  rst,

	input	logic [DATA_WIDTH-1:0]  s_axis_0_TDATA,
	input	logic  s_axis_0_TVALID,
	output	logic  s_axis_0_TREADY,

	output	logic [DATA_WIDTH-1:0]  m_axis_0_TDATA,
	output	logic  m_axis_0_TVALID,
	input	logic  m_axis_0_TREADY
);

	localparam int unsigned  ADDR_BITS = (FRAME_WORDS <= 1)? 1 : $clog2(FRAME_WORDS);
	localparam int unsigned  REP_BITS = (REPS <= 1)? 1 : $clog2(REPS);

	typedef enum logic { LOAD, REPLAY } state_t;

	(* ram_style = "auto" *) logic [DATA_WIDTH-1:0]  mem [0:FRAME_WORDS-1];
	state_t  state = LOAD;
	logic [ADDR_BITS-1:0]  wr_addr = '0;
	logic [ADDR_BITS-1:0]  rd_addr = '0;
	logic [REP_BITS-1:0]  rep = '0;

	assign s_axis_0_TREADY = state == LOAD;
	assign m_axis_0_TVALID = state == REPLAY;
	assign m_axis_0_TDATA = mem[rd_addr];

	always_ff @(posedge clk) begin
		if(rst) begin
			state <= LOAD;
			wr_addr <= '0;
			rd_addr <= '0;
			rep <= '0;
		end
		else begin
			if(state == LOAD) begin
				if(s_axis_0_TVALID && s_axis_0_TREADY) begin
					mem[wr_addr] <= s_axis_0_TDATA;
					if(wr_addr == FRAME_WORDS-1) begin
						wr_addr <= '0;
						rd_addr <= '0;
						rep <= '0;
						state <= REPLAY;
					end
					else begin
						wr_addr <= wr_addr + 1'b1;
					end
				end
			end
			else begin
				if(m_axis_0_TVALID && m_axis_0_TREADY) begin
					if(rd_addr == FRAME_WORDS-1) begin
						rd_addr <= '0;
						if(rep == REPS-1) begin
							rep <= '0;
							state <= LOAD;
						end
						else begin
							rep <= rep + 1'b1;
						end
					end
					else begin
						rd_addr <= rd_addr + 1'b1;
					end
				end
			end
		end
	end

endmodule : stream_replay
