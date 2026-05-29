/******************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 * @brief
 *	Tap into a forwarded stream with a customizable repetition of values on
 *	the tapped output.
 *****************************************************************************/

module stream_tap #(
	int unsigned  DATA_WIDTH,
	int unsigned  TAP_REP = 1,
	int unsigned  TAP_QUEUE_DEPTH = 16
)(
	input	logic  clk,
	input	logic  rst,

	input	logic [DATA_WIDTH-1:0]  idat,
	input	logic  ivld,
	output	logic  irdy,

	output	logic [DATA_WIDTH-1:0]  odat,
	output	logic  ovld,
	input	logic  ordy,

	output	logic [DATA_WIDTH-1:0]  tdat,
	output	logic  tvld,
	input	logic  trdy
);

	localparam int unsigned  CNT_BITS = (TAP_REP <= 1) ? 1 : $clog2(TAP_REP + 1);
	localparam int unsigned  QUEUE_DEPTH = (TAP_QUEUE_DEPTH < 1) ? 1 : TAP_QUEUE_DEPTH;
	localparam int unsigned  PTR_BITS = (QUEUE_DEPTH <= 1) ? 1 : $clog2(QUEUE_DEPTH);
	localparam int unsigned  LEVEL_BITS = $clog2(QUEUE_DEPTH + 1);
	typedef logic [CNT_BITS-1:0]  cnt_t;
	typedef logic [PTR_BITS-1:0]  ptr_t;
	typedef logic [LEVEL_BITS-1:0]  level_t;

	logic [DATA_WIDTH-1:0]  ODat = 'x;
	logic  OVld = 0;

	logic [QUEUE_DEPTH-1:0][DATA_WIDTH-1:0]  QDat = 'x;
	cnt_t  QRemain [QUEUE_DEPTH] = '{ default: 'x };
	ptr_t  QRd = '0;
	ptr_t  QWr = '0;
	level_t  QLevel = '0;

	uwire  oready = !OVld || ordy;
	uwire  TVld = QLevel != 0;
	uwire  tap_done = TVld && trdy && (QRemain[QRd] == cnt_t'(1));
	uwire  qready = (TAP_REP == 0) || (QLevel < level_t'(QUEUE_DEPTH)) || tap_done;
	uwire  push = ivld && irdy && (TAP_REP != 0);

	function automatic ptr_t ptr_next(input ptr_t ptr);
		return (ptr == ptr_t'(QUEUE_DEPTH-1)) ? ptr_t'(0) : ptr_t'(ptr + ptr_t'(1));
	endfunction

	assign	irdy = oready && qready;
	assign	odat = ODat;
	assign	ovld = OVld;
	assign	tdat = QDat[QRd];
	assign	tvld = TVld;

	always_ff @(posedge clk) begin
		if(rst) begin
			ODat <= 'x;
			OVld <= 0;

			QDat <= 'x;
			QRemain <= '{ default: 'x };
			QRd <= '0;
			QWr <= '0;
			QLevel <= '0;
		end
		else begin
			if(ivld && irdy) begin
				ODat <= idat;
				OVld <= 1;
			end
			else if(OVld && ordy) begin
				OVld <= 0;
			end

			if(TVld && trdy && !tap_done) begin
				QRemain[QRd] <= QRemain[QRd] - cnt_t'(1);
			end

			if(push) begin
				QDat[QWr] <= idat;
				QRemain[QWr] <= cnt_t'(TAP_REP);
				QWr <= ptr_next(QWr);
			end
			if(tap_done) begin
				QRd <= ptr_next(QRd);
			end

			case ({push, tap_done})
				2'b10: QLevel <= QLevel + level_t'(1);
				2'b01: QLevel <= QLevel - level_t'(1);
				default: QLevel <= QLevel;
			endcase
			if(tap_done && !push) begin
				QRemain[QRd] <= 'x;
			end
		end
	end

endmodule : stream_tap
