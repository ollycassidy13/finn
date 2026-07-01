/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 * @brief	Skid buffer with optional feed stages to ease long-distance routing.
 * @todo
 *	Offer knob for increasing buffer elasticity at the cost of allowable
 *	number of feed stages.
 ***************************************************************************/

module skid #(
	int unsigned  DATA_WIDTH,
	int unsigned  FEED_STAGES = 0
)(
	input	logic  clk,
	input	logic  rst,

	input	logic [DATA_WIDTH-1:0]  idat,
	input	logic  ivld,
	output	logic  irdy,

	output	logic [DATA_WIDTH-1:0]  odat,
	output	logic  ovld,
	input	logic  ordy
);

	typedef logic [DATA_WIDTH-1:0]  dat_t;

	// Required SRL Capacity
	localparam int unsigned  CAP = 2*FEED_STAGES + 2;
	initial begin // Check Capacity limit of SRL16
		if(CAP > 16) begin
			$error("%m: Requested number of %0d FEED_STAGES exceeds SRL capacity.");
			$finish;
		end
	end

	uwire  aload;
	uwire dat_t  adat;
	(* max_fanout = 128 *) uwire [3:0]  aptr;
	uwire  bvld;
	uwire  bload;
	if(FEED_STAGES == 0) begin : genNoFeedStages

		// Elasticity Control Logic
		(* max_fanout = 128 *) logic [1:0]  AVld = '0;
		logic  ARdy = 1;	// = !AVld[1]
		assign	irdy = ARdy;
		assign	bvld = |AVld;

		always_ff @(posedge clk) begin
			if(rst) begin
				AVld <= '0;
				ARdy <= 1;
			end
			else begin
				automatic logic  ardy = !AVld || bload;
				AVld <= '{ !ardy, AVld[1]? AVld[0] : ivld };
				ARdy <= ardy;
			end
		end
		assign	aload = irdy;
		assign	adat = idat;
		assign	aptr = { 3'b000, AVld[1] };

	end : genNoFeedStages
	else begin : genFeedStages

		if(1) begin : blkInputFeed

			// Credit-based Input Throttling
			(* max_fanout = 64 *) logic signed [$clog2(CAP):0]  ICnt = -CAP;  // -CAP, ..., -1, 0
			assign	irdy = ICnt[$left(ICnt)];

			// Dumb input stages to ease long-distance routing
			(* SHREG_EXTRACT = "no" *) dat_t  IDat[FEED_STAGES] = '{ default: 'x };
			(* SHREG_EXTRACT = "no" *) logic  IVld[FEED_STAGES] = '{ default: 0 };
			(* SHREG_EXTRACT = "no" *) logic  IInc[FEED_STAGES] = '{ default: 0 };

			always_ff @(posedge clk) begin
				if(rst) begin
					ICnt <= -CAP;

					IDat <= '{ default: 'x };
					IVld <= '{ default: 0 };
					IInc <= '{ default: 0 };
				end
				else begin
					automatic logic  iload = ivld && irdy;

					ICnt <= ICnt + $signed(IInc[0] == iload? 0 : IInc[0]? -1 : 1);
					assert((ICnt > -$signed(CAP)) || !IInc[0]) else begin
						$error("%m: Credit increment request beyond buffer capacity.");
						$stop;
					end

					for(int unsigned  i = 0; i < FEED_STAGES-1; i++) begin
						IDat[i] <= IDat[i+1];
						IVld[i] <= IVld[i+1];
						IInc[i] <= IInc[i+1];
					end
					IDat[FEED_STAGES-1] <= idat;
					IVld[FEED_STAGES-1] <= iload;
					IInc[FEED_STAGES-1] <= bload && bvld;
				end
			end
			assign	aload = IVld[0];
			assign	adat = IDat[0];
		end : blkInputFeed

		// Elasticity Control Logic
		(* max_fanout = 128 *) logic signed [$clog2(CAP):0]  APtr = '1;  // -1, 0, 1, ..., CAP-1
		assign	bvld = !APtr[$left(APtr)];

		always_ff @(posedge clk) begin
			if(rst)  APtr <= '1;
			else begin
				APtr <= APtr + $signed((aload == (bload && bvld))? 0 : aload? 1 : -1);
				assert((APtr < $signed(CAP-1)) || !aload) else begin
					$error("%m: Unexpected SRL load request.");
					$stop;
				end
			end
		end
		assign	aptr = $unsigned(APtr[$left(APtr)-1:0]);

	end : genFeedStages

	//-----------------------------------------------------------------------
	// Buffer Memory: SRL:CAP + Reg (no reset)

	// Elastic SRL
	localparam bit  BEHAVIORAL =
// synthesis translate_off
		1 ||
// synthesis translate_on
		0;

	uwire dat_t  bdat;
	if(BEHAVIORAL) begin : genBehav
		// This does not infer an SRL reliably for small sizes.
		(* SHREG_EXTRACT = "yes" *) dat_t  SRL[CAP];
		always_ff @(posedge clk) begin
			if(aload)  SRL <= { adat, SRL[0:CAP-2] };
		end
		assign	bdat = SRL[aptr];
	end : genBehav
	else begin : genSRL
		localparam int unsigned  SRL_SLICE_BITS = 4096;
		localparam int unsigned  SRL_SLICES =
			(DATA_WIDTH + SRL_SLICE_BITS - 1) / SRL_SLICE_BITS;
		for(genvar  c = 0; c < SRL_SLICES; c++) begin : genSlice
			localparam int unsigned  SLICE_BASE = c * SRL_SLICE_BITS;
			localparam int unsigned  SLICE_WIDTH =
				(DATA_WIDTH - SLICE_BASE > SRL_SLICE_BITS)
					? SRL_SLICE_BITS
					: DATA_WIDTH - SLICE_BASE;
			skid_srl_slice #(
				.DATA_WIDTH(SLICE_WIDTH)
			) srl_slice (
				.clk(clk),
				.ce(aload),
				.din(adat[SLICE_BASE +: SLICE_WIDTH]),
				.addr(aptr),
				.dout(bdat[SLICE_BASE +: SLICE_WIDTH])
			);
		end : genSlice
	end : genSRL

	// Output Register
	(* EXTRACT_ENABLE = "yes" *)
	dat_t  B = 'x;
	logic  BVld = 0;
	always_ff @(posedge clk) begin
		if(rst) begin
			BVld <= 0;
			B <= 'x;
		end
		else begin
			BVld <= bvld || !bload;
			if(bload)  B <= bdat;
		end
	end
	assign	bload = !BVld || ordy;
	assign	odat = B;
	assign	ovld = BVld;

endmodule : skid

module skid_srl_slice #(
	int unsigned  DATA_WIDTH
)(
	input	logic  clk,
	input	logic  ce,
	input	logic [DATA_WIDTH-1:0]  din,
	input	logic [3:0]  addr,
	output	uwire [DATA_WIDTH-1:0]  dout
);

	localparam int unsigned  APTR_BUFFER_FANOUT = 128;
	localparam int unsigned  APTR_BUFFER_REPS =
		(DATA_WIDTH + APTR_BUFFER_FANOUT - 1) / APTR_BUFFER_FANOUT;
	uwire [APTR_BUFFER_REPS-1:0][3:0]  addr_buf;

	for(genvar  r = 0; r < APTR_BUFFER_REPS; r++) begin : genAPtrBuf
		for(genvar  b = 0; b < 4; b++) begin : genBit
			(* DONT_TOUCH = "true" *)
			LUT1 #(.INIT(2'h2)) aptr_lut (.I0(addr[b]), .O(addr_buf[r][b]));
		end : genBit
	end : genAPtrBuf

	for(genvar  j = 0; j < DATA_WIDTH; j++) begin : genBit
		localparam int unsigned  APTR_BUFFER_INDEX = j / APTR_BUFFER_FANOUT;
		SRL16E srl (
			.CLK(clk),
			.CE(ce),
			.D(din[j]),
			.A3(addr_buf[APTR_BUFFER_INDEX][3]),
			.A2(addr_buf[APTR_BUFFER_INDEX][2]),
			.A1(addr_buf[APTR_BUFFER_INDEX][1]),
			.A0(addr_buf[APTR_BUFFER_INDEX][0]),
			.Q(dout[j])
		);
	end : genBit

endmodule : skid_srl_slice
