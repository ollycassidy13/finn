/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 ***************************************************************************/

module layernorm_queue #(
	int unsigned  DATA_WIDTH,
	int unsigned  ELASTICITY
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
	initial begin
		if(ELASTICITY < 2) begin
			$error("%m: ELASTICITY of %0d must be made 2 or above.", ELASTICITY);
			$finish;
		end
	end

	localparam int unsigned  PTR_WIDTH = $clog2(ELASTICITY);
	localparam int unsigned  COUNT_WIDTH = $clog2(ELASTICITY+1);
	typedef logic [PTR_WIDTH-1:0]  ptr_t;
	typedef logic [COUNT_WIDTH-1:0]  count_t;

	function automatic ptr_t inc_ptr(input ptr_t ptr);
		return  (ptr == ptr_t'(ELASTICITY-1))? '0 : ptr + ptr_t'(1);
	endfunction : inc_ptr

	count_t  Cnt = '0;
	ptr_t  WrPtr = '0;
	ptr_t  RdPtr = '0;
	logic  Rdy = 1;
	(* ram_style = "block" *) dat_t  A[ELASTICITY];
	assign	irdy = Rdy;

	logic  Vld = 0;
	dat_t  B = 'x;
	assign	odat = B;
	assign	ovld = Vld;

	uwire  bload = !Vld || ordy;
	uwire  push = Rdy && ivld;
	uwire  pop = (Cnt != 0) && bload;
	uwire count_t  CntN = Cnt + count_t'(push) - count_t'(pop);

	always_ff @(posedge clk) begin
		if(push)  A[WrPtr] <= idat;
	end

	always_ff @(posedge clk) begin
		if(rst) begin
			Cnt <= '0;
			WrPtr <= '0;
			RdPtr <= '0;
			Rdy <= 1;
			Vld <= 0;
			B <= 'x;
		end
		else begin
			// Make sure Rdy encodes what it's supposed to: space available in queue
			assert(Rdy == (Cnt < ELASTICITY)) else begin
				$error("%m: Broken Rdy computation.");
				$stop;
			end

			Cnt <= CntN;
			if(push)  WrPtr <= inc_ptr(WrPtr);
			if(pop)   RdPtr <= inc_ptr(RdPtr);
			Rdy <= CntN < ELASTICITY;
			if(bload) begin
				Vld <= Cnt != 0;
				B <= A[RdPtr];
			end
		end
	end

endmodule : layernorm_queue
