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
 *****************************************************************************/

module streamed_feedback_frames #(
    int unsigned                    ILEN_BITS,
    int unsigned                    OLEN_BITS,
    int unsigned                    IDX_BITS,
    int unsigned                    FM_SIZE,
    int unsigned                    N_DCPL_STGS = 1
) (
    input  logic                    aclk,
    input  logic                    aresetn,

    input  logic [IDX_BITS-1:0]     s_idx_tdata,
    input  logic                    s_idx_tvalid,
    output logic                    s_idx_tready,

    output logic [IDX_BITS-1:0]     m_idx_tdata,
    output logic                    m_idx_tvalid,
    input  logic                    m_idx_tready,

    input  logic [OLEN_BITS-1:0]    s_axis_tdata,
    input  logic                    s_axis_tvalid,
    output logic                    s_axis_tready,

    output logic [ILEN_BITS-1:0]    m_axis_tdata,
    output logic                    m_axis_tvalid,
    input  logic                    m_axis_tready
);

localparam int unsigned IN_BYTES = OLEN_BITS / 8;
localparam int unsigned OUT_BYTES = ILEN_BITS / 8;
localparam int unsigned FM_BEATS_IN = FM_SIZE / IN_BYTES;
localparam int unsigned FM_BEATS_OUT = FM_SIZE / OUT_BYTES;
localparam int unsigned FM_BEATS_IN_BITS = (FM_BEATS_IN <= 1) ? 1 : $clog2(FM_BEATS_IN);
localparam int unsigned FM_BEATS_OUT_BITS = (FM_BEATS_OUT <= 1) ? 1 : $clog2(FM_BEATS_OUT);

typedef enum logic[0:0] {ST_IDLE, ST_STREAM} state_t;
state_t state_C = ST_IDLE, state_N;

logic [FM_BEATS_IN_BITS-1:0] cnt_in_C = '0, cnt_in_N;
logic [FM_BEATS_OUT_BITS-1:0] cnt_out_C = '0, cnt_out_N;
logic in_done_C = 1'b0, in_done_N;

logic dwc_s_tvalid, dwc_s_tready, dwc_s_tlast;
logic [OLEN_BITS-1:0] dwc_s_tdata;
logic dwc_m_tvalid, dwc_m_tready;
logic [ILEN_BITS-1:0] dwc_m_tdata;
logic dwc_m_tlast;

logic reg_m_tvalid, reg_m_tready;
logic [ILEN_BITS-1:0] reg_m_tdata;

always_ff @(posedge aclk) begin
    if (~aresetn) begin
        state_C <= ST_IDLE;
        cnt_in_C <= '0;
        cnt_out_C <= '0;
        in_done_C <= 1'b0;
    end else begin
        state_C <= state_N;
        cnt_in_C <= cnt_in_N;
        cnt_out_C <= cnt_out_N;
        in_done_C <= in_done_N;
    end
end

always_comb begin
    state_N = state_C;
    cnt_in_N = cnt_in_C;
    cnt_out_N = cnt_out_C;
    in_done_N = in_done_C;

    s_idx_tready = 1'b0;
    m_idx_tvalid = 1'b0;
    m_idx_tdata = s_idx_tdata + 1'b1;

    dwc_s_tvalid = 1'b0;
    dwc_s_tdata = s_axis_tdata;
    dwc_s_tlast = (cnt_in_C == FM_BEATS_IN-1);
    s_axis_tready = 1'b0;
    dwc_m_tready = reg_m_tready;

    case (state_C)
        ST_IDLE: begin
            cnt_in_N = '0;
            cnt_out_N = '0;
            in_done_N = 1'b0;
            m_idx_tvalid = s_idx_tvalid;
            s_idx_tready = m_idx_tready;
            if (s_idx_tvalid && m_idx_tready) begin
                state_N = ST_STREAM;
            end
        end

        ST_STREAM: begin
            if (!in_done_C) begin
                dwc_s_tvalid = s_axis_tvalid;
                s_axis_tready = dwc_s_tready;
                if (s_axis_tvalid && dwc_s_tready) begin
                    if (cnt_in_C == FM_BEATS_IN-1) begin
                        cnt_in_N = '0;
                        in_done_N = 1'b1;
                    end else begin
                        cnt_in_N = cnt_in_C + 1'b1;
                    end
                end
            end

            if (dwc_m_tvalid && dwc_m_tready) begin
                if (cnt_out_C == FM_BEATS_OUT-1) begin
                    cnt_out_N = '0;
                    state_N = ST_IDLE;
                end else begin
                    cnt_out_N = cnt_out_C + 1'b1;
                end
            end
        end
    endcase
end

if_dwc_feedback inst_dwc_feedback (
    .aclk(aclk),
    .aresetn(aresetn),
    .s_axis_tvalid(dwc_s_tvalid),
    .s_axis_tready(dwc_s_tready),
    .s_axis_tdata(dwc_s_tdata),
    .s_axis_tkeep({IN_BYTES{1'b1}}),
    .s_axis_tlast(dwc_s_tlast),
    .m_axis_tvalid(dwc_m_tvalid),
    .m_axis_tready(dwc_m_tready),
    .m_axis_tdata(dwc_m_tdata),
    .m_axis_tkeep(),
    .m_axis_tlast(dwc_m_tlast)
);

skid #(.FEED_STAGES(N_DCPL_STGS), .DATA_WIDTH(ILEN_BITS)) inst_reg_rd (
    .clk(aclk),
    .rst(~aresetn),
    .ivld(dwc_m_tvalid),
    .irdy(reg_m_tready),
    .idat(dwc_m_tdata),
    .ovld(m_axis_tvalid),
    .ordy(m_axis_tready),
    .odat(m_axis_tdata)
);

endmodule
