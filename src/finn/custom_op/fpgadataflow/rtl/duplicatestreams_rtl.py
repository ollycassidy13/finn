# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.

import os

from finn.custom_op.fpgadataflow.duplicatestreams import DuplicateStreams
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend


class DuplicateStreams_rtl(DuplicateStreams, RTLBackend):
    """RTL AXI-stream broadcaster for DuplicateStreams."""

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(DuplicateStreams.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            DuplicateStreams.execute_node(self, context, graph)
        elif mode == "rtlsim":
            RTLBackend.execute_node(self, context, graph)
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )

    def generate_hdl(self, model, fpgapart, clk):
        top_module = self.get_verilog_top_module_name()
        data_width = self.get_instream_width()
        axi_width = ((data_width + 7) // 8) * 8
        num_outputs = self.get_nodeattr("NumOutputStreams")
        output_busif = ":".join("out%d_V" % idx for idx in range(num_outputs))
        associated_busif = "in0_V" if output_busif == "" else "in0_V:" + output_busif

        ports = [
            "\t(* X_INTERFACE_INFO = \"xilinx.com:signal:clock:1.0 ap_clk CLK\" *)",
            (
                "\t(* X_INTERFACE_PARAMETER = "
                f"\"ASSOCIATED_BUSIF {associated_busif}, ASSOCIATED_RESET ap_rst_n\" *)"
            ),
            "\tinput ap_clk,",
            "\t(* X_INTERFACE_PARAMETER = \"POLARITY ACTIVE_LOW\" *)",
            "\tinput ap_rst_n,",
            "\toutput in0_V_TREADY,",
            "\tinput in0_V_TVALID,",
            "\tinput [AXI_BITS-1:0] in0_V_TDATA,",
        ]
        for idx in range(num_outputs):
            comma = "," if idx != num_outputs - 1 else ""
            ports += [
                f"\tinput out{idx}_V_TREADY,",
                f"\toutput out{idx}_V_TVALID,",
                f"\toutput [AXI_BITS-1:0] out{idx}_V_TDATA{comma}",
            ]

        ready_terms = " & ".join("out%d_V_TREADY" % idx for idx in range(num_outputs))
        assigns = [f"\tassign in0_V_TREADY = {ready_terms};"]
        for idx in range(num_outputs):
            assigns += [
                f"\tassign out{idx}_V_TVALID = in0_V_TVALID;",
                f"\tassign out{idx}_V_TDATA = in0_V_TDATA;",
            ]

        verilog = "\n".join(
            [
                f"module {top_module} #(",
                f"\tparameter integer DATA_BITS = {data_width},",
                f"\tparameter integer AXI_BITS = {axi_width}",
                ")(",
                "\n".join(ports),
                ");",
                "",
                "\t// Consume one input word only when every duplicated output can accept it.",
                "\n".join(assigns),
                "",
                "endmodule",
                "",
            ]
        )

        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        self.set_nodeattr("gen_top_module", top_module)
        with open(os.path.join(code_gen_dir, top_module + ".v"), "w") as f:
            f.write(verilog)

        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def get_rtl_file_list(self, abspath=False):
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen") + "/" if abspath else ""
        return [code_gen_dir + self.get_nodeattr("gen_top_module") + ".v"]

    def code_generation_ipi(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        sourcefile = os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + ".v")
        return [
            "add_files -norecurse %s" % sourcefile,
            "create_bd_cell -type module -reference %s %s"
            % (self.get_nodeattr("gen_top_module"), self.onnx_node.name),
        ]
