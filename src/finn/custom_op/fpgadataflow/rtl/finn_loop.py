# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of Xilinx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import copy
import math
import numpy as np
import os
import re
import shutil
import subprocess
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import is_custom_op
from qonnx.util.basic import get_by_name, qonnx_make_model, roundup_to_integer_multiple

import finn.core.onnx_exec as oxe
from finn import xsi
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.custom_op.fpgadataflow import templates
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.util.basic import getHWCustomOp, make_build_dir
from finn.util.create import adjacency_list
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.util.fpgadataflow import is_hls_node, is_rtl_node
from finn.util.mlo_sim import mlo_prehook_func_factory

finnxsi = xsi if xsi.is_available() else None

OVERLAPPED_SCHEDULER_ARTIFACT_SUFFIXES = {".onnx", ".v", ".sv", ".vhd", ".vhdl"}
OVERLAPPED_SCHEDULER_WRAPPER_SUFFIXES = {".v", ".sv"}
OVERLAPPED_SCHEDULER_BUILTINS = {"stream_feedback"}


def collect_ip_dirs(model, ipstitch_path):
    # collect list of all IP dirs
    ip_dirs = []
    need_memstreamer = False
    for node in model.graph.node:
        node_inst = getHWCustomOp(node, model)
        ip_dir_value = node_inst.get_nodeattr("ip_path")
        assert os.path.isdir(
            ip_dir_value
        ), """The directory that should
        contain the generated ip blocks doesn't exist."""
        ip_dirs += [ip_dir_value]
        if node.op_type.startswith("MVAU") or node.op_type == "Thresholding_hls":
            if node_inst.get_nodeattr("mem_mode") == "internal_decoupled":
                need_memstreamer = True
    ip_dirs += [ipstitch_path + "/ip"]
    if need_memstreamer:
        # add RTL streamer IP
        ip_dirs.append("$::env(FINN_ROOT)/finn-rtllib/memstream")
    return ip_dirs


def _collapse_generated_ip_repo_path(path):
    """Return the generated-build root Vivado should scan for this IP."""

    if path.startswith("$"):
        return path
    norm_path = os.path.normpath(path)
    parts = norm_path.split(os.sep)
    for i, part in enumerate(parts):
        if part.startswith("finn_temp_file"):
            return os.sep.join(parts[: i + 1]) or os.sep
    for i, part in enumerate(parts):
        if part.startswith("code_gen_ipgen_") or part.startswith("vivado_stitch_proj_"):
            return os.sep.join(parts[:i]) or os.sep
    return norm_path


def _collapse_ip_repo_paths(ip_dirs, collapse_generated=True):
    """Deduplicate IP repo paths, optionally collapsing generated repos."""

    paths = list(ip_dirs)
    has_list_prefix = bool(paths and paths[0] == "list")
    if has_list_prefix:
        paths = paths[1:]

    collapsed = []
    seen = set()
    for path in paths:
        repo_path = _collapse_generated_ip_repo_path(path) if collapse_generated else path
        if repo_path in seen:
            continue
        seen.add(repo_path)
        collapsed.append(repo_path)
    if has_list_prefix:
        return ["list"] + collapsed
    return collapsed


def _ip_repo_paths_cmds(ip_dirs, append=False, collapse_generated=True, max_cmd_chars=20000):
    paths = _collapse_ip_repo_paths(ip_dirs, collapse_generated=collapse_generated)
    if paths and paths[0] == "list":
        paths = paths[1:]
    cmds = []
    if not append:
        cmds.append("set_property ip_repo_paths [list] [current_project]")
    chunk = []
    chunk_chars = 0
    for path in paths:
        path_len = len(path) + 1
        if chunk and chunk_chars + path_len > max_cmd_chars:
            cmds.append(
                "set_property ip_repo_paths "
                "[concat [get_property ip_repo_paths [current_project]] "
                "[list %s]] [current_project]" % " ".join(chunk)
            )
            chunk = []
            chunk_chars = 0
        chunk.append(path)
        chunk_chars += path_len
    if chunk:
        cmds.append(
            "set_property ip_repo_paths "
            "[concat [get_property ip_repo_paths [current_project]] "
            "[list %s]] [current_project]" % " ".join(chunk)
        )
    return cmds


def _ip_repo_paths_set_cmd(ip_dirs, collapse_generated=True):
    return "\n".join(
        _ip_repo_paths_cmds(ip_dirs, append=False, collapse_generated=collapse_generated)
    )


def _ip_repo_paths_append_cmd(ip_dirs, collapse_generated=True):
    return "\n".join(
        _ip_repo_paths_cmds(ip_dirs, append=True, collapse_generated=collapse_generated)
    )


def _add_files_copy_to_cmd(source_target, source_file):
    return "add_files -force -copy_to %s -norecurse %s" % (source_target, source_file)


def _bd_clock_freq_hz_cmds(clk_ns, has_clk2x):
    fclk_hz = round(1e9 / float(clk_ns))
    cmds = ["set_property CONFIG.FREQ_HZ %d [get_bd_ports /ap_clk]" % fclk_hz]
    if has_clk2x:
        cmds.append(
            "set_property CONFIG.FREQ_HZ %d [get_bd_ports /ap_clk2x]" % (2 * fclk_hz)
        )
    return cmds


class FINNLoop(HWCustomOp, RTLBackend):
    """Class that corresponds to the meta/container node FINN loop
    which is a placeholder for a group of fpgadataflow nodes that have been separated
    out into a FINN-ONNX model of its own and are meant to be executed in a loop."""

    def get_nodeattr_types(self):
        my_attrs = {
            "body": ("g", True, ""),
            "iteration": ("i", False, 1),
            # FINN input datatype
            "inputDataType": ("s", True, ""),
            # FINN output datatype
            "outputDataType": ("s", True, ""),
            # Path to save per-iteration execution context (cppsim only).
            # If non-empty, each iteration's full context is saved to this path.
            "iteration_context_path": ("s", False, ""),
            # FINNLoop input indexes that are dynamic loop-invariant streams.
            # These inputs are provided once per outer invocation and replayed
            # to the loop body for every iteration instead of being compiled
            # into static per-iteration memstream parameter sets.
            "dynamic_loop_inputs": ("ints", False, []),
            # Sequential is the existing MLO implementation. Overlapped is only
            # valid when an explicit graph/RTL scheduler artifact is available.
            "loop_scheduler_mode": ("s", False, "sequential", {"sequential", "overlapped"}),
            "overlapped_body_ii_cycles": ("i", False, 0),
            "overlapped_scheduler_artifact": ("s", False, ""),
            "overlapped_scheduler_dcp_ready": ("i", False, 0),
            # Clock period from PrepareIP, needed when packaging loop-local BDs.
            "clk_ns": ("f", False, 10.0),
        }
        my_attrs.update(HWCustomOp.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def _dynamic_loop_inputs(self):
        return set(int(x) for x in self.get_nodeattr("dynamic_loop_inputs"))

    def _loop_overhead_per_iter(self):
        return 40

    def _overlapped_scheduler_builtin(self):
        artifact = self.get_nodeattr("overlapped_scheduler_artifact")
        if artifact in OVERLAPPED_SCHEDULER_BUILTINS:
            return artifact
        return None

    def _uses_builtin_stream_feedback_scheduler(self):
        return (
            self.get_nodeattr("loop_scheduler_mode") == "overlapped"
            and self._overlapped_scheduler_builtin() == "stream_feedback"
            and int(self.get_nodeattr("overlapped_scheduler_dcp_ready")) == 1
        )

    def _overlapped_scheduler_artifact_candidates(self):
        artifact = self.get_nodeattr("overlapped_scheduler_artifact")
        if artifact == "":
            return []
        if artifact in OVERLAPPED_SCHEDULER_BUILTINS:
            return []
        path = Path(artifact)
        if path.is_absolute():
            return [path]
        candidates = []
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        if code_gen_dir != "":
            candidates.append(Path(code_gen_dir) / path)
        candidates.append(path)
        return candidates

    def _overlapped_scheduler_artifact_path(self):
        candidates = self._overlapped_scheduler_artifact_candidates()
        for path in candidates:
            if path.is_file():
                return path
        return candidates[0] if candidates else None

    def _overlapped_scheduler_artifact_ready(self):
        if self._overlapped_scheduler_builtin() is not None:
            return int(self.get_nodeattr("overlapped_scheduler_dcp_ready")) == 1
        path = self._overlapped_scheduler_artifact_path()
        return (
            path is not None
            and path.is_file()
            and path.suffix.lower() in OVERLAPPED_SCHEDULER_ARTIFACT_SUFFIXES
        )

    def _loop_wrapper_module_name(self):
        return f"{self.onnx_node.name}_loop_cont_wrapper"

    def _loop_wrapper_source_path(self):
        code_gen_dir = Path(self.get_nodeattr("code_gen_dir_ipgen"))
        suffix = ".v"
        if self.get_nodeattr("loop_scheduler_mode") == "overlapped":
            artifact_path = self._overlapped_scheduler_artifact_path()
            if (
                artifact_path is not None
                and artifact_path.suffix.lower() in OVERLAPPED_SCHEDULER_WRAPPER_SUFFIXES
            ):
                suffix = artifact_path.suffix.lower()
        return code_gen_dir / f"{self.onnx_node.name}_wrapper{suffix}"

    def _overlapped_scheduler_wrapper_source(self):
        if self.get_nodeattr("loop_scheduler_mode") != "overlapped":
            return None
        if int(self.get_nodeattr("overlapped_scheduler_dcp_ready")) != 1:
            raise RuntimeError(
                f"{self.onnx_node.name}: overlapped FINNLoop HDL generation requires "
                "overlapped_scheduler_dcp_ready=1 and a real scheduler wrapper RTL "
                "artifact. Estimate-only scheduler contracts must not be used for DCP."
            )
        if self._overlapped_scheduler_builtin() == "stream_feedback":
            return None
        artifact_path = self._overlapped_scheduler_artifact_path()
        if artifact_path is None or not artifact_path.is_file():
            raise RuntimeError(
                f"{self.onnx_node.name}: overlapped FINNLoop HDL generation requires "
                f"an existing scheduler wrapper RTL artifact; got {artifact_path}"
            )
        if artifact_path.suffix.lower() not in OVERLAPPED_SCHEDULER_WRAPPER_SUFFIXES:
            allowed = ", ".join(sorted(OVERLAPPED_SCHEDULER_WRAPPER_SUFFIXES))
            raise RuntimeError(
                f"{self.onnx_node.name}: DCP-ready overlapped FINNLoop HDL generation "
                f"requires a Verilog/SystemVerilog wrapper ({allowed}); got "
                f"{artifact_path}"
            )
        wrapper_module = self._loop_wrapper_module_name()
        text = artifact_path.read_text()
        if re.search(rf"\bmodule\s+{re.escape(wrapper_module)}\b", text) is None:
            raise RuntimeError(
                f"{self.onnx_node.name}: overlapped scheduler wrapper {artifact_path} "
                f"must define module {wrapper_module}"
            )
        return artifact_path

    def _write_overlapped_scheduler_wrapper(self):
        artifact_path = self._overlapped_scheduler_wrapper_source()
        if artifact_path is None:
            return False
        wrapper_path = self._loop_wrapper_source_path()
        wrapper_path.parent.mkdir(parents=True, exist_ok=True)
        if artifact_path.resolve() != wrapper_path.resolve():
            shutil.copyfile(artifact_path, wrapper_path)
        return True

    def _overlapped_exp_cycles(self):
        if self.get_nodeattr("loop_scheduler_mode") != "overlapped":
            return None
        body_ii_cycles = int(self.get_nodeattr("overlapped_body_ii_cycles"))
        if body_ii_cycles <= 0:
            raise RuntimeError(
                f"{self.onnx_node.name}: overlapped FINNLoop scheduling requires "
                "overlapped_body_ii_cycles > 0"
            )
        artifact_path = self._overlapped_scheduler_artifact_path()
        if not self._overlapped_scheduler_artifact_ready():
            allowed = ", ".join(sorted(OVERLAPPED_SCHEDULER_ARTIFACT_SUFFIXES))
            allowed = allowed + ", " + ", ".join(sorted(OVERLAPPED_SCHEDULER_BUILTINS))
            raise RuntimeError(
                f"{self.onnx_node.name}: overlapped FINNLoop scheduling requires an "
                f"existing graph/RTL artifact ({allowed}); got "
                f"{artifact_path if artifact_path is not None else 'missing'}"
            )
        iteration = self.get_nodeattr("iteration")
        return (body_ii_cycles + self._loop_overhead_per_iter()) * iteration

    def get_nodeattr(self, name):
        """Get a node attribute by name. Data is stored inside the ONNX node's
        AttributeProto container. Attribute must be part of get_nodeattr_types.
        Default value is returned if attribute is not set."""
        try:
            (dtype, req, def_val, allowed_values) = self.get_nodeattr_def(name)
            attr = get_by_name(self.onnx_node.attribute, name)
            if attr is not None:
                # dtype indicates which ONNX Attribute member to use
                # g : graph
                if dtype == "g":
                    ret = attr.__getattribute__(dtype)
                    ret = ModelWrapper(qonnx_make_model(ret))
                    return ret
                else:
                    return super().get_nodeattr(name)
            else:
                if req:
                    raise Exception(
                        """Required attribute %s unspecified in
                    a %s node"""
                        % (name, self.onnx_node.op_type)
                    )
                else:
                    # not set, return default value
                    return def_val
        except KeyError:
            raise AttributeError("Op has no such attribute: " + name)

    def set_nodeattr(self, name, value):
        """Set a node attribute by name. Data is stored inside the ONNX node's
        AttributeProto container. Attribute must be part of get_nodeattr_types."""
        try:
            (dtype, req, def_val, allowed_values) = self.get_nodeattr_def(name)
            attr = get_by_name(self.onnx_node.attribute, name)
            if attr is not None:
                # dtype indicates which ONNX Attribute member to use
                # g : graph
                if dtype == "g":
                    attr.g.CopyFrom(value)
                else:
                    super().set_nodeattr(name, value)
            else:
                super().set_nodeattr(name, value)
        except KeyError:
            raise AttributeError("Op has no such attribute: " + name)

    def _body_input_endpoint(self, ind):
        loop_body = self.get_nodeattr("body")
        tensor = loop_body.graph.input[ind].name
        consumer = loop_body.find_consumer(tensor)
        if consumer is None:
            return loop_body, tensor, None, None
        return loop_body, tensor, consumer, list(consumer.input).index(tensor)

    def _body_output_endpoint(self, ind):
        loop_body = self.get_nodeattr("body")
        tensor = loop_body.graph.output[ind].name
        producer = loop_body.find_producer(tensor)
        if producer is None:
            return loop_body, tensor, None, None
        return loop_body, tensor, producer, list(producer.output).index(tensor)

    def get_normal_input_shape(self, ind=0):
        loop_body, tensor, node, node_input_ind = self._body_input_endpoint(ind)
        if node is None:
            ishape = loop_body.get_tensor_shape(tensor)
        elif is_custom_op(node.domain):
            inst = getHWCustomOp(node)  # No model context: read only
            ishape = inst.get_normal_input_shape(node_input_ind)
        else:
            ishape = loop_body.get_tensor_shape(tensor)
        return ishape

    def get_normal_output_shape(self, ind=0):
        loop_body, tensor, node, node_output_ind = self._body_output_endpoint(ind)
        if node is None:
            oshape = loop_body.get_tensor_shape(tensor)
        elif is_custom_op(node.domain):
            inst = getHWCustomOp(node)  # No model context: read only
            oshape = inst.get_normal_output_shape(node_output_ind)
        else:
            oshape = loop_body.get_tensor_shape(tensor)
        return oshape

    def get_folded_input_shape(self, ind=0):
        loop_body, tensor, node, node_input_ind = self._body_input_endpoint(ind)
        if node is None:
            ishape = loop_body.get_tensor_shape(tensor)
        elif is_custom_op(node.domain):
            inst = getHWCustomOp(node)  # No model context: read only
            ishape = inst.get_folded_input_shape(node_input_ind)
        else:
            ishape = loop_body.get_tensor_shape(tensor)
        return ishape

    def get_folded_output_shape(self, ind=0):
        loop_body, tensor, node, node_output_ind = self._body_output_endpoint(ind)
        if node is None:
            return loop_body.get_tensor_shape(tensor)
        if is_custom_op(node.domain):
            inst = getHWCustomOp(node)  # No model context: read only
            return inst.get_folded_output_shape(node_output_ind)
        return loop_body.get_tensor_shape(tensor)

    def infer_node_datatype(self, model):
        pass

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        if ind == 0:
            idt = DataType[self.get_nodeattr("inputDataType")]
        else:
            loop_body, tensor, node, node_input_ind = self._body_input_endpoint(ind)
            if node is not None and is_custom_op(node.domain):
                inst = getHWCustomOp(node)  # No model context: read only
                idt = inst.get_input_datatype(node_input_ind)
            else:
                idt = loop_body.get_tensor_datatype(tensor)
        return idt

    def get_output_datatype(self, ind=0):
        odt = DataType[self.get_nodeattr("outputDataType")]
        return odt

    def get_instream_width(self, ind=0):
        loop_body, tensor, node, node_input_ind = self._body_input_endpoint(ind)
        if node is not None and is_custom_op(node.domain):
            inst = getHWCustomOp(node)  # No model context: read only
            iwidth = inst.get_instream_width(node_input_ind)
        else:
            iwidth = loop_body.get_tensor_datatype(tensor).bitwidth()
        return iwidth

    def get_exp_cycles(self):
        overlapped_cycles = self._overlapped_exp_cycles()
        if overlapped_cycles is not None:
            return overlapped_cycles

        loop_body = self.get_nodeattr("body")
        loop_body = loop_body.transform(AnnotateCycles())
        self.set_nodeattr("body", loop_body.graph)

        iteration = self.get_nodeattr("iteration")
        body_cycles = loop_body.analysis(dataflow_performance)["critical_path_cycles"]
        return (body_cycles + self._loop_overhead_per_iter()) * iteration

    def get_outstream_width(self, ind=0):
        loop_body, tensor, node, node_output_ind = self._body_output_endpoint(ind)
        if node is not None and is_custom_op(node.domain):
            inst = getHWCustomOp(node)  # No model context: read only
            return inst.get_outstream_width(node_output_ind)
        return loop_body.get_tensor_datatype(tensor).bitwidth()

    def get_number_output_values(self):
        loop_body, tensor, node, node_output_ind = self._body_output_endpoint(0)
        if node is not None and is_custom_op(node.domain):
            inst = getHWCustomOp(node)  # No model context: read only
            if node_output_ind == 0:
                return inst.get_number_output_values()
            return int(np.prod(inst.get_folded_output_shape(node_output_ind)[:-1]))
        return int(np.prod(loop_body.get_tensor_shape(tensor)))

    def prepare_rtlsim(self, behav=False):
        """Creates a xsi emulation library for the RTL code generated
        for this node, sets the rtlsim_so attribute to its path."""

        vivado_stitch_proj_dir = self.get_nodeattr("code_gen_dir_ipgen")
        with open(vivado_stitch_proj_dir + "/all_verilog_srcs.txt", "r") as f:
            all_verilog_srcs = f.read().split()
        top_module_file_name = os.path.basename(os.path.realpath(self.get_nodeattr("ipgen_path")))
        top_module_name = top_module_file_name.strip(".v")
        single_src_dir = make_build_dir("rtlsim_" + top_module_name + "_")
        trace_file = self.get_nodeattr("rtlsim_trace")
        debug = not (trace_file is None or trace_file == "")
        rtlsim_so = finnxsi.compile_sim_obj(
            top_module_name, all_verilog_srcs, single_src_dir, debug, behav
        )
        # save generated lib filename in attribute
        sim_base, sim_rel = rtlsim_so
        self.set_nodeattr("rtlsim_so", sim_base + "/" + sim_rel)

    def derive_characteristic_fxns(self, period):
        mlo_prehook = mlo_prehook_func_factory(self.onnx_node)
        super().derive_characteristic_fxns(period, pre_hook=mlo_prehook)

    def execute_node(self, context, graph):
        node = self.onnx_node
        inp_values = context[node.input[0]]
        if self.get_nodeattr("exec_mode") == "rtlsim":
            # prepare input io_dict
            io_dict = {"inputs": {}, "outputs": {}}
            itensor = inp_values.reshape(self.get_folded_input_shape(0))
            idt = self.get_input_datatype(0)
            iwidth = self.get_instream_width(0)
            # pack input for rtlsim
            packed_input = npy_to_rtlsim_input(itensor, idt, iwidth)
            io_dict["inputs"]["in0"] = packed_input
            io_dict["outputs"]["out0"] = []
            mlo_prehook = mlo_prehook_func_factory(self.onnx_node)
            sim = self.get_rtlsim()
            # reset and call rtlsim, including any pre/post hooks
            self.reset_rtlsim(sim)
            mlo_prehook(sim)
            self.rtlsim_multi_io(
                sim,
                io_dict,
            )
            self.close_rtlsim(sim)
            odt = self.get_output_datatype(0)
            o_folded_shape = self.get_folded_output_shape(0)
            owidth = self.get_outstream_width(0)
            packed_output = io_dict["outputs"]["out0"]
            o_folded_tensor = rtlsim_output_to_npy(
                packed_output, None, odt, o_folded_shape, owidth, odt.bitwidth()
            )
            oshape = self.get_normal_output_shape(0)
            result = o_folded_tensor.reshape(oshape)
        else:
            loop_body = self.get_nodeattr("body")
            # for each iteration run execution
            iteration = self.get_nodeattr("iteration")
            iteration_context_path = self.get_nodeattr("iteration_context_path")
            save_iteration_context = (
                iteration_context_path is not None and iteration_context_path != ""
            )
            all_iteration_contexts = {}
            dynamic_loop_inputs = self._dynamic_loop_inputs()
            for i_iter in range(iteration):
                # set the right parameters
                input_dict = {}
                for i, inp in enumerate(node.input):
                    if i == 0:
                        input_dict[loop_body.graph.input[i].name] = inp_values
                    elif i in dynamic_loop_inputs:
                        input_dict[loop_body.graph.input[i].name] = context[node.input[i]]
                    else:
                        params = context[node.input[i]]
                        input_dict[loop_body.graph.input[i].name] = params[i_iter]
                outp_dict = oxe.execute_onnx(loop_body, input_dict, return_full_exec_context=True)
                inp_values = outp_dict[loop_body.graph.output[0].name]
                # Save iteration context if enabled
                if save_iteration_context:
                    for tensor_name, tensor_val in outp_dict.items():
                        # Skip empty tensor name (dummy entry for default values)
                        if tensor_name:
                            key = f"iter_{i_iter}_{tensor_name}"
                            all_iteration_contexts[key] = tensor_val
            result = outp_dict[loop_body.graph.output[0].name]
            # Save all iteration contexts to file
            if save_iteration_context:
                np.savez(iteration_context_path, **all_iteration_contexts)
        context[node.output[0]] = np.asarray(result, dtype=np.float32)

    def generate_hdl(self, model, fpgapart, clk):
        # Generate params as part of IP preparation
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        self.set_nodeattr("clk_ns", float(clk))
        self._overlapped_scheduler_wrapper_source()
        self.generate_hdl_stream_tap()
        self.generate_hdl_dynamic_replay()
        self.generate_params(model, code_gen_dir)
        if self._write_overlapped_scheduler_wrapper():
            return
        code_gen_dict = {}
        code_gen_dict["$LOOP_CONTROL_WRAPPER_NAME$"] = [self._loop_wrapper_module_name()]
        code_gen_dict["$N_MAX_LAYERS$"] = (str(self.get_nodeattr("iteration")),)
        code_gen_dict["$N_LAYERS$"] = [str(self.get_nodeattr("iteration"))]
        code_gen_dict["$ILEN_BITS$"] = [str(self.get_instream_width(0))]
        code_gen_dict["$OLEN_BITS$"] = [str(self.get_outstream_width(0))]

        input_elements = np.prod(self.get_normal_input_shape(0))
        input_bytes = (input_elements * self.get_input_datatype(0).bitwidth() + 8 - 1) // 8
        output_elements = np.prod(self.get_normal_output_shape(0))
        output_bytes = (output_elements * self.get_output_datatype(0).bitwidth() + 8 - 1) // 8
        code_gen_dict["$INPUT_BYTES$"] = [str(input_bytes)]
        code_gen_dict["$OUTPUT_BYTES$"] = [str(output_bytes)]

        # round up to next power of 2
        input_bytes_rounded_to_power_of_2 = 2 ** (math.ceil(math.log2(input_bytes)))
        code_gen_dict["$LAYER_OFFS_INT$"] = [
            str(input_bytes_rounded_to_power_of_2)
        ]  # need to get correct value

        template_path = os.environ["FINN_ROOT"] + "/finn-rtllib/mlo/loop_control_wrapper.v"
        with open(template_path, "r") as f:
            template_wrapper = f.read()
        if self._uses_builtin_stream_feedback_scheduler():
            template_wrapper = template_wrapper.replace(
                "loop_control #(",
                "overlapped_loop_control #(",
            )
        for key, value in code_gen_dict.items():
            # transform list into long string separated by '\n'
            code_gen_line = "\n".join(value)
            template_wrapper = template_wrapper.replace(key, code_gen_line)
        with open(self._loop_wrapper_source_path(), "w") as f:
            f.write(template_wrapper)

    def generate_params(self, model, path):
        iteration = self.get_nodeattr("iteration")
        loop_node = self.onnx_node
        loop_body = self.get_nodeattr("body")
        activation_inputs = max(1, len(loop_node.output))
        dynamic_loop_inputs = self._dynamic_loop_inputs()
        for loop_input_index in range(activation_inputs, len(loop_node.input)):
            if loop_input_index in dynamic_loop_inputs:
                continue
            inp = loop_node.input[loop_input_index]
            params = model.get_initializer(inp)
            if params is None:
                producer = model.find_producer(inp)
                producer_desc = (
                    "graph input"
                    if producer is None
                    else f"{producer.name or '<unnamed>'}:{producer.op_type}"
                )
                raise RuntimeError(
                    f"{loop_node.name}: FINNLoop input {loop_input_index} ({inp}) is not a "
                    f"static initializer; producer is {producer_desc}. RTL MLO parameter "
                    "code generation currently supports only static per-iteration parameter "
                    "stacks. Dynamic invariant streams must be precomputed/folded before "
                    "loop codegen or lowered through a dedicated replay/streaming path."
                )
            param_dtype = model.get_tensor_datatype(inp)
            assert params.shape[0] == iteration
            # get node that initializer is attached to
            loop_tensor = loop_body.graph.input[loop_input_index].name
            param_node = loop_body.find_consumer(loop_tensor)
            for iter in range(iteration):
                loop_body.set_initializer(loop_tensor, params[iter])
                loop_body.set_tensor_datatype(loop_tensor, param_dtype)
                inst = getHWCustomOp(param_node, model)
                inst.generate_params(loop_body, path)
                param_file = "{}/memblock.dat".format(path)
                new_param_file = "{}/{}_memblock_{}.dat".format(path, param_node.op_type, iter)
                if param_node.op_type.startswith("MVAU") or param_node.op_type.startswith(
                    "Elementwise"
                ):
                    # rename so it doesn't get overwritten
                    shutil.move(param_file, new_param_file)
                    # also rename simd-flipped npy for external_mem MVAU nodes
                    npy_file = "{}/input_1.npy".format(path)
                    if os.path.isfile(npy_file):
                        new_npy_file = "{}/{}_input1_{}.npy".format(path, param_node.op_type, iter)
                        shutil.move(npy_file, new_npy_file)
                elif param_node.op_type.startswith("Thresholding"):
                    # get all generated Thresholding dat files
                    pe = inst.get_nodeattr("PE")
                    output_data_type = inst.get_nodeattr("outputDataType")
                    o_bitwidth = DataType[output_data_type].bitwidth()
                    param_files = []
                    for stage in range(o_bitwidth):
                        for pe_value in range(pe):
                            param_files.append(
                                path
                                + "/%s_threshs_%s_%s.dat"
                                % (
                                    param_node.name,
                                    pe_value,
                                    stage,
                                )
                            )
                    for param_file in param_files:
                        param_path = Path(param_file)
                        new_param_file = param_path.with_name(
                            param_path.stem + "_i" + str(iter) + param_path.suffix
                        )
                        shutil.move(param_path, new_param_file)
                else:
                    raise Exception

            if param_node.op_type.startswith("MVAU") or param_node.op_type.startswith(
                "Elementwise"
            ):
                # concatinate all .dat files together
                param_file = "{}/memblock_{}_id_{}.dat".format(
                    path, param_node.op_type, loop_input_index
                )
                with open(param_file, "w") as outfile:
                    for iter in range(iteration):
                        memblock_file = "{}/{}_memblock_{}.dat".format(
                            path, param_node.op_type, iter
                        )
                        with open(memblock_file, "r") as infile:
                            for line in infile:
                                outfile.write(line)
                        os.remove(memblock_file)
                # concatenate all .npy files together (simd-flipped for AXI-MM sim)
                npy_parts = []
                for iter in range(iteration):
                    npy_file = "{}/{}_input1_{}.npy".format(path, param_node.op_type, iter)
                    if os.path.isfile(npy_file):
                        npy_parts.append(np.load(npy_file))
                        os.remove(npy_file)
                if npy_parts:
                    combined_npy = np.concatenate(npy_parts, axis=1)
                    npy_out = "{}/input1_{}_id_{}.npy".format(
                        path, param_node.op_type, loop_input_index
                    )
                    np.save(npy_out, combined_npy)
                # Replace the path for the dat files in the ipgen files.
                # Adapted from transformations.fpgadataflow.replace_verilog_relpaths
                if param_node.op_type.startswith("MVAU") or param_node.op_type.startswith(
                    "Elementwise"
                ):
                    param_customop = getHWCustomOp(param_node, model)
                    ipgen_path = param_customop.get_nodeattr("code_gen_dir_ipgen")
                    if ipgen_path is not None and os.path.isdir(ipgen_path):
                        for dname, dirs, files in os.walk(ipgen_path):
                            for fname in files:
                                if fname.endswith("_memstream_wrapper.v"):
                                    fpath = os.path.join(dname, fname)
                                    with open(fpath, "r") as f:
                                        s = f.read()
                                    old = "%s/memblock.dat" % ipgen_path
                                    new = "%s/memblock_%s_id_%s.dat" % (
                                        path,
                                        param_node.op_type,
                                        loop_input_index,
                                    )
                                    s = s.replace(old, new)
                                    with open(fpath, "w") as f:
                                        f.write(s)
            elif param_node.op_type.startswith("Thresholding"):
                # concatinate all .dat files together
                pe = inst.get_nodeattr("PE")
                output_data_type = inst.get_nodeattr("outputDataType")
                o_bitwidth = DataType[output_data_type].bitwidth()
                for stage in range(o_bitwidth):
                    for pe_value in range(pe):
                        param_file = path + "/Thresholding_id_%s_threshs_%s_%s.dat" % (
                            loop_input_index,
                            pe_value,
                            stage,
                        )
                        with open(param_file, "w") as outfile:
                            for iter in range(iteration):
                                iter_file = "{}/{}_threshs_{}_{}_i{}.dat".format(
                                    path, param_node.name, pe_value, stage, iter
                                )
                                with open(iter_file, "r") as infile:
                                    cnt = 0
                                    for line in infile:
                                        if cnt == 0:
                                            hex_len = len(line.strip())
                                        cnt += 1
                                        outfile.write(line)
                                    # is power of 2?
                                    if (cnt & (cnt - 1)) != 0:
                                        # pad with max value
                                        next_pow2 = 2 ** math.ceil(math.log2(cnt))
                                        pad_val = 2**o_bitwidth - 1
                                        for _ in range(next_pow2 - cnt):
                                            # write out as hex of len hex_len
                                            outfile.write(hex(pad_val)[2:].zfill(hex_len) + "\n")
                                os.remove(iter_file)

                # Replace the path for the dat files in the ipgen files
                # Adapted from transformations.fpgadataflow.replace_verilog_relpaths
                param_customop = getHWCustomOp(param_node, model)
                ipgen_path = param_customop.get_nodeattr("ipgen_path")
                if ipgen_path is not None and os.path.isdir(ipgen_path):
                    for dname, dirs, files in os.walk(ipgen_path):
                        for fname in files:
                            if fname.endswith(".v"):
                                fpath = os.path.join(dname, fname)
                                with open(fpath, "r") as f:
                                    s = f.read()
                                old = "./%s" % param_node.name
                                new = "%s/Thresholding_id_%s" % (path, loop_input_index)
                                s = s.replace(old, new)
                                with open(fpath, "w") as f:
                                    f.write(s)

    def generate_hdl_stream_tap(self):
        """Helper function to generate verilog code for stream tap components."""
        template_path = (
            os.environ["FINN_ROOT"] + "/finn-rtllib/stream_tap/hdl/stream_tap_wrapper_template.v"
        )
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        iteration = self.get_nodeattr("iteration")
        loop_body = self.get_nodeattr("body")
        graph_inputs = [x.name for x in loop_body.graph.input]
        dynamic_loop_inputs = self._dynamic_loop_inputs()
        # TODO check if this needs to be padded
        data_width = DataType.get_smallest_possible(iteration).bitwidth()
        # pad to nearest multiple of 8
        data_width = roundup_to_integer_multiple(data_width, 8)
        for node in loop_body.graph.node:
            node_inst = getHWCustomOp(node)  # No model context: read only
            if node_inst.get_nodeattr("mlo_max_iter"):
                input_index = graph_inputs.index(node.input[1])
                if input_index in dynamic_loop_inputs:
                    continue
                # calculate TAP_REP
                # for Thresholds this value is fm size / pe
                # for all other param nodes it is 1
                tap_rep = 1
                if node.op_type == "Thresholding_rtl":
                    tap_rep = np.prod(node_inst.get_folded_input_shape(0)[:-1])
                elif node.op_type.startswith("Elementwise") and hasattr(
                    node_inst, "calc_wmem_reps"
                ):
                    tap_rep = node_inst.calc_wmem_reps()
                stname = "IN_%s" % input_index
                code_gen_dict = {
                    "$MODULE_NAME$": [stname],
                    "$DATA_WIDTH$": [str(data_width)],
                    "$TAP_REP$": [str(tap_rep)],
                }
                # apply code generation to template
                with open(template_path, "r") as f:
                    template_wrapper = f.read()
                for key, value in code_gen_dict.items():
                    # transform list into long string separated by '\n'
                    code_gen_line = "\n".join(value)
                    template_wrapper = template_wrapper.replace(key, code_gen_line)
                with open(
                    os.path.join(code_gen_dir, stname + "_stream_tap_wrapper.v"),
                    "w",
                ) as f:
                    f.write(template_wrapper)

    def generate_hdl_dynamic_replay(self):
        """Generate wrappers for dynamic loop-invariant replay streams."""

        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        iteration = self.get_nodeattr("iteration")
        for input_index in sorted(self._dynamic_loop_inputs()):
            if input_index == 0:
                raise RuntimeError(f"{self.onnx_node.name}: activation input cannot be replayed")
            data_width = self.get_instream_width_padded(input_index)
            frame_words = int(np.prod(self.get_folded_input_shape(input_index)[:-1]))
            if data_width <= 0 or frame_words <= 0:
                raise RuntimeError(
                    f"{self.onnx_node.name}: invalid dynamic replay input {input_index} "
                    f"width={data_width} frame_words={frame_words}"
                )
            module_name = f"IN_{input_index}_stream_replay_wrapper"
            wrapper = f"""\
module {module_name} #(
    parameter DATA_WIDTH = {data_width},
    parameter FRAME_WORDS = {frame_words},
    parameter REPS = {iteration}
)(
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axis_0:m_axis_0, ASSOCIATED_RESET ap_rst_n" *)
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
    input ap_clk,
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input ap_rst_n,

    input [DATA_WIDTH-1:0] s_axis_0_TDATA,
    input s_axis_0_TVALID,
    output s_axis_0_TREADY,

    output [DATA_WIDTH-1:0] m_axis_0_TDATA,
    output m_axis_0_TVALID,
    input m_axis_0_TREADY
);

    stream_replay #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAME_WORDS(FRAME_WORDS),
        .REPS(REPS)
    ) inst (
        .clk(ap_clk),
        .rst(!ap_rst_n),
        .s_axis_0_TDATA(s_axis_0_TDATA),
        .s_axis_0_TVALID(s_axis_0_TVALID),
        .s_axis_0_TREADY(s_axis_0_TREADY),
        .m_axis_0_TDATA(m_axis_0_TDATA),
        .m_axis_0_TVALID(m_axis_0_TVALID),
        .m_axis_0_TREADY(m_axis_0_TREADY)
    );

endmodule
"""
            with open(os.path.join(code_gen_dir, module_name + ".v"), "w") as f:
                f.write(wrapper)

    def _intermediate_frame_dwc_ipgen_cmds(self):
        olen_bits = self.get_outstream_width(0)
        ilen_bits = self.get_instream_width(0)
        data_bits = 256

        # DWC write path: body output width -> DMA width (256)
        dwc_sink_s_bytes = (olen_bits + 7) // 8
        dwc_sink_m_bytes = data_bits // 8
        cmd = [
            "create_ip -name axis_dwidth_converter -vendor xilinx.com "
            "-library ip -version 1.1 -module_name if_dwc_sink",
            "set_property -dict [list "
            "CONFIG.S_TDATA_NUM_BYTES {%d} "
            "CONFIG.M_TDATA_NUM_BYTES {%d} "
            "CONFIG.HAS_TLAST {1} "
            "CONFIG.HAS_TKEEP {1} "
            "] [get_ips if_dwc_sink]" % (dwc_sink_s_bytes, dwc_sink_m_bytes),
            "generate_target all [get_ips if_dwc_sink]",
        ]

        # DWC read path: DMA width (256) -> body input width
        dwc_source_s_bytes = data_bits // 8
        dwc_source_m_bytes = (ilen_bits + 7) // 8
        cmd += [
            "create_ip -name axis_dwidth_converter -vendor xilinx.com "
            "-library ip -version 1.1 -module_name if_dwc_source",
            "set_property -dict [list "
            "CONFIG.S_TDATA_NUM_BYTES {%d} "
            "CONFIG.M_TDATA_NUM_BYTES {%d} "
            "CONFIG.HAS_TLAST {1} "
            "CONFIG.HAS_TKEEP {1} "
            "] [get_ips if_dwc_source]" % (dwc_source_s_bytes, dwc_source_m_bytes),
            "generate_target all [get_ips if_dwc_source]",
        ]

        if self._uses_builtin_stream_feedback_scheduler():
            cmd += [
                "create_ip -name axis_dwidth_converter -vendor xilinx.com "
                "-library ip -version 1.1 -module_name if_dwc_feedback",
                "set_property -dict [list "
                "CONFIG.S_TDATA_NUM_BYTES {%d} "
                "CONFIG.M_TDATA_NUM_BYTES {%d} "
                "CONFIG.HAS_TLAST {1} "
                "CONFIG.HAS_TKEEP {1} "
                "] [get_ips if_dwc_feedback]" % (dwc_sink_s_bytes, dwc_source_m_bytes),
                "generate_target all [get_ips if_dwc_feedback]",
            ]
        return cmd

    def _loop_shell_source_ipgen_cmds(self):
        if not self._uses_builtin_stream_feedback_scheduler():
            return []
        return [
            'add_files -norecurse "$::env(FINN_ROOT)/finn-rtllib/mlo/overlapped_loop_control.sv"',
            'add_files -norecurse "$::env(FINN_ROOT)/finn-rtllib/mlo/infrastructure/streamed_feedback_frames.sv"',
        ]

    def ipgen_singlenode_code(self, fpgapart=None):
        prjname = "MakeLoopIP"
        block_name = self.onnx_node.name
        vivado_stitch_proj_dir = self.get_nodeattr("code_gen_dir_ipgen")

        cmd = []

        # Create Vivado axis_dwidth_converter IPs for intermediate_frames DWCs
        cmd += self._intermediate_frame_dwc_ipgen_cmds()

        # add all the generated IP dirs to ip_repo_paths
        ip_dirs = ["list"]
        # add RTL streamer IP
        ip_dirs.append("$::env(FINN_ROOT)/finn-rtllib/memstream")
        loop_model = self.get_nodeattr("body")
        for node in loop_model.graph.node:
            node_inst = getHWCustomOp(node)  # No model context: read only
            ip_dir_value = node_inst.get_nodeattr("ip_path")
            assert os.path.isdir(ip_dir_value), "IP generation directory doesn't exist."
            ip_dirs += [ip_dir_value]
        loop_body_intf_names = eval(loop_model.get_metadata_prop("vivado_stitch_ifnames"))
        body_clk2x_names = loop_body_intf_names.get("clk2x", [])
        cmd.append(_ip_repo_paths_set_cmd(ip_dirs, collapse_generated=False))
        cmd.append("update_ip_catalog")

        # create and instantiate FINNLoop node overarching block design
        cmd.append("create_bd_design %s_bd_design" % (self.onnx_node.name))
        cmd.append("create_bd_cell -type hier %s" % (self.onnx_node.name))
        clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
        rst_name = self.get_verilog_top_module_intf_names()["rst"][0]
        clk2x_name = None
        if body_clk2x_names:
            clk2x_name = self.get_verilog_top_module_intf_names()["clk2x"][0]
        # clock and reset
        cmd.append("create_bd_pin -dir I -type clk /%s/%s" % (self.onnx_node.name, clk_name))
        if clk2x_name is not None:
            cmd.append(
                "create_bd_pin -dir I -type clk /%s/%s" % (self.onnx_node.name, clk2x_name)
            )
        cmd.append("create_bd_pin -dir I -type rst /%s/%s" % (self.onnx_node.name, rst_name))
        # interfaces
        node_intf = self.get_verilog_top_module_intf_names()
        m_axis_intfs = node_intf["m_axis"]
        s_axis_intfs = node_intf["s_axis"]
        control_intfs = node_intf["ap_none"]
        mm_intfs = node_intf["aximm"]
        for intf in m_axis_intfs:
            cmd.append(
                "create_bd_intf_pin -mode Master "
                "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s" % (self.onnx_node.name, intf[0])
            )
        for intf in s_axis_intfs:
            cmd.append(
                "create_bd_intf_pin -mode Slave "
                "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s" % (self.onnx_node.name, intf[0])
            )
        for intf in mm_intfs:
            cmd.append(
                "create_bd_intf_pin -mode Master "
                "-vlnv xilinx.com:interface:aximm_rtl:1.0 /%s/%s" % (self.onnx_node.name, intf[0])
            )
        for intf in control_intfs:
            if intf == "done_if":
                cmd.append(
                    "create_bd_pin -from 1 -to 0 -dir O -type data /%s/%s"
                    % (self.onnx_node.name, intf)
                )

        # instantiate loop shell
        loop_shell_name = f"{self.onnx_node.name}/{self.onnx_node.name}_loop_cont_wrapper"
        cmd += self._loop_shell_source_ipgen_cmds()
        cmd.append(
            f"""create_bd_cell -type module -reference \
            {self.onnx_node.name}_loop_cont_wrapper {loop_shell_name}"""
        )
        # connect loop shell to clk and reset
        cmd.append(
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s]"
            % (self.onnx_node.name, rst_name, loop_shell_name, rst_name)
        )
        cmd.append(
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s]"
            % (self.onnx_node.name, clk_name, loop_shell_name, clk_name)
        )
        # "externalize" some of the loop shell signals
        ext_intf_signals = ["in0_V", "out0_V", "m_axi_hbm"]
        ext_signals = ["done_if"]
        for sig in ext_intf_signals:
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/%s] [get_bd_intf_pins %s/%s]"
                % (self.onnx_node.name, sig, loop_shell_name, sig)
            )
        for sig in ext_signals:
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s]"
                % (self.onnx_node.name, sig, loop_shell_name, sig)
            )

        # stream tap graph generation
        loop_body = self.get_nodeattr("body")
        dynamic_loop_inputs = self._dynamic_loop_inputs()
        source_target = "./ip/verilog/rtl_ops/%s" % self.onnx_node.name
        cmd.append("file mkdir %s" % source_target)
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        # create a hierarchy for this layer, with the same port names
        stg_intf = {}
        stg_intf["clk"] = self.get_verilog_top_module_intf_names()["clk"]
        stg_intf["rst"] = self.get_verilog_top_module_intf_names()["rst"]
        bd_name = f"{self.onnx_node.name}/stream_tap_graph"
        cmd.append("create_bd_cell -type hier %s" % bd_name)
        # clock and reset
        cmd.append("create_bd_pin -dir I -type clk /%s/%s" % (bd_name, clk_name))
        cmd.append("create_bd_pin -dir I -type rst /%s/%s" % (bd_name, rst_name))
        # streams
        cmd.append(
            "create_bd_intf_pin -mode Master "
            "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/m_axis_0" % bd_name
        )
        cmd.append(
            "connect_bd_intf_net [get_bd_intf_pins %s/m_axis_0] "
            "[get_bd_intf_pins %s/s_axis_core_out_fw_idx]" % (bd_name, loop_shell_name)
        )

        cmd.append(
            "create_bd_intf_pin -mode Slave "
            "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/s_axis_0" % bd_name
        )
        for input_index, inp in enumerate(loop_body.graph.input[1:], start=1):
            if input_index in dynamic_loop_inputs:
                cmd.append(
                    "create_bd_intf_pin -mode Slave "
                    "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/s_axis_%d"
                    % (bd_name, input_index)
                )
                continue
            cmd.append(
                "create_bd_intf_pin -mode Master "
                "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/m_axis_%d"
                % (bd_name, input_index)
            )
        # get stream tap (+ skid)  components
        skid_file = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/skid/skid.sv")
        stream_tap_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/stream_tap/hdl/")
        stream_replay_file = os.path.join(
            os.environ["FINN_ROOT"], "finn-rtllib/stream_replay/hdl/stream_replay.sv"
        )
        file_suffix = "_stream_tap_wrapper.v"
        replay_file_suffix = "_stream_replay_wrapper.v"
        # automatically find stream tap verilog components in code generation directory
        st_tmpl_names = []
        st_verilog_files = []
        replay_verilog_files = []
        for fname in os.listdir(code_gen_dir):
            if fname.endswith(file_suffix):
                st_verilog_files.append(os.path.join(code_gen_dir, fname))
                st_tmpl_names.append(fname[:-2])
            if fname.endswith(replay_file_suffix):
                replay_verilog_files.append(os.path.join(code_gen_dir, fname))
        sourcefiles = st_verilog_files + [stream_tap_dir + "stream_tap.sv", skid_file]
        if replay_verilog_files:
            sourcefiles += replay_verilog_files + [stream_replay_file]
        for f in sourcefiles:
            cmd += [_add_files_copy_to_cmd(source_target, f)]

        adj_list = adjacency_list(
            loop_body,
            lambda node: node.op_type == "Thresholding_rtl"
            or (
                node.op_type == "MVAU_rtl"
                and any(attr.name == "mlo_max_iter" and attr.i > 0 for attr in node.attribute)
            )
            or (
                node.op_type.startswith("Elementwise")
                and any(attr.name == "mlo_max_iter" and attr.i > 0 for attr in node.attribute)
            ),
        )

        # create map that maps each stream tap to its param node
        st_map = {}
        for input_index, inp in enumerate(loop_body.graph.input[1:], start=1):
            if input_index in dynamic_loop_inputs:
                continue
            consumer = loop_body.find_consumer(inp.name)
            st_map[consumer.name] = (input_index, "IN_%d_stream_tap_wrapper" % input_index)

        # instantiate all stream taps and connect their clk and rst
        for input_index, st_name in st_map.values():
            cmd.append(
                "create_bd_cell -type hier -reference %s /%s/%s" % (st_name, bd_name, st_name)
            )
            # connect
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk]"
                % (bd_name, clk_name, bd_name, st_name)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_rst_n]"
                % (bd_name, rst_name, bd_name, st_name)
            )
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/m_axis_%s] "
                "[get_bd_intf_pins %s/%s/m_axis_1]" % (bd_name, input_index, bd_name, st_name)
            )

        replay_map = {}
        for input_index in sorted(dynamic_loop_inputs):
            replay_name = "IN_%d_stream_replay_wrapper" % input_index
            replay_map[input_index] = replay_name
            cmd.append(
                "create_bd_cell -type hier -reference %s /%s/%s"
                % (replay_name, bd_name, replay_name)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk]"
                % (bd_name, clk_name, bd_name, replay_name)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_rst_n]"
                % (bd_name, rst_name, bd_name, replay_name)
            )
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/s_axis_%d] "
                "[get_bd_intf_pins %s/%s/s_axis_0]"
                % (bd_name, input_index, bd_name, replay_name)
            )

        # prune adj_list to remove join duplicates
        pruned_adj_list = copy.deepcopy(adj_list)

        for key in adj_list:
            if key.startswith("__INPUT") and "INPUT0" not in key:
                del pruned_adj_list[key]
            if "__OUTPUT0__" in adj_list[key] and len(adj_list[key]) > 1:
                pruned_adj_list[key].remove("__OUTPUT0__")

        pruned_adj_list = {tuple(v): k for k, v in pruned_adj_list.items()}  # exchange keys, values
        pruned_adj_list = {v: list(k) for k, v in pruned_adj_list.items()}

        # look for double edges,
        # e.g. input connected to node_x and intermediate node connected to node_x

        pruned_adj_list_copy = copy.deepcopy(pruned_adj_list)

        for key0, value0 in pruned_adj_list_copy.items():
            for key1, value1 in pruned_adj_list_copy.items():
                for val in value1:
                    if val in value0 and key0 != key1:
                        # check which src is in the topological order last
                        # key0
                        node0 = loop_body.get_node_from_name(key0)
                        id0 = (
                            loop_body.get_node_index(node0)
                            if loop_body.get_node_index(node0) is not None
                            else -1
                        )
                        # key1
                        node1 = loop_body.get_node_from_name(key1)
                        id1 = (
                            loop_body.get_node_index(node1)
                            if loop_body.get_node_index(node1) is not None
                            else -1
                        )
                        # if node0 is earlier in the graph remove val from list
                        if id0 < id1:
                            pruned_adj_list[key0].remove(val)

        # filter pruned_adj_list in case some of the values are now empty lists
        pruned_adj_list = {key: value for key, value in pruned_adj_list.items() if value != []}

        # create stg
        for src, dsts in pruned_adj_list.items():
            if all(x.startswith("__OUTPUT") for x in dsts):
                continue
            if "__INPUT0__" in src:
                src_inst_name = bd_name
                src_intf_name = "s_axis_0"
            else:
                src_inst_name = bd_name + "/" + st_map[src][1]
                src_intf_name = "m_axis_0"

            dst_intf_name = "s_axis_0"
            if len(dsts) == 1:
                dst_inst_name = st_map[dsts[0]][1]
                cmd.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                    "[get_bd_intf_pins %s/%s/%s]"
                    % (
                        src_inst_name,
                        src_intf_name,
                        bd_name,
                        dst_inst_name,
                        dst_intf_name,
                    )
                )
            # if node is a fork connect data signals directly
            # and insert AND logic for rdy and vld signals
            elif len(dsts) > 1:
                if "__INPUT0__" in src:
                    cmd.append(
                        "create_bd_cell -type ip "
                        "-vlnv xilinx.com:ip:axis_broadcaster:1.1 %s/axi_broadcaster_0"
                        % (src_inst_name)
                    )
                    cmd.append(
                        "set_property CONFIG.NUM_MI {%d} [get_bd_cells %s/axi_broadcaster_0]"
                        % (len(dsts), src_inst_name)
                    )
                    # connect component to clk, rst and input
                    cmd.append(
                        "connect_bd_net "
                        "[get_bd_pins %s/axi_broadcaster_0/aresetn] [get_bd_pins %s/%s]"
                        % (src_inst_name, bd_name, rst_name)
                    )
                    cmd.append(
                        "connect_bd_net "
                        "[get_bd_pins %s/axi_broadcaster_0/aclk] [get_bd_pins %s/%s]"
                        % (src_inst_name, bd_name, clk_name)
                    )
                    cmd.append(
                        "connect_bd_intf_net "
                        "[get_bd_intf_pins %s/s_axis_0] "
                        "[get_bd_intf_pins %s/axi_broadcaster_0/S_AXIS]"
                        % (src_inst_name, src_inst_name)
                    )
                    for id, dst in enumerate(dsts):
                        dst_inst_name = st_map[dst][1]
                        cmd.append(
                            "connect_bd_intf_net "
                            "[get_bd_intf_pins %s/axi_broadcaster_0/M0%s_AXIS] "
                            "[get_bd_intf_pins %s/%s/%s]"
                            % (src_inst_name, id, src_inst_name, dst_inst_name, dst_intf_name)
                        )
                else:
                    for id, dst in enumerate(dsts):
                        dst_inst_name = st_map[dst][1]
                        cmd.append(
                            "connect_bd_net "
                            "[get_bd_pins %s/%s_TDATA] [get_bd_pins %s/%s/%s_TDATA]"
                            % (
                                src_inst_name,
                                src_intf_name,
                                bd_name,
                                dst_inst_name,
                                dst_intf_name,
                            )
                        )
                        cmd.append(
                            "create_bd_cell -type ip "
                            "-vlnv xilinx.com:ip:util_vector_logic:2.0 %s_util_vector_logic_%d"
                            % (src_inst_name, id)
                        )
                        cmd.append(
                            "set_property CONFIG.C_SIZE {1} [get_bd_cells %s_util_vector_logic_%d]"
                            % (src_inst_name, id)
                        )
                        if id == 0:
                            cmd.append(
                                "connect_bd_net "
                                "[get_bd_pins %s/%s_TVALID] "
                                "[get_bd_pins %s_util_vector_logic_%d/Op1]"
                                % (
                                    src_inst_name,
                                    src_intf_name,
                                    src_inst_name,
                                    id,
                                )
                            )
                        elif id < len(dsts):
                            cmd.append(
                                "connect_bd_net "
                                "[get_bd_pins %s_util_vector_logic_%d/Res] "
                                "[get_bd_pins %s_util_vector_logic_%d/Op1]"
                                % (src_inst_name, id - 1, src_inst_name, id)
                            )

                        cmd.append(
                            "connect_bd_net "
                            "[get_bd_pins %s/%s/%s_TREADY] "
                            "[get_bd_pins %s_util_vector_logic_%d/Op2]"
                            % (bd_name, dst_inst_name, dst_intf_name, src_inst_name, id)
                        )

                    cmd.append(
                        "connect_bd_net "
                        "[get_bd_pins %s_util_vector_logic_%d/Res] [get_bd_pins %s/%s_TREADY]"
                        % (
                            src_inst_name,
                            len(dsts) - 1,
                            src_inst_name,
                            src_intf_name,
                        )
                    )
                    for dst in dsts:
                        dst_inst_name = st_map[dst][1]
                        dst_intf_name = "s_axis_0"
                        cmd.append(
                            "connect_bd_net "
                            "[get_bd_pins %s_util_vector_logic_%d/Res] "
                            "[get_bd_pins %s/%s/%s_TVALID]"
                            % (
                                src_inst_name,
                                len(dsts) - 1,
                                bd_name,
                                dst_inst_name,
                                dst_intf_name,
                            )
                        )
        # connect output of stream tap graph
        last_nodes = [
            key
            for key, value in adj_list.items()
            if all(x.startswith("__OUTPUT0__") for x in value)
        ]
        cmd.append(
            "connect_bd_intf_net [get_bd_intf_pins %s/m_axis_0] "
            "[get_bd_intf_pins %s/%s/m_axis_0]" % (bd_name, bd_name, st_map[last_nodes[0]][1])
        )

        # connect stream tap graph to clk and reset
        cmd.append(
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s]"
            % (self.onnx_node.name, rst_name, bd_name, rst_name)
        )
        cmd.append(
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s]"
            % (self.onnx_node.name, clk_name, bd_name, clk_name)
        )

        loop_body_ipstitch_path = loop_body.get_metadata_prop("vivado_stitch_proj")
        loop_body_vlnv = loop_body.get_metadata_prop("vivado_stitch_vlnv")
        loop_body_intf_names = eval(loop_body.get_metadata_prop("vivado_stitch_ifnames"))
        ip_dirs = ["list"]
        ip_dirs += collect_ip_dirs(loop_body, loop_body_ipstitch_path)
        cmd.append(_ip_repo_paths_set_cmd(ip_dirs, collapse_generated=False))
        cmd.append("update_ip_catalog -rebuild -scan_changes")
        finn_ip_name = f"{self.onnx_node.name}/finn_design_mlo"
        cmd.append("create_bd_cell -type ip -vlnv %s %s" % (loop_body_vlnv, finn_ip_name))
        # connect finn ip to clk and reset
        cmd.append(
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s]"
            % (self.onnx_node.name, rst_name, finn_ip_name, rst_name)
        )
        cmd.append(
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s]"
            % (self.onnx_node.name, clk_name, finn_ip_name, clk_name)
        )
        if clk2x_name is not None:
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s]"
                % (self.onnx_node.name, clk2x_name, finn_ip_name, body_clk2x_names[0])
            )
        # "externalize" some of the loop shell signals
        ext_signals = loop_body_intf_names["aximm"]
        for sig in ext_signals:
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/%s] [get_bd_intf_pins %s/%s]"
                % (self.onnx_node.name, sig[0], finn_ip_name, sig[0])
            )
        # connect components with each other
        # stream tap with finn ip
        connect_signals = loop_body_intf_names["s_axis"]
        for input_index in range(1, len(connect_signals)):
            if input_index in dynamic_loop_inputs:
                cmd.append(
                    "connect_bd_intf_net "
                    "[get_bd_intf_pins %s/%s/m_axis_0] [get_bd_intf_pins %s/s_axis_%d]"
                    % (bd_name, replay_map[input_index], finn_ip_name, input_index)
                )
            else:
                cmd.append(
                    "connect_bd_intf_net "
                    "[get_bd_intf_pins %s/m_axis_%d] [get_bd_intf_pins %s/s_axis_%d]"
                    % (bd_name, input_index, finn_ip_name, input_index)
                )
        # connect stream tap with loop wrapper
        cmd.append(
            "connect_bd_intf_net "
            "[get_bd_intf_pins %s/s_axis_0] [get_bd_intf_pins %s/m_axis_core_in_fw_idx]"
            % (bd_name, loop_shell_name)
        )
        for input_index in sorted(dynamic_loop_inputs):
            cmd.append(
                "connect_bd_intf_net "
                "[get_bd_intf_pins %s/in%d_V] [get_bd_intf_pins %s/s_axis_%d]"
                % (self.onnx_node.name, input_index, bd_name, input_index)
            )
        # connect loop wrapper with finn ip
        cmd.append(
            "connect_bd_intf_net "
            "[get_bd_intf_pins %s/m_axis_core_in] [get_bd_intf_pins %s/s_axis_0]"
            % (loop_shell_name, finn_ip_name)
        )
        cmd.append(
            "connect_bd_intf_net "
            "[get_bd_intf_pins %s/m_axis_0] [get_bd_intf_pins %s/s_axis_core_out]"
            % (finn_ip_name, loop_shell_name)
        )
        cmd.append("make_bd_pins_external  [get_bd_cells %s]" % block_name)
        cmd.append("make_bd_intf_pins_external  [get_bd_cells %s]" % block_name)
        cmd.append("set_property name in0_V [get_bd_intf_ports in0_V_0]")
        for input_index in sorted(dynamic_loop_inputs):
            cmd.append(
                "set_property name in%d_V [get_bd_intf_ports in%d_V_0]"
                % (input_index, input_index)
            )
        cmd.append("set_property name ap_clk [get_bd_ports ap_clk_0]")
        if clk2x_name is not None:
            cmd.append("set_property name ap_clk2x [get_bd_ports ap_clk2x_0]")
        cmd.append("set_property name ap_rst_n [get_bd_ports ap_rst_n_0]")
        cmd.append("set_property name out0_V [get_bd_intf_ports out0_V_0]")
        cmd.append("set_property name m_axi_hbm [get_bd_intf_ports m_axi_hbm_0]")
        cmd.append("set_property name done_if [get_bd_ports done_if_0]")
        # set property name for aximm interfaces
        ext_signals = loop_body_intf_names["aximm"]
        for sig in ext_signals:
            cmd.append(f"set_property name {sig[0]} [get_bd_intf_ports {sig[0]}_0]")
        cmd.extend(
            _bd_clock_freq_hz_cmds(
                self.get_nodeattr("clk_ns"), has_clk2x=clk2x_name is not None
            )
        )
        cmd.append("save_bd_design")
        # cmd.append("validate_bd_design")
        # cmd.append("save_bd_design")
        # create wrapper hdl (for rtlsim later on)
        bd_base = "%s/%s.srcs/sources_1/bd/%s_bd_design" % (
            vivado_stitch_proj_dir,
            prjname,
            block_name,
        )
        bd_filename = "%s/%s_bd_design.bd" % (bd_base, block_name)
        cmd.append("make_wrapper -files [get_files %s] -top" % bd_filename)
        wrapper_base = "%s/%s.gen/sources_1/bd/%s_bd_design" % (
            vivado_stitch_proj_dir,
            prjname,
            block_name,
        )
        wrapper_filename = "%s/hdl/%s_bd_design_wrapper.v" % (wrapper_base, block_name)
        cmd.append("add_files -norecurse %s" % wrapper_filename)
        cmd.append("set_property top %s_bd_design_wrapper [current_fileset]" % block_name)

        # export block design itself as an IP core
        block_vendor = "xilinx_finn"
        block_library = "finn"
        block_vlnv = "%s:%s:%s_bd_design:1.0" % (block_vendor, block_library, block_name)
        cmd.append(
            (
                "ipx::package_project -root_dir %s/ip -vendor %s "
                "-library %s -taxonomy /UserIP -module %s_bd_design -import_files"
            )
            % (vivado_stitch_proj_dir, block_vendor, block_library, block_name)
        )
        # Allow user to customize clock in deployment of stitched IP
        cmd.append("set_property ipi_drc {ignore_freq_hz true} [ipx::current_core]")
        # in some cases, the IP packager seems to infer an aperture of 64K or 4G,
        # preventing address assignment of the DDR_LOW and/or DDR_HIGH segments
        # the following is a hotfix to remove this aperture during IODMA packaging
        cmd.append(
            "ipx::remove_segment -quiet m_axi_gmem0:APERTURE_0 "
            "[ipx::get_address_spaces m_axi_gmem0 -of_objects [ipx::current_core]]"
        )
        cmd.append("set_property core_revision 2 [ipx::find_open_core %s]" % block_vlnv)
        cmd.append("ipx::create_xgui_files [ipx::find_open_core %s]" % block_vlnv)
        # mark bus interface params as user-resolvable to avoid FREQ_MHZ mismatches
        cmd.append(
            "set_property value_resolve_type user [ipx::get_bus_parameters "
            "-of [ipx::get_bus_interfaces -of [ipx::current_core ]]]"
        )
        example_data_dir = os.environ["FINN_ROOT"] + "/src/finn/qnn-data/mdd-data"
        shutil.copytree(example_data_dir, vivado_stitch_proj_dir + "/data", dirs_exist_ok=True)

        template = templates.ip_gen_loop_op

        # transform list into long string separated by '\n'
        cmd = "\n".join(cmd)
        template = template.replace("@IP_GEN@", cmd)
        template = template.replace("@PRJNAME@", prjname)
        template = template.replace("@PRJFOLDER@", vivado_stitch_proj_dir)
        template = template.replace("@FPGAPART@", fpgapart)
        template = template.replace(
            "@TOP_VERILOG_FILE@",
            str(self._loop_wrapper_source_path()),
        )
        f = open(os.path.join(vivado_stitch_proj_dir, "make_loop_ip.tcl"), "w")
        f.write(template)
        f.close()

        # create a shell script and call Vivado
        make_project_sh = vivado_stitch_proj_dir + "/make_loop_ip.sh"
        working_dir = os.environ["PWD"]
        with open(make_project_sh, "w") as f:
            f.write("#!/bin/bash \n")
            f.write("cd {}\n".format(vivado_stitch_proj_dir))
            f.write("vivado -mode batch -source make_loop_ip.tcl\n")
            f.write("cd {}\n".format(working_dir))
        bash_command = ["bash", make_project_sh]
        process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        process_compile.communicate()
        assert os.path.isfile(wrapper_filename), "IPGen failed: %s not found" % (wrapper_filename)
        self.set_nodeattr("ipgen_path", wrapper_filename)
        self.set_nodeattr("ip_path", vivado_stitch_proj_dir + "/ip")
        self.set_nodeattr("gen_top_module", "%s_bd_design_wrapper" % block_name)
        self.set_nodeattr("ip_vlnv", block_vlnv)

    def get_verilog_top_module_intf_names(self):
        # from wrapper template
        addr_bits = 64

        intf_names = {}
        intf_names["clk"] = ["ap_clk"]
        intf_names["rst"] = ["ap_rst_n"]

        intf_names["s_axis"] = []
        # AXI4S slave interface from outside loop to loop control externalize
        # to block diagram interface port and connect to fetch_start component
        intf_names["s_axis"].append(("in0_V", self.get_instream_width_padded(0)))
        for input_index in sorted(self._dynamic_loop_inputs()):
            intf_names["s_axis"].append(
                (f"in{input_index}_V", self.get_instream_width_padded(input_index))
            )

        intf_names["m_axis"] = []
        # AXI4S master interface to drive final loop output externalize
        # to block diagram interface port and connect to store_end component
        intf_names["m_axis"].append(("out0_V", self.get_outstream_width_padded(0)))

        intf_names["aximm"] = []
        # AXI4 master interface for intermediate buffering between layers
        # TODO: rename because it might not be hbm?
        intf_names["aximm"].append(["m_axi_hbm", str(addr_bits)])
        intf_names["axilite"] = []

        # using ap_none field to add control signals
        intf_names["ap_none"] = []
        # done_if should be externalize to a block diagram port
        # and connected to the axil_iw_slv_mlo component
        intf_names["ap_none"].append("done_if")

        loop_body = self.get_nodeattr("body")
        loop_body_intf = eval(loop_body.get_metadata_prop("vivado_stitch_ifnames"))
        if loop_body_intf.get("clk2x", []):
            intf_names["clk2x"] = ["ap_clk2x"]
        for intf in loop_body_intf["aximm"]:
            intf_names["aximm"].append(intf)

        return intf_names

    def code_generation_ipi(self):
        vlnv = self.get_nodeattr("ip_vlnv")
        cmd = []
        # add all the generated IP dirs to ip_repo_paths
        ip_dirs = ["list"]
        # add RTL streamer IP
        loop_body = self.get_nodeattr("body")
        loop_body_ipstitch_path = loop_body.get_metadata_prop("vivado_stitch_proj")
        ip_dirs += collect_ip_dirs(loop_body, loop_body_ipstitch_path)
        cmd.append(_ip_repo_paths_append_cmd(ip_dirs, collapse_generated=False))
        cmd.append("update_ip_catalog -rebuild -scan_changes")
        cmd.append("create_bd_cell -type ip -vlnv %s %s" % (vlnv, self.onnx_node.name))
        return cmd

    def get_rtl_file_list(self, abspath=False):
        pass
