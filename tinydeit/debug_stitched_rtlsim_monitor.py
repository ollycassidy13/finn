#!/usr/bin/env python3
"""Instrument TinyDeiT stitched RTL simulation handshakes."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper

from finn.core.rtlsim_exec import rtlsim_exec
from finn.util.mlo_sim import mlo_prehook_func_factory


def _as_bool(port) -> bool:
    return False if port is None else port.read().as_bool()


def _as_uint(port) -> int | None:
    if port is None:
        return None
    value = port.read().as_hexstr()
    try:
        return int(value, 16)
    except ValueError:
        sanitized = "".join("0" if char.lower() in {"x", "z"} else char for char in value)
        try:
            return int(sanitized, 16)
        except ValueError:
            return None


def _as_hex(port) -> str:
    return "-" if port is None else port.read().as_hexstr()


@dataclass
class ChannelStats:
    seen: int = 0
    handshakes: int = 0
    last_handshake: int = 0
    last_valid: bool = False
    last_ready: bool = False

    def update(self, tick: int, valid: bool, ready: bool) -> None:
        self.last_valid = valid
        self.last_ready = ready
        if valid:
            self.seen += 1
        if valid and ready:
            self.handshakes += 1
            self.last_handshake = tick


class TopMonitor:
    PRELOOP_CHANNELS = [
        "top_in",
        "sf0",
        "sf1",
        "sf2",
        "sf3",
        "th0",
        "sf4",
        "sf5",
        "sf6",
        "sf7",
        "sf8",
        "swg",
        "sf9",
        "sf10",
        "mvau0",
        "sf11",
        "sf12",
        "sf13",
        "sf14",
        "emul0",
        "sf15",
        "eadd0",
        "sf16",
        "ishuf3",
        "sf17",
        "addcls",
        "sf18",
        "eadd1",
        "sf19",
        "loopout",
        "top_out",
    ]

    LOOP_DEBUG_CHANNELS = [
        ("lcore_in", "debug_lcore_in"),
        ("mlo_out", "debug_mlo_out"),
        ("mlo_sf1", "debug_mlo_sf1"),
        ("sf3", "debug_mlo_sf3"),
        ("sf4", "debug_mlo_sf4"),
        ("sf6", "debug_mlo_sf6"),
        ("sf7", "debug_mlo_sf7"),
        ("sf19", "debug_mlo_sf19"),
        ("emul1", "debug_mlo_emul1"),
        ("sf22", "debug_mlo_sf22"),
        ("eadd3", "debug_mlo_eadd3"),
        ("sf27", "debug_mlo_sf27"),
        ("sf30", "debug_mlo_sf30"),
        ("sf33", "debug_mlo_sf33"),
        ("sf36", "debug_mlo_sf36"),
        ("outer1", "debug_mlo_outer1"),
        ("sf38", "debug_mlo_sf38"),
        ("sf20", "debug_mlo_sf20"),
        ("emul2", "debug_mlo_emul2"),
        ("sf23", "debug_mlo_sf23"),
        ("eadd2", "debug_mlo_eadd2"),
        ("sf26", "debug_mlo_sf26"),
        ("sf29", "debug_mlo_sf29"),
        ("sf32", "debug_mlo_sf32"),
        ("sf35", "debug_mlo_sf35"),
        ("outer2", "debug_mlo_outer2"),
        ("sf39", "debug_mlo_sf39"),
        ("ishuf0", "debug_mlo_ishuf0"),
        ("sf40", "debug_mlo_sf40"),
        ("sf21", "debug_mlo_sf21"),
        ("emul3", "debug_mlo_emul3"),
        ("sf24", "debug_mlo_sf24"),
        ("eadd1", "debug_mlo_eadd1"),
        ("sf25", "debug_mlo_sf25"),
        ("sf28", "debug_mlo_sf28"),
        ("sf31", "debug_mlo_sf31"),
        ("sf34", "debug_mlo_sf34"),
        ("outer0", "debug_mlo_outer0"),
        ("sf37", "debug_mlo_sf37"),
    ]

    LOOP_VR_FIELDS = [
        ("sf16", 15, 14),
        ("m0_sel", 13, 12),
        ("mvau0", 11, 10),
        ("sf19", 9, 8),
        ("emul1", 7, 6),
        ("eadd1", 5, 4),
        ("sf31", 3, 2),
        ("outer0", 1, 0),
    ]

    LC_BITS = [
        ("idx_if_in_v", 0),
        ("idx_if_in_r", 1),
        ("axis_if_in_v", 2),
        ("axis_if_in_r", 3),
        ("idx_if_out_v", 4),
        ("idx_if_out_r", 5),
        ("axis_if_out_v", 6),
        ("axis_if_out_r", 7),
        ("core_in_v", 8),
        ("core_in_r", 9),
        ("core_out_v", 10),
        ("core_out_r", 11),
        ("if_idx_v", 12),
        ("if_idx_r", 13),
        ("next_idx_v", 14),
        ("next_idx_r", 15),
        ("s0_in_v", 16),
        ("s0_in_r", 17),
        ("s0_out_v", 18),
        ("s0_out_r", 19),
        ("if_data_v", 20),
        ("if_data_r", 21),
        ("dma_wr_v", 22),
        ("dma_wr_r", 23),
        ("done_wr_in", 24),
        ("done_wr_out", 25),
        ("dma_rd_v", 26),
        ("dma_rd_r", 27),
        ("wr_desc_v", 28),
        ("wr_desc_r", 29),
        ("wr_data_v", 30),
        ("wr_data_r", 31),
        ("wready_reg", 32),
        ("wready_early", 33),
        ("ishuf3_v", 38),
        ("ishuf3_r", 39),
        ("sf17_v", 40),
        ("sf17_r", 41),
        ("addcls_v", 42),
        ("addcls_r", 43),
        ("sf18_v", 44),
        ("sf18_r", 45),
        ("pos_v", 46),
        ("pos_r", 47),
        ("eadd1_v", 48),
        ("eadd1_r", 49),
        ("sf19_v", 50),
        ("sf19_r", 51),
        ("addcls_s0", 52),
        ("addcls_s1", 53),
        ("addcls_in_v", 54),
        ("addcls_in_r", 55),
        ("cig_v", 56),
        ("cig_r", 57),
        ("mvau1_v", 58),
        ("mvau1_r", 59),
        ("top_in_v", 60),
        ("top_in_r", 61),
        ("loopout_v", 62),
        ("loopout_r", 63),
    ]

    LC_CHANNELS = [
        ("idx_if_in", 0, 1),
        ("axis_if_in", 2, 3),
        ("idx_if_out", 4, 5),
        ("axis_if_out", 6, 7),
        ("core_in", 8, 9),
        ("core_out", 10, 11),
        ("if_idx", 12, 13),
        ("next_idx", 14, 15),
        ("s0_in", 16, 17),
        ("s0_out", 18, 19),
        ("if_data", 20, 21),
        ("dma_wr", 22, 23),
        ("dma_rd", 26, 27),
        ("wr_desc", 28, 29),
        ("wr_data", 30, 31),
        ("pre_ishuf3", 38, 39),
        ("pre_sf17", 40, 41),
        ("pre_addcls", 42, 43),
        ("pre_sf18", 44, 45),
        ("pre_pos", 46, 47),
        ("pre_eadd1", 48, 49),
        ("pre_sf19", 50, 51),
        ("pre_addcls_in", 54, 55),
        ("pre_cig", 56, 57),
        ("pre_mvau1", 58, 59),
        ("top_in_dbg", 60, 61),
        ("loopout_dbg", 62, 63),
    ]

    PRELOOP2_CHANNELS = [
        ("cig", 0, 1),
        ("sf9", 2, 3),
        ("dwc0", 4, 5),
        ("sf10", 6, 7),
        ("mvau0_w", 8, 9),
        ("mvau0", 10, 11),
        ("sf11", 12, 13),
        ("dwc1", 14, 15),
        ("sf12", 16, 17),
        ("ishuf2", 18, 19),
        ("sf13", 20, 21),
        ("oshuf2", 22, 23),
        ("sf14", 24, 25),
        ("emul0_w", 26, 27),
        ("emul0", 28, 29),
        ("sf15", 30, 31),
        ("eadd0_w", 32, 33),
        ("eadd0", 34, 35),
        ("sf16", 36, 37),
        ("ishuf3", 38, 39),
        ("sf17", 40, 41),
        ("addcls", 42, 43),
        ("pos", 44, 45),
    ]

    def __init__(self, sim, interval: int):
        self.interval = interval
        self.done_if = sim.top.getPort("done_if")
        self.s_valid = sim.get_bus_port("s_axis_0", "tvalid")
        self.s_ready = sim.get_bus_port("s_axis_0", "tready")
        self.m_valid = sim.get_bus_port("m_axis_0", "tvalid")
        self.m_ready = sim.get_bus_port("m_axis_0", "tready")
        self.sfifo14_valid = sim.top.getPort("debug_sfifo14_out_tvalid")
        self.sfifo14_ready = sim.top.getPort("debug_sfifo14_out_tready")
        self.emul0_valid = sim.top.getPort("debug_emul0_out_tvalid")
        self.emul0_ready = sim.top.getPort("debug_emul0_out_tready")
        self.sfifo15_valid = sim.top.getPort("debug_sfifo15_out_tvalid")
        self.sfifo15_ready = sim.top.getPort("debug_sfifo15_out_tready")
        self.eadd0_valid = sim.top.getPort("debug_eadd0_out_tvalid")
        self.eadd0_ready = sim.top.getPort("debug_eadd0_out_tready")
        self.sfifo16_valid = sim.top.getPort("debug_sfifo16_out_tvalid")
        self.sfifo16_ready = sim.top.getPort("debug_sfifo16_out_tready")
        self.ishuf3_valid = sim.top.getPort("debug_ishuf3_out_tvalid")
        self.ishuf3_ready = sim.top.getPort("debug_ishuf3_out_tready")
        self.sfifo17_valid = sim.top.getPort("debug_sfifo17_out_tvalid")
        self.sfifo17_ready = sim.top.getPort("debug_sfifo17_out_tready")
        self.sfifo18_valid = sim.top.getPort("debug_sfifo18_out_tvalid")
        self.sfifo18_ready = sim.top.getPort("debug_sfifo18_out_tready")
        self.eladd1_valid = sim.top.getPort("debug_eladd1_out_tvalid")
        self.eladd1_ready = sim.top.getPort("debug_eladd1_out_tready")
        self.sfifo19_valid = sim.top.getPort("debug_sfifo19_out_tvalid")
        self.sfifo19_ready = sim.top.getPort("debug_sfifo19_out_tready")
        self.loopout_valid = sim.top.getPort("debug_loop_out_tvalid")
        self.loopout_ready = sim.top.getPort("debug_loop_out_tready")
        self.loop_debug_ports = {
            label: (
                sim.top.getPort(f"{port_base}_tvalid"),
                sim.top.getPort(f"{port_base}_tready"),
            )
            for label, port_base in self.LOOP_DEBUG_CHANNELS
        }
        self.debug_mlo_vr = sim.top.getPort("debug_mlo_vr")
        self.debug_loop_vr = sim.top.getPort("debug_loop_vr")
        self.debug_preloop_valid = sim.top.getPort("debug_preloop_valid")
        self.debug_preloop_ready = sim.top.getPort("debug_preloop_ready")
        self.debug_tap_valid = sim.top.getPort("debug_tap_valid")
        self.debug_tap_ready = sim.top.getPort("debug_tap_ready")
        self.debug_lc_vr = sim.top.getPort("debug_lc_vr")
        self.debug_preloop2_vr = sim.top.getPort("debug_preloop2_vr")
        self.aw_valid = sim.get_bus_port("m_axi_hbm", "awvalid")
        self.aw_ready = sim.get_bus_port("m_axi_hbm", "awready")
        self.aw_addr = sim.get_bus_port("m_axi_hbm", "awaddr")
        self.w_valid = sim.get_bus_port("m_axi_hbm", "wvalid")
        self.w_ready = sim.get_bus_port("m_axi_hbm", "wready")
        self.b_valid = sim.get_bus_port("m_axi_hbm", "bvalid")
        self.b_ready = sim.get_bus_port("m_axi_hbm", "bready")
        self.ar_valid = sim.get_bus_port("m_axi_hbm", "arvalid")
        self.ar_ready = sim.get_bus_port("m_axi_hbm", "arready")
        self.ar_addr = sim.get_bus_port("m_axi_hbm", "araddr")
        self.r_valid = sim.get_bus_port("m_axi_hbm", "rvalid")
        self.r_ready = sim.get_bus_port("m_axi_hbm", "rready")
        self.s_axis = ChannelStats()
        self.m_axis = ChannelStats()
        self.sfifo14 = ChannelStats()
        self.emul0 = ChannelStats()
        self.sfifo15 = ChannelStats()
        self.eadd0 = ChannelStats()
        self.sfifo16 = ChannelStats()
        self.ishuf3 = ChannelStats()
        self.sfifo17 = ChannelStats()
        self.sfifo18 = ChannelStats()
        self.eladd1 = ChannelStats()
        self.sfifo19 = ChannelStats()
        self.loopout = ChannelStats()
        self.loop_debug_stats = {
            label: ChannelStats() for label, _ in self.LOOP_DEBUG_CHANNELS
        }
        self.preloop_stats = {label: ChannelStats() for label in self.PRELOOP_CHANNELS}
        self.tap_stats = {i: ChannelStats() for i in range(34)}
        self.loop_vr_stats = {
            label: ChannelStats() for label, _, _ in self.LOOP_VR_FIELDS
        }
        self.lc_stats = {label: ChannelStats() for label, _, _ in self.LC_CHANNELS}
        self.preloop2_stats = {
            label: ChannelStats() for label, _, _ in self.PRELOOP2_CHANNELS
        }
        self.aw = ChannelStats()
        self.w = ChannelStats()
        self.b = ChannelStats()
        self.ar = ChannelStats()
        self.r = ChannelStats()
        self.aw_addrs: set[int] = set()
        self.ar_addrs: set[int] = set()
        self.last_tick = 0
        self.aximm_queue = None
        for task in sim.tasks:
            if task.__class__.__name__ == "AximmQueue":
                self.aximm_queue = task

    def __bool__(self):
        return False

    def _summary(self, tick: int) -> str:
        queue_summary = ""
        if self.aximm_queue is not None:
            queue_summary = (
                " map=%d waq=%d wdq=%d raq=%d bq=%d"
                % (
                    len(getattr(self.aximm_queue, "map", {})),
                    len(getattr(self.aximm_queue, "wa_queue", [])),
                    len(getattr(self.aximm_queue, "wd_queue", [])),
                    len(getattr(self.aximm_queue, "ra_queue", [])),
                    len(getattr(self.aximm_queue, "wr_completion_queue", [])),
            )
        )
        loop_debug = " ".join(
            "%s=%d/%d/%d%d@%d"
            % (
                label,
                stats.handshakes,
                stats.seen,
                int(stats.last_valid),
                int(stats.last_ready),
                stats.last_handshake,
            )
            for label, stats in self.loop_debug_stats.items()
        )
        preloop_debug = " ".join(
            "%s=%d/%d/%d%d@%d"
            % (
                label,
                stats.handshakes,
                stats.seen,
                int(stats.last_valid),
                int(stats.last_ready),
                stats.last_handshake,
            )
            for label, stats in self.preloop_stats.items()
        )
        debug_mlo_vr = _as_uint(self.debug_mlo_vr)
        debug_loop_vr = _as_uint(self.debug_loop_vr)
        debug_preloop_valid = _as_uint(self.debug_preloop_valid)
        debug_preloop_ready = _as_uint(self.debug_preloop_ready)
        debug_tap_valid = _as_uint(self.debug_tap_valid)
        debug_tap_ready = _as_uint(self.debug_tap_ready)
        debug_lc_vr = _as_uint(self.debug_lc_vr)
        debug_preloop2_vr = _as_uint(self.debug_preloop2_vr)
        debug_mlo_vr = 0 if debug_mlo_vr is None else debug_mlo_vr
        debug_loop_vr = 0 if debug_loop_vr is None else debug_loop_vr
        debug_preloop_valid = 0 if debug_preloop_valid is None else debug_preloop_valid
        debug_preloop_ready = 0 if debug_preloop_ready is None else debug_preloop_ready
        debug_tap_valid = 0 if debug_tap_valid is None else debug_tap_valid
        debug_tap_ready = 0 if debug_tap_ready is None else debug_tap_ready
        debug_lc_vr = 0 if debug_lc_vr is None else debug_lc_vr
        debug_preloop2_vr = 0 if debug_preloop2_vr is None else debug_preloop2_vr
        blocked_preloop = [
            label
            for i, label in enumerate(self.PRELOOP_CHANNELS)
            if ((debug_preloop_valid >> i) & 1) and not ((debug_preloop_ready >> i) & 1)
        ]
        blocked_preloop_text = ",".join(blocked_preloop) if blocked_preloop else "-"
        blocked_taps = [
            str(i)
            for i in range(34)
            if ((debug_tap_valid >> i) & 1) and not ((debug_tap_ready >> i) & 1)
        ]
        blocked_tap_text = ",".join(blocked_taps) if blocked_taps else "-"
        tap_seen = [
            "%d=%d/%d/%d%d@%d"
            % (
                i,
                stats.handshakes,
                stats.seen,
                int(stats.last_valid),
                int(stats.last_ready),
                stats.last_handshake,
            )
            for i, stats in self.tap_stats.items()
            if stats.handshakes or stats.seen or stats.last_valid or stats.last_ready
        ]
        tap_seen_text = ",".join(tap_seen) if tap_seen else "-"
        loop_vr_text = " ".join(
            "%s=%d%d"
            % (
                label,
                (debug_loop_vr >> valid_bit) & 1,
                (debug_loop_vr >> ready_bit) & 1,
            )
            for label, valid_bit, ready_bit in self.LOOP_VR_FIELDS
        )
        loop_vr_seen = " ".join(
            "%s=%d/%d/%d%d@%d"
            % (
                label,
                stats.handshakes,
                stats.seen,
                int(stats.last_valid),
                int(stats.last_ready),
                stats.last_handshake,
            )
            for label, stats in self.loop_vr_stats.items()
        )
        inner_text = (
            "wr=%d jobs=%d page=%d rdg=%d rden=%d rdrdy=%d "
            "rdi=%d rdj=%d pat=%d%d osb=%d rd=%d%d in=%d%d out=%d%d "
            "pb=%d rinc=%d wen=%d"
            % (
                (debug_mlo_vr >> 78) & 0x7FFF,
                (debug_mlo_vr >> 93) & 0x3,
                (debug_mlo_vr >> 95) & 0x1,
                (debug_mlo_vr >> 96) & 0x1,
                (debug_mlo_vr >> 97) & 0x1,
                (debug_mlo_vr >> 98) & 0x1,
                (debug_mlo_vr >> 99) & 0xFF,
                (debug_mlo_vr >> 107) & 0x3F,
                (debug_mlo_vr >> 114) & 0x1,
                (debug_mlo_vr >> 113) & 0x1,
                (debug_mlo_vr >> 115) & 0x1,
                (debug_mlo_vr >> 116) & 0x1,
                (debug_mlo_vr >> 117) & 0x1,
                (debug_mlo_vr >> 119) & 0x1,
                (debug_mlo_vr >> 118) & 0x1,
                (debug_mlo_vr >> 120) & 0x1,
                (debug_mlo_vr >> 121) & 0x1,
                (debug_mlo_vr >> 122) & 0x1,
                (debug_mlo_vr >> 123) & 0x1,
                (debug_mlo_vr >> 124) & 0x1,
            )
        )
        outer2_flags = (debug_tap_valid >> 51) & 0x1FFF
        outer2_text = (
            "wp=%d rp=%d fp=%d ovld=%d src=%d%d dst=%d%d in=%d%d out=%d%d "
            "fsm=%d%d%d canrd=%d fullblk=%d"
            % (
                (debug_tap_valid >> 34) & 0x1FFFF,
                (debug_tap_ready >> 34) & 0x1FFFF,
                (debug_preloop2_vr >> 46) & 0x1FFFF,
                (debug_preloop2_vr >> 63) & 0x1,
                (outer2_flags >> 0) & 0x1,
                (outer2_flags >> 1) & 0x1,
                (outer2_flags >> 2) & 0x1,
                (outer2_flags >> 3) & 0x1,
                (outer2_flags >> 4) & 0x1,
                (outer2_flags >> 5) & 0x1,
                (outer2_flags >> 6) & 0x1,
                (outer2_flags >> 7) & 0x1,
                (outer2_flags >> 8) & 0x1,
                (outer2_flags >> 9) & 0x1,
                (outer2_flags >> 10) & 0x1,
                (outer2_flags >> 11) & 0x1,
                (outer2_flags >> 12) & 0x1,
            )
        )
        lc_bits = " ".join(
            "%s=%d" % (label, (debug_lc_vr >> bit) & 1)
            for label, bit in self.LC_BITS
        )
        lc_seen = " ".join(
            "%s=%d/%d/%d%d@%d"
            % (
                label,
                stats.handshakes,
                stats.seen,
                int(stats.last_valid),
                int(stats.last_ready),
                stats.last_handshake,
            )
            for label, stats in self.lc_stats.items()
        )
        preloop2_seen = " ".join(
            "%s=%d/%d/%d%d@%d"
            % (
                label,
                stats.handshakes,
                stats.seen,
                int(stats.last_valid),
                int(stats.last_ready),
                stats.last_handshake,
            )
            for label, stats in self.preloop2_stats.items()
        )
        lc_states = "if_wr=%d if_rd=%d dma_wr=%d addcls_state=%d" % (
            (debug_lc_vr >> 36) & 1,
            (debug_lc_vr >> 37) & 1,
            (debug_lc_vr >> 34) & 3,
            (debug_lc_vr >> 52) & 3,
        )
        return (
            "[monitor] tick=%d "
            "done=%s s_hs=%d s_last=%d "
            "sf14_hs=%d sf14_last=%d emul0_hs=%d emul0_last=%d "
            "sf15_hs=%d sf15_last=%d eadd0_hs=%d eadd0_last=%d "
            "sf16_hs=%d sf16_last=%d ishuf3_hs=%d ishuf3_last=%d "
            "sf17_hs=%d sf17_last=%d "
            "sf18_hs=%d sf18_last=%d eadd1_hs=%d eadd1_last=%d "
            "sf19_hs=%d sf19_last=%d loopout_hs=%d loopout_last=%d "
            "m_hs=%d m_last=%d "
            "aw=%d w=%d b=%d ar=%d r=%d "
            "aw_unique=%d ar_unique=%d "
            "seen(sf14/emul0/sf15/eadd0/sf16/ishuf3/sf17/sf18/eadd1/sf19)="
            "%d/%d/%d/%d/%d/%d/%d/%d/%d/%d "
            "vr(sf14/emul0/sf15/eadd0/sf16/ishuf3/sf17/sf18/eadd1/sf19)="
            "%d%d/%d%d/%d%d/%d%d/%d%d/%d%d/%d%d/%d%d/%d%d/%d%d "
            "dbg_loop=0x%04x loop_vr=%s dbg_tap_v=0x%09x dbg_tap_r=0x%09x "
            "dbg_lc=0x%016x lc_state{%s} lc_bits{%s} "
            "raw_loop=%s raw_mlo=%s inner{%s} outer2{%s} "
            "raw_tap_v=%s raw_tap_r=%s raw_lc=%s raw_pre2=%s "
            "blocked_taps=%s tap_stats=%s "
            "dbg_pre_v=0x%08x dbg_pre_r=0x%08x blocked_pre=%s "
            "preloop{%s} preloop2{%s} lc_stats{%s} loop_vr_stats{%s} loop_dbg{%s}%s"
            % (
                tick,
                str(_as_uint(self.done_if)),
                self.s_axis.handshakes,
                self.s_axis.last_handshake,
                self.sfifo14.handshakes,
                self.sfifo14.last_handshake,
                self.emul0.handshakes,
                self.emul0.last_handshake,
                self.sfifo15.handshakes,
                self.sfifo15.last_handshake,
                self.eadd0.handshakes,
                self.eadd0.last_handshake,
                self.sfifo16.handshakes,
                self.sfifo16.last_handshake,
                self.ishuf3.handshakes,
                self.ishuf3.last_handshake,
                self.sfifo17.handshakes,
                self.sfifo17.last_handshake,
                self.sfifo18.handshakes,
                self.sfifo18.last_handshake,
                self.eladd1.handshakes,
                self.eladd1.last_handshake,
                self.sfifo19.handshakes,
                self.sfifo19.last_handshake,
                self.loopout.handshakes,
                self.loopout.last_handshake,
                self.m_axis.handshakes,
                self.m_axis.last_handshake,
                self.aw.handshakes,
                self.w.handshakes,
                self.b.handshakes,
                self.ar.handshakes,
                self.r.handshakes,
                len(self.aw_addrs),
                len(self.ar_addrs),
                self.sfifo14.seen,
                self.emul0.seen,
                self.sfifo15.seen,
                self.eadd0.seen,
                self.sfifo16.seen,
                self.ishuf3.seen,
                self.sfifo17.seen,
                self.sfifo18.seen,
                self.eladd1.seen,
                self.sfifo19.seen,
                int(self.sfifo14.last_valid),
                int(self.sfifo14.last_ready),
                int(self.emul0.last_valid),
                int(self.emul0.last_ready),
                int(self.sfifo15.last_valid),
                int(self.sfifo15.last_ready),
                int(self.eadd0.last_valid),
                int(self.eadd0.last_ready),
                int(self.sfifo16.last_valid),
                int(self.sfifo16.last_ready),
                int(self.ishuf3.last_valid),
                int(self.ishuf3.last_ready),
                int(self.sfifo17.last_valid),
                int(self.sfifo17.last_ready),
                int(self.sfifo18.last_valid),
                int(self.sfifo18.last_ready),
                int(self.eladd1.last_valid),
                int(self.eladd1.last_ready),
                int(self.sfifo19.last_valid),
                int(self.sfifo19.last_ready),
                debug_loop_vr,
                loop_vr_text,
                debug_tap_valid,
                debug_tap_ready,
                debug_lc_vr,
                lc_states,
                lc_bits,
                _as_hex(self.debug_loop_vr),
                _as_hex(self.debug_mlo_vr),
                inner_text,
                outer2_text,
                _as_hex(self.debug_tap_valid),
                _as_hex(self.debug_tap_ready),
                _as_hex(self.debug_lc_vr),
                _as_hex(self.debug_preloop2_vr),
                blocked_tap_text,
                tap_seen_text,
                debug_preloop_valid,
                debug_preloop_ready,
            blocked_preloop_text,
            preloop_debug,
            preloop2_seen,
            lc_seen,
            loop_vr_seen,
            loop_debug,
                queue_summary,
            )
        )

    def __call__(self, sim):
        tick = sim.ticks
        self.last_tick = tick
        self.s_axis.update(tick, _as_bool(self.s_valid), _as_bool(self.s_ready))
        self.m_axis.update(tick, _as_bool(self.m_valid), _as_bool(self.m_ready))
        self.sfifo14.update(tick, _as_bool(self.sfifo14_valid), _as_bool(self.sfifo14_ready))
        self.emul0.update(tick, _as_bool(self.emul0_valid), _as_bool(self.emul0_ready))
        self.sfifo15.update(tick, _as_bool(self.sfifo15_valid), _as_bool(self.sfifo15_ready))
        self.eadd0.update(tick, _as_bool(self.eadd0_valid), _as_bool(self.eadd0_ready))
        self.sfifo16.update(tick, _as_bool(self.sfifo16_valid), _as_bool(self.sfifo16_ready))
        self.ishuf3.update(tick, _as_bool(self.ishuf3_valid), _as_bool(self.ishuf3_ready))
        self.sfifo17.update(tick, _as_bool(self.sfifo17_valid), _as_bool(self.sfifo17_ready))
        self.sfifo18.update(tick, _as_bool(self.sfifo18_valid), _as_bool(self.sfifo18_ready))
        self.eladd1.update(tick, _as_bool(self.eladd1_valid), _as_bool(self.eladd1_ready))
        self.sfifo19.update(tick, _as_bool(self.sfifo19_valid), _as_bool(self.sfifo19_ready))
        self.loopout.update(tick, _as_bool(self.loopout_valid), _as_bool(self.loopout_ready))
        debug_mlo_vr = _as_uint(self.debug_mlo_vr)
        debug_mlo_vr = 0 if debug_mlo_vr is None else debug_mlo_vr
        for i, (label, _port_base) in enumerate(self.LOOP_DEBUG_CHANNELS):
            self.loop_debug_stats[label].update(
                tick,
                bool((debug_mlo_vr >> (2 * i)) & 1),
                bool((debug_mlo_vr >> ((2 * i) + 1)) & 1),
            )
        debug_preloop_valid = _as_uint(self.debug_preloop_valid)
        debug_preloop_ready = _as_uint(self.debug_preloop_ready)
        debug_preloop_valid = 0 if debug_preloop_valid is None else debug_preloop_valid
        debug_preloop_ready = 0 if debug_preloop_ready is None else debug_preloop_ready
        for i, label in enumerate(self.PRELOOP_CHANNELS):
            self.preloop_stats[label].update(
                tick,
                bool((debug_preloop_valid >> i) & 1),
                bool((debug_preloop_ready >> i) & 1),
            )
        debug_tap_valid = _as_uint(self.debug_tap_valid)
        debug_tap_ready = _as_uint(self.debug_tap_ready)
        debug_tap_valid = 0 if debug_tap_valid is None else debug_tap_valid
        debug_tap_ready = 0 if debug_tap_ready is None else debug_tap_ready
        for i in range(34):
            self.tap_stats[i].update(
                tick,
                bool((debug_tap_valid >> i) & 1),
                bool((debug_tap_ready >> i) & 1),
            )
        debug_loop_vr = _as_uint(self.debug_loop_vr)
        debug_loop_vr = 0 if debug_loop_vr is None else debug_loop_vr
        for label, valid_bit, ready_bit in self.LOOP_VR_FIELDS:
            self.loop_vr_stats[label].update(
                tick,
                bool((debug_loop_vr >> valid_bit) & 1),
                bool((debug_loop_vr >> ready_bit) & 1),
            )
        debug_lc_vr = _as_uint(self.debug_lc_vr)
        debug_lc_vr = 0 if debug_lc_vr is None else debug_lc_vr
        for label, valid_bit, ready_bit in self.LC_CHANNELS:
            self.lc_stats[label].update(
                tick,
                bool((debug_lc_vr >> valid_bit) & 1),
                bool((debug_lc_vr >> ready_bit) & 1),
            )
        debug_preloop2_vr = _as_uint(self.debug_preloop2_vr)
        debug_preloop2_vr = 0 if debug_preloop2_vr is None else debug_preloop2_vr
        for label, valid_bit, ready_bit in self.PRELOOP2_CHANNELS:
            self.preloop2_stats[label].update(
                tick,
                bool((debug_preloop2_vr >> valid_bit) & 1),
                bool((debug_preloop2_vr >> ready_bit) & 1),
            )
        self.aw.update(tick, _as_bool(self.aw_valid), _as_bool(self.aw_ready))
        self.w.update(tick, _as_bool(self.w_valid), _as_bool(self.w_ready))
        self.b.update(tick, _as_bool(self.b_valid), _as_bool(self.b_ready))
        self.ar.update(tick, _as_bool(self.ar_valid), _as_bool(self.ar_ready))
        self.r.update(tick, _as_bool(self.r_valid), _as_bool(self.r_ready))
        if _as_bool(self.aw_valid) and _as_bool(self.aw_ready):
            addr = _as_uint(self.aw_addr)
            if addr is not None:
                self.aw_addrs.add(addr)
        if _as_bool(self.ar_valid) and _as_bool(self.ar_ready):
            addr = _as_uint(self.ar_addr)
            if addr is not None:
                self.ar_addrs.add(addr)
        if self.interval and tick % self.interval == 0:
            print(self._summary(tick), flush=True)
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        default=(
            "tinydeit/build/v80_300mhz_ipgen_after_mlo_selector_pad1"
        ),
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--input",
        default=(
            "tinydeit/build/v80_300mhz_rtlverify_90k_79ea961a_"
            "split_dsp_chains_liveness350k_pad1/input.npy"
        ),
    )
    parser.add_argument(
        "--expected-output",
        default=(
            "tinydeit/build/v80_300mhz_rtlverify_90k_79ea961a_"
            "split_dsp_chains_liveness350k_pad1/expected_output.npy"
        ),
    )
    parser.add_argument("--liveness", type=int, default=300000)
    parser.add_argument("--interval", type=int, default=50000)
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Run with the MLO prehook only, without per-cycle debug monitoring.",
    )
    parser.add_argument("--force-recompile", action="store_true")
    parser.add_argument("--rtlsim-so", default=None)
    parser.add_argument("--trace", default=None)
    args = parser.parse_args()

    os.environ["LIVENESS_THRESHOLD"] = str(args.liveness)
    model_path = args.model
    if model_path is None:
        model_path = str(
            Path(args.build_dir) / "intermediate_models" / "step_create_stitched_ip.onnx"
        )
    model = ModelWrapper(model_path)
    model.set_metadata_prop("exec_mode", "rtlsim")
    if args.force_recompile:
        model.set_metadata_prop("rtlsim_so", "")
    if args.rtlsim_so is not None:
        model.set_metadata_prop("rtlsim_so", args.rtlsim_so)
    if args.trace is not None:
        model.set_metadata_prop("rtlsim_trace", args.trace)

    input_tensor = np.load(args.input)
    expected_output = np.load(args.expected_output)
    input_name = model.get_first_global_in()
    output_name = model.get_first_global_out()
    ctx = {input_name: input_tensor}

    loop_nodes = model.get_nodes_by_op_type("FINNLoop")
    if len(loop_nodes) != 1:
        raise RuntimeError(f"Expected one FINNLoop, found {len(loop_nodes)}")
    base_prehook = mlo_prehook_func_factory(loop_nodes[0])
    monitor_holder = {}

    def prehook(sim):
        base_prehook(sim)
        if args.no_monitor:
            return
        monitor = TopMonitor(sim, args.interval)
        sim.enlist(monitor)
        monitor_holder["monitor"] = monitor

    print(f"model={model_path}")
    print(f"input={args.input} shape={input_tensor.shape}")
    print(f"expected={args.expected_output} shape={expected_output.shape}")
    print(f"LIVENESS_THRESHOLD={args.liveness}")
    print(f"wrapper={model.get_metadata_prop('wrapper_filename')}")
    print(f"vivado_stitch_proj={model.get_metadata_prop('vivado_stitch_proj')}")
    print(f"rtlsim_so={model.get_metadata_prop('rtlsim_so')}")
    sim_failed = False
    try:
        rtlsim_exec(model, ctx, pre_hook=prehook)
    except AssertionError as exc:
        sim_failed = True
        print(f"rtlsim_exec failed: {exc}", flush=True)
    finally:
        monitor = monitor_holder.get("monitor")
        if monitor is not None:
            print(monitor._summary(monitor.last_tick), flush=True)
    if sim_failed:
        return
    produced = ctx[output_name]
    print(f"produced shape={produced.shape}")
    print(f"allclose={np.isclose(expected_output, produced, atol=1e-1).all()}")


if __name__ == "__main__":
    main()
