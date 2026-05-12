#!/usr/bin/env python3
"""Summarize whether BERT safety build directories produced verified DCPs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bert.common import DEFAULT_BUILD_DIR, repo_path


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open() as f:
        return json.load(f)


def summarize_build(path: Path) -> dict[str, Any]:
    timing = load_json(path / "report" / "ooc_synth_and_timing.json")
    dcp = path / "stitched_ip" / "finn_design.dcp"
    success_files = sorted(path.glob("verification_output/verify_stitched_ip_rtlsim_*_SUCCESS.npy"))
    failure_files = sorted(path.glob("verification_output/verify_stitched_ip_rtlsim_*_FAIL.npy"))
    preset = load_json(path / "preset.json") or {}
    request = load_json(path / "build_request.json") or {}
    has_reference = (path / "input.npy").is_file() and (path / "expected_output.npy").is_file()
    wns = timing.get("WNS") if timing else None
    verified_dcp = bool(
        success_files and dcp.is_file() and timing and wns is not None and wns >= 0.0
    )
    return {
        "build_dir": str(path),
        "preset": preset.get("name", request.get("preset")),
        "mode": request.get("mode"),
        "verified_dcp": verified_dcp,
        "rtlsim_success": bool(success_files),
        "rtlsim_failure": bool(failure_files),
        "has_reference_io": has_reference,
        "dcp": str(dcp) if dcp.is_file() else None,
        "dcp_bytes": dcp.stat().st_size if dcp.is_file() else 0,
        "ooc_timing": str(path / "report" / "ooc_synth_and_timing.json") if timing else None,
        "wns_ns": wns,
        "fmax_mhz": timing.get("fmax_mhz") if timing else None,
        "lut": timing.get("LUT") if timing else None,
        "ff": timing.get("FF") if timing else None,
        "dsp": timing.get("DSP") if timing else None,
        "bram": timing.get("BRAM") if timing else None,
        "uram": timing.get("URAM") if timing else None,
    }


def discover_build_dirs(root: Path) -> list[Path]:
    return sorted(
        path for path in root.iterdir() if path.is_dir() and (path / "preset.json").is_file()
    )


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = ["build", "preset", "verified", "rtlsim", "dcp", "fmax_mhz", "wns_ns"]
    print(" | ".join(headers))
    print(" | ".join("---" for _ in headers))
    for row in rows:
        print(
            " | ".join(
                [
                    Path(row["build_dir"]).name,
                    str(row["preset"]),
                    "yes" if row["verified_dcp"] else "no",
                    "yes" if row["rtlsim_success"] else "no",
                    "yes" if row["dcp"] else "no",
                    "" if row["fmax_mhz"] is None else f"{row['fmax_mhz']:.2f}",
                    "" if row["wns_ns"] is None else f"{row['wns_ns']:.3f}",
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dirs", nargs="*", help="Build directories to inspect.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Inspect every build under bert/build that has a preset.json file.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    if args.all:
        build_dirs = discover_build_dirs(DEFAULT_BUILD_DIR)
    elif args.build_dirs:
        build_dirs = [repo_path(path) for path in args.build_dirs]
    else:
        build_dirs = [
            DEFAULT_BUILD_DIR / "v80_smoke_mlo_dcp_v1",
            DEFAULT_BUILD_DIR / "v80_mlo_dcp_v1",
            DEFAULT_BUILD_DIR / "max_util_p16s32_dcp_v1",
            DEFAULT_BUILD_DIR / "max_util_p16s32_fixedfifo_dcp_v1",
        ]

    rows = [summarize_build(path) for path in build_dirs]
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print_table(rows)


if __name__ == "__main__":
    main()
