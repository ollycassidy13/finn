#!/usr/bin/env python3
"""Guardrails for the exact SO400M W6A8 DCP objective."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SO400M_W6A8_OBJECTIVE_TOKEN = (
    "static_qat_so400m_patch14_384_w6a8_qvlsq_step200_featurehead32k_freshbias_lr3e4_ep60_cal0_export"
)
DEFAULT_ESTIMATE_CANDIDATES_JSON = Path("full_siglip/build/so400m_estimate_candidates.json")
DEFAULT_LOWER_BOUND_JSON = Path("full_siglip/build/so400m_lower_bound.json")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def is_so400m_w6a8_objective_path(*paths: Path) -> bool:
    return any(SO400M_W6A8_OBJECTIVE_TOKEN in str(path) for path in paths)


def load_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else repo_root() / path


def summarize_resource_overages(candidate: dict[str, Any]) -> list[str]:
    checks = (
        candidate.get("resource_fit_details", {})
        .get("checks", {})
    )
    overages = []
    for key in ("DSP", "LUT", "BRAM_18K", "URAM"):
        detail = checks.get(key, {})
        if detail.get("ok") is not False:
            continue
        used = detail.get("used")
        limit = detail.get("limit")
        factor = detail.get("factor")
        if isinstance(used, (int, float)) and isinstance(limit, (int, float)):
            if isinstance(factor, (int, float)):
                overages.append(f"{key} {used:,.0f}/{limit:,.0f} ({factor:.1f}x)")
            else:
                overages.append(f"{key} {used:,.0f}/{limit:,.0f}")
    return overages


def evaluate_so400m_dcp_preflight(
    *,
    estimate_candidates_json: Path = DEFAULT_ESTIMATE_CANDIDATES_JSON,
) -> dict[str, Any]:
    audit = load_json_if_present(repo_path(estimate_candidates_json))
    if audit is None:
        return {
            "normal_dcp_allowed": False,
            "reason": f"estimate-candidate audit missing at {estimate_candidates_json}",
            "blockers": [
                "no exact SO400M MLO+pumped+embedding resource-fit sub-50 estimate has been audited"
            ],
        }

    checks = audit.get("checks", {})
    candidate = audit.get("best_resource_fit_under_budget_estimate")
    if (
        checks.get("resource_fit_under_budget_candidate_available") is True
        and isinstance(candidate, dict)
    ):
        return {
            "normal_dcp_allowed": True,
            "reason": "resource-fit sub-50 SO400M estimate is available",
            "candidate": {
                "path": candidate.get("path"),
                "top_cycles": candidate.get("top_cycles"),
                "latency_ms": candidate.get("latency_ms"),
            },
            "blockers": [],
        }

    blockers = []
    best = audit.get("best_current_estimate")
    cycle_budget = audit.get("cycle_budget")
    best_over_budget = False
    if isinstance(best, dict):
        top_cycles = best.get("top_cycles")
        latency_ms = best.get("latency_ms")
        if isinstance(top_cycles, int) and isinstance(cycle_budget, int):
            best_over_budget = top_cycles > cycle_budget
            if best_over_budget:
                blockers.append(
                    f"best exact estimate is {top_cycles:,} cycles "
                    f"({top_cycles / cycle_budget:.1f}x over the "
                    f"{cycle_budget:,}-cycle budget)"
                )
        if best_over_budget and isinstance(latency_ms, (int, float)):
            blockers.append(f"best exact estimate latency is {latency_ms:.1f} ms")
        overages = summarize_resource_overages(best)
        if overages:
            blockers.append("best exact estimate exceeds VCK190 resources: " + ", ".join(overages))
    elif checks.get("eligible_candidate_available") is not True:
        blockers.append("no exact MLO+pumped+embedding SO400M estimate candidate is available")

    if checks.get("resource_fit_candidate_available") is not True:
        blockers.append("no exact MLO+pumped+embedding estimate fits VCK190 resources")
    if checks.get("resource_fit_under_budget_candidate_available") is not True:
        blockers.append("no exact MLO+pumped+embedding estimate is both resource-fit and under 50 ms")

    return {
        "normal_dcp_allowed": False,
        "reason": "no resource-fit sub-50 SO400M estimate candidate is available",
        "blockers": blockers,
    }


def lower_bound_detail(lower_bound_json: Path = DEFAULT_LOWER_BOUND_JSON) -> str:
    data = load_json_if_present(repo_path(lower_bound_json))
    if data is None:
        return ""
    bound = data.get("mvau_only_dsp_limited_bound", {})
    conclusion = data.get("conclusion", {})
    if conclusion.get("sub50_possible_under_vck190_dsp_bound") is not False:
        return ""
    cycles = bound.get("cycles")
    latency = bound.get("latency_ms")
    clock = data.get("clock_mhz")
    if (
        isinstance(cycles, int)
        and isinstance(latency, (int, float))
        and isinstance(clock, (int, float))
    ):
        return (
            f" Current lower bound: {cycles:,} MVAU-only cycles, "
            f"{latency:.1f} ms at {clock:g} MHz."
        )
    return ""


def guard_so400m_w6a8_dcp(
    input_model: Path,
    output_dir: Path,
    allow_infeasible: bool,
    *,
    estimate_candidates_json: Path = DEFAULT_ESTIMATE_CANDIDATES_JSON,
    lower_bound_json: Path = DEFAULT_LOWER_BOUND_JSON,
) -> None:
    if allow_infeasible:
        return
    if not is_so400m_w6a8_objective_path(input_model, output_dir):
        return

    preflight = evaluate_so400m_dcp_preflight(
        estimate_candidates_json=estimate_candidates_json,
    )
    if preflight["normal_dcp_allowed"]:
        return

    blocker_text = ""
    blockers = preflight.get("blockers", [])
    if blockers:
        blocker_text = " Blockers: " + "; ".join(str(item) for item in blockers) + "."

    raise SystemExit(
        "Refusing SO400M patch14/384 W6A8 DCP launch: current objective checks do not "
        "have an exact full-depth MLO+pumped embedding-only estimate that is both "
        "sub-50 ms and within VCK190 DSP/LUT/BRAM_18K/URAM limits."
        f"{blocker_text}{lower_bound_detail(lower_bound_json)} "
        "Run `python -m full_siglip.check_so400m_goal_status` and inspect "
        "`full_siglip/build/so400m_goal_status.md` before spending Vivado time. "
        "Use --allow-infeasible-so400m-dcp only for an explicitly requested diagnostic DCP run."
    )
