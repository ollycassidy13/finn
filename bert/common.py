"""Shared paths and presets for the BERT safety FINN flow."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_DIR = REPO_ROOT / "bert" / "build"
DEFAULT_CACHE_DIR = REPO_ROOT / "bert" / "cache"
DEFAULT_BOARD = "V80"
DEFAULT_FPGA_PART = "xcv80-lsva4737-2MHP-e-s"
DEFAULT_CLOCK_NS = 4.0


@dataclass(frozen=True)
class CorePreset:
    name: str
    seq_len: int
    hidden: int
    intermediate: int
    layers: int
    num_classes: int
    pe: int
    simd: int
    target_fps: int
    mvau_wwidth_max: int
    ram_style: str = "auto"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


CORE_PRESETS = {
    # Fast bring-up target. This is the default for repeated local verification.
    "smoke": CorePreset(
        name="smoke",
        seq_len=4,
        hidden=8,
        intermediate=16,
        layers=2,
        num_classes=2,
        pe=1,
        simd=1,
        target_fps=1000,
        mvau_wwidth_max=36,
    ),
    # Small V80-oriented target for stitched-IP/DCP validation without full BERT compile time.
    "v80": CorePreset(
        name="v80",
        seq_len=16,
        hidden=64,
        intermediate=128,
        layers=4,
        num_classes=2,
        pe=4,
        simd=4,
        target_fps=5000,
        mvau_wwidth_max=512,
        ram_style="block",
    ),
    # BERT-base dimensions. This is the paper-size target for the trained checkpoint.
    "paper": CorePreset(
        name="paper",
        seq_len=128,
        hidden=768,
        intermediate=3072,
        layers=12,
        num_classes=2,
        pe=16,
        simd=16,
        target_fps=1000,
        mvau_wwidth_max=1000000,
        ram_style="ultra",
    ),
    # Same geometry as paper, but asks FINN to unfold more aggressively.
    "max-util": CorePreset(
        name="max-util",
        seq_len=128,
        hidden=768,
        intermediate=3072,
        layers=12,
        num_classes=2,
        pe=16,
        simd=32,
        target_fps=2000,
        mvau_wwidth_max=10000000,
        ram_style="ultra",
    ),
}


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def get_preset(name: str) -> CorePreset:
    try:
        return CORE_PRESETS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(CORE_PRESETS))
        raise ValueError(
            f"Unknown BERT safety core preset {name!r}; valid presets: {valid}"
        ) from exc


def derive_preset(base: CorePreset, name: str | None = None, **overrides: Any) -> CorePreset:
    """Return a preset with selected fields overridden for build sweeps."""

    clean_overrides = {key: value for key, value in overrides.items() if value is not None}
    if name is not None:
        clean_overrides["name"] = name
    return replace(base, **clean_overrides)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
