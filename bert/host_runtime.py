"""Host-side contract for the BERT safety FINN core.

The board-facing implementation is expected to provide a callable that accepts
one quantized activation tensor and returns one logits tensor. Tokenization,
embedding lookup, attention masking, logging, and policy thresholds remain on
the host side.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class SafetyCoreConfig:
    seq_len: int = 128
    hidden: int = 768
    num_classes: int = 2
    unsafe_index: int = 1
    unsafe_threshold: float = 0.5

    @property
    def patch_shape(self) -> tuple[int, int, int]:
        return (1, self.seq_len - 1, self.hidden)

    @property
    def logits_shape(self) -> tuple[int, int]:
        return (1, self.num_classes)


@dataclass(frozen=True)
class SafetyDecision:
    logits: np.ndarray
    probabilities: np.ndarray
    unsafe_score: float
    blocked: bool


SafetyCoreRunner = Callable[[np.ndarray], np.ndarray]


def quantize_embedding_patches(
    patches: np.ndarray, scale: float, zero_point: int = 0, dtype: np.dtype = np.uint8
) -> np.ndarray:
    """Quantize host-generated embedding patches for the FINN core input."""

    if scale <= 0.0:
        raise ValueError("scale must be positive")
    quantized = np.round((patches.astype(np.float32) / scale) + zero_point)
    info = np.iinfo(dtype)
    return np.clip(quantized, info.min, info.max).astype(dtype)


def validate_patches(patches: np.ndarray, config: SafetyCoreConfig) -> np.ndarray:
    patches = np.asarray(patches)
    if patches.shape != config.patch_shape:
        raise ValueError(f"Expected patches shape {config.patch_shape}, got {patches.shape}")
    if patches.dtype != np.uint8:
        raise ValueError(f"Expected UINT8 patches, got {patches.dtype}")
    return patches


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def make_decision(logits: np.ndarray, config: SafetyCoreConfig) -> SafetyDecision:
    logits = np.asarray(logits)
    if logits.shape != config.logits_shape:
        raise ValueError(f"Expected logits shape {config.logits_shape}, got {logits.shape}")
    probabilities = softmax(logits)
    unsafe_score = float(probabilities[0, config.unsafe_index])
    return SafetyDecision(
        logits=logits,
        probabilities=probabilities,
        unsafe_score=unsafe_score,
        blocked=unsafe_score >= config.unsafe_threshold,
    )


def run_safety_core(
    patches: np.ndarray, runner: SafetyCoreRunner, config: SafetyCoreConfig
) -> SafetyDecision:
    """Run a board or simulator-backed FINN safety core and apply host policy."""

    checked_patches = validate_patches(patches, config)
    logits = runner(checked_patches)
    return make_decision(logits, config)
