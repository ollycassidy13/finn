"""Import trained BERT student weights into the FINN safety core."""

from __future__ import annotations

import importlib
import numpy as np
from dataclasses import asdict, dataclass
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from typing import Any

from bert.common import CorePreset, repo_path, write_json


@dataclass(frozen=True)
class ImportedWeight:
    target: str
    source: str
    source_shape: list[int]
    target_shape: list[int]
    transposed: bool
    max_abs: float
    quant_scale: float


def _load_safetensors(path: Path) -> dict[str, Any]:
    try:
        safetensors_torch = importlib.import_module("safetensors.torch")
    except ImportError as exc:
        raise RuntimeError(
            f"{path} requires safetensors; install safetensors or export a PyTorch .bin/.pt file"
        ) from exc
    return dict(safetensors_torch.load_file(str(path), device="cpu"))


def _load_torch(path: Path) -> dict[str, Any]:
    torch = importlib.import_module("torch")
    payload = torch.load(str(path), map_location="cpu")
    if isinstance(payload, dict):
        for key in ["state_dict", "model_state_dict", "model"]:
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
        return payload
    raise RuntimeError(f"Unsupported checkpoint payload in {path}")


def resolve_state_path(path: str | Path) -> Path:
    path = repo_path(path)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)
    for name in [
        "model.safetensors",
        "pytorch_model.bin",
        "pytorch_model.pt",
        "state_dict.pt",
    ]:
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No supported checkpoint found in {path}")


def load_state_dict(path: str | Path) -> dict[str, Any]:
    state_path = resolve_state_path(path)
    if state_path.suffix == ".safetensors":
        state = _load_safetensors(state_path)
    else:
        state = _load_torch(state_path)
    normalized = {}
    for name, value in state.items():
        if name.startswith("module."):
            name = name[len("module.") :]
        normalized[name] = value
    return normalized


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _candidate_keys(layer: int, kind: str) -> list[str]:
    bert = f"bert.encoder.layer.{layer}"
    bare = f"encoder.layer.{layer}"
    fallback = f"encoder.layers.{layer}"
    distil = f"distilbert.transformer.layer.{layer}"
    if kind == "attention":
        suffixes = [
            "attention.output.dense.weight",
            "attention.self.value.weight",
        ]
        return [f"{prefix}.{suffix}" for prefix in [bert, bare] for suffix in suffixes] + [
            f"{fallback}.self_attn.out_proj.weight",
            f"{distil}.attention.out_lin.weight",
        ]
    if kind == "ffn_expand":
        return [
            f"{bert}.intermediate.dense.weight",
            f"{bare}.intermediate.dense.weight",
            f"{fallback}.linear1.weight",
            f"{distil}.ffn.lin1.weight",
        ]
    if kind == "ffn_project":
        return [
            f"{bert}.output.dense.weight",
            f"{bare}.output.dense.weight",
            f"{fallback}.linear2.weight",
            f"{distil}.ffn.lin2.weight",
        ]
    raise ValueError(kind)


def _head_keys() -> list[str]:
    return [
        "classifier.weight",
        "score.weight",
        "pre_classifier.weight",
    ]


def _first_matching(state: dict[str, Any], keys: list[str]) -> tuple[str, np.ndarray] | None:
    for key in keys:
        if key in state:
            return key, _to_numpy(state[key])
    return None


def _quantize_for_target(
    source: np.ndarray, target_shape: list[int]
) -> tuple[np.ndarray, bool, float, float]:
    target = tuple(target_shape)
    transposed = False
    if source.shape == target:
        aligned = source
    elif source.ndim == 2 and source.T.shape == target:
        aligned = source.T
        transposed = True
    else:
        raise ValueError(f"source shape {source.shape} does not match target shape {target}")
    max_abs = float(np.max(np.abs(aligned))) if aligned.size else 0.0
    quant_scale = 1.0 if max_abs == 0.0 else 127.0 / max_abs
    quantized = np.clip(np.round(aligned * quant_scale), -128, 127).astype(np.float32)
    return quantized, transposed, max_abs, quant_scale


def _weight_targets(model: ModelWrapper, preset: CorePreset) -> list[tuple[str, list[str]]]:
    mvau_nodes = [node for node in model.graph.node if node.op_type == "MVAU"]
    expected_mvaus = preset.layers * 3 + 1
    if len(mvau_nodes) != expected_mvaus:
        raise RuntimeError(f"Expected {expected_mvaus} MVAU nodes, found {len(mvau_nodes)}")
    targets = []
    for layer in range(preset.layers):
        base = layer * 3
        targets.extend(
            [
                (
                    mvau_nodes[base].input[1],
                    _candidate_keys(layer, "attention"),
                ),
                (mvau_nodes[base + 1].input[1], _candidate_keys(layer, "ffn_expand")),
                (mvau_nodes[base + 2].input[1], _candidate_keys(layer, "ffn_project")),
            ]
        )
    targets.append((mvau_nodes[-1].input[1], _head_keys()))
    return targets


def apply_imported_weights(
    model: ModelWrapper,
    preset: CorePreset,
    state_path: str | Path,
    manifest_path: str | Path | None = None,
    strict: bool = False,
) -> list[ImportedWeight]:
    """Quantize matching trained checkpoint tensors into FINN core initializers.

    Unmatched tensors keep their deterministic initialized values unless `strict`
    is set. The hardware graph is BERT-shaped, so the importer maps the dense
    BERT attention-output/FFN/head weights that have direct shape-compatible
    counterparts in the current FINN core.
    """

    state = load_state_dict(state_path)
    imported: list[ImportedWeight] = []
    missing = []
    mismatched = []
    for target, candidates in _weight_targets(model, preset):
        target_value = model.get_initializer(target)
        if target_value is None:
            missing.append(f"{target}: target initializer missing")
            continue
        match = _first_matching(state, candidates)
        if match is None:
            missing.append(f"{target}: none of {candidates}")
            continue
        source_name, source_value = match
        try:
            quantized, transposed, max_abs, quant_scale = _quantize_for_target(
                source_value, list(target_value.shape)
            )
        except ValueError as exc:
            mismatched.append(f"{target} <- {source_name}: {exc}")
            continue
        model.set_initializer(target, quantized)
        imported.append(
            ImportedWeight(
                target=target,
                source=source_name,
                source_shape=list(source_value.shape),
                target_shape=list(target_value.shape),
                transposed=transposed,
                max_abs=max_abs,
                quant_scale=quant_scale,
            )
        )

    if strict and (missing or mismatched):
        details = "\n".join(missing + mismatched)
        raise RuntimeError(f"Could not import all FINN core weights:\n{details}")

    if manifest_path is not None:
        write_json(
            manifest_path,
            {
                "state_path": str(resolve_state_path(state_path)),
                "preset": preset.as_dict(),
                "strict": strict,
                "imported": [asdict(item) for item in imported],
                "missing": missing,
                "mismatched": mismatched,
            },
        )
    return imported
