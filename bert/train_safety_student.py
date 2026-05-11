#!/usr/bin/env python3
"""Fine-tune and export a BERT safety student with teacher distillation.

The default path is `--init-only`: create an initialized student and export it
without training. On a GPU cluster, remove `--init-only` and provide a JSONL
dataset with `text` and `label` fields. If `transformers` is installed, the
script uses Hugging Face BERT models and a safety teacher. Otherwise init-only
falls back to a small local PyTorch encoder so ONNX export remains testable.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Iterable

from bert.common import DEFAULT_BUILD_DIR, DEFAULT_CACHE_DIR, repo_path, write_json


class FallbackSafetyStudent(nn.Module):
    """Minimal BERT-shaped fallback used only when transformers is unavailable."""

    def __init__(self, vocab_size: int = 30522, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 4,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.classifier = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        key_padding_mask = attention_mask == 0 if attention_mask is not None else None
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.classifier(x[:, 0, :])


class LogitsOnlyWrapper(nn.Module):
    """Wrap Hugging Face outputs so ONNX export sees a plain logits tensor."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits if hasattr(out, "logits") else out


def configure_local_caches(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir / "xdg")
    os.environ["HF_HOME"] = str(cache_dir / "hf_home")
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hf_hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir / "hf_hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
    os.environ["TORCH_HOME"] = str(cache_dir / "torch")


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def batches(rows: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for idx in range(0, len(rows), batch_size):
        yield rows[idx : idx + batch_size]


def split_rows(rows: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    if len(rows) < 2 or val_fraction <= 0.0:
        return rows, []
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    val_count = min(val_count, len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


def export_onnx(model: nn.Module, output_path: Path, max_length: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = LogitsOnlyWrapper(model)
    model.eval()
    input_ids = torch.zeros((1, max_length), dtype=torch.long)
    attention_mask = torch.ones((1, max_length), dtype=torch.long)
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )


def write_export_manifest(args: argparse.Namespace, export_path: Path) -> None:
    manifest = {
        "student": args.student,
        "teacher": args.teacher,
        "tokenizer": args.tokenizer or args.student,
        "teacher_tokenizer": args.teacher_tokenizer or args.teacher,
        "dataset_jsonl": args.dataset_jsonl,
        "output_dir": str(args.output_dir),
        "cache_dir": str(args.cache_dir),
        "onnx": str(export_path),
        "init_only": args.init_only,
        "random_init": args.random_init,
        "num_labels": args.num_labels,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "alpha": args.alpha,
        "temperature": args.temperature,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
    }
    write_json(args.output_dir / "student_export_manifest.json", manifest)


def make_fallback_student(args: argparse.Namespace) -> nn.Module:
    return FallbackSafetyStudent(hidden=args.hidden, layers=args.layers)


def get_transformers_class(name: str):
    return getattr(importlib.import_module("transformers"), name)


def make_hf_student_and_tokenizer(args: argparse.Namespace):
    auto_config = get_transformers_class("AutoConfig")
    auto_model = get_transformers_class("AutoModelForSequenceClassification")
    auto_tokenizer = get_transformers_class("AutoTokenizer")

    cache_dir = str(args.cache_dir)
    tokenizer = auto_tokenizer.from_pretrained(args.tokenizer or args.student, cache_dir=cache_dir)
    if args.random_init:
        config = auto_config.from_pretrained(args.student, cache_dir=cache_dir)
        config.num_labels = args.num_labels
        student = auto_model.from_config(config)
    else:
        student = auto_model.from_pretrained(
            args.student,
            num_labels=args.num_labels,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=True,
        )
    return student, tokenizer


def train_with_teacher(args: argparse.Namespace) -> nn.Module:
    auto_model = get_transformers_class("AutoModelForSequenceClassification")
    auto_tokenizer = get_transformers_class("AutoTokenizer")

    rows = read_jsonl(repo_path(args.dataset_jsonl))
    if not rows:
        raise RuntimeError("Training dataset is empty")
    train_rows, val_rows = split_rows(rows, args.val_fraction, args.seed)
    write_json(
        args.output_dir / "dataset_split.json",
        {
            "dataset_jsonl": args.dataset_jsonl,
            "total_rows": len(rows),
            "train_rows": len(train_rows),
            "validation_rows": len(val_rows),
            "val_fraction": args.val_fraction,
            "seed": args.seed,
        },
    )

    student, tokenizer = make_hf_student_and_tokenizer(args)
    teacher = auto_model.from_pretrained(
        args.teacher,
        num_labels=args.num_labels,
        cache_dir=str(args.cache_dir),
        ignore_mismatched_sizes=True,
    )
    teacher_tokenizer = auto_tokenizer.from_pretrained(
        args.teacher_tokenizer or args.teacher,
        cache_dir=str(args.cache_dir),
    )
    device = torch.device(args.device)
    student.to(device)
    teacher.to(device)
    teacher.eval()
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        student.train()
        running = 0.0
        seen = 0
        random.Random(args.seed + epoch).shuffle(train_rows)
        for batch_rows in batches(train_rows, args.batch_size):
            text = [row["text"] for row in batch_rows]
            labels = torch.tensor([int(row["label"]) for row in batch_rows], device=device)
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=args.max_length,
            )
            enc = {key: value.to(device) for key, value in enc.items()}
            teacher_enc = teacher_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=args.max_length,
            )
            teacher_enc = {key: value.to(device) for key, value in teacher_enc.items()}
            with torch.no_grad():
                teacher_logits = teacher(**teacher_enc).logits
            student_logits = student(**enc).logits
            hard_loss = F.cross_entropy(student_logits, labels)
            temperature = args.temperature
            distill_loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature * temperature)
            loss = args.alpha * distill_loss + (1.0 - args.alpha) * hard_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu()) * len(batch_rows)
            seen += len(batch_rows)

        msg = f"epoch={epoch + 1} train_loss={running / max(1, seen):.6f}"
        if val_rows:
            msg += f" val_accuracy={evaluate(student, tokenizer, val_rows, args):.4f}"
        print(msg)

    student.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    return student


@torch.no_grad()
def evaluate(model: nn.Module, tokenizer, rows: list[dict], args: argparse.Namespace) -> float:
    device = torch.device(args.device)
    model.eval()
    correct = 0
    total = 0
    for batch_rows in batches(rows, args.batch_size):
        text = [row["text"] for row in batch_rows]
        labels = torch.tensor([int(row["label"]) for row in batch_rows], device=device)
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )
        enc = {key: value.to(device) for key, value in enc.items()}
        pred = model(**enc).logits.argmax(dim=-1)
        correct += int((pred == labels).sum().detach().cpu())
        total += labels.numel()
    return correct / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", default="bert-base-uncased")
    parser.add_argument("--teacher", default="unitary/toxic-bert")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--teacher-tokenizer", default=None)
    parser.add_argument("--dataset-jsonl", default=None)
    parser.add_argument(
        "--output-dir",
        default=str((DEFAULT_BUILD_DIR / "training").relative_to(repo_path("."))),
    )
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR.relative_to(repo_path("."))))
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-labels", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--export-onnx", default="student_init.onnx")
    args = parser.parse_args()

    args.cache_dir = repo_path(args.cache_dir)
    configure_local_caches(args.cache_dir)
    set_reproducible_seed(args.seed)
    output_dir = repo_path(args.output_dir)
    args.output_dir = output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.init_only:
        try:
            student, _ = make_hf_student_and_tokenizer(args)
        except Exception as exc:
            print(f"Falling back to local initialized student: {exc}")
            student = make_fallback_student(args)
    else:
        if args.dataset_jsonl is None:
            raise RuntimeError("--dataset-jsonl is required unless --init-only is set")
        student = train_with_teacher(args)

    export_path = output_dir / args.export_onnx
    export_onnx(student, export_path, args.max_length)
    write_export_manifest(args, export_path)
    print(f"Exported student ONNX: {export_path}")
    print(f"Cache directory: {args.cache_dir}")


if __name__ == "__main__":
    main()
