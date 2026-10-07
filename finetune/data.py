"""Loading, tokenizing and batching the SFT rows (shared by train/evaluate/check)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from acharya.prompting import SYSTEM_PROMPT, completion_text, render_prompt

IGNORE_INDEX = -100


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def prompt_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """System prompt + history + final user turn (everything except the answer)."""
    messages = row["messages"]
    if messages[-1]["role"] != "assistant" or messages[-2]["role"] != "user":
        raise ValueError(f"row {row['id']} must end with a user/assistant pair")
    return [{"role": "system", "content": SYSTEM_PROMPT}, *messages[:-1]]


def prompt_text(row: dict[str, Any]) -> str:
    return render_prompt(prompt_messages(row))  # type: ignore[arg-type]


def answer_text(row: dict[str, Any]) -> str:
    return str(row["messages"][-1]["content"]).strip()


def encode(row: dict[str, Any], tokenizer: Any, max_seq_len: int) -> dict[str, Any] | None:
    """Prompt tokens are masked with -100 so the loss is computed on the answer only.

    Prompt and completion are tokenized separately and concatenated, which is
    exactly what happens at inference (the prompt is tokenized, then generated
    tokens are appended). Returns None if the example does not fit.
    """
    prompt_ids = tokenizer(prompt_text(row), add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(completion_text(answer_text(row)), add_special_tokens=False)[
        "input_ids"]
    input_ids = list(prompt_ids) + list(answer_ids)
    if len(input_ids) > max_seq_len:
        return None
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(answer_ids)
    return {"input_ids": input_ids, "labels": labels}


class SFTDataset:
    """A plain list-backed dataset (no `datasets` dependency at train time)."""

    def __init__(self, rows: Sequence[dict[str, Any]], tokenizer: Any, max_seq_len: int) -> None:
        self.items = []
        self.dropped = 0
        for row in rows:
            item = encode(row, tokenizer, max_seq_len)
            if item is None:
                self.dropped += 1
            else:
                self.items.append(item)
        if not self.items:
            raise ValueError("no examples fit into max_seq_len")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]

    @property
    def tokens(self) -> int:
        return sum(len(item["input_ids"]) for item in self.items)

    @property
    def supervised_tokens(self) -> int:
        return sum(sum(label != IGNORE_INDEX for label in item["labels"]) for item in self.items)


class Collator:
    """Right-pads a batch; padded labels are -100 so they never contribute to the loss."""

    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8) -> None:
        self.pad_token_id = pad_token_id
        self.multiple = pad_to_multiple_of

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, Any]:
        import torch

        longest = max(len(item["input_ids"]) for item in features)
        if self.multiple:
            longest = -(-longest // self.multiple) * self.multiple
        input_ids, labels, attention = [], [], []
        for item in features:
            pad = longest - len(item["input_ids"])
            input_ids.append(list(item["input_ids"]) + [self.pad_token_id] * pad)
            labels.append(list(item["labels"]) + [IGNORE_INDEX] * pad)
            attention.append([1] * len(item["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(data_dir: Path, kb_path: Path | None = None) -> dict[str, str]:
    """Fail fast if the committed dataset files were modified or are incomplete."""
    manifest = json.loads((data_dir / "MANIFEST.json").read_text())
    expected = manifest["outputs"]
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        actual = sha256_file(data_dir / name)
        if actual != expected[name]:
            raise RuntimeError(f"{data_dir / name} does not match MANIFEST.json "
                               "(rebuild with `python -m finetune.build_dataset`)")
    if kb_path is not None and sha256_file(kb_path) != expected["cards.jsonl"]:
        raise RuntimeError(f"{kb_path} does not match MANIFEST.json")
    prompt_hash = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
    if manifest["system_prompt_sha256"] != prompt_hash:
        raise RuntimeError("SYSTEM_PROMPT changed since the dataset was built")
    return expected
