"""Loading, tokenizing and batching the SFT rows (shared by train/evaluate/check)."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from acharya.prompting import SYSTEM_PROMPT, completion_text, render_prompt

IGNORE_INDEX = -100


SPLITS = ("train", "validation", "test")


def split_path(data_dir: Path, split: str) -> Path:
    """data/sft/<split>.jsonl.gz (committed compressed: the train split is ~100 MB raw)."""
    return data_dir / f"{split}.jsonl.gz"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def read_split(data_dir: Path, split: str) -> list[dict[str, Any]]:
    return read_jsonl(split_path(data_dir, split))


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
    """A plain list-backed dataset (no `datasets` dependency at train time).

    Tokenizes in batches (identical ids to `encode`, which tokenizes row by row: no special
    tokens and no padding are added), so 90k rows take seconds instead of minutes of GPU time.
    """

    def __init__(self, rows: Sequence[dict[str, Any]], tokenizer: Any, max_seq_len: int,
                 sort_by_length: bool = False) -> None:
        self.items = []
        self.dropped = 0
        chunk = 4096
        for start in range(0, len(rows), chunk):
            part = rows[start:start + chunk]
            prompts = tokenizer([prompt_text(row) for row in part],
                                add_special_tokens=False)["input_ids"]
            answers = tokenizer([completion_text(answer_text(row)) for row in part],
                                add_special_tokens=False)["input_ids"]
            for prompt_ids, answer_ids in zip(prompts, answers, strict=True):
                input_ids = list(prompt_ids) + list(answer_ids)
                if len(input_ids) > max_seq_len:
                    self.dropped += 1
                    continue
                self.items.append({"input_ids": input_ids,
                                   "labels": [IGNORE_INDEX] * len(prompt_ids) + list(answer_ids)})
        if not self.items:
            raise ValueError("no examples fit into max_seq_len")
        if sort_by_length:  # evaluation: similar lengths per batch -> little padding
            self.items.sort(key=lambda item: len(item["input_ids"]))

    @property
    def lengths(self) -> list[int]:
        return [len(item["input_ids"]) for item in self.items]

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


def padded_length(length: int, multiple: int = 8) -> int:
    return -(-length // multiple) * multiple


class TokenBudgetBatchSampler:
    """Batches of similar-length examples whose PADDED size stays under `max_tokens`.

    Why: a fixed batch of 16 sequences is mostly padding when lengths vary 60-1100 tokens,
    and small batches leave an H200 idle (round 1 ran at ~40% utilization). Here every
    micro-batch carries ~max_tokens real tokens, padding is <5%, and attention stays
    per-sequence (no packing cross-talk, works with plain SDPA).

    Deterministic given (seed, epoch): Trainer calls set_epoch(), and on resume it skips
    the already-trained batches of the same order, so resuming is exact.
    """

    def __init__(self, lengths: Sequence[int], max_tokens: int, seed: int,
                 shuffle: bool = True, pool_batches: int = 64, multiple: int = 8) -> None:
        longest = padded_length(max(lengths), multiple)
        if longest > max_tokens:
            raise ValueError(f"max_tokens={max_tokens} is smaller than the longest example "
                             f"({longest} padded tokens)")
        self.lengths = list(lengths)
        self.max_tokens = max_tokens
        self.seed = seed
        self.shuffle = shuffle
        self.pool = pool_batches
        self.multiple = multiple
        self.epoch = 0
        self._fixed: list[list[int]] | None = None
        self._cache: tuple[int, list[list[int]]] | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _pack(self, indices: list[int]) -> list[list[int]]:
        batches, current, longest = [], [], 0
        for index in indices:
            size = padded_length(self.lengths[index], self.multiple)
            if current and max(longest, size) * (len(current) + 1) > self.max_tokens:
                batches.append(current)
                current, longest = [], 0
            current.append(index)
            longest = max(longest, size)
        if current:
            batches.append(current)
        return batches

    def _fixed_batches(self) -> list[list[int]]:
        """Batch membership is fixed for the whole run, so every epoch has the same number
        of batches (Trainer's step arithmetic and resume-skipping rely on that)."""
        if self._fixed is None:
            import random

            order = list(range(len(self.lengths)))
            if not self.shuffle:
                order.sort(key=lambda i: self.lengths[i])
                self._fixed = self._pack(order)
            else:
                random.Random(f"{self.seed}-pack").shuffle(order)
                # sort inside large random pools: similar lengths per batch, random mix overall
                mean = max(1, sum(self.lengths) // len(self.lengths))
                pool_size = max(1, self.pool * self.max_tokens // mean)
                self._fixed = []
                for start in range(0, len(order), pool_size):
                    pool = sorted(order[start:start + pool_size], key=lambda i: self.lengths[i])
                    self._fixed.extend(self._pack(pool))
        return self._fixed

    def batches(self) -> list[list[int]]:
        if self._cache is not None and self._cache[0] == self.epoch:
            return self._cache[1]
        import random

        result = list(self._fixed_batches())
        if self.shuffle:
            random.Random(f"{self.seed}-{self.epoch}").shuffle(result)
            # the largest batch first: an out-of-memory error shows up at step 1, not later
            biggest = max(range(len(result)), key=lambda b: self.cost(result[b]))
            result.insert(0, result.pop(biggest))
        self._cache = (self.epoch, result)
        return result

    def worst_batches(self) -> list[list[int]]:
        """The batches with the most padded tokens and the longest sequence (memory probe)."""
        fixed = self._fixed_batches()
        by_cost = max(fixed, key=self.cost)
        by_length = max(fixed, key=lambda b: max(self.lengths[i] for i in b))
        return [by_cost] if by_cost == by_length else [by_cost, by_length]

    def cost(self, batch: list[int]) -> int:
        return max(padded_length(self.lengths[i], self.multiple) for i in batch) * len(batch)

    def __iter__(self) -> Any:
        return iter(self.batches())

    def __len__(self) -> int:
        return len(self.batches())


def sha256_file(path: Path) -> str:
    """SHA-256 of the file; for .gz files of the DECOMPRESSED content, so the hash does not
    depend on the zlib build that compressed it."""
    digest = hashlib.sha256()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(data_dir: Path, kb_path: Path | None = None) -> dict[str, str]:
    """Fail fast if the committed dataset files were modified or are incomplete."""
    manifest = json.loads((data_dir / "MANIFEST.json").read_text())
    expected = manifest["outputs"]
    for split in SPLITS:
        path = split_path(data_dir, split)
        name = path.name
        if not path.is_file() or name not in expected:
            raise RuntimeError(f"{path} is missing (rebuild with `python -m "
                               "finetune.build_dataset`, or `git pull` the data)")
        actual = sha256_file(path)
        if actual != expected[name]:
            raise RuntimeError(f"{data_dir / name} does not match MANIFEST.json "
                               "(rebuild with `python -m finetune.build_dataset`)")
    if kb_path is not None and sha256_file(kb_path) != expected["cards.jsonl"]:
        raise RuntimeError(f"{kb_path} does not match MANIFEST.json")
    prompt_hash = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
    if manifest["system_prompt_sha256"] != prompt_hash:
        raise RuntimeError("SYSTEM_PROMPT changed since the dataset was built")
    return expected
