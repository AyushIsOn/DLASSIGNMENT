"""CPU check of the dataset with the real tokenizer (run before any GPU time).

    python -m finetune.check_data --tokenizer artifacts/models/Qwen3-8B

* verifies data/sft against MANIFEST.json,
* checks that acharya.prompting.render_prompt is byte-identical to Qwen3's official
  chat template (enable_thinking=False) for every row,
* checks that the answer mask covers exactly the answer + <|im_end|>,
* reports token statistics used to size batches and estimate training time.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from finetune.data import (
    IGNORE_INDEX,
    answer_text,
    encode,
    prompt_messages,
    prompt_text,
    read_split,
    verify_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def check(tokenizer_path: Path, data_dir: Path, max_seq_len: int) -> dict[str, object]:
    from transformers import AutoTokenizer

    verify_manifest(data_dir, ROOT / "data/kb/cards.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    report: dict[str, object] = {"tokenizer": str(tokenizer_path), "max_seq_len": max_seq_len}
    for split in ("train", "validation", "test"):
        rows = read_split(data_dir, split)
        lengths, answers, too_long = [], [], 0
        for row in rows:
            official = tokenizer.apply_chat_template(
                prompt_messages(row), tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            if official != prompt_text(row):
                raise RuntimeError(f"prompt rendering differs from the chat template: {row['id']}")
            full = tokenizer.apply_chat_template(
                [*prompt_messages(row), {"role": "assistant", "content": answer_text(row)}],
                tokenize=False, enable_thinking=False,
            )
            if not full.startswith(official):
                raise RuntimeError(f"chat template prefix mismatch: {row['id']}")
            item = encode(row, tokenizer, max_seq_len)
            if item is None:
                too_long += 1
                continue
            supervised = [t for t, label in zip(item["input_ids"], item["labels"], strict=True)
                          if label != IGNORE_INDEX]
            if supervised[-1] != end_id:
                raise RuntimeError(f"answer does not end with <|im_end|>: {row['id']}")
            decoded = tokenizer.decode(supervised[:-1])
            if decoded.strip() != answer_text(row):
                raise RuntimeError(f"answer mask does not cover the answer exactly: {row['id']}")
            lengths.append(len(item["input_ids"]))
            answers.append(len(supervised))
        if too_long:
            raise RuntimeError(f"{too_long} {split} rows exceed max_seq_len={max_seq_len}")
        lengths.sort()
        report[split] = {
            "rows": len(rows),
            "tokens": sum(lengths),
            "supervised_tokens": sum(answers),
            "mean_tokens": round(statistics.mean(lengths), 1),
            "p50_tokens": lengths[len(lengths) // 2],
            "p99_tokens": lengths[int(len(lengths) * 0.99)],
            "max_tokens": lengths[-1],
            "max_answer_tokens": max(answers),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/sft")
    parser.add_argument("--max-seq-len", type=int, default=None,
                        help="default: max_seq_len from configs/train.yaml")
    args = parser.parse_args()
    if args.max_seq_len is None:
        from finetune.common import Config

        args.max_seq_len = int(Config.load().raw["max_seq_len"])
    print(json.dumps(check(args.tokenizer, args.data, args.max_seq_len), indent=2))
    print("DATA CHECK PASSED")


if __name__ == "__main__":
    main()
