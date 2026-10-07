"""Compare the base model and the fine-tuned model on the held-out test set.

    python -m finetune.evaluate                 # full test set (~1,900 questions)

The base model and the fine-tuned model are the *same* weights with the LoRA
adapter switched off/on, so the comparison isolates what fine-tuning changed.
Both get the identical prompt (system prompt + question [+ retrieved entries]).

Outputs (artifacts/run/eval/):
  generations.jsonl   every answer (resumable: finished answers are not redone)
  results.json        scores per test group / attribute, test loss & perplexity
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from acharya.prompting import clean_generation
from finetune.common import (
    Config,
    append_jsonl,
    load_base_model,
    load_tokenizer,
    log,
    stop_token_ids,
    write_json,
)
from finetune.data import Collator, SFTDataset, prompt_text, read_split, verify_manifest
from finetune.metrics import score_row

MODELS = ("base", "finetuned")
GROUP_LABELS = {
    "seen_closed": "Knowledge recall (no retrieval, new question wording)",
    "heldout_open": "Unseen conditions/terms + retrieved KB entries (RAG)",
    "unseen_closed": "Unseen Sanskrit terms, no retrieval (infer the meaning: generalization)",
    "heldout_closed": "Unseen conditions, no retrieval (control: unknowable)",
    "concepts": "General Ayurveda concepts (new wording)",
    "safety": "Safety (doses, emergencies, diagnosis requests)",
}
# Headline "generalization" = questions about things never trained on, so they cannot be
# answered from memorised training rows: reading unseen KB entries + inferring unseen terms.
GENERALIZATION_GROUPS = ("heldout_open", "unseen_closed")


def load_model(config: Config, adapter: Path) -> tuple[Any, Any]:
    from peft import PeftModel

    tokenizer = load_tokenizer(config.model_dir, padding_side="left")
    base = load_base_model(config.model_dir, for_training=False)
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
    model.eval()
    return model, tokenizer


def generate(model: Any, tokenizer: Any, prompts: list[str], max_new_tokens: int,
             use_adapter: bool) -> list[str]:
    import contextlib

    import torch
    from transformers import GenerationConfig

    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    generation = GenerationConfig(
        max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
        eos_token_id=stop_token_ids(tokenizer), pad_token_id=tokenizer.pad_token_id,
    )
    adapter_off = contextlib.nullcontext() if use_adapter else model.disable_adapter()
    with torch.inference_mode(), adapter_off:
        output = model.generate(**encoded, generation_config=generation)
    new_tokens = output[:, encoded["input_ids"].shape[1]:]
    return [clean_generation(text) for text in
            tokenizer.batch_decode(new_tokens, skip_special_tokens=False)]


def answer_loss(model: Any, tokenizer: Any, rows: list[dict[str, Any]], max_len: int,
                batch_size: int, use_adapter: bool) -> dict[str, float]:
    """Mean per-token NLL of the reference answers (teacher forcing)."""
    import contextlib

    import torch

    tokenizer.padding_side = "right"
    dataset = SFTDataset(rows, tokenizer, max_len, sort_by_length=True)
    tokenizer.padding_side = "left"
    collate = Collator(tokenizer.pad_token_id)
    total_loss, total_tokens = 0.0, 0
    adapter_off = contextlib.nullcontext() if use_adapter else model.disable_adapter()
    with torch.inference_mode(), adapter_off:
        for start in range(0, len(dataset), batch_size):
            batch = collate([dataset[i] for i in range(start, min(len(dataset),
                                                                  start + batch_size))])
            batch = {key: value.to(model.device) for key, value in batch.items()}
            logits = model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"]).logits[:, :-1].float()
            labels = batch["labels"][:, 1:]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                ignore_index=-100, reduction="sum")
            total_loss += float(loss)
            total_tokens += int((labels != -100).sum())
    mean = total_loss / max(1, total_tokens)
    return {"loss": round(mean, 4), "perplexity": round(math.exp(mean), 3),
            "tokens": total_tokens}


def summarize(rows: list[dict[str, Any]], answers: dict[tuple[str, str], dict[str, Any]]
              ) -> dict[str, Any]:
    def stats(subset: list[dict[str, Any]], model: str) -> dict[str, Any]:
        scored = [answers[(row["id"], model)]["metrics"] for row in subset
                  if (row["id"], model) in answers]
        if not scored:
            return {"n": 0}
        n = len(scored)
        return {
            "n": n,
            "accuracy": round(sum(item["correct"] for item in scored) / n, 4),
            "fact_score": round(sum(item["score"] for item in scored) / n, 4),
            "token_f1": round(sum(item["token_f1"] for item in scored) / n, 4),
            "rouge_l": round(sum(item["rouge_l"] for item in scored) / n, 4),
            "mean_words": round(sum(item["answer_words"] for item in scored) / n, 1),
        }

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_attribute: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row["meta"]["group"]].append(row)
        by_attribute[f"{row['meta']['group']}:{row['meta']['attribute']}"].append(row)
    groups = {g: {m: stats(subset, m) for m in MODELS} for g, subset in sorted(by_group.items())}
    attributes = {a: {m: stats(subset, m) for m in MODELS}
                  for a, subset in sorted(by_attribute.items())}
    knowledge = [row for row in rows if row["meta"]["group"] != "heldout_closed"]
    unseen = [row for row in rows if row["meta"]["group"] in GENERALIZATION_GROUPS]
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[f"{row['meta']['group']}:{row['meta']['source']}"].append(row)
    return {"overall_excluding_control": {m: stats(knowledge, m) for m in MODELS},
            "generalization": {m: stats(unseen, m) for m in MODELS},
            "groups": groups, "attributes": attributes,
            "group_by_source": {k: {m: stats(v, m) for m in MODELS}
                                for k, v in sorted(by_source.items())}}


def evaluate(config: Config, adapter: Path, limit: int | None, smoke: bool) -> dict[str, Any]:
    import torch

    from finetune.train import adapter_complete

    if not adapter_complete(adapter):
        raise SystemExit(f"No trained adapter in {adapter}: run `python -m finetune.train` first")
    verify_manifest(config.data_dir)
    out = config.eval_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = read_split(config.data_dir, "test")
    if limit:
        # stratified: keep every group represented
        per_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            per_group[row["meta"]["group"]].append(row)
        rows = [row for group in per_group.values()
                for row in group[: max(1, limit // len(per_group))]]
    settings = config.evaluation
    log("evaluation_start", rows=len(rows), adapter=adapter,
        device="cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(config, adapter)

    done: dict[tuple[str, str], dict[str, Any]] = {}
    generations = out / "generations.jsonl"
    if generations.exists():
        for line in generations.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                done[(record["id"], record["model"])] = record
    by_id = {row["id"]: row for row in rows}
    batch_size = int(settings["batch_size"])
    for model_name in ("finetuned", "base"):
        todo = [row for row in rows if (row["id"], model_name) not in done]
        todo.sort(key=lambda row: len(prompt_text(row)))  # similar lengths -> less padding
        started, total = time.time(), len(todo)
        for start in range(0, total, batch_size):
            batch = todo[start:start + batch_size]
            t0 = time.time()
            answers = generate(model, tokenizer, [prompt_text(row) for row in batch],
                               int(settings["max_new_tokens"]), model_name == "finetuned")
            seconds = (time.time() - t0) / len(batch)
            for row, answer in zip(batch, answers, strict=True):
                record = {"id": row["id"], "model": model_name, "answer": answer,
                          "seconds_per_answer_in_batch": round(seconds, 3),
                          "metrics": score_row(row, answer)}
                append_jsonl(generations, record)
                done[(row["id"], model_name)] = record
            finished = min(total, start + batch_size)
            rate = finished / max(1e-6, time.time() - started)
            log("generation_progress", model=model_name, done=finished, total=total,
                eta_minutes=round((total - finished) / max(rate, 1e-6) / 60, 1))

    # rescore (cheap) so metric fixes never require regenerating
    for key, record in done.items():
        if key[0] in by_id:
            record["metrics"] = score_row(by_id[key[0]], record["answer"])

    max_len = int(config.raw["max_seq_len"])
    loss_batch = int(settings["loss_batch_size"])
    losses = {}
    for model_name in MODELS:
        losses[model_name] = answer_loss(model, tokenizer, rows, max_len, loss_batch,
                                         model_name == "finetuned")
        log("test_loss", model=model_name, **losses[model_name])

    summary = summarize(rows, done)
    examples = pick_examples(rows, done)
    results = {
        "test_rows": len(rows),
        "smoke": smoke,
        "group_labels": GROUP_LABELS,
        "test_answer_loss": losses,
        **summary,
        "examples": examples,
        "generation": {"greedy": True, "max_new_tokens": settings["max_new_tokens"],
                       "batch_size": batch_size},
    }
    write_json(out / "results.json", results)
    overall = summary["overall_excluding_control"]
    general = summary["generalization"]
    log("evaluation_done", base_accuracy=overall["base"].get("accuracy"),
        finetuned_accuracy=overall["finetuned"].get("accuracy"),
        base_generalization=general["base"].get("accuracy"),
        finetuned_generalization=general["finetuned"].get("accuracy"),
        base_loss=losses["base"]["loss"], finetuned_loss=losses["finetuned"]["loss"])
    return results


def pick_examples(rows: list[dict[str, Any]], answers: dict[tuple[str, str], dict[str, Any]],
                  per_group: int = 3) -> list[dict[str, Any]]:
    """Deterministic, unbiased picks: the first rows of each group in file order."""
    picked: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        group = row["meta"]["group"]
        if counts[group] >= per_group or any((row["id"], m) not in answers for m in MODELS):
            continue
        counts[group] += 1
        picked.append({
            "id": row["id"], "group": group, "question": row["question"],
            "open_book": row["meta"]["open_book"], "reference": row["messages"][-1]["content"],
            **{m: {"answer": answers[(row["id"], m)]["answer"],
                   "correct": answers[(row["id"], m)]["metrics"]["correct"]} for m in MODELS},
        })
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--limit", type=int, help="evaluate a stratified subset (smoke tests)")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = Config.load(args.config) if args.config else Config.load()
    evaluate(config, args.adapter or config.adapter_dir, args.limit, args.smoke)


if __name__ == "__main__":
    main()
