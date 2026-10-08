"""LoRA fine-tuning of Qwen3 (14B by default) on the AcharyaGPT dataset (single GPU, bf16).

    python -m finetune.train                      # uses configs/train.yaml

Safe to re-run: an interrupted run resumes from its last checkpoint, a finished run is
skipped. A run restored on another machine/account from a recovery archive
(scripts/restore_and_resume.sh) resumes exactly where it stopped. Writes to output_dir
(configs/train.yaml):
    adapter/                  final LoRA adapter (best validation loss)
    checkpoints/              resumable checkpoints (adapter + optimizer + scheduler + RNG)
    recovery/recovery-latest.tar   portable copy of the newest + best checkpoint
    metrics.jsonl             every logged step (loss, lr, grad norm, eval loss, tokens/s)
    training_summary.json     what happened, for the report

GPU efficiency (round 1 ran at ~40% utilization): batches are built by token budget
(similar lengths, <5% padding) instead of a fixed 16 sequences, LoRA dropout is off,
gradient checkpointing is only used if a memory probe on the worst batch says it is needed,
the optimizer is fused AdamW and batches are collated in background workers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import time
from pathlib import Path
from typing import Any

from finetune.common import (
    Config,
    append_jsonl,
    cpu_allowed,
    device_info,
    load_base_model,
    load_tokenizer,
    log,
    use_bf16,
    write_json,
)
from finetune.data import Collator, SFTDataset, TokenBudgetBatchSampler, read_split, verify_manifest

ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def adapter_complete(path: Path) -> bool:
    return all((path / name).is_file() and (path / name).stat().st_size > 0
               for name in ADAPTER_FILES)


def run_fingerprint(config: Config, manifest: dict[str, str]) -> str:
    """Identity of a run: checkpoints from a different config/data are never resumed.
    Paths are left out, so a run restored into another folder or account still matches."""
    raw = json.loads(json.dumps(config.raw))
    raw.pop("output_dir", None)
    raw.get("base_model", {}).pop("local_dir", None)
    value = json.dumps({"config": raw, "data": manifest}, sort_keys=True)
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def check_model_dir(model_dir: Path, config: Config) -> None:
    shards = config.raw["base_model"]["shards"]
    missing = [name for name in shards if not (model_dir / name).is_file()]
    if missing or not (model_dir / "config.json").is_file():
        raise SystemExit(
            f"Base model not found in {model_dir} (missing {missing or ['config.json']}). "
            "Run `bash scripts/lightning_cpu_setup.sh` first (CPU, no GPU cost)."
        )
    if any(digest != "-" for digest in shards.values()):  # "-" = tiny smoke-test model
        from finetune.download_model import verified

        if not verified(model_dir, config):
            raise SystemExit(f"{model_dir} is not verified: run "
                             "`python -m finetune.download_model` (CPU) first.")


# ------------------------------------------------------------------ memory probe


def probe_memory(model: Any, dataset: SFTDataset, sampler: TokenBudgetBatchSampler,
                 collator: Collator, optimizer_bytes: int) -> float | None:
    """Forward+backward on the worst batches. Returns the peak fraction of GPU memory
    (including the not-yet-allocated optimizer state), or None on out-of-memory."""
    import torch

    total = torch.cuda.get_device_properties(0).total_memory
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        for indices in sampler.worst_batches():
            batch = collator([dataset[i] for i in indices])
            batch = {key: value.to(model.device) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            loss.backward()
            del loss, batch
            model.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated() + optimizer_bytes
        return peak / total
    except torch.cuda.OutOfMemoryError:
        return None
    finally:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()


def enable_checkpointing(model: Any) -> None:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()  # makes checkpointing work with a frozen base


def plan_memory(model: Any, dataset: SFTDataset, collator: Collator, hp: dict[str, Any],
                trainable: int, previous: dict[str, Any] | None,
                resuming: bool) -> tuple[TokenBudgetBatchSampler, bool, dict[str, Any]]:
    """Choose (tokens per micro-batch, gradient checkpointing, accumulation).

    Fastest first: big batches without checkpointing; then smaller batches without
    checkpointing (accumulation goes up so every optimizer step still sees the configured
    number of tokens); checkpointing only as a last resort. A resumed run keeps exactly the
    plan it started with (another plan would change the data order)."""
    import gc

    import torch

    seed = int(hp["seed"])
    setting = hp["gradient_checkpointing"]
    target = int(hp["max_tokens_per_batch"])
    base_accumulation = int(hp["gradient_accumulation"])

    def result(tokens: int, use: bool, accumulation: int, probe: Any) -> tuple[
            TokenBudgetBatchSampler, bool, dict[str, Any]]:
        if use and not getattr(model, "is_gradient_checkpointing", False):
            enable_checkpointing(model)
        if not use and getattr(model, "is_gradient_checkpointing", False):
            model.gradient_checkpointing_disable()
        return (TokenBudgetBatchSampler(dataset.lengths, tokens, seed), use,
                {"max_tokens": tokens, "gradient_checkpointing": use,
                 "gradient_accumulation": accumulation, "probe": probe})

    if resuming and previous:
        return result(int(previous["max_tokens"]), bool(previous["gradient_checkpointing"]),
                      int(previous.get("gradient_accumulation", base_accumulation)),
                      "kept from the interrupted run")
    if not torch.cuda.is_available():  # CPU tests
        return result(target, setting is True, base_accumulation, "skipped (no GPU)")

    longest = max(dataset.lengths)
    if setting == "auto":
        candidates = [(1, False), (2, False), (4, False), (1, True), (2, True), (4, True)]
    else:
        candidates = [(1, bool(setting)), (2, bool(setting)), (4, bool(setting))]
    limit = float(hp.get("max_memory_fraction", 0.9))
    optimizer_bytes = trainable * 4 * 3  # fp32 Adam moments + gradients
    attempts = []
    for divisor, use in candidates:
        tokens = target // divisor
        if tokens < max(2048, longest + 8):
            continue
        plan = result(tokens, use, base_accumulation * divisor, attempts)
        fraction = probe_memory(model, dataset, plan[0], collator, optimizer_bytes)
        gc.collect()
        attempts.append({"max_tokens": tokens, "gradient_checkpointing": use,
                         "peak_fraction": None if fraction is None else round(fraction, 3)})
        log("memory_probe", **attempts[-1])
        if fraction is not None and fraction <= limit:
            return plan
    raise SystemExit(f"Out of GPU memory even with small batches: {attempts}")


# ------------------------------------------------------------------ training


def train(config: Config, *, max_steps: int | None = None, smoke: bool = False,
          allow_cpu: bool = False) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import Trainer, TrainerCallback, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint

    from finetune import recovery

    out = config.output_dir
    checkpoints = out / "checkpoints"
    summary_path = out / "training_summary.json"
    if adapter_complete(config.adapter_dir) and summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        if summary.get("completed"):
            log("training_skipped", reason="adapter already trained", adapter=config.adapter_dir)
            return summary

    manifest = verify_manifest(config.data_dir)
    fingerprint = run_fingerprint(config, manifest)
    run_info_path = out / "RUN_INFO.json"
    run_info = json.loads(run_info_path.read_text()) if run_info_path.is_file() else {}
    if run_info and run_info.get("fingerprint") != fingerprint \
            and any(checkpoints.glob("checkpoint-*")):
        raise SystemExit(
            f"{checkpoints} holds checkpoints from a different config or dataset. "
            f"Move {out} away (or set ACHARYA_RUN_DIR) before training again."
        )
    resume = None
    if checkpoints.is_dir():
        last = get_last_checkpoint(str(checkpoints))
        if last and (Path(last) / "trainer_state.json").is_file():
            resume = last
            recovery.fix_checkpoint_paths(checkpoints)  # restored from another machine
    out.mkdir(parents=True, exist_ok=True)

    hp = config.train
    torch.manual_seed(int(hp["seed"]))
    check_model_dir(config.model_dir, config)
    device = device_info()
    log("device", **device)
    if not device["cuda_available"] and not (smoke or allow_cpu or cpu_allowed()):
        raise SystemExit("CUDA is not available: training needs the GPU. "
                         "Run `nvidia-smi` and check the Studio machine type.")
    bf16 = use_bf16()

    tokenizer = load_tokenizer(config.model_dir, padding_side="right")
    max_len = int(config.raw["max_seq_len"])
    train_rows = read_split(config.data_dir, "train")
    val_rows = read_split(config.data_dir, "validation")
    if smoke:
        train_rows, val_rows = train_rows[:64], val_rows[:16]
    started_tokenizing = time.time()
    train_set = SFTDataset(train_rows, tokenizer, max_len)
    val_set = SFTDataset(val_rows, tokenizer, max_len, sort_by_length=True)
    if train_set.dropped or val_set.dropped:
        raise SystemExit(f"{train_set.dropped + val_set.dropped} examples exceed max_seq_len")
    log("data", train_examples=len(train_set), train_tokens=train_set.tokens,
        supervised_tokens=train_set.supervised_tokens, validation_examples=len(val_set),
        tokenize_seconds=round(time.time() - started_tokenizing, 1))

    model = load_base_model(config.model_dir, for_training=True)
    lora = config.lora
    model = get_peft_model(model, LoraConfig(
        r=int(lora["r"]), lora_alpha=int(lora["alpha"]), lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]), bias="none", task_type="CAUSAL_LM",
    ))
    trainable, total = model.get_nb_trainable_parameters()
    log("lora", trainable_parameters=trainable, total_parameters=total,
        trainable_percent=round(100 * trainable / total, 3))

    collator = Collator(tokenizer.pad_token_id)
    sampler, checkpointing, memory = plan_memory(
        model, train_set, collator, hp, trainable, run_info.get("memory"), resume is not None)
    write_json(run_info_path, {"fingerprint": fingerprint, "config": config.raw,
                               "memory": memory})
    log("batching", **{k: v for k, v in memory.items() if k != "probe"},
        batches_per_epoch=len(sampler),
        mean_examples_per_batch=round(len(train_set) / len(sampler), 1))

    accumulation = int(memory["gradient_accumulation"])
    eval_every = int(hp["eval_every_steps"])
    steps_per_epoch = max(1, len(sampler) // accumulation)
    planned_steps = max_steps or math.ceil(steps_per_epoch * float(hp["epochs"]))
    if smoke:
        eval_every = max(1, planned_steps // 2)
    cuda = torch.cuda.is_available()
    workers = int(hp.get("dataloader_workers", 4)) if cuda else 0
    args = TrainingArguments(
        output_dir=str(checkpoints),
        num_train_epochs=float(hp["epochs"]),
        max_steps=max_steps or -1,
        per_device_train_batch_size=1,  # ignored: batches come from the token-budget sampler
        per_device_eval_batch_size=int(hp.get("eval_batch_size", 32)),
        gradient_accumulation_steps=accumulation,
        learning_rate=float(hp["learning_rate"]),
        lr_scheduler_type=str(hp["lr_scheduler"]),
        warmup_ratio=float(hp["warmup_ratio"]),
        weight_decay=float(hp["weight_decay"]),
        max_grad_norm=float(hp["max_grad_norm"]),
        optim="adamw_torch_fused" if cuda else "adamw_torch",
        bf16=bf16,
        tf32=True if cuda and torch.cuda.get_device_capability()[0] >= 8 else None,
        gradient_checkpointing=checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=eval_every,
        eval_on_start=True,  # step-0 validation loss == the untouched base model
        save_strategy="steps",
        save_steps=eval_every,
        save_total_limit=int(hp["keep_checkpoints"]),  # the best one is always kept too
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=int(hp["logging_steps"]),
        logging_first_step=True,
        report_to="none",
        seed=int(hp["seed"]),
        data_seed=int(hp["seed"]),
        remove_unused_columns=False,
        dataloader_num_workers=workers,
        dataloader_pin_memory=cuda,
        save_safetensors=True,
        include_num_input_tokens_seen="non_padding",
        disable_tqdm=False,
        use_cpu=not cuda,
    )

    started = time.time()
    limit_seconds = float(hp["max_train_minutes"]) * 60
    deadline = os.environ.get("ACHARYA_TRAIN_DEADLINE")  # unix time, set by lightning_gpu_run.sh
    if deadline:
        limit_seconds = min(limit_seconds, max(120.0, float(deadline) - started))
        log("time_budget", training_minutes_allowed=round(limit_seconds / 60, 1),
            note="training stops here, saves the best adapter, evaluation still runs")
    save_seconds = float(hp.get("save_every_minutes", 10)) * 60
    stop = {"requested": False, "reason": None}
    clock = {"last_save": time.time(), "last_log": time.time(), "last_tokens": 0}

    def on_signal(signum: int, _frame: Any) -> None:
        stop.update(requested=True, reason=f"signal {signum}")
        log("stop_requested", signal=signum, note="saving a checkpoint, then stopping")

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    packer = recovery.Packer(out)

    class Progress(TrainerCallback):
        def on_step_end(self, _args: Any, state: Any, control: Any, **_: Any) -> None:
            if time.time() - started > limit_seconds and not stop["requested"]:
                stop.update(requested=True, reason="max_train_minutes reached")
                log("time_limit", minutes=round(limit_seconds / 60, 1))
            if time.time() - clock["last_save"] >= save_seconds:
                control.should_save = True  # time-based resume point (every ~10 minutes)
            if stop["requested"]:
                control.should_save = True
                control.should_training_stop = True
            if state.global_step == 20:
                per_step = (time.time() - started) / 20
                log("eta", seconds_per_step=round(per_step, 2), planned_steps=planned_steps,
                    estimated_minutes=round(per_step * planned_steps / 60, 1),
                    time_limit_minutes=hp["max_train_minutes"])

        def on_save(self, _args: Any, state: Any, _control: Any, **_: Any) -> None:
            clock["last_save"] = time.time()
            packer.pack(state.global_step, state.best_model_checkpoint,
                        force=stop["requested"])

        def on_log(self, _args: Any, state: Any, _control: Any, logs: Any = None,
                   **_: Any) -> None:
            if not logs:
                return
            now = time.time()
            record = {"step": state.global_step, "epoch": round(state.epoch or 0, 4),
                      "elapsed_s": round(now - started, 1), **logs}
            tokens = int(state.num_input_tokens_seen or 0)
            if "loss" in logs and now > clock["last_log"] and tokens > clock["last_tokens"]:
                record["tokens_per_second"] = round(
                    (tokens - clock["last_tokens"]) / (now - clock["last_log"]))
                clock.update(last_log=now, last_tokens=tokens)
            if cuda:
                record["gpu_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            append_jsonl(out / "metrics.jsonl", record)

    if resume:
        log("resuming", checkpoint=resume)
    elif (out / "metrics.jsonl").exists():
        (out / "metrics.jsonl").unlink()  # fresh run: start the curve from scratch

    class BudgetTrainer(Trainer):
        def get_train_dataloader(self) -> Any:
            loader = DataLoader(self.train_dataset, batch_sampler=sampler,
                                collate_fn=self.data_collator, num_workers=workers,
                                pin_memory=cuda, persistent_workers=False)
            return self.accelerator.prepare(loader)

    trainer = BudgetTrainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[Progress()],
    )
    log("training_start", planned_steps=planned_steps, steps_per_epoch=steps_per_epoch,
        tokens_per_step=memory["max_tokens"] * accumulation, epochs=hp["epochs"],
        gradient_checkpointing=checkpointing)
    result = trainer.train(resume_from_checkpoint=resume)
    packer.wait()
    history = trainer.state.log_history
    eval_losses = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]
    completed = not stop["requested"] or stop["reason"] == "max_train_minutes reached"

    if completed:
        staging = out / ".adapter-staging"
        if staging.exists():
            shutil.rmtree(staging)
        trainer.model.save_pretrained(staging, safe_serialization=True)
        tokenizer.save_pretrained(staging)
        if config.adapter_dir.exists():
            shutil.rmtree(config.adapter_dir)
        os.replace(staging, config.adapter_dir)

    speeds = [r.get("tokens_per_second") for r in _read_jsonl(out / "metrics.jsonl")
              if r.get("tokens_per_second")]
    summary = {
        "completed": completed,
        "stopped_early_reason": stop["reason"],
        "global_step": trainer.state.global_step,
        "planned_steps": planned_steps,
        "epochs_completed": round(trainer.state.epoch or 0, 3),
        "train_runtime_minutes": round(result.metrics.get("train_runtime", 0) / 60, 2),
        "train_loss": result.metrics.get("train_loss"),
        "tokens_seen": trainer.state.num_input_tokens_seen,
        "median_tokens_per_second": sorted(speeds)[len(speeds) // 2] if speeds else None,
        "base_model_validation_loss": eval_losses[0][1] if eval_losses else None,
        "best_validation_loss": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "validation_curve": eval_losses,
        "train_examples": len(train_set),
        "validation_examples": len(val_set),
        "supervised_tokens_per_epoch": train_set.supervised_tokens,
        "trainable_parameters": trainable,
        "batching": memory,
        "device": device,
        "base_model": config.raw["base_model"]["repository"],
        "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)
        if cuda else None,
        "hyperparameters": {"lora": lora, "train": hp, "max_seq_len": max_len},
        "fingerprint": fingerprint,
    }
    write_json(summary_path, summary)
    log("training_done", **{k: summary[k] for k in (
        "completed", "global_step", "base_model_validation_loss", "best_validation_loss",
        "train_runtime_minutes", "median_tokens_per_second")})
    if not completed:
        raise SystemExit(f"Training stopped ({stop['reason']}). Re-run the same command to "
                         "resume from the last checkpoint.")
    return summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, help="override (for smoke tests)")
    parser.add_argument("--smoke", action="store_true", help="tiny CPU run for testing")
    parser.add_argument("--allow-cpu", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = Config.load(args.config) if args.config else Config.load()
    train(config, max_steps=args.max_steps, smoke=args.smoke, allow_cpu=args.allow_cpu)


if __name__ == "__main__":
    main()
