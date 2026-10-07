"""LoRA fine-tuning of Qwen3-8B on the AcharyaGPT dataset (single GPU, bf16).

    python -m finetune.train                      # uses configs/train.yaml

Safe to re-run: an interrupted run resumes from its last checkpoint, a finished
run is skipped. Writes to artifacts/run/:
    adapter/                  final LoRA adapter (best validation loss)
    checkpoints/              intermediate checkpoints (optimizer state etc.)
    metrics.jsonl             every logged step (loss, lr, grad norm, eval loss, memory)
    training_summary.json     what happened, for the report
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
from finetune.data import Collator, SFTDataset, read_jsonl, verify_manifest

ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def adapter_complete(path: Path) -> bool:
    return all((path / name).is_file() and (path / name).stat().st_size > 0
               for name in ADAPTER_FILES)


def run_fingerprint(config: Config, manifest: dict[str, str]) -> str:
    """Identity of a run: checkpoints from a different config/data are never resumed."""
    value = json.dumps({"config": config.raw, "data": manifest}, sort_keys=True)
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def check_model_dir(model_dir: Path, config: Config) -> None:
    missing = [name for name in config.raw["base_model"]["shards"]
               if not (model_dir / name).is_file()]
    if missing or not (model_dir / "config.json").is_file():
        raise SystemExit(
            f"Base model not found in {model_dir} (missing {missing or ['config.json']}). "
            "Run `bash scripts/lightning_cpu_setup.sh` first (CPU, no GPU cost)."
        )
    if config.model_dir.name == "Qwen3-8B":
        from finetune.download_model import verified

        if not verified(model_dir, config):
            raise SystemExit(f"{model_dir} is not verified: run "
                             "`python -m finetune.download_model` (CPU) first.")


def train(config: Config, *, max_steps: int | None = None, smoke: bool = False,
          allow_cpu: bool = False) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainerCallback, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint

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
    run_info = out / "RUN_INFO.json"
    if (run_info.is_file()
            and json.loads(run_info.read_text())["fingerprint"] != fingerprint
            and any(checkpoints.glob("checkpoint-*"))):
        raise SystemExit(
            f"{checkpoints} holds checkpoints from a different config or dataset. "
            f"Move {out} away (or set ACHARYA_RUN_DIR) before training again."
        )
    out.mkdir(parents=True, exist_ok=True)
    write_json(run_info, {"fingerprint": fingerprint, "config": config.raw})

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
    train_rows = read_jsonl(config.data_dir / "train.jsonl")
    val_rows = read_jsonl(config.data_dir / "validation.jsonl")
    if smoke:
        train_rows, val_rows = train_rows[:64], val_rows[:16]
    train_set = SFTDataset(train_rows, tokenizer, max_len)
    val_set = SFTDataset(val_rows, tokenizer, max_len)
    if train_set.dropped or val_set.dropped:
        raise SystemExit(f"{train_set.dropped + val_set.dropped} examples exceed max_seq_len")
    log("data", train_examples=len(train_set), train_tokens=train_set.tokens,
        supervised_tokens=train_set.supervised_tokens, validation_examples=len(val_set))

    model = load_base_model(config.model_dir, for_training=True)
    lora = config.lora
    model = get_peft_model(model, LoraConfig(
        r=int(lora["r"]), lora_alpha=int(lora["alpha"]), lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]), bias="none", task_type="CAUSAL_LM",
    ))
    if hp["gradient_checkpointing"]:
        model.enable_input_require_grads()  # makes checkpointing safe with a frozen base
    trainable, total = model.get_nb_trainable_parameters()
    log("lora", trainable_parameters=trainable, total_parameters=total,
        trainable_percent=round(100 * trainable / total, 3))

    micro = int(hp["micro_batch_size"])
    accumulation = int(hp["gradient_accumulation"])
    eval_every = int(hp["eval_every_steps"])
    steps_per_epoch = math.ceil(len(train_set) / (micro * accumulation))
    planned_steps = max_steps or math.ceil(steps_per_epoch * float(hp["epochs"]))
    if smoke:
        eval_every = max(1, planned_steps // 2)
    args = TrainingArguments(
        output_dir=str(checkpoints),
        num_train_epochs=float(hp["epochs"]),
        max_steps=max_steps or -1,
        per_device_train_batch_size=micro,
        per_device_eval_batch_size=micro,
        gradient_accumulation_steps=accumulation,
        learning_rate=float(hp["learning_rate"]),
        lr_scheduler_type=str(hp["lr_scheduler"]),
        warmup_ratio=float(hp["warmup_ratio"]),
        weight_decay=float(hp["weight_decay"]),
        max_grad_norm=float(hp["max_grad_norm"]),
        optim="adamw_torch",
        bf16=bf16,
        tf32=True if torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 8 else None,
        gradient_checkpointing=bool(hp["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,  # also puts the longest batch first, so OOM shows up at step 1
        eval_strategy="steps",
        eval_steps=eval_every,
        eval_on_start=True,  # step-0 validation loss == the untouched base model
        save_strategy="steps",
        save_steps=eval_every,
        save_total_limit=int(hp["keep_checkpoints"]),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=int(hp["logging_steps"]),
        logging_first_step=True,
        report_to="none",
        seed=int(hp["seed"]),
        data_seed=int(hp["seed"]),
        remove_unused_columns=False,
        dataloader_num_workers=0,
        save_safetensors=True,
        include_num_input_tokens_seen=True,
        disable_tqdm=False,
        use_cpu=not torch.cuda.is_available(),
    )

    started = time.time()
    limit_seconds = float(hp["max_train_minutes"]) * 60
    stop = {"requested": False, "reason": None}

    def on_signal(signum: int, _frame: Any) -> None:
        stop.update(requested=True, reason=f"signal {signum}")
        log("stop_requested", signal=signum, note="saving a checkpoint, then stopping")

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    class Progress(TrainerCallback):
        def on_step_end(self, _args: Any, state: Any, control: Any, **_: Any) -> None:
            if time.time() - started > limit_seconds and not stop["requested"]:
                stop.update(requested=True, reason="max_train_minutes reached")
                log("time_limit", minutes=hp["max_train_minutes"])
            if stop["requested"]:
                control.should_save = True
                control.should_training_stop = True
            if state.global_step == 20:
                elapsed = time.time() - started
                per_step = elapsed / 20
                log("eta", seconds_per_step=round(per_step, 2), planned_steps=planned_steps,
                    estimated_minutes=round(per_step * planned_steps / 60, 1))

        def on_log(self, _args: Any, state: Any, _control: Any, logs: Any = None,
                   **_: Any) -> None:
            if not logs:
                return
            record = {"step": state.global_step, "epoch": round(state.epoch or 0, 4),
                      "elapsed_s": round(time.time() - started, 1), **logs}
            if torch.cuda.is_available():
                record["gpu_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            append_jsonl(out / "metrics.jsonl", record)

    resume = None
    if checkpoints.is_dir():
        last = get_last_checkpoint(str(checkpoints))
        if last and (Path(last) / "trainer_state.json").is_file():
            resume = last
    if resume:
        log("resuming", checkpoint=resume)
    elif (out / "metrics.jsonl").exists():
        (out / "metrics.jsonl").unlink()  # fresh run: start the curve from scratch

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=Collator(tokenizer.pad_token_id),
        processing_class=tokenizer,
        callbacks=[Progress()],
    )
    log("training_start", planned_steps=planned_steps, steps_per_epoch=steps_per_epoch,
        effective_batch=micro * accumulation, epochs=hp["epochs"])
    result = trainer.train(resume_from_checkpoint=resume)
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

    summary = {
        "completed": completed,
        "stopped_early_reason": stop["reason"],
        "global_step": trainer.state.global_step,
        "planned_steps": planned_steps,
        "epochs_completed": round(trainer.state.epoch or 0, 3),
        "train_runtime_minutes": round(result.metrics.get("train_runtime", 0) / 60, 2),
        "train_loss": result.metrics.get("train_loss"),
        "tokens_seen": trainer.state.num_input_tokens_seen,
        "base_model_validation_loss": eval_losses[0][1] if eval_losses else None,
        "best_validation_loss": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "validation_curve": eval_losses,
        "train_examples": len(train_set),
        "validation_examples": len(val_set),
        "supervised_tokens_per_epoch": train_set.supervised_tokens,
        "trainable_parameters": trainable,
        "device": device,
        "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)
        if torch.cuda.is_available() else None,
        "hyperparameters": {"lora": lora, "train": hp, "max_seq_len": max_len},
        "fingerprint": fingerprint,
    }
    write_json(summary_path, summary)
    log("training_done", **{k: summary[k] for k in (
        "completed", "global_step", "base_model_validation_loss", "best_validation_loss",
        "train_runtime_minutes")})
    if not completed:
        raise SystemExit(f"Training stopped ({stop['reason']}). Re-run the same command to "
                         "resume from the last checkpoint.")
    return summary


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
