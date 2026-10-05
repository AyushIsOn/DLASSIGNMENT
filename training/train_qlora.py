"""External-only Qwen3 BF16 LoRA training entry point."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path
from typing import Any, cast

from acharya.config import canonical_json, sha256_bytes, sha256_file
from acharya.lightning import (
    LOCKED_REVISION,
    PreflightError,
    QLoRAProfile,
    select_resume_checkpoint,
)


def _rows(path: Path) -> list[dict[str, Any]]:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict) or not isinstance(value.get("messages"), list):
                raise PreflightError("training row does not contain messages")
            values.append(value)
    if not values:
        raise PreflightError("training split is empty")
    return values


def _tokenize_chat_row(
    row: dict[str, Any], tokenizer: Any, maximum_sequence_tokens: int
) -> dict[str, Any]:
    """Tokenize a chat example and compute loss only on the assistant turn.

    The Qwen chat template is deterministic, so tokenizing the same messages
    with an assistant generation prompt gives us the exact prefix to mask.
    """
    messages = row["messages"]
    if (
        not isinstance(messages, list)
        or [item.get("role") for item in messages]
        not in (["user", "assistant"], ["system", "user", "assistant"])
        or any(not str(item.get("content", "")).strip() for item in messages)
    ):
        raise PreflightError("training row needs one nonempty user and assistant turn")
    text = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    input_ids = list(encoded["input_ids"])
    if len(input_ids) > maximum_sequence_tokens:
        raise PreflightError("complete training example exceeds maximum sequence length")
    if input_ids[: len(prompt_ids)] != list(prompt_ids):
        raise PreflightError("chat template generation prefix does not match training text")
    prefix_length = len(prompt_ids)
    labels = [-100] * prefix_length + input_ids[prefix_length:]
    if not any(label != -100 for label in labels):
        raise PreflightError("maximum sequence length removed the assistant answer")
    encoded["labels"] = labels
    return cast(dict[str, Any], encoded)


def _verify_model_snapshot(profile: QLoRAProfile, snapshot: Path) -> None:
    for name, record in profile.value["base_model"]["files"].items():
        path = snapshot / name
        if not path.is_file() or path.stat().st_size != int(record["size"]):
            raise PreflightError(f"locked model file missing:{name}")
        if sha256_file(path) != record["sha256"]:
            raise PreflightError(f"locked model file hash mismatch:{name}")


def acquire_model(workspace: Path) -> dict[str, object]:
    """Download then hash-verify the locked snapshot without loading model code."""
    profile = QLoRAProfile.load(workspace)
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            profile.value["base_model"]["repository"],
            revision=LOCKED_REVISION,
            local_dir=workspace / "artifacts" / "model-cache" / LOCKED_REVISION,
        )
    )
    _verify_model_snapshot(profile, snapshot)
    return {
        "snapshot": str(snapshot),
        "revision": LOCKED_REVISION,
        "files": {
            name: sha256_file(snapshot / name)
            for name in sorted(profile.value["base_model"]["files"])
        },
    }


def training_arguments(output: Path, profile: QLoRAProfile, max_steps: int) -> dict[str, Any]:
    """Shared argument contract for the GPU runner and CPU integration smoke."""
    return dict(
        output_dir=str(output),
        max_steps=max_steps,
        per_device_train_batch_size=int(profile.training["micro_batch_size"]),
        per_device_eval_batch_size=int(profile.training["micro_batch_size"]),
        gradient_accumulation_steps=int(profile.training["gradient_accumulation_steps"]),
        learning_rate=float(profile.training["learning_rate"]),
        lr_scheduler_type=str(profile.training["scheduler"]),
        warmup_ratio=float(profile.training["warmup_ratio"]),
        optim=str(profile.training["optimizer"]),
        bf16=True,
        gradient_checkpointing=True,
        save_steps=int(profile.training["checkpoint_steps"]),
        eval_strategy="steps",
        eval_steps=int(profile.training["checkpoint_steps"]),
        prediction_loss_only=True,
        logging_steps=1,
        report_to=[],
        save_total_limit=int(profile.training["save_total_limit"]),
        load_best_model_at_end=True,
        restore_callback_states_from_checkpoint=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_first_step=True,
        log_level="info",
        disable_tqdm=False,
        save_safetensors=True,
        seed=int(profile.training["seed"]),
    )


def train(
    workspace: Path, output: Path, max_steps: int, quality_data: Path | None = None
) -> dict[str, object]:
    profile = QLoRAProfile.load(workspace)
    if max_steps < 1:
        raise PreflightError("max steps must be positive")
    pointer = json.loads(
        (workspace / "artifacts" / "state" / "active_preparation.json").read_text(encoding="utf-8")
    )
    processed = Path(str(pointer["processed_path"]))
    if not processed.is_absolute():
        processed = workspace / processed
    processed = processed.resolve()
    quality_manifest_hash = None
    if quality_data is None:
        raise PreflightError(
            "Legacy copy-target training is disabled. Supply --quality-data with reviewed QA."
        )
    processed = quality_data.resolve()
    manifest_path = processed / "QUALITY_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    benchmark = Path(__file__).resolve().parents[1] / "eval/independent_quality.jsonl"
    if not manifest.get("training_ready") or manifest.get("benchmark_sha256") != sha256_file(
        benchmark
    ):
        raise PreflightError("quality data is not approved or benchmark binding differs")
    for split in ("train", "validation", "test"):
        name = f"{split}.jsonl"
        if manifest.get("files", {}).get(name) != sha256_file(processed / name):
            raise PreflightError("quality split hash mismatch")
    evidence = processed / "REVIEW_EVIDENCE.json"
    if manifest.get("files", {}).get(evidence.name) != sha256_file(evidence):
        raise PreflightError("review evidence hash mismatch")
    quality_manifest_hash = sha256_file(manifest_path)
    train_rows = _rows(processed / "train.jsonl")
    validation_rows = _rows(processed / "validation.jsonl")
    test_rows = _rows(processed / "test.jsonl")
    if len(train_rows) < 500 or len(validation_rows) < 100 or len(test_rows) < 100:
        raise PreflightError("quality dataset is below minimum split sizes")
    from training.quality_data import audit_row

    if any(audit_row(row) for row in (*train_rows, *validation_rows, *test_rows)):
        raise PreflightError("quality dataset failed copy/OCR audit")
    run_input = {
        "qlora_sha256": profile.config_sha256,
        "preparation_fingerprint": pointer["fingerprint"],
        "train_sha256": sha256_file(processed / "train.jsonl"),
        "validation_sha256": sha256_file(processed / "validation.jsonl"),
        "planned_max_steps": max_steps,
        "quality_manifest_sha256": quality_manifest_hash,
        "training_code_sha256": sha256_file(Path(__file__)),
    }
    run_sha256 = sha256_bytes(canonical_json(run_input))

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Trainer,
        TrainerCallback,
        TrainerState,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise PreflightError("real BF16 LoRA training requires CUDA")
    snapshot = workspace / "artifacts" / "model-cache" / LOCKED_REVISION
    _verify_model_snapshot(profile, snapshot)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        local_files_only=True,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    model.enable_input_require_grads()
    lora = profile.value["lora"]
    peft_config = LoraConfig(
        r=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        bias=str(lora["bias"]),
        task_type="CAUSAL_LM",
        target_modules=list(lora["target_modules"]),
    )
    from peft import get_peft_model

    model = get_peft_model(model, peft_config)
    model.config.use_cache = False
    model.peft_config["default"].base_model_name_or_path = profile.value["base_model"]["repository"]
    model.peft_config["default"].revision = LOCKED_REVISION

    def tokenize(row: dict[str, Any]) -> dict[str, Any]:
        return _tokenize_chat_row(row, tokenizer, int(profile.training["maximum_sequence_tokens"]))

    dataset = Dataset.from_list(train_rows).map(tokenize, remove_columns=list(train_rows[0]))
    validation_dataset = Dataset.from_list(validation_rows).map(
        tokenize, remove_columns=list(validation_rows[0])
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "RUN_INPUT.json").write_bytes(
        canonical_json({**run_input, "run_sha256": run_sha256}) + b"\n"
    )
    arguments = TrainingArguments(**training_arguments(output, profile, max_steps))
    resume = select_resume_checkpoint(output, run_sha256)

    from acharya.checkpoints import pack, seal

    stop_requested = False
    started = time.monotonic()
    deadline_seconds = float(os.environ.get("ACHARYA_TRAIN_SECONDS", "21600"))
    if "ACHARYA_TRAIN_DEADLINE_UNIX" in os.environ:
        deadline_seconds = min(
            deadline_seconds,
            max(0.0, float(os.environ["ACHARYA_TRAIN_DEADLINE_UNIX"]) - time.time()),
        )

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    class BindCheckpointCallback(TrainerCallback):
        last_saved = time.monotonic()

        def on_step_end(self, _args: Any, _state: Any, control: Any, **_kwargs: Any) -> None:
            nonlocal stop_requested
            if time.monotonic() - started >= deadline_seconds:
                stop_requested = True
            if stop_requested or time.monotonic() - self.last_saved >= 180:
                control.should_save = True
            if stop_requested:
                control.should_training_stop = True

        def on_log(
            self, _args: Any, state: Any, _control: Any, logs: Any = None, **_kwargs: Any
        ) -> None:
            event = {
                "step": state.global_step,
                "elapsed_seconds": time.monotonic() - started,
                "cuda_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                "cuda_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                **(logs or {}),
            }
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)

        def on_save(self, _args: Any, state: Any, _control: Any, **_kwargs: Any) -> None:
            checkpoint = output / f"checkpoint-{int(state.global_step)}"
            if checkpoint.is_dir():
                (checkpoint / "RUN_INPUT.json").write_bytes(
                    canonical_json({**run_input, "run_sha256": run_sha256}) + b"\n"
                )

                seal(checkpoint)
                recovery = pack(workspace, checkpoint, output / "recovery-latest.tar")
                print(json.dumps({"recovery_checkpoint": recovery}), flush=True)
                self.last_saved = time.monotonic()

    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8
        ),
        processing_class=tokenizer,
        callbacks=[
            BindCheckpointCallback(),
            EarlyStoppingCallback(
                early_stopping_patience=int(profile.training["early_stopping_patience"])
            ),
        ],
    )
    if resume is None and any(output.glob("checkpoint-*")):
        raise PreflightError(
            "existing checkpoints do not match or are incomplete; use a new run folder"
        )
    print(
        json.dumps(
            {
                "resume_from": str(resume) if resume else None,
                "target_steps": max_steps,
                "train_rows": len(train_rows),
                "method": "bf16_lora",
                "trainable_parameters": model.get_nb_trainable_parameters(),
            }
        ),
        flush=True,
    )
    completed = False
    if resume:
        saved = json.loads((resume / "trainer_state.json").read_text())
        early = saved.get("stateful_callbacks", {}).get("EarlyStoppingCallback", {})
        patience_used = early.get("attributes", {}).get("early_stopping_patience_counter", 0)
        completed = int(saved["global_step"]) >= max_steps or int(patience_used) >= int(
            profile.training["early_stopping_patience"]
        )
    if completed:
        # A transfer after the final optimizer step must not perform an extra step.
        trainer.state = TrainerState.load_from_json(str(resume / "trainer_state.json"))
        trainer._load_from_checkpoint(str(resume))
        if trainer.state.best_model_checkpoint:
            trainer._load_best_model()
        metrics = {"resumed_completed_checkpoint": True}
    else:
        result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        metrics = result.metrics
    (output / "train_metrics.json").write_bytes(
        canonical_json(
            {
                **metrics,
                "global_step": trainer.state.global_step,
                "paused": stop_requested,
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "best_validation_loss": trainer.state.best_metric,
            }
        )
        + b"\n"
    )
    if stop_requested:
        raise SystemExit(75)
    trainer.save_state()
    model.save_pretrained(output / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(output / "adapter")
    return {"run_sha256": run_sha256, "global_step": trainer.state.global_step}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--acquire-only", action="store_true")
    parser.add_argument("--quality-data", type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if args.acquire_only:
        result = acquire_model(workspace)
    else:
        if args.output is None or args.max_steps is None:
            parser.error("--output and --max-steps are required for training")
        result = train(workspace, args.output.resolve(), args.max_steps, args.quality_data)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
