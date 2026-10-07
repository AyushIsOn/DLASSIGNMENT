"""End-to-end pipeline smoke test on CPU (≈2 min, no GPU, no 8B weights in RAM).

    python -m finetune.smoke

Builds a tiny random Qwen3 model with the REAL Qwen3-8B tokenizer and chat template,
then runs the same train -> evaluate -> report -> merge code the GPU run uses.
It proves the environment works; the scores it prints are meaningless.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from finetune.common import ROOT, Config, log


def tiny_model(tokenizer_dir: Path, output: Path) -> Path:
    import torch
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    if (output / "config.json").is_file():
        return output
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    config = Qwen3Config(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=16, max_position_embeddings=2048, tie_word_embeddings=False,
                         pad_token_id=tokenizer.pad_token_id, eos_token_id=151645)
    model = Qwen3ForCausalLM(config)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    return output


def run() -> None:
    from finetune import evaluate, export, report, train

    base = Config.load()
    work = ROOT / "artifacts" / "smoke"
    if (work / "run").exists():
        shutil.rmtree(work / "run")
    model_dir = tiny_model(base.model_dir, work / "tiny-qwen3")
    raw = json.loads(json.dumps(base.raw))
    raw["base_model"].update(local_dir=str(model_dir), shards={"model.safetensors": "-"})
    raw["output_dir"] = str(work / "run")
    raw["train"].update(micro_batch_size=4, gradient_accumulation=1, eval_every_steps=2,
                        logging_steps=1, keep_checkpoints=1)
    raw["evaluation"].update(batch_size=8, max_new_tokens=8, loss_batch_size=4)
    (work / "config.yaml").write_text(yaml.safe_dump(raw))
    config = Config(raw)

    summary = train.train(config, max_steps=4, smoke=True)
    assert summary["completed"] and summary["global_step"] == 4, summary
    results = evaluate.evaluate(config, config.adapter_dir, limit=10, smoke=True)
    assert results["test_rows"] >= 5, results["test_rows"]
    report.build(config)
    merged = export.merge(config, config.adapter_dir, config.output_dir / "merged")
    assert (merged / "config.json").is_file()
    log("SMOKE TEST PASSED", note="train, evaluate, report and merge all work on this machine")


if __name__ == "__main__":
    run()
