"""Turn training logs + evaluation results into charts and a Markdown report.

    python -m finetune.report        # CPU only, a few seconds

Writes artifacts/run/report/: REPORT.md, loss_curves.png, accuracy_by_group.png,
accuracy_by_attribute.png, test_loss.png, summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from finetune.common import Config, write_json

COLORS = {"base": "#94a3b8", "finetuned": "#0f766e"}
LABELS = {"base": "Qwen3-8B (base)", "finetuned": "AcharyaGPT (fine-tuned)"}


def read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pct(value: float | None) -> str:
    return "–" if value is None else f"{100 * value:.1f}%"


def plot_losses(records: list[dict[str, Any]], path: Path) -> bool:
    import matplotlib.pyplot as plt

    train = [(r["step"], r["loss"]) for r in records if "loss" in r and "eval_loss" not in r]
    evals = [(r["step"], r["eval_loss"]) for r in records if "eval_loss" in r]
    if not train and not evals:
        return False
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if train:
        ax.plot(*zip(*train, strict=True), color="#64748b", alpha=0.8, lw=1.2,
                label="training loss")
    if evals:
        ax.plot(*zip(*evals, strict=True), color=COLORS["finetuned"], marker="o", lw=2,
                label="validation loss")
        ax.annotate(f"base model: {evals[0][1]:.3f}", evals[0], textcoords="offset points",
                    xytext=(10, 8), fontsize=9)
        best = min(evals, key=lambda item: item[1])
        ax.annotate(f"best: {best[1]:.3f}", best, textcoords="offset points",
                    xytext=(-10, 12), fontsize=9, ha="right")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy loss (answer tokens)")
    ax.set_title("AcharyaGPT LoRA fine-tuning of Qwen3-8B")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def bar_chart(labels: list[str], base: list[float], tuned: list[float], title: str,
              path: Path, counts: list[int] | None = None) -> None:
    import matplotlib.pyplot as plt

    width = 0.38
    positions = range(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, 1.25 * len(labels)), 5))
    for offset, values, name in ((-width / 2, base, "base"), (width / 2, tuned, "finetuned")):
        bars = ax.bar([p + offset for p in positions], [100 * v for v in values], width,
                      color=COLORS[name], label=LABELS[name])
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{100 * value:.0f}",
                    ha="center", fontsize=8)
    names = [f"{label}\n(n={count})" for label, count in zip(labels, counts, strict=True)] \
        if counts else labels
    ax.set_xticks(list(positions), names, fontsize=8)
    ax.set_ylim(0, 110)
    ax.set_ylabel("answers judged correct (%)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build(config: Config) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    out = config.output_dir / "report"
    out.mkdir(parents=True, exist_ok=True)
    training = json.loads((config.output_dir / "training_summary.json").read_text()) \
        if (config.output_dir / "training_summary.json").is_file() else {}
    results_path = config.eval_dir / "results.json"
    if not results_path.is_file():
        raise SystemExit(f"{results_path} not found: run `python -m finetune.evaluate` first")
    results = json.loads(results_path.read_text())
    records = read_metrics(config.output_dir / "metrics.jsonl")
    has_curve = plot_losses(records, out / "loss_curves.png")

    groups = [g for g in ("seen_closed", "heldout_open", "concepts", "safety", "heldout_closed")
              if g in results["groups"]]
    short = {"seen_closed": "Knowledge recall\n(closed book)", "heldout_open":
             "Unseen + RAG\n(open book)", "concepts": "Ayurveda\nconcepts",
             "safety": "Safety", "heldout_closed": "Unseen, no RAG\n(control)"}
    bar_chart([short[g] for g in groups],
              [results["groups"][g]["base"]["accuracy"] for g in groups],
              [results["groups"][g]["finetuned"]["accuracy"] for g in groups],
              "Held-out test set: base vs fine-tuned", out / "accuracy_by_group.png",
              [results["groups"][g]["base"]["n"] for g in groups])
    attrs = sorted(a for a in results["attributes"] if a.startswith("seen_closed:")
                   and results["attributes"][a]["base"]["n"] >= 8)
    bar_chart([a.split(":", 1)[1].replace("_", " ") for a in attrs],
              [results["attributes"][a]["base"]["accuracy"] for a in attrs],
              [results["attributes"][a]["finetuned"]["accuracy"] for a in attrs],
              "Knowledge recall by question type (question wording never seen in training)",
              out / "accuracy_by_attribute.png",
              [results["attributes"][a]["base"]["n"] for a in attrs])

    import matplotlib.pyplot as plt

    losses = results["test_answer_loss"]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar([LABELS[m] for m in ("base", "finetuned")],
                  [losses[m]["perplexity"] for m in ("base", "finetuned")],
                  color=[COLORS["base"], COLORS["finetuned"]])
    for bar, model in zip(bars, ("base", "finetuned"), strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"ppl {losses[model]['perplexity']:.2f}\nloss {losses[model]['loss']:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("perplexity of the reference answers")
    ax.set_title("Test-set perplexity (lower is better)")
    ax.set_ylim(0, max(losses[m]["perplexity"] for m in losses) * 1.3)
    fig.tight_layout()
    fig.savefig(out / "test_loss.png", dpi=150)
    plt.close(fig)

    overall = results["overall_excluding_control"]
    lines = [
        "# AcharyaGPT fine-tuning report",
        "",
        "Base model: **Qwen/Qwen3-8B** (thinking disabled) · method: **LoRA** "
        f"(r={config.lora['r']}, alpha={config.lora['alpha']}, all attention + MLP projections)"
        f" · trainable parameters: **{training.get('trainable_parameters', 0):,}**",
        "",
        "## Headline",
        "",
        "| | Base Qwen3-8B | Fine-tuned AcharyaGPT |",
        "|---|---:|---:|",
        f"| Accuracy, all test questions (excl. control) | {pct(overall['base']['accuracy'])} | "
        f"**{pct(overall['finetuned']['accuracy'])}** |",
        f"| Test perplexity of reference answers | {losses['base']['perplexity']:.2f} | "
        f"**{losses['finetuned']['perplexity']:.2f}** |",
        f"| Mean answer length (words) | {overall['base']['mean_words']} | "
        f"{overall['finetuned']['mean_words']} |",
        "",
        "## Results by test group",
        "",
        "| Test group | n | Base | Fine-tuned | Δ |",
        "|---|---:|---:|---:|---:|",
    ]
    for g in groups:
        b, f = results["groups"][g]["base"], results["groups"][g]["finetuned"]
        lines.append(f"| {results['group_labels'][g]} | {b['n']} | {pct(b['accuracy'])} | "
                     f"**{pct(f['accuracy'])}** | {100 * (f['accuracy'] - b['accuracy']):+.1f} |")
    lines += ["", "![accuracy by group](accuracy_by_group.png)", "",
              "## Knowledge recall by question type", "",
              "| Question type | n | Base | Fine-tuned |", "|---|---:|---:|---:|"]
    for a in attrs:
        b, f = results["attributes"][a]["base"], results["attributes"][a]["finetuned"]
        lines.append(f"| {a.split(':', 1)[1]} | {b['n']} | {pct(b['accuracy'])} | "
                     f"**{pct(f['accuracy'])}** |")
    lines += ["", "![accuracy by attribute](accuracy_by_attribute.png)", ""]
    if has_curve:
        lines += ["## Training", "",
                  f"* steps: {training.get('global_step')} of {training.get('planned_steps')} "
                  f"({training.get('epochs_completed')} epochs) in "
                  f"{training.get('train_runtime_minutes')} min on "
                  f"{training.get('device', {}).get('gpu', 'CPU')}",
                  f"* validation loss: {training.get('base_model_validation_loss'):.4f} "
                  f"(base, step 0) → {training.get('best_validation_loss'):.4f} (best)"
                  if training.get("best_validation_loss") else "* validation loss: n/a",
                  f"* peak GPU memory: {training.get('peak_gpu_memory_gib')} GiB", "",
                  "![loss curves](loss_curves.png)", ""]
    lines += ["![test perplexity](test_loss.png)", "", "## Example answers", ""]
    for example in results.get("examples", []):
        lines += [f"**Q ({example['group']}):** {example['question']}", "",
                  f"* reference: {example['reference']}",
                  f"* base ({'✓' if example['base']['correct'] else '✗'}): "
                  f"{example['base']['answer'][:500]}",
                  f"* fine-tuned ({'✓' if example['finetuned']['correct'] else '✗'}): "
                  f"{example['finetuned']['answer'][:500]}", ""]
    lines += [
        "## How to read this", "",
        "* **Knowledge recall** asks about conditions from the Kaggle Ayurveda tables using "
        "question wordings that never appear in training; it measures whether the facts "
        "were learned rather than memorised sentences.",
        "* **Unseen + RAG** uses conditions the model never trained on, with three BM25-retrieved "
        "knowledge-base entries in the prompt (as the app does).",
        "* **Control** asks about the same unseen conditions without retrieval; neither model "
        "can know these dataset-specific facts, so low scores here are expected.",
        "* An answer is *correct* when it states the gold fact (dosha set, modern equivalent, "
        "prognosis class, classical text, ≥50% of listed symptoms…), independent of wording; "
        "see `finetune/metrics.py`. The facts come from community Kaggle datasets and are "
        "educational, not clinically validated.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    summary = {"overall": overall, "groups": results["groups"], "test_answer_loss": losses,
               "training": {k: training.get(k) for k in (
                   "global_step", "epochs_completed", "train_runtime_minutes",
                   "base_model_validation_loss", "best_validation_loss", "peak_gpu_memory_gib")}}
    write_json(out / "summary.json", summary)
    print((out / "REPORT.md").read_text()[:3000])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    build(Config.load(args.config) if args.config else Config.load())


if __name__ == "__main__":
    main()
