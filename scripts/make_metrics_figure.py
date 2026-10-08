"""One image with every evaluation metric of a run (from its real results files).

    python scripts/make_metrics_figure.py --run-dir artifacts/run      # round 1
    python scripts/make_metrics_figure.py                              # run of configs/train.yaml

Reads <run>/eval/results.json, <run>/eval/generations.jsonl, <run>/training_summary.json and
<run>/RUN_INFO.json; writes <run>/report/metrics_summary.png. Nothing is estimated or edited:
every number is read from those files (95% intervals and McNemar p-values are computed from
the per-question results).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

ORDER = ["seen_closed", "heldout_open", "unseen_closed", "concepts", "safety", "heldout_closed"]
NAMES = {
    "seen_closed": "Knowledge recall\n(new wording)",
    "heldout_open": "Unseen conditions\n+ retrieval",
    "unseen_closed": "Unseen terms\n(no retrieval)",
    "concepts": "Ayurveda\nconcepts",
    "safety": "Safety",
    "heldout_closed": "Unseen conditions\nno retrieval",
}
BASE, TUNED = "#94a3b8", "#0f766e"


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_p(only_tuned: int, only_base: int) -> float:
    n = only_tuned + only_base
    if n == 0:
        return 1.0
    k = min(only_tuned, only_base)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def group_of(row_id: str) -> str:
    """Test group from a test-row id (same scheme in round 1 and round 2)."""
    if row_id.startswith("concept-"):
        return "concepts"
    if row_id.startswith("safety-"):
        return "safety"
    if row_id.endswith("|open"):
        return "heldout_open"
    if row_id.endswith("|closed"):
        return "unseen_closed" if row_id.startswith(("nm-", "sat-")) else "heldout_closed"
    return "seen_closed"


def paired(run: Path) -> dict[str, tuple[int, int]]:
    """group -> (only fine-tuned correct, only base correct)."""
    path = run / "eval" / "generations.jsonl"
    if not path.is_file():
        return {}
    correct: dict[tuple[str, str], bool] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            correct[(record["id"], record["model"])] = bool(record["metrics"]["correct"])
    counts: dict[str, list[int]] = {}
    for (row_id, model), _ in correct.items():
        if model != "finetuned" or (row_id, "base") not in correct:
            continue
        tuned, base = correct[(row_id, "finetuned")], correct[(row_id, "base")]
        entry = counts.setdefault(group_of(row_id), [0, 0, 0])
        entry[0] += tuned and not base
        entry[1] += base and not tuned
        entry[2] += 1
    return {g: (c[0], c[1], c[2]) for g, c in counts.items()}


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.run_dir is None:
        from finetune.common import Config

        args.run_dir = Config.load().output_dir
    run = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    results = json.loads((run / "eval" / "results.json").read_text())
    training = (
        json.loads((run / "training_summary.json").read_text())
        if (run / "training_summary.json").is_file()
        else {}
    )
    info = (
        json.loads((run / "RUN_INFO.json").read_text()) if (run / "RUN_INFO.json").is_file() else {}
    )
    model = (
        training.get("base_model")
        or info.get("config", {}).get("base_model", {}).get("repository", "base model")
    ).split("/")[-1]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [g for g in ORDER if g in results["groups"]]
    pairs = paired(run)
    rows = []
    for g in groups:
        b, f = results["groups"][g]["base"], results["groups"][g]["finetuned"]
        n = b["n"]
        rows.append(
            {
                "g": g,
                "n": n,
                "b": b,
                "f": f,
                "b_ci": wilson(round(b["accuracy"] * n), n),
                "f_ci": wilson(round(f["accuracy"] * n), n),
                "p": mcnemar_p(*pairs[g][:2]) if g in pairs and pairs[g][2] == n else None,
            }
        )

    fig = plt.figure(figsize=(17, 10.2), facecolor="white")
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.2, 0.8],
        width_ratios=[1.35, 1.35, 1],
        hspace=0.42,
        wspace=0.28,
        left=0.05,
        right=0.97,
        top=0.87,
        bottom=0.09,
    )
    fig.suptitle(
        f"AcharyaGPT - evaluation results ({model} base vs LoRA fine-tuned)",
        fontsize=19,
        fontweight="bold",
        x=0.5,
        y=0.975,
    )
    fig.text(
        0.5,
        0.925,
        f"{results['test_rows']} held-out test questions  ·  same prompts for "
        "both models  ·  greedy decoding  ·  error bars = 95% Wilson confidence interval",
        ha="center",
        fontsize=11,
        color="#475569",
    )

    # A: accuracy per group with CIs
    ax = fig.add_subplot(grid[0, :2])
    x = range(len(rows))
    w = 0.38
    for offset, key, ci, color, label in (
        (-w / 2, "b", "b_ci", BASE, f"{model} (base)"),
        (w / 2, "f", "f_ci", TUNED, "AcharyaGPT (fine-tuned)"),
    ):
        values = [100 * r[key]["accuracy"] for r in rows]
        lower = [v - 100 * r[ci][0] for v, r in zip(values, rows, strict=True)]
        upper = [100 * r[ci][1] - v for v, r in zip(values, rows, strict=True)]
        bars = ax.bar(
            [i + offset for i in x],
            values,
            w,
            color=color,
            label=label,
            yerr=[lower, upper],
            capsize=3,
            error_kw={"lw": 1, "ecolor": "#334155"},
        )
        for bar, v in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                2,
                f"{v:.0f}",
                ha="center",
                fontsize=10,
                color="white",
                fontweight="bold",
            )
    ax.set_xticks(list(x), [f"{NAMES[r['g']]}\n(n={r['n']})" for r in rows], fontsize=10)
    ax.set_ylim(0, 112)
    ax.set_ylabel("answers stating the correct fact (%)")
    ax.set_title("Accuracy by test group", fontsize=13, fontweight="bold", loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # B: language-model metrics
    ax = fig.add_subplot(grid[0, 2])
    losses = results["test_answer_loss"]
    ppl = [losses["base"]["perplexity"], losses["finetuned"]["perplexity"]]
    bars = ax.bar(["base", "fine-tuned"], ppl, color=[BASE, TUNED], width=0.55)
    for bar, model_key in zip(bars, ("base", "finetuned"), strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.03,
            f"perplexity {losses[model_key]['perplexity']:.2f}\n"
            f"loss {losses[model_key]['loss']:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylim(0, max(ppl) * 1.45)
    ax.set_title(
        "Test-set perplexity of the\nreference answers (lower = better)",
        fontsize=13,
        fontweight="bold",
        loc="left",
    )
    ax.spines[["top", "right"]].set_visible(False)
    notes = []
    if training.get("best_validation_loss") is not None:
        notes.append(
            f"validation loss {training['base_model_validation_loss']:.3f} → "
            f"{training['best_validation_loss']:.3f}"
        )
    if training.get("global_step"):
        notes.append(
            f"{training['global_step']} steps · {training.get('train_runtime_minutes')} min"
            f" · {training.get('device', {}).get('gpu', 'GPU')}"
        )
    if training.get("trainable_parameters"):
        notes.append(f"{training['trainable_parameters'] / 1e6:.0f}M trainable LoRA parameters")
    ax.text(
        0.5,
        -0.16,
        "\n".join(notes),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color="#334155",
    )

    # C: full metrics table
    ax = fig.add_subplot(grid[1, :])
    ax.axis("off")
    ax.set_title(
        "All metrics per test group (base → fine-tuned)", fontsize=13, fontweight="bold", loc="left"
    )
    header = [
        "Test group",
        "n",
        "Accuracy  [fine-tuned 95% CI]",
        "Fact score",
        "Token F1",
        "ROUGE-L",
        "McNemar p",
    ]
    cells = []
    for r in rows:
        b, f = r["b"], r["f"]
        p_value = r["p"]
        cells.append(
            [
                NAMES[r["g"]].replace("\n", " "),
                str(r["n"]),
                f"{pct(b['accuracy'])} → {pct(f['accuracy'])}  "
                f"[{pct(r['f_ci'][0])}–{pct(r['f_ci'][1])}]",
                f"{b['fact_score']:.2f} → {f['fact_score']:.2f}",
                f"{b['token_f1']:.2f} → {f['token_f1']:.2f}",
                f"{b['rouge_l']:.2f} → {f['rouge_l']:.2f}",
                "–" if p_value is None else ("< 0.001" if p_value < 0.001 else f"{p_value:.3f}"),
            ]
        )
    overall = results.get("overall_excluding_control")
    if overall:
        b, f = overall["base"], overall["finetuned"]
        cells.append(
            [
                "All (excluding control)",
                str(b["n"]),
                f"{pct(b['accuracy'])} → {pct(f['accuracy'])}",
                f"{b['fact_score']:.2f} → {f['fact_score']:.2f}",
                f"{b['token_f1']:.2f} → {f['token_f1']:.2f}",
                f"{b['rouge_l']:.2f} → {f['rouge_l']:.2f}",
                "",
            ]
        )
    table = ax.table(
        cellText=cells,
        colLabels=header,
        loc="upper center",
        cellLoc="center",
        colWidths=[0.2, 0.06, 0.24, 0.12, 0.12, 0.12, 0.1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.75)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#e2e8f0")
        if row == 0:
            cell.set_facecolor("#0f766e")
            cell.set_text_props(color="white", fontweight="bold")
        elif row == len(cells) and overall:
            cell.set_facecolor("#f1f5f9")
            cell.set_text_props(fontweight="bold")
    fig.text(
        0.05,
        0.015,
        "Accuracy: answer states the gold fact (dosha set, modern name, prognosis, classical "
        "text, ≥50% of listed symptoms...), wording-independent.  "
        "Fact score: partial credit.  "
        "Token F1 / ROUGE-L: word overlap with the reference answer.\n"
        "McNemar: exact test on the paired answers (same questions, both models).  Control = "
        "unseen conditions without retrieval: dataset-specific facts there are not knowable.",
        fontsize=9,
        color="#64748b",
    )

    out = args.out or run / "report" / "metrics_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
