"""Draw docs/pipeline_flow.png: the AcharyaGPT pipeline in five steps.

    python scripts/make_flow_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ("#2563eb", "#dbeafe", "1. Data", ["NAMASTE terms", "(Ministry of Ayush)", "Kaggle Ayurveda tables"]),
    ("#d97706", "#fef3c7", "2. Dataset", ["question-answer pairs", "train / test split",
                                         "by condition"]),
    ("#16a34a", "#dcfce7", "3. Fine-tune", ["Qwen3 + LoRA", "1x H200 GPU", "Lightning AI"]),
    ("#db2777", "#fce7f3", "4. Evaluate", ["base vs fine-tuned", "on unseen questions", "accuracy, F1, perplexity"]),
    ("#7c3aed", "#ede9fe", "5. Chat app", ["safety + retrieval", "+ fine-tuned model", "iOS / Android / web"]),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(15.5, 4.2))
    ax.set_xlim(0, 15.4)
    ax.set_ylim(0.3, 4.6)
    ax.axis("off")
    ax.text(7.7, 4.25, "AcharyaGPT pipeline", ha="center", fontsize=19, fontweight="bold",
            color="#0f172a")
    width, gap, top = 2.6, 0.5, 3.6
    for index, (edge, face, title, lines) in enumerate(STEPS):
        x = 0.3 + index * (width + gap)
        ax.add_patch(FancyBboxPatch((x, 1.15), width, top - 1.15,
                                    boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc=face, ec=edge, lw=2))
        ax.text(x + width / 2, top - 0.35, title, ha="center", va="top", fontsize=14,
                fontweight="bold", color=edge)
        ax.text(x + width / 2, top - 0.95, "\n".join(lines), ha="center", va="top",
                fontsize=10.5, color="#334155", linespacing=1.7)
        if index < len(STEPS) - 1:
            ax.add_patch(FancyArrowPatch((x + width + 0.06, 2.35), (x + width + gap - 0.06, 2.35),
                                         arrowstyle="-|>", mutation_scale=20, lw=2,
                                         color="#475569"))
    ax.text(7.7, 0.75, "Educational project - not medical advice.", ha="center", fontsize=9,
            color="#94a3b8")
    out = ROOT / "docs" / "pipeline_flow.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
