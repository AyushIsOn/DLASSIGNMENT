"""Draw docs/pipeline_flow.png: how data, training, evaluation and the app fit together.

    python scripts/make_flow_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"data": "#dbeafe", "build": "#fef3c7", "train": "#dcfce7", "eval": "#fce7f3",
          "serve": "#ede9fe", "app": "#e0f2fe"}
EDGE = {"data": "#2563eb", "build": "#d97706", "train": "#16a34a", "eval": "#db2777",
        "serve": "#7c3aed", "app": "#0284c7"}


def box(ax, x, y, w, h, kind, title, lines):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=COLORS[kind], ec=EDGE[kind], lw=1.8))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=11.5,
            fontweight="bold", color="#0f172a")
    ax.text(x + w / 2, y + h - 0.62, "\n".join(lines), ha="center", va="top", fontsize=8.6,
            color="#334155", linespacing=1.45)
    return (x, y, w, h)


def arrow(ax, a, b, label=None, side="right"):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    if side == "right":
        start, end = (ax0 + aw, ay0 + ah / 2), (bx0, by0 + bh / 2)
    else:  # down
        start, end = (ax0 + aw / 2, ay0), (bx0 + bw / 2, by0 + bh)
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, lw=1.6,
                                 color="#475569", connectionstyle="arc3,rad=0"))
    if label:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.12, label, ha="center",
                va="bottom", fontsize=8, color="#475569", style="italic")


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.text(8, 8.65, "AcharyaGPT - end-to-end pipeline", ha="center", fontsize=18,
            fontweight="bold", color="#0f172a")
    ax.text(8, 8.25, "Ayurveda knowledge -> fine-tuned Qwen3 LLM -> chat API -> SwiftUI iOS app",
            ha="center", fontsize=11, color="#475569")

    # row 1: data -> build -> train -> evaluate
    sources = box(ax, 0.3, 5.35, 3.4, 2.45, "data", "1. Data sources", [
        "NAMASTE (Ministry of Ayush):", "2,483 morbidity codes +", "12,070 SAT terms",
        "Kaggle Ayurveda tables (1,337)", "Original PDF Q&A, concepts,", "safety prompts"])
    build = box(ax, 4.3, 5.35, 3.4, 2.45, "build", "2. Dataset builder", [
        "finetune/build_dataset.py", "natural Q&A from every fact", "7 question wordings / fact",
        "entity-level train/val/test split", "leakage checks", "-> data/sft, data/kb, SQLite DB"])
    train = box(ax, 8.3, 5.35, 3.4, 2.45, "train", "3. LoRA fine-tuning", [
        "Qwen3 (8B / 14B), bf16 LoRA r=64", "1x H200 on Lightning AI",
        "token-budget batches", "checkpoints every 10 min", "resumable on any account",
        "-> LoRA adapter"])
    evaluate = box(ax, 12.3, 5.35, 3.4, 2.45, "eval", "4. Evaluation", [
        "base vs fine-tuned, same prompts", "held-out test set:",
        "recall, unseen + RAG, unseen", "terms, concepts, safety",
        "fact-level scoring + perplexity", "-> REPORT.md + charts"])
    arrow(ax, sources, build)
    arrow(ax, build, train)
    arrow(ax, train, evaluate)

    # row 2: export -> serve -> app (right to left under the top row)
    export = box(ax, 12.3, 1.0, 3.4, 2.75, "train", "5. Export", [
        "merge LoRA into the model", "llama.cpp -> GGUF (Q4_K_M)", "Ollama on a Mac",
        "or: transformers on the GPU", "(lightning_serve.sh +", "Cloudflare https tunnel)"])
    box(ax, 6.3, 1.0, 5.2, 2.75, "serve", "6. Chat API (FastAPI, src/acharya)", [
        "1. safety rules: emergency / self-harm / dose requests",
        "2. BM25 retrieval over ~16k knowledge cards",
        "3. fine-tuned model writes the answer",
        "4. dose guard on the output",
        "POST /v1/chat -> answer + sources + mode"])
    box(ax, 0.3, 1.0, 5.2, 2.75, "app", "7. SwiftUI iOS app", [
        "chat UI from the original AcharyaGPT design",
        "server URL in settings (gear icon)",
        "shows answer, mode badge, sources, latency",
        "suggestions, retry, cancel, new chat",
        "iPhone / iPad, iOS 17+"])
    arrow(ax, evaluate, export, None, side="down")
    ax.add_patch(FancyArrowPatch((12.3, 2.375), (11.5, 2.375), arrowstyle="-|>", mutation_scale=16,
                                 lw=1.6, color="#475569"))
    ax.add_patch(FancyArrowPatch((6.3, 2.375), (5.5, 2.375), arrowstyle="<|-|>", mutation_scale=16,
                                 lw=1.6, color="#475569"))
    ax.text(5.9, 2.5, "HTTPS", ha="center", fontsize=8, color="#475569", style="italic")
    ax.text(8, 0.55, "Educational project - not medical advice.", ha="center", fontsize=8.5,
            color="#94a3b8")

    out = ROOT / "docs" / "pipeline_flow.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
