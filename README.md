# AcharyaGPT

An Ayurveda assistant: **Qwen3-8B fine-tuned with LoRA** on Ayurvedic knowledge tables,
with a small BM25 retrieval layer, served to a **SwiftUI iOS app**. Based on
[danushkhanna/AcharyaGPT](https://github.com/danushkhanna/AcharyaGPT) (a LangChain + OpenAI
RAG demo over one PDF). Educational only - not medical advice.

## Why the previous training showed no improvement

| Problem in the old code | Fix |
|---|---|
| The training answer was a **copy of the passage in the prompt** (`answer = context + " [1]"`; 95% five-word overlap), so the model learned to copy text, nothing else. Sushruta OCR "questions" were literally `"... passage 1533"`. | New dataset (`finetune/build_dataset.py`): natural answers to real questions; 86% closed-book, so facts must be learned. |
| Copy loss drops to ~0 immediately, so early stopping ended training after a few hundred steps. | 3 full epochs (~3,200 steps), cosine schedule, best validation checkpoint kept. |
| Evaluation put the reference in the prompt and measured word overlap with it. | Held-out test: new question wordings + entirely unseen conditions, scored on the *fact* (style-independent), base vs fine-tuned on identical prompts. |
| The app never showed the model: the server answered with extracted source sentences unless a DeBERTa verifier approved every claim. Later commits disabled training entirely. | The API sends the question (+ retrieved entries) to the fine-tuned model and returns its answer with sources. |
| iOS read the server URL from a custom `INFOPLIST_KEY_*` setting that Xcode never writes into Info.plist, so every request failed with "backend URL not configured". | URL set in the app (gear icon), default `http://127.0.0.1:8000`. |

The old 10k-line pipeline was removed; it is still in the git history (commit `9e85eb1`).

## Run it on Lightning AI (one CPU step, one GPU step)

Everything that does not need a GPU runs first on a CPU Studio. Every script can be
re-run safely: finished steps are skipped and training resumes from its last checkpoint.

**1. CPU Studio (free) - about 20 minutes**

```bash
git clone https://github.com/AyushIsOn/DLASSIGNMENT.git && cd DLASSIGNMENT
bash scripts/lightning_cpu_setup.sh
```

Installs the locked environment, downloads Qwen3-8B (16.4 GB) and checks every shard's
SHA-256, verifies the dataset and the prompt/label masking with the real tokenizer, runs
the unit tests and a full train → evaluate → report → merge pass on a tiny model. It must
end with `CPU SETUP COMPLETE`.

**2. Switch the same Studio to 1× H200, then - about 1.5-2 hours**

```bash
cd DLASSIGNMENT && bash scripts/lightning_gpu_run.sh
```

GPU preflight → LoRA training (prints an ETA after 20 steps) → base vs fine-tuned
evaluation → report → merged model → GGUF file for the Mac. It runs in its own background
session: closing the tab or pressing Ctrl+C only stops the log view; run the same command
again to re-attach. If the Studio stops, start it on the H200 again and re-run - nothing
finished is repeated. Training has a hard 3.5-hour limit so the whole run fits in the
5-hour budget even in the worst case. Make sure the Studio's auto-sleep cannot stop it
during the run. It ends with `ALL DONE`.

Results: `artifacts/run/report/REPORT.md` with loss curves and accuracy charts.

**3. (Optional) demo the app straight from the GPU**

```bash
bash scripts/lightning_serve.sh     # prints https://<random>.trycloudflare.com
```

Paste that URL into the app (gear icon). It works while the script runs.

**4. Download the results, then stop the Studio (or switch it back to CPU)**

`artifacts/gguf/` (≈5 GB Q4_K_M + Modelfile, for the Mac), `artifacts/run/report/` and
`artifacts/run/adapter/`. If the GGUF step reported `FAILED`, run
`bash scripts/export_gguf.sh` on the CPU machine - it needs no GPU.

## Run the app on a Mac (no GPU needed)

1. Install [Ollama](https://ollama.com/download) and copy the downloaded `gguf` folder to
   `artifacts/gguf/` in this repo.
2. `bash scripts/mac_serve.sh` (creates the `acharyagpt` Ollama model and starts the API
   on `http://127.0.0.1:8000`; use `--lan` for a real iPhone on the same Wi-Fi).
3. Open `AcharyaGPT(iOS)/AcharyaGPT(iOS).xcodeproj` in Xcode 15+, pick an iPhone
   simulator, press Run. For a real iPhone, select your team under Signing & Capabilities
   and set the server URL to the Mac's IP in the app's settings.

Check the server without the app:

```bash
curl -s localhost:8000/v1/chat -H 'Content-Type: application/json' \
  -d '{"message":"Which dosha is predominant in Ardita?","history":[]}'
```

## What is trained and how it is evaluated

* **Model:** Qwen/Qwen3-8B (pinned revision, thinking disabled), bf16 LoRA r=64/α=128 on
  all attention and MLP projections (175M trainable parameters), lr 2e-4 cosine, 3 epochs,
  effective batch 32, gradient checkpointing. Loss on answer tokens only. `configs/train.yaml`.
* **Data:** 34k training conversations from two Kaggle Ayurveda tables (~1,340 conditions),
  the original PDF's Q&A, 59 hand-written concepts and safety examples - see
  [DATASET_CARD.md](DATASET_CARD.md).
* **Test set (1,918 questions, never trained on):**
  *knowledge recall* - facts from training asked with question wordings never used in
  training; *unseen conditions + RAG* - conditions excluded from training, with the BM25
  entries in the prompt (as the app does); *concepts*, *safety*; and a *control* with unseen
  conditions and no retrieval. The base model and the fine-tuned model are the same weights
  with the adapter off/on and get identical prompts. Scoring (`finetune/metrics.py`) checks
  the stated fact - dosha set, modern equivalent, prognosis class, classical text, share of
  listed symptoms - not the wording; test-set perplexity is reported as well.
* **Serving:** safety rules first (emergencies, self-harm and dose requests get fixed
  replies, generated doses are removed), then BM25 top-3 entries if the match is
  confident, then the model. Same prompt format as training (`src/acharya/prompting.py`).

## Repository

```
finetune/      build_dataset, check_data, download_model, preflight, train, evaluate,
               metrics, report, export, smoke
src/acharya/   prompting (shared prompt), retrieval (BM25), safety, generation
               (transformers / Ollama), service, api (FastAPI), cli
scripts/       lightning_cpu_setup.sh, lightning_gpu_run.sh, lightning_serve.sh,
               export_gguf.sh, mac_serve.sh
data/          sft/ (train/validation/test JSONL), kb/cards.jsonl, curated/, dataset V1.0.pdf
AcharyaGPT(iOS)/  SwiftUI app + tests
```

Local checks: `uv sync --frozen --extra train --extra data --group dev && uv run pytest -q`.
Rebuild the dataset: `uv run python -m finetune.build_dataset` (identical output).
