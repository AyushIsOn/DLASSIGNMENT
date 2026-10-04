# AcharyaGPT: Lightning AI handoff

## What is ready

The archive contains source, locked dependencies, the bundled PDF, processed data,
training/validation/test JSONL, attribution, and integrity manifests. Raw Kaggle
archives and credentials stay on the preparation machine. No Kaggle token is
needed to use this archive. The active prepared corpus fingerprint is
`3f2ae80f27d0cbf11a6c64df285e807cd4018213ffa5ed9964ae8561a775e9dd`; it contains
28,169 retrieval chunks and 1,456 complete SFT examples (1,161 train, 146
validation, 149 test). GPU training has not been executed locally.

The merge includes the repository PDF, 1,000 Ayurvedic knowledge records, 446
AyurGenixAI Ayurveda profiles, 77 deduplicated healthcare profiles for retrieval,
12,241 accepted MedQuAD records for retrieval, four official AYUSH educational
passages, and 2,582 public-domain Sushruta passages. MedQuAD and the healthcare
table are deliberately retrieval-only; prescription, medicine, formulation, and
medical-intervention fields from the Kaggle profile table are excluded from the
prepared text. Review the generated `DATA_CARD.md` and `attribution.json` before
redistribution.

## Upload and install

1. Create a Lightning Studio with persistent storage. Upload `acharyagpt-lightning.tar.zst`.
2. Extract into a new directory (`tar --zstd -xf acharyagpt-lightning.tar.zst -C acharyagpt`).
   Create the destination directory first. Install the `zstd` utility if needed.
3. In that directory, install Python 3.11 and uv, then run:

```bash
uv sync --frozen --extra retrieval --extra training --group dev
uv run python -m acharya.bundle validate --workspace "$PWD"
uv run python -m acharya.bundle activate --workspace "$PWD"
```

Do not run `build-corpus` in the prepared handoff: that command selects the small
PDF demo corpus. The bundled active corpus already contains the merged data.

## Try the extracted-answer backend first

```bash
uv run acharya --workspace "$PWD" build-index
uv run acharya --workspace "$PWD" calibrate
uv run acharya --workspace "$PWD" serve --host 0.0.0.0 --port 8000
```

Check `/health/ready` and POST `{"message":"Which are the doshas involved in Eka kusta?","history":[]}`
to `/v1/chat`. Answers include citations. Keep the service private: this research
API does not implement user authentication or rate limiting. Use Studio's protected
port access or a tunnel for your own demo. An iPhone release needs a reachable
HTTPS backend URL configured in Xcode.

## GPU training

The existing profile requires exactly one A100 with at least 39 GiB VRAM, 60 GiB
free disk, persistent storage, and platform auto-stop enabled. The script limits
its own runtime but does not switch off a billed Studio; configure that in Lightning.
It downloads the pinned Qwen2.5-7B weights on the GPU machine. It never starts a paid
machine itself.

Create `quote.json` **outside the extracted project** using actual pricing and time
estimates from your Lightning account. Required fields:

```json
{
  "credits_per_hour": "REPLACE_WITH_ACTUAL_NUMBER",
  "currency_per_credit": "REPLACE_WITH_ACTUAL_NUMBER",
  "currency": "REPLACE_WITH_CURRENCY",
  "quoted_at_utc": "REPLACE_WITH_ISO_UTC_TIMESTAMP",
  "startup_minutes": "REPLACE_WITH_ESTIMATE",
  "remaining_training_minutes": "REPLACE_WITH_ESTIMATE",
  "evaluation_export_minutes": "REPLACE_WITH_ESTIMATE"
}
```

Replace placeholders with real numbers, not strings. This is a template, not a
price quote. Projections above 10 credits or 240 minutes are rejected.

```bash
export ACHARYA_PERSISTENT_STORAGE=1
export ACHARYA_AUTO_STOP=1
bash scripts/lightning_a100.sh --workspace "$PWD" --quote ../quote.json --train-steps 100
```

This performs a 5-step smoke run, a 100-step profile, held-out/adversarial checks,
adapter export, and support-verified RAG serving smoke. A failed quality/support
check is a failed check; it must not be bypassed to label training complete.
100 steps is a pilot, not a demonstrated quality improvement. The trainer masks
the user/provenance prefix from loss and evaluates the held-out validation split.
Use a larger `--train-steps` only within your measured budget. Compatible checkpoints resume
when the step target is extended; data/config changes invalidate them.

## Serve the exported adapter

After successful export and support calibration:

```bash
uv run acharya --workspace "$PWD" serve --host 0.0.0.0 --port 8000 \
  --provider peft-local --adapter "$PWD/artifacts/external-run/adapter"
```

The runtime loads the verified local base snapshot and adapter lazily on CUDA.
Generated answers require support calibration and verified citations; otherwise
responses fall back to source extraction. The default server always uses extraction.
For optional local Ollama, use `--provider ollama --model YOUR_LOCAL_MODEL`, after
`calibrate-support`; Ollama must be listening on loopback.

## iOS

Open `AcharyaGPT(iOS)/AcharyaGPT(iOS).xcodeproj` in full Xcode. Debug defaults to
`http://127.0.0.1:8000` for a simulator on the same Mac. Set
`INFOPLIST_KEY_ACHARYA_BACKEND_URL` to your actual HTTPS endpoint for a phone or Release.
No server has been deployed by this handoff. Run the shared scheme's tests on an
available iOS 17+ simulator. The preparation Mac lacked full Xcode, so only project
structure was validated.

## Remaining evidence

Full semantic retrieval, support-verifier model execution, GPU training, and actual
iOS simulator execution require their respective environments. The ten-row
committed retrieval evaluation is a smoke set, not a clinical benchmark. Source
citations and policy filters are not evidence of medical correctness. Review the
prepared DATA_CARD and attribution before extending or distributing the corpus.
