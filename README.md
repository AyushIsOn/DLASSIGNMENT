# AcharyaGPT

Offline-first Ayurvedic knowledge retrieval with deterministic ingestion, calibrated BM25/dense/reranked retrieval, cited extractive answers, medical-safety abstention, an iOS client, optional support-verified generation, and an optional external QLoRA workflow.

This is educational software, not medical advice. It must abstain or return its fixed safety response for urgent, diagnostic, personalized-treatment, and dose requests.

## Install and verify

Python 3.11 and `uv` are required. Commands do not need secrets for the default service.

```bash
uv sync --frozen --extra retrieval --extra training --group dev
bash scripts/verify_gate_a.sh
bash scripts/verify_all.sh
```

`verify_all.sh` performs a frozen core/dev install, Ruff, strict mypy, CPU tests,
the network-denied PDF demo API restart, Xcode-project structural validation, and
CPU preflight. It does not download models or run paid compute. Set
`ACHARYA_VERIFY_DATA=1` to additionally prepare/verify the configured Kaggle data.
For the prepared archive and exact Lightning steps, see [LIGHTNING_HANDOFF.md](LIGHTNING_HANDOFF.md).


## Data preparation and licensing

The strict Gate B preparation requires these exact versions:

- [MedQuAD `gpreda/medquad/1`](https://www.kaggle.com/datasets/gpreda/medquad), Apache 2.0 as recorded in `configs/datasets.yaml` and the selected Kaggle version metadata.
- [Ayurvedic Knowledge Dataset `akashkumarpr/ayurvedic-knowledge-dataset/1`](https://www.kaggle.com/datasets/akashkumarpr/ayurvedic-knowledge-dataset), license and required credit recorded in configuration and output attribution.
- [Ayurveda Healthcare Dataset `aliainaanraza/ayurveda-healthcare-dataset/2`](https://www.kaggle.com/datasets/aliainaanraza/ayurveda-healthcare-dataset), license and required credit recorded in configuration and output attribution.

The acquisition client is pinned to the [official Kaggle CLI 2.2.4 release](https://github.com/Kaggle/kaggle-cli/releases/tag/v2.2.4) and uses its [version-aware dataset request](https://github.com/Kaggle/kaggle-cli/blob/v2.2.4/src/kaggle/api/kaggle_api_extended.py). Exact archive/member hashes, ordered schemas, row expectations, size limits, score admission rules, and attribution are enforced. `rcratos/ayurveda-texts-english/1` is explicitly prohibited because its rights are unclear. Deterministic admission is not proof of medical truth.

```bash
uv run acharya --workspace "$PWD" prepare-handoff --strict
uv run acharya --workspace "$PWD" prepare-handoff --strict
uv run acharya --workspace "$PWD" verify-preparation --strict
```

If authentication/network access and the exact hash-valid cache are both unavailable, Gate B remains `BLOCKED_EXTERNAL_DATA`. The PDF is never substituted for a successful three-dataset preparation.

## Retrieval, serving, and optional providers

Gate A is local and network-independent:

```bash
uv run acharya --workspace "$PWD" build-corpus
uv run acharya --workspace "$PWD" build-index
uv run acharya --workspace "$PWD" calibrate
uv run acharya --workspace "$PWD" serve --host 127.0.0.1 --port 8000
```

The accepted default is `generation_mode=extractive`: answers are spans from allowed source chunks with citations. BM25 uses calibrated raw score, positive-IDF lexical coverage, and informative-term overlap rather than per-query normalization. Full retrieval is built and calibrated only against matching corpus/model/config hashes:

```bash
uv run acharya index build --workspace "$PWD" --retrieval-mode full
uv run acharya evaluate --workspace "$PWD" --retrieval-mode full --golden "$PWD/eval/golden.jsonl" --activate-calibration
uv run acharya calibrate-support --workspace "$PWD"
```

Optional Kiro CLI/HTTP, loopback Ollama, and local PEFT transports use the same RAG and safety gates. Generated medical text remains disabled unless every cited claim passes the calibrated [pinned DeBERTa verifier](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli); one unsupported claim rejects the whole candidate. Check Kiro without printing or sourcing the dotenv:

```bash
uv run acharya provider-check --workspace "$PWD" --provider kiro-cli --env-file "$PWD/.env.local" --record "$PWD/artifacts/provider-check/kiro.json"
```

The record contains only allowlisted metadata. A transport exit or non-exact response is not an access-success claim.

## iOS app

The SwiftUI project has no remote package dependencies or embedded provider credentials. Set `ACHARYA_BACKEND_URL` through build settings: loopback HTTP is allowed only in Debug; Release requires HTTPS. Linux can validate references:

```bash
uv run python scripts/validate_xcodeproj.py --project 'AcharyaGPT(iOS)/AcharyaGPT(iOS).xcodeproj/project.pbxproj'
```

Only macOS with Xcode 15/iOS 17 can complete app tests:

```bash
xcodebuild -project 'AcharyaGPT(iOS)/AcharyaGPT(iOS).xcodeproj' -scheme 'AcharyaGPT(iOS)' -destination 'platform=iOS Simulator,name=iPhone 15' test
```

Until that runs, the component is `PENDING_MACOS_VALIDATION`.

## Deterministic Lightning transfer

A bundle may be built only after completed strict preparation. It contains tracked source/config/lock/tests/backend/iOS/training/bootstrap files and finalized processed/SFT/reports. It excludes dotenvs/credentials, raw archives, unclear-rights data, Git metadata, model/index weights, adapters, and checkpoints. `COPY_UPLOAD_INVENTORY.json` says what to move; `raw-inventory.json` contains hashes/identifiers rather than archives.

```bash
FINGERPRINT="$(uv run python -c 'import json,pathlib; print(json.loads(pathlib.Path("artifacts/state/active_preparation.json").read_text())["fingerprint"])')"
bash scripts/prepare_handoff.sh --workspace "$PWD" --fingerprint "$FINGERPRINT" --output "$PWD/dist/lightning/acharyagpt-lightning-${FINGERPRINT}.tar.zst"
uv run python -m acharya.bundle validate-archive --archive "$PWD/dist/lightning/acharyagpt-lightning-${FINGERPRINT}.tar.zst"
```

Copy only that archive and independent quote evidence to persistent Lightning storage, clean-extract it, run `python -m acharya.bundle validate --workspace "$PWD"`, then follow `BOOTSTRAP.md`.

## Optional Gate C QLoRA

`configs/qlora.yaml` locks Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`, all four weight hashes, NF4 double quantization, bf16 compute, LoRA rank 16, 1,536 tokens, and effective batch 16. A 7B bf16 base is roughly 14 GB, while full AdamW weights, gradients, fp32 master weights, and moments exceed 80 GB before activations/checkpoints. Full tuning therefore cannot fit one A100 40 GB and is never an implicit fallback. Reversible QLoRA is expected to profile around 22–32 GB VRAM.

Local CPU verification performs exactly five no-save optimizer steps and records no training claim:

```bash
uv run python -m acharya.lightning preflight --workspace "$PWD" --cpu-only-dry-run
uv run python -m acharya.lightning fake-run --steps 100
```

The external workflow requires exactly one A100 with at least 39 GiB VRAM, completed strict preparation, clean bundle hashes, persistent storage, auto-stop, and JSON quote evidence with `credits_per_hour`, `currency_per_credit`, `currency`, `quoted_at_utc`, and startup/training/evaluation-export minute projections. It aborts before model load above 10 credits or 240 minutes and enforces the documented 10/30/20/45/45/30/45-minute stage ceilings.

```bash
export ACHARYA_PERSISTENT_STORAGE=1 ACHARYA_AUTO_STOP=1
bash scripts/lightning_a100.sh --workspace "$PWD" --quote "$PWD/quote.json" --train-steps 100
```

The script verifies integrity, acquires and hashes the model, runs a 5-step GPU smoke and real 100-optimizer-step profile, resumes only hash-bound checkpoints, evaluates held-out/adversarial/safety rows, atomically exports the adapter, optionally merges bf16 with `--merge`, and runs support-gated PEFT RAG smoke. Gate C becomes `COMPLETED` only when checkpoint state, adapter config/weights/tokenizer/base revision, manifests/file hashes, evaluation, and PEFT serving evidence all validate. Scripts, CPU checks, and fake runners leave it `PENDING_EXTERNAL_GPU`; no tuned-serving or quality-gain claim is made.

## Independent gates and security

- **Gate A:** local calibrated extractive MVP; expected to pass independently.
- **Gate B:** exact external data, full retrieval, provider evidence, iOS, and deterministic handoff. External data, provider, and macOS statuses stay distinct; Gate A fallback cannot satisfy missing Gate B work.
- **Gate C:** optional external GPU enhancement. It never blocks A or B.

CI fetches full Git history, performs frozen checks without external/GPU/macOS assumptions, and runs pinned Gitleaks 8.24.2 with redaction. Equivalent local scan:

```bash
docker run --rm -v "$PWD:/repo" zricethezav/gitleaks:v8.24.2 detect --source /repo --config /repo/.gitleaks.toml --log-opts=--all --redact --exit-code 1
```

Generated reports, raw data, corpora, indexes, caches, bundles, checkpoints, and adapters are ignored. Never commit any of them.

Content describing linked third-party materials was rephrased for compliance with licensing restrictions.
