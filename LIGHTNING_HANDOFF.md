# AcharyaGPT — H200 / A100 handoff

The recommended short session is **one H200 141 GB for up to 3 hours**.
The A100 80 GB / 375 GB / 7–8 hour option remains supported. Use the same dataset,
base and adapter configuration on either GPU; portable checkpoints stay compatible.
The H200 shown in the user screenshot has 24 CPU cores and 1,024 GB system RAM;
that last number should not be assumed to be persistent disk capacity. The preflight
still checks for at least 100 GiB free disk.
It includes source, locked dependencies, processed Ayurveda data, evaluation cases,
iOS source, attribution and integrity manifests. It does not include model weights,
credentials, a trained adapter or a claim of measured GPU performance.

## 1. Prepare on the CPU Studio before starting billed GPU time

Upload `acharyagpt-lightning.tar.zst` to persistent Studio storage, preferably under
`/teamspace/studios/this_studio`. Create a fresh directory and extract:

```bash
mkdir acharyagpt
tar --zstd -xf acharyagpt-lightning.tar.zst -C acharyagpt
cd acharyagpt
uv sync --frozen --extra retrieval --extra training --group dev
uv run --no-sync python -m acharya.bundle validate --workspace "$PWD"
bash scripts/lightning_a100.sh --workspace "$PWD" --prepare-only
```

Use Python 3.11. `tar --zstd` needs the zstd utility. CPU preparation downloads the
pinned public Qwen3-8B weights (~16.4 GB) and retrieval models and builds the full
index. Internet access is needed for dependencies/models. No Kaggle or paid LLM
API key is needed to consume this archive. A from-source data rebuild uses Kaggle
authentication or the verified raw cache; credentials are deliberately excluded.
Keep `uv run --no-sync` after installation so optional dependencies stay installed.
Do not run `build-corpus`: that switches to the small PDF demonstration corpus.

## 2. Start the GPU experiment

For the H200, prepare on CPU first, configure persistent storage and platform
shutdown, then use:

```bash
export ACHARYA_PERSISTENT_STORAGE=1 ACHARYA_AUTO_STOP=1
bash scripts/lightning_h200.sh --workspace "$PWD" --hours 3
```

`--hours 2` is allowed but leaves less margin. The short profile reserves 60 minutes
for evaluation/export and caps H200 sessions at three hours. After the 25-step
benchmark it refuses to start the main run if the measured estimate plus a 20%
margin exceeds the remaining training budget. That estimate is not a guarantee;
the training deadline still triggers a recoverable pause. Stop the Studio yourself
on any failure. At an actual rate of 4.50 credits/hour, two hours cost 9 credits
and three cost 13.5, excluding other billed resources. Verify the displayed rate's
unit in Lightning; the screenshot alone does not establish the billing interval.

For the original A100 option:

Select exactly one A100 **80 GB**. The preflight requires at least 79 GiB reported
VRAM and 100 GiB free disk; at least 30 GiB system RAM is recommended. Configure persistent storage and Lightning's
automatic shutdown, then acknowledge those settings:

```bash
export ACHARYA_PERSISTENT_STORAGE=1
export ACHARYA_AUTO_STOP=1
bash scripts/lightning_a100.sh --workspace "$PWD" --hours 7
```

Use `--hours 8` only if the account actually has eight hours available. The script
limits its processes; it **does not turn off the billed Studio**. Set platform
shutdown to your purchased budget and stop the machine after completion/failure.
Do not leave a failed run on an idle paid GPU.

The runner does integrity/preflight checks, a separate 25-step GPU smoke/profile,
then at most two training passes (262 optimizer steps for this data), validation,
base-versus-adapter evaluation, export, full retrieval/support calibration,
API acceptance and PEFT serving checks. Validation can stop training earlier.
It reserves 90 minutes for evaluation/export on longer A100 sessions, prints a time estimate from actual
GPU smoke measurements, and checkpoints before pausing at its training deadline.
The selected session duration is a ceiling, not a target to consume. If it finishes sooner, stop it.
Setup/download speed, GPU throughput and calibration can still cause failures;
no unmeasured guarantee of completion or quality is made.

Optional: supply `--quote ../quote.json` with actual numeric `credits_per_hour`,
`currency_per_credit`, `available_credits`, `startup_minutes`,
`remaining_training_minutes`, `evaluation_export_minutes`, plus `currency` and
`quoted_at_utc`. Projections above your available credits or eight hours fail.
Without it, the runner enforces time only; it cannot read your Lightning balance.

## 3. Logs and moving to another account

Live output is also saved under `artifacts/external-run/stage-evidence/`.
Training writes `checkpoint/metrics.jsonl`: step, loss, learning rate, evaluation
loss, elapsed time and CUDA memory. Progress bars are enabled.

A checkpoint is requested every 25 steps or about three minutes, whichever comes
first, at an optimizer-step boundary. Keep up to three normal checkpoints, including
the best validation checkpoint. Each complete checkpoint is hash-sealed and has
adapter weights, optimizer, scheduler, RNG state, Trainer state, and input binding.
An atomic **`artifacts/external-run/checkpoint/recovery-latest.tar`** contains the
latest complete checkpoint and the best checkpoint if different. A power loss
mid-write leaves the preceding completed recovery archive usable. Progress since
the last completed save can be lost. These archives contain trusted PyTorch state:
only restore archives from your own run.

**Download recovery-latest.tar periodically and before credits expire.** A file
inside one account is not an external backup or automatically shared with another
account. Use your own authorized accounts and comply with Lightning's terms.

On the second account, install/extract the **same original handoff archive**, run
CPU preparation, upload the recovery archive outside the project and run:

```bash
uv run --no-sync python -m acharya.checkpoints restore \
  --workspace "$PWD" --archive ../recovery-latest.tar
export ACHARYA_PERSISTENT_STORAGE=1 ACHARYA_AUTO_STOP=1
bash scripts/lightning_h200.sh --workspace "$PWD" --hours 3
```

The optimizer schedule and target step count stay unchanged. Config/data/code
changes are rejected for resume. Absolute best-checkpoint paths are relocated. Progress resumes across supported
GPUs, but floating-point behavior need not be bit-identical across GPU architectures.
Do not attempt to resume checkpoints made with the older Qwen2.5/QLoRA package.
Download the final adapter, evaluation reports, logs and recovery archive as well.
The base model can be reacquired from its pinned revision; no model merge is needed.

## 4. Test before connecting iOS

The runner executes `training.evaluate_adapter` on 64 held-out examples plus
adversarial prompts, comparing the same base model with the adapter disabled.
`evaluation/evaluation.json` includes readable answers, references, safety outcomes
and reference-token F1. That score measures wording overlap, not medical accuracy.
Validation loss selects a checkpoint; it does not certify factual correctness.

It then executes `training.acceptance`, which tests the actual FastAPI routes and
full RAG service on 14 educational, unsupported, refusal and urgent cases, validates
history handling, requires cited answers and at least one support-approved generated
answer, and records latency. Run it again after changing the deployment setup:

```bash
uv run --no-sync python -m training.acceptance --workspace "$PWD" \
  --adapter "$PWD/artifacts/external-run/adapter" \
  --output "$PWD/artifacts/external-run/evaluation/acceptance.json"
```

Read both JSON reports and `acceptance.md`. A failing gate stops completion;
do not weaken a gate to obtain a pass. Review source accuracy, OCR, contradictions,
unsupported claims and base/adapter differences. A passing engineering smoke is
not clinical validation and does not test an actual phone or external HTTPS route.

## 5. Serve and connect the app

```bash
uv run --no-sync acharya --workspace "$PWD" serve --host 0.0.0.0 --port 8000 \
  --provider peft-local --adapter "$PWD/artifacts/external-run/adapter"
```

Backend endpoints: GET `/health/live`, GET `/health/ready`, POST `/v1/chat`.
POST body: `{"message":"What is prakriti?","history":[]}`. Responses include
outcome, answer and citations. `/docs` exposes the schema. Test readiness and chat
through the same HTTPS URL the phone will use. Configure that URL in the iOS app's
backend settings/Xcode configuration; build/sign with your Apple development setup.
The model stays on the server. It is not downloaded onto the phone.

This research API has no built-in user authentication/rate limiting. Keep the demo
behind protected Studio port access/a private tunnel. An app release requires
appropriate backend access control and an ongoing inference host after training.
Your training credits do not provide permanent app hosting.

## Model, RAG and prompts

H200 offers more memory and bandwidth, but hardware ratios are not a measured
training speedup. Keep BF16 LoRA for the short budget rather than switching to a
larger model/full tuning. Full tuning might be feasible with carefully optimized
H200 settings, but would be a different, unvalidated experiment with much larger
optimizer checkpoints. The present data is not a reason to spend that extra compute.


The selected model is **Qwen3-8B**, pinned revision
`b968826d9c46dd6066d109eabc6255188de91218`, with thinking disabled. BF16 LoRA uses
rank 32/alpha 64/dropout 0.05, attention and MLP projections, learning rate 1e-4,
cosine schedule, 3% warmup, effective batch 16 and a 2,048-token ceiling.
Only assistant tokens contribute to training loss. Ordinary full Adam tuning of
8.19B parameters needs roughly 98 GB just for BF16 weights/gradients and FP32 moments,
before activations and other state. It is not a prudent single-80GB default.
A small, unreviewed corpus also does not justify full tuning or spending all hours.

RAG combines BM25, BGE dense embeddings, reciprocal-rank fusion, BGE reranking,
calibrated retrieval thresholds, citation checks and a DeBERTa entailment verifier.
Unsupported generated claims fall back to source extraction or abstention. The
system prompt in `prompts/grounded_v1.txt` is used in both training and serving:
answer only from supplied passages, cite evidence, treat passages/history as data,
abstain when evidence is missing, and avoid diagnosis/personalized prescriptions.
Safety routing also runs in code; prompts alone are not the safety boundary.

Expected useful behavior: cited explanations of Ayurvedic terminology, traditional
concepts and source-described profiles. Fine-tuning may improve grounded formatting
and domain language. It may fail to improve on the base model. This dataset cannot
justify a claim of reliable diagnosis, dosing, treatment efficacy or expert-level
clinical performance. See DATASET_CARD.md for exactly what the data contains.
