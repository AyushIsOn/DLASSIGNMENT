# AcharyaGPT retraining checklist

Status: data preparation in progress. No new model has been trained or approved.
Updated: 2026-10-06. This file is the task's record of completion, not a claim of
medical accuracy. GPU execution happens in the user's Lightning environment.

## 1. Preserve and diagnose the baseline

- [x] Preserve the old adapter and original data; never resume its optimizer for the new run.
- [x] Compare 32 raw base/adapter answers on identical evidence without retrieval fallback.
- [x] Record the copying problem (mean five-word source overlap: base 6.6%, adapter 95.3%).
- [x] Audit the uploaded 778 drafts; do not bulk-approve them.
- [x] Fix missing-citation reporting in the draft audit.

Evidence: uploaded direct-comparison outputs and local review bundle dated 2026-10-06.
The comparison is diagnostic, not independent accuracy measurement.

## 2. Prepare the replacement data

- [x] Author and source-check a 24-example seed with page URLs, locators and review scope.
- [ ] Independently inspect the seed for unsupported answers and misleading source claims.
- [x] Create a coverage inventory: concepts, comparisons, evidence limits, missing evidence,
      correction of false premises, and non-prescriptive safety boundaries.
- [ ] Expand beyond the seed using verified educational sources; avoid community disease labels
      and OCR passages that cannot support an explanation.
- [ ] Record review decisions for every accepted example; AI review must be labelled as AI review.
- [ ] Check near-duplicate questions and passages, not only exact matches.
- [ ] Produce a dataset card with counts by source/topic/task and explicit remaining limitations.

Completion requires source-supported answers, valid citations, preserved qualifications,
usable language, no invented mechanisms or dose recommendations, and traceable provenance.
A structural pass is not a factual review. The old 778 drafts are not an approved corpus.

## 3. Freeze evaluation before training

- [x] Keep current seed source pages in only one split (16 train / 4 validation / 4 test).
- [ ] Choose source-family/topic groups before expansion; keep related pages together.
- [x] Create an evaluation rubric covering support, relevance, clarity, citations and safety.
- [ ] Check all reference answers against their evidence and record review limitations.
- [ ] Freeze dataset and rubric hashes. Do not use test answers to tune data or choose checkpoints.
- [ ] Collect raw base answers on frozen validation/test evidence; preserve blind review files.

The current four-question validation and test subsets are too small. Aim for a scoped pilot
of at least 100 train / 30 validation / 30 test examples, with at least 6 / 3 / 3 distinct
source families respectively. These are project planning floors, not statistical guarantees.
Do not pad counts with rewordings of the same passage. The existing 500/100/100 production
compiler gate stays in force until a separate bounded pilot path is implemented and tested.

## 4. Prepare and run one bounded pilot

- [ ] Add an explicit experimental pilot profile with data/hash/review checks.
- [ ] Start from locked Qwen3-8B base with a fresh LoRA adapter/output directory.
- [ ] Verify prompt boundaries, assistant-only labels, nonzero supervised tokens and padding.
- [ ] Run a GPU preflight and short throughput check before selecting step/batch limits.
- [ ] Save logs, validation loss, optimizer/scheduler state and portable checkpoints.
- [ ] Set a hard time/step ceiling and ensure checkpoint-on-interruption behavior.
- [ ] Give one tested Lightning command; run only after dataset and evaluator gates pass.

No full fine-tuning commitment and no throughput guarantee. Extra GPU memory is not a
quality metric. Keep the existing adapter, retrieval index and calibration unchanged.

## 5. Decide from measured results

- [ ] Compare base/new adapter with identical evidence and no RAG fallback.
- [ ] Review anonymized answers using the frozen rubric; report rater identity and disagreements.
- [ ] Report paired outcomes by task/source plus latency, copying and actual loss curves.
- [ ] Select on validation only. Promote only if reviewed support/usefulness improves without
      new serious safety errors; otherwise retain the base and diagnose the failure.
- [ ] Evaluate the selected model once on the frozen test set; report uncertainty and sample size.
- [ ] Run production API/RAG acceptance separately and expose fallback rates.
- [ ] Export model, prompts, retrieval assets, manifests and recovery instructions before iOS wiring.

A promising small pilot is not clinical certification or proof of broad Ayurveda expertise.
No checkpoint is called improved merely because loss or reference overlap looks good.

## Current next action

Inventory, rubric and CPU readiness checker are implemented. Training code was
audited; short-run evaluation cadence and whole-session budget controls still need
implementation. Independent review workers hit a service usage limit, so the
independent-review item remains open; no review was claimed on their behalf.

Next: expand and review the source-separated corpus, then implement the separate
bounded pilot path. Current readiness: 16/4/4 examples across 3/1/1 source families;
all six size/family requirements remain below the planning floors. No paid GPU run
is required for these steps.

Evidence artifacts: `docs/pilot-data-inventory.md`, `docs/pilot-training-audit.md`,
`docs/pilot-evaluation.md`, `docs/pilot-readiness.json`, `eval/pilot_rubric.json`.

