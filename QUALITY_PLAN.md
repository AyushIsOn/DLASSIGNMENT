# Establishing answer quality

The current adapter passes engineering acceptance but copies its supplied source.
The new audit flagged all 2,593 old targets. Preserve that adapter as a comparison
baseline; new training with those targets is disabled before any model loads.

`eval/independent_quality.jsonl` contains 21 project-authored compositional,
limitations, safety and unsupported-question checks. The evaluation supplies only
questions to production RAG, never reference criteria or review notes. It compares
extractive, locked-base and adapter serving, saves incremental responses, and
produces a shuffled blind review sheet plus a separate model key. Concept matching
is an automatic screen, not a factual accuracy grade. The benchmark is post-hoc
for the old adapter, small and not clinician validated. It is a starting evaluation,
not sufficient evidence of expertise.

For each blind response, a reviewer scores factuality, relevance, clarity and
safety from 0 (wrong/poor) to 2 (correct/good), checking the actual cited sources.
Record incorrect claims, wrong citations, outdated historical statements, poor
OCR, omissions and unnecessary abstentions. Compare paired scores across models
and report counts and disagreements. A perfect copying score is not used as a
quality gate. Expert review remains necessary for health-related claims.

## Replacement training data

`training.rewrite_targets` uses the already downloaded locked Qwen3-8B **base**
to draft explanatory QA. It never uses the copying adapter or presents synthetic
labels as independent ground truth. Historical extraction records and flagged
OCR are excluded. It preserves original splits and provenance, writes each draft
incrementally and has a wall-time ceiling. The teacher is a resource-conscious
drafting aid, not an expert or a guaranteed improvement.

Draft generation requires GPU compute and has **not** been executed by Codex.
Review each candidate against its source. Reject inaccurate questions/answers;
edit copying, incoherent text and uncited claims. Keep only approved candidates
in the compiler input; set a named reviewer and all four review checks to true
only after performing that review. Do not bulk change these fields merely to pass.

`training.quality_data compile` rejects unreviewed, unsafe, uncited, largely copied,
flagged OCR and historical extraction targets, exact duplicate questions,
benchmark questions and exact source-context overlap across splits. It requires
at least 500 train / 100 validation / 100 test approved records and preserves
hash-bound review evidence. These are engineering minimums, not a claim that the
sample sizes are sufficient. Near-duplicate topics and semantic overlap still
need review. No compiled replacement dataset exists yet.

Use a fresh output directory; do not change the original runtime's active data or
resume its old optimizer. The trainer accepts `--quality-data`, verifies split and
review hashes, audits targets before GPU initialization and binds the new dataset
into checkpoint identity. The old end-to-end runner is intentionally blocked from
starting its old copy-target training. A new benchmark-based training schedule
must be measured after the replacement dataset is ready; no full-fine-tuning or
H200-throughput claim is made.

## Commands from the repaired source checkout

CPU audit and extractive quality screening (no paid GPU required):

```bash
bash scripts/quality_cpu.sh /teamspace/studios/this_studio/acharyagpt-main-fix
```

Optional GPU comparison of existing base and adapter, without retraining:

```bash
export PYTHONPATH="$PWD/src:$PWD"
RUNTIME=/teamspace/studios/this_studio/acharyagpt-main-fix
"$RUNTIME/.venv/bin/python" -u -m training.quality_eval \
  --workspace "$RUNTIME" --benchmark "$PWD/eval/independent_quality.jsonl" \
  --adapter "$RUNTIME/artifacts/external-run/adapter" --mode gpu \
  --output "$RUNTIME/artifacts/quality/independent-gpu-v1"
```

Optional draft generation, **not training**, with a 30-minute ceiling:

```bash
"$RUNTIME/.venv/bin/python" -u -m training.rewrite_targets \
  --workspace "$RUNTIME" --output "$RUNTIME/artifacts/quality/drafts.jsonl" \
  --limit 1500 --minutes 30
```

After review, compile a fresh dataset:

```bash
"$RUNTIME/.venv/bin/python" -m training.quality_data compile \
  --candidates "$RUNTIME/artifacts/quality/approved.jsonl" \
  --benchmark "$PWD/eval/independent_quality.jsonl" \
  --output "$RUNTIME/artifacts/quality/reviewed-v1"
```

Stop the GPU after comparison/drafting or failure. The commands do not stop billing.
Model weights, reviewed replacement targets and measured answer quality are not
included in this change. No new clinical accuracy claim is made.

## Retrieval development and failure traces

CPU quality screening now recalibrates against `eval/retrieval_development.jsonl`
before evaluating the unchanged independent benchmark. The development set uses
different question strings but overlaps the same foundational concepts; it is not
a claim of independent clinical validation. Calibration considers the same context
count as serving, requires labeled relevant sources for foundational development
queries and checks negative queries against every context offered to serving.

Failed response records include the failure stage and top retrieval scores/text.
These diagnostic fields are written to local evaluation artifacts, not exposed by
the chat API. Improvements must be measured on Lightning; no new benchmark result
is claimed from the local tests.

The source-question retrieval revision embeds and reranks existing source question
metadata together with passage text. Index fingerprints include this format; CPU
quality screening rebuilds the full index while retaining old index directories.
Bounded source diversity expands candidate recall without changing support
thresholds. Sentence selection normalizes grammatical number, but returned text
stays verbatim and is validated against its original source.

### Calibration recovery and runtime

The CPU workflow reuses a fingerprint-validated full index with the current
source-question passage format. Development retrieval scores are saved after
each query, bound to index, evaluation contents, context count and retrieval
implementation hashes. Interrupted calibration resumes these scores; corrupted
cache entries are recomputed. Full-mode calibration jointly searches empirical
BM25, coverage, dense and reranker thresholds, maximizing relevant-source
positive coverage subject to zero false support on the development negatives.
This is an empirical development constraint, not a guarantee on unseen queries.

The first score-cache population still requires neural inference. The repeatedly
inspected 21-case benchmark is now regression evidence; final quality claims
require fresh held-out questions and blind review. No new training is authorized
by a retrieval screening pass alone.
