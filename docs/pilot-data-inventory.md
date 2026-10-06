# Coverage and source inventory

Current corpus: `data/curated/reviewed_seed_v1/sources-and-examples.json`.
The machine-readable snapshot is `docs/pilot-readiness.json`.

| Allocation | Examples | Distinct source families | Planned minimum |
|---|---:|---:|---|
| Train | 16 | 3 | 100 examples / 6 families |
| Validation | 4 | 1 | 30 examples / 3 families |
| Test | 4 | 1 | 30 examples / 3 families |

Delhi dosha and Prakriti pages belong to one foundations family. NCCIH herb pages
are grouped by herb, not counted as independent publishers. Conceptual similarities
between research summaries remain; these family counts are not independence claims.

Current task coverage: 2 comparisons, 2 corrections, 6 insufficient-evidence
questions, 6 explanations, 7 evidence-limit questions, 1 distinction. Direct safety
boundary and multi-context integration examples remain missing. Do not fill this
 gap by rephrasing the existing four questions per page.

## Expansion order

1. Define additional source families and fix their split assignments before QA
   authoring. Related pages about the same concept/herb stay together. Reject
   community disease labels as evidence of mechanisms, and reject unreadable OCR.
2. Read and log publisher, URL, section/page, date, reuse status and relevant
   limitations. Include enough explanatory evidence to support an answer rather
   than making it up from categorical fields. Preserve traditional/clinical scope.
3. Author focused questions and supported answers; include examples where the
   evidence cannot answer. Keep no more than five variants per supplied passage.
4. Record review against source for every example. AI authorship/review remains
   labelled. Any source defect or unverified mapping is a blocker, not resolved by
   adding [1]. Preserve corrections and rejection reasons.
5. Run exact and approximate duplicate checks. Lexical similarity is only a screen;
   inspect semantic overlap and related source-page families manually.
6. Compile only after counts, source families, review scope and evaluation contract
   are satisfied. Create a fresh manifest; no retroactive bulk approval of drafts.

## Source review status

The original seed received an author source-grounding check, with citations and
prompt structure tested. An independent second review is not complete. Parallel
review workers were unavailable due to the service usage limit during this turn;
no reviewer opinions were fabricated. This document and the training audit were
completed by the main Codex agent. Expert review, if claimed later, must identify
an actual expert and record that review.

## CPU readiness command

```bash
python -m training.pilot_preflight \
  --source data/curated/reviewed_seed_v1/sources-and-examples.json \
  --plan configs/pilot_plan.json \
  --rubric eval/pilot_rubric.json \
  --output artifacts/pilot-readiness.json
```

This generates a report and never starts a GPU or marks the model ready. The
experimental planning floor does not bypass the existing production compiler.
