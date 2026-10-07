# Source-reviewed replacement curriculum

Updated 2026-10-07. This is a **60-example work in progress**, not an approved
training dataset: 40 train, 8 validation and 12 test, from 15 publisher pages.
No examples were imported or approved from the 778 generated drafts.

## Contents and allocation

| Split | Topics | Examples | Source families |
|---|---|---:|---:|
| Train | Doshas, Prakriti, classical texts, routines, substance terminology, Ayurveda overview, turmeric, ginger, licorice, fenugreek | 40 | 6 |
| Validation | Ashwagandha, bacopa | 8 | 2 |
| Test | Boswellia, Centella/gotu kola, Tinospora/Guduchi | 12 | 3 |

Related foundation pages from Delhi AYUSH, CARI and Ayusoft are one family.
Herb pages are grouped by herb, not counted as independent publishers. Allocation
was chosen before writing the new questions. Each passage has four questions.
The new material includes affirmative definitions and comparisons, as well as
uncertainty and missing-information responses.

Task counts: 9 definitions, 8 comparisons, 12 explanations, 3 corrections,
3 evidence interpretation, 11 evidence limits, 6 safety information,
7 insufficient evidence and 1 distinction.

## Source and review scope

Contexts are brief original paraphrases checked against publisher text. Each
record identifies its URL, section, date, context hash and review scope. For ginger,
licorice and fenugreek, PCIM&H pharmacopoeia catalog metadata establishes Ayurveda
relevance; those links are not used as clinical evidence or full monograph contents.
NIH pages provide the research and safety summaries.

The reviewer is **Codex (AI)**. The original 24-example seed also received a second
AI source review, leading to corrections recorded in `docs/source-review-20261007.md`.
A second pass over the 36 new examples did not complete because the reviewer service
hit its usage limit. No additional review decisions were fabricated. Final review
before training remains outstanding; none of the examples has clinical expert approval.

Snapshot provenance distinguishes live publisher HTML from indexed publisher text.
The six new non-NCCIH pages used indexed text when direct retrieval was unavailable.
Their hashes identify the actual indexed page text inspected, not live HTML.
Snapshots are retained locally, not redistributed in this repository. NCCIH
publications state they are public domain; photographs are excluded. Other pages
are represented by short factual paraphrases, with no broader reuse grant asserted.
Traditional categories are labelled as such; invented mechanisms, personal diagnoses,
prescriptions and dose recommendations are excluded.

## CPU validation

```bash
python -m training.reviewed_seed \
  --source data/curated/reviewed_seed_v1/sources-and-examples.json
python -m training.pilot_preflight \
  --source data/curated/reviewed_seed_v1/sources-and-examples.json \
  --plan configs/pilot_plan.json --rubric eval/pilot_rubric.json \
  --output artifacts/pilot-readiness.json
```

The structural validator checks identifiers, provenance, literal review booleans,
citations, copy/length rules, prompt round trips and exact duplicates. The separate
preflight checks family allocation and lexical near-duplicates. Neither proves
factual correctness or rules out semantic overlap. Optional additional AI reviews
are counted separately; a `revise` decision is never an approval.

## Remaining limitations

The set is below the pilot planning floors of 100/30/30 examples and 6/3/3 source
families. These floors are planning decisions, not statistical guarantees. Publisher
style, question patterns and broad topics overlap across splits; the base model
may already know every source. Short paraphrased contexts differ from longer,
noisy production RAG passages. Multi-context synthesis and broader coverage are
still missing. No model improvement has been measured on this set.

`training_ready` stays false. The production compiler's 500/100/100 gate is
unchanged. A separate pilot trainer, real-tokenizer check, interruption recovery
test and raw base/adapter evaluation remain required before paid use.

Note: `reviewed_seed_v1/` is the source-reviewed curriculum from PR #17. It is kept for
reference and is not used by the current training set.
