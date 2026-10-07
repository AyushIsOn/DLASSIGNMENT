# Coverage and source inventory

Updated 2026-10-07. Corpus:
`data/curated/reviewed_seed_v1/sources-and-examples.json`.
Machine-readable status: `docs/pilot-readiness.json`.

| Allocation | Examples | Source families | Planned minimum |
|---|---:|---:|---|
| Train | 40 | 6 | 100 examples / 6 families |
| Validation | 8 | 2 | 30 examples / 3 families |
| Test | 12 | 3 | 30 examples / 3 families |

Fifteen pages supply four questions each. The foundation family includes Delhi
dosha, Prakriti and classical-text pages plus CARI routines and Ayusoft substance
terminology. NCCIH and LiverTox herb pages are grouped by herb. These are topic/source
clusters, not independent publishers.

Task counts: 9 definitions, 8 comparisons, 12 explanations, 3 corrections,
3 evidence interpretation, 11 evidence limits, 6 safety information,
7 insufficient evidence and 1 distinction. Multi-context synthesis remains absent.

The original 24 examples received an additional AI source review. Three wording/locator
corrections are recorded in `source-review-20261007.md`. The 36 new examples received
author source checks; their second review remains incomplete. No row has human
clinical review. All 60 remain preparation material.

## Next expansion rules

1. Assign related source topics before authoring; retain existing held-out allocation.
2. Record the actual inspected source and its limits. Indexed text must not be
   labelled live HTML. Use short original paraphrases when reuse rights are unclear.
3. Add distinct evidence and question types, with at most five questions per context.
4. Preserve qualifiers and traditional framing. Do not derive causal explanations
   from community disease labels or turn proposed biological activity into efficacy.
5. Record additional reviews honestly. Check semantic overlap as well as the existing
   lexical screen, then freeze data and evaluation hashes before a GPU run.

Current gaps: 60 training examples, 22 validation and 18 test examples, plus one
additional validation family. Counts alone do not qualify the corpus; final source
review and the bounded pilot implementation are separate requirements.
