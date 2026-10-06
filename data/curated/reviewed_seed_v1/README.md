# Source-reviewed starting set

This is the first small replacement curriculum, not a production training dataset.
It contains 24 newly authored examples checked against six publisher pages on
2026-10-06: 16 train, 4 validation, 4 test. Entire pages are assigned to one split.
No examples were imported from the 778 generated drafts.

The reviewer is **Codex (AI)**. Source support and citation structure were checked;
this is not independent human review, clinical validation or a guarantee that the
publisher's assertions are medically established. The short contexts are original
paraphrases, not verbatim textbook extracts. Page URLs, section locators, retrieval
hashes, review notes and exclusions are in `sources-and-examples.json`. Publisher
HTML snapshots were inspected locally and are not redistributed. NCCIH pages
explicitly mark their publications public domain; images are not included. Delhi
AYUSH material is represented only by short original factual paraphrases.

## What changed

Training examples contrast concepts, correct misreadings, preserve uncertainty,
and recognize missing evidence. They do not invent causal disease explanations
from categorical labels. AYUSH material is explicitly presented as traditional
theory. NCCIH material supplies evidence limitations rather than personal advice.
Six questions require explaining that the provided passage lacks the requested
information. No answer recommends a product, dose or treatment.

Review included the association asserted, the strength of evidence, qualifiers,
route of administration where relevant, and whether each answer actually follows
from its supplied context. Training examples cover dosha roles, Prakriti,
Ayurveda's scope and turmeric research. Ashwagandha is validation-only; boswellia
is test-only. Source grouping was set before authoring questions. Their general
evidence-limit patterns still overlap, and all examples have the same author.

## Validate without a GPU

From the repository root:

```bash
python -m training.reviewed_seed \
  --source data/curated/reviewed_seed_v1/sources-and-examples.json
```

The checker enforces citation identity, copy/length checks, real prompt parser
round-trips, source allocation, exact question uniqueness and context integrity.
It does not automatically prove entailment or clinical correctness. It always
reports `training_ready: false`; the existing reviewed-data compiler's minimums
and review requirements have not been relaxed. No GPU or training is started.

## Before retraining

This small set demonstrates the replacement target style. It is too small for a
credible broad Ayurveda improvement claim. It needs more independently checked
sources and examples, particularly supported multi-concept explanations and
robust safety cases. Evaluation needs more sources and a separately reviewed
rubric; four questions from one page per held-out split are not an adequate
estimate of generalization. Do not inflate these counts with paraphrases of the
same evidence or reuse the held-out pages as training context.

The existing copying adapter stays a baseline. Any later pilot should start from
the locked base, use a fresh output directory, select checkpoints on validation
only, and compare raw answers without retrieval fallback. Test answers must not
be used to tune prompts or pick a checkpoint. Fresh sources may be needed after
repeated inspection. No training command or readiness claim is provided here.
