# Paired evaluation contract

Use `eval/pilot_rubric.json` to review raw base and new-adapter answers on identical
contexts. Do not call production retrieval/fallback for this measurement. Record
all responses, including refusals and malformed output. Reference answers and
rubric notes are never sent as inference prompts.

Freeze source families, split IDs, prompts, reference targets and rubric hashes
before training. Validation selects checkpoints; test remains out of selection.
Existing 21-question results are regression evidence, not this new test set.
Question variants from one passage are correlated; report results by source family
and task, not just an overall average.

Blind the model identities, collect reviewer ID/type/date, scores, severe flags,
and evidence notes. Missing ratings remain unreviewed. AI-generated judgments must
not be described as clinician or independent verification. A second review from
the same model is not an independent clinical assessment. For disagreements,
retain both original ratings and record adjudication rather than silently averaging.

Use paired wins/ties/losses for support and usefulness, serious error counts,
response/abstention counts and citations. Report denominator and rater coverage.
Latencies include cold-load and warm-query measurements separately. Copy fraction
and training/validation loss are diagnostics, not correctness.

Before the run, specify what usefulness improvement warrants another experiment.
Do not choose that threshold after observing validation results. A pilot candidate
must not lose source support/relevance or add serious safety errors. Unresolved
ratings or no credible improvement mean no promotion. Report uncertainty and
source clustering; the proposed 30-item held-out floor is not a statistical guarantee.

Production API/RAG acceptance remains a separate check after selection, with
whether generation was accepted or extractive fallback supplied the answer made
explicit. Neither this rubric nor an acceptance pass certifies clinical reliability.
