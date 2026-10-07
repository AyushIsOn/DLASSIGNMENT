# Source review record

Scope: source-grounding checks by AI, not medical expert review or training approval.
No GPU execution, model selection or benchmark-driven target edits occurred.

## Original 24-example seed

Reviewer: Codex `seed_review` worker, separate AI pass on 2026-10-06.
It inspected NCCIH pages directly and indexed Delhi publisher text, reporting no
clear factual contradiction in the answers, with three corrections:

| Finding | Resolution by primary agent |
|---|---|
| `nccih-turmeric-2` attributed an inferred warning/causal explanation to the passage | Changed to a factual question and answer about the product differences listed |
| Turmeric osteoarthritis context omitted route and safety section locator | Added oral route and the publisher safety heading |
| Prakriti locator named a heading not displayed on the page | Changed to the introductory definition paragraph locator |

The reviewer also found a response-pattern bias: every binary question expected a
negative answer. New questions add affirmative identification, definitions and
comparisons. This addresses coverage; it does not establish model quality.

## Expansion: 36 examples

Primary author/source checker: Codex. Three concept-page candidates were researched
by the `seed_review` AI worker; the primary agent checked publisher text and revised
the final contexts/questions. Nine page records add ginger, licorice, fenugreek,
bacopa, Centella, Tinospora, classical works, routines and substance terminology.
Answers were checked against their contexts, and contexts against the cited publisher
text. Medical summaries preserve uncertainty and do not recommend personal treatment.

Live NIH HTML is retained for the three NCCIH pages. Indexed publisher text is
retained for the six other pages; each record hashes its own inspected page result.
Full pages are not included in the repository. PCIM&H catalog entries establish
only the Ayurveda relevance of the three added NCCIH herbs.

A second review of the expansion was requested, but the worker hit a service usage
limit before returning decisions. It remains unfinished. Optional per-example
additional-review fields are absent, so the manifest reports zero recorded
additional acceptance decisions rather than inventing approval from a failed run.
The original-seed narrative review above is retained separately and is not a
row-level approval record.

## Limits and disposition

All 60 examples remain preparation material. AI source checking is neither clinical
validation nor proof that a source's claims are established. Related publishers,
concepts and evidence-caveat patterns recur across splits. No observed base/adapter
answers were used to author this expansion. Existing held-out allocations were
preserved; future checkpoint selection uses validation, not test answers.
