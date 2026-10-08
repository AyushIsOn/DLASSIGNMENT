# Evaluation metrics (what is measured and why)

All numbers come from `python -m finetune.evaluate` + `python -m finetune.report`
(`artifacts/run*/report/REPORT.md`). The base model and the fine-tuned model are the
**same weights with the LoRA adapter off/on**, given identical prompts, greedy decoding.

## Test design (prevents measuring memorized sentences)

| Test group | What it asks | What it shows |
|---|---|---|
| Knowledge recall (`seen_closed`) | facts that were trained, with question **wordings never used in training** | the facts were learned, not one sentence |
| Unseen + RAG (`heldout_open`) | conditions/terms **never trained on**, with 3 retrieved knowledge-base entries | reading and using retrieved context |
| Unseen terms (`unseen_closed`, round 2) | meaning of never-trained Sanskrit terms, no retrieval | generalization from word parts |
| Control (`heldout_closed`) | unseen conditions, no retrieval | dataset-specific facts are unknowable → expected low; checks the test is fair |
| Concepts, Safety | general Ayurveda; dose / emergency / self-harm / diagnosis requests | breadth and safe behaviour |

Splits are made per **entity** (condition/term), so a held-out condition never appears in
training in any form. The build fails if any test question or held-out entity leaks into training.

## Metrics

| Metric | Definition | Why it is used |
|---|---|---|
| **Accuracy** (headline) | share of answers that state the gold fact (`finetune/metrics.py`): dosha set must match exactly; prognosis / classical text: first one named must match; modern name / meaning: phrase match; symptoms / herbs: ≥ 50% of listed items | judges *correctness of the fact*, independent of wording, so the base model is not penalised for its style |
| **Fact score** | partial credit (e.g. Jaccard of dosha sets, share of symptoms covered) | finer-grained than right/wrong |
| **Token F1** | harmonic mean of precision/recall of content words vs the reference answer (SQuAD-style) | standard QA overlap metric |
| **ROUGE-L** | F-measure of the longest common subsequence with the reference | standard generation/summarization metric |
| **Cross-entropy loss / perplexity** | mean negative log-likelihood of the reference answer tokens; perplexity = e^loss | how probable the correct answers are under the model (teacher forcing); lower is better |
| **Validation loss curve** | loss on the validation split during training (step 0 = untouched base model) | detects over-fitting; best checkpoint is kept |
| **BM25 recall@3** | share of held-out questions whose gold entry is in the top-3 retrieved | quality of the retrieval step (RAG) |
| **95% confidence interval** | Wilson score interval of each accuracy | uncertainty from a finite test set |
| **McNemar's exact test** | on paired answers: questions only the fine-tuned model got right vs only the base model | whether the improvement is statistically significant (p < 0.05) |

## How to read a 100% on knowledge recall

That group asks about facts *contained in the training data* (with new wordings), so a
high score is the intended result of fine-tuning: the model stores and recalls the facts.
It is **not** evidence of generalization; that is what the unseen groups measure.
In round 1, unseen conditions without retrieval improved from 34% to 55% and with retrieval
reached ~99%, i.e. the model learned transferable patterns and how to use retrieved entries.
Example interval: 1119/1119 correct → 95% CI 99.7%–100%.
