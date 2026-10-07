# AcharyaGPT fine-tuning dataset

Built by `python -m finetune.build_dataset` (CPU, deterministic, ~5 s) and committed in
`data/sft/` with SHA-256 hashes in `data/sft/MANIFEST.json`. Numbers below come from
`data/sft/stats.json`.

## Sources

| Source | Used for | Rows used |
|---|---|---:|
| [Ayurvedic Knowledge Dataset](https://www.kaggle.com/datasets/akashkumarpr/ayurvedic-knowledge-dataset) (Akash Kumar, v1, CC BY 4.0) | conditions: modern equivalent, dosha, body system, prognosis, symptoms, treatment principles, classical text | 974 unique names (26 duplicate names dropped) |
| [AyurGenixAI](https://www.kaggle.com/datasets/kagglekirti123/ayurgenixai-ayurvedic-dataset) (kagglekirti123, v1, CC BY 4.0) | diseases: doshas, prakriti, symptoms, herbs, yoga, diet/lifestyle, Hindi/Marathi names | 363 unique diseases (most complete row kept per disease) |
| `data/dataset V1.0.pdf` (original AcharyaGPT) | 20 Q&A pairs, cleaned transcription in `data/curated/pdf_qa.yaml` | train only |
| `data/curated/concepts.yaml` (written for this project) | 59 general concepts (doshas, dhatus, agni, panchakarma, classical texts...) with 5 phrasings each | |
| `data/curated/safety.yaml` (written for this project) | dose requests, emergencies, self-harm, stopping medication, "diagnose me" | |

Both Kaggle files are downloaded anonymously with `kagglehub` and must match pinned
SHA-256 hashes. Formulation doses from AyurGenixAI are never used. The 10,000-row
"Ayurveda Healthcare" Kaggle table is not used (repeated synthetic variants with
mismatched symptom profiles), and neither are the Sushruta Samhita OCR scans.

## Format

Each JSONL row is a chat example: `messages` = optional history, the user turn (the
question, or retrieved knowledge-base entries + the question) and the assistant answer.
The system prompt is added at train/eval/serve time from `acharya/prompting.py`.

Answers are short natural sentences written from the table fields, e.g.

> **Q:** Which modern disease is Kandu Jwara equivalent to?
> **A:** Kandu Jwara corresponds to Fever with itching in modern medicine.

The previous version of this project trained on answers that were a verbatim copy of the
passage in the prompt (95% five-word overlap), so the model only learned to copy. Here
86% of training rows are closed book (no passage at all) and only 0.2% of answers appear
verbatim in their prompt (the 57 concept cards whose text is the answer; a test keeps
this under 1%).

## Splits (how the test avoids leakage)

Each attribute has 7 question phrasings: phrasings 0-3 are used for training (3 per
fact), 4 for validation and 5-6 for testing. Conditions are hashed into groups:

| Group | Share | Training | Testing |
|---|---:|---|---|
| seen | 84% | closed book (3 phrasings) + 35% open book copies | `seen_closed`: unseen phrasing, no retrieval |
| rag_only | 8% | open book only | – |
| held out | 8% | **never** | `heldout_open`: with retrieved entries; `heldout_closed`: control |

| Split | Rows | Open-book rows |
|---|---:|---:|
| train | 34,274 | 4,958 |
| validation | 1,178 | 405 |
| test | 1,918 | 469 |

Token counts with the Qwen3 tokenizer (checked by `finetune.check_data`): train 5.41M
tokens per epoch, of which 1.03M are supervised answer tokens; longest example 812 tokens
(no truncation at `max_seq_len=1024`). BM25 retrieval finds the right entry in the top 3
for 100% of held-out test questions and 96% of seen-condition test questions.

## Limitations

The Kaggle tables are community-made and have not been verified by clinicians; some
"modern equivalents" are descriptive rather than exact diagnoses. The model learns these
tables as they are. Outputs are educational, not medical advice.
