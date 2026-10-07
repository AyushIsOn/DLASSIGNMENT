# AcharyaGPT fine-tuning dataset (round 2)

Built by `python -m finetune.build_dataset` (CPU, deterministic, ~30 s) and committed in
`data/sft/*.jsonl.gz` with SHA-256 hashes (of the decompressed content) in
`data/sft/MANIFEST.json`. The same build writes the SQLite database
`data/db/acharya.sqlite` (not committed, 87 MB). Numbers below come from `data/sft/stats.json`.

## Sources

| Source | Used for | Entities |
|---|---|---:|
| [NAMASTE National Ayurveda Morbidity Codes](https://namaste.ayush.gov.in/ayurveda) (Ministry of Ayush, Govt. of India) | diagnostic terms: English name, NAMASTE code (+ ICD-11 TM2 code), clinical features, parent category, classical references | 2,483 terms (rows without an IAST spelling dropped, 18 duplicate IDs merged) |
| [NAMASTE Standardised Ayurveda Terminology (SAT)](https://namaste.ayush.gov.in/sat_Ayurveda) | fundamentals, body, signs & symptoms, substances, pharmacy, diet, treatment, prevention: meaning, definition, category | 12,070 terms (its diagnostic section duplicates the morbidity codes and only adds their references) |
| [Ayurvedic Knowledge Dataset](https://www.kaggle.com/datasets/akashkumarpr/ayurvedic-knowledge-dataset) (Akash Kumar, v1, CC BY 4.0) | conditions: modern equivalent, dosha, body system, prognosis, symptoms, treatment principles, classical text | 974 |
| [AyurGenixAI](https://www.kaggle.com/datasets/kagglekirti123/ayurgenixai-ayurvedic-dataset) (kagglekirti123, v1, CC BY 4.0) | doshas, prakriti, symptoms, herbs, yoga, diet/lifestyle, Hindi/Marathi names | 363 |
| `data/dataset V1.0.pdf` (original AcharyaGPT) | 20 Q&A pairs (`data/curated/pdf_qa.yaml`) | train only |
| `data/curated/concepts.yaml`, `data/curated/safety.yaml` (this project) | 59 general concepts; dose / emergency / self-harm / diagnosis requests | |

The two NAMASTE Excel files are committed in `data/raw/namaste/` (downloaded 2026-10-07,
SHA-256 pinned). The Kaggle files are downloaded with `kagglehub` and hash-checked.

Checked and **not used**: the Hugging Face sets `Tweaks/Ayurvedic_QA` (82k rows generated
from one book, mostly "Q1: who is the author…"), `jaychedaa/Ayurveda-LLM-dataset`
(LLM-written, no licence), `AmareshHebbar/ayurveda-icd-sft` (ICD-10-CM codes, not
Ayurveda); Kaggle "Ayurveda Healthcare" (synthetic variants); Sushruta Samhita OCR.

## Format

Each row: `messages` = optional history, the user turn (question, or retrieved entries +
question) and the assistant answer, plus `meta` (source, entity, attribute, split group,
gold label). The system prompt is added at train/eval/serve time from
`acharya/prompting.py`. Example:

> **Q:** What does vatakapholbana-hinapittasannipatah mean?
> **A:** Vātakaphōlbaṇa-hīnapittasannipātaḥ (वातकफोल्बण-हीनपित्तसन्निपातः) is the Ayurvedic
> diagnostic term for sannipāta vikāra due to aggravated vāta-kapha and depleted pitta.

## Splits (how the test measures generalization, not memorization)

Every attribute has 7 question phrasings: 0-3 train (3 per fact for the Kaggle tables,
2 for NAMASTE), 4 validation, 5-6 test; phrasing 6 for NAMASTE uses the diacritic-free
spelling people type. Entities are hashed into groups. NAMASTE terms are grouped by
spelling-insensitive name, so a term listed twice can never be both trained and "unseen".

| Group | Share | Training | Testing |
|---|---:|---|---|
| seen | 84% | closed book + 25-35% open-book copies | `seen_closed`: unseen wording, no retrieval |
| rag_only | 8% | open book only | – |
| held out | 8% | **never** | `heldout_open`: with retrieved entries; `unseen_closed`: infer a term's meaning without retrieval; `heldout_closed`: control |

Leakage guards (the build fails if any is violated):
- no validation/test question appears in training;
- no held-out entity or held-out card appears in training rows or contexts;
- no spelling twin of an unseen term is trained;
- a seen term's category answer is dropped if its parent term is held out.

Closed-book unseen questions whose term appears **anywhere** in training (for example
glossed inside another definition) are removed: 149 test rows. Echoing the question's term
earns no credit in scoring.

| Split | Rows | Open-book rows |
|---|---:|---:|
| train | 88,214 | 12,327 |
| validation | 1,182 | 525 |
| test | 2,242 | 720 |

Test groups: seen_closed 1,000 · heldout_open 720 · unseen_closed 300 ·
heldout_closed 138 · concepts 59 · safety 25.

Train tokens (Qwen3 tokenizer, `finetune.check_data`): 14.4M per epoch, of which 3.4M are
supervised answer tokens. The longest example is 1,316 tokens, so `max_seq_len=1536`
truncates nothing. BM25 finds the right entry in the top 3 for 98% of held-out questions.

## Limitations

The Kaggle tables are community-made and not clinically verified. NAMASTE is an official
terminology, but its English glosses are terse and contain a few spelling errors from the
source. Outputs are educational, not medical advice.
