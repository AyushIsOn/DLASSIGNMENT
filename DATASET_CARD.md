# AcharyaGPT dataset card — 4 October 2026

Prepared fingerprint: `ea1cc9d86476329512b50bdb865aeb48921ff6f81d149c4de0b810115fa81cc8`.

This is an educational research corpus for a cited Ayurveda assistant. The active
retrieval corpus contains **28,092 chunks**. Fine-tuning uses **1,454 complete
question/answer examples**, split into **1,174 train / 136 validation / 144 test**.
These are separate views of the data: retrieval windows are not training answers.

| Source | Raw rows or paragraphs | Accepted retrieval records | Retrieval chunks | Complete SFT examples |
|---|---:|---:|---:|---:|
| Ayurvedic Knowledge, Kaggle v1 | 1,000 | 1,000 | 1,000 | 1,000 |
| AyurGenixAI, Kaggle v1 | 446 | 446 | 446 | 446 |
| Repository six-page PDF | 19 Q&A | 10 | 11 | 8 |
| MedQuAD, Kaggle v1 | 16,412 | 12,241 | 23,464 | 0 |
| Sushruta, 1907 English OCR | 2,583 retained-length paragraphs | 2,582 | 3,167 | 0 |
| Project-authored AYUSH paraphrases | 4 | 4 | 4 | 0 |
| Ayurveda Healthcare, Kaggle v2 | 10,000 | 0 — quarantined | 0 | 0 |

## What the records contain

Ayurvedic Knowledge is a 12-column table: identifiers, Ayurvedic name, proposed
modern equivalent, body system, dosha, prognosis, symptoms, age, gender, treatment
principles, and source text. Prepared answers retain descriptive name/equivalent,
body system, dosha, prognosis and symptom fields with explicit labels. Treatment
principles are omitted. The source's mappings and prognosis labels are community
assertions, not validated diagnoses or evidence of treatment effectiveness.

AyurGenixAI has 34 columns and 446 rows representing 367 distinct disease names.
Prepared records retain disease name, Hindi and Marathi names, symptoms, doshas,
and constitution. They omit remedies, herbs, formulations, medicines, medical
interventions, diet/yoga recommendations, prevention and prognosis. Most text is
English; the translated names do not constitute a multilingual Q&A dataset.

The original Healthcare table repeats a small number of conditions as thousands
of numbered variants. Inspection also found mismatched profiles: the retained
Jaundice row listed nonspecific digestive symptoms, and a Migraines row listed
anxiety/insomnia/dizziness. Its uploader-supplied confidence/authentication scores
are not independent validation. The entire source is quarantined. It remains in
the local raw cache and audit inventory, but contributes no active answers.

MedQuAD contains general biomedical questions, answers, source labels and focus
areas. It contributes most retrieval text (about 84%), and is not Ayurveda SFT.
The 1907 Sushruta text is historical English OCR: blank-line paragraphs are filtered
by minimum length and split into retrieval windows. It can contain OCR errors,
front matter and obsolete clinical assertions. It is excluded from fine-tuning.
The four AYUSH-based notes are **project-authored paraphrases**, not an official
government dataset or independently reviewed training labels.

## Actual prepared example

The JSONL row below is drawn directly from the generated split. IDs and provenance
are metadata; the model receives only the chat messages.

```json
{
  "id": "chk_012b9398a2fd1b015c188f98",
  "messages": [
    {
      "content": "What are the symptoms and Ayurvedic profile of Zoonotic Infections?",
      "role": "user"
    },
    {
      "content": "Community dataset profile (not a clinical assessment). Disease: Zoonotic Infections. Hindi name: जन्तुजन्य संक्रमण. Marathi name: प्राणीजन्य संक्रमण. Symptoms: Fever, chills, nausea, muscle aches, skin rash. Doshas: Pitta. Constitution/Prakriti: Pitta",
      "role": "assistant"
    }
  ],
  "provenance": {
    "dataset": "kagglekirti123/ayurgenixai-ayurvedic-dataset/1",
    "license": "CC BY 4.0",
    "locator": "row=440",
    "source_reference": "Kaggle AyurGenixAI dataset",
    "url": "https://www.kaggle.com/datasets/kagglekirti123/ayurgenixai-ayurvedic-dataset",
    "version": 1
  }
}
```

Every retrieval record additionally carries a stable chunk ID, source, locator,
role, dataset version, license label, URL, trust tier and source reference.
CSV line numbers and historical paragraph locators are retained for inspection.

## Cleaning and splits

Archives, extracted members, ordered headers and row counts are hash/version
checked. Exact and near-duplicate answers use 5-word shingles and a 0.90 Jaccard
threshold. The repetitive Healthcare source is quarantined; selected prescriptive
fields are omitted from other community sources. Rule-based safety filtering is
applied to complete SFT answers, independently of retrieval chunk filtering. This
leaves eight complete PDF examples instead of ten accepted retrieval records.

SFT deduplication precedes a deterministic 80/10/10 hash split by normalized
question. The resulting observed proportions differ slightly from those targets.
There are zero normalized-question or exact-answer overlaps across splits.
There are 99 question groups with multiple distinct community answers; each group
stays in one split. These answers can disagree and have not been adjudicated by a
clinician. Different names for a related topic can still occur across splits, so
this is **not evidence of zero semantic leakage**. The eight PDF SFT examples all
land in training; validation/test therefore cover the community tables only.

## Training parameters and measured limits

- Base: pinned Qwen2.5-7B-Instruct, 7,615,616,512 parameters.
- QLoRA: 4-bit NF4, double quantization, BF16 compute, rank 16 / alpha 32 /
  dropout 0.05; attention and MLP projections are adapted.
- Learning rate 0.0002, cosine schedule, 3% warmup, effective batch 16,
  maximum sequence length 1,536, seed 3407; evaluation/checkpoint interval 25 steps.
- The actual Qwen tokenizer measured 81–207 tokens per full example, with no
  truncation. Training input totals 129,335 tokens, of which 84,614 are supervised
  assistant tokens per pass. Dynamic padding masks pad labels with -100; the
  system/user prefix is also masked. Provenance metadata is not tokenized.
- A 100-step pilot processes roughly 1.36 passes over 1,174 examples at batch 16.
  It is a starting experiment, not a tuned optimum. The 1,536 ceiling leaves room
  for future longer examples; dynamic padding avoids padding every row to it.

## What result to expect

The data and software now support a bounded training experiment and a working
cited extraction baseline. This small, repetitive, unreviewed SFT collection may
teach vocabulary and answer format, but cannot justify a claim of medical accuracy
or improvement over the base model. It contains about 85k supervised training
tokens per pass; adding more epochs may memorize labels rather than add knowledge.
Keep retrieval for source access and judge the adapter against the same base-model
prompts. The evaluation script saves readable base/adapter answers, references and
reference-token F1. F1 measures wording overlap, not factual or clinical correctness.
A domain expert should review disagreements and hallucinations before broader use.

The ten-row retrieval calibration retained 5/5 answerable cases with 0/5 false
support. Separate smoke queries checked doshas, prakriti, a PDF question and an
asthma profile, plus abstention/refusal/urgent routing. These are engineering
checks, not independent retrieval or medical benchmarks. No 7B GPU training,
neural retrieval/support-model execution, or clinical validation has run locally.

## Sources and reproduction

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the generated
`attribution.json` for credits and license distinctions, including MedQuAD's
upstream CC BY 4.0 license versus the mirror's Apache label. Kaggle uploaders'
license labels do not independently establish rights in every medical source.
New public Kaggle and Internet Archive download URLs were fetched again and
matched the pinned hashes. The prepared transfer archive works without raw data
or a Kaggle credential. A strict rebuild downloads supplemental sources, validates
the tracked paraphrases, and uses the original Kaggle CLI or hash-valid cache for
the other archives. No secret or model weight is included in the transfer.
