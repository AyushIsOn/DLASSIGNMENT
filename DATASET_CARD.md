# AcharyaGPT dataset card — 6 October 2026

Fingerprint: `250d0ff5acdb5adfd9664c08f4447b9b91bfa56a521a1c6316dc0de928d902ec`.

The active corpus contains **8,642 Ayurveda retrieval passages** and **2,593 grounded
training/evaluation examples**: **2,093 train / 236 validation / 264 test**. This is
an educational research collection, not clinically validated medical instruction.
Ayurveda retrieval coverage grew from 4,628 to 8,642 passages (87%). The previous
28,092 total included 23,464 general-biomedical MedQuAD passages, now removed.
We did not enlarge the dataset by duplicating rows or adding unrelated medicine.

| Source | Accepted records | Retrieval passages | SFT examples |
|---|---:|---:|---:|
| Ayurvedic Knowledge, Kaggle v1 | 1,000 | 1,000 | 1,000 |
| AyurGenixAI, Kaggle v1 | 446 | 446 | 446 |
| Repository PDF | 10 | 11 | 8 |
| Sushruta volume I, 1907 | 1,823 | 2,229 | 404 |
| Sushruta volume II, 1911 | 2,582 | 3,167 | 450 |
| Sushruta volume III, 1916 | 1,541 | 1,785 | 285 |
| Four project-authored AYUSH paraphrases | 4 | 4 | 0 |
| MedQuAD (16,412 raw records) | 0 — out of scope | 0 | 0 |
| Ayurveda Healthcare (10,000 raw records) | 0 — quality quarantine | 0 | 0 |

## What is actually inside

**Ayurvedic Knowledge:** a 12-column community table. Prepared answers label names,
proposed modern equivalents, body system, dosha, prognosis and symptoms. Treatment
principles are excluded. The original mappings/prognosis claims are uploader
assertions, not validated diagnoses. Some records concern modern disease names
through their purported Ayurvedic profiles; they are not a general biomedical QA set.

**AyurGenixAI:** 446 rows, 34 columns, 367 distinct disease names. Prepared answers
retain names (including Hindi/Marathi aliases), symptoms, doshas and constitution.
Medicines, herbs, formulations, interventions, diet/yoga, prevention and prognosis
are omitted. Text is predominantly English; this is not multilingual QA training.

**Sushruta:** three historical English translations acquired as public-domain OCR.
Passages discuss traditional concepts, anatomy, physiology and historical medicine.
Retrieval includes historical clinical assertions and OCR/front-matter noise.
Those claims must be attributed as historical, not presented as modern evidence.
The old volume attribution was corrected: Internet Archive's `00susruoft` is volume
II (1911); `01susruoft` is I (1907), and `03susruoft` is III (1916).

The **1,139 new SFT examples** are grounded extraction tasks from filtered historical
paragraphs, not newly authored expert QA. Selection requires 35–180 words, primarily
alphabetic text and domain terms; excludes dosage/administration/cure/treatment,
surgery, toxic-substance and related prescriptive stems. This heuristic reduces
obvious unsuitable material but is not an expert review. The four short AYUSH-based
notes are project-authored paraphrases, not an official government dataset.

The Healthcare table repeats numbered variants and contains mismatched symptom
profiles. Its entire collection remains quarantined. MedQuAD is entirely excluded
because the user requested Ayurveda alone. Raw caches are retained only for audit;
the transfer archive includes the active processed data, not those raw archives.

## Record format

JSONL examples contain `id`, `question`, `task_type`, `provenance` and `messages`.
Each `messages` array has the actual grounded system prompt, a user turn with
numbered context/source and question, and a cited assistant answer. Provenance
includes dataset/version, source reference, locator, URL and license. For example,
a historical row's complete question/answer and provenance are:

```json
{
  "question": "Sushruta Samhita, Volume II (1911 English translation) passage 1533",
  "task_type": "historical_extraction",
  "provenance": {
    "dataset": "public_domain_sushruta_ii/1",
    "license": "public domain",
    "locator": "paragraph=1533",
    "source_reference": "Sushruta Samhita, Volume II (1911)",
    "url": "https://archive.org/details/englishtranslati00susruoft",
    "version": 1
  },
  "answer": "The historical passage states: a case of the Kaphaja type of elephantiasis the principal vein (Sird) of the first toe should be opened by an experienced surgeon and the patient should be made to take at intervals the decoction (of the Kapha-sub- duing drugs) with honey. As an alternative, the patient should be advised to take the powders (Kalka) of Abhayd mixed with any officinal kind of urine. The affected locality should be constantly plastered with the paste [1]"
}
```

This format trains evidence-following and citation behavior. The answer is supplied
in the context by design. Held-out performance therefore measures grounded response
behavior, **not closed-book knowledge acquisition**. RAG's corpus contains the source
passages used for held-out QA, as expected for an open-book task.

## Cleaning, splits and limits

Versions, archive/member hashes, CSV schemas and counts are checked. Exact and
near-duplicate answers are removed before a deterministic question-grouped 80/10/10
split. No normalized-question IDs overlap across splits. Related concepts can
still cross splits; do not claim zero semantic leakage. Community profiles can
contradict one another; they have not been adjudicated by a domain expert.

The actual pinned Qwen3 tokenizer checked every example: **244–826 tokens**, with
**zero truncation** at the 2,048-token ceiling. Training contains 748,661 input tokens
per pass, of which **217,766 assistant tokens** are supervised. Prompts and padding
are masked from loss. Dynamic padding avoids padding every example to 2,048.
Two passes give about 436k supervised tokens; this is a modest adaptation dataset,
not enough to create a medically authoritative foundation model.

Larger public candidates were reviewed but not automatically admitted: gated
expert QA needs authorized access, modern textbook-derived QA has unclear source
rights, and large synthetic clinical dossiers add unverified prescriptions and
non-Ayurvedic content. Dataset size alone is not a quality criterion.

## Model and expected outcomes

Pinned Qwen3-8B (8,190,735,360 parameters), BF16 LoRA rank 32/alpha 64/dropout .05,
learning rate 1e-4, cosine/3% warmup, effective batch 16, maximum two passes. The
A100/H200 runner profiles actual throughput and keeps validation-selected weights.
Full tuning is not the default: ordinary optimizer state alone pushes total state
beyond a single 80 GB GPU, and the data quality does not justify adapting all weights.

Expect a cited educational assistant with retrieval, refusal and abstention behavior.
Fine-tuning may improve formatting and domain language or may not beat the base.
No measured 8B GPU accuracy, latency, clinical safety rate or improvement is claimed.
Run the base/adapter evaluation and API/RAG acceptance checks inside Lightning, then
review the actual answers before connecting iOS. Reference overlap and tiny smoke
sets cannot establish medical accuracy. A clinician should review any broader use.

## Reproduction and attribution

See THIRD_PARTY_NOTICES.md, configs/datasets.yaml and generated attribution.json
for exact URLs, hashes and license labels. Kaggle uploader labels do not independently
establish rights in every underlying text. The prepared archive needs no Kaggle
credential. Strict rebuild uses the official Kaggle client or verified cache plus
hash-pinned public text downloads. No user credential or model weight is bundled.
