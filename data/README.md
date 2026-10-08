# Data layout

```
data/
  raw/namaste/        official NAMASTE Excel files (Ministry of Ayush), unmodified, SHA-256 pinned
  dataset V1.0.pdf    original AcharyaGPT Q&A (20 pairs)
  curated/            hand-written inputs
    pdf_qa.yaml         clean transcription of the PDF
    concepts.yaml       59 general Ayurveda concepts, 5 question wordings each
    safety.yaml         dose / emergency / self-harm / diagnosis requests and replies
  sft/                fine-tuning dataset (built, committed)
    train.jsonl.gz       88,214 chat examples
    validation.jsonl.gz   1,182
    test.jsonl.gz         2,242 (held out: new wordings + never-trained terms)
    stats.json           counts per split / group / source
    MANIFEST.json        SHA-256 of every input and output (checked before training)
  kb/cards.jsonl      15,967 retrieval cards used by the chat API (BM25)
  db/acharya.sqlite   everything above in one SQLite file (built, not committed)
```

The Kaggle tables (Ayurvedic Knowledge Dataset, AyurGenixAI) are downloaded and
hash-checked by the builder. Rebuild everything (byte-identical) with:

```bash
uv run --extra data python -m finetune.build_dataset
sqlite3 data/db/acharya.sqlite "SELECT source, COUNT(*) FROM entities GROUP BY source"
```

See [../DATASET_CARD.md](../DATASET_CARD.md) for sources, splits and leakage checks.
