"""Build the fine-tuning dataset and the retrieval knowledge base (CPU only, ~10 s).

    uv run --extra data python -m finetune.build_dataset

Sources (pinned by SHA-256, downloaded anonymously with kagglehub):
  * Ayurvedic Knowledge Dataset (akashkumarpr, v1, CC BY 4.0)  - 1,000 conditions
  * AyurGenixAI (kagglekirti123, v1, CC BY 4.0)                 - 446 disease profiles
  * data/dataset V1.0.pdf (original AcharyaGPT Q&A)             - 20 Q&A pairs, read from
    the cleaned transcription data/curated/pdf_qa.yaml
  * data/curated/concepts.yaml, data/curated/safety.yaml        - hand-written

Why the old run did not improve: its targets were a copy of the passage shown in
the prompt, so the model only learned to copy. Here the answer is a short natural
sentence and, for closed-book rows, the prompt contains only the question, so the
model has to learn the facts. Every fact is trained with 3 question phrasings and
tested with 2 phrasings it never saw. A few entities are held out completely to
measure retrieval-augmented (open-book) answering.

Outputs (committed): data/sft/{train,validation,test}.jsonl, data/sft/stats.json,
data/kb/cards.jsonl, data/sft/MANIFEST.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from acharya.prompting import SYSTEM_PROMPT, user_content
from acharya.retrieval import BM25, Card
from finetune import templates as T

ROOT = Path(__file__).resolve().parents[1]
SEED = 3407

SOURCES = {
    "ak": {
        "handle": "akashkumarpr/ayurvedic-knowledge-dataset/versions/1",
        "file": "ayurvedic_knowledge_dataset.csv",
        "sha256": "f3f67afa2f88141a3252d06882961d7d01fa32cbac1f37339ca2785f0d4fedd3",
        "rows": 1000,
        "citation": "Ayurvedic Knowledge Dataset (Akash Kumar, Kaggle, CC BY 4.0)",
    },
    "ag": {
        "handle": "kagglekirti123/ayurgenixai-ayurvedic-dataset/versions/1",
        "file": "AyurGenixAI_Dataset.csv",
        "sha256": "6b722b8784e5947b762d93a0f1a6e86650ca7af27bbe21a95ab0c84afd4f37d5",
        "rows": 446,
        "citation": "AyurGenixAI Dataset (kagglekirti123, Kaggle, CC BY 4.0)",
    },
}
PDF_SHA256 = "ae5a30200145559dbd0b7ce25b58e6f1a31c272b98743d8a72353a0ddfbc5322"
PDF_CITATION = "Original AcharyaGPT dataset V1.0 (PDF)"
CONCEPT_CITATION = "AcharyaGPT curated Ayurveda concepts"

# Entity groups (by stable hash bucket 0-99).
HELDOUT_BUCKETS = range(0, 8)  # never trained: validation/test only (open-book + control)
RAG_ONLY_BUCKETS = range(8, 16)  # trained only open-book (teaches reading the context)
OPEN_BOOK_SHARE = 0.35  # extra open-book copy for this share of seen facts
VAL_SHARE = 0.08  # seen facts asked with the validation phrasing
TEST_SHARE = 0.12  # seen facts asked with a test phrasing
NONE_VALUES = {"", "none", "none specific", "nan", "n/a", "na", "-"}
CONVERSATIONAL = {"greeting", "about-acharyagpt"}  # trained, but not retrievable KB cards


# ------------------------------------------------------------------ utilities


def stable_int(*parts: object) -> int:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()
    return int(digest[:12], 16)


def bucket(entity_id: str) -> int:
    return stable_int("bucket", entity_id) % 100


def chance(share: float, *parts: object) -> bool:
    return stable_int(*parts) % 10_000 < share * 10_000


def clean(value: Any) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    text = re.sub(r"\s+", " ", text).strip()
    return "" if text.casefold() in NONE_VALUES else text


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(key: str, kaggle_dir: Path | None) -> Path:
    spec = SOURCES[key]
    if kaggle_dir is not None:
        matches = sorted(kaggle_dir.rglob(spec["file"]))
        if not matches:
            raise FileNotFoundError(f"{spec['file']} not found under {kaggle_dir}")
        path = matches[0]
    else:
        import kagglehub  # optional "data" extra

        path = Path(kagglehub.dataset_download(spec["handle"])) / spec["file"]
    actual = sha256_file(path)
    if actual != spec["sha256"]:
        raise RuntimeError(f"{key} hash mismatch: {actual} != {spec['sha256']}")
    return path


def read_csv(path: Path, expected_rows: int) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_rows:
        raise RuntimeError(f"{path.name}: expected {expected_rows} rows, found {len(rows)}")
    return rows


# --------------------------------------------------------------- knowledge base


def load_ak(path: Path) -> list[dict[str, str]]:
    entities, seen = [], set()
    for row in read_csv(path, SOURCES["ak"]["rows"]):
        name = clean(row["Ayurvedic Name"])
        key = name.casefold()
        if not name or key in seen:  # 26 duplicated names: keep the first (lowest Sr No)
            continue
        seen.add(key)
        entity = {
            "id": "ak-" + slug(clean(row["Ayurvedic Code"])),
            "kind": "ak",
            "name": name,
            "code": clean(row["Ayurvedic Code"]),
            "modern_equivalent": clean(row["Modern Equivalent"]),
            "body_system": clean(row["System / Body Part"]),
            "dosha": clean(row["Dosha Predominance"]),
            "prognosis": clean(row["Prognosis"]),
            "symptoms": clean(row["Symptoms"]),
            "treatment": clean(row["Treatment Principles"]),
            "source_text": clean(row["Source Text"]),
        }
        T.dosha_factors(entity["dosha"])  # validates every value up front
        T.system_phrase(entity["body_system"])
        T.prognosis_answer(name, entity["prognosis"])
        entities.append(entity)
    return entities


def _informativeness(row: dict[str, str]) -> int:
    fields = ("Ayurvedic Herbs", "Yoga & Physical Therapy", "Symptoms", "Hindi Name")
    return sum(bool(clean(row.get(field))) for field in fields)


def load_ag(path: Path) -> list[dict[str, str]]:
    best: dict[str, tuple[int, int, dict[str, str]]] = {}
    for index, row in enumerate(read_csv(path, SOURCES["ag"]["rows"])):
        name = clean(row["Disease"])
        if not name:
            continue
        key = slug(name)  # "Crohn's Disease" and "Crohns disease" are one entity
        score = (_informativeness(row), -index)
        if key not in best or score > best[key][:2]:
            best[key] = (*score, row)
    entities = []
    for key in sorted(best):
        row = best[key][2]
        name = clean(row["Disease"])
        entity = {
            "id": "ag-" + slug(name),
            "kind": "ag",
            "name": name,
            "hindi_name": clean(row["Hindi Name"]),
            "marathi_name": clean(row["Marathi Name"]),
            "symptoms": clean(row["Symptoms"]),
            "doshas": clean(row["Doshas"]),
            "prakriti": clean(row["Constitution/Prakriti"]),
            "herbs": clean(row["Ayurvedic Herbs"]),
            "yoga": clean(row["Yoga & Physical Therapy"]),
            "diet_lifestyle": clean(row["Diet and Lifestyle Recommendations"]),
        }
        T.dosha_factors(entity["doshas"])
        T.dosha_factors(entity["prakriti"])
        entities.append(entity)
    return entities


def ak_card(entity: dict[str, str]) -> Card:
    text = (
        f"{entity['name']} (Ayurvedic condition, code {entity['code']}). "
        f"Modern equivalent: {entity['modern_equivalent']}. "
        f"Body system: {entity['body_system']}. Dosha predominance: {entity['dosha']}. "
        f"Prognosis: {entity['prognosis']}. Symptoms: {entity['symptoms']}. "
        f"Treatment principles: {entity['treatment']}. Source: {entity['source_text']}."
    )
    source = f"{SOURCES['ak']['citation']} - {entity['source_text']}"
    return Card(entity["id"], entity["name"], source, text)


def ag_card(entity: dict[str, str]) -> Card:
    names = "; ".join(
        part for part in (
            f"Hindi: {entity['hindi_name']}" if entity["hindi_name"] else "",
            f"Marathi: {entity['marathi_name']}" if entity["marathi_name"] else "",
        ) if part
    )
    text = f"{entity['name']}" + (f" ({names})" if names else "") + ". "
    text += f"Doshas: {entity['doshas']}. Prakriti: {entity['prakriti']}. "
    text += f"Symptoms: {entity['symptoms']}. "
    if entity["herbs"]:
        text += f"Ayurvedic herbs: {entity['herbs']}. "
    if entity["yoga"]:
        text += f"Yoga: {entity['yoga']}. "
    text += f"Diet and lifestyle: {entity['diet_lifestyle']}"
    return Card(entity["id"], entity["name"], SOURCES["ag"]["citation"], text.strip())


PDF_TOPICS = ("Eka kusta", "Kaphaja kasa", "Ardita", "Grahani")


def pdf_topic(question: str, answer: str) -> str:
    """Condition a PDF Q&A pair is about (the PDF covers four), used as the card title."""
    text = f"{question} {answer}".casefold()
    for topic in PDF_TOPICS:
        if topic.casefold() in text:
            return topic
    return "Kaphaja kasa" if "kasa" in text else "AcharyaGPT Q&A"


def load_pdf_qa() -> list[tuple[str, str]]:
    """The original PDF's Q&A, from the committed cleaned transcription."""
    if sha256_file(ROOT / "data" / "dataset V1.0.pdf") != PDF_SHA256:
        raise RuntimeError("bundled PDF hash mismatch")
    pairs = yaml.safe_load((ROOT / "data/curated/pdf_qa.yaml").read_text())["pairs"]
    if len(pairs) != 20 or not all(p["question"] and p["answer"] for p in pairs):
        raise RuntimeError("data/curated/pdf_qa.yaml must contain 20 complete pairs")
    return [(p["question"].strip(), p["answer"].strip()) for p in pairs]


# ------------------------------------------------------------------- facts


def ak_facts(entity: dict[str, str]) -> list[tuple[str, str, dict[str, Any]]]:
    name = entity["name"]
    modern_alternatives = [part.strip() for part in re.split(r"\s+/\s+|/", entity[
        "modern_equivalent"]) if part.strip()]
    treatment_items = [item for item in T.split_list(entity["treatment"])
                       if "prognosis" not in item.casefold() and item.casefold() != "incurable"]
    facts = [
        ("modern_equivalent", T.modern_answer(name, entity["modern_equivalent"]),
         {"type": "phrase", "values": modern_alternatives}),
        ("dosha", T.dosha_answer(name, entity["dosha"]),
         {"type": "dosha_set", "values": T.dosha_factors(entity["dosha"])}),
        ("body_system", T.system_answer(name, entity["body_system"]),
         {"type": "system", "values": [part.strip().casefold() for part in
                                       entity["body_system"].split("/") if part.strip()]}),
        ("prognosis", T.prognosis_answer(name, entity["prognosis"]),
         {"type": "category", "values": [entity["prognosis"].casefold()]}),
        ("symptoms", T.symptoms_answer(name, entity["symptoms"]),
         {"type": "items", "values": T.split_list(entity["symptoms"])}),
        ("source_text", T.source_answer(name, entity["source_text"]),
         {"type": "category", "values": [entity["source_text"].casefold()]}),
        ("overview", T.ak_overview(entity),
         {"type": "composite", "parts": [
             {"type": "phrase", "values": modern_alternatives},
             {"type": "dosha_set", "values": T.dosha_factors(entity["dosha"])},
             {"type": "items", "values": T.split_list(entity["symptoms"])},
         ]}),
    ]
    if treatment_items:
        facts.insert(6, ("treatment", T.treatment_answer(name, entity["treatment"]),
                         {"type": "items", "values": treatment_items}))
    return facts


def ag_facts(entity: dict[str, str]) -> list[tuple[str, str, dict[str, Any]]]:
    name = entity["name"]
    facts = [
        ("doshas", T.doshas_answer(name, entity["doshas"]),
         {"type": "dosha_set", "values": T.dosha_factors(entity["doshas"])}),
        ("prakriti", T.prakriti_answer(name, entity["prakriti"]),
         {"type": "dosha_set", "values": T.dosha_factors(entity["prakriti"])}),
        ("symptoms", T.ag_symptoms_answer(name, entity["symptoms"]),
         {"type": "items", "values": T.split_list(entity["symptoms"])}),
        ("diet_lifestyle", T.diet_answer(name, entity["diet_lifestyle"]),
         {"type": "items", "values": T.split_list(entity["diet_lifestyle"], ";")}),
        ("overview", T.ag_overview(entity),
         {"type": "composite", "parts": [
             {"type": "dosha_set", "values": T.dosha_factors(entity["doshas"])},
             {"type": "items", "values": T.split_list(entity["symptoms"])},
         ]}),
    ]
    if entity["herbs"]:
        facts.append(("herbs", T.herbs_answer(name, entity["herbs"]),
                      {"type": "items", "values": T.split_list(entity["herbs"])}))
    if entity["yoga"]:
        facts.append(("yoga", T.yoga_answer(name, entity["yoga"]),
                      {"type": "items", "values": T.split_list(entity["yoga"])}))
    if entity["hindi_name"]:
        facts.append(("hindi_name", T.hindi_answer(name, entity["hindi_name"],
                                                   entity["marathi_name"]),
                      {"type": "exact", "values": [entity["hindi_name"]]}))
    return facts


# ------------------------------------------------------------------ builder


class Builder:
    def __init__(self, cards: list[Card], heldout_ids: set[str]) -> None:
        self.index = BM25(cards)
        self.cards = {card.id: card for card in cards}
        self.heldout_ids = heldout_ids
        self.rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
        self.retrieval_hits: Counter[str] = Counter()

    def contexts(self, question: str, gold_id: str | None, *, training: bool, salt: str,
                 include_gold: bool = True) -> tuple[list[str], list[str]]:
        """BM25 top-3 (gold forced in when retrieval misses it), stable shuffle."""
        exclude = set(self.heldout_ids) if training else set()
        if gold_id:
            exclude.discard(gold_id)
        hits = [hit.card.id for hit in self.index.search(question, k=6, exclude=exclude)]
        if gold_id and not training:
            self.retrieval_hits["gold_in_top3" if gold_id in hits[:3] else "gold_missed"] += 1
        chosen = [card_id for card_id in hits if card_id != gold_id][:2]
        if gold_id and include_gold:
            chosen.append(gold_id)
        else:
            chosen = [card_id for card_id in hits if card_id != gold_id][:3]
        random.Random(stable_int("ctx", salt)).shuffle(chosen)
        return [self.cards[card_id].text for card_id in chosen], chosen

    def add(self, split: str, row_id: str, question: str, answer: str, meta: dict[str, Any],
            contexts: list[str] | None = None, context_ids: list[str] | None = None,
            history: list[dict[str, str]] | None = None) -> None:
        # The system prompt is NOT stored per row: train/eval/serve all prepend
        # acharya.prompting.SYSTEM_PROMPT (its hash is recorded in MANIFEST.json).
        messages = [*(history or []),
                    {"role": "user", "content": user_content(question, contexts or ())},
                    {"role": "assistant", "content": answer}]
        meta = {**meta, "open_book": bool(contexts), "context_ids": context_ids or []}
        if split == "train":  # gold labels are only needed for evaluation
            meta = {key: meta[key] for key in
                    ("source", "entity", "attribute", "group", "open_book", "context_ids")}
        self.rows[split].append({"id": row_id, "question": question, "messages": messages,
                                 "meta": meta})


def build(kaggle_dir: Path | None, output: Path, kb_output: Path) -> dict[str, Any]:
    ak = load_ak(download("ak", kaggle_dir))
    ag = load_ag(download("ag", kaggle_dir))
    pdf_pairs = load_pdf_qa()
    concepts = yaml.safe_load((ROOT / "data/curated/concepts.yaml").read_text())["concepts"]
    safety = yaml.safe_load((ROOT / "data/curated/safety.yaml").read_text())

    entities = ak + ag
    groups = {}
    for entity in entities:
        value = bucket(entity["id"])
        if value in HELDOUT_BUCKETS:
            group = "heldout_val" if stable_int("half", entity["id"]) % 2 else "heldout_test"
        elif value in RAG_ONLY_BUCKETS:
            group = "rag_only"
        else:
            group = "seen"
        groups[entity["id"]] = group
    heldout = {key for key, group in groups.items() if group.startswith("heldout")}

    cards = [ak_card(e) for e in ak] + [ag_card(e) for e in ag]
    # Titles feed the BM25 title field: concept name / condition name, never question text.
    cards += [Card(f"concept-{c['id']}", c["id"].replace("-", " ").capitalize(),
                   CONCEPT_CITATION, c["answer"]) for c in concepts
              if c["id"] not in CONVERSATIONAL]
    cards += [Card(f"pdf-{i:02d}", pdf_topic(q, a), PDF_CITATION, f"Q: {q} A: {a}")
              for i, (q, a) in enumerate(pdf_pairs, 1)]
    builder = Builder(cards, heldout)
    templates = {"ak": T.AK_TEMPLATES, "ag": T.AG_TEMPLATES}

    # -- entity facts
    for entity in entities:
        group, kind, name, eid = groups[entity["id"]], entity["kind"], entity["name"], entity["id"]
        facts = ak_facts(entity) if kind == "ak" else ag_facts(entity)
        for attribute, answer, gold in facts:
            phrasings = templates[kind][attribute]
            meta = {"source": kind, "entity": eid, "name": name, "attribute": attribute,
                    "gold": gold, "group": group}

            def q(slot: int, phrasings: tuple[str, ...] = phrasings, name: str = name) -> str:
                return phrasings[slot].format(name=name)

            key = f"{eid}|{attribute}"
            if group == "seen":
                skip = stable_int("skip", key) % 4
                for slot in (s for s in T.TRAIN_SLOTS if s != skip):
                    builder.add("train", f"{key}|{slot}", q(slot), answer,
                                {**meta, "slot": slot})
                if chance(OPEN_BOOK_SHARE, "open", key):
                    slot = stable_int("open-slot", key) % 4
                    ctx, ids = builder.contexts(q(slot), eid, training=True, salt=key)
                    builder.add("train", f"{key}|{slot}|open", q(slot), answer,
                                {**meta, "slot": slot}, ctx, ids)
                elif chance(0.04, "miss", key):  # retrieval miss: answer from memory
                    slot = stable_int("miss-slot", key) % 4
                    ctx, ids = builder.contexts(q(slot), eid, training=True, salt=key,
                                                include_gold=False)
                    builder.add("train", f"{key}|{slot}|miss", q(slot), answer,
                                {**meta, "slot": slot}, ctx, ids)
                if chance(VAL_SHARE, "val", key):
                    builder.add("validation", f"{key}|{T.VAL_SLOT}", q(T.VAL_SLOT), answer,
                                {**meta, "slot": T.VAL_SLOT, "group": "seen_closed"})
                if chance(TEST_SHARE, "test", key):
                    slot = T.TEST_SLOTS[stable_int("test-slot", key) % 2]
                    builder.add("test", f"{key}|{slot}", q(slot), answer,
                                {**meta, "slot": slot, "group": "seen_closed"})
            elif group == "rag_only":
                for slot in sorted(T.TRAIN_SLOTS, key=lambda s: stable_int("rag", key, s))[:2]:
                    ctx, ids = builder.contexts(q(slot), eid, training=True, salt=f"{key}{slot}")
                    builder.add("train", f"{key}|{slot}|open", q(slot), answer,
                                {**meta, "slot": slot}, ctx, ids)
            else:  # held-out entities: never trained on
                split = "validation" if group == "heldout_val" else "test"
                slot = T.VAL_SLOT if split == "validation" else T.TEST_SLOTS[
                    stable_int("test-slot", key) % 2]
                ctx, ids = builder.contexts(q(slot), eid, training=False, salt=key)
                builder.add(split, f"{key}|{slot}|open", q(slot), answer,
                            {**meta, "slot": slot, "group": "heldout_open"}, ctx, ids)
                if split == "test" and chance(0.5, "control", key):
                    builder.add(split, f"{key}|{slot}|closed", q(slot), answer,
                                {**meta, "slot": slot, "group": "heldout_closed"})

    # -- reverse lookups (modern term -> Ayurvedic name), seen AK entities only
    by_modern: dict[str, list[dict[str, str]]] = defaultdict(list)
    for entity in ak:
        first = re.split(r"\s+/\s+|/", entity["modern_equivalent"])[0].strip()
        by_modern[first.casefold()].append({**entity, "modern_first": first})
    for modern_key in sorted(by_modern):
        members = [e for e in by_modern[modern_key] if groups[e["id"]] == "seen"]
        if not members:
            continue
        modern = members[0]["modern_first"]
        names = [e["name"] for e in members]
        answer = T.reverse_answer(modern, names)
        key = f"rev-{slug(modern)}"
        meta = {"source": "ak", "entity": members[0]["id"], "name": modern,
                "attribute": "reverse_name", "gold": {"type": "phrase", "values": names},
                "group": "seen"}
        phrasings = T.AK_TEMPLATES["reverse_name"]
        skip = stable_int("skip", key) % 4
        for slot in (s for s in T.TRAIN_SLOTS if s != skip):
            builder.add("train", f"{key}|{slot}", phrasings[slot].format(modern=modern), answer,
                        {**meta, "slot": slot})
        if chance(VAL_SHARE, "val", key):
            builder.add("validation", f"{key}|4", phrasings[4].format(modern=modern), answer,
                        {**meta, "slot": 4, "group": "seen_closed"})
        if chance(TEST_SHARE, "test", key):
            slot = T.TEST_SLOTS[stable_int("test-slot", key) % 2]
            builder.add("test", f"{key}|{slot}", phrasings[slot].format(modern=modern), answer,
                        {**meta, "slot": slot, "group": "seen_closed"})

    # -- two-turn conversations (follow-up refers to the first topic as "it")
    rng = random.Random(SEED)
    seen_entities = [e for e in entities if groups[e["id"]] == "seen"]
    for entity in rng.sample(seen_entities, k=min(500, len(seen_entities))):
        facts = {a: (ans, gold) for a, ans, gold in
                 (ak_facts(entity) if entity["kind"] == "ak" else ag_facts(entity))}
        first, second = rng.sample([a for a in facts if a in T.FOLLOW_UPS], k=2)
        phrasings = (T.AK_TEMPLATES if entity["kind"] == "ak" else T.AG_TEMPLATES)[first]
        history = [
            {"role": "user", "content": phrasings[rng.choice(T.TRAIN_SLOTS)].format(
                name=entity["name"])},
            {"role": "assistant", "content": facts[first][0]},
        ]
        follow = rng.choice(T.FOLLOW_UPS[second])
        meta = {"source": entity["kind"], "entity": entity["id"], "name": entity["name"],
                "attribute": second, "gold": facts[second][1], "group": "seen",
                "multi_turn": True}
        key = f"{entity['id']}|{first}>{second}"
        builder.add("train", key, follow, facts[second][0], meta, history=history)
        if chance(0.4, "multi-open", key):  # the API retrieves with follow-up + last question
            ctx, ids = builder.contexts(f"{follow} {history[0]['content']}", entity["id"],
                                        training=True, salt=key)
            builder.add("train", f"{key}|open", follow, facts[second][0], meta, ctx, ids,
                        history=history)

    # -- general concepts (paraphrases 0-2 train, 3 validation, 4 test)
    for concept in concepts:
        cid, answer, questions = concept["id"], concept["answer"].strip(), concept["questions"]
        meta = {"source": "concept", "entity": f"concept-{cid}", "name": cid,
                "attribute": "concept", "gold": {"type": "terms", "values": concept["key_terms"]},
                "group": "concepts"}
        for slot in (0, 1, 2):
            builder.add("train", f"concept-{cid}|{slot}", questions[slot], answer,
                        {**meta, "slot": slot})
        gold_card = None if cid in CONVERSATIONAL else f"concept-{cid}"
        ctx, ids = builder.contexts(questions[0], gold_card, training=True, salt=cid,
                                    include_gold=gold_card is not None)
        builder.add("train", f"concept-{cid}|0|open", questions[0], answer,
                    {**meta, "slot": 0}, ctx, ids)
        builder.add("validation", f"concept-{cid}|3", questions[3], answer, {**meta, "slot": 3})
        builder.add("test", f"concept-{cid}|4", questions[4], answer, {**meta, "slot": 4})

    # -- original AcharyaGPT PDF Q&A (train only; too few to evaluate)
    for number, (question, answer) in enumerate(pdf_pairs, 1):
        builder.add("train", f"pdf-{number:02d}", question, answer,
                    {"source": "pdf", "entity": f"pdf-{number:02d}", "name": "pdf",
                     "attribute": "pdf_qa", "gold": {"type": "none"}, "group": "seen"})

    # -- safety behaviour
    fillers = safety["fillers"]
    for split in ("train", "test"):
        for kind, prompts in safety[split].items():
            for p_index, prompt in enumerate(prompts):
                fills = 3 if "{" in prompt and split == "train" else 1
                for f_index in range(fills):
                    pick = stable_int(split, kind, p_index, f_index)
                    values = {
                        "herb": fillers["herb"][pick % len(fillers["herb"])],
                        "condition": fillers["condition"][(pick // 7) % len(fillers["condition"])],
                        "drug": fillers["drug"][(pick // 11) % len(fillers["drug"])],
                    }
                    builder.add(split, f"safety-{split}-{kind}-{p_index}-{f_index}",
                                prompt.format(**values), safety["answers"][kind].format(**values),
                                {"source": "safety", "entity": "safety", "name": kind,
                                 "attribute": f"safety_{kind}",
                                 "gold": {"type": "safety", "kind": kind}, "group": "safety"})
    diagnosis = safety["diagnosis_templates"]
    candidates = [e for e in ak if groups[e["id"]] == "seen"]
    for split, count in (("train", 150), ("test", 15)):
        picks = random.Random(f"{SEED}-{split}").sample(candidates, k=count)
        for number, entity in enumerate(picks):
            symptoms = T.join_list([T.lower_first(s) for s in T.split_list(entity["symptoms"])[:3]])
            prompt = diagnosis[split][number % len(diagnosis[split])].format(symptoms=symptoms)
            answer = safety["diagnosis_answer"].format(
                symptoms=symptoms, name=entity["name"], modern=entity["modern_equivalent"])
            builder.add(split, f"safety-{split}-diagnosis-{entity['id']}", prompt, answer,
                        {"source": "safety", "entity": entity["id"], "name": entity["name"],
                         "attribute": "safety_diagnosis",
                         "gold": {"type": "safety", "kind": "diagnosis"}, "group": "safety"})

    # -- write
    output.mkdir(parents=True, exist_ok=True)
    kb_output.mkdir(parents=True, exist_ok=True)
    for split, rows in builder.rows.items():
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"duplicate row ids in {split}")
        rows = sorted(rows, key=lambda row: stable_int("order", split, row["id"]))
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (kb_output / "cards.jsonl").open("w", encoding="utf-8") as handle:
        for card in cards:
            handle.write(json.dumps(card.__dict__, ensure_ascii=False, sort_keys=True) + "\n")

    train_questions = {re.sub(r"\W+", " ", row["question"].casefold()).strip()
                       for row in builder.rows["train"]}
    leaked = [row["id"] for row in builder.rows["test"]
              if row["meta"]["group"] in {"seen_closed", "concepts", "heldout_open",
                                          "heldout_closed"}
              and re.sub(r"\W+", " ", row["question"].casefold()).strip() in train_questions]
    if leaked:
        raise RuntimeError(f"test questions also appear in training: {leaked[:5]}")
    trained_entities = {row["meta"]["entity"] for row in builder.rows["train"]}
    if trained_entities & heldout:
        raise RuntimeError("held-out entities leaked into training")
    for row in builder.rows["train"]:
        if set(row["meta"]["context_ids"]) & heldout:
            raise RuntimeError("held-out card used as a training context")

    stats: dict[str, Any] = {
        "entities": {"ayurvedic_knowledge": len(ak), "ayurgenixai": len(ag),
                     "groups": dict(Counter(groups.values()))},
        "cards": len(cards),
        "rows": {split: len(rows) for split, rows in builder.rows.items()},
        "by_group": {split: dict(Counter(r["meta"]["group"] for r in rows))
                     for split, rows in builder.rows.items()},
        "open_book": {split: sum(r["meta"]["open_book"] for r in rows)
                      for split, rows in builder.rows.items()},
        "by_attribute_test": dict(Counter(f"{r['meta']['group']}:{r['meta']['attribute']}"
                                          for r in builder.rows["test"])),
        "heldout_bm25_recall_at_3": round(
            builder.retrieval_hits["gold_in_top3"]
            / max(1, sum(builder.retrieval_hits.values())), 4),
    }
    (output / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    manifest = {
        "inputs": {**{key: spec["sha256"] for key, spec in SOURCES.items()}, "pdf": PDF_SHA256,
                   "concepts.yaml": sha256_file(ROOT / "data/curated/concepts.yaml"),
                   "pdf_qa.yaml": sha256_file(ROOT / "data/curated/pdf_qa.yaml"),
                   "safety.yaml": sha256_file(ROOT / "data/curated/safety.yaml")},
        "outputs": {path.name: sha256_file(path) for path in
                    sorted([*output.glob("*.jsonl"), kb_output / "cards.jsonl"])},
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
    }
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kaggle-dir", type=Path, help="local folder with the Kaggle CSVs")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "sft")
    parser.add_argument("--kb-output", type=Path, default=ROOT / "data" / "kb")
    args = parser.parse_args()
    print(json.dumps(build(args.kaggle_dir, args.output, args.kb_output), indent=2))


if __name__ == "__main__":
    main()
