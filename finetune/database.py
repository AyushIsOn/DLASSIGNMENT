"""The AcharyaGPT knowledge database (SQLite, one file, built by finetune.build_dataset).

    sqlite3 data/db/acharya.sqlite "SELECT source, COUNT(*) FROM entities GROUP BY source"

Tables
  sources   every input file: licence, URL, SHA-256, how it is used
  entities  one row per condition / term (name, English, code, definition, split group)
  cards     the retrieval knowledge base (same content as data/kb/cards.jsonl)
  examples  every train / validation / test row with its group, attribute and gold label

The split group of each entity (seen / rag_only / heldout_val / heldout_test) is stored, so
leakage can be audited with plain SQL, e.g. "which test entities were ever trained on?":

    SELECT DISTINCT e.entity FROM examples e WHERE e.split = 'test'
      AND e.grp IN ('heldout_open', 'unseen_closed', 'heldout_closed')
      AND e.entity IN (SELECT entity FROM examples WHERE split = 'train');   -- 0 rows
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from acharya.retrieval import Card

SCHEMA = """
CREATE TABLE sources (id TEXT PRIMARY KEY, name TEXT, license TEXT, url TEXT, sha256 TEXT,
                      use TEXT);
CREATE TABLE entities (id TEXT PRIMARY KEY, source TEXT REFERENCES sources(id), name TEXT,
                       english TEXT, code TEXT, definition TEXT, parent TEXT, split_group TEXT,
                       data TEXT);
CREATE TABLE cards (id TEXT PRIMARY KEY, title TEXT, source TEXT, text TEXT);
CREATE TABLE examples (id TEXT, split TEXT, grp TEXT, source TEXT, entity TEXT,
                       attribute TEXT, open_book INTEGER, question TEXT, answer TEXT,
                       gold TEXT, messages TEXT, PRIMARY KEY (split, id));
CREATE INDEX examples_entity ON examples(entity);
CREATE INDEX entities_name ON entities(name);
CREATE TABLE build (key TEXT PRIMARY KEY, value TEXT);
"""


def _entity_row(entity: dict[str, Any], group: str) -> tuple[Any, ...]:
    english = entity.get("english") or entity.get("modern_equivalent") or ""
    definition = entity.get("definition") or entity.get("symptoms") or ""
    return (entity["id"], entity["kind"], entity["name"], english, entity.get("code", ""),
            definition, entity.get("parent"), group,
            json.dumps(entity, ensure_ascii=False, sort_keys=True))


def write(path: Path, *, sources: list[dict[str, Any]], entities: list[dict[str, Any]],
          groups: dict[str, str], cards: list[Card], rows: dict[str, list[dict[str, Any]]],
          stats: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".tmp")
    staging.unlink(missing_ok=True)
    connection = sqlite3.connect(staging)
    try:
        connection.executescript(SCHEMA)
        connection.executemany("INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?)",
                               [(s["id"], s["name"], s["license"], s["url"], s["sha256"],
                                 s["use"]) for s in sources])
        connection.executemany("INSERT INTO entities VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                               [_entity_row(e, groups.get(e["id"], "")) for e in entities])
        connection.executemany("INSERT INTO cards VALUES (?, ?, ?, ?)",
                               [(c.id, c.title, c.source, c.text) for c in cards])
        connection.executemany(
            "INSERT INTO examples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(row["id"], split, row["meta"]["group"], row["meta"]["source"],
              row["meta"]["entity"], row["meta"]["attribute"], int(row["meta"]["open_book"]),
              row["question"], row["messages"][-1]["content"],
              json.dumps(row["meta"].get("gold"), ensure_ascii=False),
              json.dumps(row["messages"], ensure_ascii=False))
             for split, split_rows in rows.items() for row in split_rows])
        connection.execute("INSERT INTO build VALUES ('stats', ?)", (json.dumps(stats),))
        connection.commit()
        leaked = connection.execute(
            "SELECT COUNT(DISTINCT entity) FROM examples WHERE split != 'train' "
            "AND grp IN ('heldout_open', 'unseen_closed', 'heldout_closed') "
            "AND entity IN (SELECT entity FROM examples WHERE split = 'train')").fetchone()[0]
        if leaked:
            raise RuntimeError(f"{leaked} held-out entities appear in training (database check)")
        connection.execute("VACUUM")
    finally:
        connection.close()
    staging.replace(path)
    return path
