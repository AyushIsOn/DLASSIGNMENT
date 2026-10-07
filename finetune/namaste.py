"""NAMASTE (Ministry of Ayush, Government of India) terminology -> entities, facts, cards.

Two official files from https://namaste.ayush.gov.in/ are committed under data/raw/namaste/
(pinned by SHA-256 in build_dataset.SOURCES):

  NATIONAL_AYURVEDA_MORBIDITY_CODES.xls  2,910 Ayurveda diagnostic terms: NAMASTE code,
      IAST + Devanagari term, English name, clinical definition, ICD-11 TM2 code (some)
  ayu_sat_table_combined.xlsx            Standardised Ayurveda Terminology (SAT): 14,968
      terms (fundamentals, body, signs/symptoms, substances, pharmacy, diet, treatment,
      prevention) with English translation, definition and classical references

Facts are short natural sentences built from these fields. Morbidity rows are the source of
truth for diagnostic terms; the SAT copy of the diagnostic section is used only to add the
classical references, so the same disease is never two entities.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

from acharya.retrieval import Card, fold
from finetune.templates import join_list

MORBIDITY_FILE = "NATIONAL_AYURVEDA_MORBIDITY_CODES.xls"
SAT_FILE = "ayu_sat_table_combined.xlsx"
CITATION = "NAMASTE portal, Ministry of Ayush, Government of India"
NM_CITATION = f"{CITATION} - National Ayurveda Morbidity Codes"
SAT_CITATION = f"{CITATION} - Standardised Ayurveda Terminology"
EMPTY = {"", "-", "nan", "none", "null"}
MAX_DEFINITION_WORDS = 70
MAX_FEATURES = 12  # a few definitions list 40+ features; answers stay readable

# ---------------------------------------------------------------- templates
# 0-3 train, 4 validation, 5-6 test. Slot 6 is asked with the diacritic-free spelling
# ("vatavyadhih" instead of "vātavyādhiḥ"), the way people actually type.
ASCII_SLOT = 6

NM_TEMPLATES: dict[str, tuple[str, ...]] = {
    "nm_english": (
        "What does the Ayurvedic diagnostic term {name} mean?",
        "What is {name} in English?",
        "Translate the NAMASTE term {name} into English.",
        "What condition does {name} refer to?",
        "What is the English name of {name}?",
        "Explain the meaning of the Ayurvedic disease term {name}.",
        "What does {name} mean?",
    ),
    "nm_features": (
        "What are the features of {name}?",
        "How is {name} characterized?",
        "What are the clinical features of {name} according to NAMASTE?",
        "Describe the presentation of {name}.",
        "Which signs characterize {name}?",
        "What are the characteristic features of {name} in Ayurveda?",
        "What symptoms define {name}?",
    ),
    "nm_category": (
        "Under which disease group is {name} classified?",
        "{name} is a subtype of which Ayurvedic condition?",
        "What is the parent category of {name} in the NAMASTE codes?",
        "Which broader disorder does {name} belong to?",
        "In NAMASTE, {name} falls under which category?",
        "To which group of disorders does {name} belong?",
        "Which category is {name} listed under?",
    ),
    "nm_code": (
        "What is the NAMASTE code for {name}?",
        "Give the NAMASTE morbidity code of {name}.",
        "Which NAMASTE code is assigned to {name}?",
        "What code does the NAMASTE portal use for {name}?",
        "What is the national Ayurveda morbidity code for {name}?",
        "Tell me the NAMASTE code of {name}.",
        "Which morbidity code identifies {name} in NAMASTE?",
    ),
    "nm_reverse": (
        "What is the Ayurvedic term for {name}?",
        "Which NAMASTE term means {name}?",
        "What is {name} called in Ayurvedic terminology?",
        "Give the Sanskrit diagnostic term for {name}.",
        "Which Ayurvedic diagnosis corresponds to {name}?",
        "How is {name} named in the Ayurveda morbidity codes?",
        "What Ayurvedic term is used for {name}?",
    ),
    "reference": (
        "Which classical text describes {name}?",
        "Where is {name} mentioned in the classical Ayurvedic texts?",
        "Give the classical reference for {name}.",
        "In which samhita is {name} described?",
        "What is the textual source for {name}?",
        "Which Ayurvedic classic mentions {name}?",
        "Where can I find {name} in the classical literature?",
    ),
}

SAT_TEMPLATES: dict[str, tuple[str, ...]] = {
    "sat_meaning": (
        "What does the Ayurvedic term {name} mean?",
        "What is the meaning of {name}?",
        "Define {name} in Ayurveda.",
        "Translate the Sanskrit term {name}.",
        "What is meant by {name} in Ayurveda?",
        "Explain the term {name}.",
        "What does {name} mean?",
    ),
    "sat_category": (
        "Which group of Ayurvedic terms does {name} belong to?",
        "Under which heading is {name} classified in the Standardised Ayurveda Terminology?",
        "{name} is a type of what?",
        "What is the broader category of {name}?",
        "{name} falls under which concept?",
        "Which concept is {name} a part of?",
        "What category does {name} come under?",
    ),
    "reference": NM_TEMPLATES["reference"],
}

FOLLOW_UPS = {
    "nm_english": ("What does it mean in English?", "What is it in English?"),
    "nm_features": ("What are its features?", "How is it characterized?"),
    "nm_category": ("Which group is it classified under?", "What is its parent category?"),
    "nm_code": ("What is its NAMASTE code?", "Which code does it have?"),
    "sat_meaning": ("What does it mean?", "Can you define it?"),
    "reference": ("Which text describes it?", "Where is it mentioned in the classics?"),
}

# Classical text abbreviations used in the SAT references (e.g. "Ca.Su.20/17").
BOOKS = {
    "Ca": "Charaka Samhita",
    "Su": "Sushruta Samhita",
    "Ah": "Ashtanga Hridayam",
    "As": "Ashtanga Sangraha",
    "Ma": "Madhava Nidana",
    "Bp": "Bhavaprakasha",
    "Sha": "Sharangadhara Samhita",
}


# ---------------------------------------------------------------- cleaning


def text(value: Any) -> str:
    value = "" if value is None else str(value)
    value = unicodedata.normalize("NFC", value.replace("\xa0", " ").replace("⇒", " "))
    value = re.sub(r"#san@([^#]+)#", r"\1", value)  # SAT markup around Sanskrit words
    value = re.sub(r"\s+", " ", value).strip()
    return "" if value.casefold() in EMPTY else value


def term_name(value: str) -> tuple[str, list[str]]:
    """'(a) vātaja-vṛddhiḥ' -> 'vātaja-vṛddhiḥ'; 'a/ b/ c' -> ('a', ['b', 'c'])."""
    value = re.sub(r"^\([a-z]\)\s*", "", text(value))
    names = [part.strip(" ,;") for part in re.split(r"\s*/\s*", value) if part.strip(" ,;")]
    return (names[0], names[1:]) if names else ("", [])


def english_name(value: str) -> str:
    value = text(value).replace("(TM2)", "")
    return re.sub(r"\s+", " ", value).strip(" ,;")


def ascii_name(name: str) -> str:
    """Diacritic-free spelling a user would type ('vātavyādhiḥ' -> 'vatavyadhih')."""
    folded = unicodedata.normalize("NFKD", name)
    return "".join(ch for ch in folded if not unicodedata.combining(ch))


def group_key(name: str) -> str:
    """Spelling-insensitive identity used to keep duplicates in the same split."""
    return re.sub(r"[^a-z0-9]", "", fold(name))


def without_parentheses(value: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", value).strip()


def short_definition(value: str) -> str:
    words = value.split()
    if len(words) <= MAX_DEFINITION_WORDS:
        return value.rstrip(" .")
    cut = " ".join(words[:MAX_DEFINITION_WORDS])
    end = max(cut.rfind(". "), cut.rfind("; "))
    return (cut[:end] if end > len(cut) // 2 else cut).rstrip(" .;,")


def sentence(value: str) -> str:
    value = value.strip()
    return value if value.endswith((".", "!", "?")) else value + "."


def capitalize(value: str) -> str:
    return value[:1].upper() + value[1:]


GLOSS = re.compile(r"([^,\[\]]+?)\s*\[([^\]]+)\]")


def features(definition: str) -> tuple[str, list[str]]:
    """Definition -> (answer clause, gold items).

    'the disorder is characterized by jvaraḥ [fever], śītam [feeling of cold]' ->
    ('fever (jvaraḥ) and feeling of cold (śītam)', ['fever', 'feeling of cold'])
    """
    glosses = [(sanskrit.strip(" ,;:"), gloss.strip()) for sanskrit, gloss in
               GLOSS.findall(definition)]
    glosses = [(s.split(" by ")[-1].strip(), g) for s, g in glosses if g][:MAX_FEATURES]
    if glosses:
        items = [g for _, g in glosses]
        return join_list([f"{g} ({s})" if s else g for s, g in glosses]), items
    body = re.split(r"characteri[sz]ed by:?", definition, maxsplit=1, flags=re.I)[-1]
    body = body.split(". This may be explained by")[0].strip(" .")
    clause = short_definition(body)
    items = [item.strip() for item in re.split(r",|\band\b|;", clause) if item.strip()]
    return clause, items[:MAX_FEATURES]


def references(value: str) -> list[tuple[str, str]]:
    """'Ca.Su.20/17; Su.U. 43/6' -> [('Charaka Samhita', 'Ca.Su.20/17'), ...] (book order kept,
    one entry per book)."""
    found: list[tuple[str, str]] = []
    for part in re.split(r";", text(value)):
        part = part.strip()
        match = re.match(r"([A-Z][a-z]{0,2})\.", part)
        if not match or match.group(1) not in BOOKS:
            continue
        book = BOOKS[match.group(1)]
        if all(book != existing for existing, _ in found):
            found.append((book, part.replace(" ", "")))
    return found


def reference_answer(name: str, refs: list[tuple[str, str]]) -> str:
    parts = [f"the {book} ({ref})" for book, ref in refs]
    return f"{capitalize(name)} is described in {join_list(parts)}."


def reference_gold(refs: list[tuple[str, str]]) -> dict[str, Any]:
    return {"type": "source_any", "values": [book.casefold() for book, _ in refs]}


# ---------------------------------------------------------------- loading


def read_morbidity(path: Path) -> list[dict[str, str]]:
    import xlrd  # "data" extra

    sheet = xlrd.open_workbook(str(path)).sheet_by_index(0)
    header = [str(cell.value).strip() for cell in sheet.row(0)]
    rows = []
    for index in range(1, sheet.nrows):
        values = []
        for cell in sheet.row(index):
            value = cell.value
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            values.append(str(value))
        rows.append(dict(zip(header, values, strict=True)))
    return rows


def read_sat(path: Path) -> list[dict[str, str]]:
    import openpyxl  # "data" extra

    sheet = openpyxl.load_workbook(path, read_only=True).active
    rows = list(sheet.iter_rows(values_only=True))
    header = [str(value).strip() for value in rows[0]]
    return [dict(zip(header, ["" if value is None else str(value) for value in row],
                     strict=True)) for row in rows[1:]]


TM2 = re.compile(r"^S[A-Z][0-9][0-9A-Z]$")


def split_code(raw: str) -> tuple[str, str]:
    """'EI-6.2 (SN3Y)' -> ('EI-6.2', 'SN3Y'); 'SK90 (I-15)' -> ('I-15', 'SK90')."""
    tokens = [token for token in re.split(r"[\s()]+", text(raw)) if token]
    tm2 = next((token for token in tokens if TM2.match(token)), "")
    rest = [token for token in tokens if token != tm2]
    return (rest[0] if rest else tm2), tm2


def load_morbidity(path: Path, sat_rows: list[dict[str, str]],
                   expected_rows: int) -> list[dict[str, Any]]:
    rows = read_morbidity(path)
    if len(rows) != expected_rows:
        raise RuntimeError(f"{path.name}: expected {expected_rows} rows, found {len(rows)}")
    # classical references live in the SAT copy of the diagnostic section (same codes)
    sat_refs = {text(row["term_id"]): text(row["refn"]) for row in sat_rows
                if not text(row["term_id"]).startswith("SAT")}
    by_code: dict[str, dict[str, Any]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    entities = []
    for row in rows:
        raw_code = text(row["NAMC_CODE"])
        name, synonyms = term_name(row["NAMC_term_diacritical"])
        if not name or not raw_code:
            continue  # 14% have no IAST spelling; their HK-style transliteration is unreadable
        code, tm2 = split_code(raw_code)
        definition = text(row["Long_definition"])
        entity = {
            "id": f"nm-{text(row['NAMC_ID'])}",
            "kind": "nm",
            "name": name,
            "synonyms": synonyms,
            "devanagari": text(row["NAMC_term_DEVANAGARI"]),
            "english": english_name(row["Name English"]),
            "code": code,
            "tm2": tm2,
            "definition": definition,
            "reference": sat_refs.get(raw_code, "") or sat_refs.get(code, ""),
            "parent": None,
        }
        if not entity["english"]:
            continue
        twin = by_id.get(entity["id"])
        if twin is not None:  # 18 terms are listed twice (once more under their TM2 code)
            for field in ("definition", "tm2", "reference", "devanagari"):
                twin[field] = twin[field] or entity[field]
            continue
        by_id[entity["id"]] = entity
        entities.append(entity)
        by_code.setdefault(code, entity)
    for entity in entities:
        code = entity["code"]
        for separator in (".", "-"):
            if separator in code:
                parent = by_code.get(code.rsplit(separator, 1)[0])
                if parent is not None and parent is not entity:
                    entity["parent"] = parent["id"]
                    break
    return entities


def load_sat(sat_rows: list[dict[str, str]], expected_rows: int) -> list[dict[str, Any]]:
    if len(sat_rows) != expected_rows:
        raise RuntimeError(f"SAT: expected {expected_rows} rows, found {len(sat_rows)}")
    entities: list[dict[str, Any]] = []
    by_tid: dict[tuple[str, str], dict[str, Any]] = {}
    roots: set[str] = set()
    for row in sat_rows:
        term_id = text(row["term_id"]).replace(" ", "")
        if not term_id.startswith("SAT-"):
            continue  # the diagnostic section: covered by the morbidity codes
        section = term_id.split(".")[0]
        name, synonyms = term_name(row["term_iast"])
        meaning = text(row["w_trans"])
        definition = text(row["w_def"])
        if definition.casefold() == meaning.casefold():
            definition = ""
        entity = {
            "id": "sat-" + term_id.casefold().replace(".", "-"),
            "kind": "sat",
            "name": name,
            "synonyms": synonyms,
            "devanagari": text(row["term_devanagari"]),
            "english": meaning,
            "definition": definition,
            "reference": text(row["refn"]),
            "code": term_id,
            "parent_tid": (section, text(row["parent_id"])),
            "parent": None,
        }
        by_tid[(section, text(row["t_id"]))] = entity
        if "." not in term_id:
            roots.add(entity["id"])
        if name and (meaning or definition):
            entities.append(entity)
    for entity in entities:
        parent = by_tid.get(entity.pop("parent_tid"))
        # the section roots ("fundamental terms", "signs and symptoms") are too generic
        if parent and parent["id"] not in roots and parent is not entity and parent["name"] \
                and (parent["english"] or parent["definition"]):
            entity["parent"] = parent["id"]
    return entities


# ---------------------------------------------------------------- facts


def nm_facts(entity: dict[str, Any], by_id: dict[str, dict[str, Any]],
             allow_parent: bool) -> list[tuple[str, str, dict[str, Any]]]:
    name, english = entity["name"], entity["english"]
    deva = f" ({entity['devanagari']})" if entity["devanagari"] else ""
    english_values = sorted({english, without_parentheses(english)} - {""})
    facts = [
        ("nm_english", f"{capitalize(name)}{deva} is the Ayurvedic diagnostic term for "
                       f"{english}.", {"type": "phrase", "values": english_values}),
    ]
    code_answer = f"The NAMASTE code for {name} is {entity['code']}."
    if entity["tm2"]:
        code_answer += (" It maps to the ICD-11 Traditional Medicine (TM2) code "
                        f"{entity['tm2']}.")
    facts.append(("nm_code", code_answer, {"type": "exact", "values": [entity["code"]]}))
    if entity["definition"]:
        clause, items = features(entity["definition"])
        if items:
            facts.append(("nm_features", f"{capitalize(name)} is characterized by "
                                         f"{sentence(clause)}",
                          {"type": "items", "values": items}))
    parent = by_id.get(entity["parent"] or "")
    if parent and allow_parent:
        facts.append(("nm_category",
                      f"{capitalize(name)} is classified under {parent['english']} "
                      f"({parent['name']}, NAMASTE code {parent['code']}).",
                      {"type": "phrase", "values": sorted(
                          {parent["english"], without_parentheses(parent["english"])} - {""})}))
    refs = references(entity["reference"])
    if refs:
        facts.append(("reference", reference_answer(name, refs), reference_gold(refs)))
    return facts


def sat_facts(entity: dict[str, Any], by_id: dict[str, dict[str, Any]],
              allow_parent: bool) -> list[tuple[str, str, dict[str, Any]]]:
    name, meaning = entity["name"], entity["english"]
    definition = short_definition(entity["definition"]) if entity["definition"] else ""
    deva = f" ({entity['devanagari']})" if entity["devanagari"] else ""
    if meaning and definition:
        answer = f"{capitalize(name)}{deva} means \"{meaning}\": {sentence(definition)}"
    else:
        answer = f"{capitalize(name)}{deva} means {sentence(meaning or definition)}"
    values = [value for value in (meaning, definition) if value]
    facts = [("sat_meaning", answer, {"type": "phrase", "values": values})]
    parent = by_id.get(entity["parent"] or "")
    if parent and allow_parent:
        label = parent["english"] or short_definition(parent["definition"])
        facts.append(("sat_category",
                      f"In the Standardised Ayurveda Terminology, {name} is listed under "
                      f"{label} ({parent['name']}).",
                      {"type": "phrase", "values": [label, parent["name"]]}))
    refs = references(entity["reference"])
    if refs:
        facts.append(("reference", reference_answer(name, refs), reference_gold(refs)))
    return facts


def reverse_facts(entities: list[dict[str, Any]]) -> list[tuple[str, str, str, dict[str, Any]]]:
    """English name -> NAMASTE term(s): (key, english, answer, gold)."""
    by_english: dict[str, list[dict[str, Any]]] = {}
    for entity in entities:
        by_english.setdefault(entity["english"].casefold(), []).append(entity)
    result = []
    for key in sorted(by_english):
        members = by_english[key]
        english = members[0]["english"]
        names = [member["name"] for member in members]
        answer = f"The Ayurvedic (NAMASTE) term for {english} is {join_list(names)}."
        result.append((members[0]["id"], english, answer, {"type": "phrase", "values": names}))
    return result


# ---------------------------------------------------------------- cards


def nm_card(entity: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> Card:
    codes = f"NAMASTE code {entity['code']}"
    if entity["tm2"]:
        codes += f", ICD-11 TM2 {entity['tm2']}"
    parts = [f"{entity['name']} ({entity['devanagari']}; {codes}): {entity['english']}."]
    if entity["synonyms"]:
        parts.append(f"Also called {join_list(entity['synonyms'])}.")
    parent = by_id.get(entity["parent"] or "")
    if parent:
        parts.append(f"Category: {parent['english']} ({parent['name']}).")
    if entity["definition"]:
        parts.append(sentence(capitalize(short_definition(entity["definition"]))))
    if entity["reference"]:
        parts.append(f"Classical references: {entity['reference']}.")
    title = " ".join([entity["name"], *entity["synonyms"], entity["english"]])
    return Card(entity["id"], title, NM_CITATION, " ".join(parts))


def sat_card(entity: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> Card:
    parts = [f"{entity['name']} ({entity['devanagari']}; SAT {entity['code']})"
             + (f": {entity['english']}." if entity["english"] else ".")]
    if entity["synonyms"]:
        parts.append(f"Also called {join_list(entity['synonyms'])}.")
    if entity["definition"]:
        parts.append(sentence(capitalize(short_definition(entity["definition"]))))
    parent = by_id.get(entity["parent"] or "")
    if parent:
        parts.append(f"Category: {parent['english'] or parent['name']} ({parent['name']}).")
    if entity["reference"]:
        parts.append(f"Classical references: {entity['reference']}.")
    title = " ".join([entity["name"], *entity["synonyms"]])
    return Card(entity["id"], title, SAT_CITATION, " ".join(parts))
