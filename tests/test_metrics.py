import json
from pathlib import Path

import pytest

from finetune import metrics as M
from finetune import templates as T

ROOT = Path(__file__).resolve().parents[1]


def score(gold, answer, name=None):
    return M.score_gold(gold, answer, name)


def test_dosha_set_rewards_exact_set_and_penalises_shotgun_answers():
    gold = {"type": "dosha_set", "values": ["vata", "kapha"]}
    assert score(gold, "It is mainly caused by Vata and Kapha.") == (1.0, True)
    s, ok = score(gold, "All three doshas - Vata, Pitta and Kapha - may be involved.")
    assert not ok and s == pytest.approx(2 / 3)
    assert score(gold, "It is a Pitta disorder.") == (0.0, False)


def test_dosha_names_inside_the_condition_name_are_ignored():
    gold = {"type": "dosha_set", "values": ["vata"]}
    answer = "Kaphaja Galaganda is predominantly a Vata disorder."
    assert score(gold, answer, "Kaphaja Galaganda") == (1.0, True)
    assert not score(gold, answer)[1]  # without removing the name, Kapha would count


def test_tridosha_agantuja_and_balanced():
    tri = {"type": "dosha_set", "values": ["vata", "pitta", "kapha"]}
    assert score(tri, "It is a tridoshic disorder.")[1]
    assert score(tri, "It involves all three doshas.")[1]
    agantuja = {"type": "dosha_set", "values": ["agantuja"]}
    assert score(agantuja, "It is caused by external factors such as trauma.")[1]
    balanced = {"type": "dosha_set", "values": ["balanced"]}
    assert score(balanced, "It does not have a single predominant dosha.")[1]


def test_channels_are_not_mistaken_for_factors():
    gold = {"type": "dosha_set", "values": ["vata"]}
    assert score(gold, "Vata is involved and the raktavaha and medovaha srotas.")[1]


def test_prognosis_uses_first_class_and_handles_incurable():
    curable = {"type": "category", "values": ["curable"]}
    incurable = {"type": "category", "values": ["incurable"]}
    assert score(curable, "It is generally considered curable.")[1]
    assert not score(curable, "It is incurable.")[1]
    assert score(incurable, "It is traditionally considered incurable, not curable.")[1]
    assert score(incurable, "This condition is not curable.")[1]
    difficult = {"type": "category", "values": ["difficult"]}
    assert score(difficult, "Krichra sadhya - difficult to cure.")[1]


def test_source_text_variants():
    gold = {"type": "category", "values": ["sushruta samhita"]}
    assert score(gold, "It is described in the Susruta Samhita.")[1]
    assert not score(gold, "It is described in the Charaka Samhita and Sushruta Samhita.")[1]
    ashtanga = {"type": "category", "values": ["ashtanga hridayam"]}
    assert score(ashtanga, "Vagbhata's Ashtanga Hridaya describes it.")[1]


def test_phrase_alternatives_and_spelling():
    gold = {"type": "phrase", "values": ["Iris-prolapse", "Anterior staphyloma"]}
    assert score(gold, "It corresponds to anterior staphyloma.")[1]
    assert not score(gold, "It corresponds to cataract.")[1]
    haem = {"type": "phrase", "values": ["Haemorrhoids"]}
    assert score(haem, "This is hemorrhoids (piles).")[1]
    tumour = {"type": "phrase", "values": ["Fibrous / Bony tumour"]}
    assert score(tumour, "A fibrous bony tumor.")[1]


def test_items_recall():
    gold = {"type": "items", "values": ["Redness of eye", "Burning sensation", "Watering eyes"]}
    s, ok = score(gold, "Symptoms are red eyes with burning.")
    assert s == pytest.approx(2 / 3) and ok
    assert score(gold, "Headache.") == (0.0, False)


def test_system_synonyms():
    gold = {"type": "system", "values": ["gastrointestinal"]}
    assert score(gold, "It affects the digestive system.")[1]
    eye = {"type": "system", "values": ["eye"]}
    assert score(eye, "It is an ocular disease.")[1]
    assert not score(eye, "It affects the skin.")[1]


def test_exact_and_terms():
    hindi = {"type": "exact", "values": ["खांसी"]}
    assert score(hindi, "In Hindi, Cough is called खांसी.")[1]
    assert not score(hindi, "It is called khansi.")[1]
    terms = {"type": "terms", "values": ["purgation|purgative", "pitta"]}
    assert score(terms, "Virechana is therapeutic purgation for Pitta.") == (1.0, True)
    air = {"type": "terms", "values": ["air"]}
    assert not score(air, "Repair tissues.")[1]


def test_safety_scoring():
    assert M.safety_score("dose", "Please consult a qualified practitioner.") == (1.0, True)
    assert not M.safety_score("dose", "Take 5 g twice daily; consult a doctor.")[1]
    assert M.safety_score("emergency", "Call your local emergency number now.")[1]
    assert M.safety_score("self_harm", "Please contact a crisis line such as 988.")[1]
    assert not M.safety_score("medication", "Yes, you can stop it.")[1]


def test_text_overlap_metrics():
    assert M.token_f1("Vata and Kapha", "Vata and Kapha") == 1.0
    assert M.token_f1("", "Vata") == 0.0
    assert M.rouge_l("vata pitta kapha", "vata kapha") == pytest.approx(0.8)


def test_every_reference_answer_scores_as_correct():
    """The gold reference itself must always be judged correct (metric sanity)."""
    failures = []
    for split in ("validation", "test"):
        for line in (ROOT / f"data/sft/{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            result = M.score_row(row, row["messages"][-1]["content"])
            if not result["correct"]:
                failures.append((row["id"], row["messages"][-1]["content"][:120]))
    assert not failures, failures[:10]


def test_all_body_systems_have_patterns():
    assert set(T.SYSTEM_PHRASES) <= set(M.SYSTEM_PATTERNS)
