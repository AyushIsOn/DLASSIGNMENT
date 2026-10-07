"""Question templates and answer writers for the knowledge-base attributes.

Every attribute has 7 question phrasings:
  0-3 -> training (each fact is trained with 3 of these 4, rotating per entity)
  4   -> validation
  5-6 -> test (never seen in training)
So the test measures whether the model learned the *fact*, not one fixed sentence.
"""

from __future__ import annotations

import re

TRAIN_SLOTS = (0, 1, 2, 3)
VAL_SLOT = 4
TEST_SLOTS = (5, 6)

AK_TEMPLATES: dict[str, tuple[str, ...]] = {
    "modern_equivalent": (
        "What is the modern medical equivalent of {name}?",
        "{name} corresponds to which condition in modern medicine?",
        "What is {name} called in modern medical terms?",
        "Which modern disease is {name} equivalent to?",
        "In modern medicine, what is {name} known as?",
        "What would a modern doctor call {name}?",
        "Give the modern medical name for the Ayurvedic condition {name}.",
    ),
    "dosha": (
        "Which dosha is predominant in {name}?",
        "What is the dosha predominance of {name}?",
        "Which doshas are involved in {name}?",
        "{name} is mainly caused by which dosha?",
        "Which dosha is mainly affected in {name}?",
        "According to Ayurveda, which dosha dominates in {name}?",
        "What is the doshic involvement in {name}?",
    ),
    "body_system": (
        "Which body system does {name} affect?",
        "What part of the body is affected by {name}?",
        "{name} is a disease of which body system?",
        "Which organ system is involved in {name}?",
        "Where in the body does {name} occur?",
        "Which system of the body is affected in {name}?",
        "{name} belongs to which category of body system disorders?",
    ),
    "prognosis": (
        "What is the prognosis of {name}?",
        "Is {name} curable according to Ayurveda?",
        "How curable is {name}?",
        "What is the expected outcome of treating {name}?",
        "Can {name} be cured?",
        "How does Ayurveda describe the prognosis of {name}?",
        "Is {name} easy or difficult to treat in Ayurveda?",
    ),
    "symptoms": (
        "What are the symptoms of {name}?",
        "What are the signs of {name}?",
        "How does {name} present?",
        "List the clinical features of {name}.",
        "What symptoms does {name} cause?",
        "Which symptoms are seen in {name}?",
        "What are the main clinical features of {name} in Ayurveda?",
    ),
    "source_text": (
        "Which classical text describes {name}?",
        "In which Ayurvedic text is {name} described?",
        "What is the classical source for {name}?",
        "Which samhita mentions {name}?",
        "Where is {name} described in the classical Ayurvedic literature?",
        "Which Ayurvedic classic is the source for {name}?",
        "Name the classical text that describes {name}.",
    ),
    "treatment": (
        "What are the Ayurvedic treatment principles for {name}?",
        "How is {name} managed in Ayurveda?",
        "What is the line of treatment for {name}?",
        "How does Ayurveda treat {name}?",
        "What treatment does Ayurveda describe for {name}?",
        "What is the Ayurvedic management of {name}?",
        "Which therapies are used for {name} in Ayurveda?",
    ),
    "overview": (
        "Tell me about {name}.",
        "What is {name}?",
        "Explain {name} in Ayurveda.",
        "Give an overview of {name}.",
        "Describe the Ayurvedic condition {name}.",
        "What do you know about {name}?",
        "Can you explain what {name} is?",
    ),
    "reverse_name": (
        "What is the Ayurvedic name for {modern}?",
        "What is {modern} called in Ayurveda?",
        "Which Ayurvedic condition corresponds to {modern}?",
        "What is the Ayurvedic term for {modern}?",
        "How is {modern} known in Ayurveda?",
        "What Ayurvedic disease is equivalent to {modern}?",
        "Give the Ayurvedic name of {modern}.",
    ),
}

AG_TEMPLATES: dict[str, tuple[str, ...]] = {
    "doshas": (
        "Which doshas are involved in {name} according to Ayurveda?",
        "What is the Ayurvedic dosha imbalance in {name}?",
        "Which dosha is affected in {name}?",
        "In Ayurveda, {name} is caused by an imbalance of which doshas?",
        "Which doshas does Ayurveda associate with {name}?",
        "From an Ayurvedic point of view, which doshas are disturbed in {name}?",
        "What doshas are linked to {name} in Ayurveda?",
    ),
    "prakriti": (
        "Which prakriti is more prone to {name}?",
        "People of which constitution are more likely to get {name}?",
        "Which body constitution is susceptible to {name} in Ayurveda?",
        "What prakriti is associated with {name}?",
        "Which Ayurvedic body type is prone to {name}?",
        "Which constitution (prakriti) is linked to {name}?",
        "Who is more susceptible to {name} according to Ayurvedic constitution?",
    ),
    "symptoms": (
        "What are the symptoms of {name}?",
        "What are the common signs of {name}?",
        "How does {name} present?",
        "What symptoms are seen in {name}?",
        "Which symptoms does {name} cause?",
        "What are the typical symptoms of {name}?",
        "List the symptoms of {name}.",
    ),
    "herbs": (
        "Which Ayurvedic herbs are used for {name}?",
        "What herbs does Ayurveda recommend for {name}?",
        "Name the Ayurvedic herbs traditionally used in {name}.",
        "Which herbal remedies are used in Ayurveda for {name}?",
        "What Ayurvedic herbs help with {name}?",
        "Which herbs are traditionally used to manage {name} in Ayurveda?",
        "What are the Ayurvedic herbs for {name}?",
    ),
    "yoga": (
        "What yoga practices are recommended for {name}?",
        "Which yoga or physical therapy helps {name}?",
        "What exercises does Ayurveda suggest for {name}?",
        "Which yoga is useful in {name}?",
        "What kind of yoga is suggested for {name}?",
        "Which yoga and physical therapies are used for {name}?",
        "Recommend yoga practices for {name}.",
    ),
    "diet_lifestyle": (
        "What diet and lifestyle changes are recommended for {name}?",
        "What should be the diet for {name} according to Ayurveda?",
        "What lifestyle advice does Ayurveda give for {name}?",
        "Which foods and habits are suggested in {name}?",
        "What dietary advice is given for {name}?",
        "What are the diet and lifestyle recommendations for {name}?",
        "How should someone with {name} adjust diet and lifestyle?",
    ),
    "hindi_name": (
        "What is {name} called in Hindi?",
        "What is the Hindi name for {name}?",
        "How do you say {name} in Hindi?",
        "Translate {name} into Hindi.",
        "What is the Hindi word for {name}?",
        "Give the Hindi name of {name}.",
        "In Hindi, what is {name} known as?",
    ),
    "overview": (
        "Tell me about {name} in Ayurveda.",
        "What is the Ayurvedic view of {name}?",
        "Explain {name} from an Ayurvedic perspective.",
        "Give an Ayurvedic overview of {name}.",
        "How does Ayurveda describe {name}?",
        "What does Ayurveda say about {name}?",
        "Describe {name} according to Ayurveda.",
    ),
}

# Follow-up questions for two-turn conversations ("it" refers to the first topic).
FOLLOW_UPS: dict[str, tuple[str, ...]] = {
    "modern_equivalent": ("What is its modern equivalent?",
                          "What is it called in modern medicine?"),
    "dosha": ("Which dosha is involved in it?", "Which dosha does it mainly involve?"),
    "body_system": ("Which part of the body does it affect?", "Which body system is involved?"),
    "prognosis": ("Is it curable?", "What is its prognosis?"),
    "symptoms": ("What are its symptoms?", "How does it present?"),
    "source_text": ("Which text describes it?", "Where is it described?"),
    "treatment": ("How is it treated in Ayurveda?", "What is its line of treatment?"),
    "doshas": ("Which doshas are involved in it?", "What is its dosha imbalance?"),
    "prakriti": ("Which prakriti is prone to it?", "Which constitution is more prone to it?"),
    "herbs": ("Which herbs are used for it?", "What herbs help with it?"),
    "yoga": ("What yoga helps with it?", "Which yoga practices are suggested for it?"),
    "diet_lifestyle": ("What diet is advised for it?", "What lifestyle changes help with it?"),
    "hindi_name": ("What is it called in Hindi?", "What is its Hindi name?"),
}

# ---------------------------------------------------------------- answer writers

DOSHA_PHRASES = {
    "vata": "Vata",
    "pitta": "Pitta",
    "kapha": "Kapha",
    "rakta": "Rakta (blood)",
    "meda": "Meda (fat tissue)",
    "krimi": "Krimi (parasites)",
    "ama": "Ama (undigested metabolic waste)",
    "rajas": "Rajas",
    "tamas": "Tamas",
}

SYSTEM_PHRASES = {
    "systemic": "the whole body (it is a systemic condition)",
    "skin": "the skin",
    "gastrointestinal": "the gastrointestinal (digestive) system",
    "eye": "the eyes",
    "nervous": "the nervous system",
    "urinary": "the urinary system",
    "musculoskeletal": "the musculoskeletal system",
    "gynaecology": "the female reproductive system (it is a gynaecological condition)",
    "gyn": "the female reproductive system",
    "respiratory": "the respiratory system",
    "resp": "the respiratory system",
    "nose": "the nose",
    "throat": "the throat",
    "mouth": "the mouth",
    "ear": "the ears",
    "ano-rectal": "the anorectal region",
    "mental": "the mind (it is a mental disorder)",
    "paediatric": "children (it is a paediatric condition)",
    "metabolic": "metabolism",
    "cardiovascular": "the cardiovascular system",
    "abdomen": "the abdomen",
    "blood": "the blood",
    "head": "the head",
    "teeth": "the teeth",
    "reproductive": "the reproductive system",
    "thyroid": "the thyroid gland",
    "liver": "the liver",
    "lymphatic": "the lymphatic system",
    "genital": "the genital organs",
    "parasitic": "the body through parasites (it is a parasitic condition)",
    "gums": "the gums",
    "surgical": "the body in a way that needs surgical care (it is a surgical condition)",
    "obstetric": "pregnancy and childbirth (it is an obstetric condition)",
    "chest": "the chest",
    "spleen": "the spleen",
    "geriatric": "the elderly (it is a geriatric condition)",
    "deep": "the deeper tissues",
    "deep tissue": "the deeper tissues",
    "nail": "the nails",
    "hair": "the hair",
    "pancreas": "the pancreas",
    "biliary": "the biliary system",
    "uterus": "the uterus",
}

PROGNOSIS_SENTENCES = {
    "curable": "{name} is generally considered curable.",
    "moderate": "The prognosis of {name} is moderate.",
    "difficult": "{name} is considered difficult to cure.",
    "variable": "The prognosis of {name} is variable and depends on the individual case.",
    "incurable": "{name} is traditionally considered incurable.",
}

TREATMENT_NOTE = "This is educational information, not a prescription."
HERB_NOTE = "Please consult a qualified practitioner before using any herb."


def join_list(items: list[str]) -> str:
    items = [item for item in items if item]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def split_list(value: str, separators: str = ",;") -> list[str]:
    """Split a comma list without breaking inside parentheses."""
    items, depth, current = [], 0, []
    for char in value:
        depth += char == "("
        depth -= char == ")"
        if char in separators and depth <= 0:
            items.append("".join(current))
            current = []
        else:
            current.append(char)
    items.append("".join(current))
    return [re.sub(r"\s+", " ", item).strip(" .") for item in items if item.strip(" .")]


def lower_first(text: str) -> str:
    """Lower-case the first letter of list items unless they look like proper nouns."""
    words = text.split()
    if not words:
        return text
    first = words[0]
    if first.isupper() or any(char.isupper() for char in first[1:]):
        return text
    return first[:1].lower() + text[1:]


def dosha_factors(value: str) -> list[str]:
    """Parse a 'Dosha Predominance' value into ordered factors."""
    factors: list[str] = []
    for part in re.split(r"[/,\-]", value.casefold()):
        part = part.strip()
        if not part:
            continue
        if part.startswith("tridosh"):
            names = ["vata", "pitta", "kapha"]
        elif part in {"agantuja", "balanced"} or part in DOSHA_PHRASES:
            names = [part]
        elif part.endswith("prakriti"):
            continue
        else:
            raise ValueError(f"unknown dosha factor: {part!r} in {value!r}")
        for name in names:
            if name not in factors:
                factors.append(name)
    return factors


def _factor_phrase(factor: str) -> str:
    return "external causes (Agantuja)" if factor == "agantuja" else DOSHA_PHRASES[factor]


def dosha_answer(name: str, value: str) -> str:
    """One sentence about the dosha predominance; `name` may be a pronoun like 'It'."""
    raw = value.casefold().strip()
    factors = dosha_factors(value)
    possessive = "Its" if name == "It" else f"The predominant factors in {name} are"
    if raw == "tridosha":
        return f"{name} is a tridoshic disorder involving all three doshas - Vata, Pitta and Kapha."
    if factors == ["agantuja"]:
        return (
            f"{name} is classified as Agantuja: it is caused by external factors such as injury, "
            "poison or bites rather than primarily by a dosha imbalance."
        )
    if factors == ["balanced"]:
        return f"{name} does not have a single predominant dosha."
    if factors in (["rajas"], ["rajas", "tamas"]):
        plural = "s" if len(factors) > 1 else ""
        mental = join_list([DOSHA_PHRASES[f] for f in factors])
        return f"{name} is a disorder of the mind involving the mental dosha{plural} {mental}."
    if "tridosh" in raw:
        others = [_factor_phrase(f) for f in factors if f not in {"vata", "pitta", "kapha"}]
        lead = next((DOSHA_PHRASES[p.strip()] for p in re.split(r"[/\-]", raw)
                     if p.strip() in {"vata", "pitta", "kapha"}), None)
        detail = f", with {lead} most prominent" if lead else ""
        extra = f" along with {join_list(others)}" if others else ""
        return f"{name} involves all three doshas (Vata, Pitta and Kapha){extra}{detail}."
    phrases = [_factor_phrase(f) for f in factors]
    if all(f in {"vata", "pitta", "kapha"} for f in factors):
        if len(factors) == 1:
            return f"{name} is predominantly a {phrases[0]} disorder."
        return f"{name} is mainly caused by {join_list(phrases)}."
    if name == "It":
        return f"{possessive} predominant factors are {join_list(phrases)}."
    return f"{possessive} {join_list(phrases)}."


def system_phrase(value: str) -> str:
    parts = [part.strip().casefold() for part in value.split("/") if part.strip()]
    phrases = []
    for part in parts:
        phrase = SYSTEM_PHRASES.get(part)
        if phrase is None:
            raise ValueError(f"unknown body system: {part!r}")
        phrases.append(phrase)
    return join_list(phrases)


def system_answer(name: str, value: str) -> str:
    parts = [part.strip().casefold() for part in value.split("/") if part.strip()]
    if parts == ["systemic"]:
        return f"{name} is a systemic condition that affects the whole body."
    if parts == ["mental"]:
        return f"{name} is a mental disorder: it affects the mind."
    if parts == ["paediatric"]:
        return f"{name} is a paediatric condition, seen in children."
    return f"{name} mainly affects {system_phrase(value)}."


SOURCE_KEYS = {
    "charaka samhita": "charaka",
    "sushruta samhita": "sushruta",
    "madhava nidana": "madhava",
    "ashtanga hridayam": "ashtanga",
    "bhavaprakasha": "bhavaprakasha",
    "sharangadhara samhita": "sharangadhara",
}


def prognosis_answer(name: str, value: str) -> str:
    return PROGNOSIS_SENTENCES[value.strip().casefold()].format(name=name)


def source_answer(name: str, value: str) -> str:
    return f"{name} is described in the {value.strip()}."


def modern_answer(name: str, value: str) -> str:
    return f"{name} corresponds to {value.strip()} in modern medicine."


def symptoms_answer(name: str, value: str) -> str:
    items = [lower_first(item) for item in split_list(value)]
    return f"The main symptoms of {name} are {join_list(items)}."


def treatment_answer(name: str, value: str) -> str:
    items = [item for item in split_list(value) if "prognosis" not in item.casefold()
             and item.casefold() not in {"incurable"}]
    items = [lower_first(item) for item in items]
    if not items:
        return f"No specific treatment principles are recorded for {name}. {TREATMENT_NOTE}"
    return (f"Classical Ayurvedic management of {name} includes {join_list(items)}. "
            f"{TREATMENT_NOTE}")


def reverse_answer(modern: str, names: list[str]) -> str:
    return f"In Ayurveda, {modern} corresponds to {join_list(names)}."


def ak_overview(row: dict[str, str]) -> str:
    name = row["name"]
    dosha = dosha_answer("It", row["dosha"])
    symptoms = join_list([lower_first(item) for item in split_list(row["symptoms"])])
    system = system_answer("It", row["body_system"])
    return (
        f"{name} is a condition described in the {row['source_text']} that corresponds to "
        f"{row['modern_equivalent']} in modern medicine. {system} {dosha} Its main symptoms are "
        f"{symptoms}. {prognosis_answer(name, row['prognosis'])}"
    )


def doshas_answer(name: str, value: str) -> str:
    factors = [DOSHA_PHRASES[f] for f in dosha_factors(value)]
    return f"In Ayurveda, {name} is associated with an imbalance of {join_list(factors)}."


def prakriti_answer(name: str, value: str) -> str:
    factors = [DOSHA_PHRASES[f] for f in dosha_factors(value)]
    kind = "-".join(factors)
    return f"People with a {kind} constitution (prakriti) are considered more prone to {name}."


def ag_symptoms_answer(name: str, value: str) -> str:
    items = [lower_first(item) for item in split_list(value)]
    return f"Common symptoms of {name} include {join_list(items)}."


def herbs_answer(name: str, value: str) -> str:
    return (f"Ayurvedic herbs traditionally used for {name} include "
            f"{join_list(split_list(value))}. {HERB_NOTE}")


def yoga_answer(name: str, value: str) -> str:
    return f"Yoga and physical therapy suggested for {name} include {join_list(split_list(value))}."


def diet_answer(name: str, value: str) -> str:
    items = [lower_first(item) for item in split_list(value, ";")]
    return f"For {name}, the recommended diet and lifestyle is to {join_list(items)}."


def hindi_answer(name: str, hindi: str, marathi: str) -> str:
    answer = f"In Hindi, {name} is called {hindi}."
    if marathi and marathi != hindi:
        answer += f" In Marathi it is called {marathi}."
    return answer


def ag_overview(row: dict[str, str]) -> str:
    name = row["name"]
    doshas = join_list([DOSHA_PHRASES[f] for f in dosha_factors(row["doshas"])])
    prakriti = "-".join(DOSHA_PHRASES[f] for f in dosha_factors(row["prakriti"]))
    symptoms = join_list([lower_first(item) for item in split_list(row["symptoms"])])
    text = (
        f"In Ayurveda, {name} is associated with an imbalance of {doshas}, and people with a "
        f"{prakriti} constitution are considered more prone to it. Common symptoms include "
        f"{symptoms}."
    )
    if row.get("herbs"):
        text += f" Herbs traditionally used include {join_list(split_list(row['herbs']))}."
    if row.get("yoga"):
        text += f" Helpful yoga includes {join_list(split_list(row['yoga']))}."
    return text
