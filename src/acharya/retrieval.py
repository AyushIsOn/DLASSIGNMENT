"""Small dependency-free BM25 retriever over the knowledge-base cards.

The same retriever builds the open-book training contexts and serves the app, so
the model sees the same kind of retrieved entries in training and production.
"""

from __future__ import annotations

import heapq
import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+")
# Function words plus question scaffolding ("which dosha is involved in ...") that would
# otherwise match FAQ-style cards instead of the condition the user asks about.
STOPWORDS = frozenset(
    (
        "a about above after again all also am an and any are as at be because been "
        "before being below between both but by can could did do does doing down during "
        "each few for from further had has have having he her here hers him his how i if "
        "in into is it its itself just me more most my no nor not of off on once only or "
        "other our out over own same she should so some such than that the their them "
        "then there these they this those through to too under until up very was we were "
        "what when where which while who whom why will with would you your yours ayurveda "
        "ayurvedic tell explain describe described give name list called call known mean "
        "means please according main mainly involved involve predominant predominance "
        "affect affected doctor hi hey hello thanks thank ok okay"
    ).split()
)
TITLE_WEIGHT = 2.0  # a match on the card's title (the condition name) counts double


def fold(text: str) -> str:
    """Lowercase and strip diacritics, so 'vātavyādhiḥ' and 'vatavyadhih' are one token."""
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch)).casefold()


def tokenize(text: str) -> list[str]:
    tokens = []
    for token in _TOKEN.findall(fold(text)):
        if token in STOPWORDS:
            continue
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        elif len(token) > 3 and token[-1] == "h" and token[-2] in "aiu" \
                and token[-3] not in "aeiou":
            token = token[:-1]  # Sanskrit visarga, usually not typed: "jvarah" == "jvara"
        tokens.append(token)
    return tokens


@dataclass(frozen=True)
class Card:
    id: str
    title: str
    source: str
    text: str


@dataclass(frozen=True)
class Hit:
    card: Card
    score: float


class _Field:
    """BM25 statistics for one text field."""

    def __init__(self, texts: Sequence[str], k1: float, b: float) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(text)) for text in texts]
        self.lengths = [sum(doc.values()) for doc in self.docs]
        self.average = max(1e-9, sum(self.lengths) / len(self.lengths))
        frequency: Counter[str] = Counter()
        for doc in self.docs:
            frequency.update(doc.keys())
        total = len(self.docs)
        self.idf = {term: math.log(1 + (total - count + 0.5) / (count + 0.5))
                    for term, count in frequency.items()}
        self.postings: dict[str, list[int]] = {}
        for index, doc in enumerate(self.docs):
            for term in doc:
                self.postings.setdefault(term, []).append(index)

    def add_scores(self, terms: set[str], result: dict[int, float], weight: float) -> None:
        for term in terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index in self.postings[term]:
                tf = self.docs[index][term]
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average)
                result[index] = result.get(index, 0.0) + weight * idf * tf * (self.k1 + 1) / norm


class BM25:
    """BM25 over card text plus a separately normalised title field (BM25F-style), so the
    condition named in a question outweighs generic words like "symptoms" or "dosha"."""

    def __init__(self, cards: Sequence[Card], k1: float = 1.5, b: float = 0.75) -> None:
        if not cards:
            raise ValueError("the knowledge base is empty")
        self.cards = tuple(cards)
        self._text = _Field([card.text for card in self.cards], k1, b)
        self._title = _Field([card.title for card in self.cards], k1, 0.9)
        self._by_id = {card.id: card for card in self.cards}

    def card(self, card_id: str) -> Card:
        return self._by_id[card_id]

    def scores(self, query: str) -> dict[int, float]:
        terms = set(tokenize(query))
        result: dict[int, float] = {}
        self._text.add_scores(terms, result, 1.0)
        self._title.add_scores(terms, result, TITLE_WEIGHT)
        return result

    def search(
        self, query: str, k: int = 3, exclude: Iterable[str] = (), min_score: float = 0.0
    ) -> list[Hit]:
        banned = set(exclude)
        # heap instead of a full sort: same order (score desc, then index), O(n + k log n)
        heap = [(-score, index) for index, score in self.scores(query).items()
                if score >= min_score]
        heapq.heapify(heap)
        hits = []
        while heap and len(hits) < k:
            negative, index = heapq.heappop(heap)
            card = self.cards[index]
            if card.id not in banned:
                hits.append(Hit(card, -negative))
        return hits

    @classmethod
    def from_jsonl(cls, path: Path) -> BM25:
        cards = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                cards.append(Card(value["id"], value["title"], value["source"], value["text"]))
        return cls(cards)
