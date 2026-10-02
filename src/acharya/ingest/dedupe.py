"""Deterministic exact and near-duplicate connected components."""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from acharya.ingest.normalize import Chunk

_TOKEN = re.compile(r"[\w]+", re.UNICODE)


@dataclass(frozen=True)
class DuplicateComponent:
    component_id: str
    winner_id: str
    member_ids: tuple[str, ...]
    kind: str

    def as_dict(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "winner_id": self.winner_id,
            "member_ids": list(self.member_ids),
            "kind": self.kind,
        }


def shingles(text: str, size: int = 5) -> frozenset[str]:
    tokens = [item.casefold() for item in _TOKEN.findall(text)]
    if len(tokens) < size:
        return frozenset(tokens)
    return frozenset(
        " ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)
    )


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def minhash_signature(
    values: frozenset[str], seed: int = 1729, length: int = 32
) -> tuple[int, ...]:
    rng = random.Random(seed)
    salts = [rng.getrandbits(64).to_bytes(8, "big") for _ in range(length)]
    if not values:
        return tuple(0 for _ in salts)
    return tuple(
        min(
            int.from_bytes(hashlib.sha256(salt + value.encode()).digest()[:8], "big")
            for value in values
        )
        for salt in salts
    )


def deduplicate(
    chunks: tuple[Chunk, ...], threshold: float = 0.90, shingle_size: int = 5
) -> tuple[tuple[Chunk, ...], tuple[DuplicateComponent, ...]]:
    ordered = sorted(chunks, key=lambda item: item.chunk_id)
    parents = list(range(len(ordered)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    normalized = [" ".join(_TOKEN.findall(item.text.casefold())) for item in ordered]
    chunk_shingles = [shingles(item.text, shingle_size) for item in ordered]
    if not 0 < threshold <= 1:
        raise ValueError("duplicate threshold must be in (0, 1]")
    # Exact duplicates need no pairwise comparisons. Prefix filtering is lossless
    # for Jaccard: qualifying sets must share a token in these global-order prefixes.
    frequencies = Counter(token for values in chunk_shingles for token in values)
    postings: dict[str, list[int]] = defaultdict(list)
    exact_seen: dict[str, int] = {}
    empty_seen: int | None = None
    for right, values in enumerate(chunk_shingles):
        if normalized[right] in exact_seen:
            union(exact_seen[normalized[right]], right)
            continue
        exact_seen[normalized[right]] = right
        if not values:
            if empty_seen is not None:
                union(empty_seen, right)
            empty_seen = right
            continue
        tokens = sorted(values, key=lambda token: (frequencies[token], token))
        prefix = tokens[:len(tokens) - math.ceil(threshold * len(tokens)) + 1]
        candidates = {left for token in prefix for left in postings[token]}
        for left in sorted(candidates):
            other = chunk_shingles[left]
            if min(len(values), len(other)) < threshold * max(len(values), len(other)):
                continue
            if jaccard(other, values) >= threshold:
                union(left, right)
        for token in prefix:
            postings[token].append(right)

    groups: dict[int, list[int]] = {}
    for index in range(len(ordered)):
        groups.setdefault(root(index), []).append(index)
    winners: list[Chunk] = []
    components: list[DuplicateComponent] = []
    for indexes in groups.values():
        members = tuple(sorted(ordered[index].chunk_id for index in indexes))
        winner_id = members[0]
        winner = ordered[indexes[0]]
        winners.append(winner)
        if len(members) > 1:
            texts = {normalized[index] for index in indexes}
            kind = "exact" if len(texts) == 1 else "near"
            digest = hashlib.sha256("\n".join(members).encode()).hexdigest()[:20]
            components.append(DuplicateComponent(f"dup_{digest}", winner_id, members, kind))
    return tuple(sorted(winners, key=lambda item: item.chunk_id)), tuple(
        sorted(components, key=lambda item: item.component_id)
    )
