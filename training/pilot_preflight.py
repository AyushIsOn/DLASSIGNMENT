"""CPU-only readiness inventory. Reports blockers; never authorizes training."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from acharya.config import sha256_file
from training.quality_data import words
from training.reviewed_seed import validate


def _similarity(left: str, right: str) -> float:
    def shingles(text: str) -> set[tuple[str, ...]]:
        tokens = words(text)
        return {tuple(tokens[i:i + 3]) for i in range(max(1, len(tokens) - 2))}
    a, b = shingles(left), shingles(right)
    return len(a & b) / len(a | b) if a | b else 1.0


def inspect(source: Path, plan: Path, rubric: Path) -> dict:
    summary = validate(source)
    config = json.loads(plan.read_text())
    sources = json.loads(source.read_text())
    blockers, warnings = [], []
    families = defaultdict(set)
    family_splits = defaultdict(set)
    assignments = config['source_family_assignments']
    counts = Counter()
    examples = []
    for item in sources:
        split = item['split']
        family = assignments.get(item['id'])
        if not family:
            blockers.append(f"unassigned_source_family:{item['id']}")
        else:
            families[split].add(family)
            family_splits[family].add(split)
        if len(item['examples']) > config['maximum_examples_per_context']:
            blockers.append(f"too_many_variants_for_context:{item['id']}")
        counts[split] += len(item['examples'])
        examples.extend((split, example['id'], example['question']) for example in item['examples'])
    for family, splits in family_splits.items():
        if len(splits) > 1:
            blockers.append(f'source_family_crosses_splits:{family}')
    for split in ('train', 'validation', 'test'):
        required = config['minimum_examples'][split]
        if counts[split] < required:
            blockers.append(f'examples:{split}:{counts[split]}/{required}')
        required = config['minimum_source_families'][split]
        if len(families[split]) < required:
            blockers.append(f'source_families:{split}:{len(families[split])}/{required}')
    if len(summary['kinds']) < config['minimum_task_kinds']:
        blockers.append('insufficient_task_diversity')
    threshold = config['cross_split_near_duplicate_jaccard']
    for left, right in combinations(sources, 2):
        if left['split'] != right['split']:
            score = _similarity(left['context'], right['context'])
            if score >= threshold:
                blockers.append(f"similar_contexts:{left['id']}:{right['id']}:{score:.3f}")
    for left, right in combinations(examples, 2):
        if left[0] != right[0] and _similarity(left[2], right[2]) >= threshold:
            blockers.append(f'similar_questions:{left[1]}:{right[1]}')
    warnings.extend(['Lexical similarity does not exclude semantic leakage.',
                     'AI source reviews are not independent clinical review.',
                     'Pilot trainer and GPU recovery checks remain separate gates.'])
    return {'schema_version': 1, 'counts': dict(counts),
            'source_families': {key: sorted(value) for key, value in families.items()},
            'task_kinds': summary['kinds'], 'blockers': sorted(set(blockers)),
            'warnings': warnings, 'dataset_contract_passed': not blockers,
            'ready_to_train': False, 'gpu_started': False,
            'hashes': {name: sha256_file(path) for name, path in
                       [('sources', source), ('plan', plan), ('rubric', rubric)]}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--rubric', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = inspect(args.source, args.plan, args.rubric)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
