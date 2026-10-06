"""Validate a small source-reviewed curriculum; never mark it training-ready."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from acharya.config import canonical_json, sha256_bytes, sha256_file
from acharya.rag.prompt import grounded_chat_messages
from training.quality_data import audit_row, words

SYSTEM = (
    'You are an educational Ayurveda reference assistant. Use only the supplied context. '
    'Distinguish traditional descriptions from clinical evidence. Cite factual claims with [1]. '
    'Do not infer mechanisms, diagnose, prescribe or choose a personal dose. '
    'When information is absent, say what the passage cannot establish.'
)


def render_row(source: dict, example: dict) -> dict:
    user = (f"HISTORY:\n(none)\n\nCONTEXTS:\n[1] source={source['url']} page=1\n"
            f"{source['context']}\n\nQUESTION:\n{example['question']}\n\nANSWER:\n")
    return {'id': example['id'], 'question': example['question'],
            'task_type': 'source_reviewed_seed',
            'messages': [{'role': 'system', 'content': SYSTEM},
                         {'role': 'user', 'content': user},
                         {'role': 'assistant', 'content': example['answer']}],
            'provenance': {'url': source['url'], 'source_id': source['id'],
                           'sections': source['sections'], 'checked_on': source['checked_on']}}


def validate(path: Path) -> dict:
    sources = json.loads(path.read_text())
    seen_sources, seen_ids, seen_questions = set(), set(), set()
    context_splits = {}
    counts, kinds = Counter(), Counter()
    for source in sources:
        if source['split'] not in {'train', 'validation', 'test'}:
            raise ValueError('invalid split')
        if source['url'] in seen_sources:
            raise ValueError('source page repeated; source allocation must be unique')
        seen_sources.add(source['url'])
        digest = sha256_bytes(source['context'].encode())
        if digest != source['context_sha256']:
            raise ValueError('context hash mismatch')
        normalized = ' '.join(words(source['context']))
        if normalized in context_splits and context_splits[normalized] != source['split']:
            raise ValueError('context leakage across splits')
        context_splits[normalized] = source['split']
        if not source['sections'] or not source['review_notes']:
            raise ValueError('source evidence missing')
        for example in source['examples']:
            question = tuple(words(example['question']))
            if example['id'] in seen_ids or question in seen_questions:
                raise ValueError('duplicate example or question')
            seen_ids.add(example['id'])
            seen_questions.add(question)
            review = example['review']
            if (not review['reviewer'] or not review['source_support_checked']
                    or not review['citation_checked'] or review['clinical_expert_review']):
                raise ValueError('invalid review scope')
            row = render_row(source, example)
            reasons = audit_row(row)
            if reasons:
                raise ValueError(f"{example['id']}: {reasons}")
            expected = row['messages'][:2]
            if grounded_chat_messages(SYSTEM + '\n\n' + expected[1]['content']) != expected:
                raise ValueError('prompt round-trip failure')
            counts[source['split']] += 1
            kinds[example['kind']] += 1
    if set(counts) != {'train', 'validation', 'test'}:
        raise ValueError('all splits required')
    return {'schema_version': 1, 'source_file_sha256': sha256_file(path),
            'source_pages': len(sources), 'counts': dict(counts), 'kinds': dict(kinds),
            'structural_checks_passed': True, 'training_ready': False,
            'review_type': 'AI source-grounding review; not independent or clinical review',
            'limitations': ['Tiny seed curriculum, not a qualified fine-tuning corpus.',
                           'Source pages separated; topics and authorship are not independent.',
                           'No claim of unseen sources in base-model pretraining.',
                           'No expert clinical validation or demonstrated model improvement.']}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = validate(args.source)
    if args.output:
        args.output.write_bytes(canonical_json(report) + b'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
