"""Validate a small source-reviewed curriculum; never mark it training-ready."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import date
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


def _required_text(record: dict, field: str, scope: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{scope}: nonempty {field} required')
    return value


def _checked_date(record: dict, scope: str) -> None:
    value = _required_text(record, 'checked_on', scope)
    try:
        if date.fromisoformat(value).isoformat() != value:
            raise ValueError
    except ValueError as exc:
        raise ValueError(f'{scope}: checked_on must be an ISO date') from exc


def _validate_source_provenance(source: dict) -> None:
    for field in ('id', 'title', 'url', 'context', 'context_kind', 'context_sha256',
                  'review_notes', 'reuse_note'):
        _required_text(source, field, 'source provenance')
    _checked_date(source, 'source provenance')
    sections = source.get('sections')
    if (not isinstance(sections, list) or not sections
            or any(not isinstance(section, str) or not section.strip() for section in sections)):
        raise ValueError('source evidence missing: nonempty sections required')
    if 'publisher_snapshot_sha256' in source:
        digest = source['publisher_snapshot_sha256']
        if not isinstance(digest, str) or re.fullmatch(r'[0-9a-fA-F]{64}', digest) is None:
            raise ValueError('invalid publisher snapshot SHA256')
    if ('snapshot_kind' in source
            and source['snapshot_kind'] not in ('publisher_html', 'indexed_publisher_text')):
        raise ValueError('invalid snapshot_kind')
    if ('snapshot_redistributed' in source
            and not isinstance(source['snapshot_redistributed'], bool)):
        raise ValueError('snapshot_redistributed must be boolean')


def _validate_review(example: dict) -> None:
    review = example.get('review')
    if not isinstance(review, dict):
        raise ValueError('invalid review scope: review record required')
    _required_text(review, 'reviewer', 'review')
    if (review.get('source_support_checked') is not True
            or review.get('citation_checked') is not True
            or review.get('clinical_expert_review') is not False):
        raise ValueError('invalid review scope')
    if 'independent_review' not in example:
        return
    independent = example['independent_review']
    if not isinstance(independent, dict):
        raise ValueError('invalid independent_review record')
    for field in ('reviewer', 'rationale'):
        _required_text(independent, field, 'independent_review')
    _checked_date(independent, 'independent_review')
    if (independent.get('reviewer_type') != 'ai'
            or independent.get('decision') not in ('accepted', 'revise')
            or independent.get('clinical_expert_review', False) is not False):
        raise ValueError('invalid independent_review scope: AI review is not clinical approval')


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
    if not isinstance(sources, list) or not sources:
        raise ValueError('nonempty source list required')
    seen_sources, seen_source_ids, seen_ids, seen_questions = set(), set(), set(), set()
    context_splits = {}
    counts, kinds = Counter(), Counter()
    reviewers, independent_reviewers, independent_decisions = Counter(), Counter(), Counter()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError('source record required')
        _validate_source_provenance(source)
        if source['split'] not in {'train', 'validation', 'test'}:
            raise ValueError('invalid split')
        if source['url'] in seen_sources:
            raise ValueError('source page repeated; source allocation must be unique')
        if source['id'] in seen_source_ids:
            raise ValueError('source id repeated; source identifiers must be unique')
        seen_sources.add(source['url'])
        seen_source_ids.add(source['id'])
        digest = sha256_bytes(source['context'].encode())
        if digest != source['context_sha256']:
            raise ValueError('context hash mismatch')
        normalized = ' '.join(words(source['context']))
        if normalized in context_splits and context_splits[normalized] != source['split']:
            raise ValueError('context leakage across splits')
        context_splits[normalized] = source['split']
        for example in source['examples']:
            question = tuple(words(example['question']))
            if example['id'] in seen_ids or question in seen_questions:
                raise ValueError('duplicate example or question')
            seen_ids.add(example['id'])
            seen_questions.add(question)
            _validate_review(example)
            row = render_row(source, example)
            reasons = audit_row(row)
            if reasons:
                raise ValueError(f"{example['id']}: {reasons}")
            expected = row['messages'][:2]
            if grounded_chat_messages(SYSTEM + '\n\n' + expected[1]['content']) != expected:
                raise ValueError('prompt round-trip failure')
            counts[source['split']] += 1
            kinds[example['kind']] += 1
            reviewers[example['review']['reviewer']] += 1
            if 'independent_review' in example:
                review = example['independent_review']
                independent_reviewers[review['reviewer']] += 1
                independent_decisions[review['decision']] += 1
    if set(counts) != {'train', 'validation', 'test'}:
        raise ValueError('all splits required')
    total = sum(counts.values())
    independently_reviewed = sum(independent_decisions.values())
    return {'schema_version': 1, 'source_file_sha256': sha256_file(path),
            'source_pages': len(sources), 'counts': dict(counts), 'kinds': dict(kinds),
            'structural_checks_passed': True, 'training_ready': False,
            'review_type': 'AI source-grounding review; optional additional AI review, '
                           'not clinical validation',
            'review_summary': {
                'examples_total': total,
                'primary_reviewed_examples': sum(reviewers.values()),
                'primary_reviewers': dict(reviewers),
                'independent_reviewed_examples': independently_reviewed,
                'independent_review_missing_examples': total - independently_reviewed,
                'independent_review_coverage': independently_reviewed / total,
                'independent_reviewers': dict(independent_reviewers),
                'independent_review_decisions': {
                    decision: independent_decisions[decision] for decision in ('accepted', 'revise')
                },
                'independent_reviewer_type': 'ai',
                'clinical_expert_reviewed_examples': 0,
            },
            'limitations': ['Tiny seed curriculum, not a qualified fine-tuning corpus.',
                           'Source pages separated; topics and authorship are not independent.',
                           'No claim of unseen sources in base-model pretraining.',
                           'Review metadata records AI checks; it does not verify reviewer '
                           'independence or confer clinical approval.',
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
