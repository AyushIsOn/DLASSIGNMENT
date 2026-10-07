from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from training.reviewed_seed import validate

SEED = (Path(__file__).resolve().parents[2]
        / 'data/curated/reviewed_seed_v1/sources-and-examples.json')


def test_reviewed_seed_keeps_source_pages_separate_and_training_disabled():
    report = validate(SEED)
    sources = json.loads(SEED.read_text())
    counts = Counter()
    for source in sources:
        counts[source['split']] += len(source['examples'])
    assert report['source_pages'] == 15
    assert report['counts'] == dict(counts)
    assert all(counts[split] >= floor for split, floor
               in {'train': 40, 'validation': 8, 'test': 12}.items())
    assert report['training_ready'] is False
    assert report['review_summary']['clinical_expert_reviewed_examples'] == 0


@pytest.fixture
def sources():
    """Keep mutation tests independent of optional reviews added to the seed."""
    data = json.loads(SEED.read_text())
    for source in data:
        for example in source['examples']:
            example.pop('independent_review', None)
    return data


def write_seed(tmp_path, sources):
    path = tmp_path / 'seed.json'
    path.write_text(json.dumps(sources))
    return path


@pytest.mark.parametrize('mutation,match', [
    ('citation', 'missing_or_malformed'), ('source', 'source page repeated'),
    ('context', 'hash mismatch'), ('question', 'duplicate example'),
])
def test_seed_rejects_broken_evidence_and_leakage(tmp_path, mutation, match):
    sources = json.loads(SEED.read_text())
    if mutation == 'citation':
        sources[0]['examples'][0]['answer'] = 'This uncited statement must fail review.'
    elif mutation == 'source':
        sources[-1]['url'] = sources[0]['url']
    elif mutation == 'context':
        sources[0]['context'] += ' Invented material.'
    else:
        sources[-1]['examples'][0]['question'] = sources[0]['examples'][0]['question']
    path = tmp_path / 'seed.json'
    path.write_text(json.dumps(sources))
    with pytest.raises(ValueError, match=match):
        validate(path)


def test_seed_rejects_duplicate_source_id(tmp_path, sources):
    sources[-1]['id'] = sources[0]['id']
    with pytest.raises(ValueError, match='source id repeated'):
        validate(write_seed(tmp_path, sources))


@pytest.mark.parametrize('field', [
    'id', 'title', 'url', 'checked_on', 'context', 'context_kind', 'context_sha256',
    'review_notes', 'reuse_note',
])
@pytest.mark.parametrize('value', [None, '', '  '])
def test_seed_requires_nonempty_provenance(tmp_path, sources, field, value):
    sources[0][field] = value
    with pytest.raises(ValueError, match=f'nonempty {field} required'):
        validate(write_seed(tmp_path, sources))


@pytest.mark.parametrize('sections', [None, [], [''], ['  '], 'Dosha'])
def test_seed_requires_nonempty_section_list(tmp_path, sources, sections):
    sources[0]['sections'] = sections
    with pytest.raises(ValueError, match='source evidence missing'):
        validate(write_seed(tmp_path, sources))


@pytest.mark.parametrize('digest', ['', None, 'abc', 'z' * 64, 'a' * 63])
def test_seed_rejects_invalid_optional_snapshot_hash(tmp_path, sources, digest):
    sources[0]['publisher_snapshot_sha256'] = digest
    with pytest.raises(ValueError, match='invalid publisher snapshot SHA256'):
        validate(write_seed(tmp_path, sources))


@pytest.mark.parametrize('kind', ['publisher_html', 'indexed_publisher_text'])
def test_seed_accepts_snapshot_kind_without_requiring_snapshot(tmp_path, sources, kind):
    for source in sources:
        source.pop('publisher_snapshot_sha256', None)
    sources[0]['snapshot_kind'] = kind
    assert validate(write_seed(tmp_path, sources))['training_ready'] is False


def test_seed_rejects_unknown_snapshot_kind(tmp_path, sources):
    sources[0]['snapshot_kind'] = 'unverified_copy'
    with pytest.raises(ValueError, match='invalid snapshot_kind'):
        validate(write_seed(tmp_path, sources))


@pytest.mark.parametrize('field,value', [
    ('source_support_checked', 'true'), ('source_support_checked', 1),
    ('citation_checked', 'true'), ('citation_checked', 1),
    ('clinical_expert_review', 'false'), ('clinical_expert_review', 0),
    ('clinical_expert_review', True),
])
def test_seed_requires_literal_review_flags(tmp_path, sources, field, value):
    sources[0]['examples'][0]['review'][field] = value
    with pytest.raises(ValueError, match='invalid review scope'):
        validate(write_seed(tmp_path, sources))


def test_seed_rejects_blank_primary_reviewer(tmp_path, sources):
    sources[0]['examples'][0]['review']['reviewer'] = '  '
    with pytest.raises(ValueError, match='nonempty reviewer required'):
        validate(write_seed(tmp_path, sources))


def independent_review(decision='accepted'):
    return {'reviewer': 'Second AI source reviewer', 'reviewer_type': 'ai',
            'decision': decision, 'rationale': 'Checked the answer against the supplied passage.',
            'checked_on': '2026-10-06'}


def test_seed_reports_review_coverage_decisions_and_missing_reviews(tmp_path, sources):
    path = write_seed(tmp_path, sources)
    report = validate(path)
    total = sum(report['counts'].values())
    summary = report['review_summary']
    assert summary['primary_reviewed_examples'] == total
    assert sum(summary['primary_reviewers'].values()) == total
    assert summary['independent_reviewed_examples'] == 0
    assert summary['independent_review_missing_examples'] == total
    assert summary['independent_review_coverage'] == 0
    assert summary['independent_review_decisions'] == {'accepted': 0, 'revise': 0}

    sources[0]['examples'][0]['independent_review'] = independent_review()
    sources[-1]['examples'][0]['independent_review'] = independent_review('revise')
    report = validate(write_seed(tmp_path, sources))
    summary = report['review_summary']
    assert summary['independent_reviewed_examples'] == 2
    assert summary['independent_review_missing_examples'] == total - 2
    assert summary['independent_review_coverage'] == pytest.approx(2 / total)
    assert summary['independent_reviewers'] == {'Second AI source reviewer': 2}
    assert summary['independent_review_decisions'] == {'accepted': 1, 'revise': 1}
    assert summary['independent_reviewer_type'] == 'ai'
    assert summary['clinical_expert_reviewed_examples'] == 0
    assert report['training_ready'] is False


@pytest.mark.parametrize('field,value', [
    ('reviewer', '  '), ('reviewer_type', 'clinician'), ('decision', 'approved'),
    ('rationale', ''), ('checked_on', '2026-02-30'), ('checked_on', '20261006'),
    ('clinical_expert_review', True), ('clinical_expert_review', 'false'),
])
def test_seed_rejects_invalid_independent_review(tmp_path, sources, field, value):
    review = independent_review()
    review[field] = value
    sources[0]['examples'][0]['independent_review'] = review
    with pytest.raises(ValueError, match='independent_review'):
        validate(write_seed(tmp_path, sources))


def test_seed_rejects_missing_independent_reviewer_identity(tmp_path, sources):
    review = independent_review()
    del review['reviewer']
    sources[0]['examples'][0]['independent_review'] = review
    with pytest.raises(ValueError, match='nonempty reviewer required'):
        validate(write_seed(tmp_path, sources))
