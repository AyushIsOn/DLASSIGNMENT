from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.reviewed_seed import validate

SEED = (Path(__file__).resolve().parents[2]
        / 'data/curated/reviewed_seed_v1/sources-and-examples.json')


def test_reviewed_seed_keeps_source_pages_separate_and_training_disabled():
    report = validate(SEED)
    assert report['source_pages'] == 6
    assert report['counts'] == {'train': 16, 'validation': 4, 'test': 4}
    assert report['training_ready'] is False


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
