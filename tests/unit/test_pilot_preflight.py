from __future__ import annotations

import json
from pathlib import Path

from training.pilot_preflight import _similarity, inspect

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/curated/reviewed_seed_v1/sources-and-examples.json'
PLAN = ROOT / 'configs/pilot_plan.json'
RUBRIC = ROOT / 'eval/pilot_rubric.json'


def test_small_seed_does_not_qualify_for_pilot():
    report = inspect(SOURCE, PLAN, RUBRIC)
    assert report['counts'] == {'train': 16, 'validation': 4, 'test': 4}
    assert len(report['source_families']['train']) == 3  # Related Delhi pages grouped.
    assert 'examples:train:16/100' in report['blockers']
    assert not report['ready_to_train']
    assert not report['gpu_started']
    assert len(report['hashes']['rubric']) == 64


def test_family_cross_split_detected_even_with_different_urls(tmp_path):
    plan = json.loads(PLAN.read_text())
    plan['source_family_assignments']['nccih-boswellia'] = 'nccih-turmeric'
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    report = inspect(SOURCE, path, RUBRIC)
    assert 'source_family_crosses_splits:nccih-turmeric' in report['blockers']


def test_near_duplicate_similarity_survives_small_addition():
    original = 'A description of traditional concepts and their meaning in this source.'
    assert _similarity(original, original + ' Extra.') >= 0.8
    assert _similarity(original, 'An unrelated biological experiment.') == 0.0
