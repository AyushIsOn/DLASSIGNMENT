from __future__ import annotations

import json

import pytest

from acharya.rag.prompt import grounded_chat_messages
from training.direct_compare import compare, comparison_prompt


def test_raw_comparison_excludes_reference_and_preserves_distinct_outputs(tmp_path, monkeypatch):
    row = {'id': 'case', 'messages': [
        {'role': 'system', 'content': 'Use only supplied evidence.'},
        {'role': 'user', 'content': ('HISTORY:\n(none)\n\nCONTEXTS:\n'
                                     '[1] source=test\nEvidence.\n\nQUESTION:\nExplain.')},
        {'role': 'assistant', 'content': 'SECRET_REFERENCE'},
    ]}
    monkeypatch.setattr('training.direct_compare.select_rows', lambda *args: [row])
    seen = []

    def make(workspace, adapter):
        def generate(prompt, tokens):
            messages = grounded_chat_messages(prompt)
            assert messages == row['messages'][:2]
            seen.append(prompt)
            return 'Raw adapter answer' if adapter else 'Raw base answer'
        return generate

    monkeypatch.setattr('training.direct_compare.make_generator', make)
    adapter = tmp_path / 'adapter'
    adapter.mkdir()
    (adapter / 'adapter_model.safetensors').write_bytes(b'fixture')
    output = tmp_path / 'comparison'
    compare(tmp_path, adapter, output, 1)
    assert len(seen) == 2 and seen[0] == seen[1]
    assert all('SECRET_REFERENCE' not in prompt for prompt in seen)
    rows = json.loads((output / 'responses.json').read_text())
    assert rows[0]['answer'] != rows[1]['answer']
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['fallback_used'] is False
    assert manifest['answer_quality_established'] is False


def test_bad_prompt_rejected_before_model_creation(tmp_path, monkeypatch):
    monkeypatch.setattr('training.direct_compare.select_rows',
                        lambda *args: [{'messages': [{'role': 'user', 'content': 'bad'}]}])
    def forbidden(*args):
        pytest.fail('Model creation must not happen for invalid prompts')
    monkeypatch.setattr('training.direct_compare.make_generator', forbidden)
    with pytest.raises(ValueError, match='SFT row'):
        compare(tmp_path, tmp_path, tmp_path / 'output', 1)
    assert not (tmp_path / 'output').exists()


def test_missing_history_is_rejected():
    with pytest.raises(ValueError, match='history'):
        comparison_prompt({'messages': [
            {'role': 'system', 'content': 'Instructions'},
            {'role': 'user', 'content': 'CONTEXTS: bad'},
            {'role': 'assistant', 'content': 'reference'},
        ]})
