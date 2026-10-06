from __future__ import annotations

import json

from training.direct_compare import compare


def test_raw_comparison_excludes_reference_and_preserves_distinct_outputs(tmp_path, monkeypatch):
    row = {'id': 'case', 'messages': [
        {'role': 'user', 'content': 'CONTEXTS:\n[1] source=test\nEvidence.\n\nQUESTION:\nExplain.'},
        {'role': 'assistant', 'content': 'SECRET_REFERENCE'},
    ]}
    monkeypatch.setattr('training.direct_compare.select_rows', lambda *args: [row])
    seen = []

    def make(workspace, adapter):
        def generate(prompt, tokens):
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
