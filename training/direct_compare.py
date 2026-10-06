"""Compare raw base/adapter answers on identical evidence without RAG fallback."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from acharya.config import canonical_json, sha256_file
from training.quality_data import context_text, copy_fraction
from training.runtime import make_generator


def select_rows(workspace: Path, limit: int) -> list[dict]:
    pointer = json.loads((workspace / 'artifacts/state/active_preparation.json').read_text())
    data = Path(pointer['processed_path'])
    if not data.is_absolute():
        data = workspace / data
    rows = [json.loads(line) for line in (data / 'test.jsonl').read_text().splitlines()
            if line.strip()]
    # Existing held-out prompts are diagnostic for this old adapter, not fresh validation.
    random.Random(8713).shuffle(rows)
    return rows[:limit]


def compare(workspace: Path, adapter: Path, output: Path, limit: int) -> None:
    if limit < 1:
        raise ValueError('limit must be positive')
    rows = select_rows(workspace, limit)
    if not rows:
        raise ValueError('No test examples')
    output.mkdir(parents=True, exist_ok=False)
    generators = {'base': make_generator(workspace, None),
                  'adapter': make_generator(workspace, adapter)}
    records, reviews, key = [], [], []
    rng = random.Random(5319)
    for number, row in enumerate(rows, 1):
        # Never send the copying reference answer to either generator.
        prompt = next(m['content'] for m in row['messages'] if m['role'] == 'user')
        answers = []
        for model, generate in generators.items():
            started = time.monotonic()
            answer = generate(prompt, 384)
            record = {'id': row['id'], 'model': model, 'prompt': prompt, 'answer': answer,
                      'latency_seconds': time.monotonic() - started,
                      'source_copy_fraction': copy_fraction(answer, context_text(row))}
            records.append(record)
            answers.append((model, answer))
            print(json.dumps({'case': number, 'total': len(rows), 'model': model,
                              'latency_seconds': record['latency_seconds']}), flush=True)
        rng.shuffle(answers)
        for i, (model, answer) in enumerate(answers):
            blind_id = f'{row["id"]}-{i}'
            reviews.append({'blind_id': blind_id, 'prompt': prompt, 'answer': answer,
                            'reviewer': None, 'scores_0_to_2': {
                                'factuality': None, 'relevance': None,
                                'clarity': None, 'safety': None}})
            key.append({'blind_id': blind_id, 'model': model})
        for name, content in [('responses', records), ('blind-review', reviews), ('key', key)]:
            (output / f'{name}.json').write_bytes(canonical_json(content) + b'\n')
    (output / 'manifest.json').write_bytes(canonical_json({
        'case_count': len(rows), 'retrieval_used': False, 'fallback_used': False,
        'answer_quality_established': False, 'human_review_required': True,
        'adapter_sha256': sha256_file(adapter / 'adapter_model.safetensors'),
        'limitations': ['Existing test set; not fresh independent evaluation.',
                        'Copy fraction is descriptive, not correctness.',
                        'Source text may contain OCR errors; review against source.'],
    }) + b'\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=32)
    args = parser.parse_args()
    compare(args.workspace, args.adapter, args.output, args.limit)


if __name__ == '__main__':
    main()
