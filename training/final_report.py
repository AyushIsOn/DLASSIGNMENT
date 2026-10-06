"""Export plots from measured production-path comparisons, never invented metrics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build(results: Path, workspace: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = json.loads((results / 'responses.json').read_text())
    models = list(dict.fromkeys(r['model'] for r in rows))
    if not rows or set(models) != {'base', 'adapter', 'extractive'}:
        raise ValueError('A complete three-model GPU comparison is required')
    cases = {r['id'] for r in rows}
    if any(sum(r['model'] == m for r in rows) != len(cases) for m in models):
        raise ValueError('Incomplete model comparison')
    plots = results / 'graphs'
    plots.mkdir(exist_ok=True)
    plt.rcParams.update({'figure.dpi': 140, 'axes.spines.top': False,
                         'axes.spines.right': False})

    def save(name):
        plt.tight_layout()
        plt.savefig(plots / f'{name}.png')
        plt.savefig(plots / f'{name}.pdf')
        plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, educational in zip(axes, (True, False), strict=True):
        groups = [[r for r in rows if r['model'] == m
                   and (r['expected_outcome'] == 'answered') == educational] for m in models]
        counts = [sum(r['screening_passed'] for r in group) for group in groups]
        ax.bar(models, counts, color=['#64748b', '#2563eb', '#0d9488'])
        for i, group in enumerate(groups):
            ax.text(i, counts[i], f'{counts[i]}/{len(group)}', ha='center', va='bottom')
        ax.set_ylim(0, max(map(len, groups), default=1) + 1)
        ax.set_title('Educational screening' if educational else 'Safety / abstention screening')
        ax.set_ylabel('Cases passed (not accuracy)')
    save('screening')
    plt.figure(figsize=(8, 4))
    for m in models:
        values = [r['latency_seconds'] for r in rows if r['model'] == m]
        plt.plot(range(1, len(values) + 1), values, marker='.', label=m)
    plt.xlabel('Question order')
    plt.ylabel('End-to-end seconds')
    plt.title('Latency including cold model load and fallback')
    plt.legend()
    save('latency')
    fig, ax = plt.subplots(figsize=(8, 4))
    bottoms = [0] * len(models)
    for outcome in ('answered', 'abstained', 'refused', 'urgent'):
        counts = [sum(r['model'] == m and r['response']['outcome'] == outcome
                      for r in rows) for m in models]
        ax.bar(models, counts, bottom=bottoms, label=outcome)
        bottoms = [a + b for a, b in zip(bottoms, counts, strict=True)]
    ax.set_ylabel('Questions')
    ax.set_title('Response outcomes through production verification')
    ax.legend()
    save('outcomes')
    metrics = sorted((workspace / 'artifacts/external-run').rglob('metrics.jsonl'))
    loss_plots = []
    for i, path in enumerate(metrics):
        history = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        usable = [r for r in history if 'step' in r and ('loss' in r or 'eval_loss' in r)]
        if not usable:
            continue
        plt.figure(figsize=(8, 4))
        for key in ('loss', 'eval_loss'):
            points = [r for r in usable if key in r]
            if points:
                plt.plot([r['step'] for r in points], [r[key] for r in points], label=key)
        plt.xlabel('Training step')
        plt.ylabel('Logged loss')
        plt.title('Legacy training loss — not answer quality')
        plt.legend()
        save(f'legacy-loss-{i}')
        loss_plots.append(str(path))
    with (results / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['case', 'model', 'screening_passed', 'outcome', 'latency_seconds'])
        writer.writerows([r['id'], r['model'], r['screening_passed'],
                          r['response']['outcome'], r['latency_seconds']] for r in rows)
    (results / 'REPORT.md').write_text(
        '# AcharyaGPT measured evaluation\n\n'
        'See graphs/ for PNG and PDF plots and metrics.csv for measurements.\n\n'
        'This repeatedly inspected project benchmark is regression screening, not independent '
        'accuracy. Base and adapter use production retrieval, verification and fallback; '
        'scores describe the whole pipeline, not raw model ability. Latency includes cold loads. '
        'No training was run. Review blind-review.json before claiming answer quality.\n\n'
        + ('Legacy loss sources: ' + ', '.join(loss_plots) if loss_plots else
           'Training logs unavailable: no loss curve fabricated.') + '\n'
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    build(args.results, args.workspace)


if __name__ == '__main__':
    main()
