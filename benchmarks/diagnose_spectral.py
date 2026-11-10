"""Descriptive paired diagnostics for the frozen CUDA spectral experiment.

No fitted thresholds or new training claims: sampled validation attainment and
leave-one-seed-out test loss differences describe this archive only.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


def paired_summary(rows):
    """Summarize candidate-minus-reference loss; negative favors candidate."""
    if len(rows) < 2 or len({r['seed'] for r in rows}) != len(rows):
        raise ValueError('need at least two distinct paired seeds')
    deltas = [r['test_ce_delta'] for r in rows]
    if not all(math.isfinite(x) for x in deltas):
        raise ValueError('non-finite paired loss')
    return {
        'pairs': rows,
        'mean_delta': statistics.mean(deltas),
        'median_delta': statistics.median(deltas),
        'improved_seeds': sum(x < 0 for x in deltas),
        'leave_one_out': [
            {'excluded_seed': row['seed'],
             'mean_delta': statistics.mean(deltas[:i] + deltas[i + 1:])}
            for i, row in enumerate(rows)
        ],
    }


def sustained_attainment(evaluations, threshold=0.9):
    """Earliest sampled step meeting threshold at every remaining sample.

None means the final observation misses the threshold. This is not an exact
crossing time or evidence of behavior between validation samples.
"""
    if not evaluations:
        raise ValueError('missing validation observations')
    steps = [e['step'] for e in evaluations]
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError('validation steps must increase')
    result = None
    for e in reversed(evaluations):
        accuracy = e['validation']['accuracy']
        if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError('invalid validation accuracy')
        if accuracy < threshold:
            break
        result = e['step']
    return result


def diagnose(archive):
    manifest_bytes = (archive / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    plan = json.loads((archive / 'plan.json').read_text())
    result = {'archive_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
              'interpretation': 'Descriptive post-hoc analysis; no causal mechanism or general advantage established.',
              'precision': {}}
    for precision, seeds in plan['precisions'].items():
        records = {}
        for mode in plan['modes']:
            for seed in seeds:
                name = f'raw/confirm-{precision}-{mode}-{seed}.json.gz'
                data = gzip.decompress((archive / name).read_bytes())
                if hashlib.sha256(data).hexdigest() != manifest['decompressed_raw_sha256'][name]:
                    raise ValueError(f'checksum mismatch: {name}')
                r = json.loads(data)
                if r['status'] != 'complete':
                    raise ValueError(f'incomplete run: {name}')
                records[mode, seed] = r
        contrasts = {}
        for candidate, reference in [('tiger-full', 'tiger-no-spectral'),
                                     ('tiger-spectral-fade', 'tiger-full')]:
            rows = []
            for seed in seeds:
                a, b = records[candidate, seed], records[reference, seed]
                for key in ('initial_state_sha256', 'data_sha256', 'model', 'source_sha256'):
                    if a[key] != b[key]:
                        raise ValueError(f'unmatched {key}: {precision}/{seed}')
                if [e['step'] for e in a['evaluations']] != [e['step'] for e in b['evaluations']]:
                    raise ValueError('unmatched validation schedule')
                rows.append({'seed': seed, 'test_ce_delta': a['test']['ce'] - b['test']['ce'],
                             'candidate_sustained_validation_90_step': sustained_attainment(a['evaluations']),
                             'reference_sustained_validation_90_step': sustained_attainment(b['evaluations'])})
            contrasts[candidate + '_minus_' + reference] = paired_summary(rows)
        result['precision'][precision] = contrasts
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path(__file__).parent / 'evidence/2026-10-02-cuda-spectral')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = json.dumps(diagnose(args.archive), indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end='')
