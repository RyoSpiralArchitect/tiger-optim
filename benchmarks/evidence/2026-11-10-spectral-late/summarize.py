"""Validate all frozen runs and report predeclared late-learning contrasts."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
SOURCE = '3cb4b78ce5f04837ad14d16305e7f5b505bbbe99'


def summarize(here=HERE):
    plan_bytes = (here / 'plan.json').read_bytes()
    plan = json.loads(plan_bytes)
    index = json.loads((here / 'confirmation_index.json').read_text())
    if not index['complete'] or index['plan_sha256'] != hashlib.sha256(plan_bytes).hexdigest():
        raise ValueError('incomplete experiment or wrong protocol')
    seeds = plan['precisions']['bf16']
    expected = {f'confirm-bf16-{mode}-{seed}.json' for mode in plan['modes'] for seed in seeds}
    if len(index['results']) != len(expected) or {x['file'] for x in index['results']} != expected:
        raise ValueError('missing, duplicated or unexpected runs')
    records = {}
    for entry in index['results']:
        data = gzip.decompress((here / 'raw' / (entry['file'] + '.gz')).read_bytes())
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('raw checksum mismatch')
        r = json.loads(data)
        if r['status'] != 'complete' or r['git'] != {'head': SOURCE, 'status': ''}:
            raise ValueError('incomplete run or changed source')
        c = r['config']
        for k in ('steps','batch_size','eval_interval','eval_size','test_size','lr','cpu_threads','device'):
            if c[k] != plan[k]:
                raise ValueError('config mismatch: ' + k)
        if c['precision'] != 'bf16' or len(r['steps']) != plan['steps']:
            raise ValueError('precision or update count mismatch')
        if [x['step'] for x in r['evaluations']] != list(range(0,1601,50)):
            raise ValueError('incomplete validation trajectory')
        if not all(math.isfinite(x['validation']['ce']) for x in r['evaluations']) or not math.isfinite(r['test']['ce']):
            raise ValueError('non-finite loss')
        if any(x['step'] != i or not all(math.isfinite(x[k]) for k in ('loss','lr','grad_norm')) for i,x in enumerate(r['steps'],1)):
            raise ValueError('invalid training trace')
        records[c['mode'], c['seed']] = r
    pairs = {}
    for candidate, reference in [('tiger-full','tiger-no-spectral'),('tiger-spectral-fade','tiger-full')]:
        rows = []
        for seed in seeds:
            a,b = records[candidate,seed],records[reference,seed]
            for k in ('initial_state_sha256','data_sha256','source_sha256','model'):
                if a[k] != b[k]:
                    raise ValueError('unmatched pair: ' + k)
            if [x['lr'] for x in a['steps']] != [x['lr'] for x in b['steps']]:
                raise ValueError('unmatched LR schedule')
            trajectory = [{'step': x['step'], 'ce_delta': x['validation']['ce']-y['validation']['ce']}
                          for x,y in zip(a['evaluations'],b['evaluations'])]
            if candidate == 'tiger-spectral-fade' and any(x['model_state_sha256'] != y['model_state_sha256'] for x,y in zip(a['evaluations'],b['evaluations']) if x['step'] <= 800):
                raise ValueError('full/fade first-half mismatch')
            rows.append({'seed':seed,'test_ce_delta':a['test']['ce']-b['test']['ce'],
                         'late_validation_delta_change':trajectory[-1]['ce_delta']-trajectory[16]['ce_delta'],
                         'trajectory':trajectory})
        pairs[candidate+'_minus_'+reference] = {'seeds':rows,
            'mean_test_ce_delta':statistics.mean(r['test_ce_delta'] for r in rows),
            'median_test_ce_delta':statistics.median(r['test_ce_delta'] for r in rows),
            'improved_test_seeds':sum(r['test_ce_delta']<0 for r in rows),
            'mean_late_validation_delta_change':statistics.mean(r['late_validation_delta_change'] for r in rows)}
    gate = plan['binding_gate']
    return {'source_commit':SOURCE,'plan_sha256':index['plan_sha256'],'runs':len(records),
            'updates':len(records)*plan['steps'],'environment':next(iter(records.values()))['environment'],
            'binding':[{ 'mode':mode,'seed':seed,'test_ce':r['test']['ce'],
                         'passed':r['test']['accuracy']>=gate['original_min'] and r['binding_check']['rebound']['accuracy']>=gate['rebound_min'] and r['binding_check']['old_targets']['accuracy']<=gate['obsolete_max']}
                        for (mode,seed),r in records.items()], 'contrasts':pairs}


if __name__ == '__main__':
    print(json.dumps(summarize(),indent=2,allow_nan=False))
