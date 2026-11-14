"""Verify the precision replication and derive paired loss contrasts."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent


def summarize(here=HERE):
    plan_bytes=(here/'plan.json').read_bytes();plan=json.loads(plan_bytes)
    index=json.loads((here/'index.json').read_text())
    if not index['complete'] or index['plan_sha256']!=hashlib.sha256(plan_bytes).hexdigest():
        raise ValueError('incomplete or mismatched plan')
    seeds=plan['precisions']['fp32'];modes=plan['modes']
    expected={f'confirm-fp32-{m}-{s}.json' for s in seeds for m in modes}|{'confirm-bf16-tiger-full-701.json'}
    if len(index['results'])!=len(expected) or {r['file'] for r in index['results']}!=expected:
        raise ValueError('unexpected run set')
    records={}
    for entry in index['results']:
        data=gzip.decompress((here/'raw'/(entry['file']+'.gz')).read_bytes())
        if hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('checksum mismatch')
        r=json.loads(data);c=r['config']
        if entry['file']!=f"confirm-{c['precision']}-{c['mode']}-{c['seed']}.json":raise ValueError('run identity mismatch')
        if r['git']!={'head':plan['source_commit'],'status':''} or r['status']!='complete':raise ValueError('source or status mismatch')
        for k in ('steps','lr','device','cpu_threads','batch_size','eval_interval','eval_size','test_size',*plan['model']):
            if c[k] != (plan['model'][k] if k in plan['model'] else plan[k]):raise ValueError('config mismatch: '+k)
        if len(r['steps'])!=plan['steps'] or [e['step'] for e in r['evaluations']]!=list(range(0,1601,50)):raise ValueError('incomplete trace')
        if any(x['step']!=i or not all(math.isfinite(x[k]) for k in ('loss','lr','grad_norm')) for i,x in enumerate(r['steps'],1)):raise ValueError('invalid step trace')
        if not math.isfinite(r['test']['ce']) or not all(math.isfinite(e['validation']['ce']) for e in r['evaluations']):raise ValueError('nonfinite evaluation')
        records[c['precision'],c['mode'],c['seed']]=r
    old=here.parent/'2026-11-10-spectral-late'
    old_index=json.loads((old/'confirmation_index.json').read_text())
    for entry in old_index['results']:
        data=gzip.decompress((old/'raw'/(entry['file']+'.gz')).read_bytes())
        if hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('historical checksum mismatch')
        r=json.loads(data);c=r['config'];records['historical',c['mode'],c['seed']]=r
    def matched(a,b):
        for k in ('initial_state_sha256','data_sha256','source_sha256','model'):
            if a[k]!=b[k]:raise ValueError('unmatched '+k)
        if [x['lr'] for x in a['steps']]!=[x['lr'] for x in b['steps']]:raise ValueError('unmatched LR')
    bridge=records['bf16','tiger-full',701];reference=records['historical','tiger-full',701]
    matched(bridge,reference)
    if bridge['final_state_sha256']!=reference['final_state_sha256'] or [x['loss'] for x in bridge['steps']]!=[x['loss'] for x in reference['steps']]:raise ValueError('bridge mismatch')
    for seed in seeds:
        for mode in modes:
            current,prior=records['fp32',mode,seed],records['historical',mode,seed]
            matched(current,prior)
            for key in ('torch','cuda','device','tf32','cpu_threads'):
                if current['environment'][key]!=prior['environment'][key]:raise ValueError('environment mismatch: '+key)
        a,b=records['fp32','tiger-full',seed],records['fp32','tiger-spectral-fade',seed]
        if any(x['model_state_sha256']!=y['model_state_sha256'] for x,y in zip(a['evaluations'],b['evaluations']) if x['step']<=800):raise ValueError('pre-fade mismatch')
    contrasts={}
    for a,b in [('tiger-full','tiger-no-spectral'),('tiger-spectral-fade','tiger-full')]:
        rows=[]
        for seed in seeds:
            matched(records['fp32',a,seed],records['fp32',b,seed])
            fp=records['fp32',a,seed]['test']['ce']-records['fp32',b,seed]['test']['ce']
            bf=records['historical',a,seed]['test']['ce']-records['historical',b,seed]['test']['ce']
            rows.append({'seed':seed,'fp32_delta':fp,'bf16_delta':bf,'precision_delta':fp-bf})
        contrasts[a+'_minus_'+b]={'seeds':rows,'mean_fp32_delta':statistics.mean(x['fp32_delta'] for x in rows),'median_fp32_delta':statistics.median(x['fp32_delta'] for x in rows),'fp32_improved_seeds':sum(x['fp32_delta']<0 for x in rows),'mean_precision_delta':statistics.mean(x['precision_delta'] for x in rows)}
    gate=plan['binding_gate']
    binding=[{'mode':m,'seed':s,'test_ce':records['fp32',m,s]['test']['ce'],'passed':records['fp32',m,s]['test']['accuracy']>=gate['original_min'] and records['fp32',m,s]['binding_check']['rebound']['accuracy']>=gate['rebound_min'] and records['fp32',m,s]['binding_check']['old_targets']['accuracy']<=gate['obsolete_max']} for s in seeds for m in modes]
    return {'plan_sha256':index['plan_sha256'],'source_commit':plan['source_commit'],'runs':16,'updates':25600,'bridge_exact':True,'environment':records['fp32',modes[0],seeds[0]]['environment'],'binding':binding,'contrasts':contrasts}


if __name__=='__main__':print(json.dumps(summarize(),indent=2,allow_nan=False))
