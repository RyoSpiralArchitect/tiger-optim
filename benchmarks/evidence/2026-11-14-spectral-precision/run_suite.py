"""Run a frozen-source bridge followed by five paired FP32 seeds."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    plan_bytes = (HERE / 'plan.json').read_bytes()
    plan = json.loads(plan_bytes)
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip() != plan['source_commit'] or subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip():
        raise ValueError('source must be clean frozen commit')
    output = args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=False)
    jobs = [('bf16',701,'tiger-full')]
    for i,seed in enumerate(plan['precisions']['fp32']):
        modes = plan['modes']
        jobs += [('fp32',seed,m) for m in modes[i%3:]+modes[:i%3]]
    records=[]
    for precision,seed,mode in jobs:
        name=f'confirm-{precision}-{mode}-{seed}.json'
        command=[sys.executable,str(source/'benchmarks/bench_associative_recall.py'),'--precision',precision,'--seed',str(seed),'--mode',mode,'--measure-qkv-updates','--output',name]
        options={k:plan[k] for k in ('device','cpu_threads','lr','steps','batch_size','eval_interval','eval_size','test_size')}
        options.update(plan['model'])
        for k,v in options.items():command.extend(['--'+k.replace('_','-'),str(v)])
        subprocess.run(command,cwd=output,env={**os.environ,'TIGER_FLAT_STEP':'0'},check=True)
        data=(output/name).read_bytes();r=json.loads(data)
        if precision=='bf16':
            original=json.loads(gzip.decompress((HERE.parent/'2026-11-10-spectral-late/raw'/ (name+'.gz')).read_bytes()))
            for k in ('initial_state_sha256','final_state_sha256','source_sha256','data_sha256'):
                if r[k]!=original[k]:raise ValueError('bridge mismatch: '+k)
            for k in ('loss','lr'):
                if [x[k] for x in r['steps']] != [x[k] for x in original['steps']]:raise ValueError('bridge trace mismatch: '+k)
        records.append({'file':name,'sha256':hashlib.sha256(data).hexdigest()})
        (output/'index.json').write_text(json.dumps({'plan_sha256':hashlib.sha256(plan_bytes).hexdigest(),'complete':len(records)==len(jobs),'results':records},indent=2)+'\n')


if __name__=='__main__':main()
