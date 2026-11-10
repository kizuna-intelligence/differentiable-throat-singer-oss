"""Resume all 3 CPU ablations through training, full evaluation and B report.

--watch-pids attaches to already started, owned training workers. It never
signals or changes another process. Without it, launches/resumes training.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import threading
from diffart.data import ROOT
from diffart.runtime import save_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--watch-pids',nargs=3,type=int);args=ap.parse_args()
    dest=ROOT/'artifacts/improvements';dest.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1')
    states={}
    status_lock=threading.Lock()
    def status(variant,stage):
        with status_lock:
            states[variant]=dict(stage=stage,time=time.time())
            save_json(dest/'workflow.json',states)
            print(variant,stage,flush=True)
    def run(variant,script,extra):
        with (dest/f'{variant}_{script}.log').open('a') as log:
            subprocess.run([sys.executable,'-u',f'scripts/{script}.py',*extra],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    def worker(index,variant):
        if not (dest/variant/'result.json').exists():
            status(variant,'training')
            if args.watch_pids:
                while not (dest/variant/'result.json').exists():
                    try:os.kill(args.watch_pids[index],0)
                    except ProcessLookupError:raise RuntimeError(f'{variant} training exited without result.json')
                    time.sleep(10)
            else:
                run(variant,'train_improvements',[variant,'--device','cpu','--threads','2','--batch-size','4','--resume'])
        status(variant,'evaluation')
        if not (dest/variant/'evaluation/results.json').exists():
            run(variant,'evaluate_improvements',[variant,'--device','cpu','--threads','2'])
        status(variant,'diagnosis')
        if not (dest/variant/'diagnosis/learned_parameters.json').exists():
            run(variant,'report_improvements',['--diagnose',variant])
        status(variant,'complete')
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures=[executor.submit(worker,i,v) for i,v in enumerate(['b1','b2','both'])]
        for future in futures:future.result()
    subprocess.run([sys.executable,'scripts/report_improvements.py'],cwd=ROOT,env=env,check=True)
    print('all complete',flush=True)

if __name__=='__main__':main()
