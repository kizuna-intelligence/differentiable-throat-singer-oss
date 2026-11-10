"""Finalize reports only after the training/evaluation pipeline succeeds."""
import json,os,subprocess,time
from diffart.data import ROOT
from diffart.runtime import save_json

def main():
    path=ROOT/'artifacts/pipeline_progress.json'
    while True:
        state=json.loads(path.read_text())
        if state['status']=='failed':raise SystemExit('Evaluation pipeline failed; finalization withheld')
        if state['status']=='complete':break
        time.sleep(5)
    env=os.environ.copy();env.update(PYTHONPATH=str(ROOT),OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3')
    for script in ['summarize_stage5.py','audit_outputs.py','update_status.py']:
        subprocess.run([str(ROOT/'.venv/bin/python'),'scripts/'+script],cwd=ROOT,env=env,check=True)
    save_json(ROOT/'artifacts/FINALIZED.json',dict(complete=True,status='STATUS.md',audio='listen/README.md'))
    print('Final reports and audio audit complete',flush=True)
if __name__=='__main__':main()
