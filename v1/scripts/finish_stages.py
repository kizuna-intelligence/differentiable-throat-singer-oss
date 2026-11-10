"""Continue only our own stage-4 process, then run the complete evaluation."""
import json,os,signal,subprocess,time
from pathlib import Path
from diffart.data import ROOT
from diffart.runtime import save_json

GPU='GPU-cd879302-07f2-435c-b24f-c3fdab3bbf7d'
PY=str(ROOT/'.venv/bin/python')
env=os.environ.copy();env.update(PYTHONPATH=str(ROOT),OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3')

def run(name,args):
    print('START',name,flush=True)
    save_json(ROOT/'artifacts/pipeline_progress.json',dict(stage=name,status='running',time=time.time()))
    with (ROOT/'artifacts'/f'{name}_console.log').open('a') as log:
        result=subprocess.run([PY,*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        save_json(ROOT/'artifacts/pipeline_progress.json',dict(stage=name,status='failed',returncode=result.returncode,time=time.time()))
        raise SystemExit(f'{name} failed ({result.returncode}); see log')
    print('DONE',name,flush=True)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--first-leg-pid',type=int);args=ap.parse_args()
    if args.first_leg_pid:
        proc=Path('/proc')/str(args.first_leg_pid)
        original_cmd=(proc/'cmdline').read_bytes()
        assert b'scripts/train.py\x00main\x00' in original_cmd and proc.stat().st_uid==os.getuid()
        original_start=(proc/'stat').read_text().split()[21]
        while proc.exists():
            progress=json.loads((ROOT/'artifacts/main/progress.json').read_text())
            if progress['step']>0:
                assert (proc/'cmdline').read_bytes()==original_cmd
                assert (proc/'stat').read_text().split()[21]==original_start
                save_json(ROOT/'artifacts/main/batch_transition.json',dict(first_leg_pid=args.first_leg_pid,
                    checkpoint=progress,reason='First complete epoch saved; resume model and optimizer with batch 4 for throughput',
                    stopped_process='own explicitly identified training process only',
                    sampler='Old checkpoint lacks sampler order; next epoch uses a deterministic new shuffle from saved RNG state'))
                os.kill(args.first_leg_pid,signal.SIGINT)
                break
            time.sleep(5)
        while proc.exists():time.sleep(1)
    if not (ROOT/'artifacts/main/result.json').exists():
        run('main_batched',['-u','scripts/train.py','main','--device',GPU,'--steps','36000','--patience','3','--batch-size','4','--resume'])
    run('evaluation',['-u','scripts/evaluate.py','--device',GPU])
    run('noise',['-u','scripts/diagnose_noise.py','--device',GPU])
    run('anatomy',['scripts/plot_anatomy.py'])
    run('irodori',['-u','scripts/run_original_irodori.py','--device',GPU])
    run('original_evaluation',['-u','scripts/evaluate_original.py','--device',GPU])
    run('summary',['scripts/summarize_stage5.py'])
    save_json(ROOT/'artifacts/pipeline_progress.json',dict(stage='complete',status='complete',time=time.time()))
