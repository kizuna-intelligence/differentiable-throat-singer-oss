"""Final artifact integrity and audio checks; does not alter generated sound."""
import json,subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
from diffart.data import ROOT,sha
from diffart.runtime import save_json

def main():
    rows=[]
    for path in sorted((ROOT/'listen').rglob('*.wav')):
        peak=0.;energy=0.;n=0;saturated=0
        with sf.SoundFile(path) as f:
            sr=f.samplerate;channels=f.channels;subtype=f.subtype
            for block in f.blocks(blocksize=240000,dtype='float32',always_2d=True):
                assert np.isfinite(block).all(),path
                peak=max(peak,float(np.abs(block).max()));energy+=float(np.square(block,dtype=np.float64).sum());n+=block.size
                if subtype=='PCM_16':saturated+=int(((block>=32767/32768)|(block<=-1)).sum())
        assert n>0 and sr in [24000,44100],path
        rows.append(dict(path=str(path.relative_to(ROOT)),seconds=n/channels/sr,sr=sr,channels=channels,subtype=subtype,
            peak=peak,rms=(energy/n)**.5,pcm_saturated_samples=saturated,sha256=sha(path)))
    assert all((ROOT/'listen'/name).exists() for name in ['01_再現.wav','02_外挿.wav','03_発声の変化.wav','04_モデル間比較.wav'])
    for r in rows:
        if Path(r['path']).parent==Path('listen'):assert r['pcm_saturated_samples']==0
    provenance=json.loads((ROOT/'artifacts/comparison/frontend_provenance.json').read_text())
    assert provenance['db_score_used'] is False
    for shift in [-12,-5,0,5,7,12]:
        complete=json.loads((ROOT/'listen/original_verse1'/f'irodori_{shift:+d}'/'completed.json').read_text())
        assert complete['db_score_used'] is False
        assert complete['original_front_sha256']==provenance['irodori_front_sha256']
        assert complete['checkpoint_sha256']=='c998b284e8d21bc7a69383092f717ffff238d029f816d146a03de5b75b67a624'
        assert complete['command'][complete['command'].index('--front')+1]==complete['original_front']
    save_json(ROOT/'artifacts/audio_audit.json',dict(passed=True,wav_files=len(rows),files=rows,
        total_seconds=sum(r['seconds'] for r in rows),all_finite=True,top_level_reels_not_clipped=True,
        original_score_only_for_other_model=True,pcm_files_with_saturation=[r['path'] for r in rows if r['pcm_saturated_samples']]))
    lockrows=json.loads(subprocess.check_output(['lslocks','--json'],text=True)).get('locks',[])
    save_json(ROOT/'artifacts/resources_final.json',dict(gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv'],text=True),
        shared_gpu_locks=[r for r in lockrows if 'cyborgy-gpu' in r.get('path','')]))
    save_json(ROOT/'artifacts/stage45_final_code_hashes.json',{str(p.relative_to(ROOT)):sha(p) for folder in ['scripts','diffart'] for p in sorted((ROOT/folder).glob('*.py'))})
    print('Audio audit passed:',len(rows),'WAVs;',sum(r['pcm_saturated_samples'] for r in rows),'saturated PCM samples')
if __name__=='__main__':main()
