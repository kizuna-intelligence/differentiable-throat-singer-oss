from diffart.dataset import stage1_audit
from diffart.data import ROOT
from diffart.runtime import save_json
from pathlib import Path
import numpy as np

c=stage1_audit();rows=[]
for r in c.manifest['songs']:
    if r['excluded']:continue
    d=c.load(r);a=d['f0'];b=d['score_f0'];voiced=(a>0)&(b>0)
    error=np.abs(12*np.log2(a[voiced]/b[voiced]))
    rows.append(dict(song=r['id'],score_p05=r['score_p05'],score_p95=r['score_p95'],
      both_voiced_fraction=float(voiced.mean()),median_score_recording_semitone_error=float(np.median(error)),
      label_audio_end_difference_seconds=r['label_end']-r['duration']))
save_json(ROOT/'artifacts/score_alignment_audit.json',rows)
print('STAGE1 PASS',len(rows))
