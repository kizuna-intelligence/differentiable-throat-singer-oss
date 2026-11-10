import torch
import numpy as np
from diffart.metrics import SpeakerMetrics,EcapaMetrics,Transcriptor,audio16
from diffart.data import ROOT
from diffart.runtime import save_json

torch.set_num_threads(3)
p=ROOT/'data/audio24/1st_color+3_normal.wav'
y=audio16(p)[16000*12:16000*16]
report={}
for name,cls in [('wavlm',SpeakerMetrics),('ecapa',EcapaMetrics)]:
    m=cls();e=m.embedding(y);e2=m.embedding(y.copy())
    report[name]=dict(dim=len(e),self_cosine=float(e@e2),finite=bool(np.isfinite(e).all()))
    assert np.isfinite(e).all() and e@e2>.999
    save_json(ROOT/'artifacts/metric_smoke.json',report);print(name,report[name],flush=True)
