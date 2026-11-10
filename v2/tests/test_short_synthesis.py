"""Actual CPU waveform, not a mocked synthesizer."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from diffart.songkit import cpu
from diffart.songkit.pipeline import REPO,load_spec
from diffart.songkit.frontend import inputs
class Synthesis(unittest.TestCase):
    def test_short_cpu_song(self):
        ck=torch.load(REPO/'models/throat_ritsu_v2_chimera.pt',map_location='cpu',weights_only=True)
        spec=load_spec(REPO/'examples/ievan_allvoice/trial_3_4.json');part=dict(spec['parts'][0])
        part['notes']=[dict(start=0.,dur=.5,midi=60,lyric='あ',pitched=True)]
        spec=dict(spec,seconds=.25)
        ctx,pos,f0,rows=inputs(spec,part,ck['config']['vocab'])
        model=cpu.make_model(ck['config'],ck['config']['vocab'],ck['model'],1.)
        with tempfile.TemporaryDirectory() as d:
            cpu.CACHE=Path(d);audio,controls,report=cpu.cpu_render(model,ctx,pos,f0,1234)
        self.assertEqual(len(audio),6000);self.assertTrue(np.isfinite(audio).all());self.assertGreater(float(abs(audio).max()),1e-12)
        self.assertEqual(len(controls['states']),25);self.assertEqual(report['internal_samples'],24000)
