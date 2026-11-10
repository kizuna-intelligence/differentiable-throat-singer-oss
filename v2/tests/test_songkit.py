"""Score semantics and preservation of the real CPU acoustic graph."""
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from diffart.songkit.patterns import PATTERNS,expand
from diffart.songkit.pipeline import load_spec,REPO

class SongkitTests(unittest.TestCase):
    def test_authored_lead_is_preserved(self):
        path=REPO/'examples/ievan_allvoice/song.json'
        original=json.loads(path.read_text());spec=load_spec(path)
        lead=next(p for p in spec['parts'] if p['id']=='lead')
        authored=next(p for p in original['parts'] if p['id']=='lead')
        self.assertEqual(lead['notes'],authored['notes'])
        self.assertEqual((spec['bpm'],spec['time_signature'],spec['length_beats']),(150,'2/4',60))

    def test_path_like_names_are_rejected(self):
        original=json.loads((REPO/'examples/ievan_allvoice/trial_3_4.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'candidate.json'
            for field,value in [('output_name','../escape'),('output_name',r'..\escape'),
                                ('output_name',r'C:\escape'),('part id','../escape'),
                                ('part id',r'..\escape'),('part id',r'C:\escape')]:
                candidate=copy.deepcopy(original)
                if field=='part id':candidate['parts'][0]['id']=value
                else:candidate['output_name']=value
                path.write_text(json.dumps(candidate))
                with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                    load_spec(path)

    def test_all_patterns_in_another_meter_and_key(self):
        spec=load_spec(REPO/'examples/ievan_allvoice/trial_3_4.json')
        self.assertEqual((spec['beats_per_bar'],spec['seconds'],spec['length_beats']),(3,12,24))
        for p in spec['parts']:
            self.assertTrue(all(n['start']+n['dur']<=24 for n in p['notes']))
        harmony=list(PATTERNS['harmony_third_below'](spec,{'source':'lead'}))
        self.assertEqual([n['midi'] for n in harmony[:8]],[57,59,60,62,64,65,67,69])
        sustain=list(PATTERNS['sustain_chord'](spec,{'voice':1}))
        self.assertEqual(sum(n['dur'] for n in sustain),24)
        percussion=list(PATTERNS['mouth_perc'](spec,{'step':.25,'offset':0}))
        self.assertEqual(len(percussion),96)
        self.assertTrue(all(not n['pitched'] for n in percussion))

    def test_cpu_observation_does_not_change_acoustics(self):
        from diffart.songkit import cpu
        from diffart.chimera import numba_graph
        from diffart.chimera.coefficients import fixed_parameters,initial_state
        ck=torch.load(REPO/'models/throat_ritsu_v2_chimera.pt',map_location='cpu',weights_only=True)
        m=cpu.make_model(ck['config'],ck['config']['vocab'],ck['model'],1)
        vocab=ck['config']['vocab'];ctx=torch.tensor([[[vocab['a']]*3]*2]);pos=torch.tensor([[.3,.7]]);f0=torch.tensor([[220.,220.]])
        torch.set_num_threads(2)
        with torch.inference_mode():_,geo,c,glot,ng,n=m.prepare(ctx,pos,f0)
        width=c.shape[-1];C=c[0].numpy().astype(np.float64);H=fixed_parameters(m.airway,width,c).numpy().astype(np.float64);S=initial_state(m.airway,1,width,c)[0].numpy().astype(np.float64);G=glot[0].numpy().astype(np.float64)[:,None]
        with tempfile.TemporaryDirectory() as d:
            cpu.CACHE=Path(d);numba_graph.__file__=str(Path(d)/'package/a/numba_graph.py')
            baseline=numba_graph.compile_forward(m.airway,width)
            y,area=baseline.forward(C,H,G,S,96)
            instrumented=cpu.compile_noisy_numba(m.airway,width)
            y2,area2,flow=instrumented(C,H,G,S,96,np.zeros((n,width)))
            np.testing.assert_array_equal(y,y2);np.testing.assert_array_equal(area,area2)
            self.assertTrue(np.isfinite(flow).all());self.assertGreater(np.max(np.abs(flow)),0)

if __name__=='__main__':unittest.main()
