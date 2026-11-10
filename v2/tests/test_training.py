"""Offline singing/speech preparation, real acoustic adjoint and optimizer steps."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from diffart.chimera.prepare import VOCAB, prepare
from diffart.chimera.training import Corpus, default_config, make_model, fit, evaluate


class Training(unittest.TestCase):
    def test_singing_and_speech_loop(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rows=[]
            # All content authored here: no corpus recordings, labels or text.
            for i,(kind,split) in enumerate([('singing','train'),('speech','train'),('speech','val'),('singing','test')]):
                audio=.03*np.sin(2*np.pi*(220+i)*np.arange(480)/24000)
                sf.write(root/f'{i}.wav',audio,24000,subtype='FLOAT')
                np.save(root/f'{i}.npy',np.full(2,220+i,dtype=np.float32))
                rows.append(dict(audio=f'{i}.wav',kind=kind,split=split,text='あ',
                                 phones=[[0,.02,'a']],f0=f'{i}.npy'))
            (root/'input.json').write_text(json.dumps(rows))
            prepare(root/'input.json',root/'prepared')
            corpus=Corpus(root/'prepared',frames=2)
            self.assertEqual([len(corpus.blocks[s]) for s in ('train','val','test')],[2,1,1])
            history=fit(corpus,root/'trained',epochs=2,patience=3,batch_size=1)
            self.assertEqual(len(history),2)  # Four real forward/backward optimizer updates.
            ck=torch.load(root/'trained/last.pt',map_location='cpu',weights_only=True)
            initial=torch.load(root/'trained/initial.pt',map_location='cpu',weights_only=True)
            for name in ('microphone_log_gain','source_net.2.bias','anatomy.body_logits'):
                self.assertFalse(torch.equal(ck['model'][name],initial['model'][name]),name)
            model=make_model(ck['config'],ck['model'])
            self.assertTrue(np.isfinite(evaluate(model,corpus,'test',batch_size=1)))
            self.assertEqual(ck['config']['correction_coefficient'],default_config(VOCAB)['correction_coefficient'])

    def test_acoustic_adjoint_finite_difference(self):
        torch.manual_seed(123)
        m=make_model(default_config(VOCAB)).double()
        phones=torch.full((1,2,3),VOCAB['a']);pos=torch.tensor([[.2,.8]],dtype=torch.float64)
        pitch=torch.full((1,2),220.,dtype=torch.float64)
        def objective():
            # Real output propagated through the learned glottis and acoustic recurrence.
            return m(phones,pos,pitch,seed=123).square().mean()*1e8
        value=objective();value.backward();p=m.glottis.area_raw
        analytic=p.grad.detach().clone();delta=1e-4
        with torch.no_grad():
            original=p.clone();p.copy_(original+delta);plus=objective()
            p.copy_(original-delta);minus=objective();p.copy_(original)
        expected=(plus-minus)/(2*delta)
        self.assertGreater(float(expected.abs()),1e-14)
        torch.testing.assert_close(analytic,expected,rtol=.002,atol=1e-9)

    def test_prepared_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'manifest.json').write_text(json.dumps(dict(vocab=VOCAB,items=[dict(file='../outside.npz',split='train')])))
            with self.assertRaisesRegex(ValueError,'escapes'):
                Corpus(root,frames=2)
