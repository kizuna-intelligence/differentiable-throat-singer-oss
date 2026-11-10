import json
import hashlib
import math
import unittest
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
import torch
from diffart.physics import C, transfer, Synthesizer, lf_flow, lf_parameters

torch.set_num_threads(2)
REPORT={}
class Physics(unittest.TestCase):
    def test_uniform_formants(self):
        freq=torch.linspace(0,7000,28001,dtype=torch.float64)
        results=[]
        for length in [.13,.15,.19]:
            h=transfer(torch.ones(44,dtype=torch.float64),torch.tensor(length),torch.tensor(.3),freq)
            peaks,_=find_peaks(h.abs().numpy()); actual=freq[peaks[:5]].numpy()
            expected=(2*np.arange(1,6)-1)*C/(4*length)
            error=float(np.max(np.abs(actual-expected)))
            self.assertLess(error,.5);results.append(dict(length=length,actual=actual.tolist(),expected=expected.tolist(),max_error_hz=error))
        REPORT['uniform_formants']=results
    def test_time_domain_reference(self):
        # Independent pressure-wave scattering, one-sample delay per tube.
        # Pick sr to make each of 44 section delays exactly one sample.
        n=44;length=.16;sr=n*C/length;size=16384;loss=8.
        area=1.8+.7*np.sin(np.linspace(0,7,n));z=1/area
        r=(area[:-1]-area[1:])/(area[:-1]+area[1:])
        atten=math.exp(-loss*length/n)
        forward=np.zeros(n);back=np.zeros(n);out=[]
        for t in range(size):
            f=forward*atten;b=back*atten
            out.append(2*f[-1]/z[-1])
            newf=np.empty(n);newb=np.empty(n)
            newf[0]=z[0]*(1 if t==0 else 0)+b[0]
            newf[1:]=(1+r)*f[:-1]-r*b[1:]
            newb[:-1]=r*f[:-1]+(1-r)*b[1:]
            newb[-1]=-f[-1]
            forward,back=newf,newb
        freq=torch.fft.rfftfreq(size,1/sr).double()
        h=transfer(torch.tensor(area),torch.tensor(length,dtype=torch.float64),torch.tensor(loss),freq)
        expected=torch.fft.irfft(h,n=size).numpy()
        err=float(np.linalg.norm(np.array(out)-expected)/np.linalg.norm(expected))
        self.assertLess(err,1e-5);REPORT['time_domain_relative_error']=err
    def test_lf_range_and_gradient(self):
        rd=torch.linspace(.3,2.7,49,dtype=torch.float64)[:,None].requires_grad_()
        phase=torch.linspace(0,1,1001,dtype=torch.float64)[None,:]
        flow=lf_flow(phase,rd)
        self.assertTrue(torch.isfinite(flow).all())
        self.assertLess(float(flow[:,[0,-1]].abs().max().detach()),1e-5)
        self.assertGreater(float(flow.min().detach()),-1e-5)
        self.assertLess(float(flow.max().detach()),1.001)
        flow.square().mean().backward();self.assertIsNotNone(rd.grad);self.assertTrue(torch.isfinite(rd.grad).all());self.assertGreater(float(rd.grad.norm()),0)
        REPORT['lf']={'minimum':float(flow.min().detach()),'maximum':float(flow.max().detach()),'boundary':float(flow[:,[0,-1]].abs().max().detach())}
    def test_lf_finite_difference(self):
        rd=torch.tensor([.5,1.5,2.5],dtype=torch.float64,requires_grad=True)
        phase=torch.tensor([.1,.4,.7],dtype=torch.float64)
        self.assertTrue(torch.autograd.gradcheck(lambda r:lf_flow(phase,r), (rd,),eps=1e-4,atol=2e-4,rtol=2e-3))
        REPORT['lf_implicit_gradient_check']=True
    def test_all_gradients_and_f0_isolation(self):
        torch.manual_seed(2);model=Synthesizer(12)
        phones=torch.randint(0,12,(1,12,3));pos=torch.rand(1,12);f0=torch.full((1,12),210.)
        p=model.controls(phones,pos,f0,model.speaker);q=model.controls(phones,pos,f0*2,model.speaker)
        for name in ['art','velum','breath','turbulence','location']:self.assertTrue(torch.equal(p[name],q[name]),name)
        # Padding enough for short STFT; assess all trainable tensors.
        y,p=model(phones,pos,f0,return_controls=True)
        loss=torch.log1p(torch.stft(y,256,64,return_complex=True,window=torch.hann_window(256)).abs()).mean()+model.regularization(p)
        loss.backward();rows={}
        for name,value in model.named_parameters():
            self.assertIsNotNone(value.grad,name);self.assertTrue(torch.isfinite(value.grad).all(),name)
            norm=float(value.grad.norm());self.assertGreater(norm,0,name);rows[name]=norm
        REPORT['gradient_norms']=rows;REPORT['f0_shape_isolation']=True
    @classmethod
    def tearDownClass(cls):
        import os
        report=os.environ.get('THROAT_TEST_REPORT')
        if report:
            destination=Path(report);destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_text(json.dumps(REPORT,indent=2))
if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Physics)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    REPORT['passed']=result.wasSuccessful()
    REPORT['physics_sha256']=hashlib.sha256(Path('diffart/physics.py').read_bytes()).hexdigest()
    Path('artifacts/unit_results.json').write_text(json.dumps(REPORT,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
