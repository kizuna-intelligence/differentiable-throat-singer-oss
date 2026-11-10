"""Small, self-contained pitch sanity check; not a speaker/recognition evaluator."""
import numpy as np
from scipy.signal import correlate, find_peaks
class AutocorrelationPitch:
    def extract_continuous_periodicity(self,audio):
        n=len(audio)//240;pitch=np.zeros(n);confidence=np.zeros(n)
        padded=np.pad(audio,(1024,1024))
        for i in range(n):
            x=padded[i*240:i*240+2048].astype(np.float64);x-=x.mean()
            if np.max(np.abs(x))<1e-8:continue
            x*=np.hanning(len(x));ac=correlate(x,x,mode='full',method='fft')[len(x)-1:]
            if ac[0]<=1e-20:continue
            lo,hi=12, min(757,len(ac)-2);peaks,_=find_peaks(ac[lo:hi])
            if not len(peaks):continue
            peaks+=lo;k=int(peaks[np.argmax(ac[peaks])]);den=ac[k-1]-2*ac[k]+ac[k+1]
            offset=.5*(ac[k-1]-ac[k+1])/den if abs(den)>1e-20 else 0.
            pitch[i]=24000/(k+offset);confidence[i]=max(0,min(1,ac[k]/ac[0]))
        return pitch,confidence
