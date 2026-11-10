import argparse,json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diffart.data import ROOT
from diffart.physics import Synthesizer
from diffart.dataset import Corpus
from diffart.runtime import save_json

ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',default='artifacts/main/best.pt');ap.add_argument('--out',default='artifacts/anatomy');args=ap.parse_args()
torch.set_num_threads(2)
c=Corpus();m=Synthesizer(len(c.manifest['vocab']))
m.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=True)['model']);m.eval()
x=np.linspace(0,float(m.speaker.length.detach())*100,44)
with torch.no_grad():neutral=m.speaker.area(torch.zeros(9)).numpy()
means={}
with torch.no_grad():
    for vowel in ['a','i','u','e','o']:
        all_areas=[]
        for row in [r for r in c.manifest['songs'] if r['split']=='train' and r.get('train_seconds',0)>0]:
            d=c.load(row);n=int(row['train_seconds']*100)
            ix=np.flatnonzero(d['context'][:n,1]==c.manifest['vocab'][vowel])[::10]
            if len(ix):
                p=m.controls(torch.tensor(d['context'][ix]),torch.tensor(d['pos'][ix]),torch.tensor(d['f0'][ix]),m.speaker)
                all_areas.append(m.speaker.area(p['art']).numpy())
        means[vowel]=np.concatenate(all_areas).mean(0).tolist()
fig,axes=plt.subplots(1,2,figsize=(11,4));axes[0].plot(x,neutral,label='neutral',color='black',linestyle='--')
for vowel,a in means.items():axes[0].plot(x,a,label=vowel)
axes[0].set(xlabel='Distance from glottis (cm)',ylabel='Area (cm²)',ylim=(0,8));axes[0].legend()
axes[1].plot(np.linspace(0,12,20),m.speaker.nose.detach().numpy());axes[1].set(xlabel='Nasal branch distance (cm)',ylabel='Area (cm²)',ylim=(0,8))
fig.suptitle(f'Fitted consistent solution; L = {x[-1]:.3f} cm (not anatomical ground truth)')
out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(out/'areas.png',dpi=150)
save_json(out/'parameters.json',dict(length_cm=x[-1],neutral_area_cm2=neutral.tolist(),vowel_means=means,nasal_area_cm2=m.speaker.nose.detach().tolist(),
    wall_loss_neper_per_m=float(m.speaker.wall.detach()),radiation_coefficient=float(m.speaker.rad.detach()),interpretation='One consistent inverse solution; not actual anatomy'))
