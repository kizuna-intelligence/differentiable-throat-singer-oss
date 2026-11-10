"""Independent reference-recording metrics on CPU, while model training runs."""
import json,torch
from diffart.dataset import Corpus
from diffart.data import ROOT,HOP,read_lab,sha
from diffart.metrics import label_kana
from diffart.runtime import save_json
from evaluation_support import Assessment

def main():
    torch.set_num_threads(3);corpus=Corpus();metric=Assessment(corpus,'cpu')
    refs=json.loads((ROOT/'artifacts/evaluation/references.json').read_text())
    provenance=dict(manifest_sha256=sha(ROOT/'data/manifest.json'),device='cpu',
        references=[dict(path=r['path'],sha256=sha(r['path'])) for r in refs],results=[])
    out=ROOT/'artifacts/evaluation/recordings_precomputed.json'
    if out.exists():
        previous=json.loads(out.read_text());assert previous['manifest_sha256']==provenance['manifest_sha256'];assert previous['references']==provenance['references'];provenance=previous
    for song in [r for r in corpus.manifest['songs'] if r['split']=='test']:
        if any(r['song']==song['id'] for r in provenance['results']):continue
        d=corpus.load(song);y=d['y'][:len(d['f0'])*HOP];text=label_kana([p for a,b,p in read_lab(song['lab'])])
        row=metric.assess(y,d['f0'],active_wave=y,expected_text=text)
        row.update(song=song['id'],model='recording',condition='reproduction',semitones=0,rd=None,
            metrics_device='cpu',features_sha256=sha(ROOT/'data/features'/f"{song['id']}.npz"),f0_sha256=sha(ROOT/'data/f0'/f"{song['id']}.npz"))
        provenance['results'].append(row);save_json(out,provenance);print(song['id'],row['wavlm_cosine'],row['ecapa_cosine'],row['kana_cer'],flush=True)
if __name__=='__main__':main()
