"""Independent recognizer evidence for the overfit phrase, never a substitute for audition."""
import json
import argparse
import torch
from diffart.data import ROOT
from diffart.metrics import audio16,Transcriptor,label_kana,kana,cer
from diffart.runtime import save_json

ap=argparse.ArgumentParser();ap.add_argument('--threads',type=int,default=3);args=ap.parse_args()
torch.set_num_threads(args.threads)
phrase=json.loads((ROOT/'artifacts/overfit/phrase.json').read_text())
ref=label_kana(phrase['phones']);model=Transcriptor()
result={'expected_kana':ref,'metric':'HeartTranscriptor kana CER; human audition still required','files':{}}
for name in ['reference','best']:
    path=ROOT/'artifacts/overfit'/f'{name}.wav'
    text=model.transcript(audio16(path));hyp=kana(text)
    result['files'][name]={'transcript':text,'kana':hyp,'cer':cer(ref,hyp)}
    save_json(ROOT/'artifacts/overfit/intelligibility.json',result)
    print(name,result['files'][name],flush=True)
