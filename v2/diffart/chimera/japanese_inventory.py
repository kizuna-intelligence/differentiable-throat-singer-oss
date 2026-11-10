"""Explicit mapping of the corpus's 38 labels to IPA articulatory gestures.

Targets: CHIMERA phonetics/inventory.rs, vowelTargets.generated.json and
consonantTargets.generated.json. MIT, Copyright (c) 2026 Jameson Hodge.
Japanese-specific approximations are stated in NOTES; no acoustic data fitting.
"""
import json
from pathlib import Path
import torch
from .articulators import NEUTRAL,NAMES

IPA = {'Edge':'ʔ','GlottalStop':'ʔ','I':'i̥','N':'ɴ','O':'o̥','U':'ɯ̥',
       'a':'a','b':'b','by':'bʲ','ch':'tɕ','cl':'Q','d':'d','e':'e','f':'ɸ',
       'g':'ɡ','gy':'ɡʲ','h':'h','hy':'ç','i':'i','j':'dʑ','k':'k','ky':'kʲ',
       'm':'m','my':'mʲ','n':'n','o':'o','p':'p','pau':'sil','r':'ɾ','ry':'ɾʲ',
       's':'s','sh':'ɕ','t':'t','ts':'ts','u':'ɯ','w':'ɰ','y':'j','z':'z'}
NOTES = {
    'u':'/ɯ/ unrounded: protrusion=0, width>=0.6',
    'N':'moraic nasal initially uvular; context adaptation is learned',
    'cl':'geminate inherits the following consonant closure in GesturalScore',
    'sh,ch,j':'alveolo-palatal approximated by postalveolar plus raised/front tongue body',
    'I,U,O':'unvoiced vowels preserve geometry and suppress oscillation',
    'Edge':'dataset edge approximation: glottal gesture, not an extra vowel',
}
VOWELS={'a','i','u','e','o','I','U','O'}
VOICELESS={'Edge','GlottalStop','I','U','O','ch','cl','f','h','hy','k','ky','p','pau','s','sh','t','ts'}


def targets(vocab):
    if set(vocab)!=set(IPA):
        raise ValueError(f'Expected the original 38 labels; missing={set(IPA)-set(vocab)}, extra={set(vocab)-set(IPA)}')
    data=json.loads(Path(__file__).with_name('ipa_targets.json').read_text())
    values=torch.tensor(NEUTRAL,dtype=torch.float64).repeat(len(vocab),1)
    mask=torch.zeros_like(values)
    def patch(phone,items):
        for key,value in items.items():
            ix=NAMES.index(key);values[vocab[phone],ix]=value;mask[vocab[phone],ix]=1
    for phone in vocab:
        if phone in VOWELS:
            ipa={'u':'ɯ','U':'ɯ','I':'i','O':'o'}.get(phone,phone)
            base=dict(data['vowels'].get(ipa,data['vowels']['u']))
            if phone in ('u','U'):base.update(lipProt=0.,lipWidth=max(.6,base.get('lipWidth',.6)))
            patch(phone,base);continue
        if phone in ('p','b','by','m','my'):patch(phone,{'lipAp':-.25})
        if phone in ('t','d','n','r','ry','ts'):patch(phone,{'ttX':6.25,'ttY':9.68})
        if phone in ('k','g','ky','gy','N'):patch(phone,{'tbX':2.3,'tbY':8.35,'jaw':.7})
        if phone in ('m','my','n','N'):patch(phone,{'velum':1.})
        if phone in ('s','z'):patch(phone,{'ttX':6.4,**data['consonants'][phone]})
        if phone in ('sh','ch','j'):patch(phone,{'ttX':5.75,'ttY':9.51,'tbX':3.4,'tbY':7.5})
        if phone=='f':patch(phone,{'lipAp':.032,'labiodental':0.})
        if phone=='hy':patch(phone,{'tbX':3.75,**data['consonants']['ç']})
        if phone in ('w','y'):patch(phone,{'tbX':3.6 if phone=='y' else 2.5,'tbY':7.7,'lipProt':0.})
        if phone.endswith('y'):patch(phone,{'tbX':3.6,'tbY':7.7})
    return values,mask
