import json
import unittest
import numpy as np
from diffart.data import read_lab,frame_labels,ROOT,HOP,SR

class Data(unittest.TestCase):
    def test_lab_boundary_grid(self):
        vocab={'pau':0,'a':1,'k':2}
        context,pos=frame_labels([(0.,.02,'a'),(.02,.03,'k'),(.03,.06,'a')],8,vocab)
        self.assertEqual(context[:,1].tolist(),[1,1,2,1,1,1,0,0])
        self.assertAlmostEqual(float(pos[1]),.5)
        self.assertEqual(context[2].tolist(),[1,2,1])
    @unittest.skipUnless((ROOT/'data/manifest.json').is_file(), 'Requires a separately prepared private corpus manifest')
    def test_exclusion_and_song_split(self):
        m=json.loads((ROOT/'data/manifest.json').read_text());groups={}
        for r in m['songs']:
            if r['split']=='excluded':continue
            self.assertNotIn(r['group'],m['excluded_groups'])
            self.assertIn(groups.setdefault(r['group'],r['split']),[r['split']])
        self.assertAlmostEqual(sum(r.get('train_seconds',0) for r in m['songs']),3600)
        self.assertEqual(len([r for r in m['songs'] if r['split']=='test']),4)
if __name__=='__main__':unittest.main(verbosity=2)
