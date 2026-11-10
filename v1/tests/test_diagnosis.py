"""Numerical checks for the diagnostic analysis (no shared outputs written)."""
import unittest
import numpy as np
import torch

from scripts.diagnose_bands import (
    BANDS, FREQ, SR, aggregate_interventions, band_weight,
    istft, physical_resonances, source_features, stft,
)
from diffart.physics import C, Synthesizer, lf_flow


class Diagnosis(unittest.TestCase):
    def test_stft_roundtrip_and_band_support(self):
        rng = np.random.default_rng(42)
        y = rng.normal(size=SR).astype(np.float32)
        reconstructed = istft(stft(y), len(y))
        self.assertLess(np.linalg.norm(y-reconstructed)/np.linalg.norm(y), 1e-6)
        for lo, hi in BANDS:
            w = band_weight(lo, hi)
            self.assertTrue(np.all(w[(FREQ < lo) | (FREQ >= hi)] == 0))
            self.assertTrue(np.all((w >= 0) & (w <= 1)))
            self.assertGreater(w.sum(), 10)
        self.assertTrue(np.all(band_weight(8000,12000)[FREQ <= 8000] == 0))

    def test_physical_resonance_frequency_and_width(self):
        model = Synthesizer(2)
        length, loss = float(model.speaker.length.detach()), float(model.speaker.wall.detach())
        p = {'areas':torch.ones(1,44), 'velum':torch.zeros(1)}
        result = physical_resonances(model,p,0)
        # Uniform lossy tube: |1/cosh((alpha+j*k)*L)| has quarter-wave peaks.
        expected_width = C/(np.pi*length)*np.arcsin(np.sinh(loss*length))
        for j,(freq,width) in enumerate(result):
            self.assertLess(abs(freq-(2*j+1)*C/(4*length)), 3)
            self.assertLess(abs(width-expected_width), 5)

    def test_known_periodic_source_has_small_cycle_variation(self):
        f0 = 200.
        phase = torch.arange(3200)*f0/16000
        flow = lf_flow(phase.remainder(1),torch.tensor(1.)).numpy()
        result,cycle = source_features(flow,f0)
        self.assertIsNotNone(cycle)
        self.assertGreater(result['cycles'], 30)
        self.assertLess(result['cycle_variation_ratio'], 1e-5)
        self.assertTrue(0 < result['open_fraction_10pct'] < 1)

    def test_speaker_aggregate_weights_songs_equally(self):
        rows = [{'song':'a','condition':'test','wavlm_anchor_delta':0.}]*3
        rows += [{'song':'b','condition':'test','wavlm_anchor_delta':1.}]
        result = aggregate_interventions(rows)[0]
        self.assertEqual(result['wavlm_anchor_delta'], .5)
        self.assertEqual(result['wavlm_anchor_delta_song_min'], 0.)
        self.assertEqual(result['wavlm_anchor_delta_song_max'], 1.)


if __name__ == '__main__':
    unittest.main()
