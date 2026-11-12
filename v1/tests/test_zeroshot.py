import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch

from diffart.zeroshot.data import read_score
from diffart.zeroshot.model import ZeroShotSynthesizer
from scripts.zeroshot_infer import synthesize


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models/zeroshot_v1.pt"
SCORE = ROOT / "examples/zeroshot/score_example.json"
torch.set_num_threads(2)


class ZeroShot(unittest.TestCase):
    def load_checkpoint(self):
        saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        self.assertEqual(set(saved), {"model", "config"})
        self.assertEqual(set(saved["config"]),
                         {"vocab", "embedding_dim", "encoder", "body_parameters"})
        config = saved["config"]
        self.assertEqual(config["embedding_dim"], 256)
        model = ZeroShotSynthesizer(len(config["vocab"]), config["embedding_dim"])
        model.load_state_dict(saved["model"])
        return model.eval(), config

    def test_bundled_weights_load(self):
        model, config = self.load_checkpoint()
        self.assertEqual(config["encoder"], "openvoice")
        self.assertTrue(model.body_dct.shape[0] == 8)

    def test_authored_score_synthesizes_audio_from_embedding(self):
        model, config = self.load_checkpoint()
        context, pos, f0 = read_score(SCORE, config["vocab"])
        torch.manual_seed(19)
        embedding = torch.randn(256)
        waveform, controls = synthesize(model, context, pos, f0, embedding,
                                        seed=71, return_controls=True)
        self.assertEqual(waveform.numel(), len(f0) * 240)
        self.assertTrue(torch.isfinite(waveform).all())
        self.assertGreater(float(waveform.abs().max()), 1e-7)
        self.assertAlmostEqual(waveform.numel() / 24000, 3.25, places=2)
        self.assertEqual(tuple(controls["areas"].shape), (len(f0), 44))

    def test_jsonl_training_runs_two_cpu_steps_without_encoder(self):
        _, config = self.load_checkpoint()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            sample_rate = 24000
            samples = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
            waveform = 0.15 * np.sin(2 * np.pi * 220 * samples)
            sf.write(folder / "tone.wav", waveform, sample_rate)
            rng = np.random.default_rng(23)
            row = {
                "wav": "tone.wav",
                "phonemes": ["a"],
                "frames": [200],
                "f0": [220.0] * 200,
                "speaker": "voice_a",
                "embedding": rng.normal(size=256).astype(float).tolist(),
            }
            jsonl_path = folder / "examples.jsonl"
            jsonl_path.write_text(json.dumps(row) + "\n")
            output_path = folder / "trained.pt"
            environment = os.environ.copy()
            environment["OMP_NUM_THREADS"] = "2"
            environment["OPENBLAS_NUM_THREADS"] = "2"
            environment["MKL_NUM_THREADS"] = "2"
            subprocess.run(
                [sys.executable, "scripts/zeroshot_train.py", "--jsonl", str(jsonl_path),
                 "--steps", "2", "--batch-size", "1", "--segment-frames", "8",
                 "--device", "cpu", "--out", str(output_path)],
                cwd=ROOT, env=environment, check=True, capture_output=True, text=True,
                timeout=180,
            )
            trained = torch.load(output_path, map_location="cpu", weights_only=True)
            self.assertEqual(set(trained), {"model", "config"})
            self.assertEqual(set(trained["config"]),
                             {"vocab", "embedding_dim", "encoder", "body_parameters"})
            self.assertEqual(trained["config"]["vocab"], config["vocab"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
