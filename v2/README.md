# v2 — main_round2 CHIMERA throat model

The CHIMERA branching FDTD vocal tract (internal 96 kHz), parametric glottis, Borda–Carnot losses, walls, fixed turbulence, sagittal anatomy-to-area conversion and second-order gesture motion were ported to PyTorch/Triton with a custom backward pass and **trained by gradient descent**. Junction inertance was calibrated against an independent axisymmetric FEM; this is not a general 3D solver or a claim of true anatomical identification.

The checkpoint is **main_round2** only, originally initialized from the selected Round 1 model. It uses spectral loss and physical regularization, seed 20261109, batch 16, lr 0.0005, up to 30 epochs / patience 3 (best epoch 29). It uses learned source shape, walls, conservative geometry, four tongue modes and existing nasal geometry controls. The saved optional extra objectives/controls were not used and have been removed. Nasal inertance correction is 0; oral coefficient is 1.387144840893109.

This edition includes inference and singing/speech training using the **main_round2 method**: multi-resolution log-STFT, spectral convergence, mel loss, geometry smoothness and wall regularization. The existing model tensors are unchanged and training state remains stripped. See [source and distribution checksums](models/README.md). The portable trainer provides the same objective and optimizer; it does not claim bitwise reproduction of the historical GPU run.

## Historical evaluation

| Model | WavLM cosine ↑ | ECAPA cosine ↑ | Kana CER ↓ | 4–6 kHz MAE dB ↓ |
|---|---:|---:|---:|---:|
| v1 baseline | 0.693487 | 0.304517 | 0.311554 | 4.579039 |
| Round 1 | 0.711207 | 0.297481 | 0.458384 | 5.550617 |
| main_round2 | 0.719123 | 0.301142 | 0.362180 | 5.027645 |

Historical saved four-song reproduction results, not a new evaluation. **Stage 3 acceptance criteria are not met**: speaker, formants, high band and kana criteria all fail. WavLM improvement is not overall acceptance. The optional songkit autocorrelation pitch sanity check is not the historical RMVPE/speaker/CER evaluation pipeline.
## CPU install and test

Use Python 3.12 or 3.13. From **v2/**, in an independent environment (v1 and v2 both use the import name `diffart`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0
pip install -r requirements-training.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -v
```

The current CPU suite result is **11 passed**, including four real singing/speech optimizer steps and a finite-difference acoustic-adjoint check. GPU training is not covered by this CPU verification.

The first real CPU acoustic test compiles a Numba kernel; it does not download a database, dictionary or evaluation model. Caches use a temporary directory or `CHIMERA_CACHE`, never the protected experiment directory.

Install `ffmpeg` and run the independent 3/4 example (12 seconds, three physical throats):

```bash
python scripts/make_throat_song.py examples/ievan_allvoice/trial_3_4.json --out outputs/trial --audio-only
```

Full mix and visualization (install Node.js, npm, Chromium/Google Chrome and ffmpeg separately):

```bash
(cd visualizer && npm ci --ignore-scripts && npm run build)
python scripts/make_throat_song.py examples/ievan_allvoice/trial_3_4.json --out outputs/trial
```

`CHROME_BIN` overrides Chromium executable discovery. The default video compositor includes the bottom-right text **Visualization: CHIMERA vocal synth © 2026 Jameson Hodge (MIT)**. The traditional-melody all-voice example is `examples/ievan_allvoice/song.json`; see `examples/RIGHTS.md`. The kana frontend consists of small self-authored hiragana rules, not an external dictionary. Inputs must use supported kana spellings. Change tempo, meter, key, notes and accompaniment patterns in the JSON specification. Each part has its own throat and optional physical scale/pan/gain.

## Credits and conditions

Thanks to [Jameson Hodge's CHIMERA vocal synth](https://github.com/jamesondh/chimera-vocal-synth) (MIT). Copied renderer files are unmodified; the playback adapter and TractMap extraction are ours. See `visualizer/vendor/chimera/PROVENANCE.md`, its LICENSE, and [third-party notices](../THIRD_PARTY_NOTICES.md), including Archivo OFL 1.1 and AVHershey WTFPL.

歌声データベース: **波音リツ（歌唱・提供: カノン）**. Code is MIT; weights follow [MODEL_TERMS.md](../MODEL_TERMS.md), including both-name credits, Article 7(2)'s restrictions/exceptions and Article 9(3). Database recordings, labels, MIDI, lyrics and reference audio are not shipped.

## 日本語の要約

CHIMERAの分岐FDTDを微分可能に移植し、波音リツ（カノン）の歌声から勾配降下で学習したmain_round2までの版です。歌声・話し声の準備、強制アライメント、F0、main_round2の損失・学習ループ・評価を含みます。他の声は独立した初期値から学習してください。段階3の品質条件は未達です。v1と別の環境で試験してください。重みはMITではなくDB規約に従い、可視化には著作権を常時表示します。

## Training from singing or speech

Install the additional preparation/training dependencies in the v2 environment:

```bash
pip install -r requirements-training.txt
python -m unittest discover -s tests -v
```

The acoustic backward path uses the same generated exact adjoint and block replay as main_round2. CPU uses Numba, while `--device cuda` uses Triton on CUDA or ROCm (install the matching PyTorch/Triton separately). CPU is useful for short checks; full one-second blocks and corpus training are much faster on a GPU. The CPU adjoint compiles on first use. The spectral evaluator below does not download speaker/ASR models or reproduce the historical WavLM/ECAPA/CER table.

Obtain a licensed corpus yourself. For **singing**, obtain the [Namine Ritsu singing database](https://www.canon-voice.com/voicebanks/) and read its bundled conditions, or use your own recordings. Provide the audio and phoneme intervals in seconds (or an HTK `.lab` file with 100 ns timestamps), plus measured F0. For **speech**, use a speech corpus you are licensed to use for training and for releasing models (or your own recordings), read its terms, and supply audio with its transcript or reviewed reading. Do not include the corpus in a redistribution of this repository.

The input is a user-created JSON array, e.g. this schema (filenames are placeholders; text must match your audio):

```json
[
  {"kind":"singing", "split":"train", "audio":"train.wav",
   "phones":[[0.0,1.0,"a"]], "f0":"train-f0.npy"},
  {"kind":"speech", "split":"train", "audio":"speech-train.wav", "text":"あ"},
  {"kind":"speech", "split":"val", "audio":"speech-val.wav", "text":"あ"},
  {"kind":"speech", "split":"test", "audio":"speech-test.wav", "text":"あ"}
]
```

Paths resolve relative to that JSON. Use the full recording's intervals; gaps become pauses. Instead of `phones`, use `"lab":"recording.lab"`. Supply `reading` to override a written transcript when needed. `f0` is an optional numeric `.npy` array with one Hz value per 10 ms frame, 0 for unvoiced frames. For the experimental preparation path, extract RMVPE F0 separately using your chosen licensed implementation and weights, with 24 kHz audio, hop 240, confidence threshold .03, and save that array. This repository does not bundle RMVPE code or weights. When `f0` is absent, preparation uses librosa pYIN as a portable alternative; this is not the experiment's RMVPE extraction. Review alignment plots/timings and pitch before training.

For automatic **speech alignment**, install `pip install -r requirements-speech.txt`. Obtain Narabas ONNX weights and its matching symbol vocabulary separately from the [official model release](https://github.com/darashi/narabas-models/releases/tag/v0) and [Narabas source](https://github.com/darashi/narabas/blob/b5e6511ce24dd2f14fc633ed5a23214225ca3eb7/narabas/symbols.py). Check their terms. Save `narabas-v0.onnx` and `symbols.py` in a local `tools/` directory. Only literal symbol data are read from `symbols.py`; it is not executed. PyOpenJTalk supplies Japanese phones; its first use may download its dictionary to the user cache. The preparation code performs the experiment's monotonic CTC/Viterbi alignment with 300 ms padding and the same phone normalization. External dictionaries, aligner weights, and corpus text are not bundled.

```bash
python scripts/prepare_voice.py recordings.json --out data/prepared \
  --aligner tools/narabas-v0.onnx --aligner-symbols tools/symbols.py
python scripts/train_voice.py data/prepared --out outputs/voice --device cuda
python scripts/evaluate_voice.py data/prepared outputs/voice/best.pt --split test --device cuda
```

With reviewed `phones`/`lab` for every recording, omit the aligner arguments and speech dependencies. Singing and speech can be used separately or together if they belong to the **same licensed voice**. The trainer starts from CHIMERA physical defaults and independently seeded neural controls, with a TRAIN-only global recording gain calibration. It does not read any bundled voice model or its calibrated configuration. Never initialize another voice from the Namine Ritsu weights; `--init` is only for further training of the same licensed voice. Keep train/validation/test recordings disjoint. Validation selects the best epoch; test blocks are used only when explicitly evaluated.

Defaults: seed 20261109, Adam lr .0005, batch 16, 100-frame (one-second) blocks, 30 epochs, patience 3, global gradient norm capped at 5. For the speech-only schedule use `--epochs 60 --patience 5`. `--device cpu --frames 2 --batch 1 --epochs 2` provides a tiny real training run on suitably prepared audio; it is a smoke check, not a useful voice. Checkpoints use `torch.load(weights_only=True)`. Numeric feature archives reject object arrays and paths outside the prepared directory. Use an empty output directory to prevent accidental overwrite; this portable trainer does not resume optimizer/RNG state automatically. Generated checkpoints must retain the terms of their own source corpus; the generic training code remains MIT.
