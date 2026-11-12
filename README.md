# Differentiable throat singer

A singing synthesizer built around differentiable physical models of the vocal tract. It includes two editions with different acoustic models and workflows.

| Edition | Model | Contents | Guide |
|---|---|---|---|
| v1 | Frequency-domain 44-section tract, LF/learned glottis, nasal branch | Synthesis, PINN/FEM code, weights, scripts and tests | [v1 guide](v1/README.md) |
| v1 zero-shot | Reference-conditioned 44-section throat model | Singing synthesis, optional body fitting, controls export and JSONL training | [v1 guide](v1/README.md#zero-shot-synthesis) |
| v2 | CHIMERA-derived branching 96 kHz FDTD tract | Synthesis and weights, singing/speech training, songkit, multi-throat visualization | [v2 guide](v2/README.md) |

Both editions use the Python import name `diffart`. Install and run them from separate directories and Python environments.

## Testing

### v1

From `v1/`, install the CPU requirements and run the offline test suite:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0
pip install -r requirements.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -v
```

The current suite result is **19 passed, 1 skipped**. The skipped corpus-split integration test requires a separately obtained manifest. The remaining offline tests use the included model and authored phoneme inputs.

### v2

From `v2/`, install into a separate CPU environment and run the tests:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0
pip install -r requirements-training.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -v
```

The current CPU suite result is **11 passed** (about three minutes on the verification machine, including first-use compilation). The tests cover physical behavior, acoustic adjoint finite differences, singing/speech data preparation, four real optimizer steps, held-out spectral evaluation, song specifications and a short synthesis. The training backend compiles a Numba adjoint on its first use. The included 3/4 trial can be synthesized on CPU with:

```bash
python scripts/make_throat_song.py examples/ievan_allvoice/trial_3_4.json --out outputs/trial --audio-only
```

For a rendered video, install `ffmpeg`, Node.js and a Chromium-compatible browser, then build the visualizer and run the same command without `--audio-only`:

```bash
(cd visualizer && npm ci --ignore-scripts && npm run build)
python scripts/make_throat_song.py examples/ievan_allvoice/trial_3_4.json --out outputs/trial
```

The visualizer build and full 3/4 video render have both completed successfully. GPU training was not exercised by these CPU checks. The video carries the CHIMERA attribution at bottom right. See the [v2 guide](v2/README.md) for model details and the full song example.

## License and model terms

Code is [MIT](LICENSE). The v1 and v2 weights follow the [model terms](MODEL_TERMS.md) for the **Namine Ritsu singing database (波音リツ / カノン)** and are not unrestricted MIT assets. The database requires both names to be credited and sets limits on using the weights as a base for other voices and on derivative models. The model terms describe the applicable clauses and link to the official conditions.

CHIMERA-derived visualization source retains its original notices. See [third-party notices](THIRD_PARTY_NOTICES.md), the [renderer provenance](v2/visualizer/vendor/chimera/PROVENANCE.md), and the [model terms](MODEL_TERMS.md).

No database recordings, labels, MIDI, database-derived lyrics, reference recordings, or generated audio/video are included.

## 日本語の要約

微分可能な物理声道モデルを使う歌声合成ソフトウェアです。v1には周波数領域の44区間モデルとPINN/FEM、v2にはCHIMERA由来の分岐FDTDモデルによる合成、歌声・話し声からの学習、曲の合成、複数声道の可視化が含まれます。v1とv2は別々のPython環境で利用してください。試験手順と実行結果は上記の「Testing」を参照してください。

コードのライセンスはMITです。重みには波音リツ歌声データベースの利用条件が適用され、「波音リツ」「カノン」の両名のクレジットや他の声への転用に関する条件があります。詳しくは [MODEL_TERMS.md](MODEL_TERMS.md) を確認してください。データベース由来の音声、ラベル、MIDI、歌詞、参照音声および生成物は配布物に含まれません。

## Singing and speech training

The [v2 training guide](v2/README.md#training-from-singing-or-speech) covers user-obtained audio, timed singing phones, speech forced alignment, F0, the main_round2 spectral/physical objective, and held-out evaluation. Other voices start from independent physical and neural defaults. The Namine Ritsu weights must not be used as another voice's initialization.

