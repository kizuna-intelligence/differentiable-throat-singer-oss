# differentiable-throat-singer

微分できる喉のモデルで歌声を合成する。声門の音源と、44 区間の声道の管（＋鼻腔）を物理の式で組み、その形と長さを歌声の録音から勾配で推定する。

A singing synthesizer built from a differentiable physical throat model: a glottal source and a 44-section vocal-tract tube (plus a nasal branch), whose shape and length are estimated from singing recordings by gradient descent.

```
歌声 Y(f) = 声門の音源 G(f) · 声道の管 H(f; A)
```

## 中身

| パス | 内容 |
|---|---|
| `diffart/physics.py` | 声門の音源（LF）・44 区間の管・鼻腔・乱流の雑音。すべて微分できる |
| `diffart/improved.py` | 改良: 学習できる声門の波形、開閉に合わせた息の雑音、揺らぎ（ジッター・シマー）、梨状窩 |
| `diffart/pinn/` | 声道の軸対称の音響（ヘルムホルツ方程式）の有限要素法と PINN。1 次元の管とのずれを求める |
| `scripts/` | 学習・評価・診断・可視化 |
| `models/` | 学習済みの重み（下記） |

## 学習済みの重み

| ファイル | 内容 |
|---|---|
| `models/throat_ritsu_v1.pt` | 基本のモデル（`diffart.physics.Synthesizer`）。歌声データベースの歌 約 1 時間から推定。推定した声道の長さ 15.4 cm |
| `models/throat_ritsu_v1_learned_glottis.pt` | 声門の音源を学習できる形にした版（`diffart.improved.ImprovedSynthesizer(b1=False, b2=True)`） |
| `models/zeroshot_v1.pt` | OpenVoice v2 の参照埋め込みを使うゼロショットモデル（重みの条件は [MODEL_TERMS](../MODEL_TERMS.md#zero-shot-v1-weights)） |
| `models/vocab.json` | 音素の一覧 |

```python
import json, torch, numpy as np
from diffart.physics import Synthesizer
from diffart.render import render

vocab = json.load(open("models/vocab.json"))
model = Synthesizer(len(vocab))
model.load_state_dict(torch.load("models/throat_ritsu_v1.pt", weights_only=True)["model"])
model.eval()
# context: [T, 3] 前・今・次の音素の番号、pos: [T] 音素の中の位置 (0..1)、f0: [T] Hz。10 ms ごと
T = 200
a = vocab["a"]
y = render(model, np.full((T, 3), a), np.linspace(0, 1, T, dtype=np.float32), np.full(T, 262.0, np.float32))
# y: 24 kHz の波形
```

## Install and test

From **v1/**, use a separate Python 3.12/3.13 environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0
pip install -r requirements.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -v
```

The corpus split integration test skips when the separately obtained private corpus manifest is absent. Checkpoint regression tests use the bundled v1 model and authored phoneme inputs. Optional data preparation/evaluation requires separately obtained inputs: `RITSU_DB`, `RITSU_EXCLUSION_MANIFEST`, `RMVPE_ROOT`, `RMVPE_WEIGHTS`, `WAVLM_DIR`, `HEART_TRANSCRIPTOR_DIR`, `MORA_TABLE`, `COMPARISON_SCORE`, `COMPARISON_ROOT`. No evaluation models or dictionaries are bundled. PINN code remains in `diffart/pinn/`; larger data-dependent PINN experiments require their separately prepared inputs and are not the fast offline test suite.

## Zero-shot synthesis

### 日本語

参照音声から OpenVoice v2 の tone-colour 埋め込みを取り、声道の体格を推定して音素・長さ・F0 から歌声を合成します。参照音声は 10 秒程度が目安です。本人の声の特徴を完全に再現するものではありません。参照音声の音素・F0 に合ったスコアを渡すと、`--fit-steps N` で体の値を N 回合わせ込めます。参照音声ファイルは同梱しません。

OpenVoice のコードを取得し、公式の [OpenVoice v2 checkpoint archive](https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip) を OpenVoice のルートで展開してください。tone-colour converter のチェックポイントは `checkpoints_v2/converter/` に置きます。重みはこのリポジトリに含まれません。`OPENVOICE_ROOT` に OpenVoice のルートを設定するか、各コマンドに `--openvoice-root` を渡してください。

```bash
# Run from the repository root.
git clone https://github.com/myshell-ai/OpenVoice.git
cd OpenVoice
curl -L https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip -o checkpoints_v2.zip
unzip checkpoints_v2.zip
cd ../v1
export OPENVOICE_ROOT=../OpenVoice
python scripts/zeroshot_infer.py \
  --checkpoint models/zeroshot_v1.pt \
  --reference reference.wav \
  --score examples/zeroshot/score_example.json \
  --out outputs/zeroshot.wav \
  --save-controls outputs/zeroshot_controls.npz
```

合わせ込みを使う場合は、参照音声に時間合わせした音素・長さ・F0 を持つ JSON を `--reference-score` で指定します。

```bash
python scripts/zeroshot_infer.py \
  --checkpoint models/zeroshot_v1.pt \
  --reference reference.wav \
  --score examples/zeroshot/score_example.json \
  --reference-score reference_score.json \
  --fit-steps 100 \
  --out outputs/zeroshot_fitted.wav
```

`--save-controls` は `areas`, `rd`, `velum`, `amplitude`, `breath`, `turbulence`, `location`, `f0`, `pos`, `context`, `length_cm`, `nose_area` を含む NPZ を保存します。

`zeroshot_train.py` は汎用 JSONL を読みます。1 行に `wav`（相対または絶対パス）、`phonemes`、`durations` または `frames`、`f0`（Npy パスまたは数値配列）、`speaker`（文字列）を指定します。`durations` は音素ごとの秒数、`frames` は音素ごとの 10 ms フレーム数、`f0` は 10 ms ごとの値です。OpenVoice が無い環境では、OpenVoice v2 と同じ 256 次元の `embedding` 配列または Npy パスを指定できます。

```json
{"wav":"audio.wav","phonemes":["a","s","a"],"frames":[8,6,8],"f0":"pitch.npy","speaker":"voice_1"}
```

```bash
python scripts/zeroshot_train.py --jsonl examples.jsonl --steps 1000 --out models/custom_zero_shot.pt
```

### English

The model extracts an OpenVoice v2 tone-colour embedding from a reference voice, estimates vocal-tract body parameters, and synthesizes a score from phonemes, durations, and F0. About ten seconds of reference audio is a practical starting point. The output does not fully reproduce an individual voice. To adapt the body to aligned reference audio, provide an aligned score with `--reference-score` and set `--fit-steps N`. Reference audio is not included.

Get the OpenVoice code and extract the official [OpenVoice v2 checkpoint archive](https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip) at its root. The tone-colour converter checkpoint belongs in `checkpoints_v2/converter/`. The weights are not bundled here. Set `OPENVOICE_ROOT` to the OpenVoice root or pass `--openvoice-root` to the command.

`zeroshot_train.py` reads one JSON object per JSONL line. Required keys are `wav` (relative or absolute path), `phonemes`, `durations` or `frames`, `f0` (an Npy path or numeric array), and `speaker` (string). Durations are in seconds; frame counts use 10 ms units; F0 has one value per 10 ms frame. When OpenVoice is unavailable, supply an `embedding` array or Npy path containing a 256-dimensional OpenVoice v2 embedding.

## ライセンス / License

**コードはMIT、学習済みの重みは波音リツ歌声DB規約に従います。** 重みを無条件のMITとして扱わないでください。規約は派生モデルにも適用されます（第9条3項）。「波音リツ」「カノン」の両名のクレジット（第7条4項）、他の声の基礎モデル等への利用制限と限定的例外（第7条2項）、該当時の特別表示（第7条3項）について [MODEL_TERMS.md](../MODEL_TERMS.md) と公式原文を確認してください。DBの音声・ラベル・MIDI・歌詞は含みません。

Code: [MIT](../LICENSE). Bundled weights: [database model conditions](../MODEL_TERMS.md).

## クレジット

歌声データベース: 波音リツ（歌唱・提供: カノン）

構成の参考: CHIMERA (https://github.com/jamesondh/chimera-vocal-synth, MIT)。
