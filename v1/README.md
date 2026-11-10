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

## ライセンス / License

**コードはMIT、学習済みの重みは波音リツ歌声DB規約に従います。** 重みを無条件のMITとして扱わないでください。規約は派生モデルにも適用されます（第9条3項）。「波音リツ」「カノン」の両名のクレジット（第7条4項）、他の声の基礎モデル等への利用制限と限定的例外（第7条2項）、該当時の特別表示（第7条3項）について [MODEL_TERMS.md](../MODEL_TERMS.md) と公式原文を確認してください。DBの音声・ラベル・MIDI・歌詞は含みません。

Code: [MIT](../LICENSE). Bundled weights: [database model conditions](../MODEL_TERMS.md).

## クレジット

歌声データベース: 波音リツ（歌唱・提供: カノン）

構成の参考: CHIMERA (https://github.com/jamesondh/chimera-vocal-synth, MIT)。
