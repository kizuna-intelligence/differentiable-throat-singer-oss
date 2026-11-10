# C: 軸対称声道音響の独立実験

コード・抽出形状・試運転重み・図・評価表はこのディレクトリ内に置く。主モデルの `best.pt` と `physics.py` は `extract_shapes.py` が形状抽出のためだけに CPU で読み、不変を SHA-256 で確認する。音声合成や元モデルの学習は行わない。

## 再現

リポジトリ直下から実行する。既存 `.venv` の numpy/scipy/torch/matplotlib だけで動く。

```bash
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
.venv/bin/python -m diffart.pinn.extract_shapes
.venv/bin/python -m diffart.pinn.run_uniform
.venv/bin/python -m unittest diffart.pinn.test_acoustics -v
.venv/bin/python -m diffart.pinn.train_parametric --device auto --steps 200 --points 64 --width 32 --out diffart/pinn/results/parametric_smoke
```

一様管は CPU 約 11 秒、PCA 抽出と MLP 試運転も CPU 数秒。実測は `results/` を参照。`--device auto` はローカル 3090 の UUID に限定し、既存の `/tmp/cyborgy-gpu-GPU-<UUID>.lock` と同じ排他 flock を確保してからメモリー使用量・使用率・計算プロセスを確認する。使用中・ロック中なら CPU。RTX PRO 4000・2 番 PC・17 番機は選択候補に含めない。

## 実装と検証の範囲

`uniform.py` は固定 Chebyshev 特徴と学習する複素線形出力の**スペクトル collocation 基準**。入力は `t=x/L, s=(rho/R)^2` で、学習はランダム点における Helmholtz 残差と入口・壁・出口の境界残差のみ。最小二乗で線形出力を学習する。解析的な圧力場、固有モードの分散式は学習に渡さない。周波数・半径ごとに出力を学び直すため、最終的な形状条件付き代理モデルとは区別する。

方程式は円筒座標の `p_xx + p_rhorho + p_rho/rho + k² p = 0`。`s` 座標では半径方向項が `4/R² * (s*p_ss+p_s)` となり、軸の極限も含む。断面積積分は `ds` に比例するため、内部の `s` は一様分布で抽出する。単位は m・m²・Hz、音速は固定 `350 m/s`。

時間規約は `exp(+i omega t)`。入口は `p_t=1` に正規化した体積速度源。物理単位では `p_x=-i*omega*rho_air*U/A` なので、圧力を `-i*omega*rho_air*U_in*L/A` で割った場に対応する。剛壁は `p_rho=0`。出口は `p + z/(i*k)*p_x=0`（`z=Z/(rho_air*c)`）。一様管では `H=U_out/U_in=1/(cos(kL)+i*z*sin(kL))`。

四分の一波長の共鳴位置は小さい実抵抗 `z=0.03` の簡易開口負荷で検証し、別に周波数依存の baffled-piston 放射インピーダンスでも同一境界の理論と比較する。**局所 Robin 近似の検証であり、外部放射場・モード間の放射結合の検証ではない。** 厳密な無損失・圧力開放の極では定常解が発散するので、小さい正抵抗で正則化している。0 Hz の圧力は `z=0, p(1)=0` の独立した静的ゲージ試験、伝達関数は `H(0)=1` の極限とし、非零負荷の静圧とは混同しない。

横モードは `R=4 cm` の人工的な管を使い、入口を `p_t=1+0.5*J0(alpha*sqrt(s))` とする（`alpha` は最初の `J1` の零点）。既知の入口分布を課し、管内圧力場を多項式特徴で学ぶ。学習後に別の変数分離解と比較し、場から軸方向波数の二乗を推定して遮断を確認する。断面一様の入口ではこのモードが出ない対照実験もある。

`field.py` / `train_parametric.py` は最終的な形状条件付き MLP の初期実装。入力は物理座標・周波数・対数面積 PCA 係数、出力は複素圧力の実部と虚部。半径二乗入力による軸の対称性、軸方向 Fourier 特徴、周波数 curriculum、物理座標の自動微分、傾斜壁の法線 `p_rho-r'(x)*p_x`、体積速度源、放射境界を含む。PCA は訓練曲だけから抽出し、係数を標準化して使う。形状は 44 区間の対数面積を cosine 16 項で平滑化するため、段差の厳密表現ではない。**200 step は動作試験であり未収束・未検証。任意形状への一般化も未検証。**

## 出力

- `results/shapes/area_pca.npz`, `summary.json`: 全音素の代表面積、5 母音の平均、PCA、フレームと入力ファイルの由来。これはモデルの逆推定解であって実測解剖ではない。
- `results/uniform/metrics.json`, `validation.csv`: 周波数ごとの独立点誤差、PDE/境界残差、共鳴、横モードの波数、次数増加による収束。
- `results/uniform/validation.png`, `transverse_fields.png`: 1 次元理論との比較・横モードの遮断前後の場。
- `results/uniform/*.npz`: 学習済み複素線形出力と伝達関数。
- `results/parametric_smoke/field.pt`, `training.json`: 未収束 MLP の試運転重みと残差。
- `results/tests.log`: 軸の 3 次元 Laplacian、傾斜壁法線、放射符号、体積速度正規化、独立点誤差・収束の 5 テスト。

次は段差管を独立差分／FEM 解と比較し、次いで平滑化した母音形状に進む。横モードが存在するというだけで B の補正量は決まらない。断面平均の伝達・非軸対称モード・形状による結合・放射近似も切り分ける。設計上の確認事項は [DESIGN_ISSUES_pinn.md](DESIGN_ISSUES_pinn.md)、先行研究は [survey/PINN_vocal_tract.md](../../survey/PINN_vocal_tract.md)。
