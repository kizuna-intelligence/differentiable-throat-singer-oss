"""Throat + sound visualizer: the learned vocal tract (area function on a bent tube, nasal branch, glottal source,
noise sources) animated in sync with the model's own synthesized audio, with a scrolling spectrogram and F0.
Usage: PYTHONPATH=. .venv/bin/python scripts/visualize.py <song_id> <start_seconds> <seconds> <out.mp4>
Audio in the video is the model output only (no database recording is redistributed).
"""
import json, math, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from PIL import Image, ImageDraw, ImageFont
from diffart.data import ROOT, SR, HOP
from diffart.physics import Synthesizer
from diffart.render import render

SONG, START, SECONDS, OUT = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), Path(sys.argv[4])
W, H, FPS = 1920, 1080, 30
man = json.loads((ROOT / 'data/manifest.json').read_text())
inv = {v: k for k, v in man['vocab'].items()}
model = Synthesizer(len(man['vocab']))
ck = torch.load(ROOT / 'artifacts/main/best.pt', map_location='cpu', weights_only=True)
model.load_state_dict(ck['model']); model.eval()

feat = np.load(ROOT / f'data/features/{SONG}.npz'); f0all = np.load(ROOT / f'data/f0/{SONG}.npz')['f0']
a = int(START * SR / HOP); n = int(SECONDS * SR / HOP)
ctx, pos, f0 = feat['context'][a:a + n], feat['pos'][a:a + n], f0all[a:a + n].astype(np.float32)
audio = render(model, ctx, pos, f0)
with torch.inference_mode():
    p = model.controls(torch.as_tensor(ctx)[None], torch.as_tensor(pos)[None], torch.as_tensor(f0)[None], model.speaker)
    areas = model.speaker.area(p['art'])[0].numpy()                     # [T, 44] cm²
    C = {k: v[0].numpy() for k, v in p.items() if k != 'art'}
    L_cm = float(model.speaker.length) * 100
    nose = model.speaker.nose.numpy()
peak = np.abs(audio).max(); audio = audio / max(peak, 1e-6) * 0.89

# ---- spectrogram (log-magnitude, 0-8 kHz) for the scrolling panel
nfft, hop = 1024, HOP
win = np.hanning(nfft)
pad = np.pad(audio, (nfft // 2, nfft // 2))
frames = np.stack([pad[i * hop:i * hop + nfft] * win for i in range(len(audio) // hop)])
S = 20 * np.log10(np.abs(np.fft.rfft(frames, axis=1)) + 1e-6)
fmax_bin = int(8000 / (SR / nfft)); S = S[:, :fmax_bin]
S = np.clip((S - (S.max() - 75)) / 75, 0, 1)                          # [T, F] 0..1

BG0, BG1 = (9, 12, 22), (20, 27, 46)
WHITE, DIM, FAINT, ACCENT, WARM = (246, 247, 250), (110, 120, 145), (58, 66, 88), (126, 205, 255), (255, 176, 120)
TISSUE, TISSUE_EDGE = (214, 120, 132), (255, 190, 196)
JP = '/usr/share/fonts/opentype/noto/NotoSansCJK-{}.ttc'
def font(w, s): return ImageFont.truetype(JP.format(w), s, index=0)
F_T, F_S, F_PH, F_PH2, F_TINY = font('Bold', 40), font('Regular', 26), font('Bold', 96), font('Medium', 40), font('Regular', 22)

yy, xx = np.mgrid[0:H, 0:W]
r = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H * 0.46) / (H * 0.70)) ** 2).clip(0, 1)
bg = np.zeros((H, W, 3), np.float32)
for c in range(3): bg[..., c] = BG1[c] * (1 - r) + BG0[c] * r
BG = Image.fromarray(bg.astype(np.uint8))

# ---- tube geometry: glottis at the bottom, pharynx up, a 90° bend, oral cavity forward to the lips
N = 44; LEN_PX = 760; SCALE = LEN_PX / L_cm                          # px per cm
GX, GY = 330, 860
def centerline(n):
    s = (np.arange(n + 1) / n) * LEN_PX
    ph, bend_r = 0.44 * LEN_PX, 120
    bend = math.pi / 2 * bend_r
    pts, ang = [], []
    for d in s:
        if d <= ph: pts.append((GX, GY - d)); ang.append(-math.pi / 2)
        elif d <= ph + bend:
            t = (d - ph) / bend_r; cx, cy = GX + bend_r, GY - ph
            pts.append((cx - bend_r * math.cos(t), cy - bend_r * math.sin(t))); ang.append(-math.pi / 2 + t)
        else:
            e = d - ph - bend; pts.append((GX + bend_r + e, GY - ph - bend_r)); ang.append(0.0)
    return np.array(pts), np.array(ang)
CL, ANG = centerline(N)
NORM = np.stack([np.cos(ANG - math.pi / 2), np.sin(ANG - math.pi / 2)], 1) * -1   # left-hand normal

def lerp(arr, t):
    i = min(max(t * SR / HOP, 0), len(arr) - 1); i0 = int(i); i1 = min(i0 + 1, len(arr) - 1); w = i - i0
    return arr[i0] * (1 - w) + arr[i1] * w

def ease(x): x = min(max(x, 0.0), 1.0); return x * x * (3 - 2 * x)
def mix(c1, c2, a): return tuple(int(round(c1[i] * (1 - a) + c2[i] * a)) for i in range(3))

rng = np.random.default_rng(0)
def frame(t, a_all):
    img = BG.copy(); d = ImageDraw.Draw(img)
    st = t
    A = lerp(areas, st); vel = lerp(C['velum'], st); rd = lerp(C['rd'], st); amp = lerp(C['amplitude'], st)
    br = lerp(C['breath'], st); tur = lerp(C['turbulence'], st); loc = lerp(C['location'], st); fz = lerp(f0, st)
    amp_n = min(amp / (np.percentile(C['amplitude'], 95) + 1e-6), 1.2)
    # radius from area (cm² -> cm), then px; sections are centered between centerline nodes
    rad = np.sqrt(np.clip(A, 0.05, 8) / math.pi) * SCALE * 1.15
    rad_nodes = np.concatenate([[rad[0]], (rad[:-1] + rad[1:]) / 2, [rad[-1]]])
    left = CL + NORM * rad_nodes[:, None]; right = CL - NORM * rad_nodes[:, None]
    # nasal branch from the velum region (section 13), opening with the velum
    k = 13; base = CL[k + 6]; nd = np.array([0.55, -0.83]); nd /= np.linalg.norm(nd)
    nrad = np.sqrt(np.clip(nose, 0.05, 8) / math.pi) * SCALE * 1.0
    npts = [base + nd * (i / (len(nose) - 1)) * 300 for i in range(len(nose))]
    nnorm = np.array([-nd[1], nd[0]])
    a_n = 0.25 + 0.75 * ease(vel * 3)
    nl = [tuple(pnt + nnorm * rr) for pnt, rr in zip(npts, nrad)]; nr = [tuple(pnt - nnorm * rr) for pnt, rr in zip(npts, nrad)]
    d.polygon(nl + nr[::-1], fill=mix(BG1, (120, 90, 140), a_all * a_n * 0.8), outline=mix(BG1, (190, 160, 220), a_all * a_n))
    d.polygon([tuple(x) for x in left] + [tuple(x) for x in right[::-1]], fill=mix(BG1, TISSUE, a_all * 0.85), outline=mix(BG1, TISSUE_EDGE, a_all))
    # air channel (the lumen) as a thinner bright core, wider where the tube is open
    core = [tuple(x) for x in (CL + NORM * (rad_nodes * 0.35)[:, None])] + [tuple(x) for x in (CL - NORM * (rad_nodes * 0.35)[:, None])[::-1]]
    d.polygon(core, fill=mix(BG1, (255, 225, 228), a_all * 0.35))
    # glottal source: pulsing ellipse, color by Rd (pressed = warm, breathy = cool)
    phase = (t * max(fz, 1)) % 1.0
    open_ = (0.5 + 0.5 * math.sin(2 * math.pi * phase)) * amp_n if fz > 0 else 0.0
    col = mix(WARM, ACCENT, min(max((rd - 0.5) / 2.0, 0), 1))
    gw = 26 + 30 * amp_n; gh = 4 + 16 * open_
    d.ellipse([GX - gw, GY + 18 - gh, GX + gw, GY + 18 + gh], fill=mix(BG1, col, a_all), outline=mix(BG1, WHITE, a_all * 0.6))
    # noise particles: breath at the glottis, turbulence at the constriction location
    for _ in range(int(60 * br * amp_n)):
        x = GX + rng.normal(0, 14); y = GY - rng.uniform(0, 80)
        d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=mix(BG1, ACCENT, a_all * 0.8))
    j = int(round(loc * (N - 1))); cx, cy = CL[j];
    for _ in range(int(140 * tur * amp_n)):
        x = cx + rng.normal(0, 10) + (rng.uniform(0, 60) if ANG[j] == 0 else 0); y = cy + rng.normal(0, 10)
        d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=mix(BG1, WHITE, a_all * 0.9))
    d.text((GX - 40, GY + 62), '声門', font=F_S, fill=mix(BG1, DIM, a_all), anchor='ms')
    d.text((CL[-1][0] + 40, CL[-1][1] + 8), '唇', font=F_S, fill=mix(BG1, DIM, a_all), anchor='ls')
    d.text((npts[-1][0] + 20, npts[-1][1]), '鼻腔', font=F_S, fill=mix(BG1, DIM, a_all), anchor='ls')
    # ---- right: scrolling spectrogram and F0 (4 s window, playhead at 1/3)
    X0, X1, Y0, Y1 = 1060, 1840, 200, 600
    span = 4.0; t0 = t - span / 3
    fi0 = int(t0 * SR / HOP); fi1 = fi0 + int(span * SR / HOP)
    idx = np.clip(np.arange(fi0, fi1), 0, len(S) - 1); valid = (np.arange(fi0, fi1) >= 0) & (np.arange(fi0, fi1) < len(S))
    patch = S[idx] * valid[:, None]
    img_s = (patch.T[::-1] * 255).astype(np.uint8)
    col_map = np.stack([img_s * 0.45 + 12, img_s * 0.75 + 16, img_s * 1.0 + 30], -1).clip(0, 255).astype(np.uint8)
    sp = Image.fromarray(col_map).resize((X1 - X0, Y1 - Y0), Image.BILINEAR)
    img.paste(Image.blend(Image.new('RGB', sp.size, BG1), sp, a_all), (X0, Y0))
    px = X0 + (X1 - X0) / 3
    d.line([(px, Y0 - 6), (px, 860)], fill=mix(BG1, WHITE, a_all * 0.6), width=2)
    d.text((X0, Y0 - 16), 'スペクトログラム（0〜8 kHz）', font=F_TINY, fill=mix(BG1, DIM, a_all), anchor='ls')
    # F0 panel (MIDI 55..84)
    FY0, FY1 = 660, 860; lo, hi = 55, 84
    d.rectangle([X0, FY0, X1, FY1], outline=mix(BG1, FAINT, a_all))
    pts = []
    for ii in range(fi0, fi1, 2):
        if 0 <= ii < len(f0) and f0[ii] > 0:
            m = 69 + 12 * math.log2(f0[ii] / 440); y = FY1 - (m - lo) / (hi - lo) * (FY1 - FY0)
            pts.append((X0 + (ii - fi0) / (fi1 - fi0) * (X1 - X0), y))
        else:
            if len(pts) > 1: d.line(pts, fill=mix(BG1, WARM, a_all), width=3)
            pts = []
    if len(pts) > 1: d.line(pts, fill=mix(BG1, WARM, a_all), width=3)
    d.text((X0, FY0 - 12), '声の高さ（F0）', font=F_TINY, fill=mix(BG1, DIM, a_all), anchor='ls')
    # ---- current phoneme
    i = min(int(t * SR / HOP), len(ctx) - 1); prev_, cur, nxt = (inv.get(int(v), '') for v in ctx[i])
    show = lambda s: '' if s in ('pau', 'sil', 'Edge') else s
    d.text((560, 1000), show(cur), font=F_PH, fill=mix(BG1, WHITE, a_all), anchor='ms')
    d.text((400, 990), show(prev_), font=F_PH2, fill=mix(BG1, FAINT, a_all), anchor='ms')
    d.text((720, 990), show(nxt), font=F_PH2, fill=mix(BG1, FAINT, a_all), anchor='ms')
    # ---- header and legend
    d.text((72, 92), '微分できる喉のモデル：声道の形と、そこから出る音', font=F_T, fill=mix(BG1, WHITE, a_all), anchor='ls')
    d.text((72, 134), f'推定した声道の長さ {L_cm:.1f} cm ｜ 44 区間の管 ＋ 鼻腔 ＋ 声門の音源（物理モデルの出力そのまま）', font=F_S, fill=mix(BG1, DIM, a_all), anchor='ls')
    d.text((1060, 920), f'鼻腔への開き {vel:.2f}　｜　声質 Rd {rd:.2f}（小さいほど張った声）', font=F_S, fill=mix(BG1, DIM, a_all), anchor='ls')
    d.text((W - 72, H - 40), '声：「波音リツ」歌声データベース（カノン）', font=F_TINY, fill=mix(BG1, FAINT, a_all), anchor='rs')
    return img

# ---- intro: one phrase of the original recording (for a commentary video), then a transition card
ORIG_A, ORIG_B = START + 1.99, START + 5.85                      # one phrase of the same song (pause-bounded)
orig_all, _ = sf.read(ROOT / f'data/audio24/{SONG}.wav', dtype='float32')
orig = orig_all[int(ORIG_A * SR):int(ORIG_B * SR)]; orig = orig / max(np.abs(orig).max(), 1e-6) * 0.89
ORIG_LEN = len(orig) / SR
NB = 110
oenv = np.array([np.sqrt(np.mean(c ** 2)) for c in np.array_split(orig, NB)]); oenv = (oenv / oenv.max()) ** 0.7
T_O0 = 0.8; T_O1 = T_O0 + ORIG_LEN; T_CARD = T_O1 + 0.9; CARD = 3.2; T_VIZ = T_CARD + CARD

def intro_frame(t):
    img = BG.copy(); d = ImageDraw.Draw(img)
    if t < T_CARD:
        a = ease(t / 0.5) * (1 - ease((t - (T_O1 + 0.35)) / 0.5))
        prog = min(max((t - T_O0) / ORIG_LEN, 0), 1)
        d.text((W / 2, 360), '元の声', font=F_S, fill=mix(BG1, ACCENT, a), anchor='ms')
        d.text((W / 2, 440), '歌声データベースの録音（1 フレーズ）', font=F_T, fill=mix(BG1, WHITE, a), anchor='ms')
        x0 = (W - NB * 10 + 3) / 2
        for i, e in enumerate(oenv):
            h = max(4, e * 150); x = x0 + i * 10
            d.rounded_rectangle([x, 640 - h / 2, x + 7, 640 + h / 2], radius=3, fill=mix(BG1, ACCENT if (i + 0.5) / NB <= prog else FAINT, a))
        d.text((W / 2, 800), 'この人の歌を約 1 時間分使って、喉のモデルのパラメーターを推定しました', font=F_S, fill=mix(BG1, DIM, a), anchor='ms')
    else:
        c = t - T_CARD; a = ease(c / 0.5) * (1 - ease((c - (CARD - 0.5)) / 0.45))
        d.text((W / 2, 500), '推定した喉のモデルが、', font=F_T, fill=mix(BG1, WHITE, a), anchor='ms')
        d.text((W / 2, 570), '学習に使っていない曲を歌うと', font=F_T, fill=mix(BG1, WHITE, a), anchor='ms')
        d.text((W / 2, 650), '（ここから先の音は、すべてモデルの出力です）', font=F_S, fill=mix(BG1, DIM, a), anchor='ms')
    d.text((W - 72, H - 40), '声：「波音リツ」歌声データベース（カノン）', font=F_TINY, fill=mix(BG1, FAINT, 1.0), anchor='rs')
    return img

def main():
    lead, tail = T_VIZ + 0.3, 0.8
    total = lead + SECONDS + tail
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / 'a.wav'
        full = np.concatenate([np.zeros(int(T_O0 * SR)), orig, np.zeros(int((lead - T_O0) * SR) - len(orig)),
                               audio, np.zeros(int(tail * SR))]).astype(np.float32)
        sf.write(wav, full, SR)
        enc = subprocess.Popen(['nice', '-n', '19', 'ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                                '-r', str(FPS), '-i', '-', '-i', str(wav), '-c:v', 'libx264', '-preset', 'slow', '-crf', '17',
                                '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '256k', '-movflags', '+faststart', '-shortest', str(OUT)],
                               stdin=subprocess.PIPE)
        for k in range(int(math.ceil(total * FPS))):
            tt = k / FPS
            if tt < T_VIZ:
                enc.stdin.write(intro_frame(tt).tobytes()); continue
            a_all = ease((tt - T_VIZ) / 0.5) * (1 - ease((tt - (total - 0.6)) / 0.5))
            enc.stdin.write(frame(min(max(tt - lead, 0), SECONDS - 1e-3), a_all).tobytes())
        enc.stdin.close(); assert enc.wait() == 0
    print('wrote', OUT, f'{total:.1f}s', f'L={L_cm:.2f}cm')

if __name__ == '__main__':
    if len(sys.argv) > 5 and sys.argv[5] == '--still':
        for tt in [1.0, 4.0, 7.0]: frame(tt, 1.0).save(str(OUT).replace('.mp4', f'_{tt:.0f}.png'))
    else:
        main()
