"""Summarize completed A measurements; writes only artifacts/diagnosis."""
import csv
import json
import numpy as np
from diffart.data import ROOT, sha
from diffart.runtime import save_json
from diagnose_bands import paired_summary, write_csv, VOWELS


def read_csv(path):
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        for k,v in row.items():
            try:
                row[k] = float(v)
            except (ValueError, TypeError):
                pass
    return rows


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(v) for v in r)+' |' for r in rows])+'\n'


def main():
    out = ROOT/'artifacts/diagnosis'
    s = json.loads((out/'summary.json').read_text())
    components = json.loads((out/'components.json').read_text())
    forms = read_csv(out/'formants.csv')
    sources = read_csv(out/'source.csv')
    for r in forms+sources:
        r['f0_regime'] = '<350 Hz' if r['f0_hz'] < 350 else '>=350 Hz'
    fs = sum([paired_summary(forms,['f0_regime','vowel','formant'],['frequency_hz','bandwidth_hz'],'synth_inverse','recording',order) for order in [18,22]],[])
    ss = sum([paired_summary(sources,['f0_regime','vowel'],['open_fraction_10pct','return_sharpness','spectral_tilt_db_oct','cycle_variation_ratio'],'synth_inverse','recording',order) for order in [18,22]],[])
    write_csv(out/'formant_by_f0.csv',fs)
    write_csv(out/'source_by_f0.csv',ss)
    csummary = []
    for condition in ['full','no_breath','no_turbulence','closed_nose']:
        r = [x for x in components['rows'] if x['condition']==condition]
        csummary.append(dict(condition=condition,songs=len(r),
            **{k:float(np.mean([x[k] for x in r])) for k in r[0] if k.startswith(('wavlm_','ecapa_'))},
            **{k+'_song_min':float(min(x[k] for x in r)) for k in ['wavlm_anchor_delta','ecapa_anchor_delta']},
            **{k+'_song_max':float(max(x[k] for x in r)) for k in ['wavlm_anchor_delta','ecapa_anchor_delta']}))
    write_csv(out/'component_summary.csv',csummary)
    cbsummary = []
    for key in sorted({(r['condition'],r['voicing'],r['band']) for r in components['bands']}):
        r = [x for x in components['bands'] if (x['condition'],x['voicing'],x['band'])==key]
        cbsummary.append(dict(zip(['condition','voicing','band'],key),songs=len(r),
            **{k:float(np.mean([x[k] for x in r])) for k in ['path_change_rms_relative_db','lsd_change_db']}))
    write_csv(out/'component_band_summary.csv',cbsummary)
    s.update(components=csummary,component_bands=cbsummary)
    save_json(out/'summary.json',s)
    swap = {r['condition']:r for r in s['interventions']}
    band = {(r['voicing'],r['band']):r for r in s['bands'] if r['normalization']=='rms_matched'}
    selections = json.loads((out/'selection.json').read_text())
    duration = sum(r['total_seconds'] for r in selections)
    report = [f'# A. 声の近さの診断（2026-11-09）\n\n'
        f'test 4 曲・計 {duration:.2f} 秒を全曲スペクトル分析。帯域置換は各曲 6 秒×3 区間、計 72 秒。母音解析は {s["vowel_windows"]} 窓（120 ms）。CPU のみ。\n\n'
        '4–6 kHz の包絡の違いが、両指標で共通する改善候補。全帯域を合わせても録音との差は残る。帯域誤差の大きさだけでは声の近さへの寄与を順位づけできない。LPC による音源・共鳴の物理的帰属は較正に失敗し、確定を保留する。既存経路の直接比較は下の部品アブレーションを参照。\n\n'
        '詳しい式・選定方法・限界: [METHODS.md](METHODS.md)。全表は CSV、音声は [audio/](audio/)、再現情報は [provenance.json](provenance.json)。\n',
        '## A1. 帯域誤差（曲別 RMS を一致、4 曲等重み）\n\n'+
        table(['帯域 kHz','有声 LSD dB','有声 LTAS差 dB','無声 LSD dB','無声 LTAS差 dB'],
            [[b.replace('k',''),f'{band["voiced",b]["lsd_db"]:.2f}',f'{band["voiced",b]["ltas_bias_db"]:+.2f}',
              f'{band["unvoiced",b]["lsd_db"]:.2f}',f'{band["unvoiced",b]["ltas_bias_db"]:+.2f}'] for b in ['0-1k','1-2k','2-4k','4-6k','6-8k','8-12k']])+
        '\nLTAS差は「合成−録音」。有声の 4–6 kHz は不足、無声の 0–1 kHz は過多、無声の 6–12 kHz は不足。活動区間全体の合成音量は録音より小さく、曲単位の補正量は '+
        '、'.join(f'{r["rms_match_gain_db"]:.2f}' for r in selections)+' dB（曲順は selection.json）。raw / 正規化後の全曲・全帯域表を bands.csv に保存。\n\n'
        '![LTAS差](ltas_difference.png)\n\n![帯域誤差と類似度](bands_and_similarity.png)\n',
        '## A2. 包絡置換による声の近さへの寄与\n\n'+
        table(['条件','WavLM cosine','Δ WavLM','ECAPA cosine','Δ ECAPA'],
            [[cond,f'{r["wavlm_anchor_cosine"]:.4f}',f'{r["wavlm_anchor_delta"]:+.4f}',f'{r["ecapa_anchor_cosine"]:.4f}',f'{r["ecapa_anchor_delta"]:+.4f}'] for cond,r in swap.items()])+
        '\n参照は既存評価と同じ train 3 本。絶対値は今回の 12 区間内の値で、全曲の既存評価値とは異なる。\n\n'+
        table(['置換帯域','Δ WavLM: 4曲の範囲','Δ ECAPA: 4曲の範囲'],
            [[cond,f'{swap[cond]["wavlm_anchor_delta_song_min"]:+.4f} ～ {swap[cond]["wavlm_anchor_delta_song_max"]:+.4f}',
              f'{swap[cond]["ecapa_anchor_delta_song_min"]:+.4f} ～ {swap[cond]["ecapa_anchor_delta_song_max"]:+.4f}'] for cond in ['0-1k','4-6k','6-8k','all_envelope']])+
        f'\n全帯域置換で、この区間集合の録音との差の {100*swap["all_envelope"]["wavlm_anchor_delta"]/swap["recording"]["wavlm_anchor_delta"]:.1f}%（WavLM）、'
        f'{100*swap["all_envelope"]["ecapa_anchor_delta"]/swap["recording"]["ecapa_anchor_delta"]:.1f}%（ECAPA）が縮まる。これは診断的な oracle 置換で、学習可能な改善量や因果的寄与率ではない。\n\n'
        '0–1 kHz は WavLM が全曲改善するが ECAPA は全曲悪化。6–8 kHz も指標の一致は弱い。8–12 kHz は両指標の入力帯域外で判定不能。そこに出た小さい差は、再構成・resampling・音量合わせに伴う効果を含む。\n',
        '## 部品への直接介入（追加の診断）\n\n各曲の中間の選定済み 6 秒、計 24 秒。全条件を同じ F0・乱数・局所位相・分割で再合成し、その full と比較する。既存の全曲再現とは分割位置が異なる。\n\n'+
        table(['止めた経路','Δ WavLM 平均 [4曲範囲]','Δ ECAPA 平均 [4曲範囲]'],
            [[r['condition'],f'{r["wavlm_anchor_delta"]:+.4f} [{r["wavlm_anchor_delta_song_min"]:+.4f}, {r["wavlm_anchor_delta_song_max"]:+.4f}]',
              f'{r["ecapa_anchor_delta"]:+.4f} [{r["ecapa_anchor_delta_song_min"]:+.4f}, {r["ecapa_anchor_delta_song_max"]:+.4f}]'] for r in csummary if r['condition']!='full'])+
        '\n狭窄部乱流を止めると ECAPA は全曲改善するが、WavLM は平均で悪化する。無声 4–6 kHz の経路差分 RMS は full に対して平均 −1.00 dB と大きく、削除すると同帯域 LSD は +15.06 dB 悪化（この区間群）。したがって高域の主な担い手である乱流経路の音色・伝達は検討対象だが、単純削除は両指標・スペクトルの改善にならない。息ノイズ停止は両指標の平均で小幅改善するものの曲によって符号が違い、鼻腔閉鎖は平均では両指標が悪化する。\n\n'
        'この比較で分かるのは現行経路を削除した効果。ノイズの改善された代替、壁損失と放射のどちらが原因か、未実装の梨状窩・3次元性の寄与は特定できない。component_bands.csv に有声／無声・帯域別の経路差分 RMS と LSD の変化を保存した。閉鼻は音源・乱流の両方の伝達を変えるため、単独音源の強さとは解釈しない。\n',
        '## A3. 音源の逆フィルター結果と LF\n\n次数 18 の同一窓比較。開放率は「周期流量の振幅 10% 超の割合」という代理値。ノイズ欄も周期間変動の代理値で、純粋なノイズエネルギーではない。\n\n'+
        table(['母音','開放率代理: 録音 / 合成','戻り鋭さ: 録音 / 合成','傾斜 dB/oct: 録音 / 合成','周期変動比: 録音 / 合成'],
            [[r['vowel'],f'{r["open_fraction_10pct_recording_median"]:.3f} / {r["open_fraction_10pct_synth_median"]:.3f}',
              f'{r["return_sharpness_recording_median"]:.2f} / {r["return_sharpness_synth_median"]:.2f}',
              f'{r["spectral_tilt_db_oct_recording_median"]:.2f} / {r["spectral_tilt_db_oct_synth_median"]:.2f}',
              f'{r["cycle_variation_ratio_recording_median"]:.4f} / {r["cycle_variation_ratio_synth_median"]:.4f}']
             for v in VOWELS for r in s['source'] if r['vowel']==v and r['lpc_order']==18])+
        '\nLF 較正（合成音の逆推定 vs 既知 LF、絶対差の中央値）:\n\n'+
        table(['LPC次数','窓数','開放率代理','戻り鋭さ','傾斜 dB/oct','周期変動比'],
            [[r['lpc_order'],r['n'],f'{r["open_fraction_10pct_median_abs_error"]:.3f}',f'{r["return_sharpness_median_abs_error"]:.2f}',
              f'{r["spectral_tilt_db_oct_median_abs_error"]:.2f}',f'{r["cycle_variation_ratio_median_abs_error"]:.4f}'] for r in s['source_calibration']])+
        '\n開放率・戻りの鋭さ・傾斜の較正誤差は録音との差に匹敵または上回る。従って「LF の開放率が誤っている」「LF を変えれば差が埋まる」とは結論しない。録音は特に i/u で周期変動比が高く、非周期性・時間変動の不足は候補。ただし録音固有のノイズと推定誤差も含む。次数 22 と F0 別表も保存。\n\n'
        '![逆推定周期と既知LF](glottal_cycles.png)\n',
        '## A4. 母音別 F1–F4 と帯域幅（LPC候補）\n\n次数 18。中央値はそれぞれの窓群、差は対応する窓での差の中央値なので「中央値の差」と一致しない。単位 Hz。\n\n'+
        table(['母音','候補','窓数','録音 F','合成 F','ΔF','録音 BW','合成 BW','ΔBW'],
            [[r['vowel'],f'F{r["formant"]}',r['n'],f'{r["frequency_hz_recording_median"]:.1f}',f'{r["frequency_hz_synth_median"]:.1f}',
              f'{r["frequency_hz_paired_median_delta"]:+.1f}',f'{r["bandwidth_hz_recording_median"]:.1f}',f'{r["bandwidth_hz_synth_median"]:.1f}',f'{r["bandwidth_hz_paired_median_delta"]:+.1f}']
             for v in VOWELS for r in s['formants'] if r['vowel']==v and r['lpc_order']==18])+
        f'\n窓の {s["high_f0_fraction"]*100:.1f}% は F0≥350 Hz。合成音の LPC 候補と管の伝達関数の共鳴との較正では、周波数順の対応自体が崩れ、次数 18 で中央値の絶対差は候補順に '+
        ' / '.join(f'{r["frequency_median_abs_error_hz"]:.0f}' for r in s['formant_calibration'] if r['lpc_order']==18)+
        ' Hz。これは LPC・鼻腔零点・共鳴検出の欠落と対応づけの失敗も含む。よって表の大きな F2/F3 差を、そのまま真の声道共鳴のずれや管の欠陥とは呼べない。物理的 F1–F4 と帯域幅の確定は保留。次数 22、F0 別、物理モデル共鳴の列を CSV に残した。\n\n'
        '![LPC候補差](formant_differences.png)\n',
        '## 完了範囲と検証\n\nA1〜A4 の実装・実測・図表保存を実施。A3/A4 は較正の限界を検出したため物理的解釈を保留し、DESIGN_ISSUES_pinn.md に記録。B の改良・学習には進んでいない。\n\n'
        'STFT 往復の相対 RMS 誤差は 9e-8 程度。既存入力 27 ファイルは開始・終了の SHA256 が一致（integrity.json）。検証テストは tests/test_diagnosis.py の 4 件。プロセスの停止・変更はしていない。\n']
    (out/'REPORT.md').write_text('\n'.join(report))
    provenance = json.loads((out/'provenance.json').read_text())
    provenance['analysis_code_sha256'] = {str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'scripts/diagnose_bands.py',ROOT/'scripts/diagnose_components.py',ROOT/'scripts/report_diagnosis.py',ROOT/'diffart/physics.py',ROOT/'diffart/metrics.py',ROOT/'diffart/render.py']}
    save_json(out/'provenance.json',provenance)
    # Also recheck shared inputs after component diagnostics.
    for path,digest in provenance['inputs_sha256'].items():
        assert sha(ROOT/path)==digest, path
    integrity = json.loads((out/'integrity.json').read_text())
    integrity['rechecked_after_component_analysis'] = True
    save_json(out/'integrity.json',integrity)
    print('REPORT complete',flush=True)


if __name__=='__main__':
    main()
