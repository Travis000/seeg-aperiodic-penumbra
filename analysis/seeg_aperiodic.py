#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
02_seeg_analysis_fig2.py  v2.0 — SOZ Inhibitory Signature (Route B)
=============================================================================
Major changes from v1:
  Panel A: Within-patient paired (Wilcoxon on ΔExp = SOZ − nonSOZ per patient)
           + Linear Mixed Model (Exponent ~ Is_SOZ + (1|Subject))
  Panel B: Distance capped at 63mm (within-electrode only), Exp<0.5 filtered
  Panel C: Uses engel_phase.xlsx (new Engel), cleaned data
  Outputs: to script directory (SCRIPT_DIR)

Inputs
------
  DATA_ROOT/{SubXX}/*_SEEG.edf, SEEG_Electrode_Target.xlsx,
  scalp_results.csv, engel_phase.xlsx
=============================================================================
"""

from __future__ import annotations
import os, re, sys, warnings, traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import mne
from mne.io import read_raw_edf, concatenate_raws
import yasa
from specparam import SpectralModel
from scipy import stats
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════
#  配置区
# ═══════════════════════════════════════════════════════════════════
SCRIPT_DIR      = Path(__file__).resolve().parent
# 数据根：默认与本脚本同目录（随项目移动自动生效）；否则回退到绝对路径
DATA_ROOT       = next(
    (p for p in [SCRIPT_DIR,
                 Path("."),
                 Path(".")]
     if (p / "Sub01").exists() or (p / "scalp_results.csv").exists()),
    SCRIPT_DIR)
INVENTORY_FILE  = DATA_ROOT / "SEEG_Electrode_Targets.csv"
SCALP_CSV       = DATA_ROOT / "scalp_results.csv"
ENGEL_FILE      = DATA_ROOT / "engel_phase.xlsx"
OUTPUT_CSV      = SCRIPT_DIR / "seeg_contact_results.csv"

PAIRED_SUBJECTS = [
    'Sub01','Sub02','Sub04','Sub05','Sub06','Sub07','Sub08','Sub09',
    'Sub11','Sub13','Sub14','Sub15','Sub17','Sub19','Sub20','Sub21'
]

# Signal processing
L_FREQ, H_FREQ     = 0.5, 45.0
EPOCH_DURATION      = 5.0
MIN_N2_SECONDS      = 30
PTP_FLOOR_UV        = 300e-6
PTP_PERCENTILE      = 90
PSD_FMIN, PSD_FMAX  = 1.0, 45.0
CONTACT_SPACING_MM  = 3.5

# FOOOF
FOOOF_FREQ_RANGE      = [1, 45]
FOOOF_APERIODIC       = "fixed"
FOOOF_PEAK_WIDTH      = [1, 8]
FOOOF_MAX_PEAKS       = 6
FOOOF_MIN_PEAK_HEIGHT = 0.05
FOOOF_R2_THRESHOLD    = 0.85

# ── Quality filters (NEW in v2) ──
MIN_EXPONENT        = 0.5          # Below = FOOOF fit failure
MAX_DISTANCE_MM     = 63.0         # 18 contacts × 3.5mm max

MIDLINE_CHS = ['Fz', 'Cz', 'Pz']

# Colors
COL_SOZ  = '#B2182B';  COL_NSOZ = '#2166AC'
COL_GOOD = '#2166AC';  COL_POOR = '#B2182B'
COL_BG   = '#FAFAFA'

mne.set_log_level("WARNING")
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ═══════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════

def parse_contact_string(s):
    if pd.isna(s) or str(s).strip() == '':
        return set()
    result = set()
    for part in str(s).split(','):
        part = part.strip()
        if '-' in part:
            try:
                a, b = part.split('-')
                result.update(range(int(a), int(b)+1))
            except: pass
        elif part.isdigit():
            result.add(int(part))
    return result


def load_inventory(filepath):
    inv = pd.read_csv(filepath)
    inv['Ablated_Set'] = inv['Ablated_Contacts'].apply(parse_contact_string)
    inv['SOZ_Set']     = inv['SOZ_Contacts'].apply(parse_contact_string)
    inv['Has_Ablation'] = inv['Ablated_Set'].apply(lambda s: len(s) > 0)
    inv['Has_SOZ']      = inv['SOZ_Set'].apply(lambda s: len(s) > 0)
    return inv


def load_engel(filepath):
    """engel_phase.xlsx: col0=sub01, col1=follow-up, col2=engel"""
    df = pd.read_excel(filepath)
    outcome = {}
    for _, row in df.iterrows():
        raw_sub = str(row.iloc[0]).strip()
        sub = 'Sub' + raw_sub[3:].zfill(2) if raw_sub.lower().startswith('sub') else raw_sub
        engel = str(row.iloc[2]).strip().upper()
        if engel in ('', 'NAN', 'NONE'): continue
        if   engel.startswith('IV'):   outcome[sub] = 'Poor'
        elif engel.startswith('III'):  outcome[sub] = 'Poor'
        elif engel.startswith('II'):   outcome[sub] = 'Good'
        elif engel.startswith('I'):    outcome[sub] = 'Good'
        else:                          outcome[sub] = 'Poor'
    return outcome


def load_scalp_midline_delta(scalp_csv):
    df = pd.read_csv(scalp_csv)
    mid = df[df['Channel'].isin(MIDLINE_CHS)]
    pre  = mid[mid['Condition']=='Pre'].groupby('Subject_ID')['Exponent'].mean()
    post = mid[mid['Condition']=='Post'].groupby('Subject_ID')['Exponent'].mean()
    common = pre.index.intersection(post.index)
    return {s: post[s] - pre[s] for s in common}


# ═══════════════════════════════════════════════════════════════════
#  SEEG Channel Parsing
# ═══════════════════════════════════════════════════════════════════

SCALP_SET = {'Fp1','Fp2','F3','F4','F7','F8','Fz','C3','C4','Cz',
             'P3','P4','P7','P8','Pz','T3','T4','T5','T6','T7','T8','O1','O2'}

def parse_seeg_channel(ch_name):
    raw = ch_name.strip()
    raw = re.sub(r'^(EEG|SEEG|POL|DC|BIP)\s+', '', raw)
    raw = re.sub(r'[-_](Ref|REF|LE|AVG|AV|le|ref|av|A1|A2)(\-?\d*)?$', '', raw)
    if re.match(r'^(ECG|EMG|EOG|EKG|Photic|Pulse|SpO2|IBI|Bursts|Suppr|DC|MK|Event)',
                raw, re.IGNORECASE):
        return None
    if raw in SCALP_SET:
        return None
    m = re.match(r"^([A-Za-z]+[']?)(\d+)$", raw)
    if m:
        return m.group(1), int(m.group(2))
    return None


def match_channels_to_inventory(ch_names, inv_sub):
    mapping = {}
    elec_lookup = {}
    for _, row in inv_sub.iterrows():
        elec_lookup[row['Electrode'].lower()] = row
    for ch in ch_names:
        parsed = parse_seeg_channel(ch)
        if parsed is None: continue
        elec, contact = parsed
        row = elec_lookup.get(elec.lower())
        if row is None: continue
        is_soz = contact in row['SOZ_Set']
        is_ablated = contact in row['Ablated_Set']
        has_soz = row['Has_SOZ']
        if has_soz and len(row['SOZ_Set']) > 0:
            soz_center = np.mean(list(row['SOZ_Set']))
            dist_contacts = abs(contact - soz_center)
        else:
            dist_contacts = np.nan
        mapping[ch] = {
            'electrode': row['Electrode'], 'contact': contact,
            'is_soz': is_soz, 'is_ablated': is_ablated,
            'has_soz_on_electrode': has_soz,
            'distance_contacts': dist_contacts,
            'distance_mm': dist_contacts * CONTACT_SPACING_MM
                           if not np.isnan(dist_contacts) else np.nan,
        }
    return mapping


# ═══════════════════════════════════════════════════════════════════
#  Signal Processing
# ═══════════════════════════════════════════════════════════════════

def find_staging_channel(raw, ch_mapping):
    candidates = [(ch, info['contact']) for ch, info in ch_mapping.items()
                  if not info['is_soz'] and ch in raw.ch_names]
    if candidates:
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
    seeg_chs = [ch for ch in raw.ch_names if ch in ch_mapping]
    return seeg_chs[0] if seeg_chs else raw.ch_names[0]


def stage_and_extract_n2(raw, staging_ch):
    sfreq = raw.info["sfreq"]
    try:
        sls = yasa.SleepStaging(raw, eeg_name=staging_ch)
        hypno = sls.predict()
    except Exception as e:
        print(f"    ⚠ YASA failed: {e}")
        return None, 0.0
    epoch_dur = 30.0
    n_samp_ep = int(epoch_dur * sfreq)
    total = raw.n_times
    mask = np.zeros(total, dtype=bool)
    for i, stage in enumerate(hypno):
        if stage == "N2":
            s = int(i * n_samp_ep)
            e = min(int(s + n_samp_ep), total)
            mask[s:e] = True
    n2_sec = mask.sum() / sfreq
    if n2_sec < MIN_N2_SECONDS:
        return None, n2_sec
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    stops  = np.where(diff == -1)[0] + 1
    if mask[0]:  starts = np.concatenate([[0], starts])
    if mask[-1]: stops  = np.concatenate([stops, [total]])
    segments = []
    for s, e in zip(starts, stops):
        seg = raw.copy().crop(tmin=s/sfreq, tmax=(e-1)/sfreq, include_tmax=True)
        segments.append(seg)
    if not segments:
        return None, n2_sec
    return concatenate_raws(segments, verbose=False), n2_sec


def epoch_and_reject(raw):
    events = mne.make_fixed_length_events(raw, duration=EPOCH_DURATION, overlap=0.0)
    epochs = mne.Epochs(raw, events, tmin=0,
                        tmax=EPOCH_DURATION - 1.0/raw.info["sfreq"],
                        baseline=None, preload=True, verbose=False)
    n_created = len(epochs)
    if n_created == 0:
        return None, 0, 0, 0
    data = epochs.get_data()
    ptp = data.max(axis=2) - data.min(axis=2)
    max_ptp = ptp.max(axis=1)
    p90 = np.percentile(max_ptp, PTP_PERCENTILE)
    threshold = max(p90, PTP_FLOOR_UV)
    keep = max_ptp <= threshold
    n_clean = int(keep.sum())
    thresh_uv = threshold * 1e6
    if n_clean < 6:
        return None, n_created, n_clean, thresh_uv
    return epochs[np.where(keep)[0]], n_created, n_clean, thresh_uv


def compute_psd(epochs):
    sfreq = epochs.info["sfreq"]
    n_fft = int(4.0 * sfreq)
    spectrum = epochs.compute_psd(
        method="welch", fmin=PSD_FMIN, fmax=PSD_FMAX,
        n_fft=n_fft, n_overlap=n_fft//2, verbose=False)
    psd = spectrum.get_data().mean(axis=0)
    return spectrum.freqs, psd


def fit_fooof(freqs, psd_ch):
    fm = SpectralModel(
        aperiodic_mode=FOOOF_APERIODIC,
        peak_width_limits=FOOOF_PEAK_WIDTH,
        max_n_peaks=FOOOF_MAX_PEAKS,
        min_peak_height=FOOOF_MIN_PEAK_HEIGHT,
        verbose=False)
    fm.fit(freqs, psd_ch, FOOOF_FREQ_RANGE)
    try:
        ap = fm.results.params.aperiodic.params
        met = fm.results.metrics.results
        return {
            "Exponent":  float(ap[1]),
            "Offset":    float(ap[0]),
            "R_Squared": float(met.get("gof_rsquared", np.nan)),
            "Error":     float(met.get("error_mae", np.nan)),
        }
    except Exception:
        try:
            ap = getattr(fm, 'aperiodic_params_', None) or \
                 getattr(fm, 'aperiodic_params', None)
            return {
                "Exponent":  float(ap[1]) if ap is not None else np.nan,
                "Offset":    float(ap[0]) if ap is not None else np.nan,
                "R_Squared": float(getattr(fm, 'r_squared_',
                                           getattr(fm, 'r_squared', np.nan))),
                "Error":     float(getattr(fm, 'error_',
                                           getattr(fm, 'error', np.nan))),
            }
        except Exception:
            return {"Exponent": np.nan, "Offset": np.nan,
                    "R_Squared": np.nan, "Error": np.nan}


# ═══════════════════════════════════════════════════════════════════
#  File Discovery & Per-subject Processing
# ═══════════════════════════════════════════════════════════════════

def discover_seeg_files(data_root):
    tasks = []
    for sub_dir in sorted(data_root.iterdir()):
        if not sub_dir.is_dir() or not sub_dir.name.startswith("Sub"):
            continue
        sub_id = sub_dir.name
        if sub_id not in PAIRED_SUBJECTS:
            continue
        for edf in sorted(sub_dir.glob("*.edf")):
            if re.search(r'_seeg\.edf$', edf.name, re.IGNORECASE):
                tasks.append({"subject": sub_id, "file": edf})
                break
    return tasks


def process_one(task, inv):
    sub   = task["subject"]
    fpath = task["file"]
    inv_sub = inv[inv.Subject == sub]
    results = []

    try:
        raw = read_raw_edf(str(fpath), preload=True, verbose=False)
        dur = raw.n_times / raw.info["sfreq"]
        print(f"    Loaded: {dur/3600:.1f}h, {len(raw.ch_names)}ch, "
              f"{raw.info['sfreq']:.0f}Hz")

        ch_mapping = match_channels_to_inventory(raw.ch_names, inv_sub)
        seeg_chs = [ch for ch in raw.ch_names if ch in ch_mapping]
        if not seeg_chs:
            print(f"    ⚠ No SEEG channels matched inventory")
            return results, psd_info

        n_soz = sum(1 for ch in seeg_chs if ch_mapping[ch]['is_soz'])
        print(f"    Matched {len(seeg_chs)} channels (SOZ={n_soz})")

        raw.pick_channels(seeg_chs)
        raw.set_channel_types({ch: "seeg" for ch in raw.ch_names})
        raw.filter(L_FREQ, H_FREQ, method="fir", fir_design="firwin", verbose=False)

        staging_ch = find_staging_channel(raw, ch_mapping)
        print(f"    Staging channel: {staging_ch}")
        raw_staging = raw.copy()
        raw_staging.set_channel_types({staging_ch: "eeg"})
        n2_raw, n2_sec = stage_and_extract_n2(raw_staging, staging_ch)
        print(f"    N2: {n2_sec:.0f}s ({n2_sec/60:.1f}min)")

        if n2_raw is None:
            print(f"    ⚠ Skipped — N2 too short ({n2_sec:.0f}s)")
            return results, psd_info

        available = [ch for ch in seeg_chs if ch in n2_raw.ch_names]
        if not available:
            n2_raw = raw.copy(); available = seeg_chs
        n2_raw.pick_channels(available)

        epochs, n_created, n_clean, thresh_uv = epoch_and_reject(n2_raw)
        print(f"    Epochs: {n_created} → {n_clean} clean (PTP={thresh_uv:.0f}µV)")
        if epochs is None:
            print(f"    ⚠ Skipped — too few epochs")
            return results, psd_info

        freqs, psd = compute_psd(epochs)
        n_r2_fail = 0
        n_exp_fail = 0

        for ci, ch in enumerate(epochs.ch_names):
            if ch not in ch_mapping: continue
            info = ch_mapping[ch]
            res = fit_fooof(freqs, psd[ci])

            if np.isnan(res["R_Squared"]) or res["R_Squared"] < FOOOF_R2_THRESHOLD:
                n_r2_fail += 1; continue
            if np.isnan(res["Exponent"]) or res["Exponent"] < MIN_EXPONENT:
                n_exp_fail += 1; continue

            results.append({
                "Subject": sub, "Electrode": info['electrode'],
                "Contact": info['contact'], "Channel": ch,
                "Exponent": round(res["Exponent"], 4),
                "Offset":   round(res["Offset"], 4),
                "R_Squared": round(res["R_Squared"], 4),
                "Is_SOZ": info['is_soz'], "Is_Ablated": info['is_ablated'],
                "Has_SOZ_on_Electrode": info['has_soz_on_electrode'],
                "Distance_contacts": round(info['distance_contacts'], 2)
                    if not np.isnan(info['distance_contacts']) else np.nan,
                "Distance_mm": round(info['distance_mm'], 1)
                    if not np.isnan(info['distance_mm']) else np.nan,
                "N_Epochs": n_clean,
            })

        if n_r2_fail or n_exp_fail:
            print(f"    Filtered: {n_r2_fail} R²<{FOOOF_R2_THRESHOLD}, "
                  f"{n_exp_fail} Exp<{MIN_EXPONENT}")
        n_good = len(results)
        if n_good > 0:
            soz_e = [r['Exponent'] for r in results if r['Is_SOZ']]
            ns_e  = [r['Exponent'] for r in results if not r['Is_SOZ']]
            print(f"    ✓ {n_good} contacts passed QC")
            if soz_e:  print(f"      SOZ:     {np.mean(soz_e):.3f}±{np.std(soz_e):.3f} (n={len(soz_e)})")
            if ns_e:   print(f"      Non-SOZ: {np.mean(ns_e):.3f}±{np.std(ns_e):.3f} (n={len(ns_e)})")

    except Exception as e:
        print(f"    ✖ FAILED: {e}")
        traceback.print_exc()
    return results


# ═══════════════════════════════════════════════════════════════════
#  Statistical Analyses
# ═══════════════════════════════════════════════════════════════════

def run_paired_analysis(df):
    """Within-patient paired: per-patient Mean_SOZ − Mean_NonSOZ."""
    sub_diffs = []
    for sub, grp in df.groupby('Subject'):
        soz  = grp[grp.Is_SOZ]['Exponent'].values
        nsoz = grp[~grp.Is_SOZ]['Exponent'].values
        if len(soz) >= 2 and len(nsoz) >= 2:
            sub_diffs.append({
                'Subject': sub,
                'SOZ_mean': np.mean(soz), 'NonSOZ_mean': np.mean(nsoz),
                'Delta': np.mean(soz) - np.mean(nsoz),
                'SOZ_n': len(soz), 'NonSOZ_n': len(nsoz),
            })
    sm = pd.DataFrame(sub_diffs)
    st = {}
    if len(sm) >= 5:
        w_stat, w_p = stats.wilcoxon(sm['SOZ_mean'], sm['NonSOZ_mean'],
                                     alternative='less')
        t_stat, t_p = stats.ttest_rel(sm['SOZ_mean'], sm['NonSOZ_mean'])
        d = sm['Delta'].mean() / sm['Delta'].std() if sm['Delta'].std() > 0 else 0
        st = dict(N=len(sm),
                  SOZ_mean=sm['SOZ_mean'].mean(), SOZ_std=sm['SOZ_mean'].std(),
                  NonSOZ_mean=sm['NonSOZ_mean'].mean(), NonSOZ_std=sm['NonSOZ_mean'].std(),
                  Delta_mean=sm['Delta'].mean(), Delta_std=sm['Delta'].std(),
                  Wilcoxon_W=w_stat, Wilcoxon_p=w_p,
                  ttest_t=t_stat, ttest_p=t_p, d_paired=d)
    return sm, st


def run_lmm(df):
    """LMM: Exponent ~ Is_SOZ + (1|Subject)."""
    try:
        import statsmodels.formula.api as smf
        ldf = df[['Subject','Exponent','Is_SOZ']].copy()
        ldf['SOZ_int'] = ldf['Is_SOZ'].astype(int)
        model = smf.mixedlm("Exponent ~ SOZ_int", ldf, groups=ldf["Subject"])
        res = model.fit(reml=True)
        rv = float(res.cov_re.iloc[0,0])
        return dict(
            beta=res.fe_params['SOZ_int'], se=res.bse_fe['SOZ_int'],
            t=res.tvalues['SOZ_int'],      p=res.pvalues['SOZ_int'],
            intercept=res.fe_params['Intercept'],
            random_var=rv, residual_var=res.scale,
            ICC=rv/(rv+res.scale), summary=str(res.summary()),
        )
    except Exception as e:
        print(f"  ⚠ LMM failed: {e}")
        traceback.print_exc()
        return None


def main():
    print("="*70)
    print("  SEEG Aperiodic Analysis — Figure 2 v2.0")
    print("  Within-patient paired + LMM + quality filters")
    print("="*70)

    inv = load_inventory(INVENTORY_FILE)
    print(f"  Inventory: {len(inv)} electrodes, {inv.Subject.nunique()} subjects")
    print(f"  Electrodes with SOZ: {inv.Has_SOZ.sum()}")

    engel_map = load_engel(ENGEL_FILE)
    print(f"  Engel outcomes: {len(engel_map)} subjects")
    for sub in PAIRED_SUBJECTS:
        print(f"    {sub}: {engel_map.get(sub, '?')}")

    scalp_delta = load_scalp_midline_delta(SCALP_CSV)
    print(f"  Scalp midline ΔExp: {len(scalp_delta)} subjects")

    tasks = discover_seeg_files(DATA_ROOT)
    print(f"  SEEG files found: {len(tasks)}")

    if not tasks:
        print("\n  ⚠ No SEEG files found.")
        sys.exit(1)

    all_results = []

    for task in tqdm(tasks, desc="Processing", ncols=80):
        sub = task["subject"]
        print(f"\n  ─── {sub}: {task['file'].name} ───")
        results = process_one(task, inv)
        all_results.extend(results)

    if not all_results:
        print("\n  ✖ No results."); sys.exit(1)

    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Results saved: {OUTPUT_CSV}  ({len(df)} rows)")

    # ── Summary ──
    print(f"\n  {'='*60}")
    print(f"  SUMMARY (Exponent ≥ {MIN_EXPONENT})")
    print(f"  {'='*60}")
    print(f"  Subjects: {df.Subject.nunique()}")
    print(f"  Total contacts: {len(df)}")
    print(f"  SOZ: {df.Is_SOZ.sum()}, Non-SOZ: {(~df.Is_SOZ).sum()}")

    soz = df[df.Is_SOZ]['Exponent']
    nsoz = df[~df.Is_SOZ]['Exponent']
    print(f"  SOZ exponent:     {soz.mean():.4f} ± {soz.std():.4f}")
    print(f"  Non-SOZ exponent: {nsoz.mean():.4f} ± {nsoz.std():.4f}")

    # ── PRIMARY: Within-patient paired ──
    print(f"\n  {'─'*55}")
    print(f"  PRIMARY: Within-patient paired (ΔExp = SOZ − Non-SOZ)")
    print(f"  {'─'*55}")
    sm, ps = run_paired_analysis(df)
    if ps:
        print(f"  N = {ps['N']} patients")
        print(f"  SOZ mean:     {ps['SOZ_mean']:.4f} ± {ps['SOZ_std']:.4f}")
        print(f"  Non-SOZ mean: {ps['NonSOZ_mean']:.4f} ± {ps['NonSOZ_std']:.4f}")
        print(f"  Δ (SOZ−NonSOZ): {ps['Delta_mean']:+.4f} ± {ps['Delta_std']:.4f}")
        print(f"  Cohen's d (paired): {ps['d_paired']:.3f}")
        print(f"  Wilcoxon (SOZ>NonSOZ): W={ps['Wilcoxon_W']:.0f}, p={ps['Wilcoxon_p']:.6f}")
        print(f"  Paired t: t={ps['ttest_t']:.3f}, p={ps['ttest_p']:.6f}")
        print(f"\n  Per-patient detail:")
        for _, row in sm.sort_values('Delta', ascending=False).iterrows():
            d = "↑" if row['Delta'] > 0 else "↓"
            print(f"    {row['Subject']}: Δ={row['Delta']:+.4f} {d}  "
                  f"(SOZ n={row['SOZ_n']:.0f}, NonSOZ n={row['NonSOZ_n']:.0f})")

    # ── LMM ──
    print(f"\n  {'─'*55}")
    print(f"  LMM: Exponent ~ Is_SOZ + (1|Subject)")
    print(f"  {'─'*55}")
    lmm = run_lmm(df)
    if lmm:
        print(f"  SOZ effect: β = {lmm['beta']:.4f}, SE = {lmm['se']:.4f}")
        print(f"  t = {lmm['t']:.3f}, p = {lmm['p']:.6f}")
        print(f"  Intercept: {lmm['intercept']:.4f}")
        print(f"  Random var: {lmm['random_var']:.4f}, Residual: {lmm['residual_var']:.4f}")
        print(f"  ICC: {lmm['ICC']:.3f}")
        print(f"\n{lmm['summary']}")

    # ── Distance gradient ──
    grad = df[df.Has_SOZ_on_Electrode & df.Distance_mm.notna()
              & (df.Distance_mm <= MAX_DISTANCE_MM)]
    if len(grad) > 10:
        rho, p = stats.spearmanr(grad['Distance_mm'], grad['Exponent'])
        print(f"\n  Distance gradient (≤{MAX_DISTANCE_MM}mm, N={len(grad)}):")
        print(f"    Spearman ρ = {rho:.4f}, p = {p:.2e}")

    # ── Cross-modal ──
    soz_s = df[df.Is_SOZ].groupby('Subject')['Exponent'].mean()
    cross = [(soz_s[s], scalp_delta[s]) for s in soz_s.index if s in scalp_delta]
    if len(cross) >= 5:
        x, y = zip(*cross)
        r, p = stats.pearsonr(x, y)
        rho, ps2 = stats.spearmanr(x, y)
        print(f"\n  Cross-modal (N={len(cross)}):")
        print(f"    Pearson  r={r:.4f}, p={p:.4f}")
        print(f"    Spearman ρ={rho:.4f}, p={ps2:.4f}")

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
