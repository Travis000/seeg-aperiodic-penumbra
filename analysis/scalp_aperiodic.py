#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
01_scalp_analysis.py  (v5.1  -  Production + PSD Export)
=============================================================================
Study  : Validation of Aperiodic Dynamics of Human Brain E/I Balance
         Based on the SEEG Thermocoagulation Lesion Model
Target : Brain / JNNP
Phase  : 1  -  Scalp EEG Pre/Post Aperiodic Exponent Analysis

Changes in v5.1 (from v5)
---------------------------
  Fix 1: PSD saving simplified  -  all_psds saved directly without
         error-prone pruning logic.  Downstream plotting scripts
         select the correct Post part via scalp_results.csv.
  Fix 2: YASA sleep staging channel fallback now uses a robust
         candidate list (C4->C3->Cz->P4->P3->Pz->F4->F3) instead of
         blindly falling back to ch_names[0].
  Fix 3: standard_channel_order saved in .npz for unambiguous
         channel identification during plotting.

Pipeline
--------
  1. Traverse & Load  (auto-merge split Pre; separate Post parts)
  2. Pre-process       (channel pick -> avg ref -> FIR 0.5-45 Hz)
  3. Sleep staging     (YASA -> extract N2)
  4. Artifact reject   (5-s epochs, ADAPTIVE PTP threshold)
  5. Spectral param    (specparam -> exponent & offset)
  6. Export            (scalp_results.csv + processing_log.csv
                        + scalp_psd_curves.npz)

Author : [Your Name]
Date   : 2026-02
=============================================================================
"""

from __future__ import annotations

import os
import re
import sys
import warnings
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import mne
from mne.io import read_raw_edf, concatenate_raws
import yasa
from specparam import SpectralModel
from tqdm import tqdm

# -- Configuration ------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
# 数据根：默认与本脚本同目录（随项目移动自动生效）；否则回退到绝对路径
DATA_ROOT  = next(
    (p for p in [SCRIPT_DIR,
                 Path("."),
                 Path(".")]
     if (p / "Sub01").exists() or (p / "scalp_results.csv").exists()),
    SCRIPT_DIR)
OUTPUT_CSV = DATA_ROOT / "scalp_results.csv"
LOG_CSV    = DATA_ROOT / "processing_log.csv"
PSD_NPZ    = DATA_ROOT / "scalp_psd_curves.npz"

STANDARD_1020 = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8",
    "Fz", "Cz", "Pz",
]

CHANNEL_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

# Robust candidate list for YASA sleep staging (central > parietal > frontal)
YASA_CANDIDATES = ["C4", "C3", "Cz", "P4", "P3", "Pz", "F4", "F3"]

L_FREQ, H_FREQ     = 0.5, 45.0     # 45 Hz to avoid 50 Hz line noise (China)
EPOCH_DURATION      = 5.0
MIN_N2_SECONDS      = 30            # lowered to maximize paired sample size
PSD_FMIN, PSD_FMAX  = 1.0, 45.0

# Adaptive artifact rejection  -  no ceiling, pure adaptive + floor
PTP_FLOOR_UV    = 150e-6            # Never reject epochs below this (V)
PTP_PERCENTILE  = 90                # Reject the noisiest 10%

FOOOF_FREQ_RANGE      = [1, 45]
FOOOF_APERIODIC       = "fixed"
FOOOF_PEAK_WIDTH      = [1, 8]
FOOOF_MAX_PEAKS       = 4
FOOOF_MIN_PEAK_HEIGHT = 0.05
mne.set_log_level("WARNING")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# -- Global PSD collector (for Figure 1A) -------------------------------------
# Saved as-is to .npz at the end.  Downstream scripts select the correct
# Post part by cross-referencing scalp_results.csv.
#
# Keys:
#   "freqs"                       -> (n_freqs,)       shared frequency axis
#   "standard_channel_order"      -> (19,)            STANDARD_1020 names
#   "{Sub}_{Cond}_psd"            -> (n_ch, n_freqs)  per-channel PSD
#   "{Sub}_{Cond}_chans"          -> (n_ch,)          channel names for this file
#   "{Sub}_{Cond}_{Part}_psd"     -> (n_ch, n_freqs)  for multi-part Post files
#   "{Sub}_{Cond}_{Part}_chans"   -> (n_ch,)
all_psds: dict[str, np.ndarray] = {}


# -- Helpers ------------------------------------------------------------------

def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    tqdm.write(f"[{ts}] [{level}] {msg}")


def standardize_channel_name(ch: str) -> str:
    cleaned = ch.strip()
    cleaned = re.sub(
        r"^(EEG|EMG|ECG|EOG|REF|DC|Photic|Pulse Rate|IBI|Bursts|Suppr)\s+",
        "", cleaned,
    )
    cleaned = re.sub(r"[-_](Ref|REF|LE|AVG|AV|le|ref|av|A1|A2)(\-?\d*)?$", "", cleaned)
    cleaned = re.sub(r"-\d+$", "", cleaned)
    cleaned = CHANNEL_ALIASES.get(cleaned, cleaned)
    return cleaned


def discover_files(data_root: Path) -> list[dict]:
    tasks: list[dict] = []
    subject_dirs = sorted(
        [d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("Sub")]
    )

    for sub_dir in subject_dirs:
        subject_id = sub_dir.name
        edfs = sorted(sub_dir.glob("*.edf"), key=lambda p: p.name.lower())

        pre_files:  list[Path] = []
        post_groups: dict[str, list[Path]] = {}

        for edf in edfs:
            fname = edf.stem
            fname_lower = fname.lower()

            if "seeg" in fname_lower:
                continue

            fname_normalized = re.sub(r"_+", "_", fname_lower)

            if re.search(r"pre\d*$", fname_normalized):
                pre_files.append(edf)
                continue

            m = re.search(r"post(\d*)$", fname_normalized)
            if m:
                digit = m.group(1)
                if digit:
                    post_groups[f"Post_part{digit}"] = [edf]
                else:
                    post_groups.setdefault("Post", []).append(edf)
                continue

        if pre_files:
            tasks.append({
                "subject": subject_id, "condition": "Pre",
                "files": sorted(pre_files), "part": None,
            })

        if post_groups:
            for group_key, files in sorted(post_groups.items()):
                tasks.append({
                    "subject": subject_id, "condition": "Post",
                    "files": sorted(files), "part": group_key,
                })

    return tasks


def load_and_merge(files: list[Path]) -> mne.io.Raw:
    raws = []
    for f in files:
        raw = read_raw_edf(str(f), preload=True, verbose=False)
        raws.append(raw)
    if len(raws) == 1:
        return raws[0]
    log(f"  Merging {len(raws)} split files ...")
    return concatenate_raws(raws, verbose=False)


def rename_channels(raw: mne.io.Raw) -> mne.io.Raw:
    mapping = {}
    seen: dict[str, int] = {}
    for ch in raw.ch_names:
        new = standardize_channel_name(ch)
        if new in seen:
            seen[new] += 1
            mapping[ch] = f"{new}__dup{seen[new]}"
        else:
            seen[new] = 0
            mapping[ch] = new
    raw.rename_channels(mapping)
    dupes = [c for c in raw.ch_names if "__dup" in c]
    if dupes:
        raw.drop_channels(dupes)
    return raw


def pick_standard_channels(raw: mne.io.Raw) -> mne.io.Raw:
    available = [ch for ch in STANDARD_1020 if ch in raw.ch_names]
    if not available:
        log(f"  Channels after renaming: {raw.ch_names}", "ERROR")
        raise RuntimeError("No standard 10-20 channels found.")
    raw.pick_channels(available)
    log(f"  Channels retained ({len(available)})")
    return raw


def preprocess(raw: mne.io.Raw) -> mne.io.Raw:
    raw.set_eeg_reference("average", projection=False, verbose=False)
    raw.filter(L_FREQ, H_FREQ, method="fir", fir_design="firwin", verbose=False)
    return raw


def find_yasa_channel(raw: mne.io.Raw) -> str:
    """
    Find the best available channel for YASA sleep staging.
    Preference: C4 > C3 > Cz > P4 > P3 > Pz > F4 > F3.
    Falls back to first EEG channel only if none of the candidates exist.
    """
    for candidate in YASA_CANDIDATES:
        if candidate in raw.ch_names:
            return candidate

    # Last resort: first channel that is at least an EEG channel
    eeg_chs = [ch for ch in raw.ch_names if ch in STANDARD_1020]
    if eeg_chs:
        log(f"  (!!) No central/parietal ch found; using {eeg_chs[0]} for staging",
            "WARN")
        return eeg_chs[0]

    # Should never reach here after pick_standard_channels, but just in case
    log(f"  (!!) No known EEG ch; using {raw.ch_names[0]} for staging", "WARN")
    return raw.ch_names[0]


def stage_and_extract_n2(raw: mne.io.Raw) -> tuple[mne.io.Raw | None, float]:
    sfreq = raw.info["sfreq"]

    eeg_chan = find_yasa_channel(raw)
    log(f"  YASA staging channel: {eeg_chan}")

    try:
        sls = yasa.SleepStaging(raw, eeg_name=eeg_chan)
        hypno = sls.predict()
    except Exception as e:
        log(f"  (!!) YASA failed: {e}", "WARN")
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
    log(f"  N2 total: {n2_sec:.0f} s ({n2_sec/60:.1f} min)")

    if n2_sec < MIN_N2_SECONDS:
        return None, n2_sec

    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    stops  = np.where(diff == -1)[0] + 1
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        stops = np.concatenate([stops, [total]])

    segments = []
    for s, e in zip(starts, stops):
        seg = raw.copy().crop(tmin=s/sfreq, tmax=(e-1)/sfreq, include_tmax=True)
        segments.append(seg)

    if not segments:
        return None, n2_sec

    return concatenate_raws(segments, verbose=False), n2_sec


def epoch_and_reject_adaptive(raw: mne.io.Raw) -> tuple[mne.Epochs | None, int, int, float]:
    """
    Adaptive artifact rejection (no ceiling).
    threshold = max(90th-percentile PTP, 150 µV floor)
    """
    events = mne.make_fixed_length_events(raw, duration=EPOCH_DURATION, overlap=0.0)

    epochs_all = mne.Epochs(
        raw, events,
        tmin=0, tmax=EPOCH_DURATION - 1.0 / raw.info["sfreq"],
        baseline=None, preload=True, verbose=False,
    )

    n_created = len(epochs_all)
    if n_created == 0:
        return None, 0, 0, 0.0

    data = epochs_all.get_data()                          # (n_epochs, n_ch, n_times)
    ptp_per_epoch = data.max(axis=2) - data.min(axis=2)   # (n_epochs, n_ch)
    max_ptp = ptp_per_epoch.max(axis=1)                   # (n_epochs,)

    p90 = np.percentile(max_ptp, PTP_PERCENTILE)
    threshold = max(p90, PTP_FLOOR_UV)

    keep_mask = max_ptp <= threshold
    n_clean = int(keep_mask.sum())

    threshold_uv = threshold * 1e6
    log(f"  Adaptive PTP: p90={p90*1e6:.0f}µV -> threshold={threshold_uv:.0f}µV "
        f"| {n_created} -> {n_clean} clean ({n_created - n_clean} rejected)")

    if n_clean < 6:
        return None, n_created, n_clean, threshold_uv

    good_indices = np.where(keep_mask)[0]
    epochs_clean = epochs_all[good_indices]

    return epochs_clean, n_created, n_clean, threshold_uv


def compute_psd(epochs: mne.Epochs) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns: (freqs, psd)
      freqs : (n_freqs,)
      psd   : (n_channels, n_freqs)  -  averaged across epochs
    """
    sfreq = epochs.info["sfreq"]
    n_fft = int(4.0 * sfreq)
    spectrum = epochs.compute_psd(
        method="welch", fmin=PSD_FMIN, fmax=PSD_FMAX,
        n_fft=n_fft, n_overlap=n_fft // 2, verbose=False,
    )
    psd = spectrum.get_data().mean(axis=0)  # (n_channels, n_freqs)
    return spectrum.freqs, psd


def fit_fooof(freqs: np.ndarray, psd_ch: np.ndarray) -> dict:
    fm = SpectralModel(
        aperiodic_mode=FOOOF_APERIODIC,
        peak_width_limits=FOOOF_PEAK_WIDTH,
        max_n_peaks=FOOOF_MAX_PEAKS,
        min_peak_height=FOOOF_MIN_PEAK_HEIGHT,
        verbose=False,
    )
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


def process_one(task: dict) -> tuple[list[dict], dict]:
    """
    Process a single task.
    Returns (rows_for_csv, log_entry).
    Side-effect: populates global all_psds dict.
    """
    sub   = task["subject"]
    cond  = task["condition"]
    part  = task["part"]
    files = task["files"]
    fnames = [f.name for f in files]

    log_entry = {
        "Subject_ID": sub, "Condition": cond, "Part": part or "",
        "Files": "; ".join(fnames),
        "Status": "", "Reason": "",
        "N2_seconds": "", "N_Epochs_created": "", "N_Epochs_clean": "",
        "PTP_Threshold_uV": "",
        "Mean_R2": "", "Mean_Exponent": "",
    }

    rows: list[dict] = []

    try:
        raw = load_and_merge(files)
        dur = raw.n_times / raw.info["sfreq"]
        log(f"  Loaded: {dur:.0f}s, {len(raw.ch_names)} ch, "
            f"{raw.info['sfreq']:.0f} Hz")

        raw = rename_channels(raw)
        raw = pick_standard_channels(raw)
        raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})

        raw = preprocess(raw)

        n2_raw, n2_sec = stage_and_extract_n2(raw)
        log_entry["N2_seconds"] = f"{n2_sec:.0f}"

        if n2_raw is None:
            log_entry["Status"] = "SKIPPED"
            log_entry["Reason"] = f"N2={n2_sec:.0f}s < {MIN_N2_SECONDS}s"
            log(f"  (!!) SKIPPED  -  N2 too short ({n2_sec:.0f}s)", "WARN")
            return rows, log_entry

        epochs, n_created, n_clean, thresh_uv = epoch_and_reject_adaptive(n2_raw)
        log_entry["N_Epochs_created"] = str(n_created)
        log_entry["N_Epochs_clean"]   = str(n_clean)
        log_entry["PTP_Threshold_uV"] = f"{thresh_uv:.0f}"

        if epochs is None:
            log_entry["Status"] = "SKIPPED"
            log_entry["Reason"] = f"Only {n_clean} clean epochs (need >=6)"
            log(f"  (!!) SKIPPED  -  too few epochs ({n_clean})", "WARN")
            return rows, log_entry

        # -- PSD ----------------------------------------------------------
        freqs, psd = compute_psd(epochs)

        # Store shared frequency axis (once)
        if "freqs" not in all_psds:
            all_psds["freqs"] = freqs

        # Build PSD key  -  simple and transparent
        #   Pre:              "Sub01_Pre"
        #   Single Post:      "Sub01_Post"
        #   Multi-part Post:  "Sub06_Post_Post_part1"
        if part:
            psd_key = f"{sub}_{cond}_{part}"
        else:
            psd_key = f"{sub}_{cond}"

        all_psds[f"{psd_key}_psd"]   = psd
        all_psds[f"{psd_key}_chans"] = np.array(epochs.ch_names)

        # -- FOOOF per channel --------------------------------------------
        ch_names = epochs.ch_names
        r2_list, exp_list = [], []
        for ci, ch in enumerate(ch_names):
            res = fit_fooof(freqs, psd[ci])
            r2_list.append(res["R_Squared"])
            exp_list.append(res["Exponent"])

            rows.append({
                "Subject_ID": sub, "Condition": cond, "Part": part or "",
                "Channel": ch,
                "Exponent":   round(res["Exponent"], 4),
                "Offset":     round(res["Offset"], 4),
                "R_Squared":  round(res["R_Squared"], 4),
                "FOOOF_Error": round(res["Error"], 6),
                "N_Epochs":   n_clean,
                "PTP_Threshold_uV": round(thresh_uv, 0),
            })

        mean_r2  = np.nanmean(r2_list)
        mean_exp = np.nanmean(exp_list)
        log_entry["Status"]        = "SUCCESS"
        log_entry["Mean_R2"]       = f"{mean_r2:.4f}"
        log_entry["Mean_Exponent"] = f"{mean_exp:.4f}"
        log(f"  [OK] Done  -  {len(ch_names)} ch x {n_clean} ep, "
            f"R²={mean_r2:.3f}, Exp={mean_exp:.3f}")

    except Exception as e:
        log_entry["Status"] = "FAILED"
        log_entry["Reason"] = str(e)[:200]
        log(f"  [X] FAILED: {e}", "ERROR")
        traceback.print_exc()

    return rows, log_entry


# -- Main ---------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("  Scalp EEG Aperiodic Analysis Pipeline  -  Phase 1  (v5.1)")
    print("=" * 70)

    if not DATA_ROOT.exists():
        log(f"Data root not found: {DATA_ROOT.resolve()}", "ERROR")
        sys.exit(1)

    tasks = discover_files(DATA_ROOT)
    log(f"Discovered {len(tasks)} task(s) across subjects\n")

    all_rows:  list[dict] = []
    all_logs:  list[dict] = []

    pbar = tqdm(tasks, desc="Progress", unit="file", ncols=90, colour="green",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for task in pbar:
        sub  = task["subject"]
        cond = task["condition"]
        part = task["part"]
        label = f"{sub} {cond}" + (f"[{part}]" if part else "")
        pbar.set_description(f">> {label}")

        log(f"{'-' * 60}")
        log(f"Processing {label}  ({[f.name for f in task['files']]})")

        rows, log_entry = process_one(task)
        all_rows.extend(rows)
        all_logs.append(log_entry)

    pbar.close()

    # -- Select best Post when multiple parts exist -----------------------
    df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()

    if len(df) > 0 and "Part" in df.columns:
        final_rows = []
        for (sub, cond), grp in df.groupby(["Subject_ID", "Condition"]):
            parts = grp["Part"].unique()
            if cond == "Post" and len(parts) > 1:
                best_part = None
                best_r2   = -1
                for p in parts:
                    mean_r2 = grp[grp["Part"] == p]["R_Squared"].mean()
                    log(f"  {sub} {cond} [{p}]: mean R² = {mean_r2:.4f}")
                    if mean_r2 > best_r2:
                        best_r2 = mean_r2
                        best_part = p
                log(f"  -> Selected {best_part} (R²={best_r2:.4f})")
                final_rows.append(grp[grp["Part"] == best_part])
            else:
                final_rows.append(grp)

        df = pd.concat(final_rows, ignore_index=True)

    # -- Export ------------------------------------------------------------
    print("\n" + "=" * 70)

    log_df = pd.DataFrame(all_logs)
    log_df.to_csv(LOG_CSV, index=False)
    log(f"Processing log -> {LOG_CSV.resolve()}")

    print("\n-- Processing Log Summary --")
    for _, row in log_df.iterrows():
        icon = {"SUCCESS": "[OK]", "SKIPPED": "[~~]", "FAILED": "[X]"}.get(
            row["Status"], "?")
        part_str = f" [{row['Part']}]" if row["Part"] else ""
        reason_str = f"  -  {row['Reason']}" if row["Reason"] else ""
        n2_str = f"  N2={row['N2_seconds']}s" if row['N2_seconds'] else ""
        ep_str = (f"  ep={row['N_Epochs_clean']}"
                  if row['N_Epochs_clean'] else "")
        ptp_str = (f"  PTP={row['PTP_Threshold_uV']}µV"
                   if row['PTP_Threshold_uV'] else "")
        print(f"  {icon} {row['Subject_ID']} {row['Condition']}{part_str}: "
              f"{row['Status']}{reason_str}{n2_str}{ep_str}{ptp_str}")

    if len(df) > 0:
        out_cols = [c for c in df.columns if c != "Part"]
        df[out_cols].to_csv(OUTPUT_CSV, index=False)
        log(f"Results saved -> {OUTPUT_CSV.resolve()}  ({len(df)} rows)")

        subs_pre  = set(df[df["Condition"] == "Pre"]["Subject_ID"])
        subs_post = set(df[df["Condition"] == "Post"]["Subject_ID"])
        paired = sorted(subs_pre & subs_post)
        pre_only = sorted(subs_pre - subs_post)
        post_only = sorted(subs_post - subs_pre)
        log(f"Paired subjects: {len(paired)}  -  {paired}")
        if pre_only:
            log(f"Pre only: {len(pre_only)}  -  {pre_only}")
        if post_only:
            log(f"Post only: {len(post_only)}  -  {post_only}")

        summary = (
            df.groupby(["Condition", "Channel"])["Exponent"]
            .agg(["mean", "std", "count"]).round(4)
        )
        print("\n-- Exponent Summary --")
        print(summary.to_string())
    else:
        log("No results generated.", "WARN")

    # -- Save PSD curves (Fix 1: save all, no pruning) -------------------
    if all_psds:
        # Fix 3: embed standard channel order for unambiguous plotting
        all_psds["standard_channel_order"] = np.array(STANDARD_1020)

        np.savez(PSD_NPZ, **all_psds)
        n_psd = sum(1 for k in all_psds if k.endswith("_psd"))
        log(f"PSD curves saved -> {PSD_NPZ.resolve()}  "
            f"({n_psd} recordings, "
            f"freqs shape {all_psds.get('freqs', np.array([])).shape})")
        log(f"  Keys in .npz: {sorted(all_psds.keys())[:10]} ...")
    else:
        log("No PSD data to save.", "WARN")

    sc = log_df["Status"].value_counts()
    log(f"\nFinal: {sc.get('SUCCESS',0)} [OK]  {sc.get('SKIPPED',0)} [~~]  "
        f"{sc.get('FAILED',0)} [X]")
    log("Done.\n")


if __name__ == "__main__":
    main()
