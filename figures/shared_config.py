#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
shared_config.py — Unified Style & Data Loaders for All Panels
=============================================================================
Provides:
  1. Color palette, font sizes, line widths (journal-grade consistency)
  2. Path auto-detection (DATA_ROOT → SCRIPT_DIR fallback)
  3. Reusable data loading functions

Import in every panel script:
    from shared_config import *
=============================================================================
"""

from __future__ import annotations
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from scipy import stats
from scipy.integrate import trapezoid

# ═══════════════════════════════════════════════════════════════════
#  PATH CONFIGURATION
#  DATA_ROOT must point to the folder holding the derived result files
#  (scalp_results.csv, seeg_contact_results.csv, engel_phase.xlsx, ...).
#  Those files are NOT distributed with this code release (they are
#  derived from patient recordings); see data/README.md and the paper's
#  Data availability statement. Set the environment variable APERIODIC_DATA
#  or edit the default below.
# ═══════════════════════════════════════════════════════════════════
SCRIPT_DIR = Path(__file__).resolve().parent          # figures/ folder
DATA_ROOT  = Path(os.environ.get("APERIODIC_DATA", SCRIPT_DIR.parent / "data"))

def _resolve(filename: str) -> Path:
    """Try DATA_ROOT first, then SCRIPT_DIR."""
    for root in [DATA_ROOT, SCRIPT_DIR]:
        p = root / filename
        if p.exists():
            return p
    # Return DATA_ROOT path even if missing (let caller handle error)
    return DATA_ROOT / filename

def _find_roi_file() -> Path:
    for root in [DATA_ROOT, SCRIPT_DIR]:
        for pattern in ["ROI__Inferred_Side.csv", "ROI  Inferred_Side.csv",
                        "ROI_Inferred_Side.csv", "ROI*.csv"]:
            matches = glob.glob(str(root / pattern))
            if matches:
                return Path(matches[0])
    raise FileNotFoundError("No ROI file found")


# ═══════════════════════════════════════════════════════════════════
#  COLOR PALETTE  (consistent across every panel)
# ═══════════════════════════════════════════════════════════════════
COL_PRE    = '#2166AC'     # blue — pre-surgery
COL_POST   = '#B2182B'     # red  — post-surgery
COL_GOOD   = '#4DAF4A'     # green — Engel I–II
COL_POOR   = '#E41A1C'     # red   — Engel III–IV
COL_SOZ    = '#B2182B'     # red   — seizure onset zone
COL_NSOZ   = '#2166AC'     # blue  — non-SOZ
COL_IPSI   = '#FF7F00'     # orange — ipsilateral
COL_CONTRA = '#984EA3'     # purple — contralateral
COL_BG     = '#FAFAFA'     # panel background

# ═══════════════════════════════════════════════════════════════════
#  TYPOGRAPHY & LINE STANDARDS  (顶刊统一规范)
# ═══════════════════════════════════════════════════════════════════
DPI = 300

# Font sizes
FS_PANEL_LABEL  = 14       # A / B / C panel letter
FS_AXIS_LABEL   = 11       # xlabel / ylabel
FS_AXIS_TICK    = 9        # tick numbers
FS_LEGEND       = 8        # legend entries
FS_STAT_BOX     = 7        # statistics annotation boxes
FS_PVAL_BRACKET = 9        # p-value on significance brackets
FS_ANNOTATION   = 7        # small annotation text

# Line widths
LW_MEAN_BAR     = 2.5      # group mean horizontal bar
LW_SEM_BAR      = 1.5      # SEM whisker
LW_PAIRED_LINE  = 0.7      # individual paired connection
LW_BRACKET      = 0.8      # significance bracket
LW_REGRESSION   = 1.5      # regression / trend line
LW_PSD_CURVE    = 2.0      # PSD grand-average

# Marker sizes
MS_INDIVIDUAL   = 35       # individual subject dots (scatter s=)
MS_INDIVIDUAL_SM = 18      # smaller dots (e.g. Panel B interaction)
MS_STRIP_LARGE  = 40       # strip plot dots

# Scatter common kwargs
SCATTER_KW = dict(edgecolors='white', linewidths=0.5, zorder=3)

# Alpha
ALPHA_PAIRED_LINE = 0.35
ALPHA_SEM_FILL    = 0.15
ALPHA_BAR         = 0.75
ALPHA_SCATTER     = 0.6


def setup_style():
    """Apply global matplotlib style — call once at script start."""
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 9,
        'axes.labelsize': FS_AXIS_LABEL,
        'axes.titlesize': FS_PANEL_LABEL,
        'xtick.labelsize': FS_AXIS_TICK,
        'ytick.labelsize': FS_AXIS_TICK,
        'legend.fontsize': FS_LEGEND,
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def format_ax(ax, title_letter: str = '', bg: bool = True):
    """Standard axis formatting for every panel."""
    if bg:
        ax.set_facecolor(COL_BG)
    ax.tick_params(labelsize=FS_AXIS_TICK)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if title_letter:
        ax.set_title(title_letter, fontsize=FS_PANEL_LABEL,
                     fontweight='bold', loc='left', x=-0.12, pad=8)


def add_bracket(ax, x0, x1, y, p, dy=0.02, fontsize=FS_PVAL_BRACKET):
    """Draw a significance bracket between x0 and x1 at height y."""
    ax.plot([x0, x0, x1, x1], [y, y + dy, y + dy, y],
            color='#555', lw=LW_BRACKET)
    if p < 0.001:
        txt = f'p = {p:.1e}'
    elif p < 0.01:
        txt = f'**p = {p:.4f}'
    elif p < 0.05:
        txt = f'*p = {p:.3f}'
    elif p < 0.10:
        txt = f'†p = {p:.3f}'
    else:
        txt = f'n.s. (p = {p:.2f})'
    ax.text((x0 + x1) / 2, y + dy * 1.5, txt,
            ha='center', fontsize=fontsize, fontstyle='italic', color='#444')


def draw_mean_sem(ax, x, values, color, width=0.15, cap=0.06):
    """Draw a mean ± SEM bar at position x."""
    m = values.mean()
    sem = values.std(ddof=1) / np.sqrt(len(values))
    ax.plot([x - width, x + width], [m, m], color=color, lw=LW_MEAN_BAR, zorder=4)
    ax.plot([x, x], [m - sem, m + sem], color=color, lw=LW_SEM_BAR, zorder=4)
    if cap > 0:
        ax.plot([x - cap, x + cap], [m - sem, m - sem],
                color=color, lw=LW_SEM_BAR, zorder=4)
        ax.plot([x - cap, x + cap], [m + sem, m + sem],
                color=color, lw=LW_SEM_BAR, zorder=4)


def stat_box(ax, text, x=0.98, y=0.03, ha='right', va='bottom', fontsize=FS_STAT_BOX):
    """Place a standard statistics box on the panel (journal-style: light fill,
    soft rounded border, proportional sans-serif — not the raw monospace look)."""
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=fontsize, family='sans-serif', color='#1A1A1A',
            linespacing=1.35, zorder=6,
            bbox=dict(boxstyle='round,pad=0.42', fc='#F6F6F4', ec='#B9B9B9',
                      lw=0.6, alpha=0.96))


# ═══════════════════════════════════════════════════════════════════
#  CHANNEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════
LEFT_CHS   = ['Fp1', 'F3', 'F7', 'C3', 'P3', 'P7', 'T7', 'O1']
RIGHT_CHS  = ['Fp2', 'F4', 'F8', 'C4', 'P4', 'P8', 'T8', 'O2']
MIDLINE    = ['Fz', 'Cz', 'Pz']
STANDARD_1020 = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
    'Fz', 'Cz', 'Pz',
]

CHAN_XY = {
    'Fp1': (-0.31, 0.95), 'Fp2': (0.31, 0.95),
    'F7':  (-0.81, 0.59), 'F3':  (-0.39, 0.58), 'Fz': (0.0, 0.58),
    'F4':  (0.39, 0.58),  'F8':  (0.81, 0.59),
    'T7':  (-1.0, 0.0),   'C3':  (-0.49, 0.0),  'Cz': (0.0, 0.0),
    'C4':  (0.49, 0.0),   'T8':  (1.0, 0.0),
    'P7':  (-0.81, -0.59),'P3':  (-0.39, -0.58),'Pz': (0.0, -0.58),
    'P4':  (0.39, -0.58), 'P8':  (0.81, -0.59),
    'O1':  (-0.31, -0.95),'O2':  (0.31, -0.95),
}

MIRROR_MAP = {
    'Fp1': 'Fp2', 'Fp2': 'Fp1', 'F3': 'F4', 'F4': 'F3',
    'F7': 'F8', 'F8': 'F7', 'C3': 'C4', 'C4': 'C3',
    'P3': 'P4', 'P4': 'P3', 'P7': 'P8', 'P8': 'P7',
    'T7': 'T8', 'T8': 'T7', 'O1': 'O2', 'O2': 'O1',
    'Fz': 'Fz', 'Cz': 'Cz', 'Pz': 'Pz',
}

BANDS = {
    'Delta\n(1–4)':    (1, 4),
    'Theta\n(4–8)':    (4, 8),
    'Alpha\n(8–13)':   (8, 13),
    'Beta\n(13–30)':   (13, 30),
    'Breach\n(20–45)': (20, 45),
}
BREACH_BAND = (20, 45)

# SEEG / Fig2 constants
PAIRED_SUBJECTS = [
    'Sub01','Sub02','Sub04','Sub05','Sub06','Sub07','Sub08','Sub09',
    'Sub11','Sub13','Sub14','Sub15','Sub17','Sub19','Sub20','Sub21'
]
MIN_EXPONENT    = 0.5
MAX_DISTANCE_MM = 63.0
CONTACT_SPACING_MM = 3.5


# ═══════════════════════════════════════════════════════════════════
#  DATA LOADERS
# ═══════════════════════════════════════════════════════════════════

def load_engel_phase(filepath=None) -> pd.DataFrame:
    """Load engel_phase.xlsx → DataFrame[Subject, Engel, Outcome]."""
    if filepath is None:
        filepath = _resolve("engel_phase.xlsx")
    df = pd.read_excel(filepath)
    records = []
    for _, row in df.iterrows():
        raw_sub = str(row.iloc[0]).strip()
        sub = ('Sub' + raw_sub[3:].zfill(2)
               if raw_sub.lower().startswith('sub') else raw_sub)
        e = str(row.iloc[2]).strip()
        if e.upper() in ('', 'NAN', 'NONE'):
            continue
        eu = e.upper()
        if   eu.startswith('IV'):   outcome = 'Poor'
        elif eu.startswith('III'):  outcome = 'Poor'
        elif eu.startswith('II'):   outcome = 'Good'
        elif eu.startswith('I'):    outcome = 'Good'
        else:                       outcome = 'Poor'
        records.append({'Subject': sub, 'Engel': e, 'Outcome': outcome})
    return pd.DataFrame(records)


def load_scalp_results(filepath=None) -> pd.DataFrame:
    """Load scalp_results.csv."""
    if filepath is None:
        filepath = _resolve("scalp_results.csv")
    return pd.read_csv(filepath)


def load_roi(filepath=None) -> pd.DataFrame:
    """Load ROI / Inferred Side CSV (auto-detects naming)."""
    if filepath is None:
        filepath = _find_roi_file()
    return pd.read_csv(filepath)


def load_seeg_results(filepath=None) -> pd.DataFrame:
    """Load seeg_contact_results.csv."""
    if filepath is None:
        filepath = _resolve("seeg_contact_results.csv")
    return pd.read_csv(filepath)


def get_paired_subjects(df_scalp: pd.DataFrame) -> list[str]:
    """Return sorted list of subjects with both Pre and Post data."""
    pre  = set(df_scalp[df_scalp.Condition == 'Pre'].Subject_ID)
    post = set(df_scalp[df_scalp.Condition == 'Post'].Subject_ID)
    return sorted(pre & post)


def load_paired_psds(psd_npz=None, csv_file=None):
    """Load PSD .npz and return (freqs, pre_psds, post_psds, paired_list)."""
    if psd_npz is None:
        psd_npz = _resolve("scalp_psd_curves.npz")
    if csv_file is None:
        csv_file = _resolve("scalp_results.csv")
    psd_data = np.load(psd_npz, allow_pickle=True)
    df = pd.read_csv(csv_file)
    freqs = psd_data['freqs']
    paired = get_paired_subjects(df)

    pre_psds, post_psds, valid = [], [], []
    for sub in paired:
        pre_key = f"{sub}_Pre_psd"
        if pre_key not in psd_data:
            continue
        pre_psd = np.mean(psd_data[pre_key], axis=0)

        post_key = f"{sub}_Post_psd"
        post_psd = None
        if post_key in psd_data:
            post_psd = np.mean(psd_data[post_key], axis=0)
        else:
            for k in sorted(psd_data.files):
                if k.startswith(f"{sub}_Post") and k.endswith("_psd"):
                    post_psd = np.mean(psd_data[k], axis=0)
                    break
        if post_psd is None:
            continue
        pre_psds.append(pre_psd)
        post_psds.append(post_psd)
        valid.append(sub)

    return freqs, np.array(pre_psds), np.array(post_psds), valid


def get_subject_level_data() -> pd.DataFrame:
    """Merge scalp + engel + ROI → subject-level DataFrame for Fig3/4."""
    df    = load_scalp_results()
    engel = load_engel_phase()
    roi   = load_roi()
    paired = get_paired_subjects(df)

    records = []
    for sub in paired:
        pre_exp  = df[(df.Subject_ID == sub) & (df.Condition == 'Pre')].Exponent.mean()
        post_exp = df[(df.Subject_ID == sub) & (df.Condition == 'Post')].Exponent.mean()
        eng = engel[engel.Subject == sub]
        side = roi[roi.Subject == sub]
        if len(eng) == 0 or len(side) == 0:
            continue
        records.append({
            'Subject': sub,
            'Pre': pre_exp, 'Post': post_exp,
            'Delta': post_exp - pre_exp,
            'Engel': eng.iloc[0]['Engel'],
            'Outcome': eng.iloc[0]['Outcome'],
            'Side': side.iloc[0]['Inferred Side'],
            'ROI': side.iloc[0]['ROI'],
        })
    return pd.DataFrame(records)


def load_scalp_midline_delta() -> dict:
    """Compute per-subject midline ΔExponent from scalp_results."""
    df = load_scalp_results()
    mid = df[df.Channel.isin(MIDLINE)]
    pre  = mid[mid.Condition == 'Pre'].groupby('Subject_ID')['Exponent'].mean()
    post = mid[mid.Condition == 'Post'].groupby('Subject_ID')['Exponent'].mean()
    common = pre.index.intersection(post.index)
    return {s: post[s] - pre[s] for s in common}


def cohens_d_paired(pre, post):
    """Cohen's d for paired samples (sample SD, ddof=1)."""
    diff = np.asarray(post, dtype=float) - np.asarray(pre, dtype=float)
    sd = diff.std(ddof=1)
    return diff.mean() / sd if sd > 0 else 0.0


def relative_power(psds, freqs, fmin, fmax):
    """Relative power in [fmin,fmax] as % of total (1–45 Hz)."""
    mask_band  = (freqs >= fmin) & (freqs <= fmax)
    mask_total = (freqs >= 1) & (freqs <= 45)
    rp = np.zeros(len(psds))
    for i in range(len(psds)):
        total = trapezoid(psds[i][mask_total], freqs[mask_total])
        band  = trapezoid(psds[i][mask_band],  freqs[mask_band])
        rp[i] = (band / total) * 100 if total > 0 else 0
    return rp
