# -*- coding: utf-8 -*-
"""
fig5_panels.py — Relapse-timeline panels refactored into reusable plot_panel form
so they can be composed into the merged clinical Figure 4 (panels C and D).

Refactored from fig5_relapse_prediction.py (standalone) WITHOUT changing any
statistic: same swimmer plot and same "predictive vs concurrent" relapse-status
analysis (Panel B: relapsed-@EEG subgroup good vs poor, t-test + Cohen's d).

Exposes:
    load_relapse_data()           -> sorted DataFrame M
    plot_swimmer(ax, M, letter)   -> Panel: per-patient timeline (RF-TC / relapse / EEG window)
    plot_relapse(ax, M, letter)   -> Panel: ΔExponent by relapse status × outcome
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats
from shared_config import (load_scalp_results, get_paired_subjects,
                           COL_GOOD, COL_POOR, COL_PRE, SCRIPT_DIR, DATA_ROOT,
                           FS_AXIS_LABEL, FS_LEGEND, FS_ANNOTATION,
                           FS_PANEL_LABEL, format_ax)

REL = DATA_ROOT / 'relapse_timeline.xlsx'
ENGEL = DATA_ROOT / 'engel_phase.xlsx'


def _engel_num(e):
    e = str(e).strip().upper()
    for k in ('IV', 'III', 'II', 'I'):
        if e.startswith(k):
            return k
    return None


def load_relapse_data():
    """Build the per-patient relapse/outcome/ΔExponent table (sorted for swimmer)."""
    rel = pd.read_excel(REL)
    rel['Subject'] = rel['SubID'].astype(str).str.replace('-', '', regex=False).str.strip()
    statecol = [c for c in rel.columns if 'post-EEG' in c][0]
    mocol = [c for c in rel.columns if '复发' in c][0]
    rel['relapsed'] = rel[statecol].map({'有': 1, '无': 0})
    rel['relapse_mo'] = pd.to_numeric(rel[mocol], errors='coerce')

    eng = pd.read_excel(ENGEL)
    eng['Subject'] = eng.iloc[:, 0].astype(str).str.strip().apply(
        lambda s: 'Sub' + s[3:].zfill(2) if s.lower().startswith('sub') else s)
    eng['fu_mo'] = pd.to_numeric(eng['Follow-up(mo)'], errors='coerce')
    eng['en'] = eng['engel'].apply(_engel_num)

    scalp = load_scalp_results()
    paired = get_paired_subjects(scalp)
    recs = []
    for sub in paired:
        pre = scalp[(scalp.Subject_ID == sub) & (scalp.Condition == 'Pre')].Exponent.mean()
        post = scalp[(scalp.Subject_ID == sub) & (scalp.Condition == 'Post')].Exponent.mean()
        r = rel[rel.Subject == sub]
        e = eng[eng.Subject == sub]
        if len(r) == 0 or len(e) == 0:
            continue
        recs.append(dict(Subject=sub, en=e.iloc[0]['en'], fu=e.iloc[0]['fu_mo'],
                         relapsed=int(r.iloc[0]['relapsed']),
                         rmo=r.iloc[0]['relapse_mo'], delta=post - pre))
    M = pd.DataFrame(recs)
    M['good'] = M.en.isin(['I', 'II'])
    M['order_grp'] = M.good.map({True: 0, False: 1})
    M['rmo_sort'] = M.rmo.fillna(99)
    M = M.sort_values(['order_grp', 'rmo_sort']).reset_index(drop=True)
    return M


def plot_swimmer(ax, M, letter='C'):
    """Panel: per-patient post-op timeline (swimmer plot).

    ID / Engel / ΔExp are placed in fixed aligned columns to the right of the
    time axis (instead of ragged at each lane's end) so they stop competing
    with the timeline; faint leaders connect each lane to its columns.
    """
    format_ax(ax, letter, bg=False)
    n = len(M)
    x_id, x_engel, x_delta = -1.4, 21.0, 23.0

    ax.axvspan(6, 8, color='#FFD700', alpha=0.18, zorder=0)
    ax.text(7, n + 0.15, 'post-op EEG\n(6–8 mo)', ha='center', va='bottom',
            fontsize=6.5, color='#9a7d00')

    # column headers
    for xc, lab, al in [(x_id, 'ID', 'right'), (x_engel, 'Engel', 'center'),
                        (x_delta, 'ΔExp', 'center')]:
        ax.text(xc, n + 0.15, lab, ha=al, va='bottom', fontsize=6.5,
                color='#888', fontweight='bold')

    for i, row in M.iterrows():
        y = n - 1 - i
        gcol = COL_GOOD if row.good else COL_POOR
        ax.plot([0, row.fu], [y, y], color='#bbb', lw=1.2, zorder=1)
        if row.fu < x_engel - 1.8:                       # faint leader to columns
            ax.plot([row.fu, x_engel - 1.4], [y, y], color='#e3e3e3',
                    lw=0.5, ls=(0, (1, 2)), zorder=1)
        ax.scatter(0, y, marker='>', s=42, color='#333', zorder=3)
        if np.isfinite(row.rmo):
            ax.scatter(row.rmo, y, marker='x', s=55, color='#d62728', lw=1.8, zorder=4)
        ax.text(x_id, y, row.Subject[3:], va='center', ha='right',
                fontsize=6.5, color=gcol)
        ax.text(x_engel, y, row.en, va='center', ha='center',
                fontsize=7, color=gcol, fontweight='bold')
        dcol = COL_PRE if row.delta < 0 else COL_POOR
        ax.text(x_delta, y, f"{row.delta:+.2f}", va='center', ha='center',
                fontsize=6.2, color=dcol)

    ax.set_xlim(-3, 24.5)
    ax.set_ylim(-0.8, n + 0.9)
    ax.set_yticks([])
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.spines['bottom'].set_bounds(0, 20)               # axis stops before columns
    ax.set_xlabel('Months after thermocoagulation', fontsize=FS_AXIS_LABEL)
    leg = [Line2D([0], [0], marker='>', color='w', markerfacecolor='#333', markersize=8, label='RF-TC (month 0)'),
           Line2D([0], [0], marker='x', color='#d62728', markersize=8, label='first relapse', lw=0),
           Line2D([0], [0], marker='s', color='w', markerfacecolor='#FFD700', markersize=9, label='post-op EEG (6–8 mo)', alpha=0.7)]
    ax.legend(handles=leg, fontsize=FS_LEGEND, loc='upper center',
              bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False,
              handletextpad=0.4, columnspacing=1.4)


def plot_relapse(ax, M, letter='D'):
    """Panel: ΔExponent by relapse status (@EEG) × outcome; relapsed-subgroup test."""
    format_ax(ax, letter)
    ax.axhline(0, color='#aaa', ls=':', lw=0.8)
    np.random.seed(1)
    seen = set()
    for gi, (lab, rv) in enumerate([('No relapse\n@EEG', 0), ('Relapsed\n@EEG', 1)]):
        for good, col, off in [(True, COL_GOOD, -0.13), (False, COL_POOR, 0.13)]:
            v = M[(M.relapsed == rv) & (M.good == good)].delta.values
            if len(v) == 0:
                continue
            key = 'Good (I–II)' if good else 'Poor (III–IV)'
            lbl = key if key not in seen else None
            seen.add(key)
            x = gi + off + np.random.uniform(-0.04, 0.04, len(v))
            ax.scatter(x, v, s=45, color=col, edgecolors='white', lw=0.5, zorder=3, label=lbl)
            ax.plot([gi + off - 0.08, gi + off + 0.08], [v.mean()] * 2, color=col, lw=2.5, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No relapse\n@EEG', 'Relapsed\n@EEG'], fontsize=9)
    ax.set_ylabel('Aperiodic ΔExponent (Post − Pre)', fontsize=FS_AXIS_LABEL)
    ax.legend(fontsize=FS_LEGEND, loc='upper right', framealpha=0.9)

    rg = M[M.relapsed == 1]
    gd, pr = rg[rg.good].delta, rg[~rg.good].delta
    t, p = stats.ttest_ind(gd, pr)
    dpool = np.sqrt(((len(gd) - 1) * gd.var() + (len(pr) - 1) * pr.var()) / (len(gd) + len(pr) - 2))
    d = (gd.mean() - pr.mean()) / dpool if dpool > 0 else np.nan
    y0 = M.delta.min() - 0.03
    ax.plot([1 - 0.13, 1 + 0.13], [y0, y0], color='#555', lw=0.8)
    ax.text(1, y0 - 0.015, f"p = {p:.3f}, d = {d:.2f}", ha='center', va='top',
            fontsize=7.5, fontstyle='italic', color='#333')
    return dict(good_n=len(gd), poor_n=len(pr), good_delta=gd.mean(),
                poor_delta=pr.mean(), p=p, d=d)


# ── Standalone quick test ──
if __name__ == "__main__":
    from shared_config import setup_style, DPI
    import matplotlib.pyplot as plt
    setup_style()
    M = load_relapse_data()
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 6))
    plot_swimmer(a, M, 'A')
    res = plot_relapse(b, M, 'B')
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / 'fig5_panels_test.png', dpi=DPI)
    print('Saved fig5_panels_test;', res)
    plt.close()
