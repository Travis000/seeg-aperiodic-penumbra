#!/usr/bin/env python3
"""
Fig3B — ΔExponent by Laterality × Outcome interaction plot.
Ipsilateral / Contralateral / Midline × Good / Poor.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from shared_config import *


def plot_panel(ax, df_scalp, engel, roi, paired):
    """
    Parameters
    ----------
    ax       : matplotlib Axes
    df_scalp : scalp_results DataFrame
    engel    : DataFrame[Subject, Engel, Outcome]
    roi      : DataFrame[Subject, Inferred Side, ROI]
    paired   : list of Subject_IDs
    """
    format_ax(ax, 'B')

    # Build lateralised records
    records = []
    for sub in paired:
        side = roi[roi.Subject == sub]['Inferred Side'].values[0]
        outcome = engel[engel.Subject == sub]['Outcome'].values[0]
        for cond in ['Pre', 'Post']:
            sub_df = df_scalp[(df_scalp.Subject_ID == sub) & (df_scalp.Condition == cond)]
            if side == 'Left':
                ipsi   = sub_df[sub_df.Channel.isin(LEFT_CHS)].Exponent.mean()
                contra = sub_df[sub_df.Channel.isin(RIGHT_CHS)].Exponent.mean()
            else:
                ipsi   = sub_df[sub_df.Channel.isin(RIGHT_CHS)].Exponent.mean()
                contra = sub_df[sub_df.Channel.isin(LEFT_CHS)].Exponent.mean()
            mid = sub_df[sub_df.Channel.isin(MIDLINE)].Exponent.mean()
            records.append({
                'Subject': sub, 'Condition': cond, 'Outcome': outcome,
                'Ipsilateral': ipsi, 'Contralateral': contra, 'Midline': mid,
            })
    rdf = pd.DataFrame(records)

    # Compute deltas
    deltas = []
    for sub in paired:
        outcome = rdf[rdf.Subject == sub].Outcome.values[0]
        for region in ['Ipsilateral', 'Contralateral', 'Midline']:
            pre_val  = rdf[(rdf.Subject == sub) & (rdf.Condition == 'Pre')][region].values[0]
            post_val = rdf[(rdf.Subject == sub) & (rdf.Condition == 'Post')][region].values[0]
            deltas.append({
                'Subject': sub, 'Outcome': outcome,
                'Region': region, 'Delta': post_val - pre_val,
            })
    ddf = pd.DataFrame(deltas)

    regions = ['Ipsilateral', 'Contralateral', 'Midline']
    x_base = np.arange(len(regions))
    width = 0.35
    offset_good = -width / 2
    offset_poor = width / 2

    ax.axhline(0, color='#AAA', lw=0.8, ls=':', zorder=0)

    # ── Step 1: collect raw p-values for all 6 tests ─────────────
    raw_pvals = []   # order: Good×3 then Poor×3
    means_all = {}
    sems_all  = {}
    vals_all  = {}
    for outcome in ['Good', 'Poor']:
        grp = ddf[ddf.Outcome == outcome]
        for region in regions:
            vals = grp[grp.Region == region]['Delta'].dropna().values
            _, p = stats.ttest_1samp(vals, 0)
            raw_pvals.append(p)
            key = (outcome, region)
            means_all[key] = vals.mean()
            sems_all[key]  = vals.std(ddof=1) / np.sqrt(len(vals))
            vals_all[key]  = vals

    # ── Step 2: BH-FDR correction across all 6 tests ─────────────
    _, pvals_fdr, _, _ = multipletests(raw_pvals, method='fdr_bh')
    # Map back: index order is Good×[Ipsi, Contra, Mid], Poor×[Ipsi, Contra, Mid]
    fdr_map = {}
    idx = 0
    for outcome in ['Good', 'Poor']:
        for region in regions:
            fdr_map[(outcome, region)] = pvals_fdr[idx]
            idx += 1

    # ── DIAGNOSTIC: print all raw + FDR p-values for manuscript verification ──
    print("\n  ── Fig3B FDR Diagnostics ──")
    print(f"  {'Outcome':<8} {'Region':<16} {'raw_p':>12} {'FDR_p':>12} {'Stars':>6}")
    idx = 0
    for outcome in ['Good', 'Poor']:
        for region in regions:
            rp = raw_pvals[idx]
            fp = pvals_fdr[idx]
            if   fp < 0.001: stars = '***'
            elif fp < 0.01:  stars = '**'
            elif fp < 0.05:  stars = '*'
            else:            stars = 'n.s.'
            print(f"  {outcome:<8} {region:<16} {rp:>12.6f} {fp:>12.6f} {stars:>6}")
            idx += 1
    print("  ── NOTE: Manuscript reports Good×Midline FDR p = 0.0016 → should be **")
    print("  ── If computed FDR p < 0.001, manuscript number needs updating ──\n")

    # ── Step 3: draw bars + dots + FDR-corrected stars ───────────
    for outcome, offset, color, marker in [
        ('Good', offset_good, COL_GOOD, 'o'),
        ('Poor', offset_poor, COL_POOR, 's'),
    ]:
        means = [means_all[(outcome, r)] for r in regions]
        sems  = [sems_all[(outcome, r)]  for r in regions]

        # Estimation plot: individual ΔExponent dots + mean ± SEM (replaces bars)
        np.random.seed(hash(outcome) % 2**31)
        for i, region in enumerate(regions):
            vals = np.asarray(vals_all[(outcome, region)], dtype=float)
            jitter = np.random.uniform(-0.07, 0.07, len(vals))
            ax.scatter(x_base[i] + offset + jitter, vals,
                       s=MS_INDIVIDUAL_SM, color=color, edgecolors='white',
                       linewidths=0.3, alpha=ALPHA_SCATTER, zorder=3, marker=marker,
                       label=outcome if i == 0 else None)
            draw_mean_sem(ax, x_base[i] + offset, vals, color, width=0.13, cap=0.05)

        # FDR-corrected p-value stars
        for i, region in enumerate(regions):
            p_fdr = fdr_map[(outcome, region)]
            m, s  = means_all[(outcome, region)], sems_all[(outcome, region)]
            y_pos = m + s + 0.015 if m > 0 else m - s - 0.015
            va    = 'bottom' if m > 0 else 'top'
            if   p_fdr < 0.001: label = '***'
            elif p_fdr < 0.01:  label = '**'
            elif p_fdr < 0.05:  label = '*'
            else:               label = 'n.s.'
            ax.text(x_base[i] + offset, y_pos, label,
                    ha='center', va=va, fontsize=FS_ANNOTATION,
                    color='#555', fontstyle='italic')

    ax.set_xticks(x_base)
    ax.set_xticklabels(regions, fontsize=FS_AXIS_TICK)
    ax.set_ylabel('ΔExponent (Post − Pre)', fontsize=FS_AXIS_LABEL - 1)
    ax.legend(fontsize=FS_LEGEND, loc='lower left', framealpha=0.9, edgecolor='#CCC')


# ── Standalone ──
if __name__ == "__main__":
    setup_style()
    df    = load_scalp_results()
    engel = load_engel_phase()
    roi   = load_roi()
    paired = get_paired_subjects(df)
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_panel(ax, df, engel, roi, paired)
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / 'fig3b_laterality.png')
    fig.savefig(SCRIPT_DIR / 'fig3b_laterality.pdf')
    print(f"Saved fig3b  (N={len(paired)})")
    plt.close()
