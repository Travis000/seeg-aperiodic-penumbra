#!/usr/bin/env python3
"""
Fig1C — Breach band (20–45 Hz) relative power: paired strip plot.
Statistical confirmation that breach band % is unchanged.
v22b UPDATE: Added BF₀₁ (Bayesian paired t-test) for breach equivalence.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pingouin as pg
from shared_config import *


def plot_panel(ax, freqs, pre_psds, post_psds, paired):
    """
    Returns
    -------
    rp_pre, rp_post, t_stat, p_val
    """
    format_ax(ax, 'C')
    ax.set_title('C', fontsize=FS_PANEL_LABEL, fontweight='bold',
                 loc='left', x=-0.15, pad=8)

    fmin, fmax = BREACH_BAND
    rp_pre  = relative_power(pre_psds, freqs, fmin, fmax)
    rp_post = relative_power(post_psds, freqs, fmin, fmax)

    # Paired lines
    np.random.seed(42)
    for i in range(len(paired)):
        ax.plot([0, 1], [rp_pre[i], rp_post[i]],
                color='#888', alpha=ALPHA_PAIRED_LINE, lw=LW_PAIRED_LINE, zorder=1)

    # Jittered dots
    j0 = np.random.uniform(-0.12, 0.12, len(paired))
    j1 = np.random.uniform(-0.12, 0.12, len(paired))
    ax.scatter(0 + j0, rp_pre, s=MS_STRIP_LARGE, color=COL_PRE, **SCATTER_KW)
    ax.scatter(1 + j1, rp_post, s=MS_STRIP_LARGE, color=COL_POST, **SCATTER_KW)

    # Mean ± SEM
    draw_mean_sem(ax, 0, rp_pre, COL_PRE, width=0.2, cap=0)
    draw_mean_sem(ax, 1, rp_post, COL_POST, width=0.2, cap=0)

    # Statistics
    t_stat, p_val = stats.ttest_rel(rp_post, rp_pre)
    wilcox = stats.wilcoxon(rp_post - rp_pre)

    # Bayesian paired t-test for breach equivalence  [v22b]
    bf_breach = pg.ttest(rp_post, rp_pre, paired=True)
    bf10_breach = float(bf_breach['BF10'].values[0])
    bf01_breach = 1.0 / bf10_breach

    y_top = max(rp_pre.max(), rp_post.max()) + 0.3
    ax.plot([0, 0, 1, 1], [y_top, y_top + 0.15, y_top + 0.15, y_top],
            color='#555', lw=LW_BRACKET)
    wp = wilcox.pvalue
    p_label = f'n.s. (p = {wp:.2f})' if wp >= 0.05 else f'p = {wp:.3f}'
    ax.text(0.5, y_top + 0.25, p_label, ha='center',
            fontsize=FS_PVAL_BRACKET + 0.5, fontstyle='italic', color='#444')

    stat_box(ax,
             f'N = {len(paired)} paired\n'
             f'Pre:  {rp_pre.mean():.2f} \u00b1 {rp_pre.std(ddof=1):.2f}%\n'
             f'Post: {rp_post.mean():.2f} \u00b1 {rp_post.std(ddof=1):.2f}%\n'
             f'Wilcoxon p = {wilcox.pvalue:.3f}\n'
             f'BF$_{{01}}$ = {bf01_breach:.2f}',
             x=0.96, y=0.04)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Pre', 'Post'], fontsize=FS_AXIS_LABEL)
    ax.set_ylabel('Breach band relative power (%)\n(20–45 Hz / Total)',
                  fontsize=FS_AXIS_LABEL - 1)
    ax.set_xlim(-0.5, 1.5)

    return rp_pre, rp_post, t_stat, p_val, bf01_breach


# ── Standalone ──
if __name__ == "__main__":
    setup_style()
    freqs, pre, post, paired = load_paired_psds()
    fig, ax = plt.subplots(figsize=(4, 5))
    rp_pre, rp_post, t_stat, p_val, bf01 = plot_panel(ax, freqs, pre, post, paired)
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / 'fig1c_breach.png')
    fig.savefig(SCRIPT_DIR / 'fig1c_breach.pdf')

    # v22b: Full breach statistics
    wilcox = stats.wilcoxon(rp_post - rp_pre)
    print(f"Fig 1C — Breach band statistics")
    print(f"  N = {len(paired)} paired subjects")
    print(f"  Pre:  {rp_pre.mean():.2f} ± {rp_pre.std(ddof=1):.2f}%")
    print(f"  Post: {rp_post.mean():.2f} ± {rp_post.std(ddof=1):.2f}%")
    print(f"  Paired t-test: t = {t_stat:.3f}, p = {p_val:.3f}")
    print(f"  Wilcoxon: p = {wilcox.pvalue:.3f}")
    print(f"  BF01 = {bf01:.3f}")
    if bf01 > 3:
        print(f"  → moderate evidence for null ✓")
    elif bf01 > 1:
        print(f"  → anecdotal evidence for null")
    else:
        print(f"  → evidence favours H1")
    plt.close()
