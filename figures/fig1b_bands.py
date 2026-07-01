#!/usr/bin/env python3
"""
Fig1B — Relative power by frequency band (paired Pre vs Post bar chart).
Confirms no band-specific power redistribution after thermocoagulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from shared_config import *


def plot_panel(ax, freqs, pre_psds, post_psds):
    """
    Parameters
    ----------
    ax        : matplotlib Axes
    freqs     : (n_freqs,)
    pre_psds  : (n_subjects, n_freqs)
    post_psds : (n_subjects, n_freqs)
    """
    format_ax(ax, 'B')

    band_names = list(BANDS.keys())
    x = np.arange(len(band_names))
    width = 0.35

    pre_means, post_means = [], []
    pre_sems, post_sems = [], []
    p_vals = []

    for name, (fmin, fmax) in BANDS.items():
        rp_pre  = relative_power(pre_psds, freqs, fmin, fmax)
        rp_post = relative_power(post_psds, freqs, fmin, fmax)
        pre_means.append(rp_pre.mean())
        post_means.append(rp_post.mean())
        pre_sems.append(rp_pre.std(ddof=1)  / np.sqrt(len(rp_pre)))
        post_sems.append(rp_post.std(ddof=1) / np.sqrt(len(rp_post)))
        _, p = stats.ttest_rel(rp_post, rp_pre)
        p_vals.append(p)

    ax.bar(x - width/2, pre_means, width, yerr=pre_sems,
           color=COL_PRE, alpha=ALPHA_BAR, edgecolor='#333', linewidth=0.5,
           capsize=3, error_kw={'lw': 1}, label='Pre', zorder=2)
    ax.bar(x + width/2, post_means, width, yerr=post_sems,
           color=COL_POST, alpha=ALPHA_BAR, edgecolor='#333', linewidth=0.5,
           capsize=3, error_kw={'lw': 1}, label='Post', zorder=2)

    for i, p in enumerate(p_vals):
        y_max = max(pre_means[i] + pre_sems[i], post_means[i] + post_sems[i])
        label = 'n.s.' if p >= 0.05 else f'p={p:.3f}'
        ax.text(x[i], y_max + 1.0, label, ha='center',
                fontsize=FS_ANNOTATION, fontstyle='italic', color='#666')

    ax.set_xticks(x)
    ax.set_xticklabels(band_names, fontsize=8)
    ax.set_ylabel('Relative Power (%)', fontsize=FS_AXIS_LABEL - 1)
    ax.legend(fontsize=FS_LEGEND, loc='upper right', framealpha=0.9)


# ── Standalone ──
if __name__ == "__main__":
    setup_style()
    freqs, pre, post, paired = load_paired_psds()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    plot_panel(ax, freqs, pre, post)
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / 'fig1b_bands.png')
    fig.savefig(SCRIPT_DIR / 'fig1b_bands.pdf')
    print(f"Saved fig1b  (N={len(paired)})")
    plt.close()
