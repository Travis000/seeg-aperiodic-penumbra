#!/usr/bin/env python3
"""
Fig3A — Whole-brain mean aperiodic exponent: raincloud plot (Pre vs Post).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
from shared_config import *


def _half_violin(ax, data, position, color, side='left', width=0.3):
    """Draw a half violin (kernel density) on one side."""
    if len(data) < 3:
        return
    kde = gaussian_kde(data, bw_method=0.35)
    y_range = np.linspace(data.min() - 0.1, data.max() + 0.1, 200)
    density = kde(y_range)
    density = density / density.max() * width

    if side == 'left':
        verts = (list(zip(position - density, y_range))
                 + [(position, y_range[-1]), (position, y_range[0])])
    else:
        verts = (list(zip(position + density, y_range))
                 + [(position, y_range[-1]), (position, y_range[0])])

    poly = plt.Polygon(verts, facecolor=color, edgecolor=color,
                       alpha=0.25, lw=0.8, zorder=1)
    ax.add_patch(poly)


def plot_panel(ax, df_scalp, paired):
    """
    Parameters
    ----------
    ax       : matplotlib Axes
    df_scalp : scalp_results DataFrame
    paired   : list of paired Subject_IDs
    """
    format_ax(ax, 'A')

    # Subject means
    records = []
    for sub in paired:
        pre  = df_scalp[(df_scalp.Subject_ID == sub) & (df_scalp.Condition == 'Pre')].Exponent.mean()
        post = df_scalp[(df_scalp.Subject_ID == sub) & (df_scalp.Condition == 'Post')].Exponent.mean()
        records.append({'Subject': sub, 'Pre': pre, 'Post': post})
    means = pd.DataFrame(records)

    pre_vals  = means['Pre'].values
    post_vals = means['Post'].values

    # Half violins
    _half_violin(ax, pre_vals, 0, COL_PRE, side='left', width=0.35)
    _half_violin(ax, post_vals, 1, COL_POST, side='right', width=0.35)

    # Paired lines
    for i in range(len(paired)):
        ax.plot([0, 1], [pre_vals[i], post_vals[i]],
                color='#888', alpha=ALPHA_PAIRED_LINE, lw=LW_PAIRED_LINE, zorder=1)

    # Jittered dots
    np.random.seed(42)
    j1 = np.random.uniform(0.02, 0.18, len(paired))
    j2 = np.random.uniform(-0.18, -0.02, len(paired))
    ax.scatter(0 + j1, pre_vals, s=MS_INDIVIDUAL, color=COL_PRE, **SCATTER_KW)
    ax.scatter(1 + j2, post_vals, s=MS_INDIVIDUAL, color=COL_POST, **SCATTER_KW)

    # Box: median, IQR
    for xpos, vals, col in [(0, pre_vals, COL_PRE), (1, post_vals, COL_POST)]:
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        offset = 0.22 if col == COL_PRE else -0.22
        ax.plot([xpos + offset] * 2, [q1, q3], color=col, lw=LW_MEAN_BAR,
                zorder=4, solid_capstyle='round')
        ax.plot([xpos + offset - 0.06, xpos + offset + 0.06], [med, med],
                color=col, lw=3, zorder=5)

    # Statistics
    t, p = stats.ttest_rel(post_vals, pre_vals)
    d = cohens_d_paired(pre_vals, post_vals)
    wilcox = stats.wilcoxon(post_vals - pre_vals)

    y_top = max(pre_vals.max(), post_vals.max()) + 0.06
    ax.plot([0, 0, 1, 1], [y_top, y_top + 0.02, y_top + 0.02, y_top],
            color='#555', lw=LW_BRACKET)
    ax.text(0.5, y_top + 0.035, f'p = {p:.3f}  (d = {d:.2f})',
            ha='center', fontsize=8.5, fontstyle='italic', color='#444')

    stat_box(ax,
             f'N = {len(paired)} paired\n'
             f'Paired t = {t:.2f}\n'
             f'Wilcoxon p = {wilcox.pvalue:.3f}')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Pre-surgery', 'Post-surgery'], fontsize=FS_AXIS_LABEL - 1)
    ax.set_ylabel('Aperiodic Exponent\n(whole-brain mean)', fontsize=FS_AXIS_LABEL - 1)
    ax.set_xlim(-0.6, 1.6)


# ── Standalone ──
if __name__ == "__main__":
    setup_style()
    df = load_scalp_results()
    paired = get_paired_subjects(df)
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_panel(ax, df, paired)
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / 'fig3a_wholebrain.png')
    fig.savefig(SCRIPT_DIR / 'fig3a_wholebrain.pdf')
    print(f"Saved fig3a  (N={len(paired)})")
    plt.close()
