#!/usr/bin/env python3
"""
Fig4A — Good vs Poor outcome: paired Pre/Post strip plot with mean ± SEM.
v22 UPDATE: Added BF₀₁ for poor-outcome pre-post contrast.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from scipy import stats
import pingouin as pg
from shared_config import *


def plot_panel(ax, data):
    """
    Parameters
    ----------
    ax   : matplotlib Axes
    data : subject-level DataFrame with columns
           [Subject, Pre, Post, Delta, Outcome, ...]
    """
    format_ax(ax, 'A')

    good = data[data.Outcome == 'Good']
    poor = data[data.Outcome == 'Poor']

    positions = {'good_pre': 0, 'good_post': 1,
                 'poor_pre': 2.5, 'poor_post': 3.5}

    # Background shading
    ax.axvspan(-0.5, 1.5, color=COL_GOOD, alpha=0.06, zorder=0)
    ax.axvspan(2.0, 4.0, color=COL_POOR, alpha=0.06, zorder=0)

    np.random.seed(42)

    # Plot dots + paired lines
    for grp, pre_pos, post_pos in [
        (good, positions['good_pre'], positions['good_post']),
        (poor, positions['poor_pre'], positions['poor_post']),
    ]:
        for _, row in grp.iterrows():
            jp = np.random.uniform(-0.12, 0.12)
            jq = np.random.uniform(-0.12, 0.12)
            ax.plot([pre_pos + jp, post_pos + jq],
                    [row['Pre'], row['Post']],
                    color='#888', alpha=ALPHA_PAIRED_LINE,
                    lw=LW_PAIRED_LINE, zorder=1)
            ax.scatter(pre_pos + jp, row['Pre'], s=30, color=COL_PRE,
                       edgecolors='white', linewidths=0.4, zorder=3)
            ax.scatter(post_pos + jq, row['Post'], s=30, color=COL_POST,
                       edgecolors='white', linewidths=0.4, zorder=3)

    # Mean ± SEM bars
    draw_mean_sem(ax, positions['good_pre'],  good['Pre'],  COL_PRE, width=0.2)
    draw_mean_sem(ax, positions['good_post'], good['Post'], COL_POST, width=0.2)
    draw_mean_sem(ax, positions['poor_pre'],  poor['Pre'],  COL_PRE, width=0.2)
    draw_mean_sem(ax, positions['poor_post'], poor['Post'], COL_POST, width=0.2)

    # Good bracket
    t_g, p_g = stats.ttest_rel(good['Post'], good['Pre'])
    d_g = cohens_d_paired(good['Pre'], good['Post'])
    y_top_g = max(good['Pre'].max(), good['Post'].max()) + 0.08
    ax.plot([0, 0, 1, 1], [y_top_g, y_top_g + 0.02, y_top_g + 0.02, y_top_g],
            color='#555', lw=LW_BRACKET)
    if p_g < 0.05:
        ax.text(0.5, y_top_g + 0.035, f'*p = {p_g:.3f}',
                ha='center', fontsize=7.5, fontstyle='italic', color='#333',
                fontweight='bold')
    else:
        ax.text(0.5, y_top_g + 0.035, f'n.s. (p = {p_g:.2f})',
                ha='center', fontsize=7.5, fontstyle='italic', color='#666')

    # Poor bracket
    t_p, p_p = stats.ttest_rel(poor['Post'], poor['Pre'])
    d_p = cohens_d_paired(poor['Pre'], poor['Post'])
    y_top_p = max(poor['Pre'].max(), poor['Post'].max()) + 0.08
    ax.plot([2.5, 2.5, 3.5, 3.5],
            [y_top_p, y_top_p + 0.02, y_top_p + 0.02, y_top_p],
            color='#555', lw=LW_BRACKET)
    sig_label = f'*p = {p_p:.3f}' if p_p < 0.05 else f'n.s. (p = {p_p:.2f})'
    ax.text(3.0, y_top_p + 0.035, sig_label,
            ha='center', fontsize=7.5, fontstyle='italic',
            color='#333' if p_p < 0.05 else '#666',
            fontweight='bold' if p_p < 0.05 else 'normal')

    # Y limits
    all_vals = pd.concat([data['Pre'], data['Post']])
    ax.set_ylim(all_vals.min() - 0.15, all_vals.max() + 0.22)

    # Effect size boxes
    # Bayesian paired t-test for poor-outcome group  [v22]
    bf_poor = pg.ttest(poor['Post'].values, poor['Pre'].values, paired=True)
    bf10_poor = float(bf_poor['BF10'].values[0])
    bf01_poor = 1.0 / bf10_poor

    y_bottom = all_vals.min() - 0.10
    # Good group box
    ax.text(0.5, y_bottom, f'n = {len(good)},  d = {d_g:.2f}',
            ha='center', va='bottom', fontsize=FS_ANNOTATION, color='#555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=COL_GOOD, alpha=0.8, lw=0.8))
    # Poor group box (with BF₀₁)
    ax.text(3.0, y_bottom,
            f'n = {len(poor)},  d = {d_p:.2f}\n'
            f'BF$_{{01}}$ = {bf01_poor:.2f}',
            ha='center', va='bottom', fontsize=FS_ANNOTATION, color='#555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=COL_POOR, alpha=0.8, lw=0.8))

    ax.set_xticks([0, 1, 2.5, 3.5])
    ax.set_xticklabels(['Pre', 'Post', 'Pre', 'Post'], fontsize=FS_AXIS_TICK)

    # Group labels
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(0.5, -0.12, 'Good Outcome\n(Engel I–II)',
            ha='center', fontsize=FS_AXIS_TICK, fontweight='bold',
            color=COL_GOOD, transform=trans, va='top')
    ax.text(3.0, -0.12, 'Poor Outcome\n(Engel III–IV)',
            ha='center', fontsize=FS_AXIS_TICK, fontweight='bold',
            color=COL_POOR, transform=trans, va='top')

    ax.set_ylabel('Aperiodic Exponent', fontsize=FS_AXIS_LABEL)


# ── Standalone ──
if __name__ == "__main__":
    setup_style()
    data = get_subject_level_data()
    fig, ax = plt.subplots(figsize=(6, 5.5))
    plot_panel(ax, data)
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(SCRIPT_DIR / 'fig4a_outcome.png')
    fig.savefig(SCRIPT_DIR / 'fig4a_outcome.pdf')

    good = data[data.Outcome == 'Good']
    poor = data[data.Outcome == 'Poor']
    print(f"Saved fig4a  (Good={len(good)}, Poor={len(poor)})")

    # v22: BF₀₁ verification
    bf_poor = pg.ttest(poor['Post'].values, poor['Pre'].values, paired=True)
    bf10 = float(bf_poor['BF10'].values[0])
    print(f"  Poor pre-post: BF10 = {bf10:.3f}, BF01 = {1/bf10:.3f}")
    bf_good = pg.ttest(good['Post'].values, good['Pre'].values, paired=True)
    bf10g = float(bf_good['BF10'].values[0])
    print(f"  Good pre-post: BF10 = {bf10g:.3f}")
    plt.close()
