# -*- coding: utf-8 -*-
"""
Fig 4 permutation panel — null distribution of the ANCOVA outcome effect
(Post ~ Pre + Outcome) under random shuffling of Good/Poor labels.

Honest small-N inference that gives the "null-distribution" visual of the
sister paper WITHOUT a fragile N=16 ROC/AUC: it tests the SAME quantity the
ANCOVA already reports (the outcome-group coefficient) by permutation.
"""
import numpy as np
import statsmodels.formula.api as smf
from shared_config import (format_ax, get_subject_level_data, COL_POOR,
                           FS_AXIS_LABEL, FS_ANNOTATION, stat_box)

N_PERM = 5000


def _beta_outcome(data):
    d = data.copy()
    d['Outcome_bin'] = (d['Outcome'] == 'Good').astype(int)
    return smf.ols("Post ~ Pre + Outcome_bin", data=d).fit().params['Outcome_bin']


def plot_panel(ax, data=None, letter='E'):
    format_ax(ax, letter)
    if data is None:
        data = get_subject_level_data()

    obs = _beta_outcome(data)
    rng = np.random.default_rng(0)
    labels = data['Outcome'].values.copy()
    dd = data.copy()
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        dd['Outcome'] = rng.permutation(labels)
        null[i] = _beta_outcome(dd)
    p_perm = (np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1)

    ax.hist(null, bins=40, color='#BBBBBB', edgecolor='white',
            linewidth=0.3, zorder=2)
    ax.axvline(0, color='#888', ls=':', lw=0.8, zorder=1)
    ax.axvline(obs, color=COL_POOR, lw=2.0, zorder=4)
    ymax = ax.get_ylim()[1]
    ax.text(obs, ymax * 0.55, ' observed', rotation=90, ha='right', va='center',
            fontsize=FS_ANNOTATION, color=COL_POOR, fontweight='bold')

    ax.set_xlabel('ANCOVA outcome β under label permutation', fontsize=FS_AXIS_LABEL)
    ax.set_ylabel('Permutations (count)', fontsize=FS_AXIS_LABEL)
    stat_box(ax, f'observed β = {obs:.3f}\n{N_PERM} permutations\nP_perm = {p_perm:.3f}',
             x=0.03, y=0.97, ha='left', va='top')

    print(f"  [Fig4E] observed ANCOVA outcome β={obs:.4f}, "
          f"P_perm={p_perm:.4f} ({N_PERM} perms)")
    return dict(obs=obs, p_perm=p_perm)


if __name__ == "__main__":
    from shared_config import setup_style, SCRIPT_DIR, DPI
    import matplotlib.pyplot as plt
    setup_style()
    fig, ax = plt.subplots(figsize=(5, 4.2))
    res = plot_panel(ax)
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / 'fig4_perm.png', dpi=DPI, bbox_inches='tight')
    print('Saved fig4_perm;', res)
    plt.close()
