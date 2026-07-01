#!/usr/bin/env python3
"""
Fig1A — Grand-average PSD curves: Pre vs Post (all channels averaged)
Shows no emergence of high-frequency "breach" peak after surgery.
"""

import numpy as np
import matplotlib.pyplot as plt
from shared_config import *


def plot_panel(ax, freqs, pre_psds, post_psds):
    """
    Parameters
    ----------
    ax        : matplotlib Axes
    freqs     : (n_freqs,)
    pre_psds  : (n_subjects, n_freqs) — linear power
    post_psds : (n_subjects, n_freqs) — linear power
    """
    format_ax(ax, 'A')

    # Convert to dB
    pre_db  = 10 * np.log10(pre_psds + 1e-20)
    post_db = 10 * np.log10(post_psds + 1e-20)

    pre_mean  = pre_db.mean(axis=0)
    post_mean = post_db.mean(axis=0)
    pre_sem   = pre_db.std(axis=0, ddof=1)  / np.sqrt(len(pre_db))
    post_sem  = post_db.std(axis=0, ddof=1) / np.sqrt(len(post_db))

    # SEM shading
    ax.fill_between(freqs, pre_mean - pre_sem, pre_mean + pre_sem,
                    color=COL_PRE, alpha=ALPHA_SEM_FILL, zorder=1)
    ax.fill_between(freqs, post_mean - post_sem, post_mean + post_sem,
                    color=COL_POST, alpha=ALPHA_SEM_FILL, zorder=1)

    # Mean curves
    ax.plot(freqs, pre_mean, color=COL_PRE, lw=LW_PSD_CURVE,
            label='Pre-surgery', zorder=3)
    ax.plot(freqs, post_mean, color=COL_POST, lw=LW_PSD_CURVE,
            label='Post-surgery', zorder=3)

    # Breach zone
    ax.axvspan(20, 45, color='#FFD700', alpha=0.08, zorder=0,
               label='Breach zone (20–45 Hz)')

    ax.set_xlabel('Frequency (Hz)', fontsize=FS_AXIS_LABEL)
    ax.set_ylabel('Power Spectral Density (dB)', fontsize=FS_AXIS_LABEL)
    ax.set_xlim(1, 45)
    ax.legend(fontsize=FS_LEGEND, loc='upper right', framealpha=0.9,
              bbox_to_anchor=(1.0, 0.93))

    # Annotation
    ax.text(32, ax.get_ylim()[1] - 2, 'No breach\npeak',
            fontsize=8, ha='center', fontstyle='italic', color='#999')


# ── Standalone ──
if __name__ == "__main__":
    setup_style()
    freqs, pre, post, paired = load_paired_psds()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    plot_panel(ax, freqs, pre, post)
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / 'fig1a_psd.png')
    fig.savefig(SCRIPT_DIR / 'fig1a_psd.pdf')
    print(f"Saved fig1a  (N={len(paired)})")
    plt.close()
