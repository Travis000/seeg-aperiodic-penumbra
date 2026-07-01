#!/usr/bin/env python3
"""
Fig3C — ΔExponent topographic map (lesion-locked).
Left-lesion patients mirror-flipped so ipsilateral = RIGHT on map.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator
from shared_config import *


def plot_panel(ax, df_scalp, roi, paired):
    """
    Parameters
    ----------
    ax       : matplotlib Axes
    df_scalp : scalp_results DataFrame
    roi      : DataFrame[Subject, Inferred Side]
    paired   : list of Subject_IDs
    """
    format_ax(ax, 'C', bg=False)
    ax.set_title('C', fontsize=FS_PANEL_LABEL, fontweight='bold',
                 loc='left', x=-0.05, pad=8)

    # Compute lesion-locked delta per channel
    all_channels = STANDARD_1020
    delta_by_locked_ch = {ch: [] for ch in all_channels}

    for sub in paired:
        side = roi[roi.Subject == sub]['Inferred Side'].values[0]
        for ch in all_channels:
            pre  = df_scalp[(df_scalp.Subject_ID == sub) &
                            (df_scalp.Condition == 'Pre') & (df_scalp.Channel == ch)]
            post = df_scalp[(df_scalp.Subject_ID == sub) &
                            (df_scalp.Condition == 'Post') & (df_scalp.Channel == ch)]
            if len(pre) == 0 or len(post) == 0:
                continue
            delta = post.Exponent.values[0] - pre.Exponent.values[0]
            locked_ch = MIRROR_MAP[ch] if side == 'Left' else ch
            delta_by_locked_ch[locked_ch].append(delta)

    chan_delta = {}
    for ch in all_channels:
        vals = delta_by_locked_ch[ch]
        if vals:
            chan_delta[ch] = np.mean(vals)

    xs, ys, vals = [], [], []
    for ch in all_channels:
        if ch in chan_delta and ch in CHAN_XY:
            x, y = CHAN_XY[ch]
            xs.append(x); ys.append(y); vals.append(chan_delta[ch])
    xs, ys, vals = np.array(xs), np.array(ys), np.array(vals)

    # Interpolate
    grid_x = np.linspace(-1.2, 1.2, 200)
    grid_y = np.linspace(-1.2, 1.2, 200)
    X, Y = np.meshgrid(grid_x, grid_y)
    interp = CloughTocher2DInterpolator(list(zip(xs, ys)), vals)
    Z = interp(X, Y)

    head_radius = 1.05
    Z[X**2 + Y**2 > head_radius**2] = np.nan

    vmax = max(abs(np.nanmin(Z)), abs(np.nanmax(Z)), 0.05)
    im = ax.pcolormesh(X, Y, Z, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       shading='auto', zorder=1)

    # Head outline
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(head_radius * np.cos(theta), head_radius * np.sin(theta),
            'k-', lw=1.5, zorder=3)

    # Nose
    ax.plot([-0.08, 0, 0.08],
            [head_radius, head_radius + 0.12, head_radius],
            'k-', lw=1.5, zorder=3)

    # Ears
    for sign in [-1, 1]:
        ear_x = sign * np.array([head_radius, head_radius + 0.06,
                                  head_radius + 0.08, head_radius + 0.06,
                                  head_radius])
        ear_y = np.array([0.15, 0.1, 0, -0.1, -0.15])
        ax.plot(ear_x, ear_y, 'k-', lw=1.2, zorder=3)

    # Channel markers + per-channel numeric ΔExponent labels.
    # Electrode-name labels intentionally dropped — values alone declutter the
    # map (topographic position already conveys channel identity).
    import matplotlib.patheffects as pe
    ax.scatter(xs, ys, s=9, color='black', zorder=4, marker='o')
    for ch in all_channels:
        if ch in CHAN_XY and ch in chan_delta:
            x, y = CHAN_XY[ch]
            ax.annotate(f'{chan_delta[ch]:+.2f}', (x, y), fontsize=5.2,
                        ha='center', va='top', xytext=(0, -3.2),
                        textcoords='offset points', color='black',
                        fontweight='bold', zorder=6,
                        path_effects=[pe.withStroke(linewidth=1.5,
                                                    foreground='white')])

    # Laterality labels
    ax.text(1.25, 0, 'Ipsilateral\n(lesion side)',
            fontsize=FS_ANNOTATION, ha='left', va='center',
            color=COL_IPSI, fontweight='bold', fontstyle='italic')
    ax.text(-1.25, 0, 'Contralateral',
            fontsize=FS_ANNOTATION, ha='right', va='center',
            color=COL_CONTRA, fontweight='bold', fontstyle='italic')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.15, aspect=20)
    cbar.set_label('ΔExponent', fontsize=FS_AXIS_TICK)
    cbar.ax.tick_params(labelsize=FS_ANNOTATION)

    ax.set_xlim(-1.55, 1.85)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(0, -1.28,
            'Lesion-locked ΔExponent\n(Left-lesion patients mirror-flipped)',
            ha='center', fontsize=FS_ANNOTATION, color='#666', fontstyle='italic')


# ── Standalone ──
if __name__ == "__main__":
    setup_style()
    df    = load_scalp_results()
    roi   = load_roi()
    paired = get_paired_subjects(df)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    plot_panel(ax, df, roi, paired)
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / 'fig3c_topomap.png')
    fig.savefig(SCRIPT_DIR / 'fig3c_topomap.pdf')
    print(f"Saved fig3c  (N={len(paired)})")
    plt.close()
