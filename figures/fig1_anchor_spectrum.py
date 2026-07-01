# -*- coding: utf-8 -*-
"""
Anchor panel C — representative scalp power spectrum (Pre vs Post) with the
specparam aperiodic fit overlaid; shows what the aperiodic exponent measures
(slope of the 1/f background) and previews its post-operative decline.

Data: scalp_psd_curves.npz (cached PSD, no EDF reload). Aperiodic fit recomputed
with the pipeline's specparam settings (fixed mode, 1–45 Hz).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from shared_config import (format_ax, COL_PRE, COL_POST, LW_PSD_CURVE,
                           FS_ANNOTATION, FS_AXIS_LABEL, FS_LEGEND)
from shared_config import _resolve

REP_SUBJECT = 'Sub02'          # representative good-outcome recording (Engel Ia)
MID = ['Fz', 'Cz', 'Pz']


def _fit_aperiodic(freqs, psd):
    """Return (offset, exponent) from specparam fixed-mode fit over 1–45 Hz."""
    from specparam import SpectralModel
    fm = SpectralModel(peak_width_limits=[1, 8], max_n_peaks=4,
                       min_peak_height=0.05, aperiodic_mode='fixed', verbose=False)
    fm.fit(freqs, psd, [1, 45])
    ap = np.asarray(fm.results.params.aperiodic.params)
    return float(ap[0]), float(ap[-1])


def _get_psd(d, order, base):
    """Midline-averaged PSD for a recording key (handles the Post_Post naming)."""
    mid_idx = [order.index(c) for c in MID]
    if base in d.files:
        return d[base][mid_idx].mean(axis=0)
    stem = base[:-4] if base.endswith('_psd') else base   # strip '_psd'
    for k in sorted(d.files):
        if k.startswith(stem) and k.endswith('_psd'):
            return d[k][mid_idx].mean(axis=0)
    return None


def plot_panel(ax, subject=REP_SUBJECT):
    format_ax(ax, 'C')
    d = np.load(_resolve('scalp_psd_curves.npz'), allow_pickle=True)
    freqs = d['freqs']
    order = list(d['standard_channel_order'])
    mask = (freqs >= 1) & (freqs <= 45)
    f = freqs[mask]

    pre = _get_psd(d, order, f"{subject}_Pre_psd")
    post = _get_psd(d, order, f"{subject}_Post_Post_psd")
    if post is None:
        post = _get_psd(d, order, f"{subject}_Post_psd")

    for psd, col, lab in [(pre, COL_PRE, 'Pre'), (post, COL_POST, 'Post')]:
        if psd is None:
            continue
        off, exp = _fit_aperiodic(freqs, psd)
        data_db = 10 * np.log10(psd[mask])
        ap_db = 10 * (off - exp * np.log10(f))
        ax.plot(f, data_db, color=col, lw=LW_PSD_CURVE, alpha=0.9,
                label=f'{lab}-surgery  (exponent = {exp:.2f})', zorder=3)
        ax.plot(f, ap_db, color=col, lw=1.3, ls='--', alpha=0.9, zorder=4)
        print(f"  [Fig1C] {subject} {lab}: offset={off:.2f}, exponent={exp:.3f}")

    ax.set_xscale('log')
    ax.set_xlim(1, 45)
    ax.set_xticks([1, 2, 5, 10, 20, 45])
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel('Frequency (Hz, log scale)', fontsize=FS_AXIS_LABEL)
    ax.set_ylabel('Power spectral density (dB)', fontsize=FS_AXIS_LABEL)
    ax.legend(fontsize=FS_LEGEND, loc='upper right', framealpha=0.9)
    ax.text(0.03, 0.05,
            'dashed = specparam aperiodic fit\n(steeper slope = higher exponent)',
            transform=ax.transAxes, fontsize=FS_ANNOTATION, color='#555',
            va='bottom', ha='left', fontstyle='italic')
    ax.text(0.5, 1.02, f'Representative midline EEG (Fz/Cz/Pz), '
                       f'{subject.replace("Sub", "Sub-")}',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=FS_ANNOTATION, color='#666')


if __name__ == "__main__":
    from shared_config import setup_style, SCRIPT_DIR, DPI
    setup_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    plot_panel(ax)
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / 'fig1_anchor_spectrum.png', dpi=DPI, bbox_inches='tight')
    print('Saved fig1_anchor_spectrum')
    plt.close()
