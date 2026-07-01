# -*- coding: utf-8 -*-
"""
Anchor panel B — representative along-electrode profile of the aperiodic exponent.
Shows the exponent 'dip' at SOZ contacts rising into the surrounding cortex
(the 1-D evidence of the inhibitory penumbra). Real data; the electrode is the
one whose SOZ-vs-surround contrast is closest to the MEDIAN among electrodes that
exhibit the gradient — a typical example, not the single cleanest one (avoids a
cherry-pick critique).

NOTE: no 3-D electrode coordinates / imaging exist for this cohort; the spatial
representation is therefore the along-shaft contact ordering, not an MRI render.
"""
import numpy as np
import matplotlib.pyplot as plt
from shared_config import (format_ax, load_seeg_results, MIN_EXPONENT,
                           COL_SOZ, COL_NSOZ, FS_ANNOTATION, FS_AXIS_LABEL,
                           FS_LEGEND)


def _pick_electrode(df):
    """Pick a *representative* SOZ-bearing electrode: among electrodes that exhibit
    the penumbra signature (positive SOZ-vs-surround gap, enough contacts), take the
    one whose contrast is closest to the median — a typical example rather than the
    single cleanest one (pre-empts a cherry-pick critique). Deterministic."""
    qc = df[(df.R_Squared >= 0.85) & (df.Exponent >= MIN_EXPONENT)].copy()
    cands = []
    for (sub, elec), g in qc.groupby(['Subject', 'Electrode']):
        if not bool(g.Has_SOZ_on_Electrode.any()):
            continue
        soz = g[g.Is_SOZ].Exponent
        nsoz = g[~g.Is_SOZ].Exponent
        if len(soz) >= 1 and len(nsoz) >= 4 and len(g) >= 7:
            gap = nsoz.mean() - soz.mean()
            if gap > 0:                       # must actually show the dip to illustrate it
                cands.append((gap, sub, elec, g))
    if not cands:
        return None
    gaps = np.array([c[0] for c in cands])
    med = float(np.median(gaps))
    # closest to median; deterministic tie-break by (|gap-med|, -n_contacts, sub, elec)
    cands.sort(key=lambda c: (abs(c[0] - med), -len(c[3]), c[1], c[2]))
    pick = cands[0]
    print(f"  [Fig1B] {len(cands)} eligible electrodes; gap median={med:.3f}, "
          f"min={gaps.min():.3f}, max={gaps.max():.3f}; picked representative "
          f"{pick[1]}/{pick[2]} gap={pick[0]:.3f}")
    return pick


def plot_panel(ax, df=None):
    format_ax(ax, 'B')
    if df is None:
        df = load_seeg_results()
    pick = _pick_electrode(df)
    if pick is None:
        ax.text(0.5, 0.5, 'No suitable electrode', transform=ax.transAxes,
                ha='center', va='center', color='#999')
        return
    gap, sub, elec, g = pick
    g = g.sort_values('Contact')
    x = g.Contact.values.astype(float)
    y = g.Exponent.values.astype(float)
    is_soz = g.Is_SOZ.values.astype(bool)

    ax.plot(x, y, '-', color='#999', lw=1.4, zorder=2)
    ax.scatter(x[~is_soz], y[~is_soz], s=60, color=COL_NSOZ, edgecolors='white',
               linewidths=0.6, zorder=3, label='Non-SOZ contact')
    ax.scatter(x[is_soz], y[is_soz], s=80, color=COL_SOZ, edgecolors='white',
               linewidths=0.6, zorder=4, marker='s', label='SOZ contact')

    # shade SOZ span + label
    if is_soz.any():
        sx = x[is_soz]
        ax.axvspan(sx.min() - 0.5, sx.max() + 0.5, color=COL_SOZ, alpha=0.08, zorder=0)

    y_lo, y_hi = y.min(), y.max()
    pad = (y_hi - y_lo) * 0.18 + 0.05
    ax.set_ylim(y_lo - pad, y_hi + pad)
    if is_soz.any():
        ax.text(np.mean(x[is_soz]), y_hi + pad * 0.55, 'SOZ', ha='center', va='top',
                fontsize=FS_ANNOTATION, color=COL_SOZ, fontweight='bold')

    ax.set_xlabel(f'Contact position along electrode {elec}', fontsize=FS_AXIS_LABEL)
    ax.set_ylabel('Aperiodic exponent', fontsize=FS_AXIS_LABEL)
    ax.legend(fontsize=FS_LEGEND, loc='lower right', framealpha=0.9)
    ax.text(0.5, 1.02, f'Representative electrode ({sub.replace("Sub", "Sub-")}, {elec})',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=FS_ANNOTATION, color='#666')
    print(f"  [Fig1B] representative electrode {sub}/{elec}: "
          f"{len(g)} contacts, SOZ n={int(is_soz.sum())}, surround−SOZ gap={gap:.3f}")


if __name__ == "__main__":
    from shared_config import setup_style, SCRIPT_DIR, DPI
    setup_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    plot_panel(ax)
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / 'fig1_anchor_shaft.png', dpi=DPI, bbox_inches='tight')
    print('Saved fig1_anchor_shaft')
    plt.close()
