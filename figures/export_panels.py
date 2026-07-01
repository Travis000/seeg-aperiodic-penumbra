#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_panels.py — Export EVERY figure panel as an individual file for
Adobe Illustrator assembly.

Per the author's workflow: panels are delivered SEPARATELY, with
- vector PDF (editable text, embedded Arial)  ← primary, drop into Illustrator
- 600 dpi PNG (white background)              ← raster fallback / preview
- NO panel letters (A/B/C), NO figure titles/suptitles  ← added in Illustrator

Output: JNNP投稿/2_Figures/panels_for_AI/<FigX>/<FigX_PanelY_descriptor>.{pdf,png}

Run from the panels/ directory:  python export_panels.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shutil
from pathlib import Path

from shared_config import *                       # palette, loaders, PAIRED_SUBJECTS, setup_style
from fig1a_psd import plot_panel as plot_1a
from fig1b_bands import plot_panel as plot_1b
from fig1c_breach import plot_panel as plot_1c
from fig2a_soz import plot_panel as plot_2a
from fig2b_gradient import plot_panel as plot_2b
from fig2c_crossmodal import plot_panel as plot_2c
from fig3a_wholebrain import plot_panel as plot_3a
from fig3b_laterality import plot_panel as plot_3b
from fig3c_topomap import plot_panel as plot_3c
from fig4a_outcome import plot_panel as plot_4a
from fig4b_regression import plot_panel as plot_4b
from fig4_perm import plot_panel as plot_4e
from fig5_panels import load_relapse_data, plot_swimmer, plot_relapse
from fig1_anchor_shaft import plot_panel as plot_anchor_shaft
from fig1_anchor_spectrum import plot_panel as plot_anchor_spectrum
from fig1_hero import draw_timeline, draw_illustration, draw_mechanism

OUT = SCRIPT_DIR.parent.parent.parent / "output"


def _editable_text():
    # keep text as editable text objects in PDF/PS, embed TrueType
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"


def _clean_stale():
    """Remove superseded Fig1 deliverables (old Graphviz timeline/mechanism, the
    dropped penumbra schematic, and the pre-relettering rawtrace)."""
    d = OUT / "Fig1"
    for stem in ["Fig1_A_timeline", "Fig1_F_mechanism", "Fig1_D2_mechanism",
                 "Fig1_D_penumbra_cartoon", "Fig1_E_rawtrace_N2",
                 "Fig1_D_rawtrace_N2", "Fig1_E_mechanism"]:
        for ext in ("pdf", "svg", "png"):
            p = d / f"{stem}.{ext}"
            if p.exists():
                p.unlink(); print(f"  ✗ removed stale {p.name}")


def _export_hero():
    """Export Fig1 panel A (hero): vertical study-timeline + enriched anatomical
    illustration, composed as one file. Panel letter added later in Illustrator."""
    d = OUT / "Fig1"; d.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.31, 1.0], wspace=0.0,
                          left=0.02, right=0.99, top=0.98, bottom=0.02)
    draw_timeline(fig.add_subplot(gs[0, 0]), letter="")
    draw_illustration(fig.add_subplot(gs[0, 1]))
    fig.savefig(d / "Fig1_A_hero.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(d / "Fig1_A_hero.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ✓ Fig1/Fig1_A_hero (pdf+png)")


def export(figdir, name, size, draw):
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    draw(ax)
    for loc in ("left", "center", "right"):       # strip baked-in panel letter (loc='left' title)
        ax.set_title("", loc=loc)
    d = OUT / figdir
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(d / f"{name}.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {figdir}/{name}")


def main():
    setup_style()
    _editable_text()

    # ---- load all data once ----
    df_seeg     = load_seeg_results()
    scalp_delta = load_scalp_midline_delta()
    engel       = load_engel_phase()
    engel_map   = dict(zip(engel.Subject, engel.Outcome))
    df          = load_scalp_results()
    roi         = load_roi()
    paired      = get_paired_subjects(df)
    data        = get_subject_level_data()
    M           = load_relapse_data()
    freqs, pre_psds, post_psds, bpaired = load_paired_psds()
    n_c = len(df_seeg)

    panels = [
        # Figure 1 — anchor. A = hero (timeline + illustration) via _export_hero();
        # B/C matplotlib data panels; D = low-key mechanism cascade.
        # (Raw N2-trace panel dropped — the exponent is spectral, so C carries the input.)
        ("Fig1", "Fig1_B_SEEG_trajectory",   (5.2, 4.4), lambda ax: plot_anchor_shaft(ax, df_seeg)),
        ("Fig1", "Fig1_C_spectrum_fit",      (5.2, 4.4), lambda ax: plot_anchor_spectrum(ax)),
        ("Fig1", "Fig1_D_mechanism",         (8.5, 2.2), lambda ax: draw_mechanism(ax, letter="")),
        # Figure 2 — SEEG mechanism
        ("Fig2", "Fig2_A_SOZ_vs_nonSOZ",     (5.0, 4.6), lambda ax: plot_2a(ax, df_seeg)),
        ("Fig2", "Fig2_B_distance_gradient", (5.4, 4.6), lambda ax: plot_2b(ax, df_seeg, engel)),
        ("Fig2", "Fig2_C_crossmodal",        (5.0, 4.6), lambda ax: plot_2c(ax, df_seeg, scalp_delta, engel_map)),
        # Figure 3 — scalp EEG
        ("Fig3", "Fig3_A_raincloud",         (4.6, 4.8), lambda ax: plot_3a(ax, df, paired)),
        ("Fig3", "Fig3_B_regional_delta",    (5.4, 4.8), lambda ax: plot_3b(ax, df, engel, roi, paired)),
        ("Fig3", "Fig3_C_topomap_numeric",   (6.0, 4.8), lambda ax: plot_3c(ax, df, roi, paired)),
        # Figure 4 — clinical outcome
        ("Fig4", "Fig4_A_outcome_paired",    (5.2, 4.8), lambda ax: plot_4a(ax, data)),
        ("Fig4", "Fig4_B_ANCOVA",            (5.0, 4.8), lambda ax: plot_4b(ax, data)),
        ("Fig4", "Fig4_C_swimmer_timeline",  (7.0, 4.8), lambda ax: plot_swimmer(ax, M, "C")),
        ("Fig4", "Fig4_D_relapse_status",    (4.6, 4.8), lambda ax: plot_relapse(ax, M, "D")),
        ("Fig4", "Fig4_E_permutation_null",  (5.0, 4.2), lambda ax: plot_4e(ax, data, "E")),
        # Supplementary Figure S2 — breach control
        ("FigS2", "FigS2_A_breach_PSD",      (5.2, 4.2), lambda ax: plot_1a(ax, freqs, pre_psds, post_psds)),
        ("FigS2", "FigS2_B_breach_bands",    (5.2, 4.2), lambda ax: plot_1b(ax, freqs, pre_psds, post_psds)),
        ("FigS2", "FigS2_C_breach_paired",   (4.4, 4.6), lambda ax: plot_1c(ax, freqs, pre_psds, post_psds, bpaired)),
    ]

    print("=" * 60)
    print(f"  Exporting {len(panels)} panels → {OUT}")
    print("=" * 60)
    _clean_stale()
    for figdir, name, size, draw in panels:
        export(figdir, name, size, draw)
    _export_hero()
    print(f"\n  Done. {len(panels)} matplotlib panels + hero A in {OUT}\n")


if __name__ == "__main__":
    main()
