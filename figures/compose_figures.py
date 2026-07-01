#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
compose_figures.py — Master Assembly Script
=============================================================================
Composes all individual panels into 4 publication-ready figures.
Each panel is imported from its own module and drawn onto a shared axes.

Statistical methods per figure
-------------------------------
Fig 2A : LMM (Exponent ~ Is_SOZ + [1|Subject], two-tailed) — primary
         Wilcoxon one-tailed + paired-t retained as sensitivity only
Fig 2B : Spearman gradient + Fisher z (Good vs Poor interaction, p=0.753)
         Narrative: inhibitory penumbra is a universal feature
Fig 3B : BH-FDR correction across 6 regional tests (3 regions × 2 outcomes)
         Good-Midline: p_FDR = 0.0016
Fig 4B : ANCOVA (Post ~ Pre + Outcome); r = -0.572 removed entirely
         Oldham's method (p=0.880) confirmed RTM artefact in raw correlation

Usage:
    python compose_figures.py              # build all 4 figures
    python compose_figures.py --fig 1      # build Figure 1 only
    python compose_figures.py --fig 1 3 4  # build Figures 1, 3, 4

Output:
    Figure_1_Breach.{png,pdf}
    Figure_2_SEEG_Mechanism.{png,pdf}
    Figure_3_ScalpEEG.{png,pdf}
    Figure_4_Clinical.{png,pdf}
=============================================================================
"""

import sys
import argparse
import matplotlib.pyplot as plt

from shared_config import *

# Panel imports
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
from fig5_panels import load_relapse_data, plot_swimmer, plot_relapse
from fig1_hero import draw_timeline, draw_illustration, draw_mechanism
from fig1_anchor_shaft import plot_panel as plot_anchor_shaft
from fig1_anchor_spectrum import plot_panel as plot_anchor_spectrum
from fig4_perm import plot_panel as plot_4e


def build_figure_1():
    """Figure 1 (anchor): A hero (timeline+anatomy), B along-electrode dip,
    C spectrum+specparam fit, D proposed mechanism."""
    print("\n══ Figure 1: Study design & aperiodic marker (anchor) ══")
    df_seeg = load_seeg_results()
    n_c = len(df_seeg)
    print(f"   {df_seeg.Subject.nunique()} subjects, {n_c} contacts")

    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.6],
                          left=0.04, right=0.985, top=0.94, bottom=0.05,
                          wspace=0.24, hspace=0.42)

    # A (hero) — vertical study-timeline + anatomical illustration with the
    # exponent-gradient / inhibitory-penumbra finding drawn on the real anatomy.
    ags = gs[0:2, 0:2].subgridspec(1, 2, width_ratios=[0.31, 1.0], wspace=0.0)
    draw_timeline(fig.add_subplot(ags[0, 0]))
    draw_illustration(fig.add_subplot(ags[0, 1]))
    plot_anchor_shaft(fig.add_subplot(gs[0, 2]), df_seeg)             # B  along-electrode dip
    plot_anchor_spectrum(fig.add_subplot(gs[1, 2]))                   # C  spectrum + specparam fit
    draw_mechanism(fig.add_subplot(gs[2, :]), letter='D')            # D  proposed mechanism

    fig.suptitle(
        'Figure 1. Study Design and the Aperiodic Exponent as a Marker of '
        'Excitation/Inhibition Balance',
        fontsize=12.5, fontweight='bold', y=0.965)

    _save(fig, 'Figure_1_Anchor')


def build_supp_breach():
    """Supplementary figure: absence of breach rhythm after RF-TC (3 panels)."""
    print("\n══ Supplementary: Breach Rhythm Control ══")
    freqs, pre_psds, post_psds, paired = load_paired_psds()
    print(f"   N = {len(paired)} paired subjects")

    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 0.9],
                          left=0.06, right=0.96, bottom=0.14, top=0.85,
                          wspace=0.3)

    plot_1a(fig.add_subplot(gs[0]), freqs, pre_psds, post_psds)
    plot_1b(fig.add_subplot(gs[1]), freqs, pre_psds, post_psds)
    plot_1c(fig.add_subplot(gs[2]), freqs, pre_psds, post_psds, paired)

    fig.suptitle(
        f'Supplementary Figure. Absence of Breach Rhythm After SEEG '
        f'Thermocoagulation  (N = {len(paired)} paired)',
        fontsize=12.5, fontweight='bold', y=0.97)

    _save(fig, 'Figure_S_Breach')


def build_figure_2():
    """Figure 2: Intracranial SEEG Mechanism (3 panels)."""
    print("\n══ Figure 2: SEEG Mechanism ══")
    df_seeg     = load_seeg_results()
    scalp_delta = load_scalp_midline_delta()
    engel       = load_engel_phase()               # DataFrame[Subject, Outcome]
    engel_map   = dict(zip(engel.Subject, engel.Outcome))  # for plot_2c
    n_s = df_seeg.Subject.nunique()
    n_c = len(df_seeg)
    print(f"   {n_s} subjects, {n_c} contacts")

    fig = plt.figure(figsize=(16, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.2, 1.0],
                          left=0.06, right=0.96, bottom=0.14, top=0.82,
                          wspace=0.35)

    plot_2a(fig.add_subplot(gs[0]), df_seeg)
    plot_2b(fig.add_subplot(gs[1]), df_seeg, engel)
    plot_2c(fig.add_subplot(gs[2]), df_seeg, scalp_delta, engel_map)

    fig.suptitle(
        f'Figure 2. Intracranial Evidence for Compensatory Inhibition '
        f'in the Seizure Onset Zone\n'
        f'(N = {n_s} patients, {n_c} SEEG contacts, interictal N2 sleep, '
        f'Exponent ≥ {MIN_EXPONENT})',
        fontsize=11.5, fontweight='bold', y=0.97)

    _save(fig, 'Figure_2_SEEG_Mechanism')


def build_figure_3():
    """Figure 3: Core Scalp EEG Results (3 panels)."""
    print("\n══ Figure 3: Scalp EEG Results ══")
    df     = load_scalp_results()
    engel  = load_engel_phase()
    roi    = load_roi()
    paired = get_paired_subjects(df)
    print(f"   N = {len(paired)} paired subjects")

    fig = plt.figure(figsize=(15.5, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.05, 1.5],
                          left=0.06, right=0.96, bottom=0.14, top=0.86,
                          wspace=0.32)

    plot_3a(fig.add_subplot(gs[0]), df, paired)
    plot_3b(fig.add_subplot(gs[1]), df, engel, roi, paired)
    plot_3c(fig.add_subplot(gs[2]), df, roi, paired)

    fig.suptitle(
        f'Figure 3. Scalp EEG Aperiodic Exponent Changes After SEEG '
        f'Thermocoagulation  (N = {len(paired)} paired)',
        fontsize=12, fontweight='bold', y=0.97)

    _save(fig, 'Figure_3_ScalpEEG')


def build_figure_4():
    """Figure 4: Clinical translation — outcome + relapse-timeline (2×2, merges old Fig4+Fig5)."""
    print("\n══ Figure 4: Clinical Translation (merged 2×2) ══")
    data = get_subject_level_data()
    n_g = len(data[data.Outcome == 'Good'])
    n_p = len(data[data.Outcome == 'Poor'])
    M = load_relapse_data()
    print(f"   N = {len(data)} (Good={n_g}, Poor={n_p}); relapse rows = {len(M)}")

    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.0], height_ratios=[1.0, 1.0],
                          left=0.06, right=0.97, bottom=0.08, top=0.91,
                          wspace=0.30, hspace=0.32)

    # Top row = the outcome effect and its robustness; bottom row = the relapse story.
    # Panel letters keep their content mapping (A outcome, B ANCOVA, C swimmer,
    # D relapse, E permutation) — swimmer spans 2 cells so no grid cell is left blank.
    plot_4a(fig.add_subplot(gs[0, 0]), data)            # A: Good vs Poor paired
    plot_4b(fig.add_subplot(gs[0, 1]), data)            # B: ANCOVA
    plot_4e(fig.add_subplot(gs[0, 2]), data, 'E')       # E: permutation null (validates B)
    plot_swimmer(fig.add_subplot(gs[1, 0:2]), M, 'C')   # C: relapse-timeline swimmer (wide)
    plot_relapse(fig.add_subplot(gs[1, 2]), M, 'D')     # D: ΔExponent by relapse status

    fig.suptitle(
        f'Figure 4. Early Aperiodic Shift Tracks With Surgical Outcome, '
        f'Independently of Concurrent Seizure Status  (N = {len(data)} paired)',
        fontsize=12.5, fontweight='bold', y=0.965)

    _save(fig, 'Figure_4_Clinical')


def _save(fig, name):
    """Save figure as PNG + PDF to SCRIPT_DIR."""
    out_png = SCRIPT_DIR / f'{name}.png'
    out_pdf = SCRIPT_DIR / f'{name}.pdf'
    fig.savefig(out_png, dpi=DPI, bbox_inches='tight', facecolor='white')
    fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"   ✓ {out_png.name}")
    print(f"   ✓ {out_pdf.name}")


BUILDERS = {
    1: build_figure_1,        # NEW anchor: study design + aperiodic marker
    2: build_figure_2,        # SEEG mechanism (decluttered)
    3: build_figure_3,        # Scalp EEG (recomposed)
    4: build_figure_4,        # Clinical outcome (merged old Fig4 + Fig5)
    5: build_supp_breach,     # Supplementary: breach-rhythm control
}


def main():
    setup_style()

    parser = argparse.ArgumentParser(description='Compose publication figures')
    parser.add_argument('--fig', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                        help='Which figures to build (default: all + supp breach=5)')
    args = parser.parse_args()

    print("=" * 60)
    print("  Publication Figure Composer")
    print("=" * 60)

    for n in args.fig:
        if n in BUILDERS:
            BUILDERS[n]()
        else:
            print(f"  ⚠ Unknown figure number: {n}")

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
