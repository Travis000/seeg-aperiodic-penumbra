#!/usr/bin/env python3
"""
Fig2B — Distance-from-SOZ gradient (within-electrode, ≤63mm).
Outcome-stratified: Good (Engel I–II) vs Poor (Engel III–IV).

v2.0 (v21 manuscript alignment):
  - Primary statistic: nested LMM (Exponent ~ Distance_mm + (1|Patient/Electrode))
  - Universality test: Distance × Outcome interaction LMM + patient-level slopes
  - Spearman ρ retained as descriptive effect size
  - Fisher z removed from figure; replaced by interaction p and patient-level p
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from shared_config import *


# ═══════════════════════════════════════════════════════════════════
#  Statistical analysis functions
# ═══════════════════════════════════════════════════════════════════

def run_gradient_lmm(grad_df):
    """
    PRIMARY: Exponent ~ Distance_mm + (1 | Patient / Electrode)
    Accounts for nested Contact ⊂ Electrode ⊂ Patient structure.
    """
    try:
        import statsmodels.formula.api as smf
        ldf = grad_df[['Subject', 'Electrode', 'Exponent', 'Distance_mm']].copy()
        ldf['Pat_Elec'] = ldf['Subject'] + ':' + ldf['Electrode']
        model = smf.mixedlm(
            "Exponent ~ Distance_mm", ldf,
            groups=ldf["Subject"], re_formula="1",
            vc_formula={"Electrode": "0 + C(Pat_Elec)"}
        )
        res = model.fit(reml=True, disp=False)
        return dict(
            beta=res.fe_params['Distance_mm'],
            se=res.bse['Distance_mm'],
            z=res.tvalues['Distance_mm'],
            p=res.pvalues['Distance_mm'],
            patient_var=float(res.cov_re.iloc[0, 0]),
            residual_var=res.scale,
            n_contacts=len(ldf),
            n_electrodes=ldf['Pat_Elec'].nunique(),
            n_patients=ldf['Subject'].nunique(),
            converged=True,
        )
    except Exception as e:
        print(f"  ⚠ Gradient LMM failed: {e}")
        return dict(converged=False, error=str(e))


def run_interaction_lmm(grad_df):
    """
    UNIVERSALITY TEST: Exponent ~ Distance_mm × Outcome + (1 | Patient / Electrode)
    Tests whether gradient strength differs between Good and Poor outcome.
    """
    try:
        import statsmodels.formula.api as smf
        ldf = grad_df[['Subject', 'Electrode', 'Exponent',
                        'Distance_mm', 'Outcome']].copy()
        ldf['Outcome_Poor'] = (ldf['Outcome'] == 'Poor').astype(int)
        ldf['Pat_Elec'] = ldf['Subject'] + ':' + ldf['Electrode']
        model = smf.mixedlm(
            "Exponent ~ Distance_mm * Outcome_Poor", ldf,
            groups=ldf["Subject"], re_formula="1",
            vc_formula={"Electrode": "0 + C(Pat_Elec)"}
        )
        res = model.fit(reml=True, disp=False)

        interaction_key = 'Distance_mm:Outcome_Poor'
        return dict(
            dist_beta=res.fe_params['Distance_mm'],
            dist_p=res.pvalues['Distance_mm'],
            outcome_beta=res.fe_params['Outcome_Poor'],
            outcome_p=res.pvalues['Outcome_Poor'],
            interaction_beta=res.fe_params[interaction_key],
            interaction_se=res.bse[interaction_key],
            interaction_z=res.tvalues[interaction_key],
            interaction_p=res.pvalues[interaction_key],
            converged=True,
        )
    except Exception as e:
        print(f"  ⚠ Interaction LMM failed: {e}")
        return dict(converged=False, error=str(e))


def run_patient_slopes(grad_df, min_contacts=5):
    """
    PATIENT-LEVEL SENSITIVITY: per-patient OLS slopes and Spearman ρ.
    Provides the correct analysis unit for the between-subject Outcome variable.
    """
    slopes = []
    for subj, sdf in grad_df.groupby('Subject'):
        if len(sdf) >= min_contacts:
            sl, _, _, _, _ = stats.linregress(sdf['Distance_mm'], sdf['Exponent'])
            rho, _ = stats.spearmanr(sdf['Distance_mm'], sdf['Exponent'])
            slopes.append(dict(
                Subject=subj, Outcome=sdf['Outcome'].iloc[0],
                Slope=sl, Spearman_rho=rho, N=len(sdf)))
    sdf = pd.DataFrame(slopes)
    if len(sdf) == 0:
        return None

    n_positive = (sdf['Slope'] > 0).sum()
    t_one, p_one = stats.ttest_1samp(sdf['Slope'], 0)

    good_s = sdf[sdf.Outcome == 'Good']['Slope']
    poor_s = sdf[sdf.Outcome == 'Poor']['Slope']
    t_cmp, p_cmp = stats.ttest_ind(good_s, poor_s) if len(poor_s) >= 2 else (np.nan, np.nan)
    u_cmp, p_u = (stats.mannwhitneyu(good_s, poor_s, alternative='two-sided')
                  if len(poor_s) >= 2 else (np.nan, np.nan))

    return dict(
        slopes_df=sdf,
        n_total=len(sdf), n_positive=int(n_positive),
        mean_slope=sdf['Slope'].mean(), std_slope=sdf['Slope'].std(),
        t_onesample=t_one, p_onesample=p_one,
        good_mean=good_s.mean(), good_std=good_s.std(), n_good=len(good_s),
        poor_mean=poor_s.mean(), poor_std=poor_s.std(), n_poor=len(poor_s),
        t_compare=t_cmp, p_compare=p_cmp,
        U_compare=u_cmp, p_mannwhitney=p_u,
    )


# ═══════════════════════════════════════════════════════════════════
#  Binning helper (unchanged)
# ═══════════════════════════════════════════════════════════════════

BINS_MM    = [0, 3.5, 10.5, 17.5, 24.5, 35, 50, 63]
LABELS_MM  = ['0–3.5', '3.5–10.5', '10.5–17.5', '17.5–24.5',
               '24.5–35', '35–50', '50–63']
BIN_CENTRES = [1.75, 7.0, 14.0, 21.0, 29.75, 42.5, 56.5]


def _bin_stats(df_grp: pd.DataFrame) -> pd.DataFrame:
    """Return binned mean ± SEM per distance bin for a group."""
    df_grp = df_grp.copy()
    df_grp['Dist_Bin'] = pd.cut(df_grp['Distance_mm'], bins=BINS_MM,
                                 labels=LABELS_MM, right=True)
    bs = (df_grp.groupby('Dist_Bin', observed=True)['Exponent']
          .agg(['mean', 'std', 'count']).reset_index())
    bs['sem'] = bs['std'] / np.sqrt(bs['count'])
    bs['xc']  = BIN_CENTRES[:len(bs)]
    return bs


# ═══════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════

def plot_panel(ax, df, engel=None):
    """
    Parameters
    ----------
    ax     : matplotlib Axes
    df     : seeg_contact_results DataFrame (quality-filtered)
    engel  : DataFrame[Subject, Engel, Outcome]  — loaded if None
    """
    format_ax(ax, 'B')

    # ── Load outcome labels ───────────────────────────────────────
    if engel is None:
        engel = load_engel_phase()
    outcome_map = dict(zip(engel['Subject'], engel['Outcome']))

    # ── Filter gradient contacts ──────────────────────────────────
    grad_df = df[
        df.Has_SOZ_on_Electrode &
        df.Distance_mm.notna() &
        (df.Distance_mm <= MAX_DISTANCE_MM)
    ].copy()

    grad_df['Outcome'] = grad_df['Subject'].map(outcome_map)
    grad_df = grad_df[grad_df['Outcome'].isin(['Good', 'Poor'])]

    if len(grad_df) <= 10:
        ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                ha='center', va='center', fontsize=FS_AXIS_LABEL, color='#999')
        return

    good_df = grad_df[grad_df['Outcome'] == 'Good']
    poor_df = grad_df[grad_df['Outcome'] == 'Poor']

    # ── Run analyses ──────────────────────────────────────────────
    # Descriptive Spearman (retained for interpretability)
    rho_a, _ = stats.spearmanr(grad_df['Distance_mm'], grad_df['Exponent'])

    # PRIMARY: Nested LMM
    lmm_stats = run_gradient_lmm(grad_df)

    # UNIVERSALITY: Interaction LMM + patient-level slopes
    int_stats = run_interaction_lmm(grad_df)
    pat_stats = run_patient_slopes(grad_df)

    # ── Scatter dots ──────────────────────────────────────────────
    # Raw contacts kept faint so the binned trend lines read as the signal.
    ax.scatter(good_df['Distance_mm'], good_df['Exponent'],
               color=COL_GOOD, s=7, alpha=0.16,
               edgecolors='none', zorder=2, label='Good (Engel I–II)')
    ax.scatter(poor_df['Distance_mm'], poor_df['Exponent'],
               color=COL_POOR, s=7, alpha=0.20,
               edgecolors='none', zorder=2, marker='s',
               label='Poor (Engel III–IV)')

    # ── Binned mean ± SEM lines ───────────────────────────────────
    for grp_df, color, marker, ls, lbl in [
        (good_df, COL_GOOD, 'o', '-',  'Good'),
        (poor_df, COL_POOR, 's', '--', 'Poor'),
    ]:
        bs = _bin_stats(grp_df)
        bs = bs[bs['count'] >= 3]
        if len(bs) == 0:
            continue
        ax.errorbar(
            bs['xc'], bs['mean'], yerr=bs['sem'],
            fmt=f'{marker}{ls}',
            color=color,
            markersize=6,
            markeredgecolor='white',
            markeredgewidth=0.6,
            lw=LW_REGRESSION,
            capsize=3,
            capthick=1.0,
            elinewidth=1.0,
            zorder=5,
            label=f'_{lbl} mean ± SEM',
        )

    # ── Penumbra reference line ───────────────────────────────────
    ax.axvline(28, color='#AAAAAA', ls=':', lw=1.2, zorder=1)
    ymax_data = grad_df['Exponent'].quantile(0.97)
    ax.text(29.5, ymax_data + 0.01,
            'Inhibitory\npenumbra\n(~28 mm)',
            fontsize=FS_ANNOTATION, fontstyle='italic',
            color='#777', va='top', zorder=6)

    # ── Statistics annotation boxes (v21: LMM-based) ──────────────
    def _fmt_p(p):
        return f'{p:.1e}' if p < 0.001 else f'{p:.3f}'

    # PRIMARY: Nested LMM — compact top-left box (data is sparse here: low x, low y)
    if lmm_stats.get('converged'):
        lmm_txt = (
            f"Nested LMM (primary)\n"
            f"β = {lmm_stats['beta']:.4f}/mm, p = {_fmt_p(lmm_stats['p'])}\n"
            f"N = {lmm_stats['n_contacts']} contacts (ρ = {rho_a:.2f})"
        )
    else:
        lmm_txt = f"LMM failed — see console\nSpearman ρ = {rho_a:.3f}"
    stat_box(ax, lmm_txt, x=0.02, y=0.97, ha='left', va='top')

    # Gradient consistency across outcomes (patient-level = correct unit), lower right
    if pat_stats:
        consist = (f"patient slopes +{pat_stats['n_positive']}/{pat_stats['n_total']}\n"
                   f"Good vs Poor p = {pat_stats['p_compare']:.2f} (n.s.)")
        stat_box(ax, consist, x=0.98, y=0.03, ha='right', va='bottom')

    # Full statistics to console (nothing lost; for manuscript text / legend)
    if lmm_stats.get('converged'):
        print("  [Fig2B] Nested LMM β=%.5f/mm SE=%.5f z=%.2f p=%.3e | %d contacts, "
              "%d elec, %d pts | descriptive ρ=%.3f"
              % (lmm_stats['beta'], lmm_stats['se'], lmm_stats['z'], lmm_stats['p'],
                 lmm_stats['n_contacts'], lmm_stats['n_electrodes'],
                 lmm_stats['n_patients'], rho_a))
    if int_stats.get('converged'):
        print("          Distance×Outcome interaction β=%.5f z=%.2f p=%.4f"
              % (int_stats['interaction_beta'], int_stats['interaction_z'],
                 int_stats['interaction_p']))
    if pat_stats:
        print("          Patient slopes: %d/%d positive; Good vs Poor p=%.3f (MW p=%.3f)"
              % (pat_stats['n_positive'], pat_stats['n_total'],
                 pat_stats['p_compare'], pat_stats['p_mannwhitney']))

    # ── Legend (dots only) ────────────────────────────────────────
    ax.legend(fontsize=FS_LEGEND, loc='lower right',
              framealpha=0.92, edgecolor='#CCCCCC',
              markerscale=1.4, handlelength=1.2,
              bbox_to_anchor=(0.99, 0.25))

    # ── Axes ──────────────────────────────────────────────────────
    ax.set_xlabel('Distance from SOZ center (mm)\n(within-electrode only)',
                  fontsize=FS_AXIS_LABEL - 1)
    ax.set_ylabel('Aperiodic Exponent', fontsize=FS_AXIS_LABEL)
    ax.set_xlim(-1.5, MAX_DISTANCE_MM + 3)

    y_lo = grad_df['Exponent'].quantile(0.01) - 0.05
    y_hi = grad_df['Exponent'].quantile(0.99) + 0.18
    ax.set_ylim(y_lo, y_hi)


# ═══════════════════════════════════════════════════════════════════
#  Standalone execution
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    setup_style()
    df    = load_seeg_results()
    engel = load_engel_phase()
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    plot_panel(ax, df, engel)
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / 'fig2b_gradient.png', dpi=DPI)
    fig.savefig(SCRIPT_DIR / 'fig2b_gradient.pdf')

    # ── Print full stats summary for manuscript verification ──────
    outcome_map = dict(zip(engel['Subject'], engel['Outcome']))
    grad = df[df.Has_SOZ_on_Electrode &
              df.Distance_mm.notna() &
              (df.Distance_mm <= MAX_DISTANCE_MM)].copy()
    grad['Outcome'] = grad['Subject'].map(outcome_map)
    grad = grad[grad['Outcome'].isin(['Good', 'Poor'])]

    print("\n  === Nested LMM (PRIMARY) ===")
    lmm = run_gradient_lmm(grad)
    if lmm.get('converged'):
        print(f"  β = {lmm['beta']:.6f}/mm, SE = {lmm['se']:.6f}, "
              f"z = {lmm['z']:.2f}, p = {lmm['p']:.2e}")

    print("\n  === Interaction LMM ===")
    intx = run_interaction_lmm(grad)
    if intx.get('converged'):
        print(f"  Distance×Outcome: β = {intx['interaction_beta']:.6f}, "
              f"z = {intx['interaction_z']:.2f}, p = {intx['interaction_p']:.4f}")

    print("\n  === Patient-level slopes ===")
    pat = run_patient_slopes(grad)
    if pat:
        print(f"  {pat['n_positive']}/{pat['n_total']} positive slopes")
        print(f"  One-sample t = {pat['t_onesample']:.2f}, p = {pat['p_onesample']:.4f}")
        print(f"  Good: {pat['good_mean']:.5f} ± {pat['good_std']:.5f} (n={pat['n_good']})")
        print(f"  Poor: {pat['poor_mean']:.5f} ± {pat['poor_std']:.5f} (n={pat['n_poor']})")
        print(f"  t-test p = {pat['p_compare']:.4f}, MW p = {pat['p_mannwhitney']:.4f}")

    print("\n  === Descriptive Spearman ===")
    for lbl in ['Good', 'Poor']:
        g = grad[grad['Outcome'] == lbl]
        r, p = stats.spearmanr(g['Distance_mm'], g['Exponent'])
        print(f"  {lbl:4s}: N={len(g):4d}, {g.Subject.nunique()} pts, "
              f"ρ={r:.3f}, p={p:.3e}")

    print("\nSaved fig2b_gradient")
    plt.close()
