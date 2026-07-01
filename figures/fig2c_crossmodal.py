#!/usr/bin/env python3
"""
Fig2C — Cross-modal bridge: SEEG SOZ–NonSOZ exponent contrast
vs post-surgery scalp midline ΔExponent.

Analysis hierarchy:
  1. Unstratified correlation (N=16)
  2. GLM interaction (Contrast × Outcome)
  3. Good-outcome exploratory subgroup (n=11)
  4. Partial correlation controlling for pre-op midline baseline
  5. Leave-one-out Spearman sensitivity (Good subgroup)   [v22]
  6. Cook's distance for influential point detection       [v22]

AUDIT FIX (2026-02-28): X variable = within-patient SOZ−NonSOZ contrast
(not raw SOZ exponent). Verified: all passage numbers match.
v22 UPDATE: Added LOO sensitivity + Cook's D per editorial review.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf
from shared_config import *


def _partial_corr(x, y, z):
    """Partial Pearson correlation r(x,y|z), with df = n-3."""
    r_xy = np.corrcoef(x, y)[0, 1]
    r_xz = np.corrcoef(x, z)[0, 1]
    r_yz = np.corrcoef(y, z)[0, 1]
    r_partial = (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    n = len(x)
    df = n - 3
    t_val = r_partial * np.sqrt(df / (1 - r_partial**2))
    p_val = 2 * stats.t.sf(abs(t_val), df)
    return r_partial, p_val


def plot_panel(ax, df_seeg, scalp_delta=None, engel_map=None):
    """
    Parameters
    ----------
    ax          : matplotlib Axes
    df_seeg     : seeg_contact_results DataFrame
    scalp_delta : dict {Subject: midline ΔExp}  (auto-loaded if None)
    engel_map   : dict {Subject: 'Good'/'Poor'}  (auto-loaded if None)
    """
    format_ax(ax, 'C')
    ax.set_title('C', fontsize=FS_PANEL_LABEL, fontweight='bold',
                 loc='left', x=-0.12, pad=8)

    if scalp_delta is None:
        scalp_delta = load_scalp_midline_delta()
    if engel_map is None:
        engel = load_engel_phase()
        engel_map = dict(zip(engel.Subject, engel.Outcome))

    # ── X variable: within-patient SOZ − NonSOZ exponent contrast ──
    soz_per_sub  = df_seeg[df_seeg.Is_SOZ].groupby('Subject')['Exponent'].mean()
    nsoz_per_sub = df_seeg[~df_seeg.Is_SOZ].groupby('Subject')['Exponent'].mean()

    # Pre-operative midline baseline (for partial correlation)
    df_scalp = load_scalp_results()
    pre_midline = (df_scalp[(df_scalp.Condition == 'Pre') &
                            (df_scalp.Channel.isin(MIDLINE))]
                   .groupby('Subject_ID')['Exponent'].mean())

    cross_data = []
    for sub in soz_per_sub.index:
        if sub in scalp_delta and sub in nsoz_per_sub.index:
            cross_data.append({
                'Subject': sub,
                'Contrast': soz_per_sub[sub] - nsoz_per_sub[sub],
                'Scalp_Delta': scalp_delta[sub],
                'Outcome': engel_map.get(sub, 'Unknown'),
                'Pre_Midline': pre_midline.get(sub, np.nan),
            })
    cd = pd.DataFrame(cross_data)

    if len(cd) < 5:
        ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                ha='center', va='center', fontsize=FS_AXIS_LABEL, color='#999')
        return

    good = cd[cd.Outcome == 'Good']
    poor = cd[cd.Outcome == 'Poor']

    # ── Scatter by outcome ─────────────────────────────────────
    ax.scatter(good['Contrast'], good['Scalp_Delta'],
               c=COL_GOOD, marker='o', s=55, edgecolors='white',
               linewidths=0.6, zorder=3, label='Good (Engel I–II)')
    ax.scatter(poor['Contrast'], poor['Scalp_Delta'],
               c=COL_POOR, marker='s', s=55, edgecolors='white',
               linewidths=0.6, zorder=3, label='Poor (Engel III–IV)')

    # ── Good-outcome regression line + CI band ─────────────────
    if len(good) >= 5:
        x_g = good['Contrast'].values
        y_g = good['Scalp_Delta'].values
        slope_g, intercept_g = np.polyfit(x_g, y_g, 1)
        x_fit = np.linspace(x_g.min() - 0.05, x_g.max() + 0.05, 100)
        y_fit = slope_g * x_fit + intercept_g

        ax.plot(x_fit, y_fit, '-', color=COL_GOOD, lw=LW_REGRESSION,
                alpha=0.85, zorder=2)

        # 95% CI band
        n_g = len(x_g)
        y_pred = slope_g * x_g + intercept_g
        se_resid = np.sqrt(np.sum((y_g - y_pred)**2) / (n_g - 2))
        x_mean = x_g.mean()
        se_fit = se_resid * np.sqrt(1/n_g + (x_fit - x_mean)**2 /
                                    np.sum((x_g - x_mean)**2))
        t_crit = stats.t.ppf(0.975, n_g - 2)
        ax.fill_between(x_fit,
                         y_fit - t_crit * se_fit,
                         y_fit + t_crit * se_fit,
                         color=COL_GOOD, alpha=0.12, zorder=1)

    # ── Poor-outcome regression line + CI band ──────────────────
    if len(poor) >= 3:
        x_p = poor['Contrast'].values
        y_p = poor['Scalp_Delta'].values
        slope_p, intercept_p = np.polyfit(x_p, y_p, 1)
        x_fit_p = np.linspace(x_p.min() - 0.05, x_p.max() + 0.05, 100)
        y_fit_p = slope_p * x_fit_p + intercept_p

        ax.plot(x_fit_p, y_fit_p, '-', color=COL_POOR, lw=LW_REGRESSION,
                alpha=0.85, zorder=2)

        # 95% CI band
        n_p = len(x_p)
        y_pred_p = slope_p * x_p + intercept_p
        se_resid_p = np.sqrt(np.sum((y_p - y_pred_p)**2) / max(n_p - 2, 1))
        x_mean_p = x_p.mean()
        ss_x_p = np.sum((x_p - x_mean_p)**2)
        if ss_x_p > 0 and n_p > 2:
            se_fit_p = se_resid_p * np.sqrt(1/n_p + (x_fit_p - x_mean_p)**2 / ss_x_p)
            t_crit_p = stats.t.ppf(0.975, n_p - 2)
            ax.fill_between(x_fit_p,
                             y_fit_p - t_crit_p * se_fit_p,
                             y_fit_p + t_crit_p * se_fit_p,
                             color=COL_POOR, alpha=0.10, zorder=1)

    # ── Overall dashed regression line ─────────────────────────
    slope_all, intercept_all = np.polyfit(cd['Contrast'], cd['Scalp_Delta'], 1)
    x_all = np.linspace(cd['Contrast'].min() - 0.05,
                        cd['Contrast'].max() + 0.05, 100)
    ax.plot(x_all, slope_all * x_all + intercept_all, '--', color='#999',
            lw=1.0, alpha=0.6, zorder=1)

    # ── Statistics ─────────────────────────────────────────────
    # 1. Unstratified
    rho_all, p_rho_all = stats.spearmanr(cd['Contrast'], cd['Scalp_Delta'])
    r_all, p_r_all = stats.pearsonr(cd['Contrast'], cd['Scalp_Delta'])

    # 2. GLM interaction
    cd_m = cd.copy()
    cd_m['Outcome_bin'] = (cd_m['Outcome'] == 'Good').astype(int)
    glm = smf.ols("Scalp_Delta ~ Contrast * Outcome_bin", data=cd_m).fit()
    beta_int = glm.params.get('Contrast:Outcome_bin', np.nan)
    p_int = glm.pvalues.get('Contrast:Outcome_bin', np.nan)
    ci_int = glm.conf_int().loc['Contrast:Outcome_bin']

    # 3. Good subgroup
    rho_g, p_rho_g = stats.spearmanr(good['Contrast'], good['Scalp_Delta'])
    r_g, p_r_g = stats.pearsonr(good['Contrast'], good['Scalp_Delta'])

    # 4. Partial correlation (controlling pre-op midline baseline)
    r_partial, p_partial = _partial_corr(
        good['Contrast'].values, good['Scalp_Delta'].values,
        good['Pre_Midline'].values)

    # 5. Leave-one-out Spearman sensitivity (Good subgroup)  [v22]
    loo_rhos = []
    for i in range(len(good)):
        loo = good.drop(good.index[i])
        r_loo, _ = stats.spearmanr(loo['Contrast'], loo['Scalp_Delta'])
        loo_rhos.append(r_loo)

    # 6. Cook's distance (OLS on Good subgroup)  [v22]
    ols_good = smf.ols("Scalp_Delta ~ Contrast", data=good).fit()
    influence = ols_good.get_influence()
    cooks_d = influence.cooks_distance[0]
    cooks_threshold = 4.0 / len(good)

    # ── Stats box 1: Overall + interaction (upper right) ───────
    def _fp(p):
        return f'{p:.1e}' if p < 0.001 else f'{p:.3f}'

    box1 = (f'Overall (N={len(cd)}): ρ = {rho_all:.2f}, p = {_fp(p_rho_all)}\n'
            f'Interaction p = {_fp(p_int)}')
    ax.text(0.97, 0.97, box1,
            transform=ax.transAxes, ha='right', va='top',
            fontsize=FS_STAT_BOX, family='monospace', color='#666',
            bbox=dict(boxstyle='round,pad=0.25', fc='white',
                      ec='#CCCCCC', alpha=0.92, lw=0.6))

    # Stats box 2: Good subgroup (lower left) -- compact
    box2 = (f'Good (n={len(good)}): ρ = {rho_g:.2f}, p = {_fp(p_rho_g)}\n'
            f'partial r = {r_partial:.2f}, p = {_fp(p_partial)}')
    ax.text(0.03, 0.03, box2,
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=FS_STAT_BOX, family='monospace', color=COL_GOOD,
            bbox=dict(boxstyle='round,pad=0.25', fc='white',
                      ec=COL_GOOD, alpha=0.92, lw=0.8))

    # Full statistics to console (nothing lost; for manuscript text / legend)
    print("  [Fig2C] Overall N=%d  rho=%.3f p=%.3e  r=%.3f p=%.3e"
          % (len(cd), rho_all, p_rho_all, r_all, p_r_all))
    print("          Interaction beta=%.3f p=%.4f  95%%CI[%.3f, %.3f]"
          % (beta_int, p_int, ci_int[0], ci_int[1]))
    print("          Good (n=%d): rho=%.3f p=%.4f  r=%.3f p=%.4f  partial r=%.3f p=%.4f"
          % (len(good), rho_g, p_rho_g, r_g, p_r_g, r_partial, p_partial))
    print("          Good LOO rho range [%.3f, %.3f]" % (min(loo_rhos), max(loo_rhos)))

    # ── Reference lines ───────────────────────────────────────
    ax.axhline(0, color='#999', lw=0.8, ls='--', zorder=0)
    ax.axvline(0, color='#999', lw=0.8, ls='--', zorder=0)

    ax.legend(fontsize=FS_ANNOTATION, loc='upper left', framealpha=0.9,
              edgecolor='#CCC')

    ax.set_xlabel('Pre-surgery SEEG\nSOZ\u2013NonSOZ Exponent Contrast',
                  fontsize=FS_AXIS_LABEL - 1.5)
    ax.set_ylabel('Post-surgery Scalp\nMidline \u0394Exponent',
                  fontsize=FS_AXIS_LABEL - 1.5)


# ── Standalone ──
if __name__ == "__main__":
    setup_style()
    df = load_seeg_results()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    plot_panel(ax, df)
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / 'fig2c_crossmodal.png', dpi=DPI)
    fig.savefig(SCRIPT_DIR / 'fig2c_crossmodal.pdf')

    # Print all stats for verification
    print("Fig 2C \u2014 Cross-modal Statistics")
    print("=" * 50)

    seeg = df
    scalp_delta = load_scalp_midline_delta()
    engel_phase = load_engel_phase()
    engel_map = dict(zip(engel_phase.Subject, engel_phase.Outcome))
    soz_per_sub  = seeg[seeg.Is_SOZ].groupby('Subject')['Exponent'].mean()
    nsoz_per_sub = seeg[~seeg.Is_SOZ].groupby('Subject')['Exponent'].mean()

    for sub in sorted(soz_per_sub.index):
        if sub in scalp_delta and sub in nsoz_per_sub.index:
            c = soz_per_sub[sub] - nsoz_per_sub[sub]
            d = scalp_delta[sub]
            print(f"  {sub}: Contrast={c:+.4f}  \u0394Midline={d:+.4f}  "
                  f"{engel_map.get(sub, '?')}")

    # ── LOO + Cook's D (v22 — for Supplementary Table S2) ────
    cross_data = []
    for sub in soz_per_sub.index:
        if sub in scalp_delta and sub in nsoz_per_sub.index:
            cross_data.append({
                'Subject': sub,
                'Contrast': soz_per_sub[sub] - nsoz_per_sub[sub],
                'Scalp_Delta': scalp_delta[sub],
                'Outcome': engel_map.get(sub, 'Unknown'),
            })
    cd = pd.DataFrame(cross_data)
    good = cd[cd.Outcome == 'Good'].reset_index(drop=True)

    print("\n── Leave-One-Out Spearman (Good, n=%d) ──" % len(good))
    loo_rhos = []
    for i in range(len(good)):
        loo = good.drop(i)
        r_loo, _ = stats.spearmanr(loo['Contrast'], loo['Scalp_Delta'])
        loo_rhos.append(r_loo)
        print(f"  Drop {good.loc[i, 'Subject']:>6s}: \u03c1 = {r_loo:.3f}")
    print(f"  Range: [{min(loo_rhos):.3f}, {max(loo_rhos):.3f}]")

    ols_good = smf.ols("Scalp_Delta ~ Contrast", data=good).fit()
    influence = ols_good.get_influence()
    cooks_d = influence.cooks_distance[0]
    threshold = 4.0 / len(good)
    print(f"\n── Cook's Distance (threshold 4/n = {threshold:.3f}) ──")
    for i, sub in enumerate(good['Subject']):
        flag = " \u26a0 EXCEEDS" if cooks_d[i] > threshold else ""
        print(f"  {sub}: D = {cooks_d[i]:.3f}{flag}")
    print(f"  Max Cook's D = {max(cooks_d):.3f}")

    print("\nSaved fig2c_crossmodal")
    plt.close()
