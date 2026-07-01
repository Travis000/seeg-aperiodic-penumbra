#!/usr/bin/env python3
"""
Fig4B — ANCOVA visualization: Pre vs Post exponent, colored by surgical outcome.
Two parallel OLS regression lines illustrate the outcome group effect
after controlling for baseline (β = −0.141, p = 0.020).

REPLACES: baseline vs ΔExponent correlation (r = −0.572)
REASON:   Oldham's method confirmed that correlation was regression-to-mean
          artefact (p = 0.880). ANCOVA provides causally valid inference.
v22 UPDATE: Added η²p to stats box + power analysis to console output.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats
from scipy.optimize import brentq
import pingouin as pg
from shared_config import *


def plot_panel(ax, data):
    """
    Parameters
    ----------
    ax   : matplotlib Axes
    data : subject-level DataFrame [Subject, Pre, Post, Delta, Outcome, ...]
    """
    format_ax(ax, 'B')

    good = data[data.Outcome == 'Good']
    poor = data[data.Outcome == 'Poor']

    # ── ANCOVA（主统计量）──
    data_m = data.copy()
    data_m['Outcome_bin'] = (data_m['Outcome'] == 'Good').astype(int)
    ancova = smf.ols("Post ~ Pre + Outcome_bin", data=data_m).fit()
    beta_outcome = ancova.params['Outcome_bin']
    p_outcome    = ancova.pvalues['Outcome_bin']
    beta_pre     = ancova.params['Pre']
    intercept    = ancova.params['Intercept']

    # ── 散点图 ──
    ax.scatter(good['Pre'], good['Post'],
               s=55, color=COL_GOOD, marker='o',
               edgecolors='white', linewidths=0.6, zorder=3,
               label='Good (Engel I–II)')
    ax.scatter(poor['Pre'], poor['Post'],
               s=55, color=COL_POOR, marker='s',
               edgecolors='white', linewidths=0.6, zorder=3,
               label='Poor (Engel III–IV)')

    # ── 两条平行回归线 + 95% CI 带 ──
    # ANCOVA假设两组斜率相同，截距相差β_outcome
    x_range = np.linspace(data['Pre'].min() - 0.05,
                          data['Pre'].max() + 0.05, 100)

    # Prediction with 95% CI for each group
    for outcome_val, color, label_suffix in [
        (1, COL_GOOD, 'Good'),
        (0, COL_POOR, 'Poor'),
    ]:
        pred_df = pd.DataFrame({
            'Pre': x_range,
            'Outcome_bin': outcome_val
        })
        pred = ancova.get_prediction(pred_df)
        pred_summary = pred.summary_frame(alpha=0.05)
        y_hat = pred_summary['mean']
        ci_low = pred_summary['mean_ci_lower']
        ci_upp = pred_summary['mean_ci_upper']

        # Shaded 95% CI band
        ax.fill_between(x_range, ci_low, ci_upp,
                        color=color, alpha=0.12, zorder=1)
        # Regression line
        ax.plot(x_range, y_hat, color=color, lw=LW_REGRESSION,
                ls='--', zorder=2, alpha=0.85)

    # Predicted y at mean Pre for arrow annotation
    y_poor_at_mean = intercept + beta_pre * data['Pre'].mean()
    y_good_at_mean = y_poor_at_mean + beta_outcome

    # ── 双向箭头标注组间差值β ──
    x_arrow = data['Pre'].mean()

    ax.annotate(
        '', 
        xy=(x_arrow + 0.04, y_good_at_mean),
        xytext=(x_arrow + 0.04, y_poor_at_mean),
        arrowprops=dict(arrowstyle='<->', color='#555',
                        lw=LW_BRACKET, mutation_scale=10)
    )
    ax.text(x_arrow + 0.055,
            (y_good_at_mean + y_poor_at_mean) / 2,
            f'β = {beta_outcome:.3f}',
            fontsize=FS_ANNOTATION, va='center', color='#444',
            fontstyle='italic')

    # ── 参考线：y = x（无变化线）──
    xy_min = min(data['Pre'].min(), data['Post'].min()) - 0.05
    xy_max = max(data['Pre'].max(), data['Post'].max()) + 0.05
    ax.plot([xy_min, xy_max], [xy_min, xy_max],
            color='#BBBBBB', lw=LW_BRACKET, ls=':', zorder=0,
            label='No change (Pre = Post)')

    # ── Stats box（ANCOVA + η²p）──
    n_good = len(good)
    n_poor = len(poor)
    r2 = ancova.rsquared
    delta_r2 = r2 - smf.ols("Post ~ Pre", data=data_m).fit().rsquared

    # η²p for Outcome term  [v22]
    t_outcome = ancova.tvalues['Outcome_bin']
    df_resid = ancova.df_resid
    eta2p = t_outcome**2 / (t_outcome**2 + df_resid)

    stat_box(ax,
             f'ANCOVA\n'
             f'Outcome: \u03b2 = {beta_outcome:.3f}\n'
             f'p = {p_outcome:.3f}\n'
             f'\u03b7\u00b2p = {eta2p:.2f},  \u0394R\u00b2 = {delta_r2:.2f}\n'
             f'N = {n_good + n_poor} (Good={n_good}, Poor={n_poor})\n'
             f'Shaded: 95% CI',
             x=0.97, y=0.05, ha='right', fontsize=FS_STAT_BOX)

    # ── p值标注（显眼位置）──
    if p_outcome < 0.05:
        p_label = f'*p = {p_outcome:.3f}'
        color_p = '#333'
        fw = 'bold'
    else:
        p_label = f'n.s. (p = {p_outcome:.2f})'
        color_p = '#888'
        fw = 'normal'

    ax.text(0.5, 0.97, p_label,
            transform=ax.transAxes,
            ha='center', va='top',
            fontsize=FS_PVAL_BRACKET, fontstyle='italic',
            fontweight=fw, color=color_p)

    ax.set_xlabel('Pre-operative Aperiodic Exponent', fontsize=FS_AXIS_LABEL)
    ax.set_ylabel('Post-operative Aperiodic Exponent', fontsize=FS_AXIS_LABEL)
    ax.legend(fontsize=FS_LEGEND, loc='upper left',
              framealpha=0.9, edgecolor='#CCC')

    # 等比例轴（使"无变化线"呈45度）
    ax.set_xlim(xy_min, xy_max)
    ax.set_ylim(xy_min, xy_max)


# ── Standalone ──
if __name__ == "__main__":
    setup_style()
    import shutil
    data = get_subject_level_data()

    fig, ax = plt.subplots(figsize=(5.5, 5))
    plot_panel(ax, data)
    fig.tight_layout()

    out_png = SCRIPT_DIR / 'fig4b_ancova.png'
    out_pdf = SCRIPT_DIR / 'fig4b_ancova.pdf'
    fig.savefig(out_png, dpi=DPI, bbox_inches='tight', facecolor='white')
    fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')

    # ANCOVA summary to console
    data_m = data.copy()
    data_m['Outcome_bin'] = (data_m['Outcome'] == 'Good').astype(int)
    m = smf.ols("Post ~ Pre + Outcome_bin", data=data_m).fit()
    print(m.summary())

    # v22: η²p + power analysis
    t_out = m.tvalues['Outcome_bin']
    df_res = m.df_resid
    eta2p = t_out**2 / (t_out**2 + df_res)
    f2 = eta2p / (1 - eta2p)
    print(f"\n── v22 additions ──")
    print(f"  η²p (Outcome) = {eta2p:.3f}")
    print(f"  f²  (Outcome) = {f2:.3f}")

    # Achieved power
    N = len(data_m)
    ncp = f2 * N
    f_crit = stats.f.ppf(0.95, 1, df_res)
    power = 1 - stats.f.cdf(f_crit, 1, df_res, ncp)
    print(f"  Achieved power = {power:.3f}")

    # Minimum detectable f² at 80% power
    def _power_func(f2_test, N=N, alpha=0.05):
        ncp_t = f2_test * N
        df_d = N - 3
        f_c = stats.f.ppf(1 - alpha, 1, df_d)
        return 1 - stats.f.cdf(f_c, 1, df_d, ncp_t) - 0.80

    f2_min = brentq(_power_func, 0.01, 5.0)
    print(f"  Min detectable f² (80% power, N={N}) = {f2_min:.3f}")

    # N needed for medium effect
    for n_test in range(10, 200):
        df_d = n_test - 3
        if df_d <= 0:
            continue
        ncp_t = 0.15 * n_test
        f_c = stats.f.ppf(0.95, 1, df_d)
        pw = 1 - stats.f.cdf(f_c, 1, df_d, ncp_t)
        if pw >= 0.80:
            print(f"  N for f²=0.15 at 80% power = {n_test}")
            break

    # pingouin cross-check
    anc_pg = pg.ancova(data=data, dv='Post', covar='Pre', between='Outcome')
    print(f"\n  pingouin ANCOVA η²p = "
          f"{anc_pg.loc[anc_pg.Source == 'Outcome', 'np2'].values[0]:.3f}")

    print(f"\nSaved: {out_png.name}")
    plt.close()
