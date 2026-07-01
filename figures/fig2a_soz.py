#!/usr/bin/env python3
"""
Fig2A — Within-patient paired SOZ vs Non-SOZ aperiodic exponent.
Wilcoxon signed-rank + LMM + PSD inset example.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from shared_config import *


def run_paired_analysis(df):
    """Per-patient Mean_SOZ − Mean_NonSOZ."""
    sub_diffs = []
    for sub, grp in df.groupby('Subject'):
        soz  = grp[grp.Is_SOZ]['Exponent'].values
        nsoz = grp[~grp.Is_SOZ]['Exponent'].values
        if len(soz) >= 2 and len(nsoz) >= 2:
            sub_diffs.append({
                'Subject': sub,
                'SOZ_mean': np.mean(soz), 'NonSOZ_mean': np.mean(nsoz),
                'Delta': np.mean(soz) - np.mean(nsoz),
                'SOZ_n': len(soz), 'NonSOZ_n': len(nsoz),
            })
    sm = pd.DataFrame(sub_diffs)
    st = {}
    if len(sm) >= 5:
        w_stat, w_p = stats.wilcoxon(sm['SOZ_mean'], sm['NonSOZ_mean'],
                                     alternative='less')
        t_stat, t_p = stats.ttest_rel(sm['SOZ_mean'], sm['NonSOZ_mean'])
        d = sm['Delta'].mean() / sm['Delta'].std() if sm['Delta'].std() > 0 else 0
        st = dict(N=len(sm),
                  SOZ_mean=sm['SOZ_mean'].mean(), SOZ_std=sm['SOZ_mean'].std(),
                  NonSOZ_mean=sm['NonSOZ_mean'].mean(), NonSOZ_std=sm['NonSOZ_mean'].std(),
                  Delta_mean=sm['Delta'].mean(), Delta_std=sm['Delta'].std(),
                  Wilcoxon_W=w_stat, Wilcoxon_p=w_p,
                  ttest_t=t_stat, ttest_p=t_p, d_paired=d)
    return sm, st


def run_lmm(df):
    """LMM: Exponent ~ Is_SOZ + (1|Subject)."""
    try:
        import statsmodels.formula.api as smf
        ldf = df[['Subject', 'Exponent', 'Is_SOZ']].copy()
        ldf['SOZ_int'] = ldf['Is_SOZ'].astype(int)
        model = smf.mixedlm("Exponent ~ SOZ_int", ldf, groups=ldf["Subject"])
        res = model.fit(reml=True, disp=False)
        rv = float(res.cov_re.iloc[0, 0])
        return dict(
            beta=res.fe_params['SOZ_int'], se=res.bse_fe['SOZ_int'],
            t=res.tvalues['SOZ_int'], p=res.pvalues['SOZ_int'],
            intercept=res.fe_params['Intercept'],
            random_var=rv, residual_var=res.scale,
            ICC=rv / (rv + res.scale),
        )
    except Exception as e:
        print(f"  ⚠ LMM failed: {e}")
        return None


def run_lmm_random_slope(df):
    """Sensitivity: Exponent ~ Is_SOZ + (1 + Is_SOZ | Subject)."""
    try:
        import statsmodels.formula.api as smf
        ldf = df[['Subject', 'Exponent', 'Is_SOZ']].copy()
        ldf['SOZ_int'] = ldf['Is_SOZ'].astype(int)
        model = smf.mixedlm("Exponent ~ SOZ_int", ldf,
                             groups=ldf["Subject"],
                             re_formula="1 + SOZ_int")
        res = model.fit(reml=True, disp=False)
        slope_var = float(res.cov_re.iloc[1, 1]) if res.cov_re.shape[0] >= 2 else None
        return dict(
            beta=res.fe_params['SOZ_int'], se=res.bse_fe['SOZ_int'],
            t=res.tvalues['SOZ_int'], p=res.pvalues['SOZ_int'],
            slope_var=slope_var,
            converged=True,
        )
    except Exception as e:
        print(f"  ⚠ Random-slope LMM failed: {e}")
        return dict(converged=False, error=str(e))


def plot_panel(ax, df):
    """
    Parameters
    ----------
    ax : matplotlib Axes
    df : seeg_contact_results DataFrame
    """
    format_ax(ax, 'A')

    sm, paired_stats = run_paired_analysis(df)
    lmm_stats = run_lmm(df)
    rs_stats = run_lmm_random_slope(df)

    if len(sm) > 0:
        # Paired lines
        for _, row in sm.iterrows():
            ax.plot([0, 1], [row['NonSOZ_mean'], row['SOZ_mean']],
                    color='#888', alpha=ALPHA_PAIRED_LINE, lw=LW_PAIRED_LINE, zorder=1)

        # Jitter dots
        np.random.seed(42)
        j0 = np.random.uniform(-0.06, 0.06, len(sm))
        j1 = np.random.uniform(-0.06, 0.06, len(sm))
        ax.scatter(0 + j0, sm['NonSOZ_mean'], s=45, color=COL_NSOZ, **SCATTER_KW)
        ax.scatter(1 + j1, sm['SOZ_mean'], s=45, color=COL_SOZ, **SCATTER_KW)

        # Mean ± SEM
        draw_mean_sem(ax, 0, sm['NonSOZ_mean'], COL_NSOZ, width=0.15, cap=0)
        draw_mean_sem(ax, 1, sm['SOZ_mean'], COL_SOZ, width=0.15, cap=0)

        # Bracket — use LMM p as primary (two-tailed, no one-sided concern)
        if paired_stats:
            y_top = max(sm['SOZ_mean'].max(), sm['NonSOZ_mean'].max()) + 0.05
            bracket_p = lmm_stats['p'] if lmm_stats else paired_stats['Wilcoxon_p']
            add_bracket(ax, 0, 1, y_top, bracket_p)

        # Compact on-figure box (primary LMM only); full stats → console below
        def _p(p):
            return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"
        box_lines = [f"N = {paired_stats['N']} patients (paired)"]
        if lmm_stats:
            box_lines.append(f"LMM β = {lmm_stats['beta']:.3f}, p = {_p(lmm_stats['p'])}")
        box_lines.append(f"SOZ < Non-SOZ,  d = {paired_stats['d_paired']:.2f}")
        stat_box(ax, '\n'.join(box_lines), fontsize=FS_STAT_BOX)

        # Full statistics to console (nothing lost; for manuscript text / legend)
        print("  [Fig2A] N=%d  SOZ %.3f±%.3f  Non-SOZ %.3f±%.3f  Δ=%+.3f  d=%.2f"
              % (paired_stats['N'], paired_stats['SOZ_mean'], paired_stats['SOZ_std'],
                 paired_stats['NonSOZ_mean'], paired_stats['NonSOZ_std'],
                 paired_stats['Delta_mean'], paired_stats['d_paired']))
        if lmm_stats:
            print("          LMM β=%.4f SE=%.4f z=%.2f p=%.3e ICC=%.2f"
                  % (lmm_stats['beta'], lmm_stats['se'], lmm_stats['t'],
                     lmm_stats['p'], lmm_stats['ICC']))
        if rs_stats and rs_stats.get('converged'):
            print("          Random-slope β=%.4f p=%.4f" % (rs_stats['beta'], rs_stats['p']))
        print("          Wilcoxon p=%.4f (1t); Paired-t p=%.4f (2t)"
              % (paired_stats['Wilcoxon_p'], paired_stats['ttest_p']))

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Non-SOZ\ncontacts', 'SOZ\ncontacts'], fontsize=FS_AXIS_LABEL)
    ax.set_ylabel('Aperiodic Exponent (N2 sleep)', fontsize=FS_AXIS_LABEL)
    ax.set_xlim(-0.5, 1.5)


# ── Standalone ──
if __name__ == "__main__":
    setup_style()
    df = load_seeg_results()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    plot_panel(ax, df)
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / 'fig2a_soz.png')
    fig.savefig(SCRIPT_DIR / 'fig2a_soz.pdf')
    print(f"Saved fig2a  ({df.Subject.nunique()} subjects, {len(df)} contacts)")
    plt.close()
