"""
generate_supp_table_S6.py
─────────────────────────
Non-SOZ electrode control analysis for the inhibitory penumbra gradient.

Purpose
-------
Tests whether the distance-dependent aperiodic exponent gradient observed on
SOZ-containing electrode trajectories is also present on non-SOZ electrodes.
If the gradient is absent on non-SOZ electrodes, this rules out the alternative
explanation that the gradient is a generic feature of cortical penetration depth
(cytoarchitectural variation along the trajectory).

Input
-----
seeg_contact_results.csv   (canonical quality-controlled contact-level data)

Output
------
Supplementary_Table_S6.xlsx   (two sheets: per-patient summary + group statistics)

Method
------
For each electrode, an OLS slope of Exponent ~ position is computed:
  - SOZ electrodes: Exponent ~ Distance_mm (non-SOZ contacts only, as in main analysis)
  - Non-SOZ electrodes: Exponent ~ Contact number, converted to per-mm (÷ 3.5 mm spacing)
Per-electrode slopes are averaged within each patient, then compared via
paired t-test and Wilcoxon signed-rank test across all 16 patients.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent          # panels
# 数据根：默认 panels 的上一级（随项目移动自动生效）；绝对路径兜底
_DATA_ROOT = next(
    (p for p in [SCRIPT_DIR.parent,
                 Path("."),
                 Path(".")]
     if (p / "seeg_contact_results.csv").exists()),
    SCRIPT_DIR.parent)
INPUT_FILE = _DATA_ROOT / "seeg_contact_results.csv"
OUTPUT_FILE = SCRIPT_DIR / "Supplementary_Table_S6.xlsx"
CONTACT_SPACING_MM = 3.5
MIN_CONTACTS_PER_ELECTRODE = 3


def compute_slope(group, x_col):
    """OLS slope of Exponent ~ x_col. Returns NaN if fewer than MIN contacts."""
    x = group[x_col].values
    y = group["Exponent"].values
    if len(x) < MIN_CONTACTS_PER_ELECTRODE:
        return np.nan
    slope, _, _, _, _ = stats.linregress(x, y)
    return slope


def main():
    # ── Load data ──────────────────────────────────────────────────────────
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} contacts from {df['Subject'].nunique()} patients")

    # ── SOZ electrodes: slope on non-SOZ contacts ─────────────────────────
    soz_elec = df[df["Has_SOZ_on_Electrode"] == True]
    soz_nonsoz = soz_elec[soz_elec["Is_SOZ"] == False]

    soz_slopes = (
        soz_nonsoz.groupby(["Subject", "Electrode"])
        .apply(lambda g: compute_slope(g, "Distance_mm"))
        .dropna()
        .rename("Slope_per_mm")
    )

    # ── Non-SOZ electrodes: slope on all contacts ─────────────────────────
    nonsoz_elec = df[df["Has_SOZ_on_Electrode"] == False]

    nonsoz_slopes_raw = (
        nonsoz_elec.groupby(["Subject", "Electrode"])
        .apply(lambda g: compute_slope(g, "Contact"))
        .dropna()
        .rename("Slope_per_contact")
    )
    nonsoz_slopes = (nonsoz_slopes_raw / CONTACT_SPACING_MM).rename("Slope_per_mm")

    # ── Per-patient aggregation ────────────────────────────────────────────
    pat_soz = soz_slopes.groupby(level="Subject").mean().rename("SOZ_Slope_per_mm")
    pat_nonsoz = nonsoz_slopes.groupby(level="Subject").mean().rename("NonSOZ_Slope_per_mm")

    patient_df = pd.concat([pat_soz, pat_nonsoz], axis=1).dropna()
    patient_df["Difference"] = patient_df["SOZ_Slope_per_mm"] - patient_df["NonSOZ_Slope_per_mm"]
    patient_df.index.name = "Subject"

    n_soz_elec = soz_slopes.groupby(level="Subject").size().rename("N_SOZ_Electrodes")
    n_nonsoz_elec = nonsoz_slopes.groupby(level="Subject").size().rename("N_NonSOZ_Electrodes")
    patient_df = patient_df.join(n_soz_elec).join(n_nonsoz_elec)

    # Reorder columns
    patient_df = patient_df[
        ["N_SOZ_Electrodes", "SOZ_Slope_per_mm",
         "N_NonSOZ_Electrodes", "NonSOZ_Slope_per_mm", "Difference"]
    ]

    # ── Group-level statistics ─────────────────────────────────────────────
    soz_vals = patient_df["SOZ_Slope_per_mm"]
    nonsoz_vals = patient_df["NonSOZ_Slope_per_mm"]
    diff_vals = patient_df["Difference"]

    # One-sample tests (each group vs zero)
    t_soz, p_soz = stats.ttest_1samp(soz_vals, 0)
    t_nonsoz, p_nonsoz = stats.ttest_1samp(nonsoz_vals, 0)

    # Paired comparison
    t_paired, p_paired = stats.ttest_rel(soz_vals, nonsoz_vals)
    w_stat, p_wilcox = stats.wilcoxon(soz_vals, nonsoz_vals)
    d_paired = diff_vals.mean() / diff_vals.std(ddof=1)

    # Electrode-level comparison
    t_elec, p_elec = stats.ttest_ind(soz_slopes, nonsoz_slopes)
    u_elec, p_u_elec = stats.mannwhitneyu(soz_slopes, nonsoz_slopes, alternative="two-sided")
    pooled_sd = np.sqrt(
        ((len(soz_slopes) - 1) * soz_slopes.std(ddof=1) ** 2
         + (len(nonsoz_slopes) - 1) * nonsoz_slopes.std(ddof=1) ** 2)
        / (len(soz_slopes) + len(nonsoz_slopes) - 2)
    )
    d_elec = (soz_slopes.mean() - nonsoz_slopes.mean()) / pooled_sd

    stats_rows = [
        {"Comparison": "SOZ electrodes vs zero (per-patient)",
         "N": len(soz_vals),
         "Mean_SOZ": f"{soz_vals.mean():.6f}",
         "SD_SOZ": f"{soz_vals.std(ddof=1):.6f}",
         "Mean_NonSOZ": "",
         "SD_NonSOZ": "",
         "t_or_U": f"{t_soz:.3f}",
         "p": f"{p_soz:.4f}",
         "Cohens_d": ""},
        {"Comparison": "Non-SOZ electrodes vs zero (per-patient)",
         "N": len(nonsoz_vals),
         "Mean_SOZ": "",
         "SD_SOZ": "",
         "Mean_NonSOZ": f"{nonsoz_vals.mean():.6f}",
         "SD_NonSOZ": f"{nonsoz_vals.std(ddof=1):.6f}",
         "t_or_U": f"{t_nonsoz:.3f}",
         "p": f"{p_nonsoz:.4f}",
         "Cohens_d": ""},
        {"Comparison": "Paired t-test (SOZ vs Non-SOZ, per-patient)",
         "N": len(soz_vals),
         "Mean_SOZ": "",
         "SD_SOZ": "",
         "Mean_NonSOZ": "",
         "SD_NonSOZ": "",
         "t_or_U": f"{t_paired:.3f}",
         "p": f"{p_paired:.4f}",
         "Cohens_d": f"{d_paired:.3f}"},
        {"Comparison": "Wilcoxon signed-rank (SOZ vs Non-SOZ, per-patient)",
         "N": len(soz_vals),
         "Mean_SOZ": "",
         "SD_SOZ": "",
         "Mean_NonSOZ": "",
         "SD_NonSOZ": "",
         "t_or_U": f"W={w_stat:.0f}",
         "p": f"{p_wilcox:.4f}",
         "Cohens_d": ""},
        {"Comparison": f"Independent t-test (electrode-level; {len(soz_slopes)} SOZ vs {len(nonsoz_slopes)} non-SOZ)",
         "N": len(soz_slopes) + len(nonsoz_slopes),
         "Mean_SOZ": f"{soz_slopes.mean():.6f}",
         "SD_SOZ": f"{soz_slopes.std(ddof=1):.6f}",
         "Mean_NonSOZ": f"{nonsoz_slopes.mean():.6f}",
         "SD_NonSOZ": f"{nonsoz_slopes.std(ddof=1):.6f}",
         "t_or_U": f"{t_elec:.3f}",
         "p": f"{p_elec:.4f}",
         "Cohens_d": f"{d_elec:.3f}"},
        {"Comparison": f"Mann-Whitney U (electrode-level)",
         "N": len(soz_slopes) + len(nonsoz_slopes),
         "Mean_SOZ": "",
         "SD_SOZ": "",
         "Mean_NonSOZ": "",
         "SD_NonSOZ": "",
         "t_or_U": f"U={u_elec:.0f}",
         "p": f"{p_u_elec:.4f}",
         "Cohens_d": ""},
    ]
    stats_df = pd.DataFrame(stats_rows)

    # ── Write Excel ────────────────────────────────────────────────────────
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        patient_df.to_excel(writer, sheet_name="Per-Patient Slopes")
        stats_df.to_excel(writer, sheet_name="Group Statistics", index=False)

    print(f"\nOutput written to {OUTPUT_FILE}")
    print(f"  Sheet 1: Per-Patient Slopes ({len(patient_df)} patients)")
    print(f"  Sheet 2: Group Statistics ({len(stats_df)} tests)")

    # ── Console summary ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("KEY RESULTS (for manuscript cross-check)")
    print("=" * 65)
    print(f"SOZ electrodes:     mean slope = {soz_vals.mean():.4f}/mm, "
          f"t vs 0 = {t_soz:.2f}, p = {p_soz:.4f}")
    print(f"Non-SOZ electrodes: mean slope = {nonsoz_vals.mean():.4f}/mm, "
          f"t vs 0 = {t_nonsoz:.2f}, p = {p_nonsoz:.4f}")
    print(f"Paired comparison:  t = {t_paired:.2f}, p = {p_paired:.4f}, d = {d_paired:.2f}")
    print(f"Wilcoxon:           W = {w_stat:.0f}, p = {p_wilcox:.4f}")


if __name__ == "__main__":
    main()
