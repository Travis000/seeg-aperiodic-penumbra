# data/

This folder is intentionally **empty** in the public code release.

Under the ethics approval governing these clinical recordings, patient data are not
distributed. The scripts operate on a small set of **de-identified derived files** that
are available per the paper's **Data availability statement** (individual-level anonymised
data from the corresponding author on reasonable request, subject to institutional
data-sharing agreements). Place them here, or point `APERIODIC_DATA` / `DATA_ROOT` at
their location (see the top-level `README.md`).

## Derived files the code expects

| File | Produced by | Contents (de-identified) |
|---|---|---|
| `scalp_results.csv` | `analysis/scalp_aperiodic.py` | Per-subject, per-channel aperiodic exponent / offset / R² for Pre and Post recordings |
| `seeg_contact_results.csv` | `analysis/seeg_aperiodic.py` | Per-contact exponent / offset / R², SOZ & ablated flags, along-electrode distance to SOZ |
| `engel_phase.xlsx` | hand-filled | Per-subject Engel outcome class and phase |
| `ROI*.csv` | hand-filled | Per-subject lesion side / inferred hemisphere (for mirror-flipping) |
| `ASM_regimen.csv` | hand-filled | Per-subject anti-seizure-medication regimen (Pre/Post) |
| `SEEG_Electrode_Targets.csv` | hand-filled | Per-contact SOZ / ablated / target-region labels |
| `patient_age.xlsx` | hand-filled | Per-subject age |
| `relapse_timeline.xlsx` | hand-filled | Per-subject RF-TC date, first-recurrence date, and post-operative EEG window (for the swimmer/relapse panels) |

Column-level schemas are evident from the loader functions in
`figures/shared_config.py` (`load_scalp_results`, `load_seeg_results`,
`load_engel_phase`, `load_roi`, …).

> No raw EDF recordings, identity mappings, or preprocessing/anonymisation scripts are
> part of this release.
