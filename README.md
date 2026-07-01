# Aperiodic EEG/SEEG analysis — inhibitory penumbra and seizure outcome after focal thermocoagulation

Analysis and figure-generation code for the study of the **aperiodic (1/f) EEG/SEEG
exponent** as a cross-scale marker of cortical excitation/inhibition (E/I) balance in
focal epilepsy treated with SEEG-guided radiofrequency thermocoagulation (RF-TC).

The code computes, from interictal N2 sleep only, the specparam/FOOOF aperiodic exponent
for scalp EEG and intracranial SEEG, and reproduces the paper's statistics (linear
mixed-effects models, distance-gradient modelling, ANCOVA, label-permutation test,
cross-modal correlation) and figures.

> Repository: https://github.com/Travis000/seeg-aperiodic-penumbra

## Repository layout

```
.
├── analysis/                  # signal → aperiodic exponent (method reference)
│   ├── scalp_aperiodic.py     #   scalp EEG:  N2 → epochs → PTP reject → Welch PSD → specparam
│   └── seeg_aperiodic.py      #   SEEG:       same pipeline per contact + SOZ/distance mapping
├── figures/                   # statistics + figure generation (runnable on derived data)
│   ├── shared_config.py       #   palette, typography, channel layouts, data loaders, DATA_ROOT
│   ├── fig1*.py fig2*.py …     #   one module per panel: computes the panel's statistic
│   │                           #   (LMM, ANCOVA, permutation, FDR, …) and renders it
│   └── generate_supplementary_tables.py, generate_supp_table_S6.py
├── data/                      # (empty) — see data/README.md and Data availability
├── requirements.txt
└── LICENSE
```

## Requirements

Python ≥ 3.10. Install dependencies:

```bash
pip install --pre -r requirements.txt
```

`--pre` is required because `specparam==2.0.0rc6` is a pre-release. Key packages:
`mne`, `yasa` (sleep staging), `specparam` (FOOOF), `numpy`/`scipy`/`pandas`,
`statsmodels` and `pingouin` (statistics), `matplotlib`/`openpyxl` (figures/tables).

## Data

Per the study's ethics approval, the underlying patient recordings are **not** included.
The scripts read a small set of **de-identified derived files** (per-contact / per-channel
exponents and de-identified clinical labels); these are described in
[`data/README.md`](data/README.md) and are available per the paper's **Data availability
statement** (individual-level anonymised data from the corresponding author on reasonable
request, subject to institutional data-sharing agreements).

## How to run

1. Place the derived result files in a folder and point `DATA_ROOT` at it, either by

   ```bash
   export APERIODIC_DATA=/path/to/your/data      # Windows: set APERIODIC_DATA=...
   ```

   or by editing the default in `figures/shared_config.py`.

2. (Optional, requires the anonymised recordings) recompute the exponents:

   ```bash
   python analysis/scalp_aperiodic.py
   python analysis/seeg_aperiodic.py
   ```

3. Reproduce the statistics, panels, and supplementary tables (runs on the derived files).
   Each `fig*.py` computes its statistic and renders its panel when run standalone, e.g.:

   ```bash
   cd figures
   python fig2b_gradient.py         # distance-gradient LMM
   python fig4b_regression.py       # outcome ANCOVA
   python fig4_perm.py              # label-permutation test
   python generate_supplementary_tables.py
   python generate_supp_table_S6.py
   ```

## Method summary

Per-recording pipeline (identical for scalp and SEEG): load recording → average
reference → FIR band-pass 0.5–45 Hz → automated sleep staging (YASA), keep N2 only →
5-s fixed epochs → adaptive peak-to-peak artefact rejection (with expert visual exclusion
of epileptiform/artefact segments) → Welch PSD (4-s FFT) → specparam fit (fixed aperiodic
mode, 1–45 Hz) → per-channel/contact exponent, offset, R². Intracranial contacts are
mapped to clinically defined SOZ/ablated status and along-electrode distance to the SOZ
centre. Group statistics and figures are produced by the `figures/` modules.

## Citation

If you use this code, please cite the associated paper (details on publication).

## License

MIT — see [LICENSE](LICENSE).
