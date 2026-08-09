# Analysis Outputs Reference

A data dictionary for the feature→mCherry and segmentation analysis output tables, so an
analysis report can be built locally without re-reading the generating code. Covers the
CSVs under `results/feature_to_mcherry/<LABEL>/` and `results/dataset_analysis/<LABEL>/`.

`<LABEL>` ∈ `HD1509`, `HD1883`, `SA110`, `Ew2-1`, `Ew2-2`.

> **Provenance caveat (important).** All numbers below were regenerated on
> **2026-08-06** after the `masks_3d` uint8-wrap fix (see
> `docs/utils/plan_uint8_masks3d_wrap_fix.md`). The **pre-fix (corrupted)** copies are
> archived at `results/feature_to_mcherry/<LABEL>_corrupted_pre_uint8fix/` for
> before/after diffing. `cell_population` is new (no corrupted archive).

---

## Shared conventions

- **`sample_id` / `well`** — plate well id, e.g. `E07`, `N11`. One field, two names across tables.
- **`timepoint`** — acquisition index (this processed tree has 36: `1, 11, 21, … 351`).
  **1 index = 10 min.** **`ti`** is the integer form used for sorting.
- **Targets** — per-cell mCherry intensity percentiles: `percentile_75`, `percentile_90`,
  `percentile_95` (a.k.a. p75/p90/p95). The regressions predict these from morphology.
- **R² protocol** — model is `LGBMRegressor` (LightGBM) unless a `model` column says
  otherwise. Cross-validation is **5-fold `GroupKFold` grouped by WELL**: folds hold out
  whole wells, so R² measures **generalization across wells** and never leaks cells from a
  well into its own training. `within_*` R² re-centres each group's y and ŷ before scoring.
- **DMSO (vehicle) control well** — `N11` for HD1509/HD1883/SA110; `M11` for Ew2-1/Ew2-2.
- **Morphology feature set** — 12 shape features (area, perimeter, elongation, compactness,
  circularity, feret_diameter, radius_of_gyration, major_axis, minor_axis, skewness,
  kurtosis, entropy) + 2 `gabor_*` + 4 `*_intensity` (brightfield). `morphology_informativeness`
  calls the 4 intensity columns "suspect" (see `variant` below).

---

## 1. `results/feature_to_mcherry/<LABEL>/decomposition/`

Feature→mCherry R² decomposed by well and timepoint (incarta features → mCherry percentiles).

### `r2_by_well_timepoint.csv` — **well × timepoint grid** (324 rows = 9 wells × 36 t)
| column | meaning |
|---|---|
| `well`, `timepoint`, `ti` | well, acquisition index, integer index |
| `n_rows` | per-slice cell detections in that (well, t) |
| `n_distinct_cells` | distinct physical cells (3-D-linked `cell_id`) — spheroid/collapse proxy |
| `r2_percentile_75/90/95` | R² predicting each mCherry percentile within that (well, t) subset, using the pooled 5-fold-by-well OOF model |

### `r2_by_timepoint.csv` — per timepoint, wells pooled (108 = 36 t × 3 targets)
`target`, `timepoint`, `r2`, `n_cells`, `n_wells`. The R²-over-time curve (the honest,
time-resolved view). **Use this, not the aggregated floor, for the time trend.**

### `r2_by_well.csv` — per well, timepoints pooled (27 = 9 wells × 3 targets)
`target`, `well`, `lowo_r2` (leave-one-well-out: generalize *to* this well),
`within_well_r2` (internal CV grouped by timepoint inside the well), `n_cells`.

### `nested_r2.csv` — variance decomposition (9 = 3 levels × 3 targets)
`target`, `level` ∈ {`pooled(5fold)`, `within_timepoint`, `within_well_timepoint`}, `r2`, `n`.
How much of the pooled R² survives removing well/time context (pure per-cell biology vs
drift/drug context).

### `well_collapse_summary.csv` — per-well collapse (9 rows)
`well`, `early_r2`, `thr` (0.5×early or 0.10), `t_cross` (first t where R² stays below `thr`),
`ndistinct_at_cross`, `ndistinct_max`, `ndistinct_min`.

### `r2_cumulative_by_well.csv` — cumulative-from-t1 (360 = (9 wells + `ALL`) × 36)
`well`, `cutoff_ti`, `r2_p90` (R² over cells with `ti ≤ cutoff`), `n_cells_cum`. The knee =
inclusion cutoff (where adding later collapsed timepoints starts lowering pooled R²).

### `r2_rolling_by_well.csv` — 3-timepoint rolling window (360)
`well`, `center_ti`, `r2_p90`, `n_cells_win`.

### `decomposition_summary.md`
Human-readable headline (p90 nested table + per-well LOWO + early/late R²).

---

## 2. `results/dataset_analysis/<LABEL>/cell_population/`

**Segmentation-only** (mCherry-free) population over time. See
`docs/dataset_analysis/plan_cell_population_over_time.md`.

### `cell_population.csv` — well × timepoint (324 rows)
| column | meaning |
|---|---|
| `sample_id`, `timepoint`, `ti` | well, index, integer index |
| `n_cells` | distinct physical cells (`cell_id.nunique()`) — **primary count** |
| `coverage_fraction` | mean over z of Σ(cell area)/FOV — confluence proxy (0–1). *Caveat: mean-over-z dilutes it (empty top/bottom slices); peak reads ~0.08–0.28.* |
| `drug` | plate-layout annotation (`empty`/drug/control) |
| `is_dmso` | True for the DMSO reference well |

Plots: `n_cells_over_time.png`, `coverage_over_time.png` (per-well curves, DMSO bold).

---

## 3. `results/feature_to_mcherry/<LABEL>/morphology_informativeness/`

The **aggregated** informativeness floor + univariate + noise ceiling.

### `floor_metrics.csv` (12 = 2 variants × 2 models × 3 targets)
`variant` (`with_suspect` includes the 4 BF intensity features; `without_suspect` = 14 clean),
`model` (`ridge` linear floor / `gradient_boosting` nonlinear floor), `backend`, `target`,
`tau`, `mae`, `r2`, `pinball_loss`. **`r2` is a single pooled number** (5-fold grouped-by-well).
> **Caveat:** pooled across all timepoints, so it is **time-confounded** (weighted toward the
> dense, late, degraded-target regime). For the informativeness conclusion read
> `decomposition/r2_by_timepoint.csv`, ideally within the healthy window.

### `univariate_correlations.csv` (540 rows)
`feature`, `target`, `scope`, `group_id`, `rho` (Spearman), `pvalue`, `n`.

### `noise_ceiling.csv` (3 rows)
`target`, `method`, `ceiling`, `n_conditions`, `n_replicate_wells`, `reason`.
> **Caveat:** `ceiling` is empty here — no drug/dose condition has ≥2 replicate wells with
> matched targets, so ICC(1) is not estimable (`reason` explains). Not a bug.

### `summary.json`
Run summary: `n_cells`, feature counts, `suspect_feature_names`, embedded `floor_metrics`.

---

## 4. `results/feature_to_mcherry/<LABEL>/data_quality/`

### `extreme_value_report.csv` (~1690 rows)
`source`, `value_column`, `group_type` (e.g. `sample_id`), `group_value`, `n`, `n_extreme`,
`extreme_rate`, `overall_rate`, `enrichment` (group rate / overall rate). Flags wells/features
with disproportionate extreme values.

### `well_timepoint_report.html`
Interactive Plotly report: median + IQR **per well across timepoint** for every feature and
target column. (Co-plots the mCherry targets alongside features — it is a QC report, not a
mCherry-free analysis.)

---

## Building a report — question → file

| Question | File(s) |
|---|---|
| Overall morphology→mCherry predictability | `nested_r2.csv` (pooled), `floor_metrics.csv` |
| Does predictability decline over time? | `r2_by_timepoint.csv`, `r2_cumulative_by_well.csv` |
| Pure per-cell biology vs well/time context | `nested_r2.csv` (within_* levels) |
| Which wells generalize? | `r2_by_well.csv` (`lowo_r2`) |
| Cell population / spheroid collapse | `cell_population.csv`, `well_collapse_summary.csv` |
| Which single features associate with mCherry | `univariate_correlations.csv` |
| Data quality / outlier wells | `data_quality/*` |
| Before/after the uint8 fix | diff `<LABEL>/…` vs `<LABEL>_corrupted_pre_uint8fix/…` |
