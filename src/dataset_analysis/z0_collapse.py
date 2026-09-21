"""Collapse metrics and per-drug figures for the z0 projection population.

Two jobs, both downstream of ``z0_population``:

1. **Recompute ``t_cross`` on the z0 counts** through the *shipped*
   :func:`collapse_summary.summarize_experiment`, passing ``count_column="n_objects"``.
   (Per experiment rather than via ``summarize_all_experiments``, which reads the
   CSVs itself and would consume width-flagged counts as data.) Reusing the shipped
   metric rather than reimplementing the threshold is the point of the exercise: any
   difference from the published table is then attributable to the input, not to a
   second implementation of ``_first_crossing``.

2. **Reduce each well to its merging-vs-loss signature.** ``t_cross`` is count-only, so
   it cannot tell aggregation from detection loss. The count/coverage/mean-object-size
   triple can: merging conserves foreground while cutting the count, so it must inflate
   mean object size; detection loss cuts both and leaves the size flat.

Figures mirror ``feature_to_mcherry.dataset_design_figures`` (viridis dose ramp,
rank 1 = top dose, DMSO bold black on top) but deliberately do **not** import it:
``feature_to_mcherry`` already imports ``dataset_analysis``
(``informativeness/noise_ceiling.py``, ``data/normalize.py``), so importing back would
close an import cycle.

See ``docs/dataset_analysis/plan_z0_projection_count_check.md``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.dataset_analysis.collapse_summary import summarize_experiment
from src.dataset_analysis.z0_population import drop_unmeasurable

logger = logging.getLogger(__name__)

#: The z0 count column. ``cell_population`` calls its (different) quantity ``n_cells``.
Z0_COUNT_COLUMN = "n_objects"

#: Hours per frame index, mirroring the collaborator deliverable's Table 1 convention
#: (``dataset_design_figures.TI_TO_HOURS``) so the two sets of figures agree.
#:
#: NOTE, unresolved: this maps ti 1..351 onto 0..72 h, i.e. ~12.3 min per index,
#: while the 2026-08 scope decisions state "1 index = 10 min" (frame 351 would be at
#: 58.5 h, not 72 h). Both conventions are in use in this repo. Figures here follow the
#: deliverable so they can be read beside it; the axis is labelled in frame index, with
#: hours as a secondary axis, so the ambiguity stays visible rather than baked in.
TI_TO_HOURS = 72.0 / 350.0

#: Signature thresholds. Heuristic and deliberately explicit — every underlying ratio is
#: emitted alongside the label so a reader can disagree with the cut.
COUNT_DROP_MIN = 1.5  # count must fall at least this many-fold to be called anything
COVERAGE_RETAINED_MIN = 0.5  # end coverage as a fraction of peak coverage
MEAN_AREA_GROWTH_MIN = 1.15  # objects must grow this much to read as merging

SIGNATURE_MERGING = "merging"
SIGNATURE_LOSS = "detection_loss"
SIGNATURE_STABLE = "stable"
SIGNATURE_AMBIGUOUS = "ambiguous"

SIGNATURE_COLUMNS = (
    "experiment",
    "well",
    "drug",
    "dose_rank",
    "concentration_uM",
    "is_dmso",
    "peak_n_objects",
    "end_n_objects",
    "count_drop_factor",
    "peak_coverage",
    "end_coverage",
    "coverage_retained",
    "first_mean_area",
    "end_mean_area",
    "mean_area_ratio",
    "signature",
)

#: Curve labels for the shape comparison. Absolute counts are not comparable across
#: these (plan §6.4) — each is normalised to its own peak before overlaying.
SOURCE_Z0 = "z0 projection (raw)"
SOURCE_BEST_SLICE = "best optical slice (raw)"
SOURCE_TRACKED = "tracked final_2d"

SOURCE_STYLE = {
    SOURCE_Z0: {"color": "#1f77b4", "linewidth": 1.8},
    SOURCE_BEST_SLICE: {"color": "#2ca02c", "linewidth": 1.8, "linestyle": "--"},
    SOURCE_TRACKED: {"color": "#d62728", "linewidth": 1.8, "linestyle": ":"},
}

#: Plot styling, mirrored from dataset_design_figures.
DOSE_CMAP = "viridis"
DMSO_COLOR = "black"
DMSO_LINEWIDTH = 2.4
WELL_LINEWIDTH = 1.0
PANEL_METRICS = (
    ("n_objects", "distinct objects"),
    ("coverage_fraction", "coverage (foreground fraction)"),
    ("mean_object_area_fraction", "mean object area"),
)


def summarize_z0_collapse(
    z0_csvs: Mapping[str, Union[str, Path]],
    layout: Optional[Mapping[str, Any]] = None,
    dmso_wells: Optional[Mapping[str, str]] = None,
    **metric_kwargs: Any,
) -> pd.DataFrame:
    """``t_cross`` and friends for every z0 well, via the shipped collapse metric.

    Each CSV is passed through :func:`z0_population.drop_unmeasurable` and then through
    :func:`collapse_summary.summarize_experiment`.

    Args:
        z0_csvs: ``{experiment_label: path to z0_population.csv}``.
        layout: Loaded plate layout, shared across experiments.
        dmso_wells: Optional ``{experiment_label: dmso well id}``.
        **metric_kwargs: Forwarded to ``compute_collapse_metrics`` (e.g. ``edge_n``),
            so its edge window can be kept in step with
            :func:`summarize_z0_signature`'s.

    Returns:
        A ``collapse_summary.SUMMARY_COLUMNS`` table. Its ``*_n_cells`` columns hold
        **z0 object counts**, not tracked cell counts — the column names come from the
        shared metric and are not a claim that the two are the same quantity.
    """
    frames = []
    for experiment, csv_path in z0_csvs.items():
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"{experiment}: no z0_population.csv at {path}")
        # Filter here rather than calling summarize_all_experiments, which reads the
        # CSVs itself and would consume width-flagged counts as data (plan 6.2).
        table = drop_unmeasurable(pd.read_csv(path))
        frames.append(
            summarize_experiment(
                table,
                experiment,
                layout=layout,
                dmso_well=(dmso_wells or {}).get(experiment),
                count_column=Z0_COUNT_COLUMN,
                **metric_kwargs,
            )
        )
    if not frames:
        raise ValueError("no experiments given")
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["experiment", "well"])
        .reset_index(drop=True)
    )


def _edge_mean(values: "pd.Series[Any]", edge_n: int) -> float:
    """Mean of the last ``edge_n`` samples, matching ``compute_collapse_metrics``."""
    return float(values.tail(edge_n).mean())


def _classify(
    count_drop: float, coverage_retained: float, mean_area_ratio: float
) -> str:
    """Read one well against the plan's §5 table.

    Merging conserves foreground and inflates object size; detection loss destroys
    both. A well whose count barely moves is neither. A well whose count reaches zero
    is the extreme of loss, never stability.

    Note this threshold (``COUNT_DROP_MIN`` = 1.5x) is deliberately *looser* than
    ``collapse_summary.DEFAULT_COLLAPSE_FRACTION`` (0.5, i.e. a 2.0x drop), so a well
    can read "merging" here and "no material collapse" there. The two labels answer
    different questions and are not one verdict.
    """
    grew = np.isfinite(mean_area_ratio) and mean_area_ratio >= MEAN_AREA_GROWTH_MIN
    kept = np.isfinite(coverage_retained) and coverage_retained >= COVERAGE_RETAINED_MIN

    if np.isnan(count_drop):
        # Nothing measurable — not a verdict, and emphatically not "stable".
        return SIGNATURE_AMBIGUOUS
    if np.isinf(count_drop):
        # end_n == 0: every detection is gone. This is the most extreme loss there
        # is, so it must not fall through to the "count barely moved" bucket.
        return SIGNATURE_LOSS if not kept else SIGNATURE_AMBIGUOUS
    if count_drop < COUNT_DROP_MIN:
        return SIGNATURE_STABLE
    if kept and grew:
        return SIGNATURE_MERGING
    if not kept and not grew:
        return SIGNATURE_LOSS
    return SIGNATURE_AMBIGUOUS


def summarize_z0_signature(
    z0_population: pd.DataFrame,
    experiment: str,
    collapse_summary: Optional[pd.DataFrame] = None,
    edge_n: int = 3,
) -> pd.DataFrame:
    """One merging-vs-loss row per well.

    Args:
        z0_population: A ``z0_population.csv`` table for one experiment.
        experiment: Label written to the ``experiment`` column.
        collapse_summary: Optional matching :func:`summarize_z0_collapse` output, used
            only to carry ``drug``/``dose_rank``/``concentration_uM``/``is_dmso`` so the
            two tables cannot disagree about a well's annotation.
        edge_n: Samples averaged at the end, matching ``compute_collapse_metrics``.

    Returns:
        A DataFrame with :data:`SIGNATURE_COLUMNS`, one row per well.
    """
    required = {
        "sample_id",
        "ti",
        "n_objects",
        "coverage_fraction",
        "mean_object_area_fraction",
    }
    missing = required - set(z0_population.columns)
    if missing:
        raise ValueError(f"z0_population missing columns: {sorted(missing)}")

    annotation: Dict[str, Dict[str, Any]] = {}
    if collapse_summary is not None:
        subset = collapse_summary[collapse_summary["experiment"] == experiment]
        annotation = {str(row["well"]): row.to_dict() for _, row in subset.iterrows()}

    rows: List[Dict[str, Any]] = []
    for well, group in z0_population.groupby("sample_id", sort=True):
        ordered = group.sort_values("ti")
        counts = ordered["n_objects"].astype(float)
        coverage = ordered["coverage_fraction"].astype(float)
        areas = ordered["mean_object_area_fraction"].astype(float)

        peak_n = float(counts.max())
        end_n = _edge_mean(counts, edge_n)
        peak_cov = float(coverage.max())
        end_cov = _edge_mean(coverage, edge_n)
        # Mean over the same edge window as `end_area`: anchoring on a single
        # frame lets one all-background first frame (NaN, and NaN is truthy)
        # push a genuinely merging well into `ambiguous`.
        first_area = float(areas.head(edge_n).mean()) if len(areas) else float("nan")
        end_area = _edge_mean(areas, edge_n)

        # peak 0 means the well never had a detection: nothing was lost, so this is
        # undefined rather than an infinite drop (which would read as detection_loss).
        if peak_n <= 0:
            count_drop = float("nan")
        else:
            count_drop = peak_n / end_n if end_n else float("inf")
        coverage_retained = end_cov / peak_cov if peak_cov else float("nan")
        area_ratio = (
            end_area / first_area
            if np.isfinite(first_area) and first_area
            else float("nan")
        )

        info = annotation.get(str(well), {})
        rows.append(
            {
                "experiment": experiment,
                "well": well,
                "drug": info.get("drug"),
                "dose_rank": info.get("dose_rank"),
                "concentration_uM": info.get("concentration_uM"),
                "is_dmso": info.get("is_dmso"),
                "peak_n_objects": peak_n,
                "end_n_objects": end_n,
                "count_drop_factor": count_drop,
                "peak_coverage": peak_cov,
                "end_coverage": end_cov,
                "coverage_retained": coverage_retained,
                "first_mean_area": first_area,
                "end_mean_area": end_area,
                "mean_area_ratio": area_ratio,
                "signature": _classify(count_drop, coverage_retained, area_ratio),
            }
        )

    return pd.DataFrame(rows, columns=list(SIGNATURE_COLUMNS))


def _dose_colors(wells: pd.DataFrame) -> Dict[str, tuple]:
    """Well -> RGBA along the dose ladder (rank 1 = top dose = darkest)."""
    import matplotlib.pyplot as plt

    ranked = wells.dropna(subset=["dose_rank"]).sort_values("dose_rank")
    unranked = wells[wells["dose_rank"].isna()].sort_values("well")
    ordered = [str(w) for w in list(ranked["well"]) + list(unranked["well"])]

    cmap = plt.get_cmap(DOSE_CMAP)
    if not ordered:
        return {}
    if len(ordered) == 1:
        return {ordered[0]: cmap(0.25)}
    # Stop short of the ramp's pale end so the lightest dose still reads on white.
    return {
        well: cmap(0.1 + 0.75 * index / (len(ordered) - 1))
        for index, well in enumerate(ordered)
    }


def _dose_label(row: "pd.Series[Any]") -> str:
    """``"E07  75 µM"``-style label; the well id alone when no dose is defined."""
    well = str(row["well"])
    if pd.isna(row.get("concentration_uM")):
        return well
    return f"{well}  {row['concentration_uM']:g} µM"


def plot_drug_dose_panel(
    z0_population: pd.DataFrame,
    wells: pd.DataFrame,
    experiment: str,
    drug: str,
    out_png: Union[str, Path],
    dmso_well: Optional[str] = None,
    metrics: Optional[Sequence[Tuple[str, str]]] = None,
    title: Optional[str] = None,
) -> None:
    """Stacked per-dose panels of ``metrics`` for one drug.

    Defaults to :data:`PANEL_METRICS` (count / coverage / mean object size). One
    line per
    dose, coloured along the dose ladder, with the culture's DMSO well drawn bold and
    black on top as the reference. Reading the three defaults together separates genuine
    confluence (count down, coverage held, size up) from detection loss (count down,
    coverage down, size flat).

    Args:
        metrics: ``(column, y-label)`` pairs, one panel each. Any tidy per-(well,
            timepoint) frame with a ``sample_id``/``ti`` key works, which is how
            ``image_foreground`` reuses this for its mask-free measures.
        title: Overrides the first panel's title.
    """
    import matplotlib.pyplot as plt

    drug_wells = wells[wells["drug"].astype(str) == str(drug)].copy()
    drug_wells = drug_wells[~drug_wells["is_dmso"].fillna(False).astype(bool)]
    if drug_wells.empty:
        raise ValueError(f"{experiment}: no wells for drug {drug!r}")

    panels = tuple(metrics) if metrics else PANEL_METRICS
    colors = _dose_colors(drug_wells)
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 3 * len(panels)), sharex=True)
    axes = np.atleast_1d(axes)

    for panel_index, (ax, (metric, ylabel)) in enumerate(zip(axes, panels)):
        for _, row in drug_wells.sort_values("dose_rank").iterrows():
            well = str(row["well"])
            group = z0_population[z0_population["sample_id"] == well].sort_values("ti")
            if group.empty:
                continue
            ax.plot(
                group["ti"],
                group[metric],
                marker="o",
                markersize=2.5,
                linewidth=WELL_LINEWIDTH,
                color=colors.get(well),
                label=_dose_label(row) if panel_index == 0 else None,
                zorder=2,
            )

        if dmso_well is not None:
            ref = z0_population[
                z0_population["sample_id"].astype(str).str.upper()
                == str(dmso_well).upper()
            ].sort_values("ti")
            if not ref.empty:
                ax.plot(
                    ref["ti"],
                    ref[metric],
                    color=DMSO_COLOR,
                    linewidth=DMSO_LINEWIDTH,
                    label=f"{dmso_well} (DMSO)" if panel_index == 0 else None,
                    zorder=5,
                )

        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    axes[0].set_title(
        title or f"{experiment} — {drug}: z0 projection confluence measures"
    )
    axes[0].legend(fontsize="small", ncol=2)
    axes[-1].set_xlabel("frame index (ti)")

    secondary = axes[0].secondary_xaxis(
        "top",
        functions=(lambda t: (t - 1) * TI_TO_HOURS, lambda h: h / TI_TO_HOURS + 1),
    )
    secondary.set_xlabel("hours")

    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out)


def write_drug_dose_figures(
    z0_population: pd.DataFrame,
    wells: pd.DataFrame,
    experiment: str,
    out_dir: Union[str, Path],
    dmso_well: Optional[str] = None,
    metrics: Optional[Sequence[Tuple[str, str]]] = None,
    prefix: str = "z0_confluence",
    title_fmt: Optional[str] = None,
) -> List[Path]:
    """One :func:`plot_drug_dose_panel` per drug in ``wells``. Returns the paths.

    ``metrics``, ``prefix`` and ``title_fmt`` (formatted with ``experiment`` and
    ``drug``)
    let a caller render a different measure set into differently-named files without
    duplicating the panel layout.
    """
    drugs = sorted(
        {
            str(d)
            for d in wells.loc[
                ~wells["is_dmso"].fillna(False).astype(bool), "drug"
            ].dropna()
        }
    )
    written: List[Path] = []
    for drug in drugs:
        safe = str(drug).replace(" ", "_").replace("/", "-")
        path = Path(out_dir) / f"{prefix}_{safe}.png"
        plot_drug_dose_panel(
            z0_population,
            wells,
            experiment,
            drug,
            path,
            dmso_well=dmso_well,
            metrics=metrics,
            title=(
                title_fmt.format(experiment=experiment, drug=drug)
                if title_fmt
                else None
            ),
        )
        written.append(path)
    return written


# --------------------------------------------------------------------------- #
# Step 3: normalised shape comparison
# --------------------------------------------------------------------------- #


def build_shape_comparison(
    sources: Mapping[str, "tuple[pd.DataFrame, str]"],
    experiment: str,
    well_column: str = "sample_id",
) -> pd.DataFrame:
    """Tidy, peak-normalised curves from several trees, ready to overlay.

    Absolute counts differ by construction between the trees (projection objects from
    raw inference vs 3-D-linked tracked identities, plan §6.4), so each well's curve is
    divided by its own peak within its own source. Only the resulting *shape* is
    comparable.

    Args:
        sources: ``{label: (frame, count_column)}``. Each frame needs the well column
            and ``ti``.
        experiment: Written to the ``experiment`` column.
        well_column: Column holding the well id.

    Returns:
        Long frame: ``experiment``, ``well``, ``ti``, ``source``, ``value``,
        ``value_norm``.

    Raises:
        ValueError: If ``sources`` is empty or a frame lacks a required column.
    """
    if not sources:
        raise ValueError("no sources to compare")

    frames = []
    for label, (frame, column) in sources.items():
        missing = {well_column, "ti", column} - set(frame.columns)
        if missing:
            raise ValueError(f"{label}: frame missing columns {sorted(missing)}")
        tidy = frame[[well_column, "ti", column]].rename(
            columns={well_column: "well", column: "value"}
        )
        tidy = tidy.dropna(subset=["value"]).copy()
        tidy["experiment"] = experiment
        tidy["source"] = label
        peak = tidy.groupby("well")["value"].transform("max")
        tidy["value_norm"] = tidy["value"].where(peak > 0).div(peak)
        frames.append(tidy)

    out = pd.concat(frames, ignore_index=True)
    return (
        out[["experiment", "well", "ti", "source", "value", "value_norm"]]
        .sort_values(["well", "source", "ti"])
        .reset_index(drop=True)
    )


def plot_shape_grid(
    comparison: pd.DataFrame,
    experiment: str,
    out_png: Union[str, Path],
    dmso_well: Optional[str] = None,
    ncols: int = 3,
) -> Path:
    """One facet per well, overlaying each source's peak-normalised curve."""
    import matplotlib.pyplot as plt

    wells = sorted(comparison["well"].unique())
    if not wells:
        raise ValueError("comparison holds no wells")
    nrows = -(-len(wells) // ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 2.8 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    flat = [ax for row in axes for ax in row]

    for ax, well in zip(flat, wells):
        subset = comparison[comparison["well"] == well]
        for source, group in subset.groupby("source"):
            ax.plot(
                group["ti"],
                group["value_norm"],
                label=source,
                **SOURCE_STYLE.get(str(source), {}),
            )
        is_dmso = dmso_well is not None and str(well).upper() == str(dmso_well).upper()
        ax.set_title(
            f"{well}{' (DMSO)' if is_dmso else ''}",
            fontweight="bold" if is_dmso else "normal",
            fontsize=10,
        )
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)

    for ax in flat[len(wells) :]:
        ax.set_visible(False)

    # Step 4 measures the vehicle well only, so the best-slice curve lives in the DMSO
    # facet — never facet 0 after sorting. Collect handles across every facet so the
    # curve carrying Step 4's conclusion actually appears in the legend.
    handles: Dict[str, Any] = {}
    for ax in flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    if handles:
        fig.legend(
            handles.values(),
            handles.keys(),
            loc="upper right",
            fontsize="small",
        )

    fig.suptitle(
        f"{experiment} — count over time, each curve normalised to its own peak",
        fontsize=12,
    )
    fig.supxlabel("frame index (ti)")
    fig.supylabel("fraction of that curve's peak")

    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out)
    return out


# --------------------------------------------------------------------------- #
# Step 4: localise the loss across z
# --------------------------------------------------------------------------- #


def reduce_over_z(multiz: pd.DataFrame, well_column: str = "sample_id") -> pd.DataFrame:
    """Collapse a multi-plane table to the best single optical slice per timepoint.

    "Best" is the plane with the most detections — an optimistic bound on what raw
    per-slice segmentation achieved at that frame. Comparing it against the projection
    separates a projection-specific failure from one already present in the raw planes.

    Returns:
        One row per (well, ti): ``best_slice_n_objects``, ``best_z``,
        ``best_slice_coverage``, ``mean_z_coverage``, ``n_planes``.

    Raises:
        ValueError: If ``multiz`` lacks ``z_index`` or a measurement column.
    """
    required = {well_column, "ti", "z_index", "n_objects", "coverage_fraction"}
    missing = required - set(multiz.columns)
    if missing:
        raise ValueError(f"multiz frame missing columns: {sorted(missing)}")

    rows = []
    for (well, ti), group in multiz.groupby([well_column, "ti"], sort=True):
        usable = group.dropna(subset=["n_objects"])
        if usable.empty:
            rows.append(
                {
                    well_column: well,
                    "ti": ti,
                    "best_slice_n_objects": float("nan"),
                    "best_z": None,
                    "best_slice_coverage": float("nan"),
                    "mean_z_coverage": float("nan"),
                    "n_planes": 0,
                }
            )
            continue
        # Positional, not label-based: `.loc[idxmax()]` returns a frame (and then a
        # TypeError) if the caller passed a frame with a duplicated index.
        best = usable.iloc[int(usable["n_objects"].to_numpy().argmax())]
        rows.append(
            {
                well_column: well,
                "ti": ti,
                "best_slice_n_objects": float(best["n_objects"]),
                "best_z": int(best["z_index"]),
                "best_slice_coverage": float(best["coverage_fraction"]),
                "mean_z_coverage": float(usable["coverage_fraction"].mean()),
                "n_planes": int(len(usable)),
            }
        )
    return pd.DataFrame(rows)
