"""Dataset-design figures: DMSO-vs-drug mCherry percentiles over time.

§4 of ``dataset_design_report/dataset_design_assessment.md`` compares each drug well
against its culture's DMSO well with a Cliff's δ, and §7a records the pre-confluence
window those comparisons run in. Both are tables. This module draws the raw values
behind them: one figure per (culture, drug), a line per well, DMSO as the reference,
and the window boundaries marked on the same axes -- so *when* a well pulls away from
DMSO is visible, not just that it eventually does.

Deliberate choices, each recorded in
``docs/feature_to_mcherry/plan_dmso_vs_drug_mcherry_percentile90_timeplot.md``:

- **Annotation comes from the summary table**, not from a plate-layout lookup.
  ``dataset_design.build_effect_table`` reads ``drug``/``dose_rank`` and
  ``concentration_uM`` off the same summary rows when it writes
  ``drug_effect_two_windows.csv``, so sourcing
  them here too means the figure and the CSV it illustrates cannot disagree about which
  well is which dose. ``report_figures.plate.well_to_drug_dose`` re-derives the same
  facts from the plate JSON; a second derivation is exactly the drift this figure exists
  to expose.
- **Nothing is imported from** ``src.feature_analysis``. Its package ``__init__``
  imports ``feature_trajectories``, which imports ``feature_to_mcherry.dataset_design``
  -- so an import in this direction would drag a reverse dependency (and, from
  ``dataset_design`` itself, a genuine cycle) into this figure's import graph. The DMSO
  line style below is therefore *mirrored by value* from ``feature_analysis/plots.py``:
  the two figure families are meant to look like one family, so a change to either
  should prompt a look at the other.
- **The collapse is the canonical one** (``data.collapse.collapse_slices_to_cells``),
  taking all three percentiles in a single pass because that reduction is the expensive
  step on a ~1.4M-row experiment.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")  # once, before pyplot import -- headless-safe under SLURM
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.feature_to_mcherry.data.collapse import (  # noqa: E402
    collapse_slices_to_cells,
)
from src.feature_to_mcherry.dataset_design import (  # noqa: E402
    MIN_CELLS_PER_SIDE,
)

logger = logging.getLogger(__name__)

#: Mirrored by value from ``src/feature_analysis/plots.py`` (see the module docstring
#: for why it is copied rather than imported). DMSO is bold, black, solid and on top;
#: drug wells are thin and behind it.
DMSO_LINEWIDTH = 2.4
DMSO_COLOR = "black"
DMSO_ZORDER = 5
WELL_LINEWIDTH = 1.0
WELL_ZORDER = 2
MARKER = "o"
MARKER_SIZE = 2.5

#: Frames are sampled across 72 h; ``(ti - 1) * 72 / 350`` is the convention the
#: collaborator deliverable's Table 1 uses to label them in hours.
TI_TO_HOURS = 72.0 / 350.0

#: Colour ramp for the dose ladder. Sequential (not categorical) so the ordering itself
#: carries meaning: dose_rank 1 is the top dose.
DOSE_CMAP = "viridis"


def _frame_index(timepoints: pd.Series) -> pd.Series:
    """Numeric frame index, ``NaN`` where the label does not parse.

    Deliberately ``pd.to_numeric`` rather than the ``str.isdigit()`` test that
    ``feature_analysis.feature_trajectories._ti`` uses. The canonical collapse casts its
    key columns to ``str`` (``data/collapse.py:79``), so a float-typed ``timepoint``
    column -- which is what ``pd.read_csv`` infers as soon as one cell is blank, and
    ``mcherry_metrics/io/loaders.py`` does write a blank for an unrecognised
    timepoint -- arrives here as ``"241.0"``. ``"241.0".isdigit()`` is ``False``, so
    an isdigit test would silently stack every point in the figure at a single x
    position while
    ``build_effect_table`` (which uses ``pd.to_numeric``) read the same file correctly:
    the CSV would be right and the figure illustrating it wrong. This matches
    ``timepoint_matched_delta`` instead.
    """
    return pd.to_numeric(timepoints, errors="coerce")


def build_well_timeseries(
    targets: pd.DataFrame,
    target_columns: Sequence[str],
    *,
    well_column: str = "sample_id",
    time_column: str = "timepoint",
    cell_id_column: str = "cell_id",
) -> pd.DataFrame:
    """Per ``(well, timepoint, target)`` median across cells, long form.

    Collapses z-slices to cells once for every requested target (the canonical
    reduction), then takes the per-(well, timepoint) median across cells. ``n_cells``
    is carried through because a median over three surviving cells and one over three
    hundred should not be read the same way once a well starts collapsing.

    Args:
        targets: Per-``(cell, z_slice)`` rows, from ``data.loaders.load_targets``.
        target_columns: Target columns to reduce, e.g. the three mCherry percentiles.

    Returns:
        Columns ``sample_id``, ``timepoint``, ``ti``, ``target``, ``median``,
        ``n_cells`` -- these names regardless of the ``*_column`` arguments, which
        describe the *input* frame only.

    Raises:
        ValueError: If ``target_columns`` is empty, or a key/target column is absent
            (raised by the canonical collapse, which refuses to fall back to
            slice-weighted rows).
    """
    columns = list(target_columns)
    if not columns:
        raise ValueError("target_columns is empty; nothing to plot")

    per_cell = collapse_slices_to_cells(
        targets,
        columns,
        well_column=well_column,
        time_column=time_column,
        cell_id_column=cell_id_column,
    )
    per_cell = per_cell.assign(ti=_frame_index(per_cell[time_column]))
    unparsed = int(per_cell["ti"].isna().sum())
    if unparsed:
        logger.warning(
            "dropping %d of %d per-cell rows whose timepoint did not parse as a "
            "number; they have no position on a time axis",
            unparsed,
            len(per_cell),
        )
        per_cell = per_cell.dropna(subset=["ti"])

    long = per_cell.melt(
        id_vars=[well_column, time_column, "ti"],
        value_vars=columns,
        var_name="target",
        value_name="value",
    ).dropna(subset=["value"])

    series = (
        long.groupby([well_column, time_column, "ti", "target"])["value"]
        .agg(median="median", n_cells="count")
        .reset_index()
    )
    # Always emit the canonical names, whatever the caller's input columns were called:
    # `plot_dmso_vs_drug` reads this frame by name, so a renamed key column would turn a
    # non-default `well_column` into a KeyError at plot time.
    series = series.rename(columns={well_column: "sample_id", time_column: "timepoint"})
    return series.sort_values(["target", "sample_id", "ti"]).reset_index(drop=True)


def _ti_to_hours(ti: Any) -> Any:
    """Frame index -> elapsed hours, for the secondary axis.

    Typed loosely because matplotlib calls the transform with whatever it holds --
    a scalar tick or the whole array of them.
    """
    return (ti - 1) * TI_TO_HOURS


def _hours_to_ti(hours: Any) -> Any:
    """Inverse of :func:`_ti_to_hours`; matplotlib needs both directions."""
    return hours / TI_TO_HOURS + 1


def _dose_colors(wells: pd.DataFrame) -> Dict[str, tuple]:
    """Well -> RGBA along the dose ladder (rank 1 = top dose = darkest).

    Wells with no ``dose_rank`` (a drug with no defined ladder) are spread across the
    same ramp by well name, so they stay distinguishable without implying a dose.
    """
    ranked = wells.dropna(subset=["dose_rank"]).sort_values("dose_rank")
    unranked = wells[wells["dose_rank"].isna()].sort_values("well")
    ordered = list(ranked["well"]) + list(unranked["well"])

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


def _mark_thin_points(
    ax: "plt.Axes", group: pd.DataFrame, min_cells: int, color: object
) -> None:
    """Ring the timepoints backed by fewer than ``min_cells`` cells.

    Marked rather than dropped: the points are real, but ``build_effect_table`` excludes
    them from the CSV this figure illustrates, so drawing them as ordinary points would
    make the least trustworthy stretch of the curve look like the rest of it.
    """
    thin = group[group["n_cells"] < min_cells]
    if thin.empty:
        return
    ax.scatter(
        thin["ti"],
        thin["median"],
        s=42,
        facecolor="none",
        edgecolor=color if color is not None else DMSO_COLOR,
        linewidths=0.9,
        alpha=0.9,
        zorder=WELL_ZORDER + 1,
    )


def _dose_label(row: pd.Series) -> str:
    """``"E07  75 µM"``-style label; the well id alone when no dose is defined."""
    well = str(row["well"])
    if pd.isna(row.get("concentration_uM")):
        return well
    return f"{well}  {row['concentration_uM']:g} µM"


def plot_dmso_vs_drug(
    series: pd.DataFrame,
    wells: pd.DataFrame,
    *,
    experiment: str,
    drug: str,
    target: str,
    out_dir: Path,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
    min_cells: int = MIN_CELLS_PER_SIDE,
) -> List[Path]:
    """One figure: ``target`` vs time for every dose of ``drug``, against DMSO.

    Args:
        series: Output of :func:`build_well_timeseries`; filtered to ``target`` here.
        wells: Summary rows for this (experiment, drug) plus the culture's DMSO row --
            ``well``, ``drug``, ``dose_rank``, ``concentration_uM``, ``is_dmso``,
            ``t_cross_peak``.
        out_dir: Directory to write into; created if absent.
        min_cells: Timepoints backed by fewer cells than this are ringed rather than
            hidden. ``build_effect_table`` *drops* them (``MIN_CELLS_PER_SIDE``), so
            without the marking the figure's late tail would show points the CSV it
            illustrates deliberately excluded -- and that tail is both the least
            trustworthy part and, drawn plainly, the most emphatic.

    Returns:
        The paths written, one per format. Empty if no *drug* well had data for
        ``target`` (a DMSO-only figure implies data that does not exist).
    """
    subset = series[series["target"] == target]
    drug_wells = wells[~wells["is_dmso"].astype(bool)]
    dmso_wells = wells[wells["is_dmso"].astype(bool)]
    colors = _dose_colors(drug_wells)

    fig, ax = plt.subplots(figsize=(9.5, 6))
    drug_plotted = 0
    dmso_plotted = 0
    censored: List[str] = []
    thin: List[str] = []

    for _, row in drug_wells.sort_values("dose_rank").iterrows():
        well = str(row["well"])
        group = subset[subset["sample_id"].astype(str) == well].sort_values("ti")
        if group.empty:
            logger.warning(
                "%s %s: well %s has no %s data", experiment, drug, well, target
            )
            continue
        ax.plot(
            group["ti"],
            group["median"],
            marker=MARKER,
            markersize=MARKER_SIZE,
            linewidth=WELL_LINEWIDTH,
            color=colors.get(well),
            zorder=WELL_ZORDER,
            label=_dose_label(row),
        )
        drug_plotted += 1
        _mark_thin_points(ax, group, min_cells, colors.get(well))
        if (group["n_cells"] < min_cells).any():
            thin.append(well)
        if pd.notna(row["t_cross_peak"]):
            # The well's own collapse point, in its own colour: the per-well bound on
            # the pre-confluence window that §7c' compares inside.
            ax.axvline(
                float(row["t_cross_peak"]),
                color=colors.get(well),
                linestyle="--",
                linewidth=0.9,
                alpha=0.55,
                zorder=1,
            )
        else:
            # Right-censored: never crossed within the timecourse. No marker at all --
            # drawing one at 0 would read as "collapsed immediately".
            censored.append(well)

    for _, row in dmso_wells.iterrows():
        well = str(row["well"])
        group = subset[subset["sample_id"].astype(str) == well].sort_values("ti")
        if group.empty:
            logger.warning("%s: DMSO well %s has no %s data", experiment, well, target)
            continue
        ax.plot(
            group["ti"],
            group["median"],
            marker=MARKER,
            markersize=MARKER_SIZE,
            linewidth=DMSO_LINEWIDTH,
            color=DMSO_COLOR,
            zorder=DMSO_ZORDER,
            label=f"{well} (DMSO)",
        )
        dmso_plotted += 1
        _mark_thin_points(ax, group, min_cells, DMSO_COLOR)
        if (group["n_cells"] < min_cells).any():
            thin.append(f"{well} (DMSO)")
        if pd.notna(row["t_cross_peak"]):
            # The band is the DMSO REFERENCE's window -- deliberately not labelled as
            # "the" pre-confluence window, because §7c' bounds each well at
            # `_earlier_crossing(dmso_cross, well_row["t_cross_peak"])`
            # (`dataset_design.py:329`). On the shipped 45-well summary, 15 of 40 drug
            # wells collapse before their reference does, so a single band drawn to the
            # DMSO crossing overstates the window actually used for those wells. Their
            # own dashed line carries the real bound; the caption says so.
            ax.axvspan(
                float(subset["ti"].min()),
                float(row["t_cross_peak"]),
                color="grey",
                alpha=0.10,
                zorder=0,
            )
            ax.axvline(
                float(row["t_cross_peak"]),
                color=DMSO_COLOR,
                linestyle="--",
                linewidth=1.4,
                alpha=0.8,
                zorder=DMSO_ZORDER - 1,
            )
        else:
            # Without this the band and the black boundary both silently vanish while
            # the caption still promises them -- indistinguishable, to a reader, from a
            # failed render.
            censored.append(f"{well} (DMSO)")

    if drug_plotted == 0:
        # A DMSO-only figure implies drug data that does not exist, and would still be
        # counted in the run's "wrote N figures" tally.
        plt.close(fig)
        logger.warning(
            "%s %s / %s: no drug well had data; not writing a DMSO-only figure",
            experiment,
            drug,
            target,
        )
        return []
    if dmso_plotted == 0:
        logger.warning(
            "%s %s / %s: no DMSO reference line on this figure",
            experiment,
            drug,
            target,
        )

    ax.set_xlabel("timepoint index (ti)")
    ax.set_ylabel(f"{target} mCherry intensity (median across cells)")
    secondary = ax.secondary_xaxis("top", functions=(_ti_to_hours, _hours_to_ti))
    secondary.set_xlabel("hours")

    caption = (
        f"{experiment} — {drug} vs DMSO\n"
        "colour = dose (dark = high); black = DMSO reference; dashed = that well's "
        "t_cross_peak\nshaded = the DMSO reference's window; §7c' bounds each well at "
        "the EARLIER of its own dashed line and DMSO's"
    )
    if censored:
        caption += "\nno collapse within the timecourse (no marker): " + ", ".join(
            censored
        )
    if thin:
        caption += (
            f"\nringed points: fewer than {min_cells} cells, the floor below which "
            "drug_effect_two_windows.csv drops the timepoint"
        )
    ax.set_title(caption, fontsize=10)
    ax.grid(alpha=0.25)

    # De-duplicate legend entries, mirroring the trajectory grid's legend handling: a
    # well that contributed no points must not leave a dangling entry.
    handles, labels = ax.get_legend_handles_labels()
    seen: Dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        seen.setdefault(label, handle)
    if seen:
        ax.legend(
            list(seen.values()),
            list(seen.keys()),
            fontsize=8,
            ncol=2,
            title="well / dose",
        )

    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{experiment}_{drug}"
    written: List[Path] = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi)
        written.append(path)
    plt.close(fig)
    logger.info("%s %s: wrote %s", experiment, drug, [str(p) for p in written])
    return written


def write_dmso_vs_drug_figures(
    targets: pd.DataFrame,
    summary: pd.DataFrame,
    experiment: str,
    out_dir: Path,
    *,
    target_columns: Sequence[str],
    headline_target: str,
    all_targets: bool = False,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
) -> List[Path]:
    """Every (drug, target) figure for one culture.

    The culture's DMSO well is added to *each* drug's figure: DMSO belongs to no drug,
    so selecting wells by ``drug`` alone would silently drop the reference line.

    Args:
        summary: The 45-well summary; filtered to ``experiment`` here.
        all_targets: Draw every column in ``target_columns`` rather than only
            ``headline_target`` (the §1 cross-check across percentiles).

    Returns:
        Every path written, across drugs and targets.

    Raises:
        ValueError: If ``experiment`` has no rows in ``summary``, or no DMSO well.
    """
    rows = summary[summary["experiment"].astype(str) == experiment]
    if rows.empty:
        raise ValueError(f"{experiment}: no rows in the summary table")
    dmso = rows[rows["is_dmso"].astype(bool)]
    if dmso.empty:
        raise ValueError(f"{experiment}: no DMSO well in the summary")

    series = build_well_timeseries(targets, target_columns)
    drawn = list(target_columns) if all_targets else [headline_target]

    written: List[Path] = []
    drugs = rows[~rows["is_dmso"].astype(bool)]["drug"].dropna().unique()
    for drug in sorted(drugs):
        wells = pd.concat([rows[rows["drug"] == drug], dmso], ignore_index=True)
        for target in drawn:
            nested = Path(out_dir) / target
            target_dir = out_dir if target == headline_target else nested
            written.extend(
                plot_dmso_vs_drug(
                    series,
                    wells,
                    experiment=experiment,
                    drug=str(drug),
                    target=target,
                    out_dir=Path(target_dir),
                    formats=formats,
                    dpi=dpi,
                )
            )
    return written
