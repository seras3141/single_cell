"""Before/after impact of the DMSO confidence gate, per (experiment, well).

Step 3's gate is **results-changing, not a safety rail**, so the plan requires measured
before/after counts before it is enabled anywhere -- a passing test suite is not
evidence. This script produces that evidence and writes nothing outside ``--out-dir``.

For every experiment it loads the per-slice target tree, computes the gate's three
conditions at the requested threshold, and reports what would be discarded:

* per (experiment, well): per-cell observations -- distinct ``(timepoint, cell_id)``,
  since ``cell_id`` is a track id reused across timepoints -- and slice rows
  before/after, retention, and how many of that well's timepoints each condition flags;
* per experiment: the same roll-up plus timepoint retention comparable to Step 2's
  ``kept_any``;
* a reconciliation of the gate's distinct-cell counts against ``cell_population.csv``,
  the table Step 2's threshold was actually derived from. Divergence is logged, not
  fatal: the two need not agree (``load_targets`` drops NaN-target rows), but silent
  divergence would hide a real data problem.

Read-only. Run under SLURM (``slurm/gate_step3_impact.sbatch``), not on a login node.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.feature_to_mcherry.data.loaders import load_targets
from src.feature_to_mcherry.data.normalize import (
    DEFAULT_MIN_PEAK_FRACTION,
    compute_confidence_flags,
)
from src.dataset_analysis.gate_survival import DEFAULT_ABSOLUTE_FLOOR
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

DATA_ROOT = Path(
    "/ictstr01/home/haicu/serena.sritharan/projects/single_cell/data/"
    "MF5V1_processed Timelapse samples 19.03.2024"
)
MODEL = "cellpose_sam"

#: label -> (on-disk experiment dir, DMSO reference well). Ew2-1/Ew2-2 use M11, the
#: other three N11; the five experiments are five DISTINCT cultures, never replicates.
EXPERIMENTS: Dict[str, tuple] = {
    "Ew2-1": ("Ew2-1 MF5V1 0-72h 06-03-26", "M11"),
    "Ew2-2": ("Ew2-2 MF5V1 072h 17-04-26", "M11"),
    "HD1509": ("HD1509 MF5V1 0-72h 23-02-26", "N11"),
    "HD1883": ("HD1883 MF5V1 0-72h 20-03-26", "N11"),
    "SA110": ("SA110 MF5V1 0-72h 13-02-26", "N11"),
}

TARGET_COLUMNS = ["percentile_75", "percentile_90", "percentile_95"]

CONDITIONS = ["flag_relative", "flag_absolute_floor", "flag_dmso_reference"]


def _target_csv(exp_dir: str) -> Path:
    return DATA_ROOT / exp_dir / "mcherry_metrics" / MODEL / "instance_metrics.csv"


def _reconcile(
    experiment: str, exp_dir: str, flags: pd.DataFrame
) -> Optional[pd.DataFrame]:
    """Compare the gate's distinct-cell counts against ``cell_population.csv``.

    Step 2's threshold came from that file; the gate counts from the target table it
    actually filters. Report the gap rather than assuming the two agree.
    """
    population_csv = DATA_ROOT / exp_dir / "cell_population.csv"
    if not population_csv.is_file():
        logger.warning(
            "%s: no cell_population.csv; skipping reconciliation", experiment
        )
        return None

    population = pd.read_csv(population_csv)
    time_column = "ti" if "ti" in population.columns else "timepoint"
    merged = flags.merge(
        population[["sample_id", time_column, "n_cells"]].rename(
            columns={time_column: "timepoint", "n_cells": "n_cells_population"}
        ),
        on=["sample_id", "timepoint"],
        how="outer",
        indicator=True,
    )
    merged["experiment"] = experiment
    disagreeing = merged[
        (merged["_merge"] == "both")
        & (merged["n_cells"] != merged["n_cells_population"])
    ]
    if len(disagreeing):
        logger.warning(
            "%s: %d/%d (well, timepoint)s disagree with cell_population.csv "
            "(median gap %+.1f cells). Expected where load_targets dropped NaN-target "
            "rows; investigate any large or clustered gap.",
            experiment,
            len(disagreeing),
            int((merged["_merge"] == "both").sum()),
            float(
                (disagreeing["n_cells"] - disagreeing["n_cells_population"]).median()
            ),
        )
    only_targets = int((merged["_merge"] == "left_only").sum())
    only_population = int((merged["_merge"] == "right_only").sum())
    if only_targets or only_population:
        logger.warning(
            "%s: %d (well, timepoint)s only in the target table, %d only in "
            "cell_population.csv.",
            experiment,
            only_targets,
            only_population,
        )
    return merged


def _well_rows(
    experiment: str, targets: pd.DataFrame, flags: pd.DataFrame
) -> List[dict]:
    """Per-well before/after counts, plus per-condition timepoint tallies."""
    keep_keys = set(
        map(
            tuple,
            flags.loc[~flags["low_confidence"], ["sample_id", "timepoint"]].to_numpy(),
        )
    )
    key_pairs = list(map(tuple, targets[["sample_id", "timepoint"]].to_numpy()))
    kept_mask = pd.Series(
        [pair in keep_keys for pair in key_pairs], index=targets.index
    )

    rows = []
    for well, group in targets.groupby("sample_id", sort=True):
        well_flags = flags[flags["sample_id"] == well]
        kept = group[kept_mask.loc[group.index]]
        # Per-cell OBSERVATIONS, i.e. distinct (timepoint, cell_id). `cell_id` is a
        # track id reused across timepoints, so a bare nunique() over the well would
        # collapse a cell's whole trajectory into one and report ~99% retention for a
        # well that actually lost 89% of its timepoints.
        cells_before = int(len(group[["timepoint", "cell_id"]].drop_duplicates()))
        cells_after = (
            int(len(kept[["timepoint", "cell_id"]].drop_duplicates()))
            if len(kept)
            else 0
        )
        record = {
            "experiment": experiment,
            "well": well,
            "is_dmso": well == EXPERIMENTS[experiment][1],
            "peak_cells": int(well_flags["peak"].max()),
            "timepoints_before": int(len(well_flags)),
            "timepoints_after": int((~well_flags["low_confidence"]).sum()),
            "cells_before": cells_before,
            "cells_after": cells_after,
            "rows_before": int(len(group)),
            "rows_after": int(len(kept)),
        }
        record["timepoint_retention"] = (
            record["timepoints_after"] / record["timepoints_before"]
            if record["timepoints_before"]
            else float("nan")
        )
        record["cell_retention"] = (
            cells_after / cells_before if cells_before else float("nan")
        )
        for condition in CONDITIONS:
            record[f"timepoints_{condition}"] = int(well_flags[condition].sum())
        rows.append(record)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--min-peak-fraction",
        type=float,
        default=DEFAULT_MIN_PEAK_FRACTION,
        help=(
            "Relative threshold as a fraction of each well's peak "
            "(default: %(default)s)."
        ),
    )
    ap.add_argument(
        "--absolute-floor",
        type=int,
        default=DEFAULT_ABSOLUTE_FLOOR,
        help="Degenerate-statistics floor in distinct cells (default: %(default)s).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/dmso_gate_impact"),
        help="Directory for the CSV outputs (default: %(default)s).",
    )
    args = ap.parse_args()
    setup_logging()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    well_rows: List[dict] = []
    flag_frames: List[pd.DataFrame] = []
    reconciliations: List[pd.DataFrame] = []

    for experiment, (exp_dir, dmso_well) in EXPERIMENTS.items():
        csv = _target_csv(exp_dir)
        if not csv.is_file():
            raise FileNotFoundError(f"{experiment}: no target csv at {csv}")
        targets = load_targets(csv, target_columns=TARGET_COLUMNS)
        logger.info(
            "%s: %d slice rows, %d distinct cells, DMSO well %s",
            experiment,
            len(targets),
            targets["cell_id"].nunique(),
            dmso_well,
        )

        flags = compute_confidence_flags(
            targets,
            dmso_well,
            min_peak_fraction=args.min_peak_fraction,
            absolute_floor=args.absolute_floor,
        )
        flags.insert(0, "experiment", experiment)
        flag_frames.append(flags)
        well_rows.extend(_well_rows(experiment, targets, flags))

        reconciled = _reconcile(experiment, exp_dir, flags.drop(columns=["experiment"]))
        if reconciled is not None:
            reconciliations.append(reconciled)

    per_well = pd.DataFrame(well_rows)
    per_well.to_csv(args.out_dir / "per_well_impact.csv", index=False)
    pd.concat(flag_frames, ignore_index=True).to_csv(
        args.out_dir / "per_timepoint_flags.csv.gz", index=False, compression="gzip"
    )
    if reconciliations:
        pd.concat(reconciliations, ignore_index=True).to_csv(
            args.out_dir / "population_reconciliation.csv.gz",
            index=False,
            compression="gzip",
        )

    per_experiment = (
        per_well.groupby("experiment", sort=True)
        .agg(
            wells=("well", "count"),
            timepoints_before=("timepoints_before", "sum"),
            timepoints_after=("timepoints_after", "sum"),
            cells_before=("cells_before", "sum"),
            cells_after=("cells_after", "sum"),
            rows_before=("rows_before", "sum"),
            rows_after=("rows_after", "sum"),
            median_timepoint_retention=("timepoint_retention", "median"),
        )
        .reset_index()
    )
    for stem in ("timepoints", "cells", "rows"):
        per_experiment[f"{stem}_retention"] = (
            per_experiment[f"{stem}_after"] / per_experiment[f"{stem}_before"]
        )
    per_experiment.to_csv(args.out_dir / "per_experiment_impact.csv", index=False)

    total_before = int(per_experiment["timepoints_before"].sum())
    total_after = int(per_experiment["timepoints_after"].sum())
    logger.info(
        "GATE IMPACT at min_peak_fraction=%s, absolute_floor=%s: "
        "%d/%d well-timepoints survive (%.1f%%); %d/%d distinct cells (%.1f%%).",
        args.min_peak_fraction,
        args.absolute_floor,
        total_after,
        total_before,
        100.0 * total_after / total_before if total_before else 0.0,
        int(per_experiment["cells_after"].sum()),
        int(per_experiment["cells_before"].sum()),
        (
            100.0
            * per_experiment["cells_after"].sum()
            / per_experiment["cells_before"].sum()
            if per_experiment["cells_before"].sum()
            else 0.0
        ),
    )
    print("\n=== per experiment ===")
    print(per_experiment.to_string(index=False))
    print("\n=== DMSO reference wells ===")
    print(per_well[per_well["is_dmso"]].to_string(index=False))


if __name__ == "__main__":
    main()
