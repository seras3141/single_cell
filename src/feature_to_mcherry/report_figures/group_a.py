"""Group A — headline figures (A1-A7): the report's central R^2 results, the
handcrafted-vs-deep-feature comparison, the segmentation diagnosis, and the
MAE-vs-R^2 methodological note."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..dataset_design_figures import TI_TO_HOURS
from .a5_paths import MASK_SOURCES, resolve_area_p90_csv
from .config import ReportFiguresConfig
from .manifest import FigureStatus
from .paths import resolve_experiment_paths
from .theme import EXPERIMENT_COLORS, ordered_percentiles, save_fig

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _load_baseline_ladders(config: ReportFiguresConfig) -> Dict[str, pd.DataFrame]:
    """Experiment -> baseline_ladder.csv contents, for experiments where it exists."""
    ladders: Dict[str, pd.DataFrame] = {}
    for exp in config.experiments:
        paths = resolve_experiment_paths(exp, config)
        if paths.baseline_ladder_csv is not None:
            ladders[exp] = pd.read_csv(paths.baseline_ladder_csv)
    return ladders


def figure_a1(config: ReportFiguresConfig) -> FigureStatus:
    """Cross-experiment Ridge floor ranking: R^2 per experiment, by percentile."""
    ladders = _load_baseline_ladders(config)
    missing = [exp for exp in config.experiments if exp not in ladders]

    if not ladders:
        return FigureStatus(
            figure_id="A1",
            title="Cross-experiment Ridge floor ranking",
            status="blocked",
            missing=missing,
            note="No baseline_ladder.csv found for any experiment.",
        )

    rows = []
    for exp, ladder in ladders.items():
        ridge = ladder[ladder["model"] == "ridge"]
        for _, row in ridge.iterrows():
            rows.append({"experiment": exp, "target": row["target"], "r2": row["r2"]})
    table = pd.DataFrame(rows)
    pivot = table.pivot(index="experiment", columns="target", values="r2")
    percentiles = ordered_percentiles(pivot.columns)
    if not percentiles:
        return FigureStatus(
            figure_id="A1",
            title="Cross-experiment Ridge floor ranking",
            status="blocked",
            missing=missing,
            note=(
                "No percentile targets in baseline_ladder.csv; found "
                f"{sorted(map(str, pivot.columns))}."
            ),
        )
    pivot = pivot[percentiles]
    order = pivot.mean(axis=1).sort_values(ascending=True).index
    pivot = pivot.loc[order]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.6 * len(pivot))))
    n_targets = len(percentiles)
    bar_height = 0.8 / n_targets
    y_positions = np.arange(len(pivot))
    for i, target in enumerate(percentiles):
        offsets = y_positions + i * bar_height - 0.4 + bar_height / 2
        ax.barh(offsets, pivot[target], height=bar_height, label=target)
        for y, value in zip(offsets, pivot[target]):
            ax.text(value + 0.005, y, f"{value:.2f}", va="center", fontsize=8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(pivot.index)
    data_min = float(np.nanmin(pivot.values)) if pivot.size else 0.0
    data_max = float(np.nanmax(pivot.values)) if pivot.size else 0.35
    x_min = min(0.0, data_min) - 0.02
    x_max = max(0.35, data_max) + 0.05
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("R² (Ridge, grouped CV out-of-fold)")
    ax.set_title("Cross-experiment Ridge floor ranking")
    ax.legend(title="percentile", loc="lower right")
    fig.tight_layout()

    written = save_fig(
        fig, config.output_dir, "A1_ridge_floor_ranking", config.formats, config.dpi
    )

    status = "generated" if not missing else "partial"
    return FigureStatus(
        figure_id="A1",
        title="Cross-experiment Ridge floor ranking",
        status=status,
        output_paths=[str(p) for p in written],
        missing=missing,
    )


_A2_HANDCRAFTED = "handcrafted features"
_A2_NATIVE = "scPortrait (native masks)"
_A2_INJECTED = "scPortrait (injected Cellpose masks)"
_A2_BACKEND_COLORS = {
    _A2_HANDCRAFTED: "#0072B2",
    _A2_NATIVE: "#E69F00",
    _A2_INJECTED: "#CC79A7",
}
_A2_PENDING_COLOR = "#EEEEEE"
_A2_RUNGS = {"ridge": "linear (Ridge)", "gradient_boosting": "nonlinear (boosting)"}
_A2_TITLE = "Handcrafted features vs scPortrait, linear + nonlinear floor"
# Same-day files count as pre-fix.
_MASK_FIX_AUDITED = date(2026, 8, 18)


def _predates_mask_fix(path: Path) -> bool:
    return date.fromtimestamp(path.stat().st_mtime) <= _MASK_FIX_AUDITED


def _a2_rows(
    exp: str, backend: str, floor_csv: Optional[Path], problems: List[str]
) -> List[dict]:
    """``with_suspect`` percentile-target floor rows of one backend, both rungs.

    A CSV that cannot be read, or leaves no rows, adds a ``problems`` entry.
    """
    if floor_csv is None:
        return []
    unreadable = f"{exp}: unreadable {backend} floor_metrics.csv"
    try:
        floor = pd.read_csv(floor_csv)
    except (ValueError, OSError) as error:
        problems.append(f"{unreadable} ({type(error).__name__})")
        return []
    if not {"variant", "model", "target", "r2"} <= set(floor.columns):
        problems.append(f"{unreadable} (missing columns)")
        return []
    floor = floor[(floor["variant"] == "with_suspect") & floor["model"].isin(_A2_RUNGS)]
    floor = floor[floor["target"].isin(ordered_percentiles(floor["target"].unique()))]
    floor = floor.assign(r2=pd.to_numeric(floor["r2"], errors="coerce"))
    floor = floor[np.isfinite(floor["r2"])]
    if floor.empty:
        problems.append(f"{exp}: no usable rows in {backend} floor_metrics.csv")
    return [
        {
            "experiment": exp,
            "rung": _A2_RUNGS[row["model"]],
            "backend": backend,
            "learner": row.get("backend", ""),
            "target": row["target"],
            "r2": row["r2"],
        }
        for _, row in floor.iterrows()
    ]


def _rung_targets(rows: List[dict]) -> set:
    return {(row["rung"], row["target"]) for row in rows}


def _covers_both_rungs(rows: List[dict], required: set) -> bool:
    covered = _rung_targets(rows)
    return required <= covered and {rung for rung, _ in covered} == set(
        _A2_RUNGS.values()
    )


def figure_a2(config: ReportFiguresConfig) -> FigureStatus:
    """Handcrafted features vs scPortrait, linear and nonlinear floor.

    Both rungs come from each backend's ``floor_metrics.csv``: the
    ``baseline_ladder.csv`` runs were not redone on the regenerated masks. A pre-fix
    injected floor is left out even when ``scportrait_injected_pending`` is false.
    """
    rows: List[dict] = []
    missing: List[str] = []
    problems: List[str] = []
    injected_notes: List[str] = []
    injected_absent: List[str] = []
    injected_incomplete: List[str] = []
    floors_found = False
    for exp in config.scportrait_experiments:
        paths = resolve_experiment_paths(exp, config)
        floors_found |= (
            paths.floor_metrics_csv is not None
            or paths.scportrait_floor_metrics_csv is not None
        )
        handcrafted = _a2_rows(exp, _A2_HANDCRAFTED, paths.floor_metrics_csv, problems)
        native = _a2_rows(exp, _A2_NATIVE, paths.scportrait_floor_metrics_csv, problems)
        required = _rung_targets(handcrafted) | _rung_targets(native)
        if not (
            _covers_both_rungs(handcrafted, required)
            and _covers_both_rungs(native, required)
        ):
            missing.append(exp)
        rows.extend(handcrafted)
        rows.extend(native)
        if (
            config.scportrait_injected_pending
            or exp not in config.scportrait_injected_experiments
        ):
            continue
        injected_csv = paths.scportrait_injected_floor_metrics_csv
        floors_found |= injected_csv is not None
        if injected_csv is None or not injected_csv.is_file():
            injected_notes.append(f"{exp}: no injected floor")
            injected_absent.append(exp)
        elif _predates_mask_fix(injected_csv):
            injected_notes.append(f"{exp}: injected floor predates the mask fix")
            injected_absent.append(exp)
        else:
            injected = _a2_rows(exp, _A2_INJECTED, injected_csv, problems)
            rows.extend(injected)
            if _covers_both_rungs(injected, required):
                continue
            if injected:
                injected_notes.append(f"{exp}: injected floor incomplete")
                injected_incomplete.append(exp)
            else:
                injected_absent.append(exp)
        if exp not in missing:
            missing.append(exp)

    notes = problems + injected_notes
    pending_note = "Injected-mask rung drawn as pending (scportrait_injected_pending)."
    pending_requested = config.scportrait_injected_pending and any(
        exp in config.scportrait_injected_experiments
        for exp in config.scportrait_experiments
    )
    table = pd.DataFrame(
        rows, columns=["experiment", "rung", "backend", "learner", "target", "r2"]
    )
    if table.empty:
        return FigureStatus(
            figure_id="A2",
            title=_A2_TITLE,
            status="blocked",
            missing=missing,
            note="; ".join(
                [
                    (
                        "No usable floor rows for any scPortrait experiment "
                        "(with_suspect ridge/gradient_boosting, percentile target, "
                        "finite R²)"
                        if floors_found
                        else "No floor_metrics.csv found for any scPortrait experiment"
                    )
                ]
                + ([pending_note] if pending_requested else [])
                + notes
            ),
        )

    key_columns = ["experiment", "backend", "rung", "target"]
    duplicated = table[table.duplicated(key_columns, keep=False)]
    if not duplicated.empty:
        keys = sorted(
            {"/".join(map(str, key)) for key in duplicated[key_columns].values}
        )
        notes.append(f"duplicate rows averaged: {', '.join(keys)}")
    nonlinear = table[table["rung"] == _A2_RUNGS["gradient_boosting"]]
    learners = sorted(set(nonlinear["learner"].dropna()) - {""})
    mixed_learners = len(learners) > 1
    if mixed_learners:
        notes.append(f"nonlinear rung mixes learners: {', '.join(learners)}")

    drawn_experiments = set(table["experiment"])
    injected_absent = [exp for exp in injected_absent if exp in drawn_experiments]
    injected_incomplete = [
        exp for exp in injected_incomplete if exp in drawn_experiments
    ]
    injected_experiments = [
        exp
        for exp in config.scportrait_experiments
        if exp in config.scportrait_injected_experiments and exp in drawn_experiments
    ]
    pending = config.scportrait_injected_pending and bool(injected_experiments)
    if pending:
        notes.insert(0, pending_note)

    def is_pending(backend: str) -> bool:
        return pending and backend == _A2_INJECTED

    backends = [_A2_HANDCRAFTED, _A2_NATIVE]
    if pending or injected_absent or _A2_INJECTED in set(table["backend"]):
        backends.append(_A2_INJECTED)
    present = set(zip(table["experiment"], table["target"]))
    targets = ordered_percentiles(table["target"].unique())
    groups = [
        (exp, target)
        for exp in config.scportrait_experiments
        for target in targets
        if (exp, target) in present
    ]
    bar_width = 0.8 / len(backends)
    y_max = max(0.05, float(table["r2"].max()) * 1.15)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, rung in zip(axes, _A2_RUNGS.values()):
        subset = (
            table[table["rung"] == rung]
            .groupby(["experiment", "target", "backend"])["r2"]
            .mean()
        )
        if subset.empty:
            ax.text(
                0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes
            )
        for b_idx, backend in enumerate([] if subset.empty else backends):
            offset = (b_idx - (len(backends) - 1) / 2) * bar_width
            for g_idx, (exp, target) in enumerate(groups):
                x = g_idx + offset
                key = (exp, target, backend)
                if is_pending(backend) and exp in injected_experiments:
                    ax.bar(x, y_max, width=bar_width, color=_A2_PENDING_COLOR)
                    ax.text(
                        x,
                        y_max / 2,
                        "pending GPU",
                        rotation=90,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="#4D4D4D",
                    )
                elif key in subset.index:
                    value = float(subset.loc[key])
                    ax.bar(x, value, width=bar_width, color=_A2_BACKEND_COLORS[backend])
                    ax.text(
                        x,
                        value,
                        f"{value:.2f}",
                        ha="center",
                        va="bottom" if value >= 0 else "top",
                        fontsize=7,
                    )
        if subset.empty:
            ax.set_xticks([])
        else:
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels(
                [f"{exp}\n{target}" for exp, target in groups], rotation=45, ha="right"
            )
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_title(f"{rung} floor")
    axes[0].set_ylabel("R²")
    handles = [
        Patch(color=_A2_PENDING_COLOR if is_pending(b) else _A2_BACKEND_COLORS[b])
        for b in backends
    ]

    def legend_label(backend: str) -> str:
        if is_pending(backend):
            return f"{backend} -- pending GPU re-featurization"
        if backend != _A2_INJECTED:
            return backend
        gaps = []
        if injected_absent:
            gaps.append(f"not plotted for {', '.join(injected_absent)}")
        if injected_incomplete:
            gaps.append(f"incomplete for {', '.join(injected_incomplete)}")
        return f"{backend} -- {'; '.join(gaps)}" if gaps else backend

    r2_min = float(table["r2"].min())
    y_min = 0.0 if r2_min >= 0 else r2_min - 0.2 * (y_max - r2_min)
    axes[0].set_ylim(y_min, y_max)
    labels = [legend_label(b) for b in backends]
    fig.suptitle(_A2_TITLE, y=0.99)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(backends),
        bbox_to_anchor=(0.5, 0.94),
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    if _A2_NATIVE in set(table["backend"]):
        fig.text(
            0.5,
            0.0,
            "scPortrait on native masks re-extracts its target against its own masks, "
            "so its cell population and target differ from the other bars.",
            ha="center",
            va="bottom",
            fontsize=8,
            style="italic",
        )

    written = save_fig(
        fig, config.output_dir, "A2_incarta_vs_scportrait", config.formats, config.dpi
    )

    return FigureStatus(
        figure_id="A2",
        title=_A2_TITLE,
        status=(
            "partial"
            if missing or not duplicated.empty or mixed_learners
            else "generated"
        ),
        output_paths=[str(p) for p in written],
        missing=missing,
        note="; ".join(notes),
    )


def figure_a3(config: ReportFiguresConfig) -> FigureStatus:
    """Linear-quantile instability: R^2 vs tau, one line per experiment."""
    ladders = _load_baseline_ladders(config)
    missing = [exp for exp in config.experiments if exp not in ladders]

    if not ladders:
        return FigureStatus(
            figure_id="A3",
            title="Linear-quantile instability curve",
            status="blocked",
            missing=missing,
            note="No baseline_ladder.csv found for any experiment.",
        )

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for exp, ladder in ladders.items():
        subset = ladder[ladder["model"] == "linear_quantile"].sort_values("tau")
        if subset.empty:
            continue
        ax.plot(
            subset["tau"],
            subset["r2"],
            marker="o",
            label=exp,
            color=EXPERIMENT_COLORS.get(exp),
        )
    ax.axhline(0, color="grey", linewidth=1)
    ylim = ax.get_ylim()
    ax.axhspan(ylim[0], 0, color="firebrick", alpha=0.08)
    ax.set_ylim(ylim)
    ax.set_xlabel("τ (quantile level)")
    ax.set_ylabel("R² (linear quantile, grouped CV out-of-fold)")
    ax.set_title("Linear-quantile instability across τ")
    ax.legend()
    fig.tight_layout()

    written = save_fig(
        fig,
        config.output_dir,
        "A3_linear_quantile_instability",
        config.formats,
        config.dpi,
    )

    status = "generated" if not missing else "partial"
    return FigureStatus(
        figure_id="A3",
        title="Linear-quantile instability curve",
        status=status,
        output_paths=[str(p) for p in written],
        missing=missing,
    )


def figure_a4(config: ReportFiguresConfig) -> FigureStatus:
    """Informativeness floor: linear vs nonlinear R^2, faceted by percentile."""
    floors: Dict[str, pd.DataFrame] = {}
    missing: List[str] = []
    for exp in config.experiments:
        paths = resolve_experiment_paths(exp, config)
        if paths.floor_metrics_csv is None:
            missing.append(exp)
            continue
        floors[exp] = pd.read_csv(paths.floor_metrics_csv)

    if not floors:
        return FigureStatus(
            figure_id="A4",
            title="Informativeness performance floor - linear vs nonlinear",
            status="blocked",
            missing=missing,
            note="No floor_metrics.csv found for any experiment.",
        )

    rows = []
    for exp, floor in floors.items():
        subset = floor[floor["variant"] == "with_suspect"]
        for _, row in subset.iterrows():
            model_kind = "linear" if row["model"] == "ridge" else "nonlinear"
            rows.append(
                {
                    "experiment": exp,
                    "target": row["target"],
                    "model": model_kind,
                    "r2": row["r2"],
                }
            )
    table = pd.DataFrame(rows)
    percentiles = ordered_percentiles(table["target"].unique())
    if not percentiles:
        return FigureStatus(
            figure_id="A4",
            title="Informativeness performance floor - linear vs nonlinear",
            status="blocked",
            missing=missing,
            note=(
                "No percentile targets in floor_metrics.csv; found "
                f"{sorted(map(str, table['target'].unique()))}."
            ),
        )

    # Order experiments by mean nonlinear floor (ascending), once, and reuse that
    # order across every percentile facet so experiments don't jump around
    # between panels. Experiments with no nonlinear row (linear-only) sort last.
    nonlinear_order = (
        table[table["model"] == "nonlinear"]
        .groupby("experiment")["r2"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    experiment_order = nonlinear_order + [
        exp for exp in table["experiment"].unique() if exp not in nonlinear_order
    ]

    fig, axes = plt.subplots(
        1, len(percentiles), figsize=(4 * len(percentiles), 4), squeeze=False
    )
    for idx, target in enumerate(percentiles):
        ax = axes[0][idx]
        pivot = table[table["target"] == target].pivot(
            index="experiment", columns="model", values="r2"
        )
        pivot = pivot.reindex(index=experiment_order, columns=["linear", "nonlinear"])
        pivot.plot(
            kind="bar",
            ax=ax,
            color={"linear": "#0072B2", "nonlinear": "#009E73"},
            legend=False,
        )
        ax.set_title(target)
        ax.set_ylabel("R²" if idx == 0 else "")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        for position, exp in enumerate(pivot.index):
            lin, nonlin = pivot.loc[exp, "linear"], pivot.loc[exp, "nonlinear"]
            if pd.notna(lin) and pd.notna(nonlin) and lin > 0:
                ax.annotate(
                    f"{nonlin / lin:.1f}x",
                    xy=(position, max(lin, nonlin)),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#0072B2"),
        plt.Rectangle((0, 0), 1, 1, color="#009E73"),
    ]
    fig.legend(
        handles, ["linear", "nonlinear"], loc="upper right", bbox_to_anchor=(1.0, 1.05)
    )
    fig.suptitle("Linear vs nonlinear morphology-informativeness floor")
    fig.tight_layout()

    written = save_fig(
        fig,
        config.output_dir,
        "A4_linear_vs_nonlinear_floor",
        config.formats,
        config.dpi,
    )

    status = "generated" if not missing else "partial"
    return FigureStatus(
        figure_id="A4",
        title="Informativeness performance floor - linear vs nonlinear",
        status=status,
        output_paths=[str(p) for p in written],
        missing=missing,
    )


_A5_SOURCE_LABELS = {"cellpose_sam": "cellpose_sam", "scportrait": "scPortrait"}
_A5_SOURCE_COLORS = {"cellpose_sam": "#0072B2", "scportrait": "#E69F00"}


def figure_a5(config: ReportFiguresConfig) -> FigureStatus:
    """Segmentation diagnosis: corr(area, p90) and area distribution, cellpose_sam
    vs scPortrait masks -- the evidence that scPortrait's weak mCherry
    decodability is a *segmentation* problem, not a *representation* one."""
    frames: Dict[str, Dict[str, pd.DataFrame]] = {}
    missing: List[str] = []
    for exp in config.a5_experiments:
        frames[exp] = {}
        for source in MASK_SOURCES:
            csv_path = resolve_area_p90_csv(exp, source, config)
            if csv_path is None:
                continue
            frames[exp][source] = pd.read_csv(csv_path)
        if set(frames[exp]) != set(MASK_SOURCES):
            missing.append(exp)

    if not any(frames.values()):
        return FigureStatus(
            figure_id="A5",
            title="Segmentation diagnosis - why scPortrait decodes mCherry weakly",
            status="blocked",
            missing=missing,
            note="No extracted area/percentile_90 CSV found for any experiment "
            "(run docs_local/sync_a5_area_p90.sh).",
        )

    rho_rows = []
    for exp, sources in frames.items():
        for source, df in sources.items():
            rho, _ = spearmanr(df["area"], df["percentile_90"])
            rho_rows.append({"experiment": exp, "source": source, "rho": rho})
    rho_table = pd.DataFrame(rho_rows)

    fig, (ax_rho, ax_area) = plt.subplots(1, 2, figsize=(12, 4.5))

    pivot = rho_table.pivot(index="experiment", columns="source", values="rho")
    pivot = pivot.reindex(columns=list(MASK_SOURCES))
    pivot.columns = [_A5_SOURCE_LABELS[c] for c in pivot.columns]
    pivot.plot(
        kind="bar",
        ax=ax_rho,
        color=[_A5_SOURCE_COLORS[s] for s in MASK_SOURCES],
    )
    ax_rho.set_ylabel("Spearman ρ(area, percentile_90)")
    ax_rho.set_xlabel("")
    ax_rho.tick_params(axis="x", rotation=0)
    ax_rho.set_title("(a) area predicts mCherry -- by mask source")

    for exp, sources in frames.items():
        for source, df in sources.items():
            lo, hi = np.percentile(df["area"], [0, 99])
            ax_area.hist(
                df["area"].clip(upper=hi),
                bins=60,
                range=(lo, hi),
                histtype="step",
                density=True,
                label=f"{exp} ({_A5_SOURCE_LABELS[source]})",
                color=_A5_SOURCE_COLORS[source],
                linestyle="-" if exp == config.a5_experiments[0] else "--",
            )
    ax_area.set_xlabel("cell area (px, clipped at the 99th percentile)")
    ax_area.set_ylabel("density")
    ax_area.set_title("(b) area distribution -- much wider for scPortrait")
    ax_area.legend(fontsize=8)

    fig.suptitle("Segmentation diagnosis: cellpose_sam vs scPortrait masks")
    fig.text(
        0.5,
        -0.02,
        "The full 2048-d embedding ≡ PCA-100 (no dimensionality bottleneck), and "
        "scPortrait's own area barely predicts its own target (size-blindness "
        "rejected) -- the population, set by the segmentation, is the cause.",
        ha="center",
        fontsize=8,
        style="italic",
    )
    fig.tight_layout()

    written = save_fig(
        fig, config.output_dir, "A5_segmentation_diagnosis", config.formats, config.dpi
    )

    status = "generated" if not missing else "partial"
    return FigureStatus(
        figure_id="A5",
        title="Segmentation diagnosis - why scPortrait decodes mCherry weakly",
        status=status,
        output_paths=[str(p) for p in written],
        missing=missing,
    )


# (label, sibling-directory-name-or-None) -- None means "use
# config.feature_to_mcherry_dir as-is". A hardcoded, one-off mapping (not a
# config list): this figure is
# explicitly "Ew2-1 only / workstream paused" per the brief, not a pattern to
# generalize to other experiments.
_A6_VARIANTS = [
    ("tracked", None),
    ("filtered", "feature_to_mcherry_filtered"),
    ("filtered+blur", "feature_to_mcherry_filtered_blur"),
]
_A6_EXPERIMENT = "Ew2-1"


def figure_a6(config: ReportFiguresConfig) -> FigureStatus:
    """MAE vs R^2 across Ew2-1's three population variants (tracked per-slice,
    filtered per-cell, filtered+blur per-cell) -- R^2 swings wildly while MAE
    stays stable/improves, since R^2 is variance-normalised and the population
    changes across variants. Methodological note, not a five-experiment figure."""
    variant_frames: Dict[str, tuple] = {}
    missing: List[str] = []
    for label, sibling_dir_name in _A6_VARIANTS:
        if sibling_dir_name is None:
            variant_config = config
        else:
            ftm_dir = Path(config.feature_to_mcherry_dir)
            variant_config = replace(
                config, feature_to_mcherry_dir=str(ftm_dir.parent / sibling_dir_name)
            )
        paths = resolve_experiment_paths(_A6_EXPERIMENT, variant_config)
        if paths.baseline_ladder_csv is None:
            missing.append(label)
            continue
        ridge = pd.read_csv(paths.baseline_ladder_csv)
        ridge = ridge[ridge["model"] == "ridge"]
        lightgbm = None
        if paths.floor_metrics_csv is not None:
            floor = pd.read_csv(paths.floor_metrics_csv)
            lightgbm = floor[
                (floor["model"] != "ridge") & (floor["variant"] == "with_suspect")
            ]
        variant_frames[label] = (ridge, lightgbm)

    if not variant_frames:
        return FigureStatus(
            figure_id="A6",
            title="MAE vs R^2 across population variants (Ew2-1 only)",
            status="blocked",
            missing=missing,
            note="No baseline_ladder.csv found for any of Ew2-1's population "
            "variants (tracked/filtered/filtered+blur).",
        )

    variant_order = [label for label, _ in _A6_VARIANTS if label in variant_frames]

    r2_rows = []
    mae_rows = []
    for label, (ridge, lightgbm) in variant_frames.items():
        for _, row in ridge.iterrows():
            r2_rows.append({"variant": label, "target": row["target"], "r2": row["r2"]})
            mae_rows.append(
                {
                    "variant": label,
                    "target": row["target"],
                    "model": "Ridge",
                    "mae": row["mae"],
                }
            )
        if lightgbm is not None:
            for _, row in lightgbm.iterrows():
                mae_rows.append(
                    {
                        "variant": label,
                        "target": row["target"],
                        "model": "LightGBM",
                        "mae": row["mae"],
                    }
                )

    fig, (ax_r2, ax_mae) = plt.subplots(1, 2, figsize=(11, 4.5))

    r2_table = pd.DataFrame(r2_rows)
    pivot_r2 = r2_table.pivot(index="variant", columns="target", values="r2")
    percentiles = ordered_percentiles(pivot_r2.columns)
    if not percentiles:
        # The figure was already opened above; close it rather than leaking it.
        plt.close(fig)
        return FigureStatus(
            figure_id="A6",
            title="MAE vs R^2 across population variants (Ew2-1 only)",
            status="blocked",
            missing=missing,
            note=(
                "No percentile targets in the population-variant metrics; found "
                f"{sorted(map(str, pivot_r2.columns))}."
            ),
        )
    pivot_r2 = pivot_r2.reindex(index=variant_order)[percentiles]
    pivot_r2.plot(kind="bar", ax=ax_r2)
    ax_r2.axhline(0, color="grey", linewidth=1)
    ax_r2.set_title("(a) Ridge R² swings across populations")
    ax_r2.set_ylabel("R²")
    ax_r2.set_xlabel("")
    ax_r2.tick_params(axis="x", rotation=0)

    mae_table = pd.DataFrame(mae_rows)
    # Lowest available percentile, read from the data rather than hard-coded: on a
    # DMSO-normalized run the targets are z_-prefixed, so a literal "percentile_75"
    # selected zero rows and produced an all-NaN panel with no error at all.
    mae_percentiles = ordered_percentiles(mae_table["target"].unique())
    if not mae_percentiles:
        # Falling back to None here selected zero rows and drew an all-NaN "MAE (None)"
        # panel while still reporting status="generated" -- silently wrong output, worse
        # than the crash this helper replaced. Block, as the sibling call sites do.
        plt.close(fig)
        return FigureStatus(
            figure_id="A6",
            title="MAE vs R^2 across population variants (Ew2-1 only)",
            status="blocked",
            missing=missing,
            note=(
                "No percentile targets in the MAE metrics; found "
                f"{sorted(map(str, mae_table['target'].unique()))}."
            ),
        )
    representative_target = mae_percentiles[0]
    mae_subset = mae_table[mae_table["target"] == representative_target]
    pivot_mae = mae_subset.pivot(index="variant", columns="model", values="mae")
    pivot_mae = pivot_mae.reindex(index=variant_order, columns=["Ridge", "LightGBM"])
    pivot_mae.plot(
        kind="bar", ax=ax_mae, color={"Ridge": "#0072B2", "LightGBM": "#009E73"}
    )
    ax_mae.set_title(f"(b) MAE ({representative_target}) -- the honest metric")
    ax_mae.set_ylabel("MAE")
    ax_mae.set_xlabel("")
    ax_mae.tick_params(axis="x", rotation=0)

    fig.suptitle("Ew2-1 only / workstream paused: MAE vs R² across population variants")
    fig.text(
        0.5,
        -0.02,
        "MAE is the honest cross-population metric; R² is only comparable within "
        "a fixed population.",
        ha="center",
        fontsize=8,
        style="italic",
    )
    fig.tight_layout()

    written = save_fig(
        fig,
        config.output_dir,
        "A6_mae_vs_r2_population_variants",
        config.formats,
        config.dpi,
    )

    status = "generated" if not missing else "partial"
    return FigureStatus(
        figure_id="A6",
        title="MAE vs R^2 across population variants (Ew2-1 only)",
        status=status,
        output_paths=[str(p) for p in written],
        missing=missing,
    )


_A7_LEVELS = {
    "pooled": "pooled",
    "within_timepoint": "within timepoint",
    "within_well_timepoint": "within (well, timepoint)",
}
_A7_POOLED_PATTERN = r"pooled(\([0-9]+fold\))?"
_A7_LEVEL_COLORS = ["#4D4D4D", "#8C8C8C", "#C7C7C7"]
_A7_R2_FLOOR = -1.0
_A7_TITLE = "Signal location - nested R² and R² by timepoint"


def figure_a7(config: ReportFiguresConfig) -> FigureStatus:
    """Where the feature->mCherry signal lives: pooled vs within-timepoint vs
    within-(well, timepoint) R^2 per experiment, and within-timepoint R^2 over time.

    Reads ``<ftm>/<exp>/decomposition/{nested_r2,r2_by_timepoint}.csv``, not the
    ``evaluation.nested_r2`` output, which has no per-timepoint curve.
    """
    target = config.signal_location_target
    nested: Dict[str, pd.Series] = {}
    by_timepoint: Dict[str, pd.DataFrame] = {}
    missing: List[str] = []
    notes: List[str] = []
    averaged = False
    for exp in config.experiments:
        decomposition_dir = resolve_experiment_paths(exp, config).decomposition_dir
        if decomposition_dir is None:
            missing.append(exp)
            notes.append(f"{exp}: no decomposition dir")
            continue
        nested_csv = decomposition_dir / "nested_r2.csv"
        curve_csv = decomposition_dir / "r2_by_timepoint.csv"
        absent = [csv.name for csv in (nested_csv, curve_csv) if not csv.exists()]
        if absent:
            missing.append(exp)
            notes.append(f"{exp}: missing {', '.join(absent)}")
            continue
        try:
            nested_table = pd.read_csv(nested_csv)
            curve_table = pd.read_csv(curve_csv)
        except (ValueError, OSError) as error:
            missing.append(exp)
            notes.append(
                f"{exp}: unreadable decomposition table ({type(error).__name__})"
            )
            continue
        if not (
            {"target", "level", "r2"} <= set(nested_table.columns)
            and {"target", "timepoint", "r2"} <= set(curve_table.columns)
        ):
            missing.append(exp)
            notes.append(f"{exp}: decomposition table missing columns")
            continue
        nested_table = nested_table[nested_table["target"] == target].assign(
            r2=lambda table: pd.to_numeric(table["r2"], errors="coerce")
        )
        nested_table = nested_table[np.isfinite(nested_table["r2"])]
        curve_table = curve_table[curve_table["target"] == target].assign(
            r2=lambda table: pd.to_numeric(table["r2"], errors="coerce"),
            timepoint=lambda table: pd.to_numeric(table["timepoint"], errors="coerce"),
        )
        usable = curve_table["timepoint"].notna() & np.isfinite(curve_table["r2"])
        if not usable.all() and usable.any():
            notes.append(
                f"{exp}: {int((~usable).sum())} unusable timepoint rows dropped"
            )
        curve_table = curve_table[usable]
        if nested_table.empty:
            missing.append(exp)
            notes.append(f"{exp}: no finite {target} rows in nested_r2.csv")
            continue
        if curve_table.empty:
            missing.append(exp)
            notes.append(f"{exp}: no finite {target} rows in r2_by_timepoint.csv")
            continue
        raw_levels = nested_table["level"].astype(str)
        is_pooled = raw_levels.str.fullmatch(_A7_POOLED_PATTERN)
        levels = raw_levels.where(~is_pooled, "pooled")
        pooled_labels = sorted(set(raw_levels[is_pooled]))
        several_pooled = len(pooled_labels) > 1
        if several_pooled:
            notes.append(
                f"{exp}: pooled bar left out, several pooled levels: "
                + ", ".join(pooled_labels)
            )
            nested_table, levels = nested_table[~is_pooled], levels[~is_pooled]
        if levels.duplicated().any():
            notes.append(f"{exp}: duplicate nested levels averaged")
            averaged = True
        ignored_levels = sorted(set(levels) - set(_A7_LEVELS))
        if ignored_levels:
            notes.append(f"{exp}: ignored nested levels {', '.join(ignored_levels)}")
        nested[exp] = nested_table.groupby(levels)["r2"].mean()
        absent_levels = [
            level
            for level in _A7_LEVELS
            if pd.isna(nested[exp].get(level))
            and not (several_pooled and level == "pooled")
        ]
        if absent_levels:
            notes.append(f"{exp}: missing nested levels {', '.join(absent_levels)}")
        if several_pooled or absent_levels:
            missing.append(exp)
        if curve_table["timepoint"].duplicated().any():
            notes.append(f"{exp}: duplicate timepoints averaged")
            averaged = True
        by_timepoint[exp] = curve_table.groupby("timepoint", as_index=False)[
            "r2"
        ].mean()

    if not nested:
        return FigureStatus(
            figure_id="A7",
            title=_A7_TITLE,
            status="blocked",
            missing=missing,
            note="; ".join(
                [f"No decomposition tables with target {target!r} for any experiment"]
                + notes
            ),
        )

    experiments = list(nested)
    fig, (ax_levels, ax_time) = plt.subplots(
        1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1, 1.4]}
    )
    bar_width = 0.8 / len(_A7_LEVELS)
    for l_idx, (level, label) in enumerate(_A7_LEVELS.items()):
        offset = (l_idx - (len(_A7_LEVELS) - 1) / 2) * bar_width
        values = [nested[exp].get(level, np.nan) for exp in experiments]
        positions = np.arange(len(experiments)) + offset
        ax_levels.bar(
            positions,
            values,
            width=bar_width,
            color=_A7_LEVEL_COLORS[l_idx],
            label=label,
        )
        for x, value in zip(positions, values):
            if pd.notna(value):
                ax_levels.text(
                    x,
                    value,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=7,
                )
    ax_levels.set_xticks(range(len(experiments)))
    ax_levels.set_xticklabels(experiments)
    ax_levels.axhline(0, color="grey", linewidth=0.8)
    ax_levels.set_ylabel("R² (grouped-by-well out-of-fold)")
    ax_levels.set_title("(a) nested R²")
    bottom, top = ax_levels.get_ylim()
    ax_levels.set_ylim(bottom, max(top, 0.0) + 0.3 * (top - bottom))
    ax_levels.legend(fontsize=8, loc="upper right")

    clipped_minima: Dict[str, float] = {}
    any_negative = False
    for exp in experiments:
        curve = by_timepoint[exp]
        hours = (curve["timepoint"].to_numpy(dtype=float) - 1) * TI_TO_HOURS
        r2 = curve["r2"].to_numpy(dtype=float)
        (line,) = ax_time.plot(
            hours,
            np.maximum(r2, _A7_R2_FLOOR),
            marker="o",
            markersize=3,
            color=EXPERIMENT_COLORS.get(exp),
            label=exp,
        )
        any_negative = any_negative or bool(np.nanmin(r2) < 0)
        clipped = r2 < _A7_R2_FLOOR
        if clipped.any():
            clipped_minima[exp] = float(np.nanmin(r2))
            ax_time.plot(
                hours[clipped],
                np.full(clipped.sum(), _A7_R2_FLOOR),
                "v",
                markersize=7,
                color=line.get_color(),
            )
    ax_time.axhline(0, color="grey", linewidth=0.8)
    bottom, top = ax_time.get_ylim()
    if clipped_minima:
        bottom = min(bottom, _A7_R2_FLOOR - 0.25)
    if any_negative:
        ax_time.axhspan(bottom, 0, color="firebrick", alpha=0.08)
    ax_time.set_ylim(bottom, top + 0.2 * (top - bottom))
    if clipped_minima:
        ax_time.text(
            0.01,
            0.98,
            f"▼ clipped at R² = {_A7_R2_FLOOR:g}; minima:\n"
            + "\n".join(f"{exp} {value:.2f}" for exp, value in clipped_minima.items()),
            transform=ax_time.transAxes,
            va="top",
            fontsize=8,
            color="#4D4D4D",
        )
    ax_time.set_xlabel("time (h)")
    ax_time.set_ylabel("within-timepoint R²")
    ax_time.set_title("(b) R² by timepoint")
    ax_time.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Where the signal lives ({target})")
    fig.tight_layout()

    written = save_fig(
        fig, config.output_dir, "A7_signal_location", config.formats, config.dpi
    )
    return FigureStatus(
        figure_id="A7",
        title=_A7_TITLE,
        status="partial" if missing or averaged else "generated",
        output_paths=[str(p) for p in written],
        missing=missing,
        note="; ".join(notes),
    )
