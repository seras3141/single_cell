"""Group A — headline figures (A1-A6): the report's central R^2 results, the
handcrafted-vs-deep-feature comparison, the segmentation diagnosis, and the
MAE-vs-R^2 methodological note."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .a5_paths import MASK_SOURCES, resolve_area_p90_csv
from .config import ReportFiguresConfig
from .manifest import FigureStatus
from .paths import resolve_experiment_paths
from .theme import EXPERIMENT_COLORS, PERCENTILE_ORDER, save_fig

import matplotlib.pyplot as plt


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
    percentiles = [p for p in PERCENTILE_ORDER if p in pivot.columns]
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


def _collect_a2_rung(
    exp: str,
    rung: str,
    incarta_csv,
    scportrait_csv,
    model_filter,
) -> List[dict]:
    """One rung's (incarta, scPortrait) rows for experiment ``exp``, or ``[]`` if
    either side is missing."""
    if incarta_csv is None or scportrait_csv is None:
        return []
    incarta = pd.read_csv(incarta_csv)
    scportrait = pd.read_csv(scportrait_csv)
    rows = []
    for backend, table in (("incarta", incarta), ("scPortrait", scportrait)):
        subset = model_filter(table)
        for _, row in subset.iterrows():
            rows.append(
                {
                    "experiment": exp,
                    "rung": rung,
                    "backend": backend,
                    "target": row["target"],
                    "r2": row["r2"],
                }
            )
    return rows


def figure_a2(config: ReportFiguresConfig) -> FigureStatus:
    """Incarta vs scPortrait, linear (Ridge) AND nonlinear (LightGBM) floor.

    scPortrait's weakness turned out to be its *segmentation*, not its
    *representation* (see A5) -- so the comparison must show the verdict holds at
    both rungs, not just the linear one the first pass showed.
    """
    rows: List[dict] = []
    missing: List[str] = []
    for exp in config.scportrait_experiments:
        paths = resolve_experiment_paths(exp, config)

        ridge_rows = _collect_a2_rung(
            exp,
            "Ridge",
            paths.baseline_ladder_csv,
            paths.scportrait_baseline_ladder_csv,
            lambda df: df[df["model"] == "ridge"],
        )
        lightgbm_rows = _collect_a2_rung(
            exp,
            "LightGBM",
            paths.floor_metrics_csv,
            paths.scportrait_floor_metrics_csv,
            lambda df: df[(df["model"] != "ridge") & (df["variant"] == "with_suspect")],
        )
        rows.extend(ridge_rows)
        rows.extend(lightgbm_rows)
        if not ridge_rows or not lightgbm_rows:
            missing.append(exp)

    if not rows:
        return FigureStatus(
            figure_id="A2",
            title=(
                "Handcrafted vs deep features (incarta vs scPortrait), "
                "linear + nonlinear floor"
            ),
            status="blocked",
            missing=missing,
            note="No experiment has both an incarta and a scPortrait result, at "
            "either rung.",
        )

    table = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, rung in zip(axes, ("Ridge", "LightGBM")):
        subset = table[table["rung"] == rung]
        if subset.empty:
            ax.text(
                0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            pivot = subset.pivot_table(
                index=["experiment", "target"], columns="backend", values="r2"
            )
            pivot.index = [f"{exp}\n{target}" for exp, target in pivot.index]
            pivot.plot(
                kind="bar", ax=ax, color={"incarta": "#0072B2", "scPortrait": "#E69F00"}
            )
            ax.tick_params(axis="x", rotation=45)
        ax.set_title(f"{rung} floor")
        ax.set_ylabel("R²")
        ax.set_xlabel("")
    fig.suptitle("Incarta vs scPortrait: linear + nonlinear floor")
    fig.text(
        0.5,
        -0.02,
        "Verdict holds at both rungs -- scPortrait shows no nonlinear gain. Cell "
        "populations/targets differ between backends (scPortrait re-extracts its "
        "target against its own masks).",
        ha="center",
        fontsize=8,
        style="italic",
    )
    fig.tight_layout()

    written = save_fig(
        fig, config.output_dir, "A2_incarta_vs_scportrait", config.formats, config.dpi
    )

    status = "generated" if not missing else "partial"
    return FigureStatus(
        figure_id="A2",
        title=(
            "Handcrafted vs deep features (incarta vs scPortrait), "
            "linear + nonlinear floor"
        ),
        status=status,
        output_paths=[str(p) for p in written],
        missing=missing,
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
    percentiles = [p for p in PERCENTILE_ORDER if p in table["target"].unique()]

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
    percentiles = [p for p in PERCENTILE_ORDER if p in pivot_r2.columns]
    pivot_r2 = pivot_r2.reindex(index=variant_order)[percentiles]
    pivot_r2.plot(kind="bar", ax=ax_r2)
    ax_r2.axhline(0, color="grey", linewidth=1)
    ax_r2.set_title("(a) Ridge R² swings across populations")
    ax_r2.set_ylabel("R²")
    ax_r2.set_xlabel("")
    ax_r2.tick_params(axis="x", rotation=0)

    mae_table = pd.DataFrame(mae_rows)
    representative_target = "percentile_75"
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
