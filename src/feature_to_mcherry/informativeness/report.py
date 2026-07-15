"""Human-readable qualitative summary of the informativeness results.

Descriptive only — no numeric go/no-go cutoffs or auto-verdict string, per the
implementation plan's decision that the human reader judges the numbers (see
``plan_feature_to_activity_mapping.md`` Sec. 3.6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import pandas as pd


def _markdown_table(table: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(c) for c in table.columns) + " |"
    separator = "| " + " | ".join("---" for _ in table.columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in table.to_numpy()
    ]
    return "\n".join([header, separator, *rows])


def _format_floor_table(floor_metrics: pd.DataFrame) -> str:
    columns = [
        "target",
        "model",
        "variant",
        "backend",
        "tau",
        "mae",
        "r2",
        "pinball_loss",
    ]
    return _markdown_table(floor_metrics[columns].round(4))


def _format_noise_ceiling_table(noise_ceiling: pd.DataFrame) -> str:
    table = noise_ceiling.copy()
    table["ceiling"] = table["ceiling"].round(4)
    return _markdown_table(table)


def _format_top_correlations(
    univariate: pd.DataFrame, target_columns: List[str], top_k: int
) -> str:
    lines = []
    pooled = univariate[univariate["scope"] == "pooled"].copy()
    pooled["abs_rho"] = pooled["rho"].abs()
    for target in target_columns:
        subset = (
            pooled[pooled["target"] == target]
            .sort_values("abs_rho", ascending=False, na_position="last")
            .head(top_k)
        )
        lines.append(f"**{target}** (pooled Spearman rho, top {len(subset)}):")
        for _, row in subset.iterrows():
            lines.append(
                f"- `{row['feature']}`: rho={row['rho']:.3f} "
                f"(p={row['pvalue']:.3g}, n={int(row['n'])})"
            )
        lines.append("")
    return "\n".join(lines)


def render_report(bundle: Any) -> str:
    """Render the markdown report body for a
    :class:`feature_to_mcherry.informativeness.pipeline.ResultsBundle`."""
    lines = [
        "# Morphology-informativeness feasibility gate",
        "",
        "This is a feasibility check, not a predictive model: it quantifies how much "
        "interpretable brightfield morphology (size/shape + texture) predicts per-cell "
        "mCherry percentiles, and bounds what any downstream deep-feature model could "
        "reach (see `plan_feature_to_activity_mapping.md` Sec. 3.6). The numbers below "
        "are descriptive; there is no automatic go/no-go verdict — the reader judges.",
        "",
        "## Dataset",
        "",
        f"- n_cells matched to both features and targets: {bundle.n_cells}",
        f"- morphology feature columns: {bundle.n_features_all} total "
        f"({bundle.n_features_clean} 'clean', "
        f"{len(bundle.suspect_feature_names)} 'suspect')",
        f"- suspect (size/thickness/focus-proxy) columns: "
        f"{bundle.suspect_feature_names or 'none'}",
        f"- targets: {bundle.target_columns}",
        "",
        "## Top univariate associations (pooled Spearman rho)",
        "",
        _format_top_correlations(bundle.univariate, bundle.target_columns, top_k=10),
        "A high pooled rho next to a near-zero per-group rho for the same "
        "feature/target (see the `pooled_vs_group_rho` figure) indicates a batch "
        "effect rather than "
        "per-cell biology — check the `per_group` scope rows in "
        "`univariate_correlations.csv` before trusting a pooled number.",
        "",
        "## Multivariate performance floor (grouped cross-validation)",
        "",
        _format_floor_table(bundle.floor_metrics),
        "",
        "The floor is computed both including and excluding 'suspect' features (raw "
        "brightfield intensity, a thickness/focus proxy rather than unambiguous "
        "morphology) — compare the `with_suspect` and `without_suspect` variants to "
        "see how much apparent signal may be optical rather than biological.",
        "",
        "## Noise ceiling (replicate-well agreement, ICC)",
        "",
        _format_noise_ceiling_table(bundle.noise_ceiling),
        "",
        "The noise ceiling estimates how reproducible each target is across replicate "
        "wells of the same drug/dose condition, independent of any model. A low floor "
        "next to a healthy ceiling means there is headroom for a better model; a low "
        "floor next to a low ceiling means the target itself is noisy at the per-well "
        "level, and no model can close that gap.",
        "",
        "## Figures",
        "",
    ]
    for name, paths in bundle.figures.items():
        rel_paths = ", ".join(f"`{p.name}`" for p in paths)
        lines.append(f"- **{name}**: {rel_paths}")
    lines.append("")
    return "\n".join(lines)


def write_report(bundle: Any, output_dir: Path) -> Path:
    """Render and write ``report.md`` to ``output_dir``."""
    report_path = output_dir / "report.md"
    report_path.write_text(render_report(bundle))
    return report_path
