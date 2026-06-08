"""Analytics outputs for extracted mCherry metrics."""

from .correlation import plot_metric_correlation_heatmap
from .distributions import (
    plot_area_vs_intensity,
    plot_metric_distributions,
    plot_metric_violins,
)
from .quality import write_quality_report
from .summary import build_metrics_summary, write_metrics_summary


def generate_standard_outputs(
    metrics_df,
    save_dir,
    processed_image_paths=None,
):
    """Generate the standard analytics output suite for milestone 1."""
    save_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    outputs.extend(plot_metric_distributions(metrics_df, save_dir))
    outputs.extend(plot_metric_violins(metrics_df, save_dir))
    outputs.append(plot_metric_correlation_heatmap(metrics_df, save_dir))
    outputs.append(write_metrics_summary(metrics_df, save_dir / "metrics_summary.csv"))
    outputs.append(
        write_quality_report(
            metrics_df,
            save_dir / "qc_report.txt",
            processed_image_paths=processed_image_paths,
        )
    )
    outputs.append(plot_area_vs_intensity(metrics_df, save_dir))
    return outputs


__all__ = [
    "build_metrics_summary",
    "generate_standard_outputs",
    "plot_area_vs_intensity",
    "plot_metric_correlation_heatmap",
    "plot_metric_distributions",
    "plot_metric_violins",
    "write_metrics_summary",
    "write_quality_report",
]