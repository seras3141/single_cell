
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional

try:
    from IPython.display import display as _ipy_display
    _HAS_IPY = True
except ImportError:
    _HAS_IPY = False


def _show_figure(fig) -> None:
    if _HAS_IPY:
        _ipy_display(fig)
    else:
        plt.show()
    plt.close(fig)

DEFAULT_INSTANCE_METRICS = [
    'mean_intensity',
    'max_intensity',
    'percentile_95',
    'percentile_90',
    'percentile_75',
    'sum_intensity',
]


def plot_single_metric(metrics_df: pd.DataFrame, metric: str, ax) -> None:
    """Plot one metric distribution on an existing matplotlib axis."""
    data = metrics_df[metric].dropna()
    if data.empty:
        ax.text(0.5, 0.5, f'{metric} - no data', ha='center', va='center')
        ax.set_axis_off()
        return

    color_index = ax.get_subplotspec().colspan.start
    if metric == 'sum_intensity':
        data_pos = data[data > 0]
        if data_pos.empty:
            ax.text(0.5, 0.5, 'sum_intensity - no positive data', ha='center', va='center')
            ax.set_axis_off()
            return
        log_data = np.log10(data_pos)
        clip_val = np.percentile(log_data, 99)
        plot_data = log_data[log_data <= clip_val]
        sns.histplot(plot_data, bins=50, kde=True, stat='density', color=f'C{color_index}', ax=ax)
        ax.set_title(f'{metric} (log10)')
        ax.set_xlabel('log10(Sum Intensity)')
        ax.set_ylabel('Density')
        median_log = np.median(log_data)
        ax.axvline(median_log, color='k', linestyle='--', linewidth=1.5)
        orig_median = 10 ** median_log
        ax.text(
            median_log,
            ax.get_ylim()[1] * 0.9,
            f'Median=10^{median_log:.2f}\n({orig_median:,.0f})',
            ha='right',
            va='top',
            fontsize=9,
            bbox=dict(facecolor='white', alpha=0.6),
        )
        ax.grid(alpha=0.3)
        return

    data_pos = data[data > 0]
    if data_pos.empty:
        ax.text(0.5, 0.5, f'{metric} - no positive data', ha='center', va='center')
        ax.set_axis_off()
        return

    data = np.clip(data, 0, data_pos.quantile(0.99))
    sns.histplot(data, bins=50, kde=True, stat='density', color=f'C{color_index}', ax=ax)
    ax.set_title(f'{metric} (clipped to 99th pct)')
    ax.set_xlabel('Intensity Value')
    ax.set_ylabel('Density')
    ax.axvline(data_pos.median(), color='k', linestyle='--', linewidth=1.5)
    ax.grid(alpha=0.3)


def plot_stats(metrics_df: pd.DataFrame, metrics_to_plot: Optional[list[str]] = None) -> None:
    """Plot the default notebook metric distributions and print compact stats."""
    metrics_to_plot = metrics_to_plot or DEFAULT_INSTANCE_METRICS
    metrics_to_plot = [metric for metric in metrics_to_plot if metric in metrics_df.columns]
    n_plots = len(metrics_to_plot)
    if n_plots == 0:
        print("No supported metrics available to plot.")
        return

    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_arr = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for index, metric in enumerate(metrics_to_plot):
        plot_single_metric(metrics_df, metric, axes_arr[index])

    for index in range(n_plots, len(axes_arr)):
        fig.delaxes(axes_arr[index])

    fig.suptitle('Distribution plots')
    plt.tight_layout()
    _show_figure(fig)

    print('=== SELECTED METRICS SUMMARY ===')
    for metric in metrics_to_plot:
        values = metrics_df[metric].dropna()
        if values.empty:
            print(f'{metric}: no data')
            continue
        if metric == 'sum_intensity':
            positive = values[values > 0]
            median_val = positive.median() if len(positive) else np.nan
            print(f'{metric}: n={len(values)}, median={median_val:,.0f}, mean={values.mean():.2f}, std={values.std():.2f}')
        else:
            print(f'{metric}: n={len(values)}, mean={values.mean():.2f}, median={values.median():.2f}, std={values.std():.2f}')


def plot_classification_distributions(
    metrics_df: pd.DataFrame,
    selected_metric: str,
    active_label: str = 'active',
    dead_label: str = 'dead',
) -> None:
    """Plot classified metric distributions for active and dead cells."""
    fig, ax = plt.subplots(figsize=(10, 6))
    metric_col = 'metric_value'
    if metric_col not in metrics_df.columns:
        intensity_cols = [col for col in metrics_df.columns if 'intensity' in col.lower()]
        metric_col = intensity_cols[0] if intensity_cols else selected_metric

    active_data = metrics_df[metrics_df['is_active']][metric_col]
    dead_data = metrics_df[~metrics_df['is_active']][metric_col]

    upper = metrics_df[metric_col].dropna().quantile(0.99)
    active_plot = active_data.dropna().clip(upper=upper)
    dead_plot = dead_data.dropna().clip(upper=upper)

    if len(active_plot) > 1:
        sns.kdeplot(data=active_plot, label=active_label, fill=True, ax=ax, color='green', alpha=0.6)
    if len(dead_plot) > 1:
        sns.kdeplot(data=dead_plot, label=dead_label, fill=True, ax=ax, color='red', alpha=0.6)

    ax.set_xlabel(f'{selected_metric} (clipped at 99th percentile)')
    ax.set_ylabel('Density')
    ax.set_title(f'{selected_metric} distribution by classification')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _show_figure(fig)


def plot_instance_metrics_distribution(metrics_df: pd.DataFrame, output_path: Optional[str] = None):
    """Plot distribution of instance-level metrics with log-scaling and median lines.
    
    Args:
        metrics_df: DataFrame containing instance-level metrics.
    """
    # Distribution-only metrics plot with log-scaling, median lines, and 99th-percentile clipping
    metrics_to_plot = ['mean_intensity', 'max_intensity', 'percentile_95', 'percentile_90', 'percentile_75', 'sum_intensity']
    # Keep only available metrics
    metrics_to_plot = [m for m in metrics_to_plot if m in metrics_df.columns]
    n_plots = len(metrics_to_plot)
    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    # Ensure axes is iterable
    if hasattr(axes, 'flatten'):
        axes_arr = axes.flatten()
    else:
        axes_arr = [axes]

    for i, metric in enumerate(metrics_to_plot):
        ax = axes_arr[i]
        data = metrics_df[metric].dropna()
        if data.empty:
            ax.text(0.5, 0.5, f'{metric} - no data', ha='center', va='center')
            ax.set_axis_off()
            continue

        # sum_intensity: plot log10 values (to handle large scale)
        if metric == 'sum_intensity':
            data_pos = data[data > 0]
            if data_pos.empty:
                ax.text(0.5, 0.5, 'sum_intensity - no positive data', ha='center', va='center')
                ax.set_axis_off()
                continue
            log_data = np.log10(data_pos)
            clip_val = np.percentile(log_data, 99)
            plot_data = log_data[log_data <= clip_val]
            sns.histplot(plot_data, bins=50, kde=True, stat='density', color=f'C{i}', ax=ax)
            ax.set_title(f'{metric} Distribution (log10)')
            ax.set_xlabel('log10(Sum Intensity)')
            ax.set_ylabel('Density')
            median_log = np.median(log_data)
            ax.axvline(median_log, color='k', linestyle='--', linewidth=1.5)
            orig_median = 10 ** median_log
            ax.text(median_log, ax.get_ylim()[1] * 0.9, f'Median=10^{median_log:.2f}\n({orig_median:,.0f})', 
                    ha='right', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.6))
            ax.grid(alpha=0.3)

        else:
            # Other metrics: clip at 99th percentile and use log x-scale
            data_pos = data[data > 0]
            if data_pos.empty:
                print("zero values found")
            
            # Clip data at 99th percentile
            data = np.clip(data, 0, data_pos.quantile(0.99))
            sns.histplot(data, bins=50, kde=True, stat='density', color=f'C{i}', ax=ax)
            # ax.set_xscale('log')
            ax.set_title(f'{metric} Distribution (clipped to 99th pct)')
            ax.set_xlabel('Intensity Value (log scale)')
            ax.set_ylabel('Density')
            median_val = data_pos.median()
            ax.axvline(median_val, color='k', linestyle='--', linewidth=1.5)
            # ax.text(median_val, ax.get_ylim()[1] * 0.9, f'Median={median_val:.1f}', ha='right', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.6))
            ax.grid(alpha=0.3)

    # Remove any unused axes
    for j in range(n_plots, len(axes_arr)):
        fig.delaxes(axes_arr[j])

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)

    _show_figure(fig)

    # Print a short summary of chosen metrics
    print('=== SELECTED METRICS SUMMARY ===')
    for metric in metrics_to_plot:
        s = metrics_df[metric].dropna()
        if s.empty:
            print(f'{metric}: no data')
            continue
        if metric == 'sum_intensity':
            s_pos = s[s > 0]
            median_val = s_pos.median() if len(s_pos) else np.nan
            print(f'{metric}: n={len(s)}, median={median_val:,.0f}, mean={s.mean():.2f}, std={s.std():.2f}')
        else:
            print(f'{metric}: n={len(s)}, mean={s.mean():.2f}, median={s.median():.2f}, std={s.std():.2f}')
        

def plot_active_vs_dead_distribution(metrics_df: pd.DataFrame, 
                                     threshold_metric: str,
                                     threshold_method: str, 
                                     active_label: str = 'active',
                                     dead_label: str = 'dead'):
    """Plot distribution of metrics for active vs dead cells based on thresholding."""
    # Additional statistics plot
    plt.figure(figsize=(10, 6))
    metrics_available = ['max_intensity', 'percentile_95', 'percentile_90', 'percentile_75', 'mean_intensity', 'sum_intensity']
    metrics_to_plot = [m for m in metrics_available if m in metrics_df.columns]

    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 3, i+1)
        # First get the metric and clip extreme values for better visualization
        plot_data = metrics_df[metric]
        plot_data = np.clip(plot_data, 0, plot_data.quantile(0.99))
        # Plot histograms for active vs dead cells
        active_vals = plot_data[metrics_df['is_active']]
        dead_vals = plot_data[~metrics_df['is_active']]
        
        plt.hist(dead_vals, bins=30, alpha=0.7, label=f'{dead_label}', color='blue', density=True)
        plt.hist(active_vals, bins=30, alpha=0.7, label=f'{active_label}', color='red', density=True)
        plt.xlabel(metric)
        plt.ylabel('Density')
        plt.title(f'{metric} Distribution')
        plt.legend()

    plt.suptitle(f'Active vs dead cell distribtuion when thresholded by {threshold_metric} with threshold method: {threshold_method}', y=1.02)
    plt.tight_layout()
    _show_figure(plt.gcf())
