
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional

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

    plt.show()

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
    plt.show()