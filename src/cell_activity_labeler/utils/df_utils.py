"""DataFrame utilities for filtering and aggregating classification results."""
from __future__ import annotations

import pandas as pd


def filter_images_by_sample_id(metrics_df: pd.DataFrame, sample_id: str) -> pd.DataFrame:
    """
    Return a subset of ``metrics_df`` matching the given sample ID.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Full classification results DataFrame.
    sample_id : str
        Filter to rows whose ``ID`` column equals this value.

    Returns
    -------
    pd.DataFrame
        Filtered copy of ``metrics_df``.
    """
    return metrics_df[metrics_df['ID'] == sample_id].copy().reset_index(drop=True)


def filter_images_by_z_index(
    metrics_df: pd.DataFrame,
    min_z: int = 1,
    max_z: int = 20,
) -> pd.DataFrame:
    """
    Return a subset of ``metrics_df`` matching the given sample and z-indices.

    z0 is always excluded (``min_z`` defaults to 1).  The ``z_index`` column
    is coerced to numeric before filtering so string values are handled
    transparently.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Full classification results DataFrame.
    min_z : int
        Minimum z-index to keep (inclusive).  Default 1 excludes z0.
    max_z : int
        Maximum z-index to keep (inclusive).  Default 20.

    Returns
    -------
    pd.DataFrame
        Filtered copy of ``metrics_df``.
    """
    filtered_data = metrics_df.copy()

    if 'z_index' in filtered_data.columns:
        filtered_data['z_index'] = pd.to_numeric(filtered_data['z_index'], errors='coerce')
        filtered_data = filtered_data[
            filtered_data['z_index'].between(min_z, max_z, inclusive='both')
        ]

    filtered_data = filtered_data.sort_values('z_index').reset_index(drop=True)

    return filtered_data

def filter_images_by_sample_id_and_z_index(
    metrics_df: pd.DataFrame,
    sample_id: str,
    min_z: int = 1,
    max_z: int = 20,
) -> pd.DataFrame:
    """
    Return a subset of ``metrics_df`` matching the given sample and z-indices.

    Combines filtering by sample ID and z-index into one step.  See
    ``filter_images_by_sample_id`` and ``filter_images_by_z_index`` for details.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Full classification results DataFrame.
    sample_id : str
        Filter to rows whose ``ID`` column equals this value.
    min_z : int
        Minimum z-index to keep (inclusive).  Default 1 excludes z0.
    max_z : int
        Maximum z-index to keep (inclusive).  Default 20.

    Returns
    -------
    pd.DataFrame
        Filtered copy of ``metrics_df``.
    """
    filtered_data = filter_images_by_sample_id(metrics_df, sample_id)
    filtered_data = filter_images_by_z_index(filtered_data, min_z, max_z)
    return filtered_data


def create_sample_zstack_statistics(
    filtered_images: pd.DataFrame,
    sample_id: str,
) -> pd.DataFrame:
    """
    Compute and print per-z-slice activity statistics for a sample.

    Parameters
    ----------
    filtered_images : pd.DataFrame
        Subset of the classification DataFrame for one sample.  Must contain
        the columns ``z_index``, ``is_active``, ``metric_value``, and
        ``threshold``.
    sample_id : str
        Sample identifier used in the printed summary.

    Returns
    -------
    pd.DataFrame
        Per-z-index aggregated statistics.
    """
    sample_stats = filtered_images.groupby('z_index').agg({
        'is_active':    ['count', 'sum'],
        'metric_value': ['mean', 'median'],
        'threshold':    'first',
    }).round(2)

    total_cells = filtered_images.shape[0]
    total_active = sample_stats[('is_active', 'sum')].sum()
    overall_activity_rate = (total_active / total_cells) * 100 if total_cells > 0 else 0

    print(f"\nSample {sample_id} Z-Stack Statistics:")
    print(f"   Total z-slices loaded: {len(filtered_images['z_index'].unique())}")
    print(f"   Total cells across all slices: {total_cells}")
    print(f"   Active cells: {total_active} ({overall_activity_rate:.1f}%)")
    print(f"   Dead cells: {total_cells - total_active} ({100 - overall_activity_rate:.1f}%)")

    return sample_stats


def extract_sample_info(image_name, file_handler):
    """Extract sample and z-index information from an image name using the file handler."""
    sample = file_handler.extract_sample_id(image_name)
    z_index = file_handler.extract_z_index(image_name)
    id = file_handler.extract_unique_id(image_name)
    timepoint = file_handler.extract_time_point(image_name)
    return sample, z_index, id, timepoint


def add_meta_info(df, file_handler):
    """Add 'sample', 'z_index', 'time' and 'ID' fields extracted from image names."""
    df = df.copy()
    assert file_handler is not None, "file_handler not found in globals. Cannot extract sample and z-index information."

    meta_info = df['image'].astype(str).apply(lambda x: extract_sample_info(x, file_handler))

    if not meta_info.tolist():
        raise ValueError("No sample and z-index information could be extracted from image names. Check the file naming convention and file handler configuration.")
    else:
        df['sample'], df['z_index'], df['ID'], df['time'] = zip(*meta_info)

    return df

def get_sample_z_summary(df):

    # Check the extracted information
    print("Sample and Z-index extraction results:")
    sample_z_info = df[['image', 'sample', 'z_index']].drop_duplicates().sort_values(['sample', 'z_index'])
    print(sample_z_info.head(10))

    # Get unique samples and their z-index ranges
    sample_summary = df.groupby('sample')['z_index'].agg(['min', 'max', 'count', 'nunique']).reset_index()
    print(f"\nAvailable samples ({len(sample_summary)} total):")
    print(sample_summary)

    return sample_summary
