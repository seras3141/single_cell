"""Dataset-level analysis, QC helpers, and pipeline run tracking for raw microscopy plate inventories."""

from .run_manifest import (
    RunManifest,
    StageRecord,
    create_or_load_manifest,
    find_manifests,
    STAGE_ORDER,
    MANIFEST_FILENAME,
)
from .image_quality_metrics import (
    get_image_quality_metrics,
    get_percentile_images,
    get_quality_metrics,
    max_intensity_projection_metrics,
)
from .inventory import (
    build_dataset_inventory,
    discover_image_files,
    load_expected_channels,
    parse_image_metadata,
)
from .layout import (
    build_plate_annotation_dataframe,
    get_well_annotation,
    load_plate_layout,
)
from .plotting import (
    plot_channel_completeness,
    plot_control_distribution,
    plot_drug_distribution,
    plot_plate_coverage,
    plot_z_completeness,
)
from .qc import (
    DEFAULT_EXPECTED_Z_INDICES,
    DEFAULT_PROJECTION_Z_INDEX,
    build_completeness_table,
    find_dataset_issues,
)
from .summary import build_dataset_summary, build_summary_table, write_summary_json
from .processed_inventory import (
    annotate_with_raw_issues,
    build_processed_inventory,
    build_processed_summary,
    detect_phantom_samples,
    parse_sample_stem,
    print_phantom_report,
    print_summary_table,
)

__all__ = [
    # Run manifest
    "RunManifest",
    "StageRecord",
    "create_or_load_manifest",
    "find_manifests",
    "STAGE_ORDER",
    "MANIFEST_FILENAME",
    # QC
    "DEFAULT_EXPECTED_Z_INDICES",
    "get_image_quality_metrics",
    "get_percentile_images",
    "get_quality_metrics",
    "max_intensity_projection_metrics",
    "DEFAULT_PROJECTION_Z_INDEX",
    "build_completeness_table",
    "build_dataset_inventory",
    "build_dataset_summary",
    "build_plate_annotation_dataframe",
    "build_summary_table",
    "discover_image_files",
    "find_dataset_issues",
    "get_well_annotation",
    "load_expected_channels",
    "load_plate_layout",
    "parse_image_metadata",
    "plot_channel_completeness",
    "plot_control_distribution",
    "plot_drug_distribution",
    "plot_plate_coverage",
    "plot_z_completeness",
    "write_summary_json",
    # Processed inventory
    "annotate_with_raw_issues",
    "build_processed_inventory",
    "build_processed_summary",
    "detect_phantom_samples",
    "parse_sample_stem",
    "print_phantom_report",
    "print_summary_table",
]
