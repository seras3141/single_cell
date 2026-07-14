MORPHOLOGY_FEATURES = [
    'area', 'perimeter', 'elongation', 'compactness', 'circularity',
    'feret_diameter', 'radius_of_gyration', 'major_axis', 'minor_axis'
]

INTENSITY_FEATURES = [
    'mean_intensity', 'std_intensity', 'cv_intensity', 'total_intensity'
]

TEXTURE_FEATURES = [
    'gabor_mean', 'gabor_std', 'skewness', 'kurtosis', 'entropy',
    # 'contrast', 'correlation', 'energy', 'homogeneity', 'dissimilarity',
    # 'asm', 'angular_second_moment'
]

SPATIAL_FEATURES = [
    'centroid_x', 'centroid_y', 'center_of_mass_x', 'center_of_mass_y', 
    'mass_displacement'
]

# Columns to exclude from feature analysis. ``cell_id`` is the current per-cell
# id (``instance_id`` kept for reading pre-rename CSVs; ``scportrait_cell_id``
# for the scPortrait backend). ``sample_id``/``timepoint``/``z_index`` are the
# per-cell key columns now written alongside the features.
METADATA_COLUMNS = [
    'cell_id', 'scportrait_cell_id', 'instance_id', 'filename', 'image_path',
    'image_filename', 'mask_filename', 'sample_id', 'timepoint', 'z_index',
    'z_stack', 'sample_z_id', 'processing_timestamp',
    'feature_extraction_version', 'dataset_name'
]

def get_all_feature_names() -> dict:
    """Get a dictionary of all feature names categorized by type."""
    return {
        "morphology": MORPHOLOGY_FEATURES,
        "intensity": INTENSITY_FEATURES,
        "spatial": SPATIAL_FEATURES,
        "texture": TEXTURE_FEATURES,
        "metadata": METADATA_COLUMNS
    }