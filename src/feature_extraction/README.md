# Feature Extraction

Extracts per-cell features from 2D and 3D segmented microscopy images. Each extractor takes a segmentation mask and optionally a paired intensity image, and returns a `DataFrame` with one row per cell instance.

## Module structure

```
src/feature_extraction/
├── feature_extraction_pipeline.py   # Orchestrates batch extraction across datasets
├── feature_extractor_incarta.py     # Custom 2D extractor: morphology, intensity, texture
├── feature_extractor_regionprops.py # scikit-image regionprops (2D and 3D)
├── feature_extractor_pyradiomics.py # PyRadiomics via SimpleITK (optional dependency)
├── feature_list_2d.txt              # Full description of the 2D feature set
├── feature_list_3d.txt              # Full description of the 3D feature set
└── __init__.py
```

## Extraction methods

The method is set via `feature_extraction.method` in `config/feature_extraction_config.yaml`, or passed directly to `FeatureExtractionPipeline`.

| Method | What it computes | Key dependencies |
|---|---|---|
| `incarta` *(default)* | 25 handcrafted 2D features across four groups (see below) | `scikit-image`, `scipy` |
| `regionprops` | Standard skimage `regionprops_table` properties for 2D and 3D masks | `scikit-image` |
| `pyradiomics` | Radiomic texture and shape features | `pyradiomics`, `SimpleITK` *(optional)* |

`pyradiomics` is imported with a try/except — the pipeline falls back gracefully if the package is not installed.

## Feature groups (`incarta`, 2D)

See `feature_list_2d.txt` for full definitions. Summary:

| Group | Count | Features |
|---|---|---|
| Morphology | 9 | area, perimeter, elongation, compactness, circularity, feret diameter, radius of gyration, major/minor axis |
| Intensity | 4 | mean, std, coefficient of variation, total intensity |
| Spatial | 5 | centroid x/y, center of mass x/y, mass displacement |
| Texture | 6 | Gabor mean/std, skewness, kurtosis, entropy |

Individual feature groups can be toggled in config under `feature_extraction.features`.

## Usage

**From Python:**

```python
from src.feature_extraction.feature_extraction_pipeline import FeatureExtractionPipeline

pipeline = FeatureExtractionPipeline.from_config(config)
features_df = pipeline.run(image_dirs=[...], mask_dirs=[...])
```

**From the command line:**

```bash
python scripts/run_feature_extraction.py --config config/feature_extraction_config.yaml
```

Output is written to the path set in `paths.output_dir`. Per-image CSVs and an optional combined CSV are saved depending on the `output` settings in config.

## Adding a new extractor

1. Create `feature_extractor_<name>.py` with a function that accepts `(mask, image)` and returns a `pd.DataFrame` with one row per instance.
2. Import it (with a try/except if it has optional dependencies) in `feature_extraction_pipeline.py`.
3. Add `<name>` to the valid methods list in `FeatureExtractionPipeline.__init__` and handle it in `extract_features_from_path`.
4. Add tests under `tests/feature_extraction/`.

---

## Planned: scportrait extractor

> **Status: not yet implemented**

A `feature_extractor_scportrait.py` extractor is planned for this module. [scportrait](https://github.com/MannLabs/scPortrait) is an open-source framework for single-cell image analysis that provides standardised HDF5-based data structures (SPARCSpy format) and integrates directly with segmentation outputs.

**Motivation:** scportrait enables direct access to single-cell crops stored in its HDF5 project format, removing the need to pair raw images with masks manually. This makes it a better fit for datasets already processed through a scportrait/SPARCSpy pipeline.

**Planned interface** (subject to change):

```python
# feature_extractor_scportrait.py
def get_scportrait_features(scportrait_project_path: str | Path, ...) -> pd.DataFrame:
    ...
```

**Integration steps when implementing:**

1. Implement `feature_extractor_scportrait.py` following the interface above.
2. Register `scportrait` as a valid method in `FeatureExtractionPipeline`.
3. Add `scportrait` config section to `config/feature_extraction_config.yaml`.
4. Add tests under `tests/feature_extraction/test_feature_extractor_scportrait.py`.
