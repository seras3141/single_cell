# Feature Extraction

Extracts per-cell features from 2D and 3D segmented microscopy images. Each extractor takes a segmentation mask and optionally a paired intensity image, and returns a `DataFrame` with one row per cell instance.

## Module structure

```
src/feature_extraction/
├── feature_extraction_pipeline.py   # Orchestrates batch extraction across datasets
├── feature_extractor_incarta.py     # Custom 2D extractor: morphology, intensity, texture
├── feature_extractor_regionprops.py # scikit-image regionprops (2D and 3D)
├── feature_extractor_pyradiomics.py # PyRadiomics per-cell features (separate env)
├── feature_extractor_scportrait.py  # ConvNeXt deep features via scPortrait (optional dependency)
├── scportrait_project/              # scPortrait project config and helpers
│   └── config.yml                   # CytosolOnlySegmentationCellpose + ConvNeXtFeaturizer
├── feature_list_2d.txt              # Full description of the 2D feature set
├── feature_list_3d.txt              # Full description of the 3D feature set
└── __init__.py
```

## Extraction methods

The method is set via `feature_extraction.method` in `config/feature_extraction_config.yaml`, or passed directly to `FeatureExtractionPipeline`.

| Method | What it computes | Key dependencies |
|---|---|---|
| `incarta` *(default)* | 25 handcrafted 2D features across four groups (see below) | `scikit-image`, `scipy` |
| `regionprops` | Standard skimage `regionprops_table` properties. `get_region_properties` accepts 2D and 3D arrays; the batch pipeline feeds it 2D slices only | `scikit-image` |
| `pyradiomics` | 102 PyRadiomics features per cell (`shape2D`, `firstorder`, `glcm`, `glrlm`, `glszm`, `gldm`, `ngtdm`; original image only), plus `n_pixels`, `touches_border` and `extraction_seconds`. Settings live under `feature_extraction.pyradiomics` | `pyradiomics-cuda`, `SimpleITK` *(separate `.venv-pyradiomics`)* |
| `scportrait` | ConvNeXt encoder embeddings per cell via scPortrait's segment→extract→featurize pipeline | `scportrait` *(optional, requires Python ≥ 3.11)* |

`scportrait` is imported with a try/except, so the pipeline still imports when it is not installed. `scportrait` cannot share the primary environment (it pins `cellpose<4`); install it into a separate Python 3.11 environment from [`requirements-scportrait.txt`](requirements-scportrait.txt) — see [scPortrait method](#scportrait-method) below.

`pyradiomics` imports PyRadiomics and SimpleITK only when it extracts, so the module loads anywhere. It is meant to run in a separate `.venv-pyradiomics`, where `pyradiomics-cuda` provides the `radiomics` package; it can't share `.venv` with stock PyRadiomics. That environment and its SLURM launcher are not in the repo yet. Without the backend, it fails on the first file with an `ImportError`. Labels below `min_pixels` (default 20) are skipped and counted.

### Outputs

- **Format and grouping.** `output.format` is `csv` (default) or `parquet`; `parquet` needs `pyarrow`.
  `output.granularity` is `image` (default: one file per image) or `well`. With `well`, the run writes
  `<well>.parquet` (all cells of that well) and `<well>_coverage.parquet` per well; `well` requires
  `parquet`. [`config/feature_extraction_pyradiomics_config.yaml`](../../config/feature_extraction_pyradiomics_config.yaml)
  uses per-well Parquet.
- **Coverage.** Every attempted image gets one coverage record: filename, key columns, `status`
  (`ok`/`empty`/`error`), `n_cells`, `n_skipped_small` and `seconds`. That's what tells a genuinely empty
  image from one that was never processed. The summary reports the status counts.

### Inputs, errors and exclusions

- **Loading.** Masks are read with `src.utils.image_utils.load_labels` (TIFF, zarr or HDF5, any label dtype including uint32), and BF images with `load_image`. Both must be 2D and the same shape; anything else is a per-file error.
- **Any per-file error fails the run.** Errors are logged in the run log (including those from parallel workers), and `feature_extraction_summary.txt` lists every failed file. It is written on every run, whatever `save_combined_file` says. The CLI then exits non-zero. Code defects and missing dependencies (`NameError`, `ImportError`, `NotImplementedError`) are re-raised on the first file instead, after the summary is written. `TypeError` and `AttributeError` are recorded per file, because numeric libraries also raise them for bad input data (e.g. a float label image).
- **Unpaired masks.** A mask with no matching image is an error, unless its slice is listed under `known_missing` in [`config/data_exclusions.yaml`](../../config/data_exclusions.yaml). Images with no mask (e.g. z0 projections) are only counted.
- **Excluded stacks.** Stacks listed under `excluded_stacks` are still extracted. The `feature_to_mcherry` loaders drop their rows.

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

`scripts/run_feature_extraction.py` supports three input modes; direct options override the config.

```bash
# 1. Config-driven batch (image/mask dirs from the config file)
python scripts/run_feature_extraction.py --config config/feature_extraction_config.yaml

# 2. Batch over a directory (mask dir optional — not needed for scportrait)
python scripts/run_feature_extraction.py \
  --method scportrait \
  --image-dir data/sample_data/HD1883 \
  --image-pattern "*_BF.tif" \
  --output-dir tmp/scportrait_sample/HD1883

# 3. Single image (--mask-file required for non-scportrait methods)
python scripts/run_feature_extraction.py \
  --method scportrait \
  --image-file "data/.../pMF5V1_E07_t1_z10_BF.tif" \
  --output-dir tmp/scportrait_single
```

Output is written to `--output-dir` (or `paths.output_dir` from config). Per-image CSVs and a combined CSV are saved depending on the `output` settings in config.

> **scPortrait needs its own environment.** It pins `cellpose<4`, which conflicts with the cellpose-sam (cellpose 4.x) stack used elsewhere, and it requires Python 3.11. Install it into a separate environment from [`requirements-scportrait.txt`](requirements-scportrait.txt) (see the header of that file for setup steps), and run on a GPU node (via SLURM) for ConvNeXt featurization.

## Adding a new extractor

1. Create `feature_extractor_<name>.py` with a function that accepts `(mask, image)` and returns a `pd.DataFrame` with one row per instance.
2. Import it (with a try/except if it has optional dependencies) in `feature_extraction_pipeline.py`.
3. Add `<name>` to the valid methods list in `FeatureExtractionPipeline.__init__` and handle it in `extract_features_from_path`.
4. Add tests under `tests/feature_extraction/`.

---

## scportrait method

`method: scportrait` runs scPortrait's full segment→extract→featurize workflow on each brightfield image, returning a DataFrame of ConvNeXt encoder embeddings (one row per cell).

The pipeline internally duplicates the single BF image to satisfy the two-channel requirement of `CytosolOnlySegmentationCellpose`. A per-image project directory is created under `feature_extraction.scportrait.project_location/<image_stem>/`.

**Config reference** (`config/feature_extraction_config.yaml`):

```yaml
feature_extraction:
  method: "scportrait"
  scportrait:
    project_location: "tmp/scportrait_projects"
    config_path: "src/feature_extraction/scportrait_project/config.yml"
    channel_names: ["brightfield", "brightfield_ch1"]
    overwrite: true
    debug: false
    save_plots: true   # saves segmentation/extraction/featurization PNGs per image
```

The scPortrait project config (`scportrait_project/config.yml`) specifies:
- `CytosolOnlySegmentationCellpose` with `cyto3`
- `HDF5CellExtraction` at 128 × 128 px
- `ConvNeXtFeaturizer` with `channel_selection: 0` (brightfield channel)
