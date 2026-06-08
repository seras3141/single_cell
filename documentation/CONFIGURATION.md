# Configuration

The pipeline uses YAML configuration files loaded via [OmegaConf](https://omegaconf.readthedocs.io/) with typed dataclass schemas. Do not pass raw dicts through module boundaries — use the schema types defined in `src/utils/config_schemas.py`.

---

## Main configuration (`config/config.yaml`)

```yaml
segmentation:
  model_type: "cyto3"
  flow_threshold: 0.4
  min_cell_size: 30

tracking:
  search_range: 5
  memory: 1
  blur_threshold: 0.5

preprocessing:
  test_fraction: 0.2
  random_seed: 42
  patch_size: 32
  stride_size: 16
```

---

## Inference configuration (`config/inference_config.yaml`)

```yaml
model:
  name: "cyto3"
  flow_threshold: 0.4
  gpu: true

input:
  directory: "data/processed/split/test"
  pattern: "*_BF.tif"

output:
  directory: "results/inference"
  save_overlays: true
  save_metadata: true
```

---

## Feature extraction configuration (`config/feature_extraction_config.yaml`)

```yaml
paths:
  input_dir: "data/segmented"
  output_dir: "data/features_output"

feature_extraction:
  n_jobs: -1
  features:
    morphology: true
    intensity: true
    spatial: true
    texture: true
```

---

## Adding a new configuration section

1. Add a typed dataclass schema to `src/utils/config_schemas.py`.
2. Register it in `ConfigManager`.
3. Reference the new section via dot notation: `config.get("your_section.your_key")`.

---

## Supported file formats

| Type | Formats |
|---|---|
| Input images | Multi-well plate TIFF (standardised naming — see `src/utils/file_utils.py`) |
| Segmentation masks | TIFF (default), Zarr, HDF5 for 3D data |
| Feature tables | CSV (one row per cell) |
| Configuration | YAML |

Output format selection is handled by `OutputManager` — do not hardcode `.tif` suffixes in pipeline code.
