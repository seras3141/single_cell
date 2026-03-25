# Configuration Reference

Configuration files for the single cell analysis pipeline are located in the `config/` directory. All scripts accept a `--config` flag pointing to one of these files, and individual values can be overridden at runtime with `--override key=value`.

---

## Preprocessing Configuration

**File:** `config/preprocessing_config.yaml`

```yaml
paths:
  input_dir: "data/raw"
  output_dir: "data/processed"

preprocessing:
  test_size: 0.2
  random_state: 42
  split_by_group: true
  split_folder: "split_data"
  out_3d_folder: "3d_data"

  # Optional: override wavelength-to-channel mappings
  # Defaults to config/wavelength_config.yaml if omitted
  wavelength_mappings:
    1: "AnnexinV"
    2: "mCherry"
    3: "AnnexinV"

  # Optional: default plate number (overrides auto-detection from filepath)
  # plate_number: "2126"

  # Z-range for 2D-to-3D combination
  # z0 is the 2D projection and is skipped by default (z_min: 1)
  z_min: 1       # first z-slice to include (inclusive)
  # z_max: 20   # last z-slice to include (inclusive); omit for no upper limit

quality:
  blur_detection:
    patch_size: [32, 32]
    stride_size: [8, 8]
```

---

## Inference Configuration

**File:** `config/inference_config.yaml`

```yaml
inference:
  input_dir: "data/processed/split/test"
  output_dir: "results/inference"
  model_name: "cyto3"
  flow_threshold: 0.4
  diameter: 30
  gpu: true
  save_overlays: true
  save_metadata: true
```

---

## Postprocessing Configuration

**File:** `config/postprocessing_config.yaml`

```yaml
postprocessing:
  tracking:
    search_range: 5
    memory: 1
    min_mass: 100

  filtering:
    blur_threshold: 0.5
    min_area: 50
    max_area: 5000

  output:
    save_tracks: true
    save_filtered: true
    save_statistics: true
```

---

## Wavelength Configuration

**File:** `config/wavelength_config.yaml`

Maps wavelength indices to channel names. Used by `ConfigurableFileHandler` as the default mapping when `preprocessing.wavelength_mappings` is not set in the preprocessing config.

```yaml
wavelength_mappings:
  1: "BF"
  2: "mCherry"
  3: "AnnexinV"
```
