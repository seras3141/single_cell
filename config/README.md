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

  # Wavelength-to-channel mappings differ by experiment:
  #   Ew2-1, Ew2-2:           {1: FlipGFP, 2: mCherry, 3: BF}  ← default
  #   HD1509, HD1883, SA110:  {1: BF,      2: mCherry, 3: FlipGFP}
  # Leave null or omit to use the default (Ew2 convention).
  # Or use --experiment-name CLI flag to select automatically.
  wavelength_mappings:

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

**Wavelength mappings** are now hardcoded constants in `src/utils/file_utils.py` — `wavelength_config.yaml` is no longer loaded at runtime.

Two configurations are defined:

```python
# Ew2-1, Ew2-2 (default)
WAVELENGTH_MAPPINGS_EW2   = {1: "FlipGFP", 2: "mCherry", 3: "BF"}

# HD1509, HD1883, SA110
WAVELENGTH_MAPPINGS_HD_SA = {1: "BF", 2: "mCherry", 3: "FlipGFP"}
```

Select via `--experiment-name` CLI flag or pass `wavelength_mappings=EXPERIMENT_WAVELENGTH_MAPPINGS[name]` directly.
