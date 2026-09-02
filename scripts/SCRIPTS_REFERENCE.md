# Scripts Reference

Detailed usage, arguments, and configuration for each script in the `scripts/` directory.

---

## `run_preprocessing.py`

Complete data preprocessing pipeline including blur analysis and train/test splitting. Uses `ConfigurableFileHandler` for flexible file renaming with configurable wavelength-to-channel mappings and plate number extraction.

**Usage:**
```bash
# Full preprocessing pipeline using a config file
python scripts/run_preprocessing.py \
    --config config/preprocessing_config.yaml

# Specify input/output via CLI
python scripts/run_preprocessing.py \
    -i data/raw -o data/processed \
    --test-size 0.2 \
    --random-seed 42 \
    --patch-size 32 \
    --stride-size 16

# Select experiment by name (sets wavelength mappings automatically)
python scripts/run_preprocessing.py \
    -i data/raw -o data/processed \
    --experiment-name HD1509

# Override wavelength mappings manually at runtime
python scripts/run_preprocessing.py \
    -i data/raw -o data/processed \
    --wavelengths "1:BF,2:mCherry,3:FlipGFP" \
    --plate "2126"

# Override individual config values using dot notation
python scripts/run_preprocessing.py \
    --config config/preprocessing_config.yaml \
    --override preprocessing.plate_number=2126 paths.output_dir=data/out
```

**Arguments:**

| Argument | Description |
|---|---|
| `-i / --input-dir` | Path to the raw dataset directory |
| `-o / --output-dir` | Root output directory for all processed data |
| `--test-size` | Fraction of data for the test set (e.g. `0.2`) |
| `--random-seed` | Random seed for reproducibility |
| `--split-folder` | Folder name for split datasets |
| `--patch-size` | Patch size for blur detection |
| `--stride-size` | Stride size for blur detection |
| `--combine-pattern` | Regex pattern for 2D-to-3D grouping |
| `--experiment-name` | Experiment name — auto-sets wavelength mappings (e.g. `Ew2-1`, `HD1509`, `SA110`). Overridden by `--wavelengths`. |
| `--wavelengths` | Wavelength-to-channel mappings as `key:value` pairs (e.g. `1:FlipGFP,2:mCherry,3:BF`). Overrides `--experiment-name`. |
| `--plate` | Default plate number for file renaming (overrides auto-detection from filepath) |
| `--z-min` | Minimum z-index to include in 3D stacks (default: `1`; skips z0, the 2D projection) |
| `--z-max` | Maximum z-index to include in 3D stacks (default: no limit) |
| `--overwrite` | Overwrite existing output files |
| `--config` | Path to a YAML configuration file |
| `--override` | Override config values in dot notation (e.g. `paths.input_dir=data/input`) |

**Key Features:**
- Group-aware train/test splitting
- Blur heatmap generation
- 2D to 3D stack combination
- Configurable file renaming via `ConfigurableFileHandler`
  - Wavelength-to-channel mappings loaded from `config/wavelength_config.yaml` by default
  - Overridable per-run via `--wavelengths` CLI arg or `preprocessing.wavelength_mappings` in config
  - Plate number auto-detected from filepath or set explicitly via `--plate`

**Configuration:** See [`config/preprocessing_config.yaml`](../config/README.md#preprocessing-configuration).

---

## `run_inference.py`

Batch inference with organized output structure and visualization.

**Usage:**
```bash
# Run inference on test data
python scripts/run_inference.py \
    --input-dir data/processed/split/test \
    --output-dir results/inference \
    --model-name cyto3 \
    --flow-threshold 0.4 \
    --gpu

# Batch inference with config file
python scripts/run_inference.py \
    --config config/inference_config.yaml

# Custom model inference
python scripts/run_inference.py \
    --input-dir data/test \
    --output-dir results/inference \
    --model-name cyto3 \
    --diameter 30 \
    --flow-threshold 0.4
```

**Key Features:**
- Organized output structure
- Z-stack support
- Visualization overlays
- Batch processing
- GPU acceleration

**Configuration:** See [`config/inference_config.yaml`](../config/README.md#inference-configuration).

---

## `run_postprocessing.py`

3D cell tracking and blur-based quality filtering.

**Usage:**
```bash
# 3D cell tracking and blur filtering
python scripts/run_postprocessing.py \
    --input-dir results/inference \
    --output-dir results/tracked \
    --search-range 5 \
    --memory 1 \
    --blur-threshold 0.5

# Single file processing
python scripts/run_postprocessing.py \
    --single-file results/inference/sample_3d.tif \
    --output-dir results/tracked \
    --image-file data/test/sample_BF_3d.tif

# With configuration file
python scripts/run_postprocessing.py \
    --config config/postprocessing_config.yaml \
    --input-dir results/inference \
    --output-dir results/tracked
```

**Key Features:**
- 3D cell tracking across z-stacks
- Blur-based quality filtering
- Configurable processing order
- Comprehensive result reporting

**Configuration:** See [`config/postprocessing_config.yaml`](../config/README.md#postprocessing-configuration).

---

## `run_pipeline.py`

Complete end-to-end pipeline from raw data to analyzed results.

**Usage:**
```bash
python scripts/run_pipeline.py \
    --input-dir data/raw \
    --output-dir data/processed \
    --config config/config.yaml \
    --steps prepare segment-2d track
```

**Pipeline Steps:**
1. Data preprocessing (blur analysis, train/test split)
2. Model training (if requested)
3. Inference (segmentation prediction)
4. Postprocessing (tracking and filtering)

---

## `run_dataset_design_report.py`

Assembles the dataset-design assessment: per-culture pre-collapse windows, the seeding-density/time-to-collapse correlation, and the rank-based drug effect per well over two windows — plus the figures showing the raw mCherry values behind those tables.

**Usage:**
```bash
# Tables and figures (the default)
python scripts/run_dataset_design_report.py \
    --summary results/dataset_analysis/all_experiments_cell_population_summary.csv \
    --output-dir <report-dir>

# Tables only — skip the figures
python scripts/run_dataset_design_report.py --no-figures

# Draw all three percentile targets, not just the headline one
python scripts/run_dataset_design_report.py --all-targets

# Omit Step 2's relative gate from the cutoff table
python scripts/run_dataset_design_report.py --gate-threshold -1
```

**Arguments:**
- `--summary` — the 45-well cell-population summary CSV (default: `results/dataset_analysis/...`)
- `--output-dir` — where tables and figures are written
- `--gate-threshold` — relative gate recorded alongside the verdict cutoff; negative omits it
- `--figures` / `--no-figures` — draw the per-(culture, drug) time plots; on by default
- `--all-targets` — draw `percentile_75` and `percentile_95` as well as the headline target, into per-target subdirectories

**Outputs:**
- `cutoff_table.csv` — per-culture pre-collapse keep-window
- `density_vs_collapse.csv` / `.png` — peak density vs time-to-collapse, censored wells kept as their own marker
- `drug_effect_two_windows.csv` — Cliff's δ per well, both windows
- `drug_effect_summary.md` — the readable roll-up
- `figures/<headline-target>_dmso_vs_drug/<culture>_<drug>.{png,pdf}` — one figure per culture and drug: a line per well coloured by dose, the DMSO well as a bold black reference, and each well's `t_cross_peak` as a dashed line

**Notes:**
- Every run **regenerates all of the above**, figures or not. Point `--output-dir` at a scratch directory first if the target directory holds anything you are not prepared to have rewritten.
- Targets are read from `mcherry_metrics/<model>/instance_metrics.csv`, one CSV per experiment (68 MB–695 MB each); the per-slice feature CSVs are never loaded. Budget ~64 GB and a few minutes for all five cultures.
- `scripts/verify_dataset_design_numbers.py` re-checks the generated numbers against the claims made about them.

---

## `launch_gui_safe.py`

Safe GUI launcher with dependency checking and Qt backend configuration.

**Usage:**
```bash
# Launch GUI with dependency checking
python scripts/launch_gui_safe.py

# Launch with specific backend
python scripts/launch_gui_safe.py --backend pyqt5

# Check dependencies only
python scripts/launch_gui_safe.py --check-only

# Force backend configuration
python scripts/launch_gui_safe.py --backend pyqt5 --force-config
```

**Key Features:**
- Automatic Qt backend detection
- Dependency conflict resolution
- Safe GUI launching with error handling
- Cross-platform compatibility
