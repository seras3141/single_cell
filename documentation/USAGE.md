# Usage Guide

This document covers CLI and Python API usage for the single-cell analysis pipeline.

---

## Command Line Interface

### Full pipeline

```bash
python scripts/run_pipeline.py \
    --input-dir data/raw \
    --output-dir results \
    --config config/config.yaml \
    --preprocess \
    --inference \
    --postprocess
```

### Individual stages

#### 1. Preprocessing

```bash
python scripts/run_preprocessing.py data/raw data/processed \
    --test-size 0.2 \
    --random-seed 42 \
    --patch-size 32 \
    --stride-size 16
```

| Option | Default | Description |
|---|---|---|
| `--test-size` | 0.2 | Fraction of data held out for test set |
| `--random-seed` | 42 | Seed for reproducibility |
| `--patch-size` | 32 | Patch size for blur detection |
| `--stride-size` | 16 | Stride size for blur detection |
| `--overwrite` | false | Overwrite existing output files |

#### 2. Inference

```bash
# From command-line arguments
python scripts/run_inference.py \
    --input-dir data/processed/split/test \
    --output-dir results/inference \
    --model-name cyto3 \
    --flow-threshold 0.4 \
    --gpu

# From a config file
python scripts/run_inference.py --config config/inference_config.yaml
```

#### 3. Postprocessing

```bash
# Directory-level processing
python scripts/run_postprocessing.py \
    --input-dir results/inference \
    --output-dir results/tracked \
    --search-range 5 \
    --memory 1 \
    --blur-threshold 0.5

# Single file
python scripts/run_postprocessing.py \
    --single-file results/inference/sample_3d.tif \
    --output-dir results/tracked \
    --image-file data/test/sample_BF_3d.tif
```

#### 4. Feature extraction

```bash
# From a config file
python scripts/run_feature_extraction.py \
    --config config/feature_extraction_config.yaml

# From command-line arguments
python scripts/run_feature_extraction.py \
    --input-dir data/segmented \
    --output-dir results/features \
    --image-pattern "*_BF.tif" \
    --mask-pattern "Cells_*.tif" \
    --n-jobs 8
```

#### 5. Visualisation GUI

```bash
python scripts/launch_gui_safe.py
```

---

## Python API

### Inference pipeline

```python
from src.inference.inference_pipeline import InferencePipeline
from src.inference.cellpose_predictor import CellposePredictor
from src.inference.output_manager import OutputManager

predictor = CellposePredictor(model_type="cyto3", gpu=True)
output_manager = OutputManager(base_output_dir="results")
pipeline = InferencePipeline(predictor, output_manager)

results = pipeline.run_inference(input_dir="data/test")
```

### Configuration

```python
from src.utils.config import get_config

config = get_config()
val = config.get("segmentation.cellpose.model_type")  # dot-notation, type-safe
```

CLI overrides use the same dot notation:

```bash
python scripts/run_inference.py \
    segmentation.cellpose.model_type=cyto2 \
    segmentation.cellpose.flow_threshold=0.5
```

---

## Notebooks

Interactive examples are in `notebooks/`:

| Notebook | Description |
|---|---|
| `00_data_filter.ipynb` | Data filtering |
| `00_data_visualization.ipynb` | Data visualisation |
| `01_data_preparation.ipynb` | Data preparation |
| `01_data_preprocessing.ipynb` | Preprocessing walkthrough |
| `02_cellpose_evaluation_example.ipynb` | Cellpose model evaluation with metrics and plots |
| `02a_cellpose_inference_example.ipynb` | Cellpose inference walkthrough |
| `03_custom_configuration_example.ipynb` | Custom configuration |
| `04_prediction_postprocessing.ipynb` | 3D cell tracking with blur filtering |
| `05_eval_segmentation.ipynb` | Segmentation evaluation |
| `feature_visualization_tutorial.ipynb` | Feature visualisation, dimensionality reduction, heatmaps |
