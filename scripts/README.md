# Scripts Directory

The `scripts/` directory contains command-line interfaces for all major pipeline operations. These scripts provide easy-to-use interfaces for processing data and running analysis workflows.

## Available Scripts

See [SCRIPTS_REFERENCE.md](SCRIPTS_REFERENCE.md) for full usage, arguments, and key features for each script.

| Script | Description |
|---|---|
| `run_preprocessing.py` | Data preprocessing: train/test split, blur heatmaps, 2D→3D stacking |
| `run_inference.py` | Batch cell segmentation inference |
| `run_postprocessing.py` | 3D cell tracking and quality filtering |
| `run_pipeline.py` | End-to-end pipeline runner |
| `launch_gui_safe.py` | Safe napari GUI launcher |

## Common Usage Patterns

### Complete Workflow

```bash
# 1. Preprocess data
python scripts/run_preprocessing.py \
    -i data/raw -o data/processed

# 2. Train model (optional)
# python scripts/run_training.py \
#     --train-dir data/processed/split/train \
#     --test-dir data/processed/split/test \
#     --output-dir models/trained

# 3. Run inference
python scripts/run_inference.py \
    --input-dir data/processed/split/test \
    --output-dir results/inference \
    --model-path models/trained/cellpose_model.pth

# 4. Postprocess results
python scripts/run_postprocessing.py \
    --input-dir results/inference \
    --output-dir results/tracked

# 5. Visualize results
python scripts/launch_gui_safe.py
```

### Quick Analysis

```bash
# Quick analysis with pretrained model
python scripts/run_inference.py \
    --input-dir data/test \
    --output-dir results/quick \
    --model-name cyto3

python scripts/run_postprocessing.py \
    --input-dir results/quick \
    --output-dir results/tracked
```

### Configuration-Based Processing

```bash
# Use configuration files for reproducible processing
python scripts/run_preprocessing.py \
    --config config/preprocessing_config.yaml

python scripts/run_training.py \
    --config config/training_config.yaml

python scripts/run_inference.py \
    --config config/inference_config.yaml

python scripts/run_postprocessing.py \
    --config config/postprocessing_config.yaml
```

## Configuration Files

See [config/README.md](../config/README.md) for all configuration file schemas and options.

## Python API

See [API_USAGE.md](API_USAGE.md) for examples of calling scripts and pipeline functions programmatically.

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and solutions.

---

*For detailed usage of individual scripts, run them with the `--help` flag.*
