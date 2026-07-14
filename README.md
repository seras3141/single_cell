# Single Cell Analysis Pipeline

A modular pipeline for single-cell segmentation, tracking, and feature extraction from brightfield microscopy images, built around [Cellpose](https://github.com/MouseLand/cellpose).

---

## Prerequisites

- Python 3.10+
- Conda (Anaconda or Miniconda)
- CUDA-capable GPU (optional, for faster inference)

---

## Installation

```bash
# 1. Clone with submodule
git clone <repository-url>
cd single_cell
git submodule update --init --recursive

# 2. Create and activate a conda environment
conda create -n cellpose-env python=3.10
conda activate cellpose-env

# 3. Install PyTorch (adjust cuda version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# 4. Install dependencies
pip install -e ./github/cellpose
pip install -e .
```

---

## Quick Start

```bash
# Run the full pipeline
python scripts/run_pipeline.py \
    --input-dir data/raw \
    --output-dir results \
    --config config/config.yaml \
    --preprocess --inference --postprocess

# Or run individual stages
python scripts/run_preprocessing.py data/raw data/processed
python scripts/run_inference.py --input-dir data/processed/split/test --output-dir results/inference
python scripts/run_postprocessing.py --input-dir results/inference --output-dir results/tracked
python scripts/run_feature_extraction.py --config config/feature_extraction_config.yaml

# Launch the visualisation GUI
python scripts/launch_gui_safe.py
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

> Run tests from the repository root — imports break if you run from inside `src/`.

---

## Documentation

- [Modules](documentation/MODULES.md) — description of each source module
- [Usage](documentation/USAGE.md) — full CLI reference and Python API examples
- [Configuration](documentation/CONFIGURATION.md) — configuration file reference and supported formats
- [Dataset](documentation/DATASET.md) — experimental design, file naming convention, plate layout, and data format
