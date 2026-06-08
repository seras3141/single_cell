# Core Modules

This document describes the core modules of the single-cell analysis pipeline.

---

## Preprocessing (`src/preprocessing/`)

Data preprocessing pipeline with blur analysis and group-aware dataset splitting.

**Key features:**
- Blur heatmap generation for quality assessment
- Group-aware train/test splitting
- 2D to 3D stack combination
- Standardised file naming and organisation

**CLI entry point:** `scripts/run_preprocessing.py`

---

## Inference (`src/inference/`)

Cellpose-based inference pipeline for running cell segmentation predictions.

**Key features:**
- Modular predictor architecture
- Structured output management via `OutputManager`
- Z-stack and batch processing support
- Visualisation and logging

**CLI entry point:** `scripts/run_inference.py`

---

## Postprocessing (`src/postprocessing/`)

3D cell tracking and quality filtering pipeline.

**Key features:**
- 3D cell tracking across z-stacks using trackpy
- Blur-based quality filtering (supports inverted thresholds)
- Configurable processing workflows
- Comprehensive result reporting

**CLI entry point:** `scripts/run_postprocessing.py`

---

## Feature Extraction (`src/feature_extraction/`)

Comprehensive feature extraction from 2D instance segmentations.

**Key features:**
- 23 morphological, intensity, spatial, and texture features per cell
- Multi-backend support
- Parallel processing with configurable batch sizes

**Extracted features:**
| Category | Features |
|---|---|
| Morphology | Area, perimeter, elongation, compactness, circularity, Feret diameter |
| Intensity | Mean, std, CV, total intensity |
| Spatial | Centroids, centre of mass, mass displacement |
| Texture | Gabor filters, skewness, kurtosis, entropy |

**CLI entry point:** `scripts/run_feature_extraction.py`

---

## Feature Visualisation (`src/feature_visualization/`)

Scatter plots, distribution plots, and dimensionality reduction visualisation.

**Architecture note:** Dimensionality reduction (UMAP/PCA/t-SNE) must happen *before* data reaches a `Plotter`. Plotters receive pre-reduced data and render only.

---

## Utils (`src/utils/`)

Core utilities shared across modules.

| Module | Purpose |
|---|---|
| `file_utils.py` | Unified file handling and multi-well plate naming |
| `blur_measure.py` | Image blur / sharpness detection |
| `conversion.py` | 2D/3D format conversion |
| `config.py` | OmegaConf configuration loading and validation |
| `logging_utils.py` | Project-wide logger |

---

## Visualisation (`src/visualize/`)

3D/4D visualisation tools using Napari.

**Key features:**
- 3D/4D TIFF stack visualisation
- Prediction overlay visualisation
- Interactive GUI with dependency checking

**CLI entry point:** `scripts/launch_gui_safe.py`

---

## Cell Activity Labeler (`src/cell_activity_labeler/`)

Sub-package for activity labelling. Can be installed independently — treat it as a dependency rather than modifying its internals when working on the main pipeline.

---

## mCherry Metrics (`src/mcherry_metrics/`)

Sub-package for extracting mCherry channel metrics from segmented cells.
