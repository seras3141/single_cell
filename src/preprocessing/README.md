# Preprocessing Module

The `src/preprocessing` module provides utilities for:

- Organizing datasets and splitting into train/test sets
- Standardizing file naming and grouping samples
- Combining 2D images into 3D stacks (via utility modules)
- Measuring image blur, including patchwise and dataset-wide blur heatmaps

---

## Quickstart

Run the entire preprocessing pipeline in one command:

```sh
python scripts/run_preprocessing.py data/raw data/processed \
  --test-size 0.2 \
  --random-seed 42 \
  --image-pattern "t1_*_w1_*.tif" \
  --mask-pattern "Cells_*.tif"
```

This will:
1. Split your dataset into train/test sets
2. Combine 2D images into 3D stacks
3. Generate blur heatmaps for quality assessment

For more control over individual steps, see the detailed pipeline below.

## Demo

Refer the [data_preparation](notebooks/01_data_preparation.ipynb) for data splitting and [data_preprocessing](notebooks/01_data_preprocessing.ipynb) for blur map generation.

## Detailed Preprocessing Pipeline

Before training models or running analysis, raw datasets need to be preprocessed. This involves three key steps:
1. Dataset Splitting & Organization
2. Combining 2D z-stacks to create 3D data
3. Create Blur Maps from 3D data

## 1. Dataset Splitting & Organization

Split images and masks into train/test sets, keeping groups (e.g., wells, positions) together.

**API Example:**
```python
from src.preprocessing.dataset_split import train_test_split_directory, get_groups_from_filenames
from src.utils.file_utils import BF_IF_FileHandler

train_test_split_directory(
    data_dir="data/raw",
    output_dir="data/processed/split",
    test_size=0.2,
    random_state=42,
    image_pattern="t1_*_w1_*.tif",
    mask_pattern="Cells_*.tif",
    file_handler=BF_IF_FileHandler(),
)
```
- Input: `data/raw/t1_*_w1_*.tif`, `data/raw/Cells_*.tif`
- Output: `data/processed/split/train/images/`, `data/processed/split/test/images/`, etc.

**CLI Command:**
```sh
python -m src.preprocessing.dataset_split \
  data/raw \
  data/processed/split \
  --test-size 0.2 \
  --random-seed 42 \
  --image-pattern "t1_*_w1_*.tif" \
  --mask-pattern "Cells_*.tif"
```

---

## 2. Combining 2D to 3D

While not directly in `src/preprocessing`, use utility functions (e.g., `src/utils/conversion.py`) to combine 2D slices into 3D TIFFs (to create blur maps).

**API Example:**
```python
from src.utils.conversion import combine_2d_to_3d

combine_2d_to_3d(
    input_dir="data/processed/split",
    output_dir="data/processed/3d_images",
    pattern="(.*)_z(\\d+)(?:_(BF|Cells))?\\.(tif|tiff)",
    recursive=True,
)
```

**CLI Command:**
```sh
python -m src.utils.conversion combine \
  --input data/processed/split/ \
  --output data/processed/3d_images \
  --pattern "(.*)_z(\\d+)(?:_(BF|Cells))?\\.(tif|tiff)"
```

---

## 3. Blur Measurement & Filtering

Measure patchwise-blur map for each image, or at dataset level. 
<!-- Filter out blurry images before training. -->

**Patchwise Blur:**
```python
from src.utils.blur_measure import measure_patchwise_blur

blur_map = measure_patchwise_blur(image, patch_size=32, stride_size=16)
```

**Dataset Blur Heatmaps:**
```python
from src.preprocessing.blur_analysis import generate_blur_heatmap_batch

generate_blur_heatmap_batch(
    input_dir="data/processed/3d_images",
    output_dir="data/processed/blur_heatmaps",
    pattern="*_BF_3d.tif",
    patch_size=32,
    stride_size=16,
    normalize=True,
    overwrite=False,
)
```

**CLI Command:**
```sh
python -m src.preprocessing.blur_analysis \
  --input data/processed/3d_images \
  --output data/processed/blur_heatmaps \
  --pattern "*_BF_3d.tif" \
  --patch-size 32 \
  --stride-size 16 \
  --overwrite
```

## 4. Example Workflow

```python
from src.preprocessing import train_test_split_directory
from src.preprocessing.blur_analysis import generate_blur_heatmap_batch
from src.utils.conversion import combine_2d_to_3d

# Split dataset
train_test_split_directory(...)

# Combine 2D to 3D
combine_2d_to_3d(...)

# Generate blur heatmaps
generate_blur_heatmap_batch(...)
```
## Config file

Run the entire preprocessing pipeline using the config file:

```sh
python scripts/run_preprocessing.py data/raw data/processed \
  --config config/preprocessing_config.yaml
```


## Output Structure

After preprocessing, your data will be organized as:

```
data/
├── raw/                              # Original data
│   ├── plate1/
│   ├── plate2/
│   └── ...
└── processed/
    ├── blur_heatmaps/               # Quality assessment maps
    │   ├── image1_blur_heatmap.tif
    │   ├── image2_blur_heatmap.tif
    │   └── ...
    └── 3d_images/                       # 3D z-stacks
    │   ├── image1_BF_3d.tif
    │   └── image1_Cells_3d.tif
    │   └── ...
    └── split/                       # Train/test split
        ├── train/
        └── test/
```

---

Adjust parameters and file patterns as needed for your dataset.
