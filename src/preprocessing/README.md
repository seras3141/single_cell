# Preprocessing Module

The `src/preprocessing` module provides utilities for:

- Organizing datasets and splitting into train/test sets
- Standardizing file naming and grouping
- Combining 2D images into 3D stacks (via utility modules)
- Measuring and filtering image blur, including patchwise and dataset-wide blur heatmaps

---

## Detailed Preprocessing Pipeline

Before training models or running analysis, raw datasets need to be preprocessed. This involves two key steps:

Before training or analysis, preprocess your raw dataset:

```python
# Generate blur heatmaps for quality assessment
from src.preprocessing.blur_analysis import measure_dataset_blur_heatmaps

blur_results = measure_dataset_blur_heatmaps(
    input_dir="data/raw/images",
    output_dir="data/processed/blur_heatmaps"
)

# Create train/test split with group awareness
from src.preprocessing.dataset_split import split_dataset
from src.utils.file_utils import BF_IF_FileHandler

file_patterns = {
    'images': 'data/raw/*/t1_*_w1_*.tif',
    'masks': 'data/raw/*/Cells_*.tif'
}

split_results = split_dataset(
    file_patterns=file_patterns,
    output_dir="data/processed/split",
    test_fraction=0.2,
    file_handler=BF_IF_FileHandler()
)
```

### Step 1: Generate Blur Heatmaps

Generate blur quality heatmaps for all images to enable quality assessment and filtering:

```python
from src.preprocessing.blur_analysis import measure_dataset_blur_heatmaps

# Generate blur heatmaps for the entire dataset
results = measure_dataset_blur_heatmaps(
    input_dir="data/raw/images",
    output_dir="data/processed/blur_heatmaps",
    pattern="*.tif",
    patch_size=32,           # Size of patches for blur detection
    stride_size=16,          # Stride between patches
    normalize=True,          # Normalize values to [0,1] range
    center_values=True,      # Center values on patches
    overwrite=False          # Skip existing files
)

print(f"Generated blur heatmaps for {len(results)} images")
```

**Command Line Usage:**
```bash
python -m src.preprocessing.blur_analysis \
    --input data/raw/images \
    --output data/processed/blur_heatmaps \
    --patch-size 32 \
    --stride-size 16 \
    --normalize
```

### Step 2: Split Dataset into Train/Test

Create group-aware train/test splits ensuring all images from the same experimental group stay together:

```python
from src.preprocessing.dataset_split import split_dataset
from src.utils.file_utils import BF_IF_FileHandler

# Define file patterns for images and masks
file_patterns = {
    'images': 'data/raw/*/t1_*_w1_*.tif',
    'masks': 'data/raw/*/Cells_*.tif'
}

# Split dataset with group awareness
results = split_dataset(
    file_patterns=file_patterns,
    output_dir="data/processed/split",
    test_fraction=0.2,        # 20% for testing
    random_seed=42,           # For reproducibility
    file_handler=BF_IF_FileHandler(),  # Handles standardized naming
)

print(f"Train set: {len(results['train']['images'])} images")
print(f"Test set: {len(results['test']['images'])} images")
```

**Alternative Directory-based Split:**
```python
from src.preprocessing.dataset_split import train_test_split_directory

# Split an existing directory structure
train_test_split_directory(
    input_dir="data/raw",
    output_dir="data/processed/split",
    test_fraction=0.2,
    file_handler=BF_IF_FileHandler(),
    random_seed=42
)
```

### Complete Preprocessing Workflow

```python
from pathlib import Path
from src.preprocessing.blur_analysis import measure_dataset_blur_heatmaps  
from src.preprocessing.dataset_split import split_dataset
from src.utils.file_utils import BF_IF_FileHandler

# Setup paths
raw_data_dir = Path("data/raw")
processed_dir = Path("data/processed")

# Step 1: Generate blur heatmaps
print("Step 1: Generating blur heatmaps...")
blur_results = measure_dataset_blur_heatmaps(
    input_dir=raw_data_dir / "images",
    output_dir=processed_dir / "blur_heatmaps",
    pattern="*.tif",
    normalize=True,
    overwrite=False
)

# Step 2: Split dataset
print("Step 2: Creating train/test split...")
file_patterns = {
    'images': str(raw_data_dir / '*' / 't1_*_w1_*.tif'),
    'masks': str(raw_data_dir / '*' / 'Cells_*.tif')
}

split_results = split_dataset(
    file_patterns=file_patterns,
    output_dir=processed_dir / "split",
    test_fraction=0.2,
    random_seed=42,
    file_handler=BF_IF_FileHandler(),
)

print(f"✅ Preprocessing complete!")
print(f"   📊 Blur heatmaps: {len(blur_results)} images")
print(f"   🎯 Train set: {len(split_results['train']['images'])} images") 
print(f"   🧪 Test set: {len(split_results['test']['images'])} images")
```

### Output Structure

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
    └── split/                       # Train/test split
        ├── train/
        └── test/
```

### Preprocessing Benefits

- **🎯 Quality Control**: Blur heatmaps identify low-quality regions for filtering
- **🔄 Group Awareness**: Train/test split maintains experimental groups
- **📁 Organization**: Standardized structure for downstream analysis
- **🔧 Reproducibility**: Configurable parameters with seeded random splits
- **💾 Efficiency**: Skip existing files to avoid reprocessing

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

- **CLI Command:**
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

## 2. Blur Measurement & Filtering

Measure blur at patch, image, or dataset level. Filter out blurry images before training.

**Patchwise Blur:**
```python
from src.preprocessing.blur_measure import measure_patchwise_blur

blur_map = measure_patchwise_blur(image, patch_size=32, stride_size=16)
```

**Dataset Blur Heatmaps:**
```python
from src.preprocessing.blur_analysis import measure_dataset_blur_heatmaps

measure_dataset_blur_heatmaps(
    input_dir="data/processed/split/train/3d_images",
    output_dir="data/processed/blur_heatmaps",
    pattern="*.tif",
    patch_size=32,
    stride_size=16,
    normalize=True,
    overwrite=False
)
```

**CLI Command:**
```sh
python -m src.preprocessing.blur_analysis \
  --input data/processed/split/train/3d_images \
  --output data/processed/blur_heatmaps \
  --pattern "*.tif" \
  --patch-size 32 \
  --stride-size 16 \
  --overwrite
```

---

## 3. Combining 2D to 3D

While not directly in `src/preprocessing`, use utility functions (e.g., `src/utils/conversion.py`) to combine 2D slices into 3D TIFFs.

**CLI Command:**
```sh
python -m src.utils.conversion combine \
  --input data/processed/split/train/images \
  --output data/processed/split/train/3d_images \
  --pattern "(.*)_z(\\d+)(?:_(BF|Cells))?\\.(tif|tiff)"
```

---

## 4. Example Workflow

```python
from src.preprocessing import train_test_split_directory, measure_dataset_blur_heatmaps
from src.utils.conversion import combine_2d_to_3d

# Split dataset
train_test_split_directory(...)

# Combine 2D to 3D
combine_2d_to_3d(...)

# Generate blur heatmaps
measure_dataset_blur_heatmaps(...)
```

---

## Available Functions

- `train_test_split_directory`, `get_groups_from_filenames`
- `measure_patchwise_blur`, `measure_image_blur`, `measure_blur_heatmap`
- `analyze_dataset_blur`, `filter_blurry_images`

---

Adjust parameters and file patterns as needed for your dataset.
