# Cell Segmentation Inference Pipeline

This module provides a comprehensive inference pipeline for running cell segmentation predictions using Cellpose models on test datasets.

<!-- TODO Add sample image (Input, GT, Pred) -->

## Quick Start

Run the entire inference pipeline in one command:

```bash
# Basic inference using command line arguments
python scripts/run_inference.py \
    --input-dir data/sample_plates_split/test \
    --output-dir outputs/inference_results \
    --model-name cyto3 \
    --flow-threshold 0.3 \
    --min-size 25 \
    --diameter 30

# Using dedicated inference config (for inference-only pipelines)
python scripts/run_inference.py \
    --config config/inference_config.yaml \
    --input-dir data/sample_plates_split/split \
    --output-dir outputs/inference_results
```

For more control over individual steps, see the detailed pipeline below.

## Detailed Inference Pipeline

### 1. API Usage

**Configuration-Based Usage** (Recommended)

```python
from src.inference import InferencePipeline

# Create pipeline from main config with overrides
pipeline = InferencePipeline.from_config(
    config_path="config/config.yaml",
    model_name="cyto3",
    output_dir="results",
    dataset_name="test"
)

# Run inference
results = pipeline.run_inference(input_dir="data/test")
```

**Simple Pipeline:**

```python
from src.inference import CellposePredictor, OutputManager, InferencePipeline

# Initialize predictor
predictor = CellposePredictor(
    model_type="cyto3",
    gpu=True,
    flow_threshold=0.4
)

# Set up output management
output_manager = OutputManager(
    base_output_dir="results",
    model_name="cyto3", 
    dataset_name="test"
)

# Create and run pipeline
pipeline = InferencePipeline(predictor, output_manager)
results = pipeline.run_inference(
    input_dir="data/test",
    file_pattern="*_BF.tif"
)
```

### 2. Command Line Usage

```bash
# Basic inference
python scripts/run_inference.py \
    --input-dir data/sample_plates_split/test \
    --output-dir outputs/inference_results \
    --model-name cyto3

# With custom parameters
python scripts/run_inference.py \
    --input-dir data/sample_plates_split/test \
    --output-dir outputs/inference_results \
    --model-name cyto3 \
    --flow-threshold 0.3 \
    --min-size 25 \
    --diameter 30

# Using inference-specific configuration file
python scripts/run_inference.py \
    --config config/inference_config.yaml \
    --input-dir data/sample_plates_split/test \
    --output-dir outputs/inference_results

# Skip overlays for faster processing
python scripts/run_inference.py \
    --input-dir data/sample_plates_split/test \
    --output-dir outputs/inference_results \
    --config config/inference_config.yaml \
    --no-overlays

# Skip metadata for minimal output
python scripts/run_inference.py \
    --input-dir data/sample_plates_split/test \
    --output-dir outputs/inference_results \
    --config config/inference_config.yaml \
    --no-metadata --no-overlays
```

## Output Structure

The pipeline creates organized output directories:

```
results/
└── pred/
    └── {model_name}/
        └── {dataset_name}/
            ├── masks/           # Segmentation masks (.tif)
            ├── overlays/        # Visualization overlays (.png)
            ├── metadata/        # Prediction metadata (.json)
            ├── run_summary.json # Overall run statistics
            └── inference.log    # Detailed log file
```


<!-- ## Configuration

### Inference Config Example

```yaml
# config/inference_config.yaml
segmentation:
  cellpose:
    model_type: "cyto3"
    gpu: true
    flow_threshold: 0.4
    cellprob_threshold: 0.0
    min_size: 30

inference:
  file_patterns:
    - "*_BF.tif"
  output:
    save_overlays: true
    save_metadata: true
  processing:
    process_z_stacks: false

paths:
  test_data: "data/test"
  output_root: "results"
``` -->

<!-- TODO : Update this part with different models, and training integration
## Advanced Usage

### 1. Custom Model Loading

```python
# Load custom trained model
predictor = CellposePredictor(model_type="cyto3")
predictor.load_model("path/to/custom/model")

# Or initialize with custom model
predictor = CellposePredictor(
    model_type="cyto3",
    model_path="path/to/custom/model"
)
```

### 2. Z-Stack Processing

```python
# Process as individual 2D slices
results = pipeline.run_inference(
    input_dir="data/z_stacks",
    process_z_stacks=True,
    save_individual_slices=True
)

# Process entire volume as 3D
# TODO : Update this
predictor = CellposePredictor(model_type="cyto3")
masks, metadata = predictor.predict_3d(
    z_stack_array,
    do_2d=False  # True for slice-by-slice, False for 3D
)
```

### 3. Batch Processing with Progress Tracking

```python
def progress_callback(current, total, file_path):
    print(f"Processing {current}/{total}: {file_path.name}")

results = pipeline.run_inference(
    input_dir="data/large_dataset",
    progress_callback=progress_callback
)
```

### 4. Custom Output Organization

```python
output_manager = OutputManager(
    base_output_dir="custom_results",
    model_name="my_model",
    dataset_name="experiment_1",
    create_subdirs=False  # Flat structure
)
``` -->

<!-- ## Model Parameters

### Cellpose Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_type` | "cyto3" | Pretrained model type |
| `gpu` | True | Use GPU acceleration |
| `channels` | [0, 0] | [cytoplasm, nucleus] channels |
| `diameter` | None | Expected cell diameter (auto if None) |
| `flow_threshold` | 0.4 | Flow error threshold (0.0-1.0) |
| `cellprob_threshold` | 0.0 | Cell probability threshold |
| `min_size` | 30 | Minimum cell size in pixels |
| `normalize` | True | Normalize input images |
| `invert` | False | Invert image intensities |

### Model Types

- **cyto3**: Latest cytoplasm model (recommended for brightfield)
- **cyto2**: Cytoplasm model for fluorescence
- **cyto**: Original cytoplasm model
- **nuclei**: Nuclear segmentation model -->



<!-- ## Integration with Training Pipeline

The inference pipeline is designed to work seamlessly with trained models:

```python
# After training
trained_model_path = "models/my_trained_model"

# Use for inference
predictor = CellposePredictor(model_type="cyto3")
predictor.load_model(trained_model_path)

# Run inference with trained model
pipeline = InferencePipeline(predictor, output_manager)
results = pipeline.run_inference(input_dir="data/test")
``` -->

<!-- ## Troubleshooting

### Common Issues and Solutions

1. **"Object of type ndarray is not JSON serializable"**
   ```
   ERROR: Failed to save metadata: Object of type ndarray is not JSON serializable
   ```
   **Solution**: This has been fixed in the current version. Ensure you're using the latest code.

2. **"AttributeError: module 'tifffile' has no attribute 'imsave'"**
   ```
   ERROR: AttributeError: module 'tifffile' has no attribute 'imsave'
   ```
   **Solution**: This has been fixed. The code now uses `tifffile.imwrite()`.

3. **OpenMP Error on Windows**:
   ```bash
   # Set environment variable before running
   set KMP_DUPLICATE_LIB_OK=TRUE
   python scripts/run_inference.py ...
   ```

4. **GPU Memory Issues**: 
   ```python
   # Use CPU instead
   python scripts/run_inference.py --no-gpu ...
   
   # Or reduce batch processing (process files one by one)
   # This is already the default behavior
   ```

5. **No files found**:
   ```python
   # Debug file finding
   from pathlib import Path
   files = list(Path("data/test").glob("*_BF.tif"))
   print(f"Found files: {files}")
   
   # Check your file pattern
   python scripts/run_inference.py \
       --input-dir data/test \
       --file-pattern "*.tif"  # Adjust pattern as needed
   ```

6. **Model Loading Issues**:
   ```python
   # Check if Cellpose is properly installed
   import cellpose
   print(f"Cellpose version: {cellpose.__version__}")
   
   # Verify CUDA availability
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   ```

7. **Permission/Path Issues on Windows**:
   ```bash
   # Use absolute paths
   python scripts/run_inference.py \
       --input-dir "E:\sera\Helmholtz\single_cell\data\test" \
       --output-dir "E:\sera\Helmholtz\single_cell\outputs"
   ``` -->


## Examples

See `notebooks/02_cellpose_inference.ipynb` for comprehensive usage examples including:
- Basic inference setup
- Configuration-based inference with OmegaConf
- CLI override patterns
- Single file processing
- Custom model usage
- Error handling patterns


## Configuration

You can:

1. **Use the main config**: [`config/config.yaml`](config/config.yaml ) (default)
2. **Use inference-specific config**: [`config/inference_config.yaml`](config/inference_config.yaml ) (for inference-only)
3. **Override parameters via CLI**: Use `--override key.subkey=value` syntax

### CLI Override Examples

```bash
# Override model type
python scripts/run_inference.py \
    --input-dir data/test \
    --override segmentation.cellpose.model_type=cyto2

# Override multiple parameters
python scripts/run_inference.py \
    --input-dir data/test \
    --override segmentation.cellpose.flow_threshold=0.3 \
    --override segmentation.cellpose.min_size=25

# Multiple overrides in one command
python scripts/run_inference.py \
    --input-dir data/test \
    --override segmentation.cellpose.model_type=cyto2 \
    --override segmentation.cellpose.gpu=false \
    --override segmentation.cellpose.diameter=25
```

## Performance Benchmarks

<!-- ### Test Results -->

<!-- **Hardware**: NVIDIA GPU (CUDA 12.3), Windows 11  
**Dataset**: 20 brightfield images (1024x1024, 16-bit TIF)  
**Model**: Cellpose cyto3  

| Metric | Value |
|--------|-------|
| **Total Processing Time** | ~2-3 minutes |
| **Average per Image** | ~6-9 seconds |
| **Total Cells Detected** | 10,301 cells |
| **Average Cells per Image** | ~515 cells | -->

### Performance Modes

```bash
# Fastest: masks only
python scripts/run_inference.py \
    --input-dir data/test \
    --output-dir outputs/fast \
    --no-metadata --no-overlays
# ~2-3 seconds per image

# Balanced: masks + metadata  
python scripts/run_inference.py \
    --input-dir data/test \
    --output-dir outputs/balanced \
    --no-overlays
# ~3-4 seconds per image

# Complete: all outputs (default)
python scripts/run_inference.py \
    --input-dir data/test \
    --output-dir outputs/complete
# ~6-9 seconds per image
```
