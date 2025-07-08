# Cell Segmentation Inference Pipeline

This module provides a comprehensive inference pipeline for running cell segmentation predictions using Cellpose models on test datasets. The pipeline is designed following machine learning best practices with organized output structure and extensive configuration options.

## Features

- **Modular Design**: Separate components for predictors, output management, and pipeline orchestration
- **Multiple Model Support**: Currently supports Cellpose with easy extensibility for other models
- **Organized Output**: Structured output directories following the pattern `{output_dir}/{model_name}/{dataset}`
- **Z-Stack Support**: Handle both 2D images and 3D Z-stacks
- **Comprehensive Logging**: Detailed logging and metadata tracking
- **Configuration-Driven**: YAML-based configuration for reproducible experiments
- **Visualization**: Automatic generation of segmentation overlays
- **Batch Processing**: Efficient processing of large datasets

## Quick Start

### 1. Basic Usage

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

# Using configuration file (recommended)
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

### 3. Configuration-Based Usage

```python
from src.inference import InferencePipeline

# Create pipeline from config
pipeline = InferencePipeline.from_config(
    config_path="config/inference_config.yaml",
    model_name="cyto3",
    output_dir="results",
    dataset_name="test"
)

# Run inference
results = pipeline.run_inference(input_dir="data/test")
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

### File Naming Convention

- **Masks**: `{original_name}_masks.tif`
- **Overlays**: `{original_name}_overlay.png`  
- **Metadata**: `{original_name}_metadata.json`
- **Z-stacks**: `{original_name}_stack.tif` (full stack), `{original_name}_z{idx:03d}_masks.tif` (individual slices)

## Architecture

### Core Components

1. **BasePredictor**: Abstract base class defining the predictor interface
2. **CellposePredictor**: Concrete implementation for Cellpose models
3. **OutputManager**: Handles file organization and saving
4. **InferencePipeline**: Orchestrates the entire inference process

### Class Hierarchy

```
BasePredictor (ABC)
├── CellposePredictor
└── [Future predictors: OmniposePredictor, CustomPredictor, etc.]

OutputManager
├── save_prediction()
├── save_z_stack_prediction()
└── finalize_run()

InferencePipeline
├── run_inference()
├── run_inference_single()
└── from_config() [class method]
```

## Configuration

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
```

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
predictor = CellposePredictor(model_type="cyto3")
masks, metadata = predictor.predict_z_stack(
    z_stack_array,
    process_2d=False  # True for slice-by-slice, False for 3D
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
```

## Model Parameters

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
- **nuclei**: Nuclear segmentation model

## Error Handling and Logging

The pipeline includes comprehensive error handling and logging:

```python
# Validation before running
validation = pipeline.validate_setup()
if not validation['overall']:
    print(f"Setup issues: {validation}")

# Detailed logging
import logging
logging.basicConfig(level=logging.INFO)

# Results include error information
results = pipeline.run_inference(input_dir="data/test")
if results['failed_files']:
    for failed in results['failed_files']:
        print(f"Failed: {failed['file']} - {failed['error']}")
```

## Performance Considerations

### GPU Usage

```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# Monitor GPU memory during inference
predictor = CellposePredictor(gpu=True)
# Process in batches for large datasets
```

### Memory Management

```python
# For large Z-stacks, process slice-by-slice
results = pipeline.run_inference(
    input_dir="data/large_stacks",
    process_z_stacks=True  # Processes each slice individually
)

# Or process in chunks
chunk_size = 10  # Process 10 files at a time
# Implementation would require custom batching logic
```

## Integration with Training Pipeline

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
```

## Troubleshooting

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
   ```

### Debugging Steps

1. **Test with a single file first**:
   ```python
   from src.inference import CellposePredictor
   
   predictor = CellposePredictor(model_type="cyto3", gpu=True)
   info = predictor.get_model_info()
   print(f"Model status: {info}")
   ```

2. **Check the logs**:
   ```bash
   # Logs are saved in the output directory
   cat outputs/test_inference/pred/cyto3/test/inference.log
   ```

3. **Validate setup**:
   ```python
   from src.inference import InferencePipeline
   
   pipeline = InferencePipeline.from_config("config/inference_config.yaml")
   validation = pipeline.validate_setup()
   print(f"Setup validation: {validation}")
   ```

## Examples

See `examples/inference_example.py` for comprehensive usage examples including:
- Basic inference setup
- Configuration-based inference  
- Single file processing
- Custom model usage
- Error handling patterns

## Future Extensions

The modular design allows for easy extension:

- **New Models**: Implement `BasePredictor` for other segmentation models
- **Custom Outputs**: Extend `OutputManager` for different file formats
- **Preprocessing**: Add preprocessing steps to the pipeline
- **Postprocessing**: Add cell tracking, feature extraction, etc.
- **Cloud Integration**: Add cloud storage backends
- **Distributed Processing**: Add support for cluster computing

## Performance Benchmarks

### Test Results (Real Data)

**Hardware**: NVIDIA GPU (CUDA 12.3), Windows 11  
**Dataset**: 20 brightfield images (1024x1024, 16-bit TIF)  
**Model**: Cellpose cyto3  

| Metric | Value |
|--------|-------|
| **Total Processing Time** | ~2-3 minutes |
| **Average per Image** | ~6-9 seconds |
| **Total Cells Detected** | 10,301 cells |
| **Average Cells per Image** | ~515 cells |
| **Success Rate** | 100% (0 failed files) |
| **Output Files Generated** | 80 files (20 masks + 20 metadata + 20 overlays + summaries) |

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
