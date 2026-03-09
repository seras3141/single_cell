# Scripts Directory

The `scripts/` directory contains command-line interfaces for all major pipeline operations. These scripts provide easy-to-use interfaces for processing data and running analysis workflows.

## Available Scripts

### `run_preprocessing.py`
Complete data preprocessing pipeline including blur analysis and train/test splitting.

**Usage:**
```bash
# Full preprocessing pipeline
python scripts/run_preprocessing.py data/raw data/processed \
    --test-size 0.2 \
    --random-seed 42 \
    --patch-size 32 \
    --stride-size 16

# With advanced options
python scripts/run_preprocessing.py data/raw data/processed \
    --test-size 0.2 \
    --random-seed 42 \
    --image-pattern "t1_*_w1_*.tif" \
    --mask-pattern "Cells_*.tif" \
    --patch-size 32 \
    --stride-size 16 \
    --overwrite
```

**Key Features:**
- Group-aware train/test splitting
- Blur heatmap generation
- 2D to 3D stack combination
- Standardized file naming


### `run_inference.py`
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

### `run_postprocessing.py`
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

### `run_pipeline.py`
Complete end-to-end pipeline from raw data to analyzed results.

**Usage:**
```bash
# Run entire pipeline
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

### `launch_gui_safe.py`
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

## Common Usage Patterns

### Complete Workflow

```bash
# 1. Preprocess data
python scripts/run_preprocessing.py data/raw data/processed

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

### Preprocessing Configuration

```yaml
# config/preprocessing_config.yaml
preprocessing:
  test_size: 0.2
  random_seed: 42
  image_pattern: "t1_*_w1_*.tif"
  mask_pattern: "Cells_*.tif"
  patch_size: 32
  stride_size: 16
  overwrite: false
```

### Inference Configuration

```yaml
# config/inference_config.yaml
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

### Postprocessing Configuration

```yaml
# config/postprocessing_config.yaml
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

## Best Practices

### Command Line Usage

1. **Use Configuration Files**: For reproducible experiments
2. **Validate Paths**: Ensure input and output paths exist
3. **Check Dependencies**: Use `--check-only` flags when available
4. **Monitor Progress**: Use verbose flags for detailed output
5. **Handle Errors**: Check return codes and log files

### Performance Optimization

1. **Use GPU**: Enable GPU acceleration when available
2. **Batch Processing**: Process multiple files together
3. **Memory Management**: Monitor memory usage for large datasets
4. **Parallel Processing**: Use multi-core processing when supported

### Error Handling

1. **Check Logs**: Review log files for detailed error information
2. **Validate Inputs**: Ensure input data is in correct format
3. **Test with Small Datasets**: Test workflows with small data first
4. **Use Dry Run**: Use dry run options when available

## Integration with Python API

Scripts can be integrated with Python code:

```python
import subprocess
import sys

def run_preprocessing(input_dir, output_dir, **kwargs):
    """Run preprocessing script from Python"""
    cmd = [
        sys.executable, "scripts/run_preprocessing.py",
        input_dir, output_dir
    ]
    
    for key, value in kwargs.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr

# Usage
success, stdout, stderr = run_preprocessing(
    "data/raw", "data/processed",
    test_size=0.2,
    random_seed=42
)
```

## Troubleshooting

### Common Issues

1. **Path Issues**: Use absolute paths or ensure working directory is correct
2. **Permission Issues**: Ensure write permissions for output directories
3. **Memory Issues**: Reduce batch sizes or use smaller datasets
4. **GPU Issues**: Check CUDA availability and driver versions
5. **Dependency Issues**: Use `launch_gui_safe.py` for GUI-related problems

### Performance Issues

1. **Slow Processing**: Enable GPU acceleration and parallel processing
2. **Memory Usage**: Monitor memory usage and adjust batch sizes
3. **Disk Space**: Ensure sufficient disk space for outputs
4. **Network Issues**: Use local processing for large datasets

### Configuration Issues

1. **Invalid Configurations**: Validate configuration files
2. **Missing Parameters**: Check required parameters for each script
3. **Type Errors**: Ensure correct data types in configuration
4. **Path Errors**: Use absolute paths in configuration files

---

*For detailed usage of individual scripts, run them with the `--help` flag.*
