# Python API Usage

Scripts can be called programmatically from Python using `subprocess` or by importing pipeline functions directly.

## Calling Scripts via subprocess

```python
import subprocess
import sys

def run_preprocessing(input_dir, output_dir, **kwargs):
    """Run preprocessing script from Python"""
    cmd = [
        sys.executable, "scripts/run_preprocessing.py",
        "-i", input_dir,
        "-o", output_dir,
    ]

    for key, value in kwargs.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr

# Usage
success, stdout, stderr = run_preprocessing(
    "data/raw", "data/processed",
    test_size=0.2,
    random_seed=42,
    wavelengths="1:FlipGFP,2:mCherry,3:BF",
    plate="2126",
)
```

## Calling Pipeline Functions Directly

The underlying functions used by each script can be imported and called directly:

```python
from src.utils.config import get_config_manager
from scripts.run_preprocessing import run_preprocessing_from_config

config_manager = get_config_manager(
    cli_args={"config": "config/preprocessing_config.yaml"}
)
run_preprocessing_from_config(config_manager.to_dict())
```

## Using ConfigurableFileHandler Directly

```python
from src.utils.file_utils import ConfigurableFileHandler

from src.utils.file_utils import ConfigurableFileHandler, EXPERIMENT_WAVELENGTH_MAPPINGS

# Use the named lookup for known experiments
handler = ConfigurableFileHandler(
    wavelength_mappings=EXPERIMENT_WAVELENGTH_MAPPINGS["HD1509"],
    plate_number="2126",
)

# Or pass mappings directly for custom setups
handler = ConfigurableFileHandler(
    wavelength_mappings={1: "FlipGFP", 2: "mCherry", 3: "BF"},
    plate_number="2126",
)

renamed = handler.rename_file("Plate 2126/t1_A01_s1_w2_z1.tif", "image")
# Output: p2126_A01_t1_z1_mCherry.tif
```
