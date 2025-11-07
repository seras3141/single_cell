import tifffile as tiff
import numpy as np
from pathlib import Path
from typing import Union


def load_image(file_path: Union[str, Path]) -> np.ndarray:
    file_path = Path(file_path)

    if file_path.suffix.lower() in ['.tif', '.tiff']:
        image = tiff.imread(str(file_path))
    else:
        # Try with PIL for other formats
        from PIL import Image
        image = np.array(Image.open(file_path))
    
    return image

