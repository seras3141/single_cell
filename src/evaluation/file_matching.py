from pathlib import Path
import logging
from typing import List, Dict, Union, Optional, Any
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_image_file(file_path: Path) -> np.ndarray:
    """
    Load a single image file.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Loaded image as numpy array
    """
    try:
        from PIL import Image
        import numpy as np
        
        # Try PIL first for common image formats
        img = Image.open(file_path)
        try:
            # Check if it's a multi-frame image
            n_frames = getattr(img, 'n_frames', 1)
            if n_frames > 1:
                # Multi-frame TIFF
                frames = []
                for i in range(n_frames):
                    img.seek(i)
                    frames.append(np.array(img))
                return np.array(frames)
            else:
                return np.array(img)
        except AttributeError:
            # Single frame image
            return np.array(img)
            
    except Exception as e:
        logger.warning(f"PIL failed to load {file_path}, trying other methods: {e}")
        
        try:
            # Try with skimage
            from skimage import io
            return io.imread(str(file_path))
        except Exception as e2:
            logger.warning(f"skimage failed to load {file_path}: {e2}")
            
            try:
                # Try with opencv
                import cv2
                img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError(f"Could not load image: {file_path}")
                return img
            except Exception as e3:
                logger.error(f"All image loading methods failed for {file_path}: {e3}")
                raise


def load_masks_from_directory(
    directory: Path,
    file_suffix: str = "_masks.tif",
    # supported_extensions: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Load all mask files from a directory.
    
    Args:
        directory: Directory containing mask files
        file_pattern: Glob pattern for file matching
        supported_extensions: List of supported file extensions
        
    Returns:
        Dictionary mapping filenames to loaded masks
    """
    # if supported_extensions is None:
    #     supported_extensions = ['.png', '.tif', '.tiff', '.jpg', '.jpeg']

    if file_suffix not in ["_masks.tif", "_Cells.tif"]:
        raise NotImplementedError("Complex file patterns are not supported yet. Use simple patterns like '*.png'.")

    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    masks = {}

    files = directory.glob(f"*{file_suffix}")

    # Filter by supported extensions
    # supported_files = [f for f in files if f.suffix.lower() in supported_extensions]
    
    # if not supported_files:
    #     raise ValueError(f"No supported files found in {directory}. "
    #                     f"Supported extensions: {supported_extensions}")

    supported_files = list(files)
    
    logger.info(f"Loading {len(supported_files)} files from {directory}")
    
    for file_path in tqdm(sorted(supported_files), desc="Loading masks"):
        try:
            mask = load_image_file(file_path)
            
            # Use filename without suffix as key
            key = file_path.name.replace(file_suffix, "")
            masks[key] = mask
            
            logger.debug(f"Loaded {key}: shape {mask.shape}, dtype {mask.dtype}")
            
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            continue
    
    return masks


def load_single_file(file_path: Path) -> np.ndarray:
    """
    Load a single file containing multiple masks.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Loaded array (possibly 3D for multiple masks)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Loading single file: {file_path}")
    
    return load_image_file(file_path)


def match_prediction_to_labels(
    predictions: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray]
) -> List[tuple]:
    """
    Match prediction masks to label masks by filename.
    
    Args:
        predictions: Dictionary of prediction masks
        labels: Dictionary of label masks
        
    Returns:
        List of (pred_mask, label_mask, image_id) tuples
    """
    matched_pairs = []
    
    # Find common keys
    pred_keys = set(predictions.keys())
    label_keys = set(labels.keys())
    common_keys = pred_keys & label_keys
    
    if not common_keys:
        logger.error("No matching files found between predictions and labels!")
        logger.info(f"Prediction files: {sorted(pred_keys)}")
        logger.info(f"Label files: {sorted(label_keys)}")
        raise ValueError("No matching files found")
    
    missing_in_pred = label_keys - pred_keys
    missing_in_labels = pred_keys - label_keys
    
    if missing_in_pred:
        logger.warning(f"Labels without predictions: {sorted(missing_in_pred)}")
    if missing_in_labels:
        logger.warning(f"Predictions without labels: {sorted(missing_in_labels)}")
    
    logger.info(f"Found {len(common_keys)} matching pairs")
    
    for key in sorted(common_keys):
        pred_mask = predictions[key]
        label_mask = labels[key]
        
        # Validate shapes
        if pred_mask.shape != label_mask.shape:
            logger.warning(f"Shape mismatch for {key}: pred {pred_mask.shape} vs label {label_mask.shape}")
            # Try to handle 3D vs 2D
            if len(pred_mask.shape) == 3 and len(label_mask.shape) == 2:
                if pred_mask.shape[0] == 1:
                    pred_mask = pred_mask[0]
                else:
                    logger.error(f"Cannot handle 3D prediction with {pred_mask.shape[0]} frames")
                    continue
            elif len(label_mask.shape) == 3 and len(pred_mask.shape) == 2:
                if label_mask.shape[0] == 1:
                    label_mask = label_mask[0]
                else:
                    logger.error(f"Cannot handle 3D label with {label_mask.shape[0]} frames")
                    continue
            else:
                logger.error(f"Cannot resolve shape mismatch for {key}")
                continue
        
        matched_pairs.append((pred_mask, label_mask, key))
    
    return matched_pairs



def get_matching_prediction_label_files(
    predictions_path: Union[str, Path],
    labels_path: Union[str, Path],
    prediction_file_pattern: str = "*",
    label_file_pattern: str = "*"
) -> List[tuple]:
    """
    Get matching prediction and label files from directories or single files.
    
    Args:
        predictions_path: Path to predictions (directory or file)
        labels_path: Path to labels (directory or file)
        file_pattern: Pattern for matching files in directories
    Returns:
        List of (pred_mask, label_mask, image_id) tuples
    """

    predictions_path = Path(predictions_path)
    labels_path = Path(labels_path)
    prediction_file_suffix = prediction_file_pattern.replace("*", "")
    label_file_suffix = label_file_pattern.replace("*", "")
    
    logger.info("Starting evaluation...")
    logger.info(f"Predictions: {predictions_path}, with pattern: {prediction_file_pattern}")
    logger.info(f"Labels: {labels_path}, with pattern: {label_file_pattern}")

    # Load predictions
    logger.info("Loading predictions...")
    if predictions_path.is_dir():
        predictions = load_masks_from_directory(predictions_path, prediction_file_suffix)
    else:
        pred_array = load_single_file(predictions_path)
        if len(pred_array.shape) == 3:
            # Multiple masks in one file
            predictions = {f"image_{i:04d}": pred_array[i] for i in range(pred_array.shape[0])}
        else:
            # Single mask
            predictions = {"image_0000": pred_array}
    
    # Load labels
    logger.info("Loading labels...")
    if labels_path.is_dir():
        labels = load_masks_from_directory(labels_path, label_file_suffix)
    else:
        label_array = load_single_file(labels_path)
        if len(label_array.shape) == 3:
            # Multiple masks in one file
            labels = {f"image_{i:04d}": label_array[i] for i in range(label_array.shape[0])}
        else:
            # Single mask
            labels = {"image_0000": label_array}
    
    # Match predictions to labels
    logger.info("Matching predictions to labels...")
    matched_pairs = match_prediction_to_labels(predictions, labels)

    return matched_pairs
