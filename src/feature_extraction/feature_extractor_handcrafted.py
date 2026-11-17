import numpy as np
# import cv2
from skimage.measure import regionprops, label, perimeter
from skimage.filters import gabor
from scipy.stats import skew, kurtosis, entropy
from scipy.ndimage import center_of_mass
import pandas as pd
from joblib import Parallel, delayed
from typing import Dict


def compute_morphology_features(mask):
    """Compute morphology features from a binary mask."""
    props = regionprops(mask.astype(int))[0]
    
    area = props.area
    perim = perimeter(mask)
    major_axis = props.major_axis_length
    minor_axis = props.minor_axis_length
    elongation = major_axis / minor_axis if minor_axis > 0 else 0
    compactness = (perim ** 2) / (4 * np.pi * area) if area > 0 else 0
    circularity = (4 * np.pi * area) / (perim ** 2) if perim > 0 else 0
    feret_diameter = props.feret_diameter_max
    radius_gyration = props.inertia_tensor_eigvals[0] ** 0.5 if props.inertia_tensor_eigvals[0] > 0 else 0

    return {
        "area": area,
        "perimeter": perim,
        "elongation": elongation,
        "compactness": compactness,
        "circularity": circularity,
        "feret_diameter": feret_diameter,
        "radius_of_gyration": radius_gyration,
        "major_axis": major_axis,
        "minor_axis": minor_axis
    }


def compute_intensity_features(mask, image):
    """Compute intensity features from a binary mask and corresponding image."""
    masked_values = image[mask.astype(bool)]
    mean_intensity = np.mean(masked_values)
    std_intensity = np.std(masked_values)
    cv_intensity = std_intensity / mean_intensity if mean_intensity > 0 else 0
    total_intensity = np.sum(masked_values)

    return {
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "cv_intensity": cv_intensity,
        "total_intensity": total_intensity
    }


def compute_spatial_features(mask, image):
    """Compute spatial features from a binary mask and corresponding image."""
    props = regionprops(mask.astype(int), intensity_image=image)[0]
    centroid_y, centroid_x = props.centroid
    com_y, com_x = center_of_mass(image * mask)
    mass_displacement = np.sqrt((centroid_x - com_x) ** 2 + (centroid_y - com_y) ** 2)

    return {
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "center_of_mass_x": com_x,
        "center_of_mass_y": com_y,
        "mass_displacement": mass_displacement
    }


def compute_texture_features(mask, image):
    masked_img = image * mask
    masked_values = masked_img[mask.astype(bool)]

    # Gabor filter (example frequency and theta)
    filt_real, _ = gabor(masked_img, frequency=0.6, theta=0)
    gabor_vals = filt_real[mask.astype(bool)]
    
    gabor_mean = np.mean(gabor_vals)
    gabor_std = np.std(gabor_vals)

    img_skewness = skew(masked_values)
    img_kurtosis = kurtosis(masked_values)
    # Histogram over the ACTUAL intensity range of the masked pixels. A previous
    # hardcoded ``range=(0, 255)`` assumed 8-bit imagery; the brightfield inputs
    # are uint16 (values ~1000-11000), so no pixel fell in [0, 255], every bin
    # count was 0, and ``density=True`` divided by a zero total -> an all-NaN
    # histogram -> ``entropy`` returned NaN for every cell. Omitting ``range``
    # lets numpy span [min, max] of the data. ``scipy.stats.entropy`` renormalizes
    # ``pk`` internally, so ``density`` is immaterial here.
    hist, _ = np.histogram(masked_values, bins=256, density=True)
    img_entropy = entropy(hist + 1e-10)  # Avoid log(0)

    return {
        "gabor_mean": gabor_mean,
        "gabor_std": gabor_std,
        "skewness": img_skewness,
        "kurtosis": img_kurtosis,
        "entropy": img_entropy
    }


# === Example usage ===
# mask = (instance_mask == instance_id)
# features = {}
# features.update(compute_morphology_features(mask))
# features.update(compute_intensity_features(mask, image))
# features.update(compute_spatial_features(mask, image))
# features.update(compute_texture_features(mask, image))

# === Include previous feature functions here (unchanged) ===
# - compute_morphology_features
# - compute_intensity_features
# - compute_spatial_features
# - compute_texture_features

def extract_instance_features(instance_id: int, label_mask: np.ndarray, image: np.ndarray) -> Dict:
    """
    Extract features for a single instance given its ID, label mask, and corresponding image.
    Args:
        instance_id (int): The ID of the instance to extract features for.
        label_mask (np.ndarray): The labeled mask containing instance IDs.
        image (np.ndarray): The corresponding intensity image.
    Returns:
        Dict: A dictionary of extracted features for the instance.
    """
    mask = (label_mask == instance_id).astype(np.uint8)

    # Output the per-cell id as ``cell_id`` to match the mcherry_metrics CSV
    # contract (feature and target tables join on this name); the function
    # argument stays ``instance_id`` as it is the raw Cellpose label value.
    features = {
        "cell_id": instance_id
    }
    features.update(compute_morphology_features(mask))
    features.update(compute_intensity_features(mask, image))
    features.update(compute_spatial_features(mask, image))
    features.update(compute_texture_features(mask, image))
    
    return features


def extract_all_instance_features(label_mask: np.ndarray, image: np.ndarray, n_jobs: int = -1) -> pd.DataFrame:
    instance_ids = np.unique(label_mask)
    instance_ids = instance_ids[instance_ids != 0]  # Exclude background (label 0)

    results = Parallel(n_jobs=n_jobs)(
        delayed(extract_instance_features)(instance_id, label_mask, image)
        for instance_id in instance_ids
    )

    df = pd.DataFrame(results) # type: ignore
    return df
