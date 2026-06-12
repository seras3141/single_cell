import random

import numpy as np
from numpy.fft import fft2
from scipy.ndimage import laplace
from skimage import io
from skimage.measure import shannon_entropy
from tqdm import tqdm


def laplacian_variance(image: np.ndarray) -> float:
    return float(np.var(laplace(image)))


def fft_sharpness(image: np.ndarray) -> float:
    return float(np.mean(np.abs(fft2(image))))


def get_quality_metrics(image: np.ndarray) -> tuple[float, float, float]:
    return laplacian_variance(image), fft_sharpness(image), float(shannon_entropy(image))


def get_image_quality_metrics(images: list[str]) -> dict[str, dict[str, float]]:
    metrics = {}
    for img_path in tqdm(images):
        img = io.imread(img_path, as_gray=True)
        lv, fs, se = get_quality_metrics(img)
        metrics[img_path] = {
            "laplacian_variance": lv,
            "fft_sharpness": fs,
            "shannon_entropy": se,
        }
    return metrics


def get_percentile_images(
    metrics: dict[str, dict[str, float]],
    metric_name: str,
    low_pct: int = 5,
    mid_pct: int = 10,
    high_pct: int = 5,
    sample_size: int = 3,
) -> dict[str, list[str]]:
    sorted_items = sorted(metrics.items(), key=lambda x: x[1][metric_name])
    n = len(sorted_items)
    low_n = max(1, int(n * low_pct / 100))
    high_n = max(1, int(n * high_pct / 100))
    mid_n = max(1, int(n * mid_pct / 100))

    lowest = sorted_items[:low_n]
    highest = sorted_items[-high_n:]
    mid_start = (n - mid_n) // 2
    mid = sorted_items[mid_start : mid_start + mid_n]

    return {
        "highest": [item[0] for item in random.sample(highest, min(len(highest), sample_size))],
        "mid": [item[0] for item in random.sample(mid, min(len(mid), sample_size))],
        "lowest": [item[0] for item in random.sample(lowest, min(len(lowest), sample_size))],
    }


def max_intensity_projection_metrics(
    base_id_groups: dict[str, list[str]],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    projections: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}

    for base_id, paths in tqdm(base_id_groups.items()):
        imgs = []
        for p in paths:
            img = io.imread(p)
            if img.ndim == 3:
                img = img.mean(axis=-1)
            imgs.append(img)
        max_proj = np.stack(imgs, axis=0).max(axis=0)
        projections[base_id] = max_proj

        lv, fs, se = get_quality_metrics(max_proj)
        metrics[base_id] = {
            "laplacian_variance": lv,
            "fft_sharpness": fs,
            "shannon_entropy": se,
        }

    return projections, metrics
