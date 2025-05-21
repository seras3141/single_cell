import numpy as np
from scipy.ndimage import laplace
import tifffile as tiff
import os
from tqdm import tqdm
from glob import glob

def measure_patchwise_blur(img, patch_size=(50, 50), stride_size=(50, 50)):
    """
    Measure patchwise blur using Laplacian variance and find areas of highest focus
    across multiple z-stacks in a 3D TIFF file.

    Args:
        img (ndarray): 2D image array.
        patch_size (tuple): Size of the patches (height, width).
        stride_size (tuple): Stride size for sliding the patch.

    Returns:
        ndarray: 2D array representing the blur map.
    """
    height, width = img.shape
    stride_y, stride_x = stride_size
    pad_y, pad_x = patch_size[0] // 2, patch_size[1] // 2

    # Pad the image to ensure the i_th pixel is centered
    padded_img = np.pad(img, ((pad_y, pad_y), (pad_x, pad_x)), mode='reflect')
    blur_map = np.zeros(((height - 1) // stride_y + 1, (width - 1) // stride_x + 1))

    # Perform convolution-like operation with patches
    for i in range(0, height, stride_y):
        for j in range(0, width, stride_x):
            patch = padded_img[i:i + patch_size[0], j:j + patch_size[1]]
            if patch.size == 0:
                continue
            laplacian = laplace(patch)
            variance = np.var(laplacian)
            blur_map[i // stride_y, j // stride_x] = variance
    return blur_map

def resize_heatmap(normalized_heatmap, tiff_data, stride_size):
    resized_heatmap = np.zeros_like(tiff_data, dtype=np.float32)
    for z_index in range(normalized_heatmap.shape[0]):
        kron_result = np.kron(
            normalized_heatmap[z_index],
            np.ones((stride_size[0], stride_size[1]))
        )
        # Crop or pad the result to match the original TIFF data dimensions
        resized_heatmap[z_index, :kron_result.shape[0], :kron_result.shape[1]] = kron_result

    return resized_heatmap



def measure_blur_heatmap(tiff_path, patch_size=(50, 50), stride_size=(50,50)):
    """
    """
    # Load the 3D TIFF file
    tiff_data = tiff.imread(tiff_path)
    if tiff_data.ndim != 3:
        raise ValueError("The input TIFF file must be a 3D stack.")

    z_focus_areas = {}

    for z_index, z_slice in enumerate(tiff_data):
        # Initialize the blur map for the current z-slice
        z_focus_areas[z_index] = measure_patchwise_blur(z_slice, patch_size, stride_size)

    # Assert that every slice exists in the z_focus_areas dictionary
    assert len(z_focus_areas) == tiff_data.shape[0], "Not all slices are accounted for in z_focus_areas."

    # Convert the dictionary to a 3D array
    blur_map_3d = np.array([z_focus_areas[z] for z in sorted(z_focus_areas.keys())])

    # Normalize the 3D array across z-slices
    normalized_heatmap = (blur_map_3d - np.min(blur_map_3d, axis=0)) / (
        np.max(blur_map_3d, axis=0) - np.min(blur_map_3d, axis=0) + 1e-8
    )

    # Resize the 3D heatmap to the original size of the image, accounting for stride size
    resized_heatmap = resize_heatmap(normalized_heatmap, tiff_data, stride_size)
    return resized_heatmap

    # # Find the coordinates of the patches with the highest focus
    # max_focus_indices = np.argwhere(blur_map == np.max(blur_map))
    # z_focus_areas[z_index] = [(int(x * patch_size[0]), int(y * patch_size[1])) for x, y in max_focus_indices]

    # return z_focus_areas

def test_measure_blur_heatmap():
    input_path = "data/BF+IF Experiments_3D_train_test_dataset/train/p2126_J03_BF.tif"

    patch_size = (32, 32)
    stride_size = (8, 8)

    output_path = os.path.join("data/BF+IF Experiments_3D_train_test_dataset/train/blur_heatmaps", os.path.basename(tiff_path).replace("_BF.tif", f"_blur_heatmap_{patch_size[0]}_{stride_size[0]}.tif"))

    heatmap = measure_blur_heatmap(input_path, patch_size, stride_size, output_folder)

    # Save the resized heatmap as a TIFF file in a different output folder
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tiff.imwrite(output_path, heatmap.astype(np.float32))


# Example usage
if __name__ == "__main__":

    input_directory = "data/BF+IF Experiments_3D_train_test_dataset/test"  # Replace with your input directory
    output_directory = "data/BF+IF Experiments_3D_train_test_dataset/blur_heatmaps"

    patch_size = (32, 32)
    stride_size = (8, 8)

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)


    image_files = glob(os.path.join(input_directory, "*_BF.tif"))

    # Iterate through all TIFF files in the input directory
    for file_name in tqdm(image_files, desc="Processing TIFF files"):
        input_path = file_name  # file_name already contains the full path
        output_path = os.path.join(
            output_directory,
            os.path.basename(file_name).replace("_BF.tif", f"_blur_heatmap_{patch_size[0]}_{stride_size[0]}.tif")
        )

        # Measure blur heatmap
        heatmap = measure_blur_heatmap(input_path, patch_size, stride_size)

        # Save the resized heatmap as a TIFF file
        tiff.imwrite(output_path, heatmap.astype(np.float32))