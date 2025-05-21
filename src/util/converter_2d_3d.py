import os
import re
from PIL import Image
import tifffile as tiff
from tqdm import tqdm 
from collections import defaultdict

def combine_2d_to_3d(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Group files by base name and suffix (_BF or _Cells)
    file_groups = defaultdict(list)
    for file_name in tqdm(os.listdir(input_dir)):
        if file_name.endswith(".tif") or file_name.endswith(".tiff"):
            match = re.match(r"(.*)_z(\d+)(?:_(BF|Cells))?\.(tif|tiff)", file_name)
            if match:
                base_name = match.group(1)
                z_index = int(match.group(2))
                suffix = match.group(3)
                key = f"{base_name}_{suffix}"
                file_groups[key].append((z_index, file_name))

    # Combine 2D TIFFs into 3D TIFFs
    for key, files in file_groups.items():
        # Sort files by z-index
        files.sort(key=lambda x: x[0])
        images = []
        for _, file_name in files:
            file_path = os.path.join(input_dir, file_name)
            images.append(Image.open(file_path))

        # Save as 3D TIFF
        output_path = os.path.join(output_dir, f"{key}_3d.tif")
        tiff.imwrite(output_path, [img for img in images], photometric='minisblack')
        print(f"Saved 3D TIFF: {output_path}")

if __name__ == "__main__":
    input_directory = "data/BF+IF Experiments_2D_train_test_dataset/predictions_test/pretrained_cell2d_cyto3_FlowT0.4/masks"  # Replace with your input directory
    output_directory = "data/BF+IF Experiments_2D_train_test_dataset/predictions_test/pretrained_cell2d_cyto3_FlowT0.4/3d_view"  # Replace with your output directory
    combine_2d_to_3d(input_directory, output_directory)