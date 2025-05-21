import os
import re
import random
from pathlib import Path
from PIL import Image
import tifffile as tiff
from tqdm import tqdm
import json

from util.train_test_split import DatasetSplitter
from util.file_renamer import BF_IF_FileRenamer, DatasetPaths
from util.converter_2d_3d import combine_2d_to_3d

def split_2d_files(input_dir, output_dir, test_size=0.2, random_state=42):
    """Split 2D TIFF files into train and test sets."""
    random.seed(random_state)
    
    # Group files by base name (excluding z-stack index)
    file_groups = {}
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".tif") or file_name.endswith(".tiff"):
            match = re.match(r"(.*)_z(\d+)\.tif", file_name)
            if match:
                base_name = match.group(1)
                z_index = int(match.group(2))
                if base_name not in file_groups:
                    file_groups[base_name] = []
                file_groups[base_name].append((z_index, file_name))
    
    # Split into train and test sets
    base_names = list(file_groups.keys())
    random.shuffle(base_names)
    num_test = max(1, int(len(base_names) * test_size))
    test_base_names = set(base_names[:num_test])
    train_base_names = set(base_names[num_test:])
    
    train_files = {k: file_groups[k] for k in train_base_names}
    test_files = {k: file_groups[k] for k in test_base_names}
    
    # Save split information
    split_info = {
        "train": list(train_base_names),
        "test": list(test_base_names)
    }
    with open(os.path.join(output_dir, "split.json"), "w") as f:
        json.dump(split_info, f, indent=4)
    
    return train_files, test_files

def main():

    input_dir = "data/Plate 2426"
    output_dir = "data/Plate 2426_2D_train_test_dataset"
    output_dir_3d = "data/Plate 2426_3D_train_test_dataset"

    paths = DatasetPaths(
        image_path=os.path.join(input_dir, "**/t1_*_w1_*.tif"),
        mask_path=os.path.join(input_dir, "**/*_V5/Cells_*.tif"),
        output_dir=output_dir
    )

    file_handler = BF_IF_FileRenamer()

    '''
    paths_3D = DatasetPaths(
        image_path="data/BF+IF Experiments Labeled_3D/**/*_BF_3D.tif",
        mask_path="data/BF+IF Experiments Labeled_3D/**/*_Cells_3D.tif",
        output_dir="data/BF+IF Experiments_3D_train_test_dataset"
    )

    file_handler = BF_IF_FileRenamer_3D()
    '''

    
    splitter = DatasetSplitter(paths=paths, file_handler=file_handler)
    splitter.process()


    # Convert train set to 3D
    train_input_dir = os.path.join(output_dir, "train")
    train_output_dir = os.path.join(output_dir_3d, "train")
    combine_2d_to_3d(train_input_dir, train_output_dir)

    # Convert test set to 3D
    test_input_dir = os.path.join(output_dir, "test")
    test_output_dir = os.path.join(output_dir_3d, "test")
    combine_2d_to_3d(test_input_dir, test_output_dir)

if __name__ == "__main__":
    main()
