import argparse
import os
from src.preprocessing.dataset_split import train_test_split_directory
from src.utils.file_utils import BF_IF_FileHandler
from src.utils.conversion import combine_2d_to_3d
from src.preprocessing.blur_analysis import measure_dataset_blur_heatmaps

def main():
    parser = argparse.ArgumentParser(description="Run full preprocessing pipeline: split, combine 2D to 3D, blur heatmaps.")
    parser.add_argument("dataset_path", help="Path to raw dataset directory (input)")
    parser.add_argument("output_root", help="Root output directory for all processed data")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data for test set")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--image-pattern", default="t1_*_w1_*.tif", help="Glob pattern for image files")
    parser.add_argument("--mask-pattern", default="Cells_*.tif", help="Glob pattern for mask files")
    parser.add_argument("--patch-size", type=int, default=32, help="Patch size for blur detection")
    parser.add_argument("--stride-size", type=int, default=16, help="Stride size for blur detection")
    parser.add_argument("--combine-pattern", default=r"(.+?)_z(\d+)(?:_(BF|Cells))?\.(tif)", help="Regex for 2D to 3D grouping")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    # Step 1: Split dataset
    split_dir = os.path.join(args.output_root, "split")
    print(f"Splitting dataset into train/test at {split_dir} ...")
    train_test_split_directory(
        data_dir=args.dataset_path,
        output_dir=split_dir,
        test_size=args.test_size,
        random_state=args.random_seed,
        image_pattern=args.image_pattern,
        mask_pattern=args.mask_pattern,
        file_handler=BF_IF_FileHandler()
    )

    # Step 2: Combine 2D to 3D (for train set only)
    input_2d_dir = split_dir
    output_3d_dir = os.path.join(args.output_root, "3d_images")
    print(f"Combining 2D images into 3D stacks at {output_3d_dir} ...")
    combine_2d_to_3d(
        input_dir=input_2d_dir,
        output_dir=output_3d_dir,
        pattern=args.combine_pattern,
        recursive=True,
    )

    # Step 3: Generate blur heatmaps
    blur_dir = os.path.join(args.output_root, "blur_heatmaps")
    print(f"Generating blur heatmaps at {blur_dir} ...")
    measure_dataset_blur_heatmaps(
        input_dir=output_3d_dir,
        output_dir=blur_dir,
        pattern="*_BF_3d.tif",
        patch_size=args.patch_size,
        stride_size=args.stride_size,
        normalize=True,
        overwrite=args.overwrite
    )
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()
