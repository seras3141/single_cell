import os
import pandas as pd
import tifffile as tiff
import numpy as np
from tqdm import tqdm
from pathlib import Path

import SimpleITK as sitk
from joblib import Parallel, delayed

try:
    from radiomics import featureextractor
except ImportError:
    featureextractor = None


def extract_radiomic_features(
        brightfield_image_path : str | Path, 
        segmentation_image_path : str | Path, 
        output_csv_path : str | Path = None,
        visualize : bool = False
    ) -> pd.DataFrame:
    """
    Extract PyRadiomics features from a brightfield image and segmentation image.

    Parameters:
        brightfield_image_path (str): Path to the brightfield image (.tif file).
        segmentation_image_path (str): Path to the segmentation image (.tif file).
        output_csv_path (str): Path to save the extracted features as a CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the extracted features.
    """

    brightfield_image_path = str(brightfield_image_path)
    segmentation_image_path = str(segmentation_image_path)

    if not os.path.exists(brightfield_image_path):
        raise FileNotFoundError(f"Brightfield image not found: {brightfield_image_path}")
    if not os.path.exists(segmentation_image_path):
        raise FileNotFoundError(f"Segmentation image not found: {segmentation_image_path}")

    segmentation_image = tiff.imread(segmentation_image_path)
    brightfield_image = tiff.imread(brightfield_image_path)

    # plate = os.path.basename(brightfield_image_path).split('_')[0]

    # with open("data/BF+IF Experiments Labeled/meta.csv") as f:
    #     meta = pd.read_csv(f)
    #     time_point = int(meta.loc[meta['plate'] == plate, 'time'].values[0])

    # brightfield_image = sitk.ReadImage(brightfield_image_path)
    # segmentation_image = sitk.ReadImage(segmentation_image_path)

    radiomics_props = get_radiomics_features(brightfield_image, segmentation_image)

    if output_csv_path is not None:
        radiomics_props.to_csv(output_csv_path, index=False)

    # # Save features to a CSV file
    # with open(output_csv_path, 'w') as csv_file:
    #     csv_file.write("Feature,Value\n")
    #     for feature_name, feature_value in features.items():
    #         csv_file.write(f"{feature_name},{feature_value}\n")

    # print(f"Features extracted and saved to {output_csv_path}")
    
    # if visualize:
    #     visualize_region_properties(radiomics_props)

    return radiomics_props


def get_radiomics_features_2d(
        brightfield_image_sitk : sitk.Image,
        segmentation_image_sitk : sitk.Image,
        extractor: "featureextractor.RadiomicsFeatureExtractor",
        n_jobs: int = -1,
    ) -> pd.DataFrame:

    unique_labels, count = np.unique(segmentation_image_sitk, return_counts=True)
    unique_labels = unique_labels[count > 10]  # filter out labels with count less than 10
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background (label 0)

    # Iterate through each label in the segmentation image
    def _extract_features(label):
        return extractor.execute(brightfield_image_sitk, segmentation_image_sitk, label=int(label))

    if n_jobs == 0:
        all_features = []
        for label in unique_labels:
            features_label = _extract_features(label)
            features_label['label'] = label
            all_features.append(features_label)
        features_df = pd.DataFrame(all_features, columns=all_features[0].keys())
        # pop all columns that are not features (starting with 'diagnostics'))
        features_df = features_df[[col for col in features_df.columns if not col.startswith('diagnostics')]]
        return features_df


    # Use joblib to parallelize the feature extraction for each label
    features_list = Parallel(n_jobs=-1)(delayed(_extract_features)(label) for label in unique_labels)

    # Concatenate the results
    features_df = pd.DataFrame(features_list, columns=[k for k in features_list[0].keys() if not k.startswith('diagnostics')])
    # HOW CAN I FORCE THE LABELS TO BE IN ORDER HERE
    features_df['label'] = unique_labels

    return features_df


def get_radiomics_features(brightfield_image, segmentation_image, extractor=None) -> pd.DataFrame:
    # Initialize the feature extractor with default settings
    extractor = extractor or featureextractor.RadiomicsFeatureExtractor()

    if brightfield_image.ndim == 2 and segmentation_image.ndim == 2:
        # Single 2D image
        brightfield_image_sitk = sitk.GetImageFromArray(brightfield_image)
        segmentation_image_sitk = sitk.GetImageFromArray(segmentation_image)

        features_df = get_radiomics_features_2d(
            brightfield_image_sitk=brightfield_image_sitk,
            segmentation_image_sitk=segmentation_image_sitk,
            extractor=extractor,
        )

        return features_df

    elif brightfield_image.ndim == 3 and segmentation_image.ndim == 3:

        # Initialize an empty DataFrame to store features for all z-stacks
        all_features = []

        # Iterate through z-stacks in the segmentation image
        for z in tqdm(range(segmentation_image.shape[0]), desc="Processing z-stacks"):
            brightfield_image_z = brightfield_image[z]
            segmentation_image_z = segmentation_image[z]

            # Convert the images to SimpleITK format
            brightfield_image_z_sitk = sitk.GetImageFromArray(brightfield_image_z)
            segmentation_image_z_sitk = sitk.GetImageFromArray(segmentation_image_z)

            features_df_z = get_radiomics_features_2d(
                brightfield_image_sitk=brightfield_image_z_sitk,
                segmentation_image_sitk=segmentation_image_z_sitk,
                extractor=extractor
            )
            features_df_z['z'] = z  # Add z-stack information

            all_features.append(features_df_z)

        # Combine all features into a single DataFrame
        features_df = pd.concat(all_features, ignore_index=True)

        return features_df
    else:
        raise ValueError("Brightfield and segmentation images must both be 2D or both be 3D.")

def test_feature_extractor_pyradiomics():
    data_dir = "/Users/serenasritharan/Projects/single-cell"

    brightfield_image_path = Path(data_dir) / "data/Plate 2426_preprocessed_2D/test/p2426_B06_z10_BF.tif"
    segmentation_image_path = Path(data_dir) / "data/Plate 2426_preprocessed_2D/test/p2426_B06_z10_Cells.tif"
    prediction_image_path = Path(data_dir) / "data/Plate 2426_preprocessed_2D/inference/cyto3/test_tracking_results/final_2d/p2426_B06_z10_masks.tif"

    # brightfield_image_path = Path(data_dir) / "data/BF+IF Experiments_3D_train_test_dataset/train/p2126_J03_BF.tif"
    # segmentation_image_path = Path(data_dir) / "data/BF+IF Experiments_3D_train_test_dataset/train/p2126_J03_Cells.tif"
    # prediction_image_path = Path(data_dir) / "data/BF+IF Experiments_2D_train_test_dataset/predictions_test/pretrained_cell2d_cyto3_FlowT0.4/3d_view_tracked/p2126_J03_3d_filtered_0.5.tif"

    # Create the output directory for radiomics CSV files if it doesn't exist
    radiomics_csv_dir = Path(data_dir) / "feature_extractor_pyradiomics/BF+IF Experiments_2D_train_test_dataset/predictions_test/pretrained_cell2d_cyto3_FlowT0.4/radiomics_csv"
    radiomics_csv_dir.mkdir(parents=True, exist_ok=True)

    # Generate the output CSV path
    output_csv_path = radiomics_csv_dir / (prediction_image_path.stem + ".csv")

    gt_props = extract_radiomic_features(brightfield_image_path, segmentation_image_path, visualize=False)
    gt_props['y'] = ['gt'] * len(gt_props['label'])

    pred_props = extract_radiomic_features(brightfield_image_path, prediction_image_path, output_csv_path, visualize=False)
    pred_props['y'] = ['pred'] * len(pred_props['label'])

    combined_df = pd.concat([gt_props, pred_props], ignore_index=True)

    # visualize_region_properties(combined_df, drop=['label', 'z_stack', 'y'], labels='y')

if __name__ == "__main__":
    test_feature_extractor_pyradiomics()