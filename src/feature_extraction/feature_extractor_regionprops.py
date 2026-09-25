# Extract skimage regionprops features from a segmentation mask and an optional intensity image

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import tifffile as tiff
from skimage.measure import regionprops_table

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


def get_region_properties(segmentation_mask, intensity_image=None):

    if segmentation_mask.ndim ==2:
        # Extract region properties for the current z-stack
        properties = regionprops_table(
            segmentation_mask,
            intensity_image=intensity_image,
            properties=[
                'label',
                'area',
                'eccentricity',
                # 'bbox',
                # 'centroid',
                'mean_intensity',
                'max_intensity',
                'min_intensity'
            ]
        )
        # Rename skimage's ``label`` id column to ``cell_id`` for consistency
        # with the incarta/scPortrait outputs and the mcherry_metrics contract.
        return pd.DataFrame(properties).rename(columns={'label': 'cell_id'})

    elif segmentation_mask.ndim == 3:

        # Initialize an empty list to store region properties for all z-stacks
        all_properties = []

        # Iterate through each z-stack
        for z in range(segmentation_mask.shape[0]):
            segmentation_slice = segmentation_mask[z]
            intensity_slice = intensity_image[z] if intensity_image is not None else None

            # Extract region properties for the current z-stack
            properties = regionprops_table(
                segmentation_slice,
                intensity_image=intensity_slice,
                properties=[
                    'label',
                    'area',
                    'eccentricity',
                    # 'bbox',
                    # 'centroid',
                    'mean_intensity',
                    'max_intensity',
                    'min_intensity'
                ]
            )

            # Add the z-stack value to the properties
            properties['z_stack'] = [z] * len(properties['label'])

            # Append the properties to the list
            all_properties.append(pd.DataFrame(properties))

        # Combine all properties into a single DataFrame
        combined_df = pd.concat(all_properties, ignore_index=True)

        # Rename skimage's ``label`` id column to ``cell_id`` (see 2D branch).
        return combined_df.rename(columns={'label': 'cell_id'})
    else:
        raise ValueError("Segmentation mask must be either 2D or 3D.")


def extract_regionprops_features(
    brightfield_image_path: Union[str, Path],
    segmentation_image_path: Union[str, Path],
    output_csv_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Extract regionprops features from a brightfield image and segmentation image.

    Parameters:
        brightfield_image_path (str): Path to the brightfield image (.tif file).
        segmentation_image_path (str): Path to the segmentation image (.tif file).
        output_csv_path (str): Path to save the extracted features as a CSV file.
    """

    segmentation_image = tiff.imread(segmentation_image_path)
    brightfield_image = tiff.imread(brightfield_image_path)

    # plate = os.path.basename(brightfield_image_path).split('_')[0]

    # with open("data/BF+IF Experiments Labeled/meta.csv") as f:
    #     meta = pd.read_csv(f)
    #     time_point = int(meta.loc[meta['plate'] == plate, 'time'].values[0])

    region_props = get_region_properties(segmentation_image, intensity_image=brightfield_image)
    # region_props['time'] = time_point

    if output_csv_path is not None:
        region_props.to_csv(output_csv_path, index=False)

    # # Save features to a CSV file
    # with open(output_csv_path, 'w') as csv_file:
    #     csv_file.write("Feature,Value\n")
    #     for feature_name, feature_value in features.items():
    #         csv_file.write(f"{feature_name},{feature_value}\n")

    # print(f"Features extracted and saved to {output_csv_path}")
    

    return region_props
