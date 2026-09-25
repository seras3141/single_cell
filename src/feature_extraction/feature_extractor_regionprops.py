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
    region_props = get_region_properties(
        segmentation_image, intensity_image=brightfield_image
    )
    if output_csv_path is not None:
        region_props.to_csv(output_csv_path, index=False)
    return region_props
