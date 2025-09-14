# Python script to extract PyRadiomics features from a brightfield image and segmentation image

from glob import glob
import os
import radiomics
import SimpleITK as sitk

from radiomics import featureextractor
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
import tifffile as tiff
import umap
from tqdm import tqdm
import matplotlib.pyplot as plt

logger = radiomics.logging.getLogger("radiomics")
logger.setLevel(radiomics.logging.ERROR)


def visualize_region_properties(region_props, drop: list = ['label'], labels=None, out_name=None):
    # Drop the 'label' column and use it as color
    if labels:
        col = region_props[labels]
        region_props_features = region_props.drop(columns=labels)
    else:
        region_props_features = region_props
        col = None

    for c in drop:
        if c in region_props_features.columns:
            region_props_features = region_props_features.drop(columns=c)

    print("Labels:", labels)
    print("Region Properties Columns:", region_props.columns)

    # Convert categorical values to color values if labels are provided
    if labels and col is not None:
        unique_labels = col.unique()
        label_to_color = {label: idx for idx, label in enumerate(unique_labels)}
        col = col.map(label_to_color)

    # Perform UMAP dimensionality reduction

    if region_props_features.shape[1] == 0:
        xlabel, ylabel = region_props_features.columns
        embedding = region_props_features.to_numpy()
    else:
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(region_props_features)
        xlabel = "UMAP Dimension 1"
        ylabel = "UMAP Dimension 2"

    # Visualize the UMAP embedding
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=col, cmap='Spectral', s=5)
    plt.title('UMAP Visualization of Region Properties')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Add a color bar with labels
    if labels and col is not None:
        cbar = plt.colorbar(scatter, ticks=range(len(unique_labels)))
        cbar.ax.set_yticklabels(unique_labels)
        cbar.set_label('Labels')

    if out_name:
        plt.savefig(out_name)

    plt.show()


def get_region_properties(segmentation_mask, intensity_image=None):
    # Read segmentation and intensity images using tifffile

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

    return combined_df


# def read_image(image_path):
#     """
#     Read an image from a file path using SimpleITK.

#     Parameters:
#         image_path (str): Path to the image file.

#     Returns:
#         SimpleITK.Image: The loaded image.
#     """
#     # Transform input
#     im_vect = sitk.ReadImage('path/to/image.tiff')
#     # im_vect = sitk.JoinSeries(im_vect)  # Add 3rd dimension, NO LONGER NECESSARY

#     # Build full mask
#     im_size = numpy.array(im_vect.GetSize())[::-1]  # flip x, y, z to z, y, x
#     ma_arr = numpy.ones(im_size)
#     ma = sitk.GetImageFromArray(ma_arr)
#     ma.CopyInformation(im_vect)

#     return sitk.ReadImage(image_path)

def get_radiomics_features(brightfield_image, segmentation_image):
    # Initialize the feature extractor with default settings
    extractor = featureextractor.RadiomicsFeatureExtractor()

    # Initialize an empty DataFrame to store features for all z-stacks
    all_features = []

    # Iterate through z-stacks in the segmentation image
    for z in tqdm(range(segmentation_image.shape[0]), desc="Processing z-stacks"):
        brightfield_image_z = brightfield_image[z]
        segmentation_image_z = segmentation_image[z]

        # Convert the images to SimpleITK format
        brightfield_image_z_sitk = sitk.GetImageFromArray(brightfield_image_z)
        segmentation_image_z_sitk = sitk.GetImageFromArray(segmentation_image_z)
        brightfield_image_z_sitk.CopyInformation(brightfield_image_z_sitk)  # Copy metadata
        segmentation_image_z_sitk.CopyInformation(segmentation_image_z_sitk)  # Copy metadata

        # Iterate through each label in the segmentation image
        unique_labels, count = np.unique(segmentation_image_z, return_counts=True)
        unique_labels = unique_labels[count > 10]  # filter out labels with count less than 10
        unique_labels = unique_labels[unique_labels != 0]  # Exclude background (label 0)

        for label in unique_labels:
            # Extract features for the current label in the z-stack
            features_label = extractor.execute(brightfield_image_z_sitk, segmentation_image_z_sitk, label= int(label))

            # Add the label and z-stack value to the features
            features_label['label'] = label
            features_label['z_stack'] = z

            # Append the features to the list
            all_features.append(pd.DataFrame([features_label]))


    # Combine all features into a single DataFrame
    features_df = pd.concat(all_features, ignore_index=True)


def extract_features(brightfield_image_path, segmentation_image_path, output_csv_path=None, visualize=False):
    """
    Extract PyRadiomics features from a brightfield image and segmentation image.

    Parameters:
        brightfield_image_path (str): Path to the brightfield image (.tif file).
        segmentation_image_path (str): Path to the segmentation image (.tif file).
        output_csv_path (str): Path to save the extracted features as a CSV file.
    """

    segmentation_image = tiff.imread(segmentation_image_path)
    brightfield_image = tiff.imread(brightfield_image_path)
    plate = os.path.basename(brightfield_image_path).split('_')[0]

    with open("data/BF+IF Experiments Labeled/meta.csv") as f:
        meta = pd.read_csv(f)
        time_point = int(meta.loc[meta['plate'] == plate, 'time'].values[0])

    # brightfield_image = sitk.ReadImage(brightfield_image_path)
    # segmentation_image = sitk.ReadImage(segmentation_image_path)

    # radiomics_props = get_radiomics_features(brightfield_image, segmentation_image)

    region_props = get_region_properties(segmentation_image, intensity_image=brightfield_image)
    region_props['time'] = time_point

    # temporarily disable saving
    if output_csv_path is not None and False:
        region_props_csv = output_csv_path.replace(".csv", "_region_props.csv")
        region_props.to_csv(region_props_csv, index=False)

    


    if output_csv_path is not None and False:
        features_df.to_csv(output_csv_path, index=False)

    # # Save features to a CSV file
    # with open(output_csv_path, 'w') as csv_file:
    #     csv_file.write("Feature,Value\n")
    #     for feature_name, feature_value in features.items():
    #         csv_file.write(f"{feature_name},{feature_value}\n")

    # print(f"Features extracted and saved to {output_csv_path}")
    

    if visualize:
        visualize_region_properties(region_props)

    return region_props

def test_feature_extractor():
    data_dir = "/Users/serenasritharan/Projects/single-cell"

    brightfield_image_path = os.path.join(data_dir, "data/BF+IF Experiments_3D_train_test_dataset/train/p2126_J03_BF.tif")
    segmentation_image_path = os.path.join(data_dir, "data/BF+IF Experiments_3D_train_test_dataset/train/p2126_J03_Cells.tif")
    prediction_image_path = os.path.join(data_dir, "data/BF+IF Experiments_2D_train_test_dataset/predictions_test/pretrained_cell2d_cyto3_FlowT0.4/3d_view_tracked/p2126_J03_3d_filtered_0.5.tif")

    # Create the output directory for radiomics CSV files if it doesn't exist
    radiomics_csv_dir = os.path.join(data_dir, "data/BF+IF Experiments_2D_train_test_dataset/predictions_test/pretrained_cell2d_cyto3_FlowT0.4/radiomics_csv")
    os.makedirs(radiomics_csv_dir, exist_ok=True)

    # Generate the output CSV path
    output_csv_path = os.path.join(radiomics_csv_dir, os.path.basename(prediction_image_path).replace(".tif", ".csv"))

    gt_props = extract_features(brightfield_image_path, segmentation_image_path, visualize=False)
    gt_props['y'] = ['gt'] * len(gt_props['label'])

    pred_props = extract_features(brightfield_image_path, prediction_image_path, output_csv_path, visualize=False)
    pred_props['y'] = ['pred'] * len(pred_props['label'])

    combined_df = pd.concat([gt_props, pred_props], ignore_index=True)

    visualize_region_properties(combined_df, drop=['label', 'z_stack', 'y'], labels='y')

if __name__ == "__main__":
    data_dir = "/Users/serenasritharan/Projects/single-cell"

    brightfield_images = glob(os.path.join(data_dir, "data/BF+IF Experiments_3D_train_test_dataset/test/*_BF_3d.tif"))
    prediction_images = glob(os.path.join(data_dir, "data/BF+IF Experiments_2D_train_test_dataset/predictions_test/pretrained_cell2d_cyto3_FlowT0.4/3d_view_tracked/*_3d_filtered_0.5.tif"))
    segmentation_images = glob(os.path.join(data_dir, "data/BF+IF Experiments_3D_train_test_dataset/test/*_Cells_3d.tif"))

    all_props = []

    for brightfield_image, prediction_image in tqdm(zip(brightfield_images, prediction_images), desc="Processing images"):
        print(f"Processing {brightfield_image} and {prediction_image}")
        props = extract_features(brightfield_image, prediction_image, visualize=False)
        all_props.append(props)

    combined_df = pd.concat(all_props, ignore_index=True)

    visualize_region_properties(combined_df, drop=['label', 'z_stack', 'time'], labels='time', out_name='tmp5.png')


