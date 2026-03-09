# from skimage.measure import regionprops
# import os
# import tifffile
# import numpy as np
# from tqdm import tqdm
# import trackpy as tp
# import pandas as pd
# from glob import glob
# from .blur_measure import measure_blur_heatmap

# def get_label_centers(segmentation_mask, sharpness_image = None, blur_thresh = .5, inv=False):
#     """
#     Get the center coordinates of labels from an instance segmentation mask.

#     Parameters:
#         segmentation_mask (ndarray): Instance segmentation mask.

#     Returns:
#         list of tuple: List of (y, x) coordinates for the centers of each label.
#     """

#     def comp(x, y, inv):
#         if inv:
#             return x > y
#         else:
#             return x < y

#     if sharpness_image is not None:
#         # TODO add sharpness as intensity image
#         regions_unfiltered = regionprops(segmentation_mask, intensity_image=sharpness_image, extra_properties=[blur_intensity])
#         regions = list(filter(lambda x: comp(x['blur_intensity'], blur_thresh, inv), regions_unfiltered))
#         # regions = regions_unfiltered
#     else:
#         regions = regionprops(segmentation_mask)

#     # Filter regions based on area percentiles
#     areas = [region.area for region in regions]
#     # lower_bound, upper_bound = np.percentile(areas, [0.1, 99.9])
#     regions_SF = [region for region in regions if 10 <= region.area <= 5000]

#     # Get the coordinates of the centroids of each region
#     props = dict(
#         x = [region.centroid[0] for region in regions],
#         y = [region.centroid[1] for region in regions],
#         label = [region.label for region in regions],
#         # coords = [region.coords for region in regions]
#         )
#     return pd.DataFrame.from_dict(props)


# #TODO implement blur intensity
# def blur_intensity(regionmask, intensity_image):
#         return np.mean(intensity_image[regionmask])

# '''
# def get_centers_from_tif_files(directory):
#     """
#     Read multiple .tif files containing instance segmentation labels at different z-stacks,
#     and get the center of each instance, ordered by z-stack.

#     Parameters:
#         directory (str): Path to the directory containing .tif files.

#     Returns:
#         dict: A dictionary where keys are file names and values are lists of (z, y, x) coordinates for the centers.
#     """
#     centers_dict = {}
#     for file_name in os.listdir(directory):
#         if file_name.endswith('.tif'):
#             file_path = os.path.join(directory, file_name)
#             segmentation_mask = tifffile.imread(file_path)
            
#             # Extract z-stack number from the file name (assuming format includes '_z<number>.tif')
#             zstack = int(file_name.split('_z')[-1].split('.tif')[0])
            
#             centers = get_label_centers(segmentation_mask)
#             # Add z-stack information to each center
#             centers_with_z = [(zstack, y, x) for y, x in centers]
            
#             if file_name not in centers_dict:
#                 centers_dict[file_name] = []
#             centers_dict[file_name].extend(centers_with_z)
    
#     # Sort centers by z-stack for each file
#     for file_name in centers_dict:
#         centers_dict[file_name].sort(key=lambda center: center[0])  # Sort by z-stack
    
#     return centers_dict
# '''

# def get_centers_from_3d_mask(segmentation_stack, sharpness_image=None, **kwargs):
#     """
#     Get the center coordinates of labels from a single 3D .tif file, layer by layer.

#     Parameters:
#         file_path (str): Path to the 3D .tif file.

#     Returns:
#         list of tuple: List of (z, y, x) coordinates for the centers of each label.
#     """
#     centers_with_z = []

#     if sharpness_image is None:
#         sharpness_image = [None] * len(segmentation_stack)

#     for z, segmentation_mask in enumerate(segmentation_stack):
#         props = get_label_centers(segmentation_mask, sharpness_image=sharpness_image[z], **kwargs)
#         centers_with_z.append((z, props))

#     return centers_with_z

# def track_3d_centers(segmentation_stack, sharpness_image=None, **kwargs):
#     """
#     Track the centers across z-stacks using trackpy and create a new 3D .tif file
#     with tracked indices.

#     Parameters:
#         segmentation_stack (ndarray): 3D numpy array with untracked labels.

#     Returns:
#         ndarray: 3D numpy array with labels replaced by corresponding particle IDs.
#     """
#     # Get centers from the 3D .tif file
#     centers_with_z = get_centers_from_3d_mask(segmentation_stack, sharpness_image=sharpness_image, **kwargs)

#     # Flatten the centers into a DataFrame for tracking
#     data = pd.concat(
#         [props.assign(frame=z) for z, props in centers_with_z], 
#         ignore_index=True
#     )

#     # Perform tracking using trackpy
#     tracked = tp.link_df(data, search_range=5, memory=1)

#     # Create a new 3D array with the same shape as the input file
#     tracked_stack = np.zeros_like(segmentation_stack, dtype=np.int32)

#     # Replace label values in the segmentation stack with particle IDs
#     for _, row in tracked.iterrows():
#         z, particle_id = int(row['frame']), int(row['particle']) + 1
#         label = int(row['label'])
#         tracked_stack[z][segmentation_stack[z, :, :] == label] = particle_id

#     return tracked_stack

# def test_tracking():
#     input_file = "data/BF+IF Experiments_2D_train_test_dataset/predictions_test/pretrained_cell2d_cyto3_FlowT1.0/3d_view/p2126_J03_3d.tif"
#     output_file = "data/BF+IF Experiments_2D_train_test_dataset/predictions_test/pretrained_cell2d_cyto3_FlowT1.0/3d_view_tracked/p2126_J03_3d.tif"

#     os.makedirs(os.path.dirname(output_file), exist_ok=True)

#     # Example usage
#     segmentation_stack = tifffile.imread(input_file).astype(int)
#     sharpness_image = tifffile.imread("data/BF+IF Experiments_3D_train_test_dataset/train/blur_heatmaps/p2126_J03_blur_heatmap_32.tif")

#     tracked_stack = track_3d_centers(segmentation_stack, sharpness_image = sharpness_image)

#     # Save the tracked 3D array as a new .tif file
#     tifffile.imwrite(output_file, tracked_stack)

# def main():

#     # image_directory = "data/BF+IF Experiments_3D_train_test_dataset/test"
#     # mask_directory = "data/BF+IF Experiments_2D_train_test_dataset/predictions_test/pretrained_cell2d_cyto3_FlowT0.4/3d_view"
#     # output_directory = "data/BF+IF Experiments_2D_train_test_dataset/predictions_test/pretrained_cell2d_cyto3_FlowT0.4/3d_view_tracked"  # Replace with your input directory
#     # blur_directory = "data/BF+IF Experiments_3D_train_test_dataset/blur_heatmaps"  

#     '''
#     # 2426
#     image_directory = "data/Plate 2426_3D_train_test_dataset/test"
#     mask_directory = "data/Plate 2426_2D_train_test_dataset/predictions_test/pretrained_cell2d_cyto3_FlowT0.2/3d_view"
#     output_directory = "data/Plate 2426_2D_train_test_dataset/predictions_test/pretrained_cell2d_cyto3_FlowT0.2/3d_view_tracked"  # Replace with your input directory
#     blur_directory = "data/Plate 2426_3D_train_test_dataset/blur_heatmaps"  
#     '''

#     # old_plates
#     image_directory = "data/BF+IF Experiments_3D_train_test_dataset/train"
#     mask_directory = "data/BF+IF Experiments_2D_train_test_dataset/predictions_test/pretrained_cell2d_cyto3_FlowT0.4/3d_view"
#     output_directory = "data/BF+IF Experiments_2D_train_test_dataset/predictions_test/pretrained_cell2d_cyto3_FlowT0.4/3d_view_tracked_SF"  # Replace with your input directory
#     blur_directory = "data/BF+IF Experiments_3D_train_test_dataset/blur_heatmaps"  


#     blur_thresh = .5
#     inv = False

#     os.makedirs(output_directory, exist_ok=True)
#     os.makedirs(blur_directory, exist_ok=True)

#     mask_files = glob(os.path.join(mask_directory, "*_3d.tif"))

#     for mask_file in tqdm(mask_files):
#         output_file = os.path.join(output_directory, os.path.basename(mask_file).replace("_3d.tif", f"_3d_filtered_{blur_thresh:.1f}.tif"))

#         # Read the segmentation stack
#         segmentation_stack = tifffile.imread(mask_file).astype(int)

#         # Read the corresponding sharpness image
#         sharpness_image_name = os.path.join(blur_directory, os.path.basename(mask_file).replace("_3d.tif", "_blur_heatmap_32_8.tif"))
#         if not os.path.exists(sharpness_image_name):
#             # Measure blur heatmap
#             img_file = os.path.join(image_directory, os.path.basename(mask_file).replace("_3d.tif", "_BF_3d.tif"))
#             sharpness_image = measure_blur_heatmap(img_file, patch_size=32, stride_size=8)

#             # Save the resized heatmap as a TIFF file
#             tifffile.imwrite(sharpness_image_name, sharpness_image.astype(np.float32))
#         else:
#             sharpness_image = tifffile.imread(sharpness_image_name)

#         # Call the tracking function
#         tracked_stack = track_3d_centers(segmentation_stack, sharpness_image=sharpness_image, blur_thresh=blur_thresh, inv=inv)

#         # Save the tracked 3D array as a new .tif file
#         tifffile.imwrite(output_file, tracked_stack)


# if __name__ == "__main__":
#     main()
#     # test_tracking()
