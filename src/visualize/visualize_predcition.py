import napari
import numpy as np
import tifffile as tiff
import os
from skimage.measure import label
from skimage.morphology import closing, square, remove_small_objects

data_dir = "/Users/serenasritharan/Projects/single-cell"

'''
# (replace these with actual file paths later)
zstack_file = os.path.join(data_dir, "data/BF+IF Experiments_3D_train_test_dataset/test/p2126_J03_BF.tif")
segmentation_label_file = os.path.join(data_dir, "data/BF+IF Experiments_3D_train_test_dataset/test/p2126_J03_Cells.tif")
prediction_file = os.path.join(data_dir, "data/BF+IF Experiments_2D_train_test_dataset/predictions_test/pretrained_cell2d_cyto3_FlowT0.4/3d_view/p2126_J03_3d.tif")
blur_map_file = os.path.join(data_dir, "data/BF+IF Experiments_3D_train_test_dataset/blur_heatmaps/p2126_J03_blur_heatmap_32_8.tif")
filtered_prediction_file = os.path.join(data_dir, "data/BF+IF Experiments_2D_train_test_dataset/predictions_test/pretrained_cell2d_cyto3_FlowT0.4/3d_view_tracked_SF/p2126_J03_3d_filtered_0.5.tif")
'''

# Plate 2426
zstack_file = "data/Plate 2426_3D_train_test_dataset/test/p2426_B04_BF_3d.tif"
segmentation_label_file = "data/Plate 2426_3D_train_test_dataset/test/p2426_B04_Cells_3d.tif"
prediction_file = "data/Plate 2426_2D_train_test_dataset/predictions_test/pretrained_cell2d_cyto3_FlowT0.2/3d_view/p2426_B04_3d.tif"
blur_map_file = "data/Plate 2426_3D_train_test_dataset/blur_heatmaps/p2426_B04_blur_heatmap_32_8.tif"
filtered_prediction_file = "data/Plate 2426_2D_train_test_dataset/predictions_test/pretrained_cell2d_cyto3_FlowT0.2/3d_view_tracked_inv/p2426_B04_3d_filtered_0.5.tif"



# Load data from the specified file paths
zstack = tiff.imread(zstack_file)
segmentation_label = tiff.imread(segmentation_label_file).astype(np.uint32)
prediction = tiff.imread(prediction_file).astype(np.uint32)
blur_map = tiff.imread(blur_map_file)
filtered_prediction = tiff.imread(filtered_prediction_file).astype(np.uint32)

# Start napari viewer
viewer = napari.Viewer()

# Add layers to the viewer
viewer.add_image(zstack, name="3D Z-Stacks", colormap="gray")
viewer.add_labels(segmentation_label, name="Segmentation Label")
viewer.add_labels(prediction, name="Prediction", opacity=0.5)
viewer.add_image(blur_map, name="Blur Map", colormap="I Blue", opacity=.2)
viewer.add_labels(filtered_prediction, name="Filtered Prediction", opacity=0.5)

# Run napari
napari.run()