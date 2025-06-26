from cellpose import io, models, utils, plot, transforms
import os
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
import tifffile

def save_predictions(img, masks, flows, image_file=None, n=None, save_dir=None, label=None):


    """Save prediction results"""

    # Save results
    base_name = image_file.stem.replace('_BF', '')
    io.imsave(os.path.join(save_dir, f'{base_name}_masks.tif'), masks)

    # Optional: Save outlines overlaid on original image
    # outlines = utils.outlines_list(masks)
    fig = plt.figure(figsize=(40,10))
    plot.show_segmentation(fig, img, masks, flows, label=label)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{base_name}_outlines.tif'))


    # Preserve input folder structure in output
    # rel_path = Path(files[idx]).parent
    # out_path = Path(out_dir) / rel_path
    # out_path.mkdir(parents=True, exist_ok=True)

    Image.fromarray(masks).save(os.path.join(save_dir, base_name + '_mask.tif'))



# Load the trained model
# get_model_params(pretrained_model)
pretrained_model = 'models/cellpose_ctyo3_brightfield_model_v1.1'
model = models.CellposeModel(pretrained_model=pretrained_model, gpu=False)

# Directory containing images to predict
train_dir = "data/BF+IF Experiments_2D_train_test_dataset/train"
test_dir = "data/BF+IF Experiments_2D_train_test_dataset/test"

save_dir = "data/BF+IF Experiments_2D_train_test_dataset/predictions_2D_" + pretrained_model.split('/')[-1] + '_FlowT5.0'
os.makedirs(save_dir, exist_ok=True)

# Get all images in the prediction directory
image_files = sorted([f for f in Path(test_dir).glob("*_BF.tif")])

for image_file in image_files:
    # Load image
    # img = io.imread(str(image_file))
    img = tifffile.imread(str(image_file))
    label = tifffile.imread(str(image_file).replace("_BF.tif", "_Cells.tif"))
    
    # Normalize
    # img = transforms.normalize_img(img)

    # Run prediction
    masks, flows, styles = model.eval(img,
                                    channels=[0,0],  # Single grayscale channel
                                    diameter=None,    # Auto-diameter
                                    normalize=True,
                                    invert=False,
                                    flow_threshold=5.0,
                                    cellprob_threshold=0.0,
                                    min_size=30)
    
    img_out = transforms.convert_image(img, channels=None, channel_axis=None, z_axis=None)
    img_out = transforms.normalize99(img_out, copy=False)
    
    save_predictions(img_out, masks, flows[0], image_file=image_file, save_dir=save_dir, label=label)
    

print("Prediction completed. Results saved in:", save_dir)