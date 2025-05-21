from cellpose import io, models, utils, plot, transforms
import os
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
import tifffile
from tqdm import tqdm




def save_predictions(img, masks, flows, image_file=None, n=None, save_dir=None, label=None):


    """Save prediction results"""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'outlines'), exist_ok=True)

    # Save results
    base_name = image_file.stem.replace('_BF', '')
    io.imsave(os.path.join(save_dir, 'masks', f'{base_name}.tif'), masks)
    # io.imsave(os.path.join(save_dir, f'{base_name}_masks.tif'), masks)

    # Optional: Save outlines overlaid on original image
    # outlines = utils.outlines_list(masks)
    fig = plt.figure(figsize=(40,10))
    plot.show_segmentation(fig, img, masks, flows, label=label)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'outlines', f'{base_name}.tif'))
    plt.close(fig)

    # Preserve input folder structure in output
    # rel_path = Path(files[idx]).parent
    # out_path = Path(out_dir) / rel_path
    # out_path.mkdir(parents=True, exist_ok=True)

    # Image.fromarray(masks).save(os.path.join(save_dir, base_name + '_mask.tif'))


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description='Run Omnipose 3D prediction with custom parameters')
    parser.add_argument('--flow-threshold', type=float, default=0.4, help='Flow threshold value')
    # parser.add_argument('--flow-threshold', type=float, default=0, help='Flow threshold value')  # Not used for 3D
    # parser.add_argument('--flow-factor', type=float, default=10, help='Flow factor value')
    # parser.add_argument('--anisotropy', type=float, default=15, help='Anisotropy value')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    # Add argument parserflow
    args = get_args()   

    # Load the trained model
    # get_model_params(pretrained_model)
    # pretrained_model = "slurm/models/cellpose_ctyo3_brightfield_model_v1.1"
    # pretrained_model = 'models/cellpose_ctyo3_brightfield_model_v1.1'
    model_type = "cyto3"


    model = models.CellposeModel(model_type=model_type, gpu=True)

    # Directory containing images to predict
    train_dir = "data/BF+IF Experiments_2D_train_test_dataset/train"
    test_dir = "data/BF+IF Experiments_2D_train_test_dataset/test"

    save_dir = "data/BF+IF Experiments_2D_train_test_dataset/predictions_test/" + 'pretrained_cell2d_' + model_type.split('/')[-1] + f'_FlowT{args.flow_threshold:.1f}'
    os.makedirs(save_dir, exist_ok=True)

    # Get all images in the prediction directory
    image_files = sorted([f for f in Path(test_dir).glob("*_BF.tif")])

    for image_file in tqdm(image_files):
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
                                        flow_threshold=args.flow_threshold,
                                        cellprob_threshold=0.0,
                                        min_size=30)
        
        img_out = transforms.convert_image(img, channels=None, channel_axis=None, z_axis=None)
        img_out = transforms.normalize99(img_out, copy=False)
        
        save_predictions(img_out, masks, flows[0], image_file=image_file, save_dir=save_dir, label=label)
        

    print("Prediction completed. Results saved in:", save_dir)