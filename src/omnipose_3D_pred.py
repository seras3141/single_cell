from cellpose_omni import io, transforms, models, core
import os
from pathlib import Path
import time
from PIL import Image
from tifffile import imwrite
import argparse

def prepare_images(files):
    """Read and preprocess images from file paths"""
    imgs = []
    for file_path in files:
        # Read image
        img = io.imread(file_path)
        
        # # Normalize
        # img = normalize99(img)
        
        imgs.append(img)
    
    return imgs

def setup_paths(base_dir, out_dir):
    """Set up input and output directories"""
    os.makedirs(out_dir, exist_ok=True)
    files = io.get_image_files(base_dir, look_one_level_down=True)
    files = [f for f in files if not Path(f).stem.startswith(('Cells', 'Nuclei'))]
    return files

def get_model_params(use_GPU=False):
    """Define model and its parameters"""
    # define cellpose model
    model_name = 'cyto2'

    # this model was trained on 2D slices 
    dim = 2
    nclasses = 2 # cellpose models have no boundary field, just flow and distance 

    # Cellpose defaults to 2 channels; 
    # this is the setup for grayscale in that case
    nchan = 2

    # no rescaling for this model
    diam_mean = 0


    model = models.CellposeModel(gpu=use_GPU, model_type=model_name, net_avg=False, 
                                diam_mean=diam_mean, nclasses=nclasses, dim=dim, nchan=nchan)
    

    # segmentation parameters 
    # chans = [0,0]
    # omni = 1
    # rescale = False
    # mask_threshold = 0 
    # net_avg = 0
    # verbose = 1
    # tile = 0
    # compute_masks = 1
    # rescale = None
    # flow_threshold=0.
    # do_3D=True
    # flow_factor=10

    '''
    params = {
        'channels': [0,0],  # segment based on first channel, no second channel
        'rescale': False,     # no rescaling
        'mask_threshold': 0,
        'flow_threshold': 0,
        'transparency': True,
        'omni': True,
        'cluster': True,
        'resample': False,
        'verbose': False,
        'tile': False,
        'niter': None,
        'augment': False,
        'affinity_seg': False,
        'net_avg': False,
        'do_3D': True,
        'flow_factor': 10,
        'z_axis': 0,
    }
    '''

    params = {
        'mask_threshold': 0,
        'flow_threshold': 0,
        'cluster': True,
        'resample': False,
        'niter': None,
        'augment': False,
        'affinity_seg': False,
        'flow_factor': 10,
        'z_axis': 0,
    }

    return model, params

def save_predictions(masks, flows, files, n, out_dir):
    """Save prediction results"""
    for idx, i in enumerate(n):
        maski = masks[idx]
        
        # Preserve input folder structure in output
        rel_path = Path(files[i]).parent
        out_path = Path(out_dir) / rel_path
        out_path.mkdir(parents=True, exist_ok=True)
        
        out_name = str(out_path / Path(files[i]).stem)
        imwrite(out_name + '_mask.tif', maski, imagej=True)

def get_args():
    parser = argparse.ArgumentParser(description='Run Omnipose 3D prediction with custom parameters')
    parser.add_argument('--mask-threshold', type=float, default=0, help='Mask threshold value')
    parser.add_argument('--flow-threshold', type=float, default=0, help='Flow threshold value')
    parser.add_argument('--flow-factor', type=float, default=10, help='Flow factor value')
    parser.add_argument('--anisotropy', type=float, default=10, help='Anisotropy value')
    args = parser.parse_args()

    return args


def main():
    # Add argument parser
    args = get_args()

    # Configuration
    base_dir = 'data/BF+IF Experiments Labeled_3D'
    out_dir = f'data/BF+IF Experiments Labeled_3D/pretrained_omni3d_pred/maskT{args.mask_threshold}_flowT{args.flow_threshold}_factor{args.flow_factor}_aniso{args.anisotropy}'

    use_GPU = core.use_gpu()
    use_GPU = False
    
    # Setup
    files = setup_paths(base_dir, out_dir)
    n = [0]  # Select which images to segment
    files = [files[i] for i in n]
    print(f"Processing {len(files)} files", flush=True)
    
    # Read and prepare images
    imgs = prepare_images(files)
    
    # Get model and parameters
    model, params = get_model_params(use_GPU)
    
    # Run predictions
    print("use_gpu", use_GPU, flush=True)
    tic = time.time()
    # masks, flows, styles = model.eval(imgs, **params)

    masks, flows, styles = model.eval(imgs,
                                   channels=[0,0],
                                   rescale=None,
                                   net_avg=0,
                                   transparency=True, 
                                   verbose=0, 
                                   tile=0,
                                   compute_masks=1, 
                                   do_3D=True, 
                                   omni=True,
                                   mask_threshold=args.mask_threshold,
                                   flow_threshold=args.flow_threshold,
                                   flow_factor=args.flow_factor,
                                   anisotropy=args.anisotropy)

    net_time = time.time() - tic
    print('total segmentation time: {}s'.format(net_time))
    
    # Save results
    save_predictions(masks, flows, files, n, out_dir)

if __name__ == '__main__':
    main()


