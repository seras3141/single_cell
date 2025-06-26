import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import os
from PIL import Image
from tqdm import tqdm
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from cellpose_omni import io, transforms, models, core
from omnipose.utils import normalize99

# set up plotting defaults
# from omnipose.plot import imshow
# omnipose.plot.setup()

# This checks to see if you have set up your GPU properly.
# CPU performance is a lot slower, but not a problem if you 
# are only processing a few images.

# This checks to see if you have set up your GPU properly.
# CPU performance is a lot slower, but not a problem if you 
# are only processing a few images.
# use_GPU = core.use_gpu()

# for plotting
mpl.rcParams['figure.dpi'] = 300
# plt.style.use('dark_background')

from cellpose_omni import io
import omnipose

class OmniposeDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = list(file_paths)
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Ensure idx is a single integer
        if isinstance(idx, (list, range)):
            return [self.__getitem__(i) for i in idx]
        
        image = io.imread(self.file_paths[idx])
        
        if len(image.shape) > 2:
            image = np.mean(image, axis=-1)
        image = normalize99(image)
        
        image = torch.from_numpy(image).float()
        image = image.unsqueeze(0)  # Add channel dimension
        
        if self.transform:
            image = self.transform(image)
        
        # Calculate slice coordinates
        # For each dimension, return (0, dimension_size)
        subs = [(0, s) for s in image.shape[1:]]  # Skip channel dimension
            
        return image, idx, subs

    @staticmethod
    def collate_fn(batch):
        # Handle case where batch is a list containing a single list of items
        if len(batch) == 1 and isinstance(batch[0], list):
            batch = batch[0]
            
        # Unzip the batch into separate lists
        images, indices, subs = zip(*batch)
        # Stack images into a single tensor
        images = torch.stack(images, dim=0)
        return images, list(indices), list(subs)

# dataloader = DataLoader(dataset, 
#                        batch_size=4,
#                        shuffle=False,
#                        num_workers=2,
#                        pin_memory=True if use_GPU else False)

# imgs = [io.imread(f) for f in files]

# print some info about the images.
# for i in imgs:
#     print('Original image shape:',i.shape)
#     print('data type:',i.dtype)
#     print('data range: min {}, max {}\n'.format(i.min(),i.max()))
# nimg = len(imgs)
# print('\nnumber of images:',nimg)


# fig = plt.figure(figsize=[40]*2,frameon=False) # initialize figure
# print('\n')
# for k in tqdm(n):
#     img = transforms.move_min_dim(imgs[k]) # move the channel dimension last
#     if len(img.shape)>2:
#         # imgs[k] = img[:,:,1] # could pick out a specific channel
#         imgs[k] = np.mean(img,axis=-1) # or just turn into grayscale 
        
#     imgs[k] = normalize99(imgs[k])
    # imgs[k] = np.pad(imgs[k],10,'edge')
    # print('new shape: ', imgs[k].shape)
    # plt.subplot(1,len(files),k+1)
    # plt.imshow(imgs[k],cmap='gray')
    # plt.axis('off')


def setup_paths(base_dir, out_dir):
    """Set up input and output directories"""
    os.makedirs(out_dir, exist_ok=True)
    files = io.get_image_files(base_dir, look_one_level_down=True)
    files = [f for f in files if not Path(f).stem.startswith(('Cells', 'Nuclei'))]
    return files

def prepare_images(files):
    """Read and preprocess images from file paths"""
    imgs = []
    for file_path in files:
        # Read image
        img = io.imread(file_path)
        
        # Convert to grayscale if needed
        if len(img.shape) > 2:
            img = np.mean(img, axis=-1)
            
        # Normalize
        img = normalize99(img)
        
        imgs.append(img)
    
    return imgs

def get_model_params(use_GPU=False):
    """Define model and its parameters"""
    model_name = 'cyto2'
    model = models.CellposeModel(gpu=use_GPU, model_type=model_name)
    
    params = {
        'channels': [0, 0],  # segment based on first channel, no second channel
        'rescale': None,     # no rescaling
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
        Image.fromarray(maski).save(out_name + '_mask.tif')

def main():
    # Configuration
    base_dir = 'data/BF+IF Experiments Labeled'
    out_dir = 'data/BF+IF Experiments Labeled_omni2d_pred'

    use_GPU = core.use_gpu()
    use_GPU = False
    
    # Setup
    files = setup_paths(base_dir, out_dir)
    n = range(10)  # Select which images to segment
    files = [files[i] for i in n]
    print(f"Processing {len(files)} files", flush=True)
    
    # Read and prepare images
    imgs = prepare_images(files)
    
    # Get model and parameters
    model, params = get_model_params(use_GPU)
    
    # Run predictions
    print("use_gpu", use_GPU, flush=True)
    tic = time.time()
    masks, flows, styles = model.eval(imgs, **params)
    net_time = time.time() - tic
    print('total segmentation time: {}s'.format(net_time))
    
    # Save results
    save_predictions(masks, flows, files, n, out_dir)

if __name__ == '__main__':
    main()
