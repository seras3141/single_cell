from cellpose_omni import io, transforms, models, core
import os
from pathlib import Path
import time
import argparse
import numpy as np
from tifffile import imread, imwrite
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class CellDataset3D(Dataset):
    """Custom Dataset for 3D cell images and masks"""
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        
        # Get all image files
        self.image_files = sorted([f for f in self.image_dir.glob('*.tif')])
        self.mask_files = sorted([f for f in self.mask_dir.glob('*.tif')])
        
        assert len(self.image_files) == len(self.mask_files), "Number of images and masks must match"
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image and mask
        img = imread(self.image_files[idx])
        mask = imread(self.mask_files[idx])
        
        # Convert to torch tensors
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()
        
        # Add channel dimension if needed
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        
        # Apply transforms if any
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        
        return img, mask

def prepare_data(image_dir, mask_dir, batch_size=1, num_workers=4):
    """Prepare data loaders for training"""
    # Create dataset
    dataset = CellDataset3D(image_dir, mask_dir)
    
    # Create data loader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader

def get_args():
    parser = argparse.ArgumentParser(description='Train Omnipose 3D model')
    parser.add_argument('--image-dir', type=str, required=True, help='Directory containing training images')
    parser.add_argument('--mask-dir', type=str, required=True, help='Directory containing training masks')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--output-dir', type=str, default='models', help='Directory to save model checkpoints')
    args = parser.parse_args()
    return args

def get_model_params(use_GPU):
    """Define model and its parameters"""
    # define cellpose model
    # model_name = 'cyto2'
    model_name = None

    # this model was trained on 2D slices 
    dim = 3
    nclasses = 2 # cellpose models have no boundary field, just flow and distance 

    # Cellpose defaults to 2 channels; 
    # this is the setup for grayscale in that case
    nchan = 2

    # no rescaling for this model
    diam_mean = 0


    model = models.CellposeModel(gpu=use_GPU, model_type=model_name, net_avg=False, 
                                diam_mean=diam_mean, nclasses=nclasses, dim=dim, nchan=nchan)

    
    return model, {}


def main():
    
    use_GPU = False

    # # Run predictions
    # print("use_gpu", use_GPU, flush=True)
    # tic = time.time()
    # # masks, flows, styles = model.eval(imgs, **params)


    # masks, flows, styles = model.eval(imgs,
    #                                channels=[0,0],
    #                                rescale=None,
    #                                net_avg=0,
    #                                transparency=True, 
    #                                verbose=0, 
    #                                tile=0,
    #                                compute_masks=1, 
    #                                do_3D=True, 
    #                                omni=True,
    #                                mask_threshold=args.mask_threshold,
    #                                flow_threshold=args.flow_threshold,
    #                                flow_factor=args.flow_factor,
    #                                anisotropy=args.anisotropy)

    # net_time = time.time() - tic
    # print('total segmentation time: {}s'.format(net_time))
    

    io.logger_setup()

    # Add argument parser
    # args = get_args()

    # Set paths for training and test data
    train_dir = "data/BF+IF Experiments_3D_train_test_dataset/train"
    test_dir = "data/BF+IF Experiments_3D_train_test_dataset/test"

    # Load training and test data
    output = io.load_train_test_data(train_dir, test_dir, 
                                    image_filter="_BF", # Filter for brightfield images
                                    mask_filter="_Cells", # Filter for mask images
                                    look_one_level_down=True)

    images, labels, links, image_names, test_images, test_labels, test_links, image_names_test = output

    # Get model and parameters
    model, params = get_model_params(use_GPU)


    # Train the model
    model_path = model.train(
        train_data=images, train_labels=labels,
        test_data=test_images, test_labels=test_labels,
        train_links=links, test_links=test_links,
        channels=[0,0], normalize=True,
        save_path="cellpose_3d_brightfield_model",
        weight_decay=1e-4, SGD=True, learning_rate=0.1,
        n_epochs=100, 
        rescale=False
        )



if __name__ == '__main__':
    main() 