import os
import csv 
import numpy as np
from PIL import Image

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms


class MaskGenerator:
    def __init__(self, input_size, mask_patch_size, model_patch_size, mask_ratio):
        # We are validating the input parameters
        if input_size <= 0:
            raise ValueError("The input size needs to be a positive integer")
        if mask_patch_size <= 0 or model_patch_size <= 0:
            raise ValueError("Patch sizes need to be positive integers")
        if not (0.0 <= mask_ratio <= 1.0):
            raise ValueError("the mask ratio must be between 0 and 1")
        if input_size % mask_patch_size != 0:
            raise ValueError("The input size must be divisible by the mask patch size")
        if mask_patch_size % model_patch_size != 0:
            raise ValueError("The mask patch size must be divisible by the model patch size")
        
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = float(mask_ratio)

        self.mask_grid = input_size // mask_patch_size
        self.model_grid = input_size // model_patch_size
        # we are calculating the scale factor between the mask patch size and 
        # the model patch size, which will be used to determine how many model
        #  patches fit into one mask patch
        self.scale = mask_patch_size // model_patch_size

        #how many blockks exist in total
        self.num_mask_blocks = self.mask_grid * self.mask_grid
        #how many we are actually masking
        self.num_to_mask = int(round(self.num_mask_blocks * self.mask_ratio))


    #returns a token_mask where each entry is typically 0 or 1, 
    # indicating whether the corresponding patch is masked (1) or not (0).    
    def __call__(self) -> np.ndarray:
        # i make a flat array of zeros so nothing is masked yet
        coarse_flat = np.zeros(self.num_mask_blocks, dtype=int)

        # we are randomly picking what blocks we want to mask
        if self.num_to_mask > 0:
            masked_idx = np.random.choice(self.num_mask_blocks, size=self.num_to_mask, replace=False)
            coarse_flat[masked_idx] = 1

        # this makes it a 2d coarse grid, so we make it a square matrix
        coarse_mask = coarse_flat.reshape(self.mask_grid, self.mask_grid)
        # we repeat the rows and columns of the coarse mask to create a token mask
        # that matches the models patch size.
        token_mask = np.repeat(np.repeat(coarse_mask, self.scale, axis=0), self.scale, axis=1)

        return token_mask



class SimMIMDataset:

    def __init__(self, img_size: int, in_chans: int, mask_generator, train: bool = True,
        use_random_resized_crop: bool = True, use_hflip: bool = False, mean=None, std=None):

        if img_size <= 0:
            raise ValueError("Eror the image size has to be positive.")
        if in_chans not in (1, 3):
            raise ValueError("the input channels must be either 1 or 3")
        
        self.img_size = img_size
        self.in_chans = in_chans
        self.mask_generator = mask_generator
        self.train = train

        if mean is None or std is None:
            if in_chans == 1:
                mean = [0.5]
                std = [0.5]
            # i got this fromt the imagenet dataset.
            # we are only doing greyscale but i thought it would be good to have the option
            # for 3 channels as well, so i just used the imagenet mean and std.
            else:
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]