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