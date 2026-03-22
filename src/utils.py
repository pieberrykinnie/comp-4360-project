import os
import torch
import torch.distributed as dist

# this function is used to check if the distributed training is initialized
def is_dist_initialized():

    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True