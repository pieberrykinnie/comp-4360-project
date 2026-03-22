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

def reduce_tebsore(tensor):
    if not is_dist_initialized():
        return tensor

    reduced_tensor = tensor.clone()
    # here i am adding together the tensor vals
    dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)
    #then do the avg
    reduced_tensor /= dist.get_world_size()
    return reduced_tensor