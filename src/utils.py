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

def get_grad_norm(parameters):
    total_norm = 0.0
    #we looping through all the parameters
    for p in parameters:
        if p.grad is not None:
            # we r computing the L2 norm of the gradients
            param_norm = p.grad.data.norm(2).item()
            #squeare them
            total_norm += param_norm ** 2
    #then we take the square root
    total_norm = total_norm ** 0.5
    return total_norm