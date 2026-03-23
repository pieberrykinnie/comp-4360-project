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

#doing this so we can compute the gradient norm for the model parameters
#can help with detecting issues like exploding gradients during training
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

#i created this so we can save a checkpoint to disk
def save_checkpoint(save_path, model, optimizer=None, scheduler=None, epoch=None, best_metric=None, logger=None):
    checkpoint = {
        "model": model.state_dict()
    }

# only save if it exists, otherwise we can skip it
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if best_metric is not None:
        checkpoint["best_metric"] = best_metric

    # we check if folder actually exists
    save_dir = os.path.dirname(save_path)
    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)
    
    torch.save(checkpoint, save_path)

    if logger is not None:
        logger.info(f"Saved checkpoint to {save_path}")
    else:
        print(f"Saved checkpoint to {save_path}")

#this is so we can load a checkpoint from disk
def load_checkpoint(load_path, model, optimizer=None, scheduler=None, logger=None, map_location="cpu"):
    if not os.path.isfile(load_path):
        raise FileNotFoundError(f"Checkpoint file not found: {load_path}")

    checkpoint = torch.load(load_path, map_location=map_location)
    model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    epoch = checkpoint.get("epoch", -1)
    best_metric = checkpoint.get("best_metric", None)

    start_epoch = epoch + 1 if epoch >= 0 else 0

    if logger is not None:
        logger.info(f"Loaded checkpoint from {load_path} (epoch: {epoch}, best_metric: {best_metric})")
    else:
        print(f"Loaded checkpoint from {load_path} (epoch: {epoch}, best_metric: {best_metric})")

    return start_epoch, best_metric

def auto_resume_helper(output_dir):
    if not os.path.isdir(output_dir):
        return None
    checkpoint_files = []

    for file_name in os.listdir(output_dir):
        if file_name.endswith(".pth"):
            full_path = os.path.join(output_dir, file_name)
            checkpoint_files.append(full_path)
    
    if len(checkpoint_files) == 0:
        return None
    
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    return latest_checkpoint