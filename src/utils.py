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

def reduce_tensor(tensor):
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
def save_checkpoint(config, epoch, model, max_accuracy, optimizer=None, lr_scheduler=None, logger=None):
    save_state = {
        "model": model.state_dict(),
        "epoch": epoch,
        "max_accuracy": max_accuracy,
    }

    if optimizer is not None:
        save_state["optimizer"] = optimizer.state_dict()

    if lr_scheduler is not None:
        save_state["lr_scheduler"] = lr_scheduler.state_dict()

    save_dir = config.OUTPUT
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"ckpt_epoch_{epoch}.pth")

    if logger is not None:
        logger.info(f"Saving checkpoint to {save_path}")

    torch.save(save_state, save_path)

    if logger is not None:
        logger.info(f"Checkpoint saved: {save_path}")
    else:
        print(f"Checkpoint saved: {save_path}")



#this is so we can load a checkpoint from disk
def load_checkpoint(config, model, optimizer=None, lr_scheduler=None, logger=None):

    load_path = config.MODEL.RESUME

    if not os.path.isfile(load_path):
        raise FileNotFoundError(f"Checkpoint file not found: {load_path}")

    if logger is not None:
        logger.info(f"Loading checkpoint from {load_path}")

    checkpoint = torch.load(load_path, map_location="cpu")

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if lr_scheduler is not None and "lr_scheduler" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    if "epoch" in checkpoint:
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint["epoch"] + 1
        config.freeze()

    max_accuracy = checkpoint.get("max_accuracy", 0.0)

    if logger is not None:
        logger.info(f"Loaded checkpoint successfully")
        logger.info(f"Resuming from epoch {config.TRAIN.START_EPOCH}")
        logger.info(f"Max accuracy from checkpoint: {max_accuracy}")

    del checkpoint
    torch.cuda.empty_cache()

    return max_accuracy

def auto_resume_helper(output_dir, logger=None):

    if not os.path.isdir(output_dir):
        return None

    checkpoint_files = []

    for file_name in os.listdir(output_dir):
        if file_name.endswith(".pth"):
            checkpoint_files.append(os.path.join(output_dir, file_name))

    if len(checkpoint_files) == 0:
        if logger is not None:
            logger.info(f"No checkpoints found in {output_dir}")
        return None

    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)

    if logger is not None:
        logger.info(f"Latest checkpoint found: {latest_checkpoint}")

    return latest_checkpoint

def load_pretrained(config, model, logger=None):

    load_path = config.PRETRAINED

    if not os.path.isfile(load_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {load_path}")

    if logger is not None:
        logger.info(f"Loading pretrained weights from {load_path}")

    checkpoint = torch.load(load_path, map_location="cpu")
    checkpoint_model = checkpoint["model"]

    # If the checkpoint came from SimMIM pretraining, the backbone is often under "encoder."
    if any(key.startswith("encoder.") for key in checkpoint_model.keys()):
        new_state_dict = {}
        for key, value in checkpoint_model.items():
            if key.startswith("encoder."):
                new_key = key.replace("encoder.", "", 1)
                new_state_dict[new_key] = value
        checkpoint_model = new_state_dict

        if logger is not None:
            logger.info("Removed 'encoder.' prefix from pretrained checkpoint keys")

    msg = model.load_state_dict(checkpoint_model, strict=False)

    if logger is not None:
        logger.info(msg)
        logger.info(f"Loaded pretrained weights successfully from {load_path}")

    del checkpoint
    torch.cuda.empty_cache()


def load_pretrained(config, model, logger=None):

    load_path = config.PRETRAINED

    if load_path is None or load_path == "":
        raise ValueError("config.PRETRAINED is empty. Please provide a pretrained checkpoint path.")

    if not os.path.isfile(load_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {load_path}")

    if logger is not None:
        logger.info(f"Loading pretrained checkpoint from: {load_path}")
    else:
        print(f"Loading pretrained checkpoint from: {load_path}")

    checkpoint = torch.load(load_path, map_location="cpu")

    if "model" in checkpoint:
        checkpoint_model = checkpoint["model"]
    else:
        checkpoint_model = checkpoint

    current_model_state = model.state_dict()
    filtered_state = {}

    removed_prefix_count = 0
    skipped_missing_key = []
    skipped_shape_mismatch = []

    for key, value in checkpoint_model.items():
        new_key = key

        if new_key.startswith("encoder."):
            new_key = new_key.replace("encoder.", "", 1)
            removed_prefix_count += 1

        if new_key not in current_model_state:
            skipped_missing_key.append(new_key)
            continue

        if current_model_state[new_key].shape != value.shape:
            skipped_shape_mismatch.append(
                (new_key, tuple(value.shape), tuple(current_model_state[new_key].shape))
            )
            continue

        filtered_state[new_key] = value

    load_msg = model.load_state_dict(filtered_state, strict=False)

    if logger is not None:
        logger.info(f"Loaded {len(filtered_state)} matching pretrained tensors")
        logger.info(f"Removed 'encoder.' prefix from {removed_prefix_count} keys")
        logger.info(f"Skipped {len(skipped_missing_key)} keys not found in current model")
        logger.info(f"Skipped {len(skipped_shape_mismatch)} keys with shape mismatch")
        logger.info(f"Missing keys reported by PyTorch: {load_msg.missing_keys}")
        logger.info(f"Unexpected keys reported by PyTorch: {load_msg.unexpected_keys}")

        if len(skipped_missing_key) > 0:
            logger.info(f"Example missing-key skips: {skipped_missing_key[:10]}")

        if len(skipped_shape_mismatch) > 0:
            logger.info(f"Example shape mismatches: {skipped_shape_mismatch[:10]}")
    else:
        print(f"Loaded {len(filtered_state)} matching pretrained tensors")
        print(f"Removed 'encoder.' prefix from {removed_prefix_count} keys")
        print(f"Skipped {len(skipped_missing_key)} keys not found in current model")
        print(f"Skipped {len(skipped_shape_mismatch)} keys with shape mismatch")
        print(f"Missing keys reported by PyTorch: {load_msg.missing_keys}")
        print(f"Unexpected keys reported by PyTorch: {load_msg.unexpected_keys}")

        if len(skipped_missing_key) > 0:
            print(f"Example missing-key skips: {skipped_missing_key[:10]}")

        if len(skipped_shape_mismatch) > 0:
            print(f"Example shape mismatches: {skipped_shape_mismatch[:10]}")

    del checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()