# ------
# This file is mostly the same as the original repo
# ------


import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.utils import AverageMeter
from src.config import get_config
from src.models import build_model
from src.data import build_loader # TODO add __init__.py in data/ for pretrain and finetune loader
from lr_scheduler import build_scheduler # TODO not implemented yet
from src.optimizer import build_optimizer
from src.logger import create_logger
from src.utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper


# we probably dont need "from apex import amp" (used for mixed precision. it's outdated, and we probably not need it for our model to work)
amp = None


def parse_option():
    parser = argparse.ArgumentParser('SimMIM pre-training script', add_help=False)
    
    parser.add_argument(
        '--cfg',
        type=str,
        required=True,
        metavar="FILE",
        help='path to config file',
    )
    parser.add_argument(
        '--opts',
        help="modify config options by adding KEY-VALUE pairs.",
        default=None,
        nargs='+',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help="batch size for single GPU",
    )
    parser.add_argument(
        '--data-path',
        type=str,
        help="path to dataset",
    )
    parser.add_argument(
        '--resume',
        help="resume from checkpoint",
    )
    parser.add_argument(
        '--accumulation-steps',
        type=int,
        help="gradient accumulation steps",
    )
    parser.add_argument(
        '--use-checkpoint',
        action='store_true',
        help="whether to use gradient checkpointing to save memory",
    )
    parser.add_argument(
        '--amp-opt-level',
        type=str,
        default='O0', # does not use mixed precision with amp by default
        choices=['O0', 'O1', 'O2'],
        help="Mixed precision opt level, if O0, no amp is used",
    )
    parser.add_argument(
        '--output',
        default='output',
        type=str,
        metavar='PATH',
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument(
        '--tag',
        help="tag of experiment",
    )
    
    # distributed training
    # (we probably won't need it since we're only using one GPU, but keeping it incase we train on multiple GPUs)
    parser.add_argument(
        '--local-rank',
        type=int,
        required=True,
        help="local rank for DistributedDataParallel",
    )
    
    args = parser.parse_args()
    config = get_config(args)
    
    return args, config


def main(config):
    data_loader_train = build_loader(config, logger, is_pretrain=True)
    
    logger.info(f"Creating model: {config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config, is_pretrain=True)
    model.cuda()
    logger.info(str(model))
    
    optimizer = build_optimizer(config, model, logger, is_pretrain=True)
    if config.AMP_OPT_LEVEL != "O0" and amp != None:
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    if distributed:
        model = torch.nn.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    
    # estimates computation cost per forward pass
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")
    
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT, logger)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f"auto resumeing from {resume_file}")
        else:
            logger.info(f"no checkpoint found in {config.OUTPUT}, ignoring auto resume")
    
    if config.MODEL.RESUME:
        load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
    
    # training loop
    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)
        
        train_one_epoch(config, model, data_loader_train, optimizer, epoch, lr_scheduler)
        if (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0) and (
            epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)
        ):
            save_checkpoint(config, epoch, model_without_ddp, 0., optimizer, lr_scheduler, logger)
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")


def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler):
    model.train()
    optimizer.zero_grad() # clear old gradients
    
    num_steps = len(data_loader) # number of batches in this epoch
    batch_time = AverageMeter()  # tracks how long batches take
    loss_meter = AverageMeter()  # tracks loss
    norm_meter = AverageMeter()  # tracks gradient norm
    
    start = time.time()
    epoch_start = time.time()
    # end = time.time() # idk why the original repo have this here
    for idx, (img, mask, targets) in enumerate(data_loader):
        # move img and mask input to GPU memory
        img = img.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        
        # forward pass
        loss = model(img, mask) # loss calculation does not need "targets" (labels) for pretraining
        
        # gradient accumulation by dividing the total batch size into "ACCUMULATION_STEPS" smaller batches
        # and updating weights after all smaller batches are done
        # (update weights after every few smaller batches)
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = loss / config.TRAIN.ACCUMULATION_STEPS # if # accumulation steps is mm then each batch should contribute only 1/mm to the loss
            # trying to use automatic mixed precision (compute gradients in lower precision training faster and uses less memory)
            # depends on Apex for AMP which has not been installed (it's outdated), so defaults to AMP_OPT_LEVEL = "O0"
            # we shouldnt need it anyway
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                
                # shrinks gradient if gradient norms are too large
                if config.TRAIN.CLIP_GRAD:
                    # grad_norm = size of the gradients
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                # backpropagation: computes gradients of the loss w.r.t. model parameters
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
                    
            # only update weights every "ACCUMULATION_STEPS" batches
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()                                    # updates weights using gradients
                optimizer.zero_grad()                               # clears gradients after the update
                lr_scheduler.step_update(epoch * num_steps + idx)   # updates learning rate according to scheduler
        else: # no gradient accumulation (update weights after every batch)
            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)
        
        torch.cuda.synchronize() # waits for the GPU to finish work before continuing
        
        loss_meter.update(loss.item(), img.size(0)) # tracks batch loss, weighted by batch size
        if torch.is_tensor(grad_norm):
            grad_norm = grad_norm.item()
        norm_meter.update(grad_norm)                # stores the latest gradient norm
        end = time.time()
        batch_time.update(end - start)
        start = end # for per-batch timing
        
        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']                                    # current learning rate
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)     # peak GPU memory used
            etas = batch_time.avg * (num_steps - idx)                               # estimate of time remaining for the epoch
            
            logger.info(
                f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t"
                f"eta {datetime.timedelta(seconds=int(etas))} lr{lr:.6f}\t"
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"loss {loss_meter.val:.4f} ({loss_meter.avg: .4f})\t"
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
                f"mem {memory_used:.0f}MB"
            )
    
    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


# setup the whole training environment if this file is directly run
if __name__ == '__main__':
    args, config = parse_option()
    
    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed. Note: make sure config.AMP_OPT_LEVEL == 'O0', from apex import amp is outdated. If we really want amp, we can use pyTorch's."
    
    # RANK is the ID of the current process (GPU), WORLD_SIZE is the total number of processes (GPUs) participating
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    distributed = (rank != -1 and world_size != -1)
    
    torch.cuda.set_device(config.LOCAL_RANK)    # choose the GPU to use (LOCAL_RANK = 1 -> GPU 1)
    if distributed:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
        dist.barrier()                 

    global_rank = dist.get_rank() if distributed else 0
    actual_world_size = dist.get_world_size() if distributed else 1
    
    # set random seeds
    global_rank = dist.get_rank() if distributed else 0
    seed = config.SEED + global_rank # dist.get_rank() so seeds are not identical among GPUs
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    
    # linear scale the learning rate according to total batch size, may not be optimal
    # 512 is the reference batch size original paper tuned the learning rate for
    original_reference_batch_size = 512.0 # unused
    # using 32 since chexpert is significantly smaller than imagenet (original paper dataset)
    # can change depending on the learning rate
    reference_batch_size = 32.0
    actual_world_size = dist.get_world_size() if distributed else 1
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * actual_world_size / reference_batch_size
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * actual_world_size / reference_batch_size
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * actual_world_size / reference_batch_size
    
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    
    # update the LR values
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR =  linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=global_rank, name=f"{config.MODEL.NAME}")
    
    if global_rank == 0: # to avoid duplicates from other GPUs if they exist
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")
        
    # print config
    logger.info(config.dump())
    
    main(config)