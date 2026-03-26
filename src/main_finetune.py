# ------------------------------------------
# This file is also mostly same, except for loss calculation logic.
# ------------------------------------------

import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn                       # need nn.BCEWithLogitsLoss for multi label binary classification
from timm.utils import AverageMeter

# compute AUROC (area under the receiver operating characteristic curve)
# representing probability that a model ranks a random positive example higher than a random negative example
from sklearn.metrics import roc_auc_score
from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler # not implemented yet
from optimizer import build_optimizer
from logger import create_logger
# TODO load_pretrained not implemented yet
from utils import load_checkpoint, load_pretrained, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor

# dont need amp from Apex (outdated)
# amp = None for simplicity
amp = None

def parse_option():
    parser = argparse.ArgumentParser('ViT finetuning on Chexpert script')
    
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
        '--pretrained',
        type=str,
        help="path to pretrained model",
    )
    parser.add_argument(
        '--resume',
        help="resume from checkpoint",
    )
    parser.add_argument(
        '--accumulation_steps',
        type=int,
        help="gradient accumulation steps",
    )
    parser.add_argument(
        '--use_checkpoint',
        action='store_true',
        help="whether to use gradient checkpointing to save memory",
    )
    parser.add_argument(
        '--amp-opt-level',
        type=str,
        default='00',
        choices=['00', '01', '02'],
        help="mixed precision opt level, if 00, no amp is used"
    )
    parser.add_argument(
        '--output',
        default='output',
        type=str,
        metavar='PATH',
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)"
    )
    parser.add_argument(
        '--tag',
        help="tag of experiment",
    )
    parser.add_argument(
        '--eval',
        action='store_true',
        help="perform evaluation only"
    )
    parser.add_argument(
        '--throughput',
        action='store_true',
        help='test throughput only'
    )
    parser.add_argument(
        '--local-rank',
        type=int,
        required=True,
        help="local rank for DistributedDataParallel"
    )
    
    args = parser.parse_args()
    config = get_config(args)
    
    return args, config


def main(config):
    # TODO check mixup_fn
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config, logger, is_train=False)
    logger.info(f"Creating model: {config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config, is_train=False)
    model.cuda()
    logger.info(str(model))
    
    # check this chunk works with data_finetune
    optimizer = build_optimizer(config, model, logger, is_train=False)
    if config.AMP_OPT_LEVEL == '00' and amp is not None:
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")
    
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    
    # BCEWithLogitsLoss treats each of the 14 outputs independently (an image can have multiple findings)
    criterion = nn.BCEWithLogitsLoss()
    # replaces "max_accuracy" because AUROC is our main metric for Chexpert
    best_mean_auc = 0.0
    
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT, logger)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f"auto resuming from {resume_file}")
        else:
            logger.info(f"no checkpoint found in {config.OUTPUT}, ignoring auto resume")
    
    # TODO make sure the functions match with data_finetune.py
    if config.MODEL.RESUME: # continue a previous finetuning run
        best_mean_auc = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        mean_auc, val_loss, per_class_auc = validate(config, data_loader_val, model)
        logger.info(f"Mean AUROC on the {len(dataset_val)} validation images: {mean_auc:.4f}")
        if config.EVAL_MODE:
            return
    elif config.PRETRAINED: # load weights from the pretraining stage
        load_pretrained(config, model_without_ddp, logger)
    
    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return
    
    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCH):
        data_loader_train.sampler.set_epoch(epoch)
        
        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, best_mean_auc, optimizer, lr_scheduler, logger)
            
        mean_auc, val_loss, per_class_auc = validate(config, data_loader_val, model)
        logger.info(f"Mean AUROC on the {len(dataset_val)} validation images: {mean_auc:.4f}")
        best_mean_auc = max(best_mean_auc, mean_auc)
        logger.info(f"Best mean AUROC: {best_mean_auc:.4f}")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")


def train_one_epoch(conig, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler):
    model.train()
    optimizer.zero_grad()
    
    logger.info(f"Current learning rate for different parameter groups: {[it["lr"] for it in optimizer.param_groups]}")
    
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    
    start = time.time()
    epoch_start = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.float().cuda(non_blocking=True)
        
        # for chexpert multi-label classification we dont need mixup (can use for ablation if we want)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        
        outputs = model(samples)
        
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = criterion(outputs, targets)
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if config.AMP_OPT_LEVEL != '00':
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
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != '00':
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
        
        torch.cuda.synchronize()
        
        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        end = time.time()
        batch_time.update(end - start)
        start = end
        
        if idx % config.PRINT_FREQ == 0:
            lr = optimzer.param_groups[-1]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t"
                f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t"
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
                f"mem {memory_used:.0f}MB"
            )
    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start
    logger.info(f"EPOCH {epoch} traning takes {datetime.timedelta(seconds=int(epoch_time))}")


if __name__ == '__main__':
    args, config = parse_option()
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")
