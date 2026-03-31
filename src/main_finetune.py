# ------------------------------------------
# This file is also mostly same, except for loss calculation logic.
# ------------------------------------------

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import (
    load_checkpoint,
    load_pretrained,
    save_checkpoint,
    get_grad_norm,
    auto_resume_helper,
    reduce_tensor,
)
from src.logger import create_logger
from src.optimizer import build_optimizer
from src.lr_scheduler import build_scheduler
from src.data import build_loader
from src.models import build_model
from src.config import get_config

import time
import argparse
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
# need nn.BCEWithLogitsLoss for multi label binary classification
import torch.nn as nn
from timm.utils import AverageMeter

# compute AUROC (area under the receiver operating characteristic curve)
# representing probability that a model ranks a random positive example higher than a random negative example
from sklearn.metrics import roc_auc_score

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
        default='O0',
        choices=['O0', 'O1', 'O2'],
        help="mixed precision opt level, if O0, no amp is used"
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
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(
        config, logger, is_pretrain=False)
    logger.info(f"Creating model: {config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config, is_pretrain=False)
    model.cuda()
    logger.info(str(model))

    # check this chunk works with data_finetune
    optimizer = build_optimizer(config, model, logger, is_pretrain=False)
    if config.AMP_OPT_LEVEL == 'O0' and amp is not None:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    # BCEWithLogitsLoss treats each of the 14 outputs independently (an image can have multiple findings)
    criterion = nn.BCEWithLogitsLoss()
    # replaces "max_accuracy" because AUROC is our main metric for Chexpert
    best_label_acc = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT, logger)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}"
                )
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f"auto resuming from {resume_file}")
        else:
            logger.info(
                f"no checkpoint found in {config.OUTPUT}, ignoring auto resume")

    if config.MODEL.RESUME:
        best_label_acc = load_checkpoint(
            config, model_without_ddp, optimizer, lr_scheduler, logger)
        val_mean_auc, val_loss, val_per_class_auc = validate(
            config, data_loader_val, model)

        logger.info(
            f"Validation after resume -  "
            f"mean_auc: {val_mean_auc:.4f} | "
            f"loss: {val_loss:.4f}"
        )
        if config.EVAL_MODE:
            return
    elif config.PRETRAINED:
        load_pretrained(config, model_without_ddp, logger)

    if config.EVAL_MODE:
        val_mean_auc, val_loss, val_per_class_auc = validate(
            config, data_loader_val, model)
        logger.info(
            f"Evaluation only - "
            f"mean_auc: {val_mean_auc:.4f} | "
            f"loss: {val_loss:.4f}"
        )
        return

    logger.info("Start fine-tuning")
    start_time = time.time()

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            config=config,
            model=model,
            criterion=criterion,
            data_loader=data_loader_train,
            optimizer=optimizer,
            epoch=epoch,
            lr_scheduler=lr_scheduler,
        )

        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp,
                            best_label_acc, optimizer, lr_scheduler, logger)

        val_mean_auc, val_loss, val_per_class_auc = validate(
            config, data_loader_val, model)

        logger.info(
            f"Validation - Epoch {epoch}: "
            f"mean_auc: {val_mean_auc:.4f} | "
            f"loss: {val_loss:.4f}"
        )

        best_label_acc = max(best_label_acc, max(val_per_class_auc))
        logger.info(
            f"Best validation label accuracy so far: {best_label_acc:.2f}%")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Fine-tuning time {total_time_str}")


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, lr_scheduler):
    global logger, device, use_cuda

    model.train()
    optimizer.zero_grad()

    logger.info(
        f"Current learning rate(s): {[group['lr'] for group in optimizer.param_groups]}"
    )

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    label_acc_meter = AverageMeter()

    start = time.time()
    epoch_start = time.time()

    for idx, (images, targets) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        targets = targets.float().cuda(non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            loss.backward()

            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())

            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != 'O0':
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.TRAIN.CLIP_GRAD)
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
            lr = optimizer.param_groups[-1]['lr']
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
    logger.info(
        f"EPOCH {epoch} traning takes {datetime.timedelta(seconds=int(epoch_time))}")


# gathers tensors from all GPUs and concatenates them
def gather_tensor(tensor):
    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


# Warning: probs/targets (322-323) stay on local GPU so, currently AUROC here is per local GPU, not global AUROC
# measure how well the fine tuned chexpert model is doing
# for efficiency, validate doesnt track gradients since there is no back propagation
@torch.no_grad()
def validate(config, data_loader, model):
    # BCEWithLogitsLoss combines sigmoid and binary cross entropy
    # (expects raw logits from the model)
    # loss used for multi label classification in checpert
    criterion = nn.BCEWithLogitsLoss()
    model.eval()

    # batch_time.val stores time for current batch
    # batch_time.avg stores average time so far
    batch_time = AverageMeter()
    # loss_meter.val stores the current reduced loss
    # loss_meter.avg stores the average validation loss across all samples
    loss_meter = AverageMeter()

    all_probs = []      # collect probabilities/predictions from all batches
    all_targets = []    # collect true labels from all batches

    start = time.time()
    # idx is the batch index
    # each batch contains input chest xray images, and multilabel vector for each image (0,1 classificatin for all 14 pathologies)
    # target might look like this: [1,0,0,1,1,0,0,0,0,0,0,0,0,0] (14D vector for 14 classes/pathologies)
    for idx, (images, targets) in enumerate(data_loader):
        # move data to GPU memory
        images = images.cuda(non_blocking=True)
        targets = targets.float().cuda(non_blocking=True)
        # logits are raw scores for the 14 classifications (for 1 image) before sigmoid
        logits = model(images)
        # our loss, compares the predicted logits by the model with the true multilabel targets
        loss = criterion(logits, targets)

        # if validation is running on multiple GPUs, each GPU computes its own batch loss
        # reduced_loss is the average loss across all GPUs for a batch
        reduced_loss = reduce_tensor(loss)
        # update loss average (weighted by number of samples in the batch)
        loss_meter.update(reduced_loss.item(), targets.size(0))

        # convert logits to probabilities (between 0 and 1, inclusive) for AUROC
        probs = torch.sigmoid(logits)
        probs = gather_tensor(probs)
        targets = gather_tensor(targets)
        # store predicted probabilities and true labels for every batch
        # move back to CPU (for AUROC/scikit-learn)
        all_probs.append(probs.cpu())
        all_targets.append(targets.cpu())

        end = time.time()
        batch_time.update(end - start)
        start = end
        # logs every "PRINT_FREQ" batches
        if idx % config.PRINT_FREQ == 0:
            # converts max allocated Bytes to MB
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} (avg {batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} (avg {loss_meter.avg:.4f})\t'
                f'Mem {memory_used:.0f}MB'
            )
    # list of tensors, one tensor per batch
    # concatenate all stored tensors from each batch to make one big tensor (number of examples, 14 classes) for the whole dataset
    # then convert to numpy arrays (for scikit-learn)
    # so now we have one big np array of probabilities, and one of true labels
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    per_class_auc = []                      # store AUROC for each pathology
    for c in range(all_targets.shape[1]):   # loop over all 14 classes
        # extract one class (one column == one pathology) at a time
        # true label for that class/pathology across the whole set
        y_true = all_targets[:, c]
        # predicted probabilities for that class across the whole set
        y_score = all_probs[:, c]
        # roc_auc_score needs both positive and negative examples, otherwise just stores nan for that class
        if len(np.unique(y_true)) < 2:
            per_class_auc.append(float("nan"))
        else:
            # for one class/pathology, roc_auc_score measures how well the model
            # ranks positive (contains pathology) images higher than negative images
            # its basically the % of correct rankings, where correct means disease image
            # has lower probability than non disease image
            per_class_auc.append(roc_auc_score(y_true, y_score))
    # averages the AUROC per class (ignores class with nan)
    # mean AUROC = overall model performance across all diseases (high means model is good overall)
    mean_auc = float(np.nanmean(per_class_auc))

    logger.info(f'Mean AUROC {mean_auc:.4f}')
    logger.info(f'Per-class AUROC {per_class_auc}')

    return mean_auc, loss_meter.avg, per_class_auc


# measures inference speed
@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, targets) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):     # warmup
            model(images)
        torch.cuda.synchronize()    # waits for GPU to finish work
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):     # time this model run (whole 30 times)
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(
            f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


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

    # Read LOCAL_RANK from the environment (populated by torchrun)
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        config.defrost()
        config.LOCAL_RANK = local_rank
        config.freeze()

    # choose the GPU to use (LOCAL_RANK = 1 -> GPU 1)
    torch.cuda.set_device(config.LOCAL_RANK)
    # helps distributed GPUs talk to each other (we will probably just use one GPU anyway)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    # wait here until all GPUs/processes are synchronized
    torch.distributed.barrier()

    # set random seeds
    # dist.get_rank() so seeds are not identical among GPUs
    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    # 512 is the reference batch size original paper tuned the learning rate for
    # not using (from original paper, trained on imagenet)
    original_reference_batch_size = 512.0
    # using 32 since chexpert is significantly smaller than imagenet (original paper dataset)
    # can change depending on the learning rate
    reference_batch_size = 32.0
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * \
        dist.get_world_size() / reference_batch_size
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * \
        config.DATA.BATCH_SIZE * dist.get_world_size() / reference_batch_size
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * \
        dist.get_world_size() / reference_batch_size

    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS

    # update the LR values
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT,
                           dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:  # to avoid duplicates from other GPUs if they exist
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
