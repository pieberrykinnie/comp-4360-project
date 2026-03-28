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
from src.data import build_loader
from src.lr_scheduler import build_scheduler
from src.optimizer import build_optimizer
from src.logger import create_logger
from src.utils import (
    load_checkpoint,
    load_pretrained,
    save_checkpoint,
    get_grad_norm,
    auto_resume_helper,
    reduce_tensor,
)


logger = None
distributed = False
device = None
use_cuda = False
global_rank = 0
actual_world_size = 1


def parse_option():
    parser = argparse.ArgumentParser("CheXpert fine-tuning script", add_help=False)

    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="modify config options by adding KEY-VALUE pairs",
        default=None,
        nargs="+",
    )

    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument("--data-path", type=str, help="optional dataset root override")
    parser.add_argument("--pretrained", type=str, help="path to pre-trained model")
    parser.add_argument("--resume", type=str, help="resume from checkpoint")
    parser.add_argument("--accumulation-steps", type=int, help="gradient accumulation steps")
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="whether to use gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--amp-opt-level",
        type=str,
        default="O0",
        choices=["O0", "O1", "O2"],
        help="mixed precision opt level, keep O0 for this project",
    )
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder",
    )
    parser.add_argument("--tag", type=str, help="tag of experiment")
    parser.add_argument("--eval", action="store_true", help="evaluation only")
    parser.add_argument("--throughput", action="store_true", help="throughput only")
    parser.add_argument(
        "--local-rank",
        type=int,
        default=0,
        help="local rank for DistributedDataParallel",
    )

    args = parser.parse_args()
    config = get_config(args)
    return args, config


def compute_multilabel_metrics(logits, targets, threshold=0.5):
    """
    Compute simple multi-label metrics.

    label_acc:
        element-wise label accuracy across all 14 labels

    exact_match:
        percentage of samples where every label matches
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    label_acc = (preds == targets).float().mean() * 100.0
    exact_match = (preds == targets).all(dim=1).float().mean() * 100.0

    return label_acc, exact_match


def main(config):
    global logger, distributed, device, use_cuda

    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(
        config, logger, is_pretrain=False
    )

    logger.info(f"Creating model: {config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config, is_pretrain=False)
    model.to(device)
    logger.info(str(model))

    optimizer = build_optimizer(config, model, logger, is_pretrain=False)
    model_without_ddp = model

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config.LOCAL_RANK] if use_cuda else None,
            broadcast_buffers=False,
        )
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    if hasattr(model_without_ddp, "flops"):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    # Multi-label classification loss
    criterion = torch.nn.BCEWithLogitsLoss()

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
            logger.info(f"no checkpoint found in {config.OUTPUT}, ignoring auto resume")

    if config.MODEL.RESUME:
        best_label_acc = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        val_label_acc, val_exact_match, val_loss = validate(config, data_loader_val, model, criterion)
        logger.info(
            f"Validation after resume - "
            f"label_acc: {val_label_acc:.2f}% | "
            f"exact_match: {val_exact_match:.2f}% | "
            f"loss: {val_loss:.4f}"
        )
        if config.EVAL_MODE:
            return
    elif config.PRETRAINED:
        load_pretrained(config, model_without_ddp, logger)

    if config.EVAL_MODE:
        val_label_acc, val_exact_match, val_loss = validate(config, data_loader_val, model, criterion)
        logger.info(
            f"Evaluation only - "
            f"label_acc: {val_label_acc:.2f}% | "
            f"exact_match: {val_exact_match:.2f}% | "
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

        if global_rank == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, best_label_acc, optimizer, lr_scheduler, logger)

        val_label_acc, val_exact_match, val_loss = validate(config, data_loader_val, model, criterion)

        logger.info(
            f"Validation - Epoch {epoch}: "
            f"label_acc: {val_label_acc:.2f}% | "
            f"exact_match: {val_exact_match:.2f}% | "
            f"loss: {val_loss:.4f}"
        )

        best_label_acc = max(best_label_acc, val_label_acc)
        logger.info(f"Best validation label accuracy so far: {best_label_acc:.2f}%")

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
        images = images.to(device, non_blocking=use_cuda)
        targets = targets.to(device, non_blocking=use_cuda)

        outputs = model(images)
        loss = criterion(outputs, targets)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = loss / config.TRAIN.ACCUMULATION_STEPS

        loss.backward()

        if config.TRAIN.CLIP_GRAD:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
        else:
            grad_norm = get_grad_norm(model.parameters())

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step_update(epoch * num_steps + idx)

        if use_cuda:
            torch.cuda.synchronize()

        label_acc, _ = compute_multilabel_metrics(outputs.detach(), targets)

        loss_meter.update(loss.item(), images.size(0))

        if torch.is_tensor(grad_norm):
            grad_norm = grad_norm.item()
        norm_meter.update(grad_norm)

        if torch.is_tensor(label_acc):
            label_acc = label_acc.item()
        label_acc_meter.update(label_acc, images.size(0))

        end = time.time()
        batch_time.update(end - start)
        start = end

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]["lr"]
            memory_used = 0.0
            if use_cuda:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

            etas = batch_time.avg * (num_steps - idx)

            logger.info(
                f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t"
                f"eta {datetime.timedelta(seconds=int(etas))} "
                f"lr {lr:.6f}\t"
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"label_acc {label_acc_meter.val:.2f} ({label_acc_meter.avg:.2f})\t"
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
                f"mem {memory_used:.0f}MB"
            )

    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start
    logger.info(f"EPOCH {epoch} fine-tuning takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model, criterion):
    global device, use_cuda

    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    label_acc_meter = AverageMeter()
    exact_match_meter = AverageMeter()

    end = time.time()

    for idx, (images, targets) in enumerate(data_loader):
        images = images.to(device, non_blocking=use_cuda)
        targets = targets.to(device, non_blocking=use_cuda)

        outputs = model(images)
        loss = criterion(outputs, targets)

        label_acc, exact_match = compute_multilabel_metrics(outputs, targets)

        loss = reduce_tensor(loss)
        label_acc = reduce_tensor(label_acc)
        exact_match = reduce_tensor(exact_match)

        loss_meter.update(loss.item(), targets.size(0))
        label_acc_meter.update(label_acc.item(), targets.size(0))
        exact_match_meter.update(exact_match.item(), targets.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = 0.0
            if use_cuda:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

            logger.info(
                f"Val: [{idx}/{len(data_loader)}]\t"
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"label_acc {label_acc_meter.val:.2f} ({label_acc_meter.avg:.2f})\t"
                f"exact_match {exact_match_meter.val:.2f} ({exact_match_meter.avg:.2f})\t"
                f"mem {memory_used:.0f}MB"
            )

    return label_acc_meter.avg, exact_match_meter.avg, loss_meter.avg


if __name__ == "__main__":
    args, config = parse_option()

    # This project is keeping AMP off for simplicity
    if config.AMP_OPT_LEVEL != "O0":
        raise ValueError("This fine-tuning script currently expects AMP_OPT_LEVEL = 'O0'")

    # Check whether distributed training is active
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        distributed = True
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = 0
        world_size = 1
        distributed = False
        print("Running in single-process mode")

    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{config.LOCAL_RANK}" if use_cuda else "cpu")

    if use_cuda:
        torch.cuda.set_device(config.LOCAL_RANK)

    if distributed:
        backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        dist.barrier()

    global_rank = dist.get_rank() if distributed else 0
    actual_world_size = dist.get_world_size() if distributed else 1

    seed = config.SEED + global_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Fine-tuning usually uses a smaller reference batch size than the original ImageNet setup
    reference_batch_size = 32.0
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * actual_world_size / reference_batch_size
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * actual_world_size / reference_batch_size
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * actual_world_size / reference_batch_size

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr *= config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr *= config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr *= config.TRAIN.ACCUMULATION_STEPS

    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(
        output_dir=config.OUTPUT,
        dist_rank=global_rank,
        name=f"{config.MODEL.NAME}",
    )

    if global_rank == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    logger.info(config.dump())

    main(config)