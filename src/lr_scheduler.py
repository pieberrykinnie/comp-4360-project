from collections import Counter
from bisect import bisect_right

import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler

from config import Config
from torch.optim import Optimizer


def build_scheduler(config: Config, optimizer: Optimizer, n_iter_per_epoch: int) -> Scheduler:
    num_steps: int = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    warmup_steps: int = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)
    decay_steps: int = int(
        config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS * n_iter_per_epoch
    )

    lr_scheduler: Scheduler
    scheduler_name: str = config.TRAIN.LR_SCHEDULER.NAME

    if scheduler_name == "cosine":
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            cycle_mul=1.,
            lr_min=config.TRAIN.MIN_LR,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    elif scheduler_name == "step":
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    else:
        raise NotImplementedError(
            f"Learning rate scheduler {scheduler_name} not implemented!"
        )

    return lr_scheduler
