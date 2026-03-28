from bisect import bisect_right
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
    multi_steps: list[int] = [
        i * n_iter_per_epoch for i in config.TRAIN.LR_SCHEDULER.MULTISTEPS
    ]

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
    elif scheduler_name == "linear":
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min_rate=0.01,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
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
    elif scheduler_name == "multistep":
        lr_scheduler = MultiStepLRScheduler(
            optimizer,
            milestones=multi_steps,
            gamma=config.TRAIN.LR_SCHEDULER.GAMMA,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    else:
        raise NotImplementedError(
            f"Learning rate scheduler {scheduler_name} not implemented!"
        )

    return lr_scheduler


class LinearLRScheduler(Scheduler):
    t_initial: int
    lr_min_rate: float
    warmup_t: int
    warmup_lr_init: float
    t_in_epochs: bool
    warmup_steps: list[float]

    def __init__(
        self,
        optimizer: Optimizer,
        t_initial: int,
        lr_min_rate: float,
        warmup_t: int = 0,
        warmup_lr_init: float = 0.,
        t_in_epochs: bool = True,
        noise_range_t: list[int] | tuple[int, int] | int | None = None,
        noise_pct: float = 0.67,
        noise_std: float = 1.0,
        noise_seed: int = 42,
        initialize: bool = True,
    ) -> None:
        super().__init__(
            optimizer,
            param_group_field="lr",
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize
        )

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs

        if self.warmup_t != 0:
            self.warmup_steps = [(v - warmup_lr_init) /
                                 self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t: int) -> list[float]:
        lrs: list[float]

        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t: int = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t))
                   for v in self.base_values]

        return lrs

    def get_epoch_values(self, epoch: int) -> list[float] | None:
        return self._get_lr(epoch) if self.t_in_epochs else None

    def get_update_values(self, num_updates: int) -> list[float] | None:
        return self._get_lr(num_updates) if not self.t_in_epochs else None


class MultiStepLRScheduler(Scheduler):
    milestones: list[int]
    gamma: float
    warmup_t: int
    warmup_lr_init: float
    t_in_epochs: bool
    warmup_steps: list[float]

    def __init__(
        self,
        optimizer: Optimizer,
        milestones,
        gamma: float = 0.1,
        warmup_t: int = 0,
        warmup_lr_init: float = 0.,
        t_in_epochs: bool = True,
    ) -> None:
        super().__init__(optimizer, param_group_field="lr")

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs

        if self.warmup_t != 0:
            self.warmup_steps = [(v - warmup_lr_init) /
                                 warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

        assert (self.warmup_t <= min(self.milestones))

    def _get_lr(self, t: int) -> list[float]:
        lrs: list[float]

        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            lrs = [v * (self.gamma ** bisect_right(self.milestones, t))
                   for v in self.base_values]

        return lrs

    def get_epoch_values(self, epoch: int) -> list[float] | None:
        return self._get_lr(epoch) if self.t_in_epochs else None

    def get_update_values(self, num_updates: int) -> list[float] | None:
        return self._get_lr(num_updates) if not self.t_in_epochs else None
