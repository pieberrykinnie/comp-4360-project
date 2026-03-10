from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ConfigData:
    """
    Configuration settings related to the data pipeline.

    BATCH_SIZE: Batch size for a single GPU, could be overwritten by command line argument
    DATA_PATH: Path to dataset, could be overwritten by command line argument
    DATASET: Dataset name
    IMG_SIZE: Input image size
    INTERPOLATION: Interpolation to resize image (random, bilinear, bicubic)
    PIN_MEMORY: Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
    NUM_WORKERS: Number of data loading threads
    MASK_PATCH_SIZE: [SimMIM] Mask patch size for MaskGenerator
    MASK_RATIO: [SimMIM] Mask ratio for MaskGenerator
    """
    BATCH_SIZE: int = 128
    DATA_PATH: Path = ""
    DATASET: str = "imagenet"
    IMG_SIZE: int = 224
    INTERPOLATION: str = "bicubic"
    PIN_MEMORY: bool = True
    NUM_WORKERS: int = 8
    MASK_PATCH_SIZE: int = 32
    MASK_RATIO: float = 0.6


@dataclass
class ConfigModelSwin:
    """
    Configuration settings specific to the Swin Transformer.

    TODO: update description of parameters
    """
    PATCH_SIZE: int = 4
    IN_CHANS: int = 3
    EMBED_DIM: int = 96
    DEPTHS: list[int] = field(default_factory=lambda: [2, 2, 6, 2])
    NUM_HEADS: list[int] = field(default_factory=lambda: [3, 6, 12, 24])
    WINDOW_SIZE: int = 7
    MLP_RATIO: float = 4.
    QKV_BIAS: bool = True
    QK_SCALE: float | None = None
    APE: bool = False
    PATCH_NORM: bool = True


@dataclass
class ConfigModelVit:
    """
    Configuration settings specific to the Vision Transformer.

    TODO: update description of parameters
    """
    PATCH_SIZE: int = 16
    IN_CHANS: int = 3
    EMBED_DIM: int = 768
    DEPTH: int = 12
    NUM_HEADS: int = 12
    MLP_RATIO: int = 4
    QKV_BIAS: bool = True
    INIT_VALUES: float = 0.1
    USE_APE: bool = False
    USE_RPB: bool = False
    USE_SHARED_RPB: bool = True
    USE_MEAN_POOLING: bool = False


@dataclass
class ConfigModel:
    """
    Configuration settings related to the model in general.

    TYPE: Model type
    NAME: Model name
    RESUME: Checkpoint to resume, could be overwritten by command line argument
    NUM_CLASSES: Number of classes, overwritten in data preparation
    DROP_RATE: Dropout rate
    DROP_PATH_RATE: Drop path rate
    LABEL_SMOOTHING: Label Smoothing
    SWIN: Swin Transformer parameters
    VIT: Vision Transformer parameters
    """
    TYPE: str = "swin"
    NAME: str = "swin_tiny_patch4_window7_224"
    RESUME: Path = ""
    NUM_CLASSES: int = 1000
    DROP_RATE: float = 0.0
    DROP_PATH_RATE: float = 0.1
    LABEL_SMOOTHING: float = 0.1
    SWIN: ConfigModelSwin = field(default_factory=ConfigModelSwin)
    VIT: ConfigModelVit = field(default_factory=ConfigModelVit)


@dataclass
class ConfigTrainLRScheduler:
    """
    Configuration settings related to the learning rate scheduler.

    NAME: Name of the scheduler used
    DECAY_EPOCHS: Epoch interval to decay LR, used in StepLRScheduler
    DECAY_RATE: LR decay rate, used in StepLRScheduler
    GAMMA: Gamma value, used in MultiStepLRScheduler
    MULTISTEPS: Multi steps value, used in MultiStepLRScheduler
    """
    NAME: str = "cosine"
    DECAY_EPOCHS: int = 30
    DECAY_RATE: float = 0.1
    GAMMA: float = 0.1
    MULTISTEPS: list[float] = field(default_factory=list)


@dataclass
class ConfigTrainOptimizer:
    """
    Configuration settings related to the optimizer used in training.

    NAME: Name of the optimization algorithm used
    EPS: Optimizer Epsilon
    BETAS: Optimizer Betas
    MOMENTUM: SGD momentum
    """
    NAME: str = "adamw"
    EPS: float = 1e-8
    BETAS: tuple[float, float] = field(default_factory=lambda: (0.9, 0.999))
    MOMENTUM: float = 0.9


@dataclass
class ConfigTrain:
    """
    Configuration settings related to the training process.

    START_EPOCH: The starting epoch for training
    EPOCHS: Total number of epochs to train for
    WARMUP_EPOCHS: Number of initial epochs for warmup
    WEIGHT_DECAY: Weight decay
    BASE_LR: Base learning rate
    WARMUP_LR: Learning rate for warmup epochs
    MIN_LR: Minimal learning rate
    CLIP_GRAD: Clip gradient norm
    AUTO_RESUME: Auto resume from latest checkpoint
    ACCUMULATION_STEPS: Gradient accumulation steps, could be overwritten by command line argument
    USE_CHECKPOINT: Whether to use gradient checkpointing to save memory, could be overwritten by command line argument
    LR_SCHEDULER: LR scheduler configuration
    OPTIMIZER: Optimizer configuration
    LAYER_DECAY: [SimMIM] Layer decay for fine-tuning
    """
    START_EPOCH: int = 0
    EPOCHS: int = 300
    WARMUP_EPOCHS: int = 20
    WEIGHT_DECAY: float = 0.05
    BASE_LR: float = 5e-4
    WARMUP_LR: float = 5e-7
    MIN_LR: float = 5e-6
    CLIP_GRAD: float = 5.0
    AUTO_RESUME: bool = True
    ACCUMULATION_STEPS: int = 0
    USE_CHECKPOINT: bool = False
    LR_SCHEDULER: ConfigTrainLRScheduler = field(
        default_factory=ConfigTrainLRScheduler)
    OPTIMIZER: ConfigTrainOptimizer = field(
        default_factory=ConfigTrainOptimizer)
    LAYER_DECAY: float = 1.0


@dataclass
class ConfigAug:
    """
    Configuration settings related to data augmentation.

    COLOR_JITTER: Color jitter factor
    AUTO_AUGMENT: Use AutoAugment policy. "v0" or "original"
    REPROB: Random erase prob
    REMODE: Random erase mode
    RECOUNT: Random erase count
    MIXUP: Mixup alpha, mixup enabled if > 0
    CUTMIX: Cutmix alpha, cutmix enabled if > 0
    CUTMIX_MINMAX: Cutmix min/max ratio, overrides alpha and enables cutmix if set
    MIXUP_PROB: Probability of performing mixup or cutmix when either/both is enabled
    MIXUP_SWITCH_PROB: Probability of switching to cutmix when both mixup and cutmix enabled
    MIXUP_MODE: How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
    """
    COLOR_JITTER: float = 0.4
    AUTO_AUGMENT: str = "rand-m9-mstd0.5-inc1"
    REPROB: float = 0.25
    REMODE: str = "pixel"
    RECOUNT: int = 1
    MIXUP: float = 0.8
    CUTMIX: float = 1.0
    CUTMIX_MINMAX: float | None = None
    MIXUP_PROB: float = 1.0
    MIXUP_SWITCH_PROB: float = 0.5
    MIXUP_MODE: str = "batch"


@dataclass
class ConfigTest:
    """
    Configuration settings related to testing.

    CROP: Whether to use center crop when testing
    """
    CROP: bool = True


@dataclass
class Config:
    """
    Configuration settings. May be overridden using the command line.

    BASE: Base config files
    DATA: Data settings
    MODEL: Model settings
    TRAIN: Training settings
    AUG: Augmentation settings
    TEST: Testing settings
    AMP_OPT_LEVEL: Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2'), overwritten by command line argument
    OUTPUT: Path to output folder, overwritten by command line argument
    TAG: Tag of experiment, overwritten by command line argument
    SAVE_FREQ: Frequency to save checkpoint
    PRINT_FREQ: Frequency to logging info
    SEED: Fixed random seed
    EVAL_MODE: Perform evaluation only, overwritten by command line argument
    THROUGHPUT_MODE: Test throughput only, overwritten by command line argument
    LOCAL_RANK: Local rank for DistributedDataParallel, given by command line argument
    PRETRAINED: [SimMIM] path to pre-trained model
    """
    BASE: list[str] = field(default_factory=lambda: [""])
    DATA: ConfigData = field(default_factory=ConfigData)
    MODEL: ConfigModel = field(default_factory=ConfigModel)
    TRAIN: ConfigTrain = field(default_factory=ConfigTrain)
    AUG: ConfigAug = field(default_factory=ConfigAug)
    TEST: ConfigTest = field(default_factory=ConfigTest)
    AMP_OPT_LEVEL: str = ""
    OUTPUT: Path = ""
    TAG: str = "default"
    SAVE_FREQ: int = 1
    PRINT_FREQ: int = 10
    SEED: int = 0
    EVAL_MODE: bool = False
    THROUGHPUT_MODE: bool = False
    LOCAL_RANK: int = 0
    PRETRAINED: Path = ""
