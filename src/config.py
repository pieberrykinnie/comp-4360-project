from dataclasses import dataclass
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
    BATCH_SIZE: int
    DATA_PATH: Path
    DATASET: str
    IMG_SIZE: int
    INTERPOLATION: str
    PIN_MEMORY: bool
    NUM_WORKERS: int
    MASK_PATCH_SIZE: int
    MASK_RATIO: float


@dataclass
class ConfigModelSwin:
    """
    Configuration settings specific to the Swin Transformer.

    TODO: update description of parameters
    """
    PATCH_SIZE: int
    IN_CHANS: int
    EMBED_DIM: int
    DEPTHS: list[int]
    NUM_HEADS: list[int]
    WINDOW_SIZE: int
    MLP_RATIO: float
    QKV_BIAS: bool
    QK_SCALE: float | None
    APE: bool
    PATCH_NORM: bool


@dataclass
class ConfigModelVit:
    """
    Configuration settings specific to the Vision Transformer.

    TODO: update description of parameters
    """
    PATCH_SIZE: int
    IN_CHANS: int
    EMBED_DIM: int
    DEPTH: int
    NUM_HEADS: int
    MLP_RATIO: int
    QKV_BIAS: bool
    INIT_VALUES: float
    USE_APE: bool
    USE_RPB: bool
    USE_SHARED_RPB: bool
    USE_MEAN_POOLING: bool


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
    TYPE: str
    NAME: str
    RESUME: Path
    NUM_CLASSES: int
    DROP_RATE: float
    DROP_PATH_RATE: float
    LABEL_SMOOTHING: float
    SWIN: ConfigModelSwin
    VIT: ConfigModelVit


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
    NAME: str
    DECAY_EPOCHS: int
    DECAY_RATE: float
    GAMMA: float
    MULTISTEPS: list[float]


@dataclass
class ConfigTrainOptimizer:
    """
    Configuration settings related to the optimizer used in training.

    NAME: Name of the optimization algorithm used
    EPS: Optimizer Epsilon
    BETAS: Optimizer Betas
    MOMENTUM: SGD momentum
    """
    NAME: str
    EPS: float
    BETAS: tuple[float, float]
    MOMENTUM: float


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
    START_EPOCH: int
    EPOCHS: int
    WARMUP_EPOCHS: int
    WEIGHT_DECAY: float
    BASE_LR: float
    WARMUP_LR: float
    MIN_LR: float
    CLIP_GRAD: float
    AUTO_RESUME: bool
    ACCUMULATION_STEPS: int
    USE_CHECKPOINT: bool
    LR_SCHEDULER: ConfigTrainLRScheduler
    OPTIMIZER: ConfigTrainOptimizer
    LAYER_DECAY: float


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
    COLOR_JITTER: float
    AUTO_AUGMENT: str
    REPROB: float
    REMODE: str
    RECOUNT: int
    MIXUP: float
    CUTMIX: float
    CUTMIX_MINMAX: float | None
    MIXUP_PROB: float
    MIXUP_SWITCH_PROB: float
    MIXUP_MODE: str


@dataclass
class ConfigTest:
    """
    Configuration settings related to testing.

    CROP: Whether to use center crop when testing
    """
    CROP: bool


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
    BASE: list[str]
    DATA: ConfigData
    MODEL: ConfigModel
    TRAIN: ConfigTrain
    AUG: ConfigAug
    TEST: ConfigTest
    AMP_OPT_LEVEL: str
    OUTPUT: Path
    TAG: str
    SAVE_FREQ: int
    PRINT_FREQ: int
    SEED: int
    EVAL_MODE: bool
    THROUGHPUT_MODE: bool
    LOCAL_RANK: int
    PRETRAINED: Path
