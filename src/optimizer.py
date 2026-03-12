from logging import Logger
from torch import optim
from torch.nn import Module
from torch.nn.parameter import Parameter


def build_optimizer(config, model: Module, logger: Logger, is_pretrain: bool) -> optim.Optimizer:
    """
    Interface to return the appropriate optimizer.

    Args:
        config: The configuration class being used.
        model: The model being used.
        logger: The logger.
        is_pretrain: Whether the optimizer is for the pretraining or finetuning process.

    Return: The optimizer object.
    """
    if is_pretrain:
        return build_pretrain_optimizer(config, model, logger)
    else:
        raise NotImplementedError(
            "Finetuning optimizer not currently implemented."
        )


def build_pretrain_optimizer(config, model: Module, logger: Logger) -> optim.Optimizer:
    """
    Build an optimizer for the pre-training process.

    Args:
        config: The configuration class being used.
        model: The model being used.
        logger: The logger.

    Return: The optimizer object.
    """
    logger.info("> Building optimizer for Pre-training stage...")

    # Get model's parameters that will not be penalized during weight decay
    skip: set[str] = {}
    skip_keywords: set[str] = {}

    if hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()
        logger.info(f"No weight decay: {skip}")
    if hasattr(model, "no_weight_decay_keywords"):
        skip_keywords = model.no_weight_decay_keywords()
        logger.info(f"No weight decay keywords: {skip_keywords}")

    # Get model's trainable parameters and which to remove weight decay for
    parameters: list[dict[str, Parameter | float]] = get_pretrain_param_groups(
        model, logger, skip, skip_keywords
    )

    # Get configured optimizer
    opt_lower: str = config.TRAIN.OPTIMIZER.NAME.lower()

    optimizer: optim.Optimizer

    if opt_lower == "sgd":
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
                              nesterov=True, lr=config.TRAIN.BASE_LR,
                              weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS,
                                betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR,
                                weight_decay=config.TRAIN.WEIGHT_DECAY)
    # OG implementation just let the function return None; anti-pattern removed
    else:
        raise NotImplementedError(f"Optimizer {opt_lower} is not supported!")

    logger.info(f"Optimizer initialized: {optimizer}")

    return optimizer


def get_pretrain_param_groups(
    model: Module, logger: Logger, skip: set[str], skip_keywords: set[str]
) -> list[dict[str, Parameter | float]]:
    """
    Return a list of parameters from the model, indicating which ones not to
    perform weight decay on.

    Args:
        model: The model being used.
        logger: The logger.
        skip: The list of parameter names to skip for weight decay.
        skip_keywords: The list of keywords whose parameters including them will be skipped.

    Return: An iterable of parameter dicts to pass to the optimizer.
    """
    has_decay: list[Parameter] = []
    no_decay: list[Parameter] = []
    has_decay_name: list[str] = []
    no_decay_name: list[str] = []

    for name, param in model.named_parameters():
        # Only consider trainable parameters
        if param.requires_grad:
            # Skip applying weight decay to:
            # - Scalar parameters
            # - Any bias parameter
            # - Skipped parameters
            no_decay_param: bool = len(param.shape) == 1 or name.endswith(".bias") \
                or name in skip or any([keyword in name for keyword in skip_keywords])

            if no_decay_param:
                no_decay.append(param)
                no_decay_name.append(name)
            else:
                has_decay.append(param)
                has_decay_name.append(name)

    logger.info(f"Parameters without decay: {no_decay_name}")
    logger.info(f"Parameters with decay: {has_decay_name}")

    return [{"params": has_decay}, {"params": no_decay, "weight_decay": 0.}]
