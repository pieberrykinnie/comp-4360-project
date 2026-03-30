import torch


def log_param_group(group_name, names, params, weight_decay, logger, max_num=5):

    total_scalar = sum(p.numel() for p in params)

    msg = (f"{group_name} param group: {len(params)} tensors, {total_scalar} scalars, weight_decay={weight_decay}\n"
           f"Example tensors: {', '.join(names[:max_num])}")

    if logger is not None:
        logger.info(msg)
    else:
        print(msg)

    examples = names[:max_num]

    if examples:
        ex_msg = f" examples: {examples}"
        if logger is not None:
            logger.info(ex_msg)
        else:
            print(ex_msg)


def should_use_weight_decay(param_name, param_tensor):
    if param_name.endswith('.bias') or param_tensor.ndim == 1:
        return False
    return True


def build_pretrain_param_groups(model, weight_decay, logger=None):
    decay_params, no_decay_params = [], []
    decay_names, no_decay_names = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if should_use_weight_decay(name, param):
            decay_params.append(param)
            decay_names.append(name)
        else:
            no_decay_params.append(param)
            no_decay_names.append(name)

    log_param_group("decay group", decay_names,
                    decay_params, weight_decay, logger)
    log_param_group("no_decay group", no_decay_names,
                    no_decay_params, 0.0, logger)

    if logger is not None:
        logger.info(f"Pretrain param grouping:")
        logger.info(
            f"decay params: {len(decay_params)} tensors, wd={weight_decay}")
        logger.info(f"no_decay params: {len(no_decay_params)} tensors, wd=0.0")

    group_param = [
        {'params': decay_params, 'weight_decay': float(weight_decay)},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

    return group_param


def build_finetune_param_groups(model, weight_decay, logger=None):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if should_use_weight_decay(name, param):
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    if logger is not None:
        logger.info(f"Finetune param grouping:")
        logger.info(
            f"decay params: {len(decay_params)} tensors, wd={weight_decay}")
        logger.info(f"no_decay params: {len(no_decay_params)} tensors, wd=0.0")

    group_param = [
        {'params': decay_params, 'weight_decay': float(weight_decay)},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

    return group_param


def create_adamw_optimizer(group_param, learning_rate, betas=(0.9, 0.999), eps=1e-8, logger=None):
    optimizer = torch.optim.AdamW(group_param, lr=float(
        learning_rate), betas=betas, eps=float(eps))

    if logger is not None:
        logger.info(
            f" created the adamw optimizer with lr={learning_rate}, betas={betas}, eps={eps}")

    return optimizer


def build_optimizer(config, model, logger=None, is_pretrain=True):

    learning_rate = config.TRAIN.BASE_LR
    weight_decay = config.TRAIN.WEIGHT_DECAY
    betas = config.TRAIN.OPTIMIZER.BETAS
    eps = config.TRAIN.OPTIMIZER.EPS
    optimizer_name = config.TRAIN.OPTIMIZER.NAME.lower()

    if optimizer_name != "adamw":
        raise ValueError(
            f"This project optimizer.py currently only supports AdamW, "
            f"but got OPTIMIZER.NAME = '{config.TRAIN.OPTIMIZER.NAME}'"
        )

    if is_pretrain:
        if logger is not None:
            logger.info("Building optimizer for pretraining")
        group_param = build_pretrain_param_groups(model, weight_decay, logger)
    else:
        if logger is not None:
            logger.info("Building optimizer for fine-tuning")
        group_param = build_finetune_param_groups(model, weight_decay, logger)

    optimizer = create_adamw_optimizer(
        group_param=group_param,
        learning_rate=learning_rate,
        betas=betas,
        eps=eps,
        logger=logger
    )

    return optimizer
