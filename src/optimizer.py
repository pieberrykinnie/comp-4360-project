import torch

def should_use_weight_decay(param_name, param_tensor):
    if param_name.endswith('bias') or param_tensor.ndim == 1:
        return False
    return True

def build_optimizer(model, learning_rate, weight_decay, betas =(0.9, 0.999), eps=1e-8, logger=None):
   
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if should_use_weight_decay(name, param):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    if logger is not None:
        logger.info(f"Optimizer grouping:")
        logger.info(f"decay params: {len(decay_params)} tensors, wd={weight_decay}")
        logger.info(f"no_decay params: {len(no_decay_params)} tensors, wd=0.0")

    group_param = [
        {'params': decay_params, 'weight_decay': float(weight_decay)},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(group_param, lr=float(learning_rate), betas=betas, eps=float(eps))

    if logger is not None:
        logger.info(f" created the adamw optimizer with lr={learning_rate}, weight_decay={weight_decay}, betas={betas}, eps={eps}")

    return optimizer