import math

def warmup_cosine_schedule(base_lr: float, warmup_steps: int, total_steps: int) -> list[float]:
    """
    Returns one Python float learning rate per step.
    """
    lrs = []
    for step in range(total_steps):
        if step < warmup_steps:
            lrs.append(base_lr * (step+1)/warmup_steps)
            
        else:
            coef = math.cos(math.pi * (step - warmup_steps) / (total_steps-warmup_steps))
            lrs.append((base_lr/2.0)* (1 + coef))

    return lrs
            
