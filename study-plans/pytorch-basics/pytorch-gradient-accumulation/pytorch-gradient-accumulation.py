import torch

def gradient_accumulation(w_init: torch.Tensor, micro_batches: list, lr: float, accum_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (final_weights, last_mean_gradient) as float32 tensors.
    """
    w = w_init.detach().clone().requires_grad_(True)
    last_avg_grad = None
    
    for i , (inputs, targets) in enumerate(micro_batches):
        y = torch.dot(w,inputs)
        loss = (y - targets)**2
        loss.backward()
        
        if (i+1)%accum_steps ==0:
            last_avg_grad = w.grad.clone() / accum_steps
            with torch.no_grad():
                w -= lr * last_avg_grad
            w.grad.zero_()

    return w.detach() , last_avg_grad.detach()
        
        