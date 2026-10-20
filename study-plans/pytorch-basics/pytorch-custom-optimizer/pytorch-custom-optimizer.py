import torch

class CustomSGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.01, momentum: float = 0.0):
        defaults = dict(lr = lr, momentum = momentum)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Returns the closure loss when provided, otherwise None.
        """
        loss = None
        if closure is not None:
            loss= closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if momentum != 0:
                    state = self.state[p]
                    if 'velocity' not in state:
                        state['velocity'] = torch.zeros_like(p.data)
                    v = state['velocity']
                    v.mul_(momentum).add_(d_p)
                    p.data.add_(v, alpha= -lr)
                else:
                    p.data.add_(d_p, alpha= -lr)

        return loss


            
            