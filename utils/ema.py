import torch
from torch import nn

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {n: p.clone().detach() for n, p in model.state_dict().items()
                       if p.dtype.is_floating_point}
        self.backup = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.state_dict().items():
            if n in self.shadow and p.dtype.is_floating_point:
                self.shadow[n].mul_(self.decay).add_(p, alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        msd = model.state_dict()
        for n in self.shadow:
            self.backup[n] = msd[n].clone()
            msd[n].copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model: nn.Module):
        msd = model.state_dict()
        for n in self.backup:
            msd[n].copy_(self.backup[n])
        self.backup = {}
