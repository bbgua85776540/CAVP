import random
import torch

def set_seed(s:int):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True

def circ_diff(x1, x0):
    d = x1 - x0
    return (d + 0.5) % 1.0 - 0.5

def circ_add(x, dx):
    return torch.remainder(x + dx, 1.0)

def clamp01(y):
    return y.clamp(0.0, 1.0 - 1e-6)

def thirds_slices(T:int):
    a = T // 3; b = (2 * T) // 3
    return slice(0, a), slice(a, b), slice(b, T)

def thirds_label(T:int):
    a = T // 3; b = (2 * T) // 3
    return f"{1:02d}-{a:02d}", f"{a+1:02d}-{b:02d}", f"{b+1:02d}-{T:02d}"
