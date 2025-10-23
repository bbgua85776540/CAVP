import torch
from .geo import circ_diff, circ_add, clamp01

@torch.no_grad()
def build_constvel_prior(vp_hist: torch.Tensor, k: int, gamma: float, Tf: int):
    B, T, _ = vp_hist.shape
    dx = circ_diff(vp_hist[:, -k:, 0][:, 1:], vp_hist[:, -k:, 0][:, :-1]).mean(dim=1)
    dy = (vp_hist[:, -k:, 1][:, 1:] - vp_hist[:, -k:, 1][:, :-1]).mean(dim=1)
    v0 = torch.stack([dx, dy], dim=-1)
    last = vp_hist[:, -1, :]
    pos = last.clone(); outs = []; v = v0.clone()
    for _ in range(Tf):
        xn = circ_add(pos[:, 0], v[:, 0]); yn = clamp01(pos[:, 1] + v[:, 1])
        pos = torch.stack([xn, yn], dim=-1); outs.append(pos); v = v * gamma
    return torch.stack(outs, dim=1), v0

@torch.no_grad()
def build_constvel_nodamp_laststep(vp_hist: torch.Tensor, Tf: int):
    dlast_x = circ_diff(vp_hist[:, -1, 0], vp_hist[:, -2, 0])
    dlast_y = (vp_hist[:, -1, 1] - vp_hist[:, -2, 1])
    v = torch.stack([dlast_x, dlast_y], dim=-1)
    last = vp_hist[:, -1, :]; pos = last.clone(); outs = []
    for _ in range(Tf):
        xn = circ_add(pos[:, 0], v[:, 0]); yn = clamp01(pos[:, 1] + v[:, 1])
        pos = torch.stack([xn, yn], dim=-1); outs.append(pos)
    return torch.stack(outs, dim=1)

@torch.no_grad()
def build_copylast_prior(vp_hist: torch.Tensor, Tf: int):
    return vp_hist[:, -1:, :].expand(-1, Tf, -1).contiguous()
