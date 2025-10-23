import torch

@torch.no_grad()
def _hw_onehot_to_xy(mask_hw: torch.Tensor, H: int, W: int):
    # mask_hw: (B,T,H,W) strict one-hot (or near one-hot); use argmax
    B, T, Hh, Ww = mask_hw.shape
    assert Hh == H and Ww == W, f"Expected (H,W)=({H},{W}), got ({Hh},{Ww})"
    flat = mask_hw.view(B, T, -1).argmax(dim=-1)  # (B,T)
    y = (flat // W).float() + 0.5  # cell center
    x = (flat %  W).float() + 0.5
    return torch.stack([x / W, y / H], dim=-1)  # (B,T,2)

@torch.no_grad()
def _grid_onehot_to_xy(onehot: torch.Tensor, GH: int, GW: int):
    # onehot: (B,T,GH,GW)
    B, T, Ghh, Gww = onehot.shape
    assert Ghh == GH and Gww == GW, f"Expected (GH,GW)=({GH},{GW}), got ({Ghh},{Gww})"
    flat = onehot.view(B, T, -1).argmax(dim=-1)  # (B,T)
    gy = (flat // GW).float() + 0.5
    gx = (flat %  GW).float() + 0.5
    return torch.stack([gx / GW, gy / GH], dim=-1)

@torch.no_grad()
def ensure_hist_xy(vp_hist, H:int, W:int):
    # Accept: (B,Th,2) or (B,Th,H,W)
    if vp_hist.dim() == 3 and vp_hist.size(-1) == 2:
        return vp_hist.float()
    if vp_hist.dim() == 4 and vp_hist.size(-2) == H and vp_hist.size(-1) == W:
        return _hw_onehot_to_xy(vp_hist.float(), H, W)
    raise ValueError(f"vp_hist shape not supported: {tuple(vp_hist.shape)}")

@torch.no_grad()
def ensure_target_xy(target, GH:int, GW:int, H:int=None, W:int=None):
    # Accept: (B,T,2) or (B,T,GH,GW) or (B,T,H,W) if H,W provided
    if target.dim() == 3 and target.size(-1) == 2:
        return target.float()
    if target.dim() == 4 and target.size(-2) == GH and target.size(-1) == GW:
        return _grid_onehot_to_xy(target.float(), GH, GW)
    if H is not None and W is not None and target.dim() == 4 and target.size(-2) == H and target.size(-1) == W:
        return _hw_onehot_to_xy(target.float(), H, W)
    raise ValueError(f"target shape not supported: {tuple(target.shape)} (need (B,T,2) or (B,T,{GH},{GW}) or (B,T,{H},{W}))")
