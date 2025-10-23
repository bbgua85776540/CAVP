import torch
from .geo import thirds_slices, thirds_label

@torch.no_grad()
def coords_to_indices(xy, GH, GW):
    x = (xy[..., 0] * GW).floor().clamp(0, GW - 1).long()
    y = (xy[..., 1] * GH).floor().clamp(0, GH - 1).long()
    return y, x

@torch.no_grad()
def metrics_top1_md(pred_xy, gt_xy, GH, GW):
    B, T, _ = pred_xy.shape
    gy, gx = coords_to_indices(gt_xy, GH, GW)
    py, px = coords_to_indices(pred_xy, GH, GW)
    hit = ((gy == py) & (gx == px)).float()
    dx_raw = (gx - px).abs()
    dx_wrap = torch.minimum(dx_raw, torch.tensor(GW, device=dx_raw.device) - dx_raw)
    dy = (gy - py).abs()
    md = (dx_wrap + dy).float()
    s1, s2, s3 = thirds_slices(T); agg = lambda a: a.mean().item()
    return {
        "top1_all": agg(hit), "md_all": agg(md),
        "top1_01": agg(hit[:, s1]), "md_01": agg(md[:, s1]),
        "top1_02": agg(hit[:, s2]), "md_02": agg(md[:, s2]),
        "top1_03": agg(hit[:, s3]), "md_03": agg(md[:, s3]),
        "seg": list(thirds_label(T)),
    }
