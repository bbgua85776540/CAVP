import torch
import torch.nn.functional as F
from cavp.config import Cfg
from cavp.utils.geo import circ_diff
from cavp.utils.geo import thirds_slices

def thirds_weights(T, w1, w2, w3, device):
    s1, s2, s3 = thirds_slices(T)
    w = torch.ones(T, device=device); w[s1] = w1; w[s2] = w2; w[s3] = w3
    return w.view(1, T, 1)

def loss_fn(pred_xy, gt_xy, cfg: Cfg, gate=None, cv_gain=None):
    B, T, _ = pred_xy.shape
    bw = thirds_weights(T, cfg.bucket_w_01, cfg.bucket_w_02, cfg.bucket_w_03, pred_xy.device)

    dx = circ_diff(pred_xy[..., 0], gt_xy[..., 0]).unsqueeze(-1)
    dy = (pred_xy[..., 1] - gt_xy[..., 1]).unsqueeze(-1)
    pos_l1 = F.smooth_l1_loss(torch.cat([dx, dy], dim=-1), torch.zeros_like(pred_xy), reduction='none')
    pos_l1 = (pos_l1 * bw).mean()

    def d1(x): return (x[:, 1:] - x[:, :-1]).unsqueeze(-1)
    pvx = d1(pred_xy[..., 0]); pvy = d1(pred_xy[..., 1])
    gvx = d1(gt_xy[..., 0]); gvy = d1(gt_xy[..., 1])
    vx_l1 = F.smooth_l1_loss((pvx.squeeze(-1) - gvx.squeeze(-1)).unsqueeze(-1), torch.zeros_like(gvx), reduction='none')
    vy_l1 = F.smooth_l1_loss(pvy - gvy, torch.zeros_like(gvy), reduction='none')
    bw_v = thirds_weights(T - 1, cfg.bucket_w_01, cfg.bucket_w_02, cfg.bucket_w_03, pred_xy.device)
    vel_l1 = ((vx_l1 + vy_l1) * bw_v).mean()

    total = cfg.loss_w_pos * pos_l1 + cfg.loss_w_vel * vel_l1
    parts = {"pos": float(pos_l1.item()), "vel": float(vel_l1.item())}

    if (gate is not None) and cfg.gate_tv_lambda > 0:
        gt = gate.squeeze(-1); tv = (gt[:, 1:] - gt[:, :-1])**2
        tvm = tv.mean(); total += cfg.gate_tv_lambda * tvm; parts["gate_tv"] = float(tvm.item())

    if (cv_gain is not None) and cfg.cv_gain_reg_lambda > 0:
        t = torch.linspace(1, T, steps=T, device=pred_xy.device, dtype=pred_xy.dtype) / T
        w_reg = cfg.gain_reg_early + (cfg.gain_reg_late - cfg.gain_reg_early) * t
        w_reg = w_reg.view(1, T, 1).expand(B, T, 1)
        gr = ((cv_gain - 1.0)**2).unsqueeze(-1) * w_reg
        grm = gr.mean()
        total += cfg.cv_gain_reg_lambda * grm
        parts["gain_reg"] = float(grm.item())
    return total, parts
