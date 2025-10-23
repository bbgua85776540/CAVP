import os, time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from cavp.config import Cfg, asdict_cfg
from cavp.utils import set_seed
from cavp.utils import build_constvel_prior, build_constvel_nodamp_laststep, build_copylast_prior
from cavp.utils import metrics_top1_md
from cavp.utils import ensure_hist_xy, ensure_target_xy
from cavp.utils import EMA
from cavp.models import ResidualGRU_MoP_Video
from cavp.training.losses import loss_fn

from data.dataloader_stub import get_train_val_loaders  # absolute import preserved

def train_one_epoch(model, loader, optim, cfg: Cfg, epoch: int, ema: EMA = None):
    model.train(); dev = cfg.device; t0 = time.time()
    avg = {"loss": 0.0, "pos": 0.0, "vel": 0.0, "tv": 0.0, "gr": 0.0}; n = 0
    for step, batch in enumerate(loader, 1):
        vp_hist = batch["vp_hist"].to(dev)
        target = batch["target"].to(dev)
        video  = batch.get("video", None)
        if video is not None: video = video.to(dev).float()

        # Robust conversions
        hist_xy = ensure_hist_xy(vp_hist, cfg.H, cfg.W)
        gt_xy   = ensure_target_xy(target, cfg.GRID_H, cfg.GRID_W, cfg.H, cfg.W)

        prior_cv, _ = build_constvel_prior(hist_xy, cfg.velocity_k, cfg.damped_decay, cfg.T_future)
        prior_cl    = build_copylast_prior(hist_xy, cfg.T_future)

        pred, g, cv_gain = model(hist_xy, prior_cv, prior_cl, video if cfg.enable_video else None)
        loss, parts = loss_fn(pred, gt_xy, cfg, g, cv_gain)

        optim.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); optim.step()
        if ema is not None: ema.update(model)

        bs = hist_xy.size(0); n += bs
        avg["loss"] += loss.item() * bs; avg["pos"] += parts["pos"] * bs; avg["vel"] += parts["vel"] * bs
        avg["tv"] += parts.get("gate_tv", 0.0) * bs; avg["gr"] += parts.get("gain_reg", 0.0) * bs
        if step % cfg.log_interval == 0:
            dt = time.time() - t0
            print(f"[train] epoch {epoch} step {step}/{len(loader)} "
                  f"loss={avg['loss']/n:.4f} (pos={avg['pos']/n:.4f}, vel={avg['vel']/n:.4f}, tv={avg['tv']/n:.5f}, gr={avg['gr']/n:.5f}) "
                  f"lr={optim.param_groups[0]['lr']:.2e} dt={dt:.1f}s")
    return {k: float(v / max(n, 1)) for k, v in avg.items()}

@torch.no_grad()
def validate(model, loader, cfg: Cfg, ema: EMA = None):
    model.eval(); dev = cfg.device; applied = False
    if ema is not None: ema.apply_shadow(model); applied = True
    hit_sum = md_sum = 0.0; cnt = 0
    b01 = {"hit": 0.0, "md": 0.0, "cnt": 0}; b02 = {"hit": 0.0, "md": 0.0, "cnt": 0}; b03 = {"hit": 0.0, "md": 0.0, "cnt": 0}
    for batch in loader:
        vp_hist = batch["vp_hist"].to(dev)
        target  = batch["target"].to(dev)
        video   = batch.get("video", None)
        if video is not None: video = video.to(dev).float()

        hist_xy = ensure_hist_xy(vp_hist, cfg.H, cfg.W)
        gt_xy   = ensure_target_xy(target, cfg.GRID_H, cfg.GRID_W, cfg.H, cfg.W)

        prior_cv, _ = build_constvel_prior(hist_xy, cfg.velocity_k, cfg.damped_decay, cfg.T_future)
        prior_cl = build_copylast_prior(hist_xy, cfg.T_future)
        pred, _, _ = model(hist_xy, prior_cv, prior_cl, video if cfg.enable_video else None)

        m = metrics_top1_md(pred, gt_xy, cfg.GRID_H, cfg.GRID_W); B = hist_xy.size(0)
        hit_sum += m["top1_all"] * B; md_sum += m["md_all"] * B; cnt += B
        for b, kk in zip([b01, b02, b03], [("top1_01", "md_01"), ("top1_02", "md_02"), ("top1_03", "md_03")]):
            b["hit"] += m[kk[0]] * B; b["md"] += m[kk[1]] * B; b["cnt"] += B
    if applied: ema.restore(model)

    top1 = hit_sum / max(cnt, 1); md = md_sum / max(cnt, 1)
    f = lambda b: (b["hit"] / b["cnt"], b["md"] / b["cnt"])
    t1, m1 = f(b01); t2, m2 = f(b02); t3, m3 = f(b03)
    seg1, seg2, seg3 = m["seg"] if cnt > 0 else ("01-10", "11-20", "21-30")
    print(f"[Val] top1_all={top1:.4f}  md_all={md:.4f}")
    print(f"      Buckets {seg1} / {seg2} / {seg3}  Top1/MD:  {t1:.4f}/{m1:.4f} | {t2:.4f}/{m2:.4f} | {t3:.4f}/{m3:.4f}")
    return {"top1_all": float(top1), "md_all": float(md),
            "t1": float(t1), "m1": float(m1), "t2": float(t2), "m2": float(m2), "t3": float(t3), "m3": float(m3),
            "seg": [seg1, seg2, seg3]}

@torch.no_grad()
def baseline_eval_full(val_loader, cfg: Cfg):
    dev = cfg.device
    def agg(pred_fn, name):
        hit_sum = md_sum = 0.0; cnt = 0
        for batch in val_loader:
            vp_hist = batch["vp_hist"].to(dev)
            target  = batch["target"].to(dev)
            hist_xy = ensure_hist_xy(vp_hist, cfg.H, cfg.W)
            gt_xy   = ensure_target_xy(target, cfg.GRID_H, cfg.GRID_W, cfg.H, cfg.W)
            pred_xy = pred_fn(hist_xy)
            m = metrics_top1_md(pred_xy, gt_xy, cfg.GRID_H, cfg.GRID_W)
            B = hist_xy.size(0); hit_sum += m["top1_all"] * B; md_sum += m["md_all"] * B; cnt += B
        print(f"[Baseline FullVal][{name}] top1={hit_sum/max(cnt,1):.4f}  md_wrap={md_sum/max(cnt,1):.4f}")

    agg(lambda h: build_copylast_prior(h, cfg.T_future), "CopyLast")
    agg(lambda h: build_constvel_nodamp_laststep(h, cfg.T_future), "ConstVel(k=1,no-damp)")
    agg(lambda h: build_constvel_prior(h, cfg.velocity_k, cfg.damped_decay, cfg.T_future)[0], "ConstVel-like(k=5,γ=0.92)")

def train_validate_one(cfg: Cfg, run_name: str, out_dir_ckpt: str):
    set_seed(cfg.seed)
    train_loader, val_loader = get_train_val_loaders(cfg)

    print("\n[Sanity] Evaluating baselines on FULL validation set ...")
    baseline_eval_full(val_loader, cfg)

    print(f"[Run] Training {run_name} (seed={cfg.seed}) ...")
    model = ResidualGRU_MoP_Video(cfg).to(cfg.device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = CosineAnnealingLR(optim, T_max=cfg.num_epochs, eta_min=cfg.lr * 0.1)
    ema = EMA(model, cfg.ema_decay) if cfg.enable_ema else None

    best = -1e9; best_ep = -1; best_val = None; hist = []
    for ep in range(1, cfg.num_epochs + 1):
        tr = train_one_epoch(model, train_loader, optim, cfg, ep, ema)
        va = validate(model, val_loader, cfg, ema)
        score = va["top1_all"] - 0.05 * va["md_all"] - cfg.score_md03_coef * va["m3"]
        hist.append({"epoch": ep, "train": tr, "val": va, "score": float(score)})

        if score > best:
            best = score; best_ep = ep; best_val = va
            os.makedirs(out_dir_ckpt, exist_ok=True)
            torch.save({"cfg": asdict_cfg(cfg), "state_dict": model.state_dict()},
                       os.path.join(out_dir_ckpt, f"{run_name}_seed{cfg.seed}_best.pth"))
            print(f"[Save] {out_dir_ckpt}/{run_name}_seed{cfg.seed}_best.pth")
        sch.step()

    return {"name": run_name, "seed": cfg.seed, "best_epoch": best_ep, "best_val": best_val, "history": hist}
