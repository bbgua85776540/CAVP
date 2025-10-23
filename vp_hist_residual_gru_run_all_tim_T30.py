# -*- coding: utf-8 -*-
"""
vp_hist_residual_gru_run_all_tim_T30.py  —  T30_v3A (Attempt 3/3) SWEEP
一键跑：三随机种子 + 消融（Full / HistOnly / NoCVGain / NoVideo）
输出：
  - logs/T30_v3A_Attempt3/sweep_flat.csv
  - logs/T30_v3A_Attempt3/sweep_agg.csv
  - logs/T30_v3A_Attempt3/sweep_results.json
  - ckpts/T30_v3A_Attempt3/<variant>_seedX_best.pth
数据契约：from data.dataloader_stub import get_train_val_loaders
"""

import os, time, random, json, math, csv
from dataclasses import dataclass, asdict
from statistics import mean

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.dataloader_stub import get_train_val_loaders  #

# ======================
# 配置（锁定 Attempt 3/3；仅时间尺度改为 30/60）
# ======================
@dataclass
class Cfg:
    # 数据/网格
    H:int=144; W:int=256; GRID_H:int=9; GRID_W:int=16
    hist_len:int=30; T_future:int=30; process_frame_nums:int=60
    batch_size:int=16
    # 训练
    num_epochs:int=16; lr:float=3e-4; weight_decay:float=0.01
    grad_clip:float=1.0; seed:int=42
    device:str="cuda" if torch.cuda.is_available() else "cpu"
    log_interval:int=400
    # 先验
    velocity_k:int=5; damped_decay:float=0.92
    # Late-Heavy & 评分
    bucket_w_01:float=0.90; bucket_w_02:float=1.00; bucket_w_03:float=1.25
    score_md03_coef:float=0.03
    # 损失
    loss_w_pos:float=1.0; loss_w_vel:float=0.3
    # 残差/门控
    residual_scale:float=0.08; residual_boost:float=0.14; residual_scale_max:float=0.22
    move_thr:float=0.0270; gate_sigma:float=0.015
    gate_tv_lambda:float=3e-3
    # 视频分支
    enable_video:bool=True; use_future_video:bool=True
    vid_hidden:int=256; film_dim:int=256; dropout_p:float=0.12
    smooth_ks:int=11
    # 主干
    model_hidden:int=160
    # cv_gain（基础区间）
    enable_cv_gain:bool=True; cv_gain_min:float=0.92; cv_gain_max:float=1.08
    cv_gain_reg_lambda:float=2.5e-4
    # Attempt 3/3：双尺度与时间调度（沿用与 T90 相同的相对策略）
    alpha_time_bias:float=0.15
    fast_ks:int=5; slow_ks:int=21
    gain_widen_mid:float=0.6
    gain_clip_min:float=0.88; gain_clip_max:float=1.12
    end_shrink:float=0.5
    gain_reg_early:float=0.5; gain_reg_late:float=1.0
    # EMA
    enable_ema:bool=True; ema_decay:float=0.999

# ========== Utils ==========
def set_seed(s:int):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True

def circ_diff(x1, x0): d=x1-x0; return (d+0.5)%1.0-0.5
def circ_add(x,dx): return torch.remainder(x+dx,1.0)
def clamp01(y): return y.clamp(0.0, 1.0-1e-6)

def thirds_slices(T):
    a=T//3; b=(2*T)//3
    return slice(0,a), slice(a,b), slice(b,T)

def thirds_label(T):
    a=T//3; b=(2*T)//3
    return f"{1:02d}-{a:02d}", f"{a+1:02d}-{b:02d}", f"{b+1:02d}-{T:02d}"

@torch.no_grad()
def build_constvel_prior(vp_hist:torch.Tensor,k:int,gamma:float,Tf:int):
    B,T,_=vp_hist.shape
    dx = circ_diff(vp_hist[:, -k:, 0][:, 1:], vp_hist[:, -k:, 0][:, :-1]).mean(dim=1)
    dy = (vp_hist[:, -k:, 1][:, 1:] - vp_hist[:, -k:, 1][:, :-1]).mean(dim=1)
    v0 = torch.stack([dx,dy],dim=-1)
    last=vp_hist[:,-1,:]; pos=last.clone(); outs=[]; v=v0.clone()
    for _ in range(Tf):
        xn=circ_add(pos[:,0],v[:,0]); yn=clamp01(pos[:,1]+v[:,1])
        pos=torch.stack([xn,yn],dim=-1); outs.append(pos); v=v*gamma
    return torch.stack(outs,dim=1), v0

@torch.no_grad()
def build_constvel_nodamp_laststep(vp_hist:torch.Tensor,Tf:int):
    dlast_x = circ_diff(vp_hist[:, -1, 0], vp_hist[:, -2, 0])
    dlast_y = (vp_hist[:, -1, 1] - vp_hist[:, -2, 1])
    v = torch.stack([dlast_x, dlast_y], dim=-1)
    last = vp_hist[:, -1, :]; pos=last.clone(); outs=[]
    for _ in range(Tf):
        xn=circ_add(pos[:,0],v[:,0]); yn=clamp01(pos[:,1]+v[:,1])
        pos=torch.stack([xn,yn],dim=-1); outs.append(pos)
    return torch.stack(outs,dim=1)

@torch.no_grad()
def build_copylast_prior(vp_hist:torch.Tensor,Tf:int):
    return vp_hist[:, -1:, :].expand(-1, Tf, -1).contiguous()

@torch.no_grad()
def coords_to_indices(xy, GH, GW):
    x=(xy[...,0]*GW).floor().clamp(0,GW-1).long()
    y=(xy[...,1]*GH).floor().clamp(0,GH-1).long()
    return y,x

@torch.no_grad()
def metrics_top1_md(pred_xy, gt_xy, GH, GW):
    B,T,_=pred_xy.shape
    gy,gx=coords_to_indices(gt_xy,GH,GW)
    py,px=coords_to_indices(pred_xy,GH,GW)
    hit=((gy==py)&(gx==px)).float()
    dx_raw=(gx-px).abs()
    dx_wrap=torch.minimum(dx_raw, torch.tensor(GW,device=dx_raw.device)-dx_raw)
    dy=(gy-py).abs()
    md=(dx_wrap+dy).float()
    s1,s2,s3=thirds_slices(T); agg=lambda a: a.mean().item()
    return {"top1_all":agg(hit),"md_all":agg(md),
            "top1_01":agg(hit[:,s1]),"md_01":agg(md[:,s1]),
            "top1_02":agg(hit[:,s2]),"md_02":agg(md[:,s2]),
            "top1_03":agg(hit[:,s3]),"md_03":agg(md[:,s3]),
            "seg":list(thirds_label(T))}

# ========== Video enc ==========
class FrameCNN(nn.Module):
    def __init__(self,out_dim=64):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(3,16,5,2,2), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16,32,3,2,1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32,64,3,2,1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64,out_dim,3,2,1), nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
        ); self.gap=nn.AdaptiveAvgPool2d(1)
    def forward(self,x): f=self.net(x); return self.gap(f).squeeze(-1).squeeze(-1)

class VideoEncoder(nn.Module):
    def __init__(self,cnn_dim=64,t_hidden=128,out_dim=256,hist_len=30):
        super().__init__()
        self.cnn=FrameCNN(out_dim=cnn_dim)
        self.tgru=nn.GRU(cnn_dim,t_hidden,num_layers=1,batch_first=True,bidirectional=True)
        self.proj=nn.Linear(2*t_hidden,out_dim)
        self.hist_len=hist_len
    @staticmethod
    def smooth_time(x, ks:int):
        if ks<=1: return x
        pad=ks//2
        xbt=x.transpose(1,2)  # (B,V,T)
        xbt=F.avg_pool1d(F.pad(xbt,(pad,pad),mode="replicate"), kernel_size=ks, stride=1, padding=0)
        return xbt.transpose(1,2).contiguous()
    def forward(self,video, ks_fast:int, ks_slow:int, base_ks:int):
        B,C,T,H,W=video.shape
        x=video.permute(0,2,1,3,4).contiguous().view(B*T,C,H,W)
        f=self.cnn(x).view(B,T,-1)
        tfeat,_=self.tgru(f); tfeat=self.proj(tfeat)     # (B,T,feat)
        hist_sum=tfeat[:,:self.hist_len,:].mean(dim=1)   # (B,feat)
        fut_raw=tfeat[:,self.hist_len:,:]                # (B,Tf,feat)
        fut_base=self.smooth_time(fut_raw, base_ks)
        fut_fast=self.smooth_time(fut_raw, ks_fast)
        fut_slow=self.smooth_time(fut_raw, ks_slow)
        return fut_base, fut_fast, fut_slow, hist_sum

# ========== Model ==========
class ResidualGRU_MoP_Video(nn.Module):
    def __init__(self,cfg:Cfg):
        super().__init__(); self.cfg=cfg; H=cfg.model_hidden
        self.enc=nn.GRU(4,H,1,batch_first=True); self.dec=nn.GRU(3,H,1,batch_first=True)
        self.drop=nn.Dropout(cfg.dropout_p)
        self.head_pos=nn.Sequential(nn.Linear(H,H),nn.ReLU(inplace=True),nn.Dropout(cfg.dropout_p),nn.Linear(H,2))

        self.video_enc=VideoEncoder(64,cfg.vid_hidden//2,cfg.vid_hidden,cfg.hist_len)
        self.film_gamma=nn.Sequential(nn.Linear(cfg.vid_hidden,cfg.film_dim),nn.ReLU(inplace=True),nn.Dropout(cfg.dropout_p),nn.Linear(cfg.film_dim,H))
        self.film_beta =nn.Sequential(nn.Linear(cfg.vid_hidden,cfg.film_dim),nn.ReLU(inplace=True),nn.Dropout(cfg.dropout_p),nn.Linear(cfg.film_dim,H))
        self.hist_inject=nn.Linear(cfg.vid_hidden,H)

        # 双尺度融合门
        self.alpha_gate = nn.Sequential(
            nn.Linear(H+2, H//2), nn.ReLU(inplace=True), nn.Dropout(cfg.dropout_p),
            nn.Linear(H//2, 1), nn.Sigmoid()
        )
        # 门控/增益的附加视频投影
        self.gate_vid_proj=nn.Linear(cfg.vid_hidden,16)
        self.head_gate=nn.Sequential(nn.Linear(H+2+16,H//2),nn.ReLU(inplace=True),nn.Dropout(cfg.dropout_p),nn.Linear(H//2,1),nn.Sigmoid())
        self.head_gain=nn.Sequential(nn.Linear(H+16,H//2),nn.ReLU(inplace=True),nn.Dropout(cfg.dropout_p),nn.Linear(H//2,1),nn.Sigmoid())

    def forward(self,hist_xy,prior_cv_xy,prior_cl_xy,video=None):
        B,Th,_=hist_xy.shape; Tf=prior_cv_xy.shape[1]; dev=hist_xy.device

        # 历史编码
        dx_hist=torch.zeros_like(hist_xy)
        dx_hist[:,1:,0]=circ_diff(hist_xy[:,1:,0],hist_xy[:,:-1,0])
        dx_hist[:,1:,1]=(hist_xy[:,1:,1]-hist_xy[:,:-1,1]).clamp(-1.0,1.0)
        enc_in=torch.cat([hist_xy,dx_hist],dim=-1); _,h=self.enc(enc_in)

        # 时间归一
        t_norm=torch.linspace(1,Tf,steps=Tf,device=dev,dtype=hist_xy.dtype)/Tf
        t_norm_ = t_norm.view(1,Tf,1).expand(B,Tf,1)

        # 视频特征
        if self.cfg.enable_video and (video is not None):
            fut_base, fut_fast, fut_slow, hist_sum = self.video_enc(
                video, self.cfg.fast_ks, self.cfg.slow_ks, self.cfg.smooth_ks
            )
            if not self.cfg.use_future_video:
                hist_rep = hist_sum.unsqueeze(1).expand(B,Tf,-1).contiguous()
                fut_base = fut_fast = fut_slow = hist_rep

            # prior 步长作“速度先验”
            prior_d=torch.zeros_like(prior_cv_xy)
            prior_d[:,0,0]=circ_diff(prior_cv_xy[:,0,0],hist_xy[:,-1,0])
            prior_d[:,0,1]=(prior_cv_xy[:,0,1]-hist_xy[:,-1,1]).clamp(-1.0,1.0)
            if Tf>1:
                prior_d[:,1:,0]=circ_diff(prior_cv_xy[:,1:,0],prior_cv_xy[:,:-1,0])
                prior_d[:,1:,1]=(prior_cv_xy[:,1:,1]-prior_cv_xy[:,:-1,1]).clamp(-1.0,1.0)
            speed=torch.sqrt(prior_d[...,0]**2 + prior_d[...,1]**2 + 1e-12).unsqueeze(-1)
            alpha_in=torch.cat([speed, t_norm_], dim=-1)
            ctx=h.transpose(0,1).expand(B,Tf,-1)
            alpha=self.alpha_gate(torch.cat([ctx, alpha_in], dim=-1)).squeeze(-1)
            alpha = torch.clamp(alpha + self.cfg.alpha_time_bias * t_norm.view(1,Tf), 0.0, 1.0)
            vid_fut = alpha.unsqueeze(-1)*fut_slow + (1.0-alpha).unsqueeze(-1)*fut_fast  # (B,Tf,V)

            # FiLM
            gamma=self.film_gamma(vid_fut); beta=self.film_beta(vid_fut)
            s_film=(0.6 + 1.0 * t_norm).view(1,Tf,1)
            gamma=gamma*s_film; beta=beta*s_film
            gamma=self.drop(gamma); beta=self.drop(beta)

            gate_vid=self.gate_vid_proj(vid_fut)
            h = h + self.hist_inject(hist_sum).view(1,B,-1)
        else:
            gamma=beta=None
            gate_vid=torch.zeros(B,Tf,16,device=dev,dtype=hist_xy.dtype)

        # 解码输入：prior 位移 + t_norm
        prior_d=torch.zeros_like(prior_cv_xy)
        prior_d[:,0,0]=circ_diff(prior_cv_xy[:,0,0],hist_xy[:,-1,0])
        prior_d[:,0,1]=(prior_cv_xy[:,0,1]-hist_xy[:,-1,1]).clamp(-1.0,1.0)
        if Tf>1:
            prior_d[:,1:,0]=circ_diff(prior_cv_xy[:,1:,0],prior_cv_xy[:,:-1,0])
            prior_d[:,1:,1]=(prior_cv_xy[:,1:,1]-prior_cv_xy[:,:-1,1]).clamp(-1.0,1.0)
        dec_in=torch.cat([prior_d,t_norm_],dim=-1)
        dec_out,_=self.dec(dec_in,h)
        if (gamma is not None) and (beta is not None):
            dec_out=dec_out*(1+gamma)+beta

        # 时间型 cv_gain
        cv_gain=None
        if self.cfg.enable_cv_gain:
            base_min, base_max = self.cfg.cv_gain_min, self.cfg.cv_gain_max
            peak = 2.0/3.0
            rel = torch.where(t_norm<=peak, t_norm/peak, (1.0 - (t_norm - peak)/(1.0-peak)))
            w_tri = 1.0 + self.cfg.gain_widen_mid * rel
            lo = 1.0 - (1.0-base_min) * w_tri
            hi = 1.0 + (base_max-1.0) * w_tri
            lo = lo.clamp(min=self.cfg.gain_clip_min); hi = hi.clamp(max=self.cfg.gain_clip_max)

            g_in=torch.cat([dec_out, gate_vid], dim=-1)
            g_raw=self.head_gain(g_in).squeeze(-1)  # (B,Tf)
            cv_gain = lo.view(1,Tf) + (hi.view(1,Tf) - lo.view(1,Tf)) * g_raw

            # 末段回落
            rel_end = torch.clamp((t_norm - peak) / (1.0 - peak), min=0.0, max=1.0)
            end_decay = 1.0 - self.cfg.end_shrink * rel_end
            cv_gain = 1.0 + (cv_gain - 1.0) * end_decay.view(1,Tf)

            # 重建 prior_cv_xy
            pos_x=hist_xy[:,-1,0]; pos_y=hist_xy[:,-1,1]; cv_adj=[]
            for t in range(Tf):
                dx=prior_d[:,t,0]*cv_gain[:,t]; dy=prior_d[:,t,1]*cv_gain[:,t]
                pos_x=circ_add(pos_x,dx); pos_y=clamp01(pos_y+dy)
                cv_adj.append(torch.stack([pos_x,pos_y],dim=-1))
            prior_cv_xy=torch.stack(cv_adj,dim=1)

        # saccade-aware 残差尺度
        speed=torch.sqrt(prior_d[...,0]**2 + prior_d[...,1]**2 + 1e-12)
        gate_sa=torch.sigmoid((speed-self.cfg.move_thr)/(self.cfg.gate_sigma+1e-8))
        dyn_scale=(self.cfg.residual_scale + self.cfg.residual_boost*gate_sa).clamp(max=self.cfg.residual_scale_max).unsqueeze(-1)

        # MoP 门控
        aux=torch.stack([speed, t_norm_.squeeze(-1)], dim=-1)
        gate_in=torch.cat([dec_out,aux,gate_vid],dim=-1)
        g=self.head_gate(gate_in)  # (B,Tf,1)

        # 先验插值 + 残差
        dx_mix=circ_diff(prior_cv_xy[...,0], prior_cl_xy[...,0]) * g.squeeze(-1)
        x_mix=circ_add(prior_cl_xy[...,0], dx_mix)
        y_mix=clamp01(prior_cl_xy[...,1] + g.squeeze(-1)*(prior_cv_xy[...,1]-prior_cl_xy[...,1]))
        prior_mix=torch.stack([x_mix,y_mix],dim=-1)

        res=torch.tanh(self.head_pos(dec_out)) * dyn_scale
        x=circ_add(prior_mix[:,:,0],res[:,:,0]); y=clamp01(prior_mix[:,:,1]+res[:,:,1])
        pred=torch.stack([x,y],dim=-1)
        return pred, g, cv_gain

# ========== Loss ==========
def thirds_weights(T,w1,w2,w3,device):
    s1,s2,s3=thirds_slices(T)
    w=torch.ones(T,device=device); w[s1]=w1; w[s2]=w2; w[s3]=w3
    return w.view(1,T,1)

def loss_fn(pred_xy, gt_xy, cfg:Cfg, gate=None, cv_gain=None):
    B,T,_=pred_xy.shape
    bw=thirds_weights(T,cfg.bucket_w_01,cfg.bucket_w_02,cfg.bucket_w_03,pred_xy.device)

    dx=circ_diff(pred_xy[...,0], gt_xy[...,0]).unsqueeze(-1)
    dy=(pred_xy[...,1]-gt_xy[...,1]).unsqueeze(-1)
    pos_l1=F.smooth_l1_loss(torch.cat([dx,dy],dim=-1), torch.zeros_like(pred_xy), reduction='none')
    pos_l1=(pos_l1*bw).mean()

    def d1(x): return (x[:,1:]-x[:,:-1]).unsqueeze(-1)
    pvx=d1(pred_xy[...,0]); pvy=d1(pred_xy[...,1])
    gvx=d1(gt_xy[...,0]); gvy=d1(gt_xy[...,1])
    vx_l1=F.smooth_l1_loss(circ_diff(pvx.squeeze(-1), gvx.squeeze(-1)).unsqueeze(-1), torch.zeros_like(gvx), reduction='none')
    vy_l1=F.smooth_l1_loss(pvy-gvy, torch.zeros_like(gvy), reduction='none')
    bw_v=thirds_weights(T-1,cfg.bucket_w_01,cfg.bucket_w_02,cfg.bucket_w_03,pred_xy.device)
    vel_l1=((vx_l1+vy_l1)*bw_v).mean()

    total = cfg.loss_w_pos*pos_l1 + cfg.loss_w_vel*vel_l1
    parts={"pos":float(pos_l1.item()), "vel":float(vel_l1.item())}

    if (gate is not None) and cfg.gate_tv_lambda>0:
        gt=gate.squeeze(-1); tv=(gt[:,1:]-gt[:,:-1])**2
        tvm=tv.mean(); total += cfg.gate_tv_lambda*tvm; parts["gate_tv"]=float(tvm.item())

    if (cv_gain is not None) and cfg.cv_gain_reg_lambda>0:
        t=torch.linspace(1,T,steps=T,device=pred_xy.device,dtype=pred_xy.dtype)/T
        w_reg = cfg.gain_reg_early + (cfg.gain_reg_late - cfg.gain_reg_early) * t
        w_reg = w_reg.view(1,T,1).expand(B,T,1)
        gr = ((cv_gain-1.0)**2).unsqueeze(-1) * w_reg
        grm = gr.mean()
        total += cfg.cv_gain_reg_lambda * grm
        parts["gain_reg"]=float(grm.item())
    return total, parts

# ========== EMA ==========
class EMA:
    def __init__(self,model:nn.Module,decay:float=0.999):
        self.decay=decay
        self.shadow={n:p.clone().detach() for n,p in model.state_dict().items() if p.dtype.is_floating_point}
        self.backup={}
    @torch.no_grad()
    def update(self,model:nn.Module):
        for n,p in model.state_dict().items():
            if n in self.shadow and p.dtype.is_floating_point:
                self.shadow[n].mul_((self.decay)).add_(p,alpha=1.0-self.decay)
    @torch.no_grad()
    def apply_shadow(self,model:nn.Module):
        self.backup={}
        msd=model.state_dict()
        for n in self.shadow:
            self.backup[n]=msd[n].clone()
            msd[n].copy_(self.shadow[n])
    @torch.no_grad()
    def restore(self,model:nn.Module):
        msd=model.state_dict()
        for n in self.backup:
            msd[n].copy_(self.backup[n])
        self.backup={}

# ========== Train / Val / Baseline ==========
def train_one_epoch(model,loader,optim,cfg:Cfg,epoch:int,ema:EMA=None):
    model.train(); dev=cfg.device; t0=time.time()
    avg={"loss":0.0,"pos":0.0,"vel":0.0,"tv":0.0,"gr":0.0}; n=0
    for step,batch in enumerate(loader,1):
        vp_hist=batch["vp_hist"].to(dev).float()
        target =batch["target"].to(dev).float()
        video  =batch.get("video",None)
        if video is not None: video=video.to(dev).float()

        prior_cv,_=build_constvel_prior(vp_hist,cfg.velocity_k,cfg.damped_decay,cfg.T_future)
        prior_cl =build_copylast_prior(vp_hist,cfg.T_future)

        pred,g,cv_gain=model(vp_hist,prior_cv,prior_cl, video if cfg.enable_video else None)
        loss,parts=loss_fn(pred,target,cfg,g,cv_gain)

        optim.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip); optim.step()
        if ema is not None: ema.update(model)

        bs=vp_hist.size(0); n+=bs
        avg["loss"]+=loss.item()*bs; avg["pos"]+=parts["pos"]*bs; avg["vel"]+=parts["vel"]*bs
        avg["tv"]+=parts.get("gate_tv",0.0)*bs; avg["gr"]+=parts.get("gain_reg",0.0)*bs
        if step % cfg.log_interval == 0:
            dt=time.time()-t0
            print(f"[train] epoch {epoch} step {step}/{len(loader)} "
                  f"loss={avg['loss']/n:.4f} (pos={avg['pos']/n:.4f}, vel={avg['vel']/n:.4f}, tv={avg['tv']/n:.5f}, gr={avg['gr']/n:.5f}) "
                  f"lr={optim.param_groups[0]['lr']:.2e} dt={dt:.1f}s")
    return {k:float(v/max(n,1)) for k,v in avg.items()}

@torch.no_grad()
def validate(model,loader,cfg:Cfg,ema:EMA=None):
    model.eval(); dev=cfg.device; applied=False
    if ema is not None: ema.apply_shadow(model); applied=True
    hit_sum=md_sum=0.0; cnt=0
    b01={"hit":0.0,"md":0.0,"cnt":0}; b02={"hit":0.0,"md":0.0,"cnt":0}; b03={"hit":0.0,"md":0.0,"cnt":0}
    for batch in loader:
        vp_hist=batch["vp_hist"].to(dev).float()
        target =batch["target"].to(dev).float()
        video  =batch.get("video",None)
        if video is not None: video=video.to(dev).float()
        prior_cv,_=build_constvel_prior(vp_hist,cfg.velocity_k,cfg.damped_decay,cfg.T_future)
        prior_cl=build_copylast_prior(vp_hist,cfg.T_future)
        pred,_,_=model(vp_hist,prior_cv,prior_cl, video if cfg.enable_video else None)

        m=metrics_top1_md(pred,target,cfg.GRID_H,cfg.GRID_W); B=vp_hist.size(0)
        hit_sum+=m["top1_all"]*B; md_sum+=m["md_all"]*B; cnt+=B
        for b,kk in zip([b01,b02,b03],[("top1_01","md_01"),("top1_02","md_02"),("top1_03","md_03")]):
            b["hit"]+=m[kk[0]]*B; b["md"]+=m[kk[1]]*B; b["cnt"]+=B
    if applied: ema.restore(model)

    top1=hit_sum/max(cnt,1); md=md_sum/max(cnt,1)
    f=lambda b:(b["hit"]/b["cnt"], b["md"]/b["cnt"])
    t1,m1=f(b01); t2,m2=f(b02); t3,m3=f(b03)
    seg1,seg2,seg3=thirds_label(cfg.T_future)
    print(f"[Val] top1_all={top1:.4f}  md_all={md:.4f}")
    print(f"      Buckets {seg1} / {seg2} / {seg3}  Top1/MD:  {t1:.4f}/{m1:.4f} | {t2:.4f}/{m2:.4f} | {t3:.4f}/{m3:.4f}")
    return {"top1_all":float(top1),"md_all":float(md),
            "t1":float(t1),"m1":float(m1),"t2":float(t2),"m2":float(m2),"t3":float(t3),"m3":float(m3),
            "seg":[seg1,seg2,seg3]}

@torch.no_grad()
def baseline_eval_full(val_loader,cfg:Cfg):
    dev=cfg.device
    def agg(pred_fn,name):
        hit_sum=md_sum=0.0; cnt=0
        for batch in val_loader:
            vp_hist=batch["vp_hist"].to(dev).float()
            target =batch["target"].to(dev).float()
            pred_xy=pred_fn(vp_hist)
            m=metrics_top1_md(pred_xy,target,cfg.GRID_H,cfg.GRID_W)
            B=vp_hist.size(0); hit_sum+=m["top1_all"]*B; md_sum+=m["md_all"]*B; cnt+=B
        print(f"[Baseline FullVal][{name}] top1={hit_sum/max(cnt,1):.4f}  md_wrap={md_sum/max(cnt,1):.4f}")
    agg(lambda h: build_copylast_prior(h,cfg.T_future), "CopyLast")
    agg(lambda h: build_constvel_nodamp_laststep(h,cfg.T_future), "ConstVel(k=1,no-damp)")
    agg(lambda h: build_constvel_prior(h,cfg.velocity_k,cfg.damped_decay,cfg.T_future)[0], "ConstVel-like(k=5,γ=0.92)")

# ========== 单次训练 ==========
def train_validate_one(cfg:Cfg, run_name:str, out_dir_ckpt:str):
    set_seed(cfg.seed)
    train_loader,val_loader=get_train_val_loaders(cfg)

    print("\n[Sanity] Evaluating baselines on FULL validation set ...")
    baseline_eval_full(val_loader,cfg)

    print(f"[Run] Training {run_name} (seed={cfg.seed}) ...")
    model=ResidualGRU_MoP_Video(cfg).to(cfg.device)
    optim=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    sch=CosineAnnealingLR(optim,T_max=cfg.num_epochs,eta_min=cfg.lr*0.1)
    ema=EMA(model,cfg.ema_decay) if cfg.enable_ema else None

    best=-1e9; best_ep=-1; best_val=None; hist=[]
    for ep in range(1,cfg.num_epochs+1):
        tr=train_one_epoch(model,train_loader,optim,cfg,ep,ema)
        va=validate(model,val_loader,cfg,ema)
        score=va["top1_all"] - 0.05*va["md_all"] - cfg.score_md03_coef*va["m3"]
        hist.append({"epoch":ep,"train":tr,"val":va,"score":float(score)})

        if score>best:
            best=score; best_ep=ep; best_val=va
            os.makedirs(out_dir_ckpt,exist_ok=True)
            torch.save({"cfg":asdict(cfg),"state_dict":model.state_dict()},
                       os.path.join(out_dir_ckpt, f"{run_name}_seed{cfg.seed}_best.pth"))
            print(f"[Save] {out_dir_ckpt}/{run_name}_seed{cfg.seed}_best.pth")
        sch.step()

    return {"name":run_name,"seed":cfg.seed,"best_epoch":best_ep,"best_val":best_val,"history":hist}

# ========== Sweep（3 种子 + 消融） ==========
def sweep():
    out_dir_log = "logs/T30_v3A_Attempt3"
    out_dir_ckpt = "ckpts/T30_v3A_Attempt3"
    os.makedirs(out_dir_log, exist_ok=True); os.makedirs(out_dir_ckpt, exist_ok=True)

    VARIANTS = [
        ("T30_v3A_full",             dict(enable_video=True,  use_future_video=True,  enable_cv_gain=True)),
        ("T30_v3A_hist_only_video",  dict(enable_video=True,  use_future_video=False, enable_cv_gain=True)),
        ("T30_v3A_no_cv_gain",       dict(enable_video=True,  use_future_video=True,  enable_cv_gain=False)),
        ("T30_v3A_no_video",         dict(enable_video=False, use_future_video=False, enable_cv_gain=False)),
    ]
    SEEDS = [7, 17, 42]

    all_results=[]; flat_rows=[]
    CSV_HEADER = [
        "variant","seed","top1_all","md_all",
        "top1_01","md_01","top1_02","md_02","top1_03","md_03",
        "best_epoch","seg_01","seg_02","seg_03"
    ]

    for vname, overrides in VARIANTS:
        for sd in SEEDS:
            cfg = Cfg()
            assert cfg.T_future>=30 and cfg.process_frame_nums>=60
            for k,v in overrides.items(): setattr(cfg, k, v)
            cfg.seed = sd

            res = train_validate_one(cfg, vname, out_dir_ckpt)
            all_results.append(res)

            va = res["best_val"]; seg = va["seg"]
            flat_rows.append([
                vname, sd,
                f'{va["top1_all"]:.6f}', f'{va["md_all"]:.6f}',
                f'{va["t1"]:.6f}', f'{va["m1"]:.6f}',
                f'{va["t2"]:.6f}', f'{va["m2"]:.6f}',
                f'{va["t3"]:.6f}', f'{va["m3"]:.6f}',
                res["best_epoch"], seg[0], seg[1], seg[2]
            ])

    with open(os.path.join(out_dir_log, "sweep_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    flat_csv = os.path.join(out_dir_log, "sweep_flat.csv")
    with open(flat_csv, "w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(CSV_HEADER); w.writerows(flat_rows)
    print(f"[LOG] Wrote flat results to {flat_csv}")

    # 聚合 mean±std（population）
    agg = {}
    for row in flat_rows:
        vname=row[0]
        if vname not in agg: agg[vname] = {"top1_all":[], "md_all":[], "t1":[], "m1":[], "t2":[], "m2":[], "t3":[], "m3":[]}
        agg[vname]["top1_all"].append(float(row[2])); agg[vname]["md_all"].append(float(row[3]))
        agg[vname]["t1"].append(float(row[4])); agg[vname]["m1"].append(float(row[5]))
        agg[vname]["t2"].append(float(row[6])); agg[vname]["m2"].append(float(row[7]))
        agg[vname]["t3"].append(float(row[8])); agg[vname]["m3"].append(float(row[9]))

    def m_s(a):
        m=mean(a); s=(sum((x-m)**2 for x in a)/len(a))**0.5
        return f"{m:.4f}", f"{s:.4f}"

    agg_csv = os.path.join(out_dir_log, "sweep_agg.csv")
    with open(agg_csv, "w", newline="", encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["variant",
                    "top1_mean","top1_std","md_mean","md_std",
                    "top1_01_mean","top1_01_std","md_01_mean","md_01_std",
                    "top1_02_mean","top1_02_std","md_02_mean","md_02_std",
                    "top1_03_mean","top1_03_std","md_03_mean","md_03_std"])
        for vname, d in agg.items():
            t1m,t1s = m_s(d["top1_all"]); mdm,mds = m_s(d["md_all"])
            a1m,a1s = m_s(d["t1"]); m1m,m1s = m_s(d["m1"])
            a2m,a2s = m_s(d["t2"]); m2m,m2s = m_s(d["m2"])
            a3m,a3s = m_s(d["t3"]); m3m,m3s = m_s(d["m3"])
            w.writerow([vname, t1m,t1s, mdm,mds,
                        a1m,a1s, m1m,m1s,
                        a2m,a2s, m2m,m2s,
                        a3m,a3s, m3m,m3s])
    print(f"[LOG] Wrote aggregated results to {agg_csv}")

# ========== 入口 ==========
if __name__ == "__main__":
    sweep()
