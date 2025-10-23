from dataclasses import dataclass, asdict
import torch

@dataclass
class Cfg:
    # Data / grid
    H:int = 144; W:int = 256; GRID_H:int = 9; GRID_W:int = 16
    hist_len:int = 30; T_future:int = 30; process_frame_nums:int = 60
    batch_size:int = 16
    # Train
    num_epochs:int = 16; lr:float = 3e-4; weight_decay:float = 0.01
    grad_clip:float = 1.0; seed:int = 42
    device:str = "cuda" if torch.cuda.is_available() else "cpu"
    log_interval:int = 400
    # Priors
    velocity_k:int = 5; damped_decay:float = 0.92
    # Late-Heavy & scoring
    bucket_w_01:float = 0.90; bucket_w_02:float = 1.00; bucket_w_03:float = 1.25
    score_md03_coef:float = 0.03
    # Loss weights
    loss_w_pos:float = 1.0; loss_w_vel:float = 0.3
    # Residual / gating
    residual_scale:float = 0.08; residual_boost:float = 0.14; residual_scale_max:float = 0.22
    move_thr:float = 0.0270; gate_sigma:float = 0.015
    gate_tv_lambda:float = 3e-3
    # Video branch
    enable_video:bool = True; use_future_video:bool = True
    vid_hidden:int = 256; film_dim:int = 256; dropout_p:float = 0.12
    smooth_ks:int = 11
    # Backbone
    model_hidden:int = 160
    # cv_gain (base interval)
    enable_cv_gain:bool = True; cv_gain_min:float = 0.92; cv_gain_max:float = 1.08
    cv_gain_reg_lambda:float = 2.5e-4
    # Dual-scale & time schedule
    alpha_time_bias:float = 0.15
    fast_ks:int = 5; slow_ks:int = 21
    gain_widen_mid:float = 0.6
    gain_clip_min:float = 0.88; gain_clip_max:float = 1.12
    end_shrink:float = 0.5
    gain_reg_early:float = 0.5; gain_reg_late:float = 1.0
    # EMA
    enable_ema:bool = True; ema_decay:float = 0.999

def asdict_cfg(cfg: "Cfg"):
    return asdict(cfg)
