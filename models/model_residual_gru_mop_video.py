import torch
import torch.nn as nn
from cavp.config import Cfg
from cavp.utils.geo import circ_diff, circ_add, clamp01

from .video_encoder import VideoEncoder

class ResidualGRU_MoP_Video(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__(); self.cfg = cfg; H = cfg.model_hidden
        self.enc = nn.GRU(4, H, 1, batch_first=True); self.dec = nn.GRU(3, H, 1, batch_first=True)
        self.drop = nn.Dropout(cfg.dropout_p)
        self.head_pos = nn.Sequential(nn.Linear(H, H), nn.ReLU(inplace=True), nn.Dropout(cfg.dropout_p), nn.Linear(H, 2))

        self.video_enc = VideoEncoder(64, cfg.vid_hidden // 2, cfg.vid_hidden, cfg.hist_len)
        self.film_gamma = nn.Sequential(nn.Linear(cfg.vid_hidden, cfg.film_dim), nn.ReLU(inplace=True), nn.Dropout(cfg.dropout_p), nn.Linear(cfg.film_dim, H))
        self.film_beta  = nn.Sequential(nn.Linear(cfg.vid_hidden, cfg.film_dim), nn.ReLU(inplace=True), nn.Dropout(cfg.dropout_p), nn.Linear(cfg.film_dim, H))
        self.hist_inject = nn.Linear(cfg.vid_hidden, H)

        # Dual-scale fusion gate
        self.alpha_gate = nn.Sequential(
            nn.Linear(H + 2, H // 2), nn.ReLU(inplace=True), nn.Dropout(cfg.dropout_p),
            nn.Linear(H // 2, 1), nn.Sigmoid()
        )
        # Video projection for gates/gain
        self.gate_vid_proj = nn.Linear(cfg.vid_hidden, 16)
        self.head_gate = nn.Sequential(nn.Linear(H + 2 + 16, H // 2), nn.ReLU(inplace=True), nn.Dropout(cfg.dropout_p), nn.Linear(H // 2, 1), nn.Sigmoid())
        self.head_gain = nn.Sequential(nn.Linear(H + 16, H // 2), nn.ReLU(inplace=True), nn.Dropout(cfg.dropout_p), nn.Linear(H // 2, 1), nn.Sigmoid())

    def forward(self, hist_xy, prior_cv_xy, prior_cl_xy, video=None):
        B, Th, _ = hist_xy.shape; Tf = prior_cv_xy.shape[1]; dev = hist_xy.device

        # Encode history with deltas
        dx_hist = torch.zeros_like(hist_xy)
        dx_hist[:, 1:, 0] = circ_diff(hist_xy[:, 1:, 0], hist_xy[:, :-1, 0])
        dx_hist[:, 1:, 1] = (hist_xy[:, 1:, 1] - hist_xy[:, :-1, 1]).clamp(-1.0, 1.0)
        enc_in = torch.cat([hist_xy, dx_hist], dim=-1); _, h = self.enc(enc_in)

        # Normalized time
        t_norm = torch.linspace(1, Tf, steps=Tf, device=dev, dtype=hist_xy.dtype) / Tf
        t_norm_ = t_norm.view(1, Tf, 1).expand(B, Tf, 1)

        # Video features
        if self.cfg.enable_video and (video is not None):
            fut_base, fut_fast, fut_slow, hist_sum = self.video_enc(
                video, self.cfg.fast_ks, self.cfg.slow_ks, self.cfg.smooth_ks
            )
            if not self.cfg.use_future_video:
                hist_rep = hist_sum.unsqueeze(1).expand(B, Tf, -1).contiguous()
                fut_base = fut_fast = fut_slow = hist_rep

            # Prior step deltas as "speed prior"
            prior_d = torch.zeros_like(prior_cv_xy)
            prior_d[:, 0, 0] = circ_diff(prior_cv_xy[:, 0, 0], hist_xy[:, -1, 0])
            prior_d[:, 0, 1] = (prior_cv_xy[:, 0, 1] - hist_xy[:, -1, 1]).clamp(-1.0, 1.0)
            if Tf > 1:
                prior_d[:, 1:, 0] = circ_diff(prior_cv_xy[:, 1:, 0], prior_cv_xy[:, :-1, 0])
                prior_d[:, 1:, 1] = (prior_cv_xy[:, 1:, 1] - prior_cv_xy[:, :-1, 1]).clamp(-1.0, 1.0)
            speed = torch.sqrt(prior_d[..., 0]**2 + prior_d[..., 1]**2 + 1e-12).unsqueeze(-1)
            alpha_in = torch.cat([speed, t_norm_], dim=-1)
            ctx = h.transpose(0, 1).expand(B, Tf, -1)
            alpha = self.alpha_gate(torch.cat([ctx, alpha_in], dim=-1)).squeeze(-1)
            alpha = torch.clamp(alpha + self.cfg.alpha_time_bias * t_norm.view(1, Tf), 0.0, 1.0)
            vid_fut = alpha.unsqueeze(-1) * fut_slow + (1.0 - alpha).unsqueeze(-1) * fut_fast  # (B,Tf,V)

            # FiLM modulation
            gamma = self.film_gamma(vid_fut); beta = self.film_beta(vid_fut)
            s_film = (0.6 + 1.0 * t_norm).view(1, Tf, 1)
            gamma = self.drop(gamma * s_film); beta = self.drop(beta * s_film)

            gate_vid = self.gate_vid_proj(vid_fut)
            h = h + self.hist_inject(hist_sum).view(1, B, -1)
        else:
            gamma = beta = None
            gate_vid = torch.zeros(B, Tf, 16, device=dev, dtype=hist_xy.dtype)

        # Decoder input: prior deltas + t_norm
        prior_d = torch.zeros_like(prior_cv_xy)
        prior_d[:, 0, 0] = circ_diff(prior_cv_xy[:, 0, 0], hist_xy[:, -1, 0])
        prior_d[:, 0, 1] = (prior_cv_xy[:, 0, 1] - hist_xy[:, -1, 1]).clamp(-1.0, 1.0)
        if Tf > 1:
            prior_d[:, 1:, 0] = circ_diff(prior_cv_xy[:, 1:, 0], prior_cv_xy[:, :-1, 0])
            prior_d[:, 1:, 1] = (prior_cv_xy[:, 1:, 1] - prior_cv_xy[:, :-1, 1]).clamp(-1.0, 1.0)
        dec_in = torch.cat([prior_d, t_norm_], dim=-1)
        dec_out, _ = self.dec(dec_in, h)
        if (gamma is not None) and (beta is not None):
            dec_out = dec_out * (1 + gamma) + beta

        # Time-shaped cv_gain (triangular widen mid)
        cv_gain = None
        if self.cfg.enable_cv_gain:
            base_min, base_max = self.cfg.cv_gain_min, self.cfg.cv_gain_max
            peak = 2.0 / 3.0
            rel = torch.where(t_norm <= peak, t_norm / peak, (1.0 - (t_norm - peak) / (1.0 - peak)))
            w_tri = 1.0 + self.cfg.gain_widen_mid * rel
            lo = 1.0 - (1.0 - base_min) * w_tri
            hi = 1.0 + (base_max - 1.0) * w_tri
            lo = lo.clamp(min=self.cfg.gain_clip_min); hi = hi.clamp(max=self.cfg.gain_clip_max)

            g_in = torch.cat([dec_out, gate_vid], dim=-1)
            g_raw = self.head_gain(g_in).squeeze(-1)  # (B,Tf)
            cv_gain = lo.view(1, Tf) + (hi.view(1, Tf) - lo.view(1, Tf)) * g_raw

            # Late decay
            rel_end = torch.clamp((t_norm - peak) / (1.0 - peak), min=0.0, max=1.0)
            end_decay = 1.0 - self.cfg.end_shrink * rel_end
            cv_gain = 1.0 + (cv_gain - 1.0) * end_decay.view(1, Tf)

            # Rebuild prior_cv_xy with adjusted gain
            pos_x = hist_xy[:, -1, 0]; pos_y = hist_xy[:, -1, 1]; cv_adj = []
            for t in range(Tf):
                dx = prior_d[:, t, 0] * cv_gain[:, t]; dy = prior_d[:, t, 1] * cv_gain[:, t]
                pos_x = circ_add(pos_x, dx); pos_y = clamp01(pos_y + dy)
                cv_adj.append(torch.stack([pos_x, pos_y], dim=-1))
            prior_cv_xy = torch.stack(cv_adj, dim=1)

        # Saccade-aware residual scaling
        speed = torch.sqrt(prior_d[..., 0]**2 + prior_d[..., 1]**2 + 1e-12)
        gate_sa = torch.sigmoid((speed - self.cfg.move_thr) / (self.cfg.gate_sigma + 1e-8))
        dyn_scale = (self.cfg.residual_scale + self.cfg.residual_boost * gate_sa).clamp(max=self.cfg.residual_scale_max).unsqueeze(-1)

        # Mixture-of-priors gate
        aux = torch.stack([speed, t_norm_.squeeze(-1)], dim=-1)
        gate_in = torch.cat([dec_out, aux, gate_vid], dim=-1)
        g = self.head_gate(gate_in)  # (B,Tf,1)

        # Blend priors + residual
        dx_mix = circ_diff(prior_cv_xy[..., 0], prior_cl_xy[..., 0]) * g.squeeze(-1)
        x_mix = circ_add(prior_cl_xy[..., 0], dx_mix)
        y_mix = clamp01(prior_cl_xy[..., 1] + g.squeeze(-1) * (prior_cv_xy[..., 1] - prior_cl_xy[..., 1]))
        prior_mix = torch.stack([x_mix, y_mix], dim=-1)

        res = torch.tanh(self.head_pos(dec_out)) * dyn_scale
        x = circ_add(prior_mix[:, :, 0], res[:, :, 0]); y = clamp01(prior_mix[:, :, 1] + res[:, :, 1])
        pred = torch.stack([x, y], dim=-1)
        return pred, g, cv_gain
