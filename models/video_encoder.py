import torch
import torch.nn as nn
import torch.nn.functional as F

class FrameCNN(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, out_dim, 3, 2, 1), nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
        ); self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        f = self.net(x)
        return self.gap(f).squeeze(-1).squeeze(-1)

class VideoEncoder(nn.Module):
    def __init__(self, cnn_dim=64, t_hidden=128, out_dim=256, hist_len=30):
        super().__init__()
        self.cnn = FrameCNN(out_dim=cnn_dim)
        self.tgru = nn.GRU(cnn_dim, t_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * t_hidden, out_dim)
        self.hist_len = hist_len

    @staticmethod
    def smooth_time(x, ks: int):
        if ks <= 1: return x
        pad = ks // 2
        xbt = x.transpose(1, 2)  # (B,V,T)
        xbt = F.avg_pool1d(F.pad(xbt, (pad, pad), mode="replicate"), kernel_size=ks, stride=1, padding=0)
        return xbt.transpose(1, 2).contiguous()

    def forward(self, video, ks_fast: int, ks_slow: int, base_ks: int):
        B, C, T, H, W = video.shape
        x = video.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        f = self.cnn(x).view(B, T, -1)
        tfeat, _ = self.tgru(f); tfeat = self.proj(tfeat)       # (B,T,feat)
        hist_sum = tfeat[:, :self.hist_len, :].mean(dim=1)      # (B,feat)
        fut_raw = tfeat[:, self.hist_len:, :]                    # (B,Tf,feat)
        fut_base = self.smooth_time(fut_raw, base_ks)
        fut_fast = self.smooth_time(fut_raw, ks_fast)
        fut_slow = self.smooth_time(fut_raw, ks_slow)
        return fut_base, fut_fast, fut_slow, hist_sum
