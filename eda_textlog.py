#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eda_textlog.py — 360° 视口预测数据 EDA（文本日志·加强版 + 稀有运动分析）
- 仅输出一份**纯文本报告**（打印到终端 + 写入文件），便于复制给我分析
- 支持：字典 batch，字段 vp_hist(必需), target(可选), video_id(str), user_id(str)
- 固定缺省 Cfg：H=144, W=256, GRID_H=9, GRID_W=16, hist_len=30, T_future=60, process_frame_nums=90, batch_size=16
- 新增：按**每条序列**自适应阈值选取 Top-q 运动步（默认 q=0.2），并输出方向/转角/爆发段/起停/运动占用与转移的文本摘要
"""
import os, json, math, argparse
import numpy as np
import torch

# -------------------- Dataloader --------------------
try:
    from data.dataloader_stub import get_train_val_loaders
except Exception as e:
    raise RuntimeError("未能导入 get_train_val_loaders，请确认 data/dataloader_stub.py 可用：%s" % e)

# -------------------- Utils --------------------
EPS = 1e-8

def to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

def circ_delta(x1, x2):
    """水平经度 x∈[0,1] 的圆周最小有向差，范围 [-0.5,0.5)"""
    d = (x2 - x1 + 0.5) % 1.0 - 0.5
    return d

def grid_index_from_xy(x, y, gw=16, gh=9):
    gx = np.clip((x * gw).astype(int), 0, gw-1)
    gy = np.clip((y * gh).astype(int), 0, gh-1)
    return gy, gx  # (row, col)

def entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())

def summarize_vec(vec):
    vec = np.asarray(vec)
    if vec.size == 0:
        return {"count": 0}
    return {
        "count": int(vec.size),
        "mean": float(np.mean(vec)),
        "std":  float(np.std(vec)),
        "p10":  float(np.percentile(vec, 10)),
        "p50":  float(np.percentile(vec, 50)),
        "p90":  float(np.percentile(vec, 90)),
        "p95":  float(np.percentile(vec, 95)),
        "max":  float(np.max(vec)),
    }

def topk_cells(H, k=10):
    """返回 [(r,c,count,ratio), ...]，按计数降序；同时返回总计数"""
    flat = H.ravel()
    total = float(flat.sum()) + 1e-12
    idx = np.argsort(-flat)[:k]
    out = []
    W = H.shape[1]
    for i in idx:
        r = int(i // W); c = int(i % W)
        cnt = float(flat[i])
        if cnt <= 0: break
        out.append((r, c, int(cnt), float(cnt/total)))
    return out, int(total)

def sector_counts(angles, bins=8):
    """angles in radians [-pi,pi]; return list of (sector_index, count, ratio)"""
    if len(angles) == 0:
        return [], 0
    ang = np.mod(np.asarray(angles) + 2*np.pi, 2*np.pi)
    hist, edges = np.histogram(ang, bins=bins, range=(0, 2*np.pi))
    total = int(hist.sum())
    out = [(i, int(hist[i]), float(hist[i]/max(1,total))) for i in range(bins)]
    return out, total

# -------------------- Target Probe --------------------
class TargetProbe:
    """推断 target 的形态与 shape"""
    def __init__(self):
        self.kind = None  # 'grid_dist', 'grid_index', 'cont_xy', 'unknown'
        self.shape = None

    def infer(self, target):
        if target is None:
            self.kind, self.shape = None, None
            return self.kind
        t = torch.as_tensor(target)
        self.shape = tuple(t.shape)
        if t.ndim == 4:
            self.kind = 'grid_dist'    # (B, T, GH, GW)
        elif t.ndim == 3 and t.shape[-1] == 2:
            self.kind = 'cont_xy'      # (B, T, 2)
        elif t.ndim == 2:
            self.kind = 'grid_index'   # (B, T)
        else:
            self.kind = 'unknown'
        return self.kind

# -------------------- Aggregators --------------------
class Aggregates:
    def __init__(self, name='split', gh=9, gw=16, T_future=60, move_topq=0.2):
        self.name = name
        self.gh, self.gw, self.T_future = gh, gw, T_future
        self.move_topq = float(move_topq)
        self.sample_count = 0   # 累计样本条目数（sum of B）
        # 质量
        self.nan_hist = 0
        self.out_of_range = 0
        # 运动学（全步）
        self.speed_h = []  # |Δx|（圆）
        self.speed_v = []  # |Δy|
        self.acc_h = []
        self.acc_v = []
        self.heading = []  # Δx 有向
        # 停留
        self.dwell_frames = []
        # 占用（全步 & 未来）
        self.occ_hist = np.zeros((gh, gw), dtype=np.float64)
        self.occ_future = np.zeros((self.T_future, gh, gw), dtype=np.float64)
        self.entropy_future = []
        # 转移矩阵（全步）
        self.trans = np.zeros((gh*gw, gh*gw), dtype=np.float64)

        # ====== 运动模式（稀有运动步） ======
        self.total_steps = 0
        self.move_steps = 0
        self.move_step_mag = []      # |dx|+|dy|（只统计运动步）
        self.move_heading_angle = [] # atan2(dy, dx) in radians（只运动步）
        self.turn_angles = []        # 相邻运动步向量夹角
        self.burst_len = []          # 连续运动步段长度
        self.burst_disp = []         # 连续运动步段总位移（|dx|+|dy|累计）
        self.start_events = 0        # 停留->运动 次数
        self.stop_events  = 0        # 运动->停留 次数
        self.move_occ_hist = np.zeros((gh, gw), dtype=np.float64)   # 仅运动步位置占用（对齐 t+1）
        self.move_trans = np.zeros((gh*gw, gh*gw), dtype=np.float64) # 仅运动步转移

    def _movement_mask(self, step_mag):
        """基于每条序列的分位数阈值选取运动步（稀有）"""
        if step_mag.size == 0:
            return np.zeros_like(step_mag, dtype=bool), 0.0
        thr = np.percentile(step_mag, 100*(1.0 - self.move_topq))
        return (step_mag >= thr) & (step_mag > 0), float(thr)

    def update_hist(self, vp_hist):
        # vp_hist: (B, Th=30, 2)，x wrap，y 不 wrap
        X = torch.as_tensor(vp_hist)
        B, Th, _ = X.shape
        x = X[..., 0].numpy()
        y = X[..., 1].numpy()

        # 质量
        nan_mask = np.isnan(x) | np.isnan(y)
        self.nan_hist += int(nan_mask.sum())
        oor = ((x < -EPS) | (x > 1+EPS) | (y < -EPS) | (y > 1+EPS))
        self.out_of_range += int(oor.sum())

        # 运动学（全步）
        if Th >= 2:
            dx = circ_delta(x[:, :-1], x[:, 1:])
            dy = y[:, 1:] - y[:, :-1]
            step_mag = np.abs(dx) + np.abs(dy)
            self.speed_h.extend(np.abs(dx).ravel())
            self.speed_v.extend(np.abs(dy).ravel())
            self.heading.extend(dx.ravel())
            # 加速度
            if Th >= 3:
                ddx = dx[:, 1:] - dx[:, :-1]
                ddy = dy[:, 1:] - dy[:, :-1]
                self.acc_h.extend(np.abs(ddx).ravel())
                self.acc_v.extend(np.abs(ddy).ravel())
            # 停留：|dx|+|dy| 小于自适应阈值的连续段长度
            thr_dwell = np.percentile(step_mag, 30) if step_mag.size > 0 else 0.0
            for seq in step_mag:
                run = 0
                for v in seq:
                    if v < thr_dwell:
                        run += 1
                    else:
                        if run > 0:
                            self.dwell_frames.append(run)
                            run = 0
                if run > 0:
                    self.dwell_frames.append(run)

            # ====== 稀有运动步筛选与统计 ======
            for b in range(B):
                sm = step_mag[b]  # (Th-1,)
                m_mask, m_thr = self._movement_mask(sm)
                self.total_steps += sm.size
                m_count = int(m_mask.sum())
                self.move_steps += m_count

                # 运动步标量与方向
                if m_count > 0:
                    mdx = dx[b][m_mask]
                    mdy = dy[b][m_mask]
                    self.move_step_mag.extend((np.abs(mdx) + np.abs(mdy)).tolist())
                    ang = np.arctan2(mdy, mdx)  # [-pi, pi]
                    self.move_heading_angle.extend(ang.tolist())

                    # turn angles：仅在相邻运动步之间计算
                    if mdx.size >= 2:
                        v1 = np.stack([mdx[:-1], mdy[:-1]], axis=-1)  # (M-1,2)
                        v2 = np.stack([mdx[1:],  mdy[1:]],  axis=-1)
                        # 角度 = atan2(|v1×v2|, v1·v2)
                        cross = v1[:,0]*v2[:,1] - v1[:,1]*v2[:,0]
                        dot   = v1[:,0]*v2[:,0] + v1[:,1]*v2[:,1]
                        ang12 = np.arctan2(np.abs(cross), dot)  # [0,pi]
                        self.turn_angles.extend(ang12.tolist())

                # 运动爆发段（连续 True 的段） & 起停
                run_len = 0
                run_disp = 0.0
                if sm.size>0 and m_mask[0]==1:
                    self.start_events += 1
                for t in range(sm.size):
                    state = 1 if m_mask[t] else 0
                    if state == 1:
                        run_len += 1
                        run_disp += float(sm[t])
                        # 运动占用：对齐下一帧位置 (t+1)
                        gy, gx = grid_index_from_xy(np.array([x[b, t+1]]), np.array([y[b, t+1]]), self.gw, self.gh)
                        self.move_occ_hist[gy[0], gx[0]] += 1
                    if t < sm.size-1:
                        if state==0 and m_mask[t+1]==1:
                            self.start_events += 1
                        if state==1 and m_mask[t+1]==0:
                            self.stop_events += 1
                    if (state==1 and (t==sm.size-1 or m_mask[t+1]==0)) and run_len>0:
                        self.burst_len.append(run_len)
                        self.burst_disp.append(run_disp)
                        run_len, run_disp = 0, 0.0

                # 运动步转移矩阵（仅取发生运动的 t 步的转移 t->t+1）
                if m_count > 0:
                    gy_all, gx_all = grid_index_from_xy(x[b], y[b], self.gw, self.gh)
                    flat = gy_all * self.gw + gx_all
                    idxs = np.where(m_mask)[0]  # 0..Th-2
                    for t in idxs:
                        self.move_trans[flat[t], flat[t+1]] += 1

        # 占用 + 转移（全步）
        x_clip = np.clip(x, 0, 1)
        y_clip = np.clip(y, 0, 1)
        gy, gx = grid_index_from_xy(x_clip, y_clip, self.gw, self.gh)
        for b in range(B):
            for t in range(Th):
                self.occ_hist[gy[b, t], gx[b, t]] += 1
            if Th >= 2:
                flat = gy[b, :] * self.gw + gx[b, :]
                for t in range(Th-1):
                    self.trans[flat[t], flat[t+1]] += 1

        self.sample_count += B

    def update_future(self, target, probe):
        if target is None or probe.kind is None:
            return
        t = torch.as_tensor(target).numpy()
        if probe.kind == 'grid_dist':
            B, T, GH, GW = t.shape
            T = min(T, self.T_future)
            self.occ_future[:T] += t[:, :T].sum(axis=0)
            for step in range(T):
                p = t[:, step].reshape(B, -1)
                p = p / (p.sum(-1, keepdims=True) + 1e-12)
                self.entropy_future.append(float(np.mean([entropy(pi) for pi in p])))
        elif probe.kind == 'grid_index':
            B, T = t.shape
            T = min(T, self.T_future)
            for step in range(T):
                idx = t[:, step]
                gy = (idx // self.gw).clip(0, self.gh-1)
                gx = (idx %  self.gw).clip(0, self.gw-1)
                for b in range(B):
                    self.occ_future[step, gy[b], gx[b]] += 1
            K = self.gh * self.gw
            self.entropy_future = [math.log(K)] * T
        elif probe.kind == 'cont_xy':
            B, T, _ = t.shape
            T = min(T, self.T_future)
            x = np.clip(t[:, :T, 0], 0, 1)
            y = np.clip(t[:, :T, 1], 0, 1)
            for step in range(T):
                gy, gx = grid_index_from_xy(x[:, step], y[:, step], self.gw, self.gh)
                H = np.zeros((self.gh, self.gw), dtype=np.float64)
                for b in range(B):
                    H[gy[b], gx[b]] += 1
                self.occ_future[step] += H
            for step in range(T):
                p = self.occ_future[step].ravel()
                p = p / (p.sum() + 1e-12)
                self.entropy_future.append(entropy(p))

# -------------------- Drift / Leakage --------------------
def kl_divergence_from_hists(a, b, bins=100, value_range=None):
    if len(a) == 0 and len(b) == 0:
        return 0.0
    if len(a) == 0 or len(b) == 0:
        return float('nan')
    if value_range is None:
        lo = min(np.min(a), np.min(b))
        hi = max(np.max(a), np.max(b))
        if not np.isfinite(lo): lo = 0.0
        if not np.isfinite(hi): hi = 1.0
        value_range = (lo, hi if hi>lo else lo+1e-6)
    ha, _ = np.histogram(a, bins=bins, range=value_range)
    hb, _ = np.histogram(b, bins=bins, range=value_range)
    pa = (ha + 1e-12) / (ha.sum() + 1e-12)
    pb = (hb + 1e-12) / (hb.sum() + 1e-12)
    return float(np.sum(pa * np.log(pa / pb)))

def chi2_distance(p, q):
    p = p.astype(np.float64); q = q.astype(np.float64)
    p = p / (p.sum() + 1e-12); q = q / (q.sum() + 1e-12)
    denom = p + q + 1e-12
    return float(0.5 * np.sum((p - q)**2 / denom))

def compute_train_val_drift(aggs_train, aggs_val):
    drift = {}
    drift['KL_speed_h'] = kl_divergence_from_hists(aggs_train.speed_h, aggs_val.speed_h, bins=100, value_range=(0.0, 0.5))
    # 速度垂直动态范围
    vr_v = None
    if len(aggs_train.speed_v) and len(aggs_val.speed_v):
        lo = min(np.min(aggs_train.speed_v), np.min(aggs_val.speed_v))
        hi = max(np.max(aggs_train.speed_v), np.max(aggs_val.speed_v))
        vr_v = (float(lo), float(hi if hi>lo else lo+1e-6))
    drift['KL_speed_v'] = kl_divergence_from_hists(aggs_train.speed_v, aggs_val.speed_v, bins=100, value_range=vr_v)
    drift['KL_heading'] = kl_divergence_from_hists(aggs_train.heading, aggs_val.heading, bins=100, value_range=(-0.5, 0.5))
    # 历史占用 χ²
    drift['Chi2_occ_hist'] = chi2_distance(aggs_train.occ_hist.ravel(), aggs_val.occ_hist.ravel())
    # 未来 t=60 占用 χ²（若可）
    t_last = min(aggs_train.T_future-1, aggs_val.T_future-1)
    drift['Chi2_occ_future_last'] = chi2_distance(aggs_train.occ_future[t_last].ravel(), aggs_val.occ_future[t_last].ravel())
    # 未来熵均值
    if len(aggs_train.entropy_future) and len(aggs_val.entropy_future):
        drift['Entropy_future_mean_train'] = float(np.mean(aggs_train.entropy_future))
        drift['Entropy_future_mean_val']   = float(np.mean(aggs_val.entropy_future))
    return drift

def compute_train_val_movement_drift(aggs_train, aggs_val):
    """新增：运动子集上的漂移比较"""
    drift = {}
    # 运动步位移幅度 KL
    vr = None
    if len(aggs_train.move_step_mag) and len(aggs_val.move_step_mag):
        lo = min(np.min(aggs_train.move_step_mag), np.min(aggs_val.move_step_mag))
        hi = max(np.max(aggs_train.move_step_mag), np.max(aggs_val.move_step_mag))
        vr = (float(lo), float(hi if hi>lo else lo+1e-6))
    drift['KL_move_step_mag'] = kl_divergence_from_hists(aggs_train.move_step_mag, aggs_val.move_step_mag, bins=100, value_range=vr)
    # 转角 KL（[0,pi]）
    drift['KL_turn_angle'] = kl_divergence_from_hists(aggs_train.turn_angles, aggs_val.turn_angles, bins=72, value_range=(0.0, math.pi))
    # 运动占用 χ²
    drift['Chi2_move_occ_hist'] = chi2_distance(aggs_train.move_occ_hist.ravel(), aggs_val.move_occ_hist.ravel())
    # 运动比率差异
    r_train = aggs_train.move_steps / (aggs_train.total_steps + 1e-12)
    r_val   = aggs_val.move_steps   / (aggs_val.total_steps   + 1e-12)
    drift['move_ratio_train'] = float(r_train)
    drift['move_ratio_val']   = float(r_val)
    drift['move_ratio_gap']   = float(abs(r_train - r_val))
    return drift

# -------------------- Reporter --------------------
def render_report(cfg, info):
    """
    info: dict {
       'probe': {...}, 'overlap': {...},
       'drift': {...}, 'move_drift': {...},
       'train': {'aggs': Aggregates, 'users': set, 'videos': set},
       'val'  : {'aggs': Aggregates, 'users': set, 'videos': set},
       'topk': int, 'picks': list[int], 'move_topq': float
    }
    返回文本字符串
    """
    lines = []
    def L(s=''): lines.append(s)

    # Header
    L("# 360° 视口预测 EDA 文本报告（含稀有运动分析）")
    L("")
    L("## 配置（Cfg）")
    for k in ['H','W','GRID_H','GRID_W','hist_len','T_future','process_frame_nums','batch_size']:
        L(f"- {k}: {getattr(cfg,k,None)}")
    L(f"- move_topq（每序列取 Top-q 作为运动步）: {info['move_topq']}")
    L("")

    # Probe
    L("## 标签形态探测（Probe）")
    L(f"- train target: kind={info['probe']['train']['target_kind']}, shape={info['probe']['train']['target_shape']}")
    L(f"- val   target: kind={info['probe']['val']['target_kind']}, shape={info['probe']['val']['target_shape']}")
    L("")

    def section_split(tag, dat):
        ag = dat['aggs']
        users = dat['users']; videos = dat['videos']
        L(f"## Split: {tag}")
        L(f"- 样本数（序列条目）：{ag.sample_count}")
        L(f"- 去重用户数：{len(users)}；去重视频数：{len(videos)}")
        L("")

        # 质量
        L("### 数据质量")
        total_hist = int(ag.occ_hist.sum())
        L(f"- 总历史帧计数：{total_hist}")
        L(f"- NaN 帧数：{ag.nan_hist}；越界帧数：{ag.out_of_range}")
        L("")

        # 运动学
        L("### 运动学统计（全步，逐步差分）")
        for name, vec in [('speed_h', ag.speed_h), ('speed_v', ag.speed_v), ('acc_h', ag.acc_h), ('acc_v', ag.acc_v), ('heading', ag.heading)]:
            stats = summarize_vec(vec)
            L(f"- {name}: {json.dumps(stats, ensure_ascii=False)}")
        L("")

        # 停留
        stats_dw = summarize_vec(ag.dwell_frames)
        L(f"### 停留分布（连续低运动步长段）\n- dwell_frames: {json.dumps(stats_dw, ensure_ascii=False)}")
        L("")

        # 空间分布：历史占用 Top-K
        L("### 历史占用（9x16）Top-K")
        top_hist, total_hist = topk_cells(ag.occ_hist, k=info['topk'])
        cover_hist = int((ag.occ_hist>0).sum())
        L(f"- 历史覆盖格数：{cover_hist} / {ag.gh*ag.gw}")
        L(f"- Top-{info['topk']} 热点 (row,col,count,ratio): {top_hist}")
        L("")

        # 未来占用（若有）
        picks = info['picks']
        L("### 未来占用 Top-K（若 target 可用）")
        if ag.occ_future.sum() > 0:
            for t in picks:
                t = min(t, ag.T_future-1)
                top_f, tot = topk_cells(ag.occ_future[t], k=info['topk'])
                cov = int((ag.occ_future[t]>0).sum())
                L(f"- t={t+1}: 覆盖格 {cov}/{ag.gh*ag.gw}；Top-{info['topk']}={top_f}")
        else:
            L("- 无未来标签或未统计未来分布")
        L("")

        # 未来熵
        if len(ag.entropy_future) > 0:
            mean_ent = float(np.mean(ag.entropy_future))
            ent_picks = {}
            for t in picks:
                if t < len(ag.entropy_future):
                    ent_picks[f"t={t+1}"] = float(ag.entropy_future[t])
            L(f"### 未来熵\n- 平均熵：{mean_ent:.6f}； 取样熵：{json.dumps(ent_picks, ensure_ascii=False)}")
        else:
            L("### 未来熵\n- 无")
        L("")

        # 转移矩阵摘要（全步）
        L("### 历史转移摘要（全步）")
        total_trans = float(ag.trans.sum())
        if total_trans > 0:
            diag_sum = float(np.trace(ag.trans))
            self_ratio = diag_sum / total_trans
            flat = ag.trans.ravel()
            idx = np.argsort(-flat)[:info['topk']]
            W = ag.gw
            items_str = []
            for i in idx:
                cnt = int(flat[i])
                if cnt == 0: break
                s_idx = int(i // (W*ag.gh)); t_idx = int(i % (W*ag.gh))
                sr, sc = divmod(s_idx, W); tr, tc = divmod(t_idx, W)
                items_str.append(f"({sr},{sc})->({tr},{tc})|count={cnt}|ratio={(cnt/total_trans):.6f}")
            L(f"- 总转移数：{int(total_trans)}；自转移比例（停留同格）：{self_ratio:.6f}")
            L(f"- Top-{info['topk']} 转移：[{'; '.join(items_str)}]")
        else:
            L("- 无可统计的转移")
        L("")

        # ===== 运动模式（稀有运动步） =====
        L("### 运动模式（稀有运动步 Top-q）")
        move_ratio = ag.move_steps / (ag.total_steps + 1e-12)
        L(f"- 运动步/总步：{ag.move_steps}/{ag.total_steps}（ratio={move_ratio:.6f}）")
        L(f"- 运动步幅度：{json.dumps(summarize_vec(ag.move_step_mag), ensure_ascii=False)}")
        sectors, stotal = sector_counts(ag.move_heading_angle, bins=8)
        L(f"- 方向分布（8扇区，0=东，每45°一扇区）：总计={stotal}，明细={sectors}")
        L(f"- 转角（相邻运动向量夹角，弧度）：{json.dumps(summarize_vec(ag.turn_angles), ensure_ascii=False)}")
        L(f"- 爆发段长度：{json.dumps(summarize_vec(ag.burst_len), ensure_ascii=False)}")
        L(f"- 爆发段累计位移：{json.dumps(summarize_vec(ag.burst_disp), ensure_ascii=False)}")
        L(f"- 起停统计：start_events={ag.start_events}, stop_events={ag.stop_events}")
        top_move, tot_move = topk_cells(ag.move_occ_hist, k=info['topk'])
        cov_move = int((ag.move_occ_hist>0).sum())
        L(f"- 运动占用覆盖格：{cov_move}/{ag.gh*ag.gw}；Top-{info['topk']}={top_move}")
        total_mtrans = float(ag.move_trans.sum())
        if total_mtrans > 0:
            flat = ag.move_trans.ravel()
            idx = np.argsort(-flat)[:info['topk']]
            W = ag.gw
            items_str = []
            for i in idx:
                cnt = int(flat[i])
                if cnt == 0: break
                s_idx = int(i // (W*ag.gh)); t_idx = int(i % (W*ag.gh))
                sr, sc = divmod(s_idx, W); tr, tc = divmod(t_idx, W)
                items_str.append(f"({sr},{sc})->({tr},{tc})|count={cnt}|ratio={(cnt/total_mtrans):.6f}")
            self_ratio_m = float(np.trace(ag.move_trans)) / total_mtrans
            L(f"- 运动转移：总数={int(total_mtrans)}；自转移比例={self_ratio_m:.6f}")
            L(f"- 运动 Top-{info['topk']} 转移：[{'; '.join(items_str)}]")
        else:
            L("- 运动转移：无")
        L("")

    # 各 split
    section_split('train', info['train'])
    section_split('val',   info['val'])

    # 泄露与漂移
    L("## Train/Val ID 泄露检查")
    ov = info['overlap']
    for k,v in ov.items():
        L(f"- {k}: {v}")
    L("")
    L("## Train/Val 分布漂移（全量）")
    dr = info['drift']
    for k,v in dr.items():
        L(f"- {k}: {v}")
    L("")
    L("## Train/Val 分布漂移（仅运动子集）")
    mdr = info['move_drift']
    for k,v in mdr.items():
        L(f"- {k}: {v}")
    L("")
    return "\n".join(lines)

# -------------------- Runner --------------------
def run(cfg, out_path='./eda_report.txt', max_batches=None, topk=15, picks=(0,9,29,59), move_topq=0.2):
    train_loader, val_loader = get_train_val_loaders(cfg)
    probe_train = TargetProbe()
    probe_val   = TargetProbe()

    aggs_train = Aggregates(name='train', gh=getattr(cfg, 'GRID_H', 9), gw=getattr(cfg, 'GRID_W', 16),
                            T_future=getattr(cfg, 'T_future', 60), move_topq=move_topq)
    aggs_val   = Aggregates(name='val',   gh=getattr(cfg, 'GRID_H', 9), gw=getattr(cfg, 'GRID_W', 16),
                            T_future=getattr(cfg, 'T_future', 60), move_topq=move_topq)

    users_train, users_val   = set(), set()
    videos_train, videos_val = set(), set()

    def _loop(loader, tag, aggs, probe, users_set, videos_set):
        nb = 0
        for batch in loader:
            if not isinstance(batch, dict):
                raise TypeError("期望 batch 为字典：包含 'vp_hist', 'target'(可选), 'video_id'(str), 'user_id'(str)")
            vp_hist = batch['vp_hist']
            target  = batch.get('target', None)
            video_id = batch.get('video_id', None)
            user_id  = batch.get('user_id',  None)

            aggs.update_hist(to_numpy(vp_hist))
            if target is not None:
                tgt_np = to_numpy(target)
                if probe.kind is None:
                    probe.infer(tgt_np)
                aggs.update_future(tgt_np, probe)

            # 记录 ID（需求为 str；若传 list/tuple 也兼容）
            if isinstance(user_id, (list, tuple)):
                for u in user_id: users_set.add(str(u))
            elif user_id is not None:
                users_set.add(str(user_id))

            if isinstance(video_id, (list, tuple)):
                for v in video_id: videos_set.add(str(v))
            elif video_id is not None:
                videos_set.add(str(video_id))

            nb += 1
            if max_batches and nb >= max_batches:
                break
        print(f'[{tag}] 扫描 {nb} 个 batch. target_kind={probe.kind}, shape={probe.shape}')
        return nb

    _loop(train_loader, 'train', aggs_train, probe_train, users_train, videos_train)
    _loop(val_loader,   'val',   aggs_val,   probe_val,   users_val,   videos_val)

    # 泄露 / 漂移
    overlap = {
        'users_train': len(users_train), 'users_val': len(users_val),
        'videos_train': len(videos_train), 'videos_val': len(videos_val),
        'users_overlap_count': len(users_train & users_val),
        'videos_overlap_count': len(videos_train & videos_val),
    }
    drift = compute_train_val_drift(aggs_train, aggs_val)
    move_drift = compute_train_val_movement_drift(aggs_train, aggs_val)

    # 汇总信息
    info = {
        'probe': {
            'train': {'target_kind': probe_train.kind, 'target_shape': probe_train.shape},
            'val'  : {'target_kind': probe_val.kind,   'target_shape': probe_val.shape},
        },
        'overlap': overlap,
        'drift': drift,
        'move_drift': move_drift,
        'train': {'aggs': aggs_train, 'users': users_train, 'videos': videos_train},
        'val'  : {'aggs': aggs_val,   'users': users_val,   'videos': videos_val},
        'topk': int(topk),
        'picks': list(picks),  # e.g., t=1,10,30,60
        'move_topq': float(move_topq),
    }

    text = render_report(cfg, info)

    # 输出
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print("\n" + "="*80 + "\n" + text + "\n" + "="*80 + f"\n报告已写入: {out_path}\n")

# -------------------- CLI --------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='./eda_report.txt', help='输出文本路径')
    parser.add_argument('--max_batches', type=int, default=None, help='最多扫描的 batch 数，None 为全量')
    parser.add_argument('--topk', type=int, default=15, help='Top-K 列表长度（热点格子/转移对）')
    parser.add_argument('--picks', type=str, default='0,9,29,59', help='展示的未来步索引(从0计)，如 0,9,29,59')
    parser.add_argument('--move_topq', type=float, default=0.2, help='每条序列按步长Top-q抽取“运动步”，例如 0.2 代表取Top 20%')
    args = parser.parse_args()

    class Cfg: pass
    cfg = Cfg()
    cfg.H = 144
    cfg.W = 256
    cfg.GRID_H = 9
    cfg.GRID_W = 16
    cfg.hist_len = 30
    cfg.T_future = 60
    cfg.process_frame_nums = 90
    cfg.batch_size = 16

    picks = tuple(int(x) for x in args.picks.split(',')) if args.picks else (0,9,29,59)
    run(cfg, out_path=args.out, max_batches=args.max_batches, topk=args.topk, picks=picks, move_topq=args.move_topq)
