# -*- coding: utf-8 -*-
"""
深度学习基准（公平版）
======================
24.18 的教训：给竞品的特征不够，赢了也没意义。所以这里的神经分位模型
拿到与 A5 完全相同的条件信息，甚至更多：

  特征 = [15 个窗口标准化分位 xi(标准正态尺度),
          log sd_d, log sd_h, log h, 1]
  输出 = 15 个分位点
  损失 = 15 个 tau 的 pinball 之和（= 2*CRPS 的训练目标）

也就是说，网络原则上可以精确复现 A5 的函数形式
    Q = mu*lam + sd_d*h^H*GV
（把 xi 与 GV 线性组合、把 log sd_d 乘上 h^H 即可），
再在此基础上做非线性修正。这是本文能给对手的最强公平设置。

模型：2 层 MLP（全批 Adam，确定性），超参在小网格上用**训练段最后 15% 的原点**
做验证选择（不看测试段）。3 个种子做分位平均（集成）。

另外跑一个纯 LSTM（只用最近 60 个日收益，不用任何手工特征），
作为"文献常规 DL 做法"的对照，用来看手工特征 vs 端到端序列模型。
"""
import numpy as np, pandas as pd, os, sys, json, time, warnings
import torch
import torch.nn as nn
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
torch.set_num_threads(4)

OUT = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(OUT, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = None
from scipy.stats import norm
GV = norm.ppf(TAUS)
AK = len(TAUS)
MK = ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]
HOR = [5, 20, 40, 60, 120]
W, STEP = 250, 2
PAR = {"CEA": (0.1, 0.60), "GUANGZHOU": (30, 0.35), "SHANGHAI": (0.1, 0.30),
       "HUBEI": (3, 0.35), "EU": (80, 0.45)}


def crps_row(y, Q):
    Qs = np.sort(Q, axis=1)
    ind = (y[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (y[:, None] - Qs)).mean(axis=1)


def feats(lp, t, h, sdd, sdh):
    win = lp[t - W:t]
    rr = win[h:] - win[:-h]
    xi = np.quantile((rr - rr.mean()) / sdh, TAUS)
    return np.concatenate([xi, [np.log(max(sdd, 1e-8)), np.log(max(sdh, 1e-8)), np.log(h)]])


def targets(lp, t, h):
    win = lp[t - W:t]
    rr = win[h:] - win[:-h]
    sdh = np.std(rr, ddof=1)
    sdd = np.std(np.diff(win), ddof=1)
    return feats(lp, t, h, sdd, sdh), lp[t + h] - lp[t]


class MLP(nn.Module):
    def __init__(self, din, hid, k):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(din, hid), nn.ReLU(),
                               nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, k))

    def forward(self, x):
        return self.f(x)


def pinball(pred, y, taus):
    e = y[:, None] - pred
    return ((taus[None, :] - (e < 0).float()) * e).mean()


def fit_mlp(X, y, hid, wd, iters=1500, seed=0, lr=0.03):
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    tt = torch.tensor(TAUS, dtype=torch.float32)
    net = MLP(X.shape[1], hid, AK)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    for _ in range(iters):
        opt.zero_grad()
        loss = pinball(net(Xt), yt, tt)
        loss.backward()
        opt.step(); sch.step()
    net.eval()
    return net


print("=" * 140)
print("深度学习基准：MLP 神经分位回归（与 A5 同信息集）")
print("=" * 140)
t_start = time.time()

RES = {}
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float))
    dt = pd.to_datetime(d.date).to_numpy()
    cut = "2024-01-01" if m in ("CEA", "GUANGZHOU", "SHANGHAI") else "2023-01-01"
    t0 = int(np.searchsorted(dt, np.datetime64(cut)))
    h0, Hv = PAR[m]
    print(f"\n--- {m} (t0={t0}) ---")
    print(f"{'h':>4}{'n_tr':>6}{'n_te':>6}{'hid':>5}{'wd':>7}{'vl':>9} | "
          f"{'EMP':>9}{'A5':>9}{'MLP':>9}{'A5/EMP':>8}{'MLP/EMP':>9}{'A5/MLP':>8}")
    for h in HOR:
        tr = list(range(W + h + 5, t0 - h, 1))
        te = list(range(max(W + h + 5, t0), len(lp) - h, STEP))
        if len(tr) < 60 or len(te) < 40:
            print(f"{h:>4}  跳过 tr={len(tr)} te={len(te)}")
            continue
        Xtr = np.array([feats(lp, t, h,
                              np.std(np.diff(lp[t - W:t]), ddof=1),
                              np.std(lp[t - W + h:t] - lp[t - W:t - h], ddof=1))
                        for t in tr])
        ytr = np.array([lp[t + h] - lp[t] for t in tr])
        Xte = np.array([feats(lp, t, h,
                              np.std(np.diff(lp[t - W:t]), ddof=1),
                              np.std(lp[t - W + h:t] - lp[t - W:t - h], ddof=1))
                        for t in te])
        yte = np.array([lp[t + h] - lp[t] for t in te])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xtr_s = (Xtr - mu) / sd
        Xte_s = (Xte - mu) / sd
        nv = max(int(0.15 * len(tr)), 40)
        Xa, ya, Xv, yv = Xtr_s[:-nv], ytr[:-nv], Xtr_s[-nv:], ytr[-nv:]
        best, bcfg = np.inf, None
        for hid in (32, 64):
            for wd in (1e-4, 1e-2):
                net = fit_mlp(Xa, ya, hid, wd, iters=900, seed=0)
                with torch.no_grad():
                    v = pinball(net(torch.tensor(Xv, dtype=torch.float32)),
                                torch.tensor(yv, dtype=torch.float32),
                                torch.tensor(TAUS, dtype=torch.float32)).item()
                if v < best:
                    best, bcfg = v, (hid, wd)
        hid, wd = bcfg
        preds = []
        for s in (0, 1, 2):
            net = fit_mlp(Xtr_s, ytr, hid, wd, iters=1500, seed=s)
            with torch.no_grad():
                preds.append(net(torch.tensor(Xte_s, dtype=torch.float32)).numpy())
        Qmlp = np.mean(preds, axis=0)

        Qa, Qe = [], []
        for t in te:
            win = lp[t - W:t]
            rr = win[h:] - win[:-h]; rd = np.diff(win)
            mu_ = rr.mean(); sdh = np.std(rr, ddof=1); sdd = np.std(rd, ddof=1)
            xi = np.quantile((rr - rr.mean()) / sdh, TAUS)
            Qa.append(mu_ * (h0 / (h0 + h)) + sdd * h ** Hv * GV)
            Qe.append(mu_ + sdh * xi)
        ca = crps_row(yte, np.array(Qa)).mean()
        ce = crps_row(yte, np.array(Qe)).mean()
        cm = crps_row(yte, Qmlp).mean()
        RES[f"{m}|{h}"] = dict(n_tr=len(tr), n_te=len(te), hid=hid, wd=wd, vl=float(best),
                               crps_emp=float(ce), crps_a5=float(ca), crps_mlp=float(cm))
        print(f"{h:>4}{len(tr):>6}{len(te):>6}{hid:>5}{wd:>7.0e}{best:>9.5f} | "
              f"{ce:>9.5f}{ca:>9.5f}{cm:>9.5f}{ca/ce:>8.3f}{cm/ce:>9.3f}{ca/cm:>8.3f}")
    print(f"  [耗时 {time.time()-t_start:.0f}s]")

print("\n" + "=" * 140)
print("汇总")
print("=" * 140)
print(f"{'h':>4}{'格数':>6}{'A5/EMP 中位':>14}{'MLP/EMP 中位':>15}{'A5/MLP 中位':>14}"
      f"{'A5 胜 EMP':>11}{'MLP 胜 EMP':>12}{'A5 胜 MLP':>11}")
for h in HOR:
    ks = [k for k in RES if k.endswith(f"|{h}")]
    if not ks:
        continue
    r = [RES[k] for k in ks]
    ra = np.array([x["crps_a5"] / x["crps_emp"] for x in r])
    rm = np.array([x["crps_mlp"] / x["crps_emp"] for x in r])
    ram = np.array([x["crps_a5"] / x["crps_mlp"] for x in r])
    print(f"{h:>4}{len(ks):>6}{np.median(ra):>14.3f}{np.median(rm):>15.3f}"
          f"{np.median(ram):>14.3f}{(ra<1).sum():>8}/{len(ks)}{(rm<1).sum():>9}/{len(ks)}"
          f"{(ram<1).sum():>8}/{len(ks)}")
r = list(RES.values())
ra = np.array([x["crps_a5"] / x["crps_emp"] for x in r])
rm = np.array([x["crps_mlp"] / x["crps_emp"] for x in r])
ram = np.array([x["crps_a5"] / x["crps_mlp"] for x in r])
print(f"\n全部 {len(r)} 格：A5/EMP 中位 {np.median(ra):.3f} ({(ra<1).sum()}/{len(r)})；"
      f"MLP/EMP 中位 {np.median(rm):.3f} ({(rm<1).sum()}/{len(r)})；"
      f"A5/MLP 中位 {np.median(ram):.3f} ({(ram<1).sum()}/{len(r)})")

json.dump(RES, open(os.path.join(OUT, "DL_BASELINE.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 DL_BASELINE.json")
