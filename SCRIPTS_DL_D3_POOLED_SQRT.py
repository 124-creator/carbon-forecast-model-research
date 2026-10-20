# -*- coding: utf-8 -*-
"""
深度学习基准（最强公平版）：√h 标准化 + 跨市场池化
==================================================
前两版的两个问题，这一版都解决：
  dl1：逐市场训练，数据少 → 池化。
  dl2：直接预测 y 的水平，网络在长期限外推出灾难性的分位点。

本版采用任何合格的 DL 实践者都会做的标准化：
    目标 z = y / (sd_d * sqrt(h))     <- 经典 √h 标度，不含本文的 H
    输出 15 个 z 的分位点，再乘回 sd_d*sqrt(h)
    特征 xi(15 个窗口标准化分位) + log h   <- 尺度信息由标准化提供，
                                              不额外喂 sd，避免 log 特征的线性外推爆炸

这样网络只学**形状**（偏离高斯、偏离 √h），尺度交给经典 √h。
它是本文最需要的那个对手：如果 A5 还能赢它，赢的部分就只能归功于 H。
"""
import numpy as np, pandas as pd, os, sys, json, time, warnings
import torch, torch.nn as nn
from scipy.stats import norm
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
torch.set_num_threads(4)

OUT = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(OUT, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS)
AK = len(TAUS)
MK = ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]
HOR = [5, 20, 40, 60, 120]
W, STEP = 250, 2
PAR = {"CEA": (0.1, 0.60), "GUANGZHOU": (30, 0.35), "SHANGHAI": (0.1, 0.30),
       "HUBEI": (3, 0.35), "EU": (80, 0.45)}
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}

DATA = {}
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float))
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    DATA[m] = (lp, dt, t0)


def crps_row(y, Q):
    Qs = np.sort(Q, axis=1)
    ind = (y[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (y[:, None] - Qs)).mean(axis=1)


def feats(lp, t, h):
    win = lp[t - W:t]
    rr = win[h:] - win[:-h]
    sdh = np.std(rr, ddof=1); sdd = np.std(np.diff(win), ddof=1)
    xi = np.quantile((rr - rr.mean()) / sdh, TAUS)
    return np.concatenate([xi, [np.log(h)]]), sdd, sdh


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


def fit(X, y, hid, wd, iters, seed):
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    tt = torch.tensor(TAUS, dtype=torch.float32)
    net = MLP(X.shape[1], hid, AK)
    opt = torch.optim.Adam(net.parameters(), lr=0.03, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    for _ in range(iters):
        opt.zero_grad()
        loss = pinball(net(Xt), yt, tt)
        loss.backward(); opt.step(); sch.step()
    net.eval()
    return net


def build(m, h):
    lp, dt, t0 = DATA[m]
    targ = np.arange(W + h + 5, len(lp) - h)
    F, Z, SC = [], [], []
    for t in targ:
        f, sdd, sdh = feats(lp, t, h)
        F.append(f); Z.append((lp[t + h] - lp[t]) / (sdd * np.sqrt(h))); SC.append(sdd * np.sqrt(h))
    F = np.array(F); Z = np.array(Z); SC = np.array(SC)
    ok = dt[targ + h] < np.datetime64(CUT[m])
    return F, Z, SC, targ, ok


print("=" * 152)
print("深度学习基准（最强公平版）：√h 标准化目标 + 跨市场池化 + 只学形状")
print("=" * 152)
t0all = time.time()
RES = {}
for m in MK:
    lp, dt, t0 = DATA[m]
    h0, Hv = PAR[m]
    print(f"\n--- {m} (t0={t0}) ---")
    print(f"{'h':>4}{'n_pool':>8}{'n_te':>6}{'hid':>5}{'wd':>7} | "
          f"{'EMP':>9}{'A5':>9}{'MLPs':>9}{'A5/EMP':>8}{'S/EMP':>8}{'A5/S':>7}")
    for h in HOR:
        cache = {mm: build(mm, h) for mm in MK}
        Fm, Zm, SCm, targ_m, ok_m = cache[m]
        Fp, Zp = [], []
        for mm in MK:
            F2, Z2, S2, tg2, ok2 = cache[mm]
            Fp.append(F2[ok2]); Zp.append(Z2[ok2])
        Fp = np.vstack(Fp); Zp = np.concatenate(Zp)
        trm = np.where(ok_m)[0]
        nv = max(int(0.15 * len(trm)), 40)
        mu, sd = Fm[trm[:-nv]].mean(0), Fm[trm[:-nv]].std(0) + 1e-8
        Fps = (Fp - mu) / sd
        Fvs = (Fm[trm[-nv:]] - mu) / sd
        zv = Zm[trm[-nv:]]
        best, bcfg = np.inf, None
        for hid in (32, 64):
            for wd in (1e-4, 1e-2):
                net = fit(Fps, Zp, hid, wd, 600, 0)
                with torch.no_grad():
                    v = pinball(net(torch.tensor(Fvs, dtype=torch.float32)),
                                torch.tensor(zv, dtype=torch.float32),
                                torch.tensor(TAUS, dtype=torch.float32)).item()
                if v < best:
                    best, bcfg = v, (hid, wd)
        hid, wd = bcfg
        te = np.arange(max(W + h + 5, t0), len(lp) - h, STEP)
        Fte, SCte = [], []
        for t in te:
            f, sdd, sdh = feats(lp, t, h)
            Fte.append(f); SCte.append(sdd * np.sqrt(h))
        Fte = (np.array(Fte) - mu) / sd
        SCte = np.array(SCte)
        yte = np.array([lp[t + h] - lp[t] for t in te])
        preds = []
        for s in (0, 1, 2):
            net = fit(Fps, Zp, hid, wd, 1500, s)
            with torch.no_grad():
                preds.append(net(torch.tensor(Fte, dtype=torch.float32)).numpy())
        Qz = np.mean(preds, axis=0)
        Qs = Qz * SCte[:, None]
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
        cs = crps_row(yte, Qs).mean()
        RES[f"{m}|{h}"] = dict(n_pool=len(Zp), n_te=len(te), hid=hid, wd=wd,
                               crps_emp=float(ce), crps_a5=float(ca), crps_std=float(cs))
        print(f"{h:>4}{len(Zp):>8}{len(te):>6}{hid:>5}{wd:>7.0e} | "
              f"{ce:>9.5f}{ca:>9.5f}{cs:>9.5f}{ca/ce:>8.3f}{cs/ce:>8.3f}{ca/cs:>7.3f}")
    print(f"  [耗时 {time.time()-t0all:.0f}s]")

print("\n" + "=" * 152)
print("汇总")
print("=" * 152)
print(f"{'h':>4}{'格数':>6}{'A5/EMP':>10}{'√hMLP/EMP':>12}{'A5/√hMLP':>11}"
      f"{'A5胜EMP':>10}{'S胜EMP':>9}{'A5胜S':>8}")
for h in HOR:
    ks = [k for k in RES if k.endswith(f"|{h}")]
    r = [RES[k] for k in ks]
    ra = np.array([x["crps_a5"] / x["crps_emp"] for x in r])
    rs = np.array([x["crps_std"] / x["crps_emp"] for x in r])
    ras = np.array([x["crps_a5"] / x["crps_std"] for x in r])
    print(f"{h:>4}{len(ks):>6}{np.median(ra):>10.3f}{np.median(rs):>12.3f}{np.median(ras):>11.3f}"
          f"{(ra<1).sum():>7}/{len(ks)}{(rs<1).sum():>6}/{len(ks)}{(ras<1).sum():>5}/{len(ks)}")
r = list(RES.values())
ra = np.array([x["crps_a5"] / x["crps_emp"] for x in r])
rs = np.array([x["crps_std"] / x["crps_emp"] for x in r])
ras = np.array([x["crps_a5"] / x["crps_std"] for x in r])
print(f"\n全部 {len(r)} 格：A5/EMP {np.median(ra):.3f} ({(ra<1).sum()}/{len(r)})；"
      f"√hMLP/EMP {np.median(rs):.3f} ({(rs<1).sum()}/{len(r)})；"
      f"A5/√hMLP {np.median(ras):.3f} ({(ras<1).sum()}/{len(r)})")

json.dump(RES, open(os.path.join(OUT, "DL_STANDARDIZED.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 DL_STANDARDIZED.json")
