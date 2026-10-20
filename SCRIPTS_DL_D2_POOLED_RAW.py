# -*- coding: utf-8 -*-
"""
深度学习基准（池化版，最强公平设置）
====================================
dl1.py 的 MLP 是逐市场训练的，CEA 只有 332 个训练原点，审稿人会说"没给够数据"。
这里把五个市场的训练原点**池化**，给神经网络约 3-5 倍的数据。

防泄漏：为目标市场 m 训练时，只允许使用满足
    date(origin) + h < date(t0_m)
的样本（不论来自哪个市场）。即目标日必须严格早于该市场的测试起点。
这比"用全部市场到各自 t0"更严，因为各市场 t0 不同（EU/HUBEI 是 2023，其余 2024）。

信息集与 A5 完全相同：15 个窗口标准化分位 + log sd_d + log sd_h + log h。
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


def crps_row(y, Q):
    Qs = np.sort(Q, axis=1)
    ind = (y[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (y[:, None] - Qs)).mean(axis=1)


DATA = {}
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float))
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    DATA[m] = (lp, dt, t0)


def build(m, h):
    lp, dt, t0 = DATA[m]
    lo = W + h + 5
    hi = len(lp) - h
    targ = np.arange(lo, hi)
    X, y = [], []
    for t in targ:
        win = lp[t - W:t]
        rr = win[h:] - win[:-h]
        sdh = np.std(rr, ddof=1); sdd = np.std(np.diff(win), ddof=1)
        xi = np.quantile((rr - rr.mean()) / sdh, TAUS)
        X.append(np.concatenate([xi, [np.log(max(sdd, 1e-8)), np.log(max(sdh, 1e-8)),
                                      np.log(h)]]))
        y.append(lp[t + h] - lp[t])
    X = np.array(X); y = np.array(y)
    okt = dt[targ + h] < np.datetime64(CUT[m])
    return X, y, targ, okt


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


print("=" * 152)
print("深度学习基准（跨市场池化）：与 A5 同信息集，约 3-5 倍训练样本")
print("=" * 152)
t0all = time.time()
RES = {}
for m in MK:
    lp, dt, t0 = DATA[m]
    h0, Hv = PAR[m]
    print(f"\n--- {m} (t0={t0}) ---")
    print(f"{'h':>4}{'n_pool':>8}{'n_te':>6}{'hid':>5}{'wd':>7} | "
          f"{'EMP':>9}{'A5':>9}{'MLPp':>9}{'A5/EMP':>8}{'P/EMP':>8}{'A5/P':>7}"
          f"{'A5胜EMP':>9}{'P胜EMP':>8}")
    for h in HOR:
        cache = {mm: build(mm, h) for mm in MK}
        Xm, ym, targ_m, ok_m = cache[m]
        # 池化训练集：目标日必须早于 m 的测试起点
        Xp, yp = [], []
        for mm in MK:
            X2, y2, tg2, ok2 = cache[mm]
            sel = ok2
            Xp.append(X2[sel]); yp.append(y2[sel])
        Xp = np.vstack(Xp); yp = np.concatenate(yp)
        trm = np.where(ok_m)[0]
        nv = max(int(0.15 * len(trm)), 40)
        Xtr_own = Xm[trm[:-nv]]; ytr_own = ym[trm[:-nv]]
        Xv = Xm[trm[-nv:]]; yv = ym[trm[-nv:]]
        mu, sd = Xtr_own.mean(0), Xtr_own.std(0) + 1e-8
        Xps = (Xp - mu) / sd
        Xvs = (Xv - mu) / sd
        best, bcfg = np.inf, None
        for hid in (32, 64):
            for wd in (1e-4, 1e-2):
                net = fit(Xps, yp, hid, wd, 600, 0)
                with torch.no_grad():
                    v = pinball(net(torch.tensor(Xvs, dtype=torch.float32)),
                                torch.tensor(yv, dtype=torch.float32),
                                torch.tensor(TAUS, dtype=torch.float32)).item()
                if v < best:
                    best, bcfg = v, (hid, wd)
        hid, wd = bcfg
        te = np.arange(max(W + h + 5, t0), len(lp) - h, STEP)
        Xte = np.array([np.concatenate([
            np.quantile((lambda win: (lambda rr: (rr - rr.mean()) / np.std(rr, ddof=1))(
                win[h:] - win[:-h]))(lp[t - W:t]), TAUS),
            [np.log(max(np.std(np.diff(lp[t - W:t]), ddof=1), 1e-8)),
             np.log(max(np.std(lp[t - W + h:t] - lp[t - W:t - h], ddof=1), 1e-8)),
             np.log(h)]]) for t in te])
        yte = np.array([lp[t + h] - lp[t] for t in te])
        Xte_s = (Xte - mu) / sd
        preds = []
        for s in (0, 1, 2):
            net = fit(Xps, yp, hid, wd, 1500, s)
            with torch.no_grad():
                preds.append(net(torch.tensor(Xte_s, dtype=torch.float32)).numpy())
        Qp = np.mean(preds, axis=0)
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
        cp = crps_row(yte, Qp).mean()
        RES[f"{m}|{h}"] = dict(n_pool=len(yp), n_te=len(te), hid=hid, wd=wd, vl=float(best),
                               crps_emp=float(ce), crps_a5=float(ca), crps_pool=float(cp))
        print(f"{h:>4}{len(yp):>8}{len(te):>6}{hid:>5}{wd:>7.0e} | "
              f"{ce:>9.5f}{ca:>9.5f}{cp:>9.5f}{ca/ce:>8.3f}{cp/ce:>8.3f}{ca/cp:>7.3f}"
              f"{'✓' if ca<ce else '·':>8}{'✓' if cp<ce else '·':>8}")
    print(f"  [耗时 {time.time()-t0all:.0f}s]")

print("\n" + "=" * 152)
print("汇总")
print("=" * 152)
print(f"{'h':>4}{'格数':>6}{'A5/EMP 中位':>14}{'池化/EMP 中位':>16}{'A5/池化 中位':>16}"
      f"{'A5胜EMP':>10}{'池化胜EMP':>11}{'A5胜池化':>10}")
for h in HOR:
    ks = [k for k in RES if k.endswith(f"|{h}")]
    r = [RES[k] for k in ks]
    ra = np.array([x["crps_a5"] / x["crps_emp"] for x in r])
    rp = np.array([x["crps_pool"] / x["crps_emp"] for x in r])
    rap = np.array([x["crps_a5"] / x["crps_pool"] for x in r])
    print(f"{h:>4}{len(ks):>6}{np.median(ra):>14.3f}{np.median(rp):>16.3f}{np.median(rap):>16.3f}"
          f"{(ra<1).sum():>7}/{len(ks)}{(rp<1).sum():>8}/{len(ks)}{(rap<1).sum():>7}/{len(ks)}")
r = list(RES.values())
ra = np.array([x["crps_a5"] / x["crps_emp"] for x in r])
rp = np.array([x["crps_pool"] / x["crps_emp"] for x in r])
rap = np.array([x["crps_a5"] / x["crps_pool"] for x in r])
print(f"\n全部 {len(r)} 格：A5/EMP {np.median(ra):.3f} ({(ra<1).sum()}/{len(r)})；"
      f"池化/EMP {np.median(rp):.3f} ({(rp<1).sum()}/{len(r)})；"
      f"A5/池化 {np.median(rap):.3f} ({(rap<1).sum()}/{len(r)})")

json.dump(RES, open(os.path.join(OUT, "DL_POOLED.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 DL_POOLED.json")
