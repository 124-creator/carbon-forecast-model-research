# -*- coding: utf-8 -*-
"""
深度学习基准（终版）：跨市场 + 跨期限联合训练
=============================================
dl1/dl2/dl3 的共同缺陷：按 (市场,h) 各训一个模型。这样
  - 池化时 log h 在格内是常数，网络**学不到期限依赖**；
  - 每格样本量被期限切碎。

终版把 5 个市场 x 5 个期限的**全部**训练原点放进一个模型：
    特征 = xi(15 个窗口标准化分位) + log h
    目标 = y / (sd_d * sqrt(h))        <- 经典 √h 标度，不含本文的 H
    输出 15 个 z 分位，再乘回 sd_d*sqrt(h)
样本量约 2 万个，log h 成为有效特征，网络可以自己学出 h^{H-1/2} 那种弯曲。

每个目标市场单独训一个（5 个模型），因为市场间形状差异是真实存在的；
但样本跨市场共享。这已经是本文能给出的最强 DL 对手。
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


def feats_all(lp, h):
    """对某市场某期限，返回全部可用原点的 (特征, z 目标, 尺度, 绝对原点, 目标日是否在训练段)"""
    lo, hi = W + h + 5, len(lp) - h
    t = np.arange(lo, hi)
    F = np.empty((len(t), AK + 1)); Z = np.empty(len(t)); S = np.empty(len(t))
    for i, tt in enumerate(t):
        win = lp[tt - W:tt]
        rr = win[h:] - win[:-h]
        sdh = np.std(rr, ddof=1); sdd = np.std(np.diff(win), ddof=1)
        F[i, :AK] = np.quantile((rr - rr.mean()) / sdh, TAUS)
        F[i, AK] = np.log(h)
        S[i] = sdd * np.sqrt(h)
        Z[i] = (lp[tt + h] - lp[tt]) / S[i]
    return F, Z, S, t


class MLP(nn.Module):
    def __init__(self, din, hid, k):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(din, hid), nn.ReLU(),
                               nn.Linear(hid, hid), nn.ReLU(),
                               nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, k))

    def forward(self, x):
        return self.f(x)


def pinball(pred, y, taus):
    e = y[:, None] - pred
    return ((taus[None, :] - (e < 0).float()) * e).mean()


def fit(X, y, hid, wd, iters, seed, lr=0.02):
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    tt = torch.tensor(TAUS, dtype=torch.float32)
    net = MLP(X.shape[1], hid, AK)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    bs = min(1024, len(y))
    nb = max(len(y) // bs, 1)
    for it in range(iters):
        idx = torch.randint(0, len(y), (bs,)) if len(y) > bs else slice(None)
        opt.zero_grad()
        loss = pinball(net(Xt[idx]), yt[idx], tt)
        loss.backward(); opt.step(); sch.step()
    net.eval()
    return net


print("=" * 152)
print("深度学习基准（终版）：跨市场 + 跨期限联合训练，约 2 万训练原点")
print("=" * 152)
t0all = time.time()

# 预计算每个市场每个期限的特征
PRE = {m: {h: feats_all(DATA[m][0], h) for h in HOR} for m in MK}
for m in MK:
    print(f"{m}: " + ", ".join(f"h{h}={len(PRE[m][h][1])}" for h in HOR))

RES = {}
for m in MK:
    lp, dt, t0 = DATA[m]
    h0, Hv = PAR[m]
    # 池化训练集：所有市场所有期限，目标日必须早于 m 的测试起点
    Xp, yp = [], []
    for mm in MK:
        for h in HOR:
            F, Z, S, t = PRE[mm][h]
            ok = (DATA[mm][1][t + h] < np.datetime64(CUT[m]))
            Xp.append(F[ok]); yp.append(Z[ok])
    Xp = np.vstack(Xp); yp = np.concatenate(yp)
    # 标准化：用 m 自己的训练段统计
    own_ok = np.concatenate([(DATA[m][1][t + h] < np.datetime64(CUT[m]))
                             for h in HOR for t in [PRE[m][h][3]]])
    Fm = np.vstack([PRE[m][h][0] for h in HOR])
    mu, sd = Fm[own_ok].mean(0), Fm[own_ok].std(0) + 1e-8
    Xps = (Xp - mu) / sd
    # 验证集：m 自己训练段里跨期限分层的 15%
    nown = int(own_ok.sum()); nv = max(int(0.15 * nown), 200)
    Fi = Fm[own_ok]
    Zi = np.concatenate([PRE[m][h][1][(DATA[m][1][PRE[m][h][3] + h] < np.datetime64(CUT[m]))] for h in HOR])
    perm = np.random.default_rng(0).permutation(len(Zi))
    Xv = ((Fi[perm[:nv]] - mu) / sd); zv = Zi[perm[:nv]]
    print(f"\n--- {m} --- 池化 {len(yp)} 条，验证 {len(zv)} 条")
    best, bcfg = np.inf, None
    for hid in (64, 128):
        for wd in (1e-5, 1e-3):
            net = fit(Xps, yp, hid, wd, 2000, 0)
            with torch.no_grad():
                v = pinball(net(torch.tensor(Xv, dtype=torch.float32)),
                            torch.tensor(zv, dtype=torch.float32),
                            torch.tensor(TAUS, dtype=torch.float32)).item()
            print(f"    hid={hid} wd={wd:.0e} 验证 pinball={v:.5f}")
            if v < best:
                best, bcfg = v, (hid, wd)
    hid, wd = bcfg
    nets = [fit(Xps, yp, hid, wd, 4000, s) for s in (0, 1, 2)]
    print(f"{'h':>4}{'n_te':>6} | {'EMP':>9}{'A5':>9}{'DL':>9}"
          f"{'A5/EMP':>8}{'DL/EMP':>8}{'A5/DL':>7}")
    for h in HOR:
        F, Z, S, t = PRE[m][h]
        te_mask = t >= max(W + h + 5, t0)
        te = t[te_mask][::STEP]
        Fte = ((F[te_mask][::STEP] - mu) / sd)
        Ste = S[te_mask][::STEP]
        yte = np.array([lp[tt + h] - lp[tt] for tt in te])
        preds = []
        with torch.no_grad():
            Xt = torch.tensor(Fte, dtype=torch.float32)
            for net in nets:
                preds.append(net(Xt).numpy())
        Qd = np.mean(preds, axis=0) * Ste[:, None]
        Qa, Qe = [], []
        for tt in te:
            win = lp[tt - W:tt]
            rr = win[h:] - win[:-h]; rd = np.diff(win)
            mu_ = rr.mean(); sdh = np.std(rr, ddof=1); sdd = np.std(rd, ddof=1)
            xi = np.quantile((rr - rr.mean()) / sdh, TAUS)
            Qa.append(mu_ * (h0 / (h0 + h)) + sdd * h ** Hv * GV)
            Qe.append(mu_ + sdh * xi)
        ca = crps_row(yte, np.array(Qa)).mean()
        ce = crps_row(yte, np.array(Qe)).mean()
        cd = crps_row(yte, Qd).mean()
        RES[f"{m}|{h}"] = dict(n_te=len(te), hid=hid, wd=wd, crps_emp=float(ce),
                               crps_a5=float(ca), crps_dl=float(cd))
        print(f"{h:>4}{len(te):>6} | {ce:>9.5f}{ca:>9.5f}{cd:>9.5f}"
              f"{ca/ce:>8.3f}{cd/ce:>8.3f}{ca/cd:>7.3f}")
    print(f"  [耗时 {time.time()-t0all:.0f}s]")

print("\n" + "=" * 152)
print("汇总")
print("=" * 152)
print(f"{'h':>4}{'格数':>6}{'A5/EMP':>10}{'DL/EMP':>10}{'A5/DL':>9}"
      f"{'A5胜EMP':>10}{'DL胜EMP':>10}{'A5胜DL':>9}")
for h in HOR:
    ks = [k for k in RES if k.endswith(f"|{h}")]
    r = [RES[k] for k in ks]
    ra = np.array([x["crps_a5"] / x["crps_emp"] for x in r])
    rd = np.array([x["crps_dl"] / x["crps_emp"] for x in r])
    rad = np.array([x["crps_a5"] / x["crps_dl"] for x in r])
    print(f"{h:>4}{len(ks):>6}{np.median(ra):>10.3f}{np.median(rd):>10.3f}{np.median(rad):>9.3f}"
          f"{(ra<1).sum():>7}/{len(ks)}{(rd<1).sum():>7}/{len(ks)}{(rad<1).sum():>6}/{len(ks)}")
r = list(RES.values())
ra = np.array([x["crps_a5"] / x["crps_emp"] for x in r])
rd = np.array([x["crps_dl"] / x["crps_emp"] for x in r])
rad = np.array([x["crps_a5"] / x["crps_dl"] for x in r])
print(f"\n全部 {len(r)} 格：A5/EMP {np.median(ra):.3f} ({(ra<1).sum()}/{len(r)})；"
      f"DL/EMP {np.median(rd):.3f} ({(rd<1).sum()}/{len(r)})；"
      f"A5/DL {np.median(rad):.3f} ({(rad<1).sum()}/{len(r)})")

json.dump(RES, open(os.path.join(OUT, "DL_JOINT.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 DL_JOINT.json")
