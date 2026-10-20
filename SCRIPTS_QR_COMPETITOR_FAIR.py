# -*- coding: utf-8 -*-
"""
竞品复现（公平版）：给 QR 同样的信息集
=====================================
qrc1 的 QR 只喂了滞后日收益，结构上无法表达**条件尺度**，因此被 EMP 打败 21/25，
这不是公平比较。本脚本给竞品本文模型所用的全部信息：

  QR        —— 特征 [1, r(t-1), r(t-2), r(t-3), r(t-5), r(t-10)]            （qrc1 原版，留作对照）
  QR+vol    —— 再加上 [log sd_d(t), log sd_h(t)]，即本文的日频尺度与期限尺度
  QR+conf   —— 各自叠加 PIT 再校准（分布型 conformal）

若 A5 在 QR+vol 下仍然占优，才能说"增益不是只相对于稻草人"。
"""
import numpy as np, pandas as pd, os, sys, json, warnings
from scipy.stats import norm
from sklearn.linear_model import QuantileRegressor
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

OUT = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(OUT, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
MK = ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]
HOR = [5, 20, 40, 60, 120]
W, STEP, BLK = 250, 2, 60
LAGS = [1, 2, 3, 5, 10]
PAR = {"CEA": (0.1, 0.60), "GUANGZHOU": (30, 0.35), "SHANGHAI": (0.1, 0.30),
       "HUBEI": (3, 0.35), "EU": (80, 0.45)}


def crps_row(y, Q):
    Qs = np.sort(Q, axis=1)
    ind = (y[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (y[:, None] - Qs)).mean(axis=1)


def dm(a, b, L=10):
    d = a - b; n = len(d); db = d.mean()
    s = (d * d).mean()
    for l in range(1, L + 1):
        s += 2 * (1 - l / (L + 1)) * (d[l:] * d[:-l]).mean()
    if s <= 0:
        return 0.0, 1.0
    t = db / np.sqrt(s / n)
    return float(t), float(1 - norm.cdf(abs(t)))


def pit_recal(Xc, yc, Xt):
    """在校准样本上算 PIT，再把目标分位按 PIT 经验分布重映射。"""
    Qc = np.sort(Xc, axis=1)
    u = np.clip(np.array([np.interp(yc[i], Qc[i], TAUS) for i in range(len(yc))]),
                TAUS[0], TAUS[-1])
    us = np.quantile(u, TAUS, method="linear")
    Qt = np.sort(Xt, axis=1)
    return np.vstack([np.interp(us, TAUS, Qt[i]) for i in range(len(Qt))])


DATA = {}
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float))
    dt = pd.to_datetime(d.date).to_numpy()
    cut = "2024-01-01" if m in ("CEA", "GUANGZHOU", "SHANGHAI") else "2023-01-01"
    DATA[m] = dict(lp=lp, t0=int(np.searchsorted(dt, np.datetime64(cut))))

print("=" * 168)
print("表 1  公平竞品：QR 在获得本文同一信息集（含条件尺度）后的表现")
print("=" * 168)

RES = {}
for m in MK:
    lp, t0 = DATA[m]["lp"], DATA[m]["t0"]
    h0, Hv = PAR[m]
    r_all = np.diff(lp, prepend=lp[0])
    print(f"\n--- {m} (len={len(lp)}, t0={t0}) ---")
    print(f"{'h':>4}{'n_te':>6}{'EMP':>10}{'A5':>10}{'QR':>10}{'QRv':>10}{'QRc':>10}{'QRvc':>10}"
          f"{'A5/QRv':>9}{'A5/QRvc':>9}{'DM':>7}{'p':>7}")
    for h in HOR:
        start = max(W + h + 5, t0)
        ts_te = list(range(start, len(lp) - h, STEP))
        if len(ts_te) < 40:
            continue
        acc = {k: [] for k in ("emp", "a5", "qr", "qrv", "qrc", "qrvc")}
        Ys = []

        def rows(origins):
            X1, X2, y = [], [], []
            for t in origins:
                f1 = [1.0] + [r_all[t - k] for k in LAGS]
                win = lp[t - W:t]
                rr = win[h:] - win[:-h]; rd = np.diff(win)
                sdd = np.std(rd, ddof=1); sdh = np.std(rr, ddof=1)
                f2 = f1 + [np.log(max(sdd, 1e-9)), np.log(max(sdh, 1e-9))]
                X1.append(f1); X2.append(f2)
                y.append(lp[t + h] - lp[t])
            return np.array(X1), np.array(X2), np.array(y)

        for k in range(0, len(ts_te), BLK):
            blk = ts_te[k:k + BLK]
            tstart = blk[0]
            tr = list(range(W + h + 5, tstart - h + 1, 1))
            if len(tr) < 60:
                continue
            X1, X2, ytr = rows(tr)
            nf = max(min(int(round(0.7 * len(tr))), len(tr) - 25), 40)
            B1 = np.empty((len(TAUS), X1.shape[1])); B2 = np.empty((len(TAUS), X2.shape[1]))
            for j, tau in enumerate(TAUS):
                q1 = QuantileRegressor(quantile=tau, alpha=0.0, solver="highs").fit(X1[:nf], ytr[:nf])
                q2 = QuantileRegressor(quantile=tau, alpha=0.0, solver="highs").fit(X2[:nf], ytr[:nf])
                B1[j] = q1.coef_; B2[j] = q2.coef_
            Qt1, Qt2, yt = rows(blk)
            P1 = Qt1 @ B1.T; P2 = Qt2 @ B2.T
            acc["qr"].append(P1)
            acc["qrv"].append(P2)
            acc["qrc"].append(pit_recal(X1[nf:] @ B1.T, ytr[nf:], P1))
            acc["qrvc"].append(pit_recal(X2[nf:] @ B2.T, ytr[nf:], P2))
            mu = np.empty(len(blk)); sdh = np.empty(len(blk))
            sdd = np.empty(len(blk)); xi = np.empty((len(blk), len(TAUS)))
            for i, t in enumerate(blk):
                win = lp[t - W:t]
                rr = win[h:] - win[:-h]; rd = np.diff(win)
                mu[i] = rr.mean(); sdh[i] = np.std(rr, ddof=1); sdd[i] = np.std(rd, ddof=1)
                xi[i] = np.quantile((rr - rr.mean()) / sdh[i], TAUS)
            L = h0 / (h0 + h)
            acc["emp"].append(mu[:, None] + sdh[:, None] * xi)
            acc["a5"].append(mu[:, None] * L + (sdd * h ** Hv)[:, None] * np.tile(norm.ppf(TAUS), (len(mu), 1)))
            Ys.append(yt)

        if not Ys:
            continue
        y = np.concatenate(Ys)
        C = {k: crps_row(y, np.vstack(v)) for k, v in acc.items()}
        t_, pv = dm(C["a5"], C["qrvc"])
        RES[f"{m}|{h}"] = dict(n=len(y), **{k: float(v.mean()) for k, v in C.items()},
                               d_qrv=float((C["a5"].mean() / C["qrv"].mean() - 1) * 100),
                               d_qrvc=float((C["a5"].mean() / C["qrvc"].mean() - 1) * 100),
                               dmt=t_, dmp=pv)
        print(f"{h:>4}{len(y):>6}{C['emp'].mean():>10.5f}{C['a5'].mean():>10.5f}"
              f"{C['qr'].mean():>10.5f}{C['qrv'].mean():>10.5f}"
              f"{C['qrc'].mean():>10.5f}{C['qrvc'].mean():>10.5f}"
              f"{RES[f'{m}|{h}']['d_qrv']:>8.1f}%{RES[f'{m}|{h}']['d_qrvc']:>8.1f}%{t_:>7.2f}{pv:>7.3f}")

print("\n" + "=" * 168)
print("表 2  汇总")
print("=" * 168)
for lab, key in (("A5 vs QR（无尺度信息）", "qr"), ("A5 vs QR+vol（公平）", "qrv"),
                 ("A5 vs QR+conf", "qrc"), ("A5 vs QR+vol+conf（最强竞品）", "qrvc")):
    print(f"\n  {lab}")
    print(f"  {'h':>4}{'中位':>10}{'胜格':>9}")
    for h in HOR:
        ks = [f"{m}|{h}" for m in MK if f"{m}|{h}" in RES]
        if not ks:
            continue
        dd = [(RES[k]["a5"] / RES[k][key] - 1) * 100 for k in ks]
        print(f"  {h:>4}{np.median(dd):>9.1f}%{sum(1 for x in dd if x < 0):>6}/{len(dd)}")
    allv = [(RES[k]["a5"] / RES[k][key] - 1) * 100 for k in RES]
    print(f"  全部  中位 {np.median(allv):+.1f}%  胜 {sum(1 for x in allv if x<0)}/{len(allv)}")

print(f"\n  QR+vol+conf 相对 EMP：胜 {sum(1 for k in RES if RES[k]['qrvc']<RES[k]['emp'])}/{len(RES)}")
print(f"  A5 相对 QR+vol+conf 且 DM p<0.05："
      f"{sum(1 for k in RES if RES[k]['dmp']<0.05 and RES[k]['dmt']<0)}/{len(RES)}")
print(f"  A5 相对 QR+vol+conf 且 DM 方向为正（A5 更差）："
      f"{sum(1 for k in RES if RES[k]['dmp']<0.05 and RES[k]['dmt']>0)}/{len(RES)}")

json.dump(RES, open(os.path.join(OUT, "QR_COMPETITOR_FAIR.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 QR_COMPETITOR_FAIR.json")
