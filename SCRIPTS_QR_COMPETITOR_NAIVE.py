# -*- coding: utf-8 -*-
"""
竞品复现：QR（线性分位回归）+ conformal 再校准
==============================================
动机：本文主基准 EMP（窗口经验分位）是"无参数稻草人"。审稿人会问
"你们的增益是不是只相对于稻草人？" 这里复现文献里最主流的分布预测竞品：

  Q1  QR        —— 线性分位回归，15 个分位点，特征为滞后日收益
  Q2  QR+conf   —— 在 Q1 之上加 PIT 再校准（分布型 conformal）

协议与本文完全一致：同样的测试原点、同样的 CRPS、同样的 DM。
为避免不公平，竞品可以用**日频**（STEP=1）的估计样本，本文仍用 STEP=2 评估。
估计样本严格截断在 t_start - h，故训练目标日全部早于该块第一个测试目标日。
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
GV = norm.ppf(TAUS)
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
    g0 = (d * d).mean()
    s = g0
    for l in range(1, L + 1):
        g = (d[l:] * d[:-l]).mean()
        s += 2 * (1 - l / (L + 1)) * g
    if s <= 0:
        return 0.0, 1.0
    t = db / np.sqrt(s / n)
    return float(t), float(1 - norm.cdf(abs(t)))


def feats(lp, t):
    r = np.diff(lp)
    return [1.0] + [r[t - 1 - k] if t - 1 - k >= 0 else 0.0 for k in LAGS]


DATA = {}
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float))
    dt = pd.to_datetime(d.date).to_numpy()
    cut = "2024-01-01" if m in ("CEA", "GUANGZHOU", "SHANGHAI") else "2023-01-01"
    DATA[m] = dict(lp=lp, t0=int(np.searchsorted(dt, np.datetime64(cut))))

print("=" * 150)
print("表 1  QR 与 QR+conformal 相对本文模型（CRPS，越低越好）")
print("      QR: 15 分位线性分位回归，特征 [1, r(t-1), r(t-2), r(t-3), r(t-5), r(t-10)]")
print("      QR+conf: 在同一 QR 上做 PIT 再校准（分布型 conformal）")
print("=" * 150)

RES = {}
for m in MK:
    lp, t0 = DATA[m]["lp"], DATA[m]["t0"]
    h0, Hv = PAR[m]
    print(f"\n--- {m} (len={len(lp)}, t0={t0}) ---")
    print(f"{'h':>4}{'n_te':>6}{'EMP':>10}{'A5':>10}{'QR':>10}{'QR+conf':>10}"
          f"{'A5 vs QR':>11}{'A5 vs QRc':>11}{'DM(A5-QRc)':>12}{'p':>8}")
    for h in HOR:
        start = max(W + h + 5, t0)
        ts_te = list(range(start, len(lp) - h, STEP))
        if len(ts_te) < 40:
            continue
        Q_emp, Q_a5, Q_qr, Q_qc, Y = [], [], [], [], []

        def build_rows(origins):
            X = np.array([feats(lp, t) for t in origins])
            if len(origins) == 0:
                return X, np.array([])
            return X, np.array([lp[t + h] - lp[t] for t in origins])

        def recentre(QQ):
            """PIT 再校准：用校准样本的 PIT 经验分布重映射 tau。"""
            return QQ

        for k in range(0, len(ts_te), BLK):
            blk = ts_te[k:k + BLK]
            tstart = blk[0]
            tr = list(range(W + h + 5, tstart - h + 1, 1))
            if len(tr) < 100:
                tr = list(range(W + h + 5, tstart - h + 1, 1))
            if len(tr) < 60:
                continue
            Xtr, ytr = build_rows(tr)
            nf = int(round(0.7 * len(tr)))
            nf = max(min(nf, len(tr) - 25), 40)
            Xf, yf = Xtr[:nf], ytr[:nf]
            Xc, yc = Xtr[nf:], ytr[nf:]
            B = np.empty((len(TAUS), Xf.shape[1]))
            for j, tau in enumerate(TAUS):
                qr = QuantileRegressor(quantile=tau, alpha=0.0, solver="highs")
                qr.fit(Xf, yf)
                B[j] = qr.coef_
            # 校准样本的 PIT
            Qc = Xc @ B.T
            Qs = np.sort(Qc, axis=1)
            u = np.empty(len(yc))
            for i in range(len(yc)):
                u[i] = np.interp(yc[i], Qs[i], TAUS)
            u = np.clip(u, TAUS[0], TAUS[-1])
            # 目标分位的重映射
            ustar = np.quantile(u, TAUS, method="linear")

            Xt, yt = build_rows(blk)
            Qt = Xt @ B.T
            Qts = np.sort(Qt, axis=1)
            Qcal = np.empty_like(Qts)
            for i in range(len(yt)):
                Qcal[i] = np.interp(ustar, TAUS, Qts[i])
            # 本文模型
            mu = np.empty(len(blk)); sdh = np.empty(len(blk))
            sdd = np.empty(len(blk)); xi = np.empty((len(blk), len(TAUS)))
            for i, t in enumerate(blk):
                win = lp[t - W:t]
                rr = win[h:] - win[:-h]; rd = np.diff(win)
                mu[i] = rr.mean(); sdh[i] = np.std(rr, ddof=1); sdd[i] = np.std(rd, ddof=1)
                xi[i] = np.quantile((rr - rr.mean()) / sdh[i], TAUS)
            L = h0 / (h0 + h)
            Q_emp.append(mu[:, None] + sdh[:, None] * xi)
            Q_a5.append(mu[:, None] * L + (sdd * h ** Hv)[:, None] * np.tile(GV, (len(mu), 1)))
            Q_qr.append(Qts)
            Q_qc.append(Qcal)
            Y.append(yt)

        if not Y:
            continue
        y = np.concatenate(Y)
        ce = crps_row(y, np.vstack(Q_emp)).mean()
        ca = crps_row(y, np.vstack(Q_a5)).mean()
        cq = crps_row(y, np.vstack(Q_qr)).mean()
        cc = crps_row(y, np.vstack(Q_qc)).mean()
        t_, pv = dm(crps_row(y, np.vstack(Q_a5)), crps_row(y, np.vstack(Q_qc)))
        RES[f"{m}|{h}"] = dict(n=len(y), emp=float(ce), a5=float(ca), qr=float(cq),
                               qrc=float(cc), dq=(ca / cq - 1) * 100,
                               dqc=(ca / cc - 1) * 100, dmt=t_, dmp=pv)
        print(f"{h:>4}{len(y):>6}{ce:>10.5f}{ca:>10.5f}{cq:>10.5f}{cc:>10.5f}"
              f"{(ca/cq-1)*100:>10.1f}%{(ca/cc-1)*100:>10.1f}%{t_:>12.2f}{pv:>8.3f}")

print("\n" + "=" * 150)
print("表 2  汇总")
print("=" * 150)
for lab, key in (("A5 vs QR", "dq"), ("A5 vs QR+conformal", "dqc")):
    print(f"\n  {lab}")
    print(f"  {'h':>4}{'中位':>10}{'胜格':>9}")
    for h in HOR:
        ks = [f"{m}|{h}" for m in MK if f"{m}|{h}" in RES]
        if not ks:
            continue
        dd = [RES[k][key] for k in ks]
        print(f"  {h:>4}{np.median(dd):>9.1f}%{sum(1 for x in dd if x < 0):>6}/{len(dd)}")
    allv = [RES[k][key] for k in RES]
    print(f"  全部  中位 {np.median(allv):+.1f}%  胜 {sum(1 for x in allv if x<0)}/{len(allv)}")

print(f"\n  QR+conformal 相对 EMP：胜 {sum(1 for k in RES if RES[k]['qrc']<RES[k]['emp'])}/{len(RES)}")
print(f"  A5 相对 QR+conformal 且 DM p<0.05："
      f"{sum(1 for k in RES if RES[k]['dmp']<0.05 and RES[k]['dmt']<0)}/{len(RES)}")

json.dump(RES, open(os.path.join(OUT, "QR_COMPETITOR.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 QR_COMPETITOR.json")
