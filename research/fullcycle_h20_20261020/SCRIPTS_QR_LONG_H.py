# -*- coding: utf-8 -*-
"""
QR vs 参数密度在长期限：realistic 校准版
=========================================
修正 qrlong1.py 的致命问题：原版用**测试集自身**挑 k 使 PICP95=0.95（oracle 校准，
不可达）。本版严格区分：
  k 的确定     → 只用训练信息
     QR        : 训练样本内分位残差
     VR/Emp/G  : 训练期起点(t<t0)的滚动预测残差
  评估         → 测试期，用上一步固定的 k，不再调整
同时报告测试期 PICP95，以暴露校准是否迁移失败。

模型：
  QR   每期限线性分位回归（6 特征，全部训练段）
  VR   幂律尺度 sigma1*h^H + Gaussian 形状（H 由窗口 VR(20) 反解）
  EMP  本地经验分位（零特征零参数）
  GAU  本地经验 h 期均值/标准差 + Gaussian 形状
"""
import numpy as np, pandas as pd, os, sys, json, warnings
from scipy.stats import norm
from scipy.optimize import linprog
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

OUT = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(OUT, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS); JM = 7; J05 = 0; J95 = 14
W = 250
HOR = [20, 40, 60, 120]
KS = np.arange(0.30, 2.01, 0.01)


def qreg_lp(X, y, tau):
    n_, k = X.shape
    c = np.concatenate([np.zeros(k), [tau] * n_ + [1 - tau] * n_])
    A = np.hstack([X, np.eye(n_), -np.eye(n_)])
    r = linprog(c, A_eq=A, b_eq=y, bounds=[(None, None)] * k + [(0, None)] * (2 * n_), method="highs")
    return r.x[:k] if r.success else None


def ewma(lp, lam):
    m = np.empty(len(lp)); m[0] = lp[0]
    for i in range(1, len(lp)):
        m[i] = lam * m[i - 1] + (1 - lam) * lp[i]
    return m


def feats(lp, t, K=20, M=None):
    r = np.diff(lp, prepend=lp[0])
    return np.array([1.0, M[t] - lp[t], r[t], r[t - 1] if t > 0 else 0.0,
                     r[t - 5] if t > 5 else 0.0, lp[t] - lp[t - K] if t > K else 0.0])


def crps_q(yt, Q):
    Qs = np.sort(Q, axis=1)
    ind = (yt[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (yt[:, None] - Qs)).mean(axis=1)


def pick_k(y, Q):
    """只用给定(y,Q)挑 k 使 PICP95 最接近 0.95；平局取最大覆盖<-的k"""
    med = Q[:, JM][:, None]
    cov = [((y >= med[:, 0] + k * (Q[:, J05] - med[:, 0])) &
            (y <= med[:, 0] + k * (Q[:, J95] - med[:, 0]))).mean() for k in KS]
    cov = np.array(cov)
    return float(KS[int(np.argmin(np.abs(cov - 0.95)))])


def apply_k(Q, k):
    med = Q[:, JM][:, None]
    return med + k * (Q - med)


def powerlaw_H(win):
    v1 = np.var(np.diff(win), ddof=1)
    if v1 <= 0 or len(win) < 40:
        return None
    kk = min(20, len(win) // 3)
    vr = np.var(win[kk:] - win[:-kk], ddof=1) / (kk * v1)
    if vr <= 0:
        return None
    return 0.5 + np.log(vr) / (2 * np.log(kk))


def roll_models(lp, ts, h):
    Qv = np.full((len(ts), len(TAUS)), np.nan)
    Qe = np.full((len(ts), len(TAUS)), np.nan)
    Qg = np.full((len(ts), len(TAUS)), np.nan)
    for r, t in enumerate(ts):
        win = lp[max(0, t - W):t]
        if len(win) < h + 40:
            continue
        rr = win[h:] - win[:-h]
        Hh = powerlaw_H(win)
        s1 = np.sqrt(np.var(np.diff(win), ddof=1))
        if Hh is not None and s1 > 0:
            Qv[r] = s1 * (h ** Hh) * GV
        Qe[r] = np.quantile(rr, TAUS)
        sd = np.std(rr, ddof=1)
        if sd > 0:
            Qg[r] = rr.mean() + sd * GV
    return Qv, Qe, Qg


def picp95(y, Q):
    return float(((y >= Q[:, J05]) & (y <= Q[:, J95])).mean())


RES = {}
for m in ["CEA", "GUANGZHOU", "SHANGHAI"]:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64("2024-01-01")))
    M = ewma(lp, 0.7)
    print(f"\n{'='*124}\n{m}   训练起点数={t0}   测试起点数={len(lp)-t0}")
    print(f"{'h':>4}{'n_te':>6}{'QR':>9}{'VR':>9}{'经验':>9}{'Gauss':>9}"
          f"{'QR k':>7}{'VR k':>7}{'经验 k':>8}{'Gauss k':>8}"
          f"{'QR_PICC':>9}{'VR_PICC':>9}{'QR vs VR':>10}{'QR vs 最优':>11}")
    RES[m] = {}
    for h in HOR:
        ts_te = [t for t in range(max(W + h + 5, t0), len(lp) - h)]
        ts_tr = [t for t in range(max(W + h + 5, t0 - 420), t0 - h - 1)]
        if len(ts_te) < 50 or len(ts_tr) < 60:
            continue
        # ---- QR：全部训练段（最大化 QR 的优势，避免因样本截断而受冤）----
        tr = [i for i in range(h + 25, t0) if i + h < t0]
        if len(tr) < 150:
            continue
        Xtr = np.array([feats(lp, i, M=M) for i in tr])
        ytr = lp[np.array(tr) + h] - lp[tr]
        Xte = np.array([feats(lp, t, M=M) for t in ts_te])
        B = []; ok = True
        for tau in TAUS:
            b = qreg_lp(Xtr, ytr, tau)
            if b is None:
                ok = False; break
            B.append(b)
        if not ok:
            continue
        B = np.array(B).T                      # (k_feat, n_tau)
        Qqr_te = Xte @ B
        Qqr_tr = Xtr @ B                      # 训练样本内，用于定 k
        k_qr = pick_k(ytr, Qqr_tr)
        # ---- 滚动三模型 ----
        Qv_tr, Qe_tr, Qg_tr = roll_models(lp, ts_tr, h)
        y_tr = lp[np.array(ts_tr) + h] - lp[ts_tr]
        g_tr = np.isfinite(Qv_tr).all(1) & np.isfinite(Qe_tr).all(1) & np.isfinite(Qg_tr).all(1)
        if g_tr.sum() < 50:
            continue
        k_vr = pick_k(y_tr[g_tr], Qv_tr[g_tr])
        k_em = pick_k(y_tr[g_tr], Qe_tr[g_tr])
        k_ga = pick_k(y_tr[g_tr], Qg_tr[g_tr])
        Qv_te, Qe_te, Qg_te = roll_models(lp, ts_te, h)
        y_te = lp[np.array(ts_te) + h] - lp[ts_te]
        g_te = np.isfinite(Qv_te).all(1) & np.isfinite(Qe_te).all(1) & np.isfinite(Qg_te).all(1)
        if g_te.sum() < 50:
            continue
        y = y_te[g_te]
        A_qr = apply_k(Qqr_te[g_te], k_qr)
        A_vr = apply_k(Qv_te[g_te], k_vr)
        A_em = apply_k(Qe_te[g_te], k_em)
        A_ga = apply_k(Qg_te[g_te], k_ga)
        c = dict(qr=float(crps_q(y, A_qr).mean()), vr=float(crps_q(y, A_vr).mean()),
                 emp=float(crps_q(y, A_em).mean()), gauss=float(crps_q(y, A_ga).mean()))
        best = min(c, key=c.get)
        print(f"{h:>4}{int(g_te.sum()):>6}{c['qr']:>9.5f}{c['vr']:>9.5f}{c['emp']:>9.5f}"
              f"{c['gauss']:>9.5f}{k_qr:>7.2f}{k_vr:>7.2f}{k_em:>8.2f}{k_ga:>8.2f}"
              f"{picp95(y,A_qr):>9.3f}{picp95(y,A_vr):>9.3f}"
              f"{(c['qr']/c['vr']-1)*100:>+9.2f}%{(c['qr']/c[best]-1)*100:>+10.2f}%")
        RES[m][h] = dict(n=int(g_te.sum()), **c, best=best, k_qr=k_qr, k_vr=k_vr,
                         k_em=k_em, k_ga=k_ga, picc_qr=picp95(y, A_qr), picc_vr=picp95(y, A_vr))

print(f"\n{'='*124}")
print("汇总（realistic 校准，k 由训练信息确定）")
print(f"{'市场':<12}{'h':>5}{'QR':>9}{'VR':>9}{'经验':>9}{'Gauss':>9}{'最优':>9}{'QR比最优差':>12}{'VR比最优差':>12}")
WIN = {}
for m in RES:
    for h in sorted(RES[m]):
        r = RES[m][h]
        v = {"QR": r["qr"], "VR": r["vr"], "经验": r["emp"], "Gauss": r["gauss"]}
        b = min(v, key=v.get)
        WIN[b] = WIN.get(b, 0) + 1
        print(f"{m:<12}{h:>5}{r['qr']:>9.5f}{r['vr']:>9.5f}{r['emp']:>9.5f}{r['gauss']:>9.5f}"
              f"{b:>9}{(r['qr']/v[b]-1)*100:>+11.2f}%{(r['vr']/v[b]-1)*100:>+11.2f}%")
print(f"\n  12 个格子的获胜者计数：{WIN}")
json.dump(RES, open(os.path.join(OUT, "QR_LONG_H_REALISTIC.json"), "w"), indent=1,
          ensure_ascii=False, default=float)
print("\n已写出 QR_LONG_H_REALISTIC.json")
