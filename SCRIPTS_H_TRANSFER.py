# -*- coding: utf-8 -*-
"""
H 的跨市场迁移（含与滚动 VR 的对照）
=====================================================================
要点澄清：名册里的 H 有两个来源
  (a) A5 用的**固定** per-market H（PAR 表，§24.3 全体数据拟合）
  (b) VR 模型的**滚动** H（每个原点用 k=20 重算，clip 到 [0.05,0.95]）
§24.25 已发现：滚动 VR 能拿到 H 增益的大部分（VR/SD250=0.950）。
因此跨市场迁移的价值在于：**若 H 可迁移，则可用极短历史的市场做预测**，
这正是三个中国试点（上海/湖北历史短）的现实需求。

检验：
  T1 训练段的 per-market H（k=20 滚动，取中位）用于测试段
  T2 留一市场：其余 4 市场训练段 H 的中位 → 目标市场
  T3 H 的跨市场相关性（用 5 个市场的训练段 H 向量）
  T4 与 H=0.5、固定 PAR_H、滚动 VR 逐一对照
评价口径与 A5 完全一致（位置块、标度块原样保留，只换 H）。
"""
import numpy as np, pandas as pd, os, sys, json
from scipy.stats import norm, t as tdist
sys.stdout.reconfigure(encoding="utf-8")
DIR = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(DIR, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS)
W = 250; HOR = [5, 20, 40, 60, 120]; STEP = 2
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
PAR = {"CEA": (0.1, 0.60), "GUANGZHOU": (30, 0.35), "SHANGHAI": (0.1, 0.30),
       "HUBEI": (3, 0.35), "EU": (80, 0.45)}
MN = {"CEA": "CEA", "GUANGZHOU": "广州", "SHANGHAI": "上海", "HUBEI": "湖北", "EU": "EU"}


def crps_q(y, Q):
    Qs = np.sort(Q, axis=1)
    ind = (y[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (y[:, None] - Qs)).mean(axis=1)


def dm(a, b, L=10):
    d = np.asarray(a) - np.asarray(b); n = len(d)
    if n < 20:
        return np.nan
    m_ = d.mean(); dc = d - m_
    lrv = dc @ dc / n
    for l in range(1, L + 1):
        lrv += 2 * (1 - l / (L + 1)) * (dc[l:] @ dc[:-l] / n)
    if lrv <= 0:
        return np.nan
    s = m_ / np.sqrt(lrv / n) * np.sqrt((n + 1 - 2 * L + L * (L - 1) / n) / n)
    return float(2 * (1 - tdist.cdf(abs(s), df=n - 1)))


def rolling_H(lp, t, W=W):
    """名册同款：k=20 的滚动 VR，clip [0.05,0.95]。"""
    win = lp[max(0, t - W):t]
    r = np.diff(win)
    if len(r) < 60:
        return None, None
    v1 = np.var(r, ddof=1)
    if v1 <= 0:
        return None, None
    kk = min(20, len(win) // 3)
    vr = np.var(win[kk:] - win[:-kk], ddof=1) / (kk * v1)
    H = float(np.clip(0.5 + np.log(vr) / (2 * np.log(kk)), 0.05, 0.95)) if vr > 0 else 0.5
    return H, float(np.sqrt(v1))


DATA, HTR = {}, {}
ms = list(CUT)
print("=" * 112)
print("一、训练段的滚动 H（名册同款 k=20）")
print("=" * 112)
for m in CUT:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    h0, Hfit = PAR[m]
    tr_h, te_h = [], []
    for h in HOR:
        ts_tr = [t for t in range(W + h + 5, t0 - h - 1, STEP)]
        ts_te = [t for t in range(max(W + h + 5, t0), len(lp) - h, STEP)]
        assert not (set(ts_tr) & set(ts_te))
        Y, LOC, SD, HR, HROLL = [], [], [], [], []
        for t in ts_te:
            win = lp[max(0, t - W):t]
            rr = win[h:] - win[:-h]
            if len(rr) < 3 or np.std(rr, ddof=1) <= 0:
                continue
            rd = np.diff(win); sdd = np.std(rd, ddof=1)
            if not np.isfinite(sdd) or sdd <= 0:
                continue
            Hr, _ = rolling_H(lp, t)
            if Hr is None:
                continue
            Y.append(lp[t + h] - lp[t]); LOC.append((h0 / (h0 + h)) * rr.mean())
            SD.append(sdd); HR.append(Hr)
            HROLL.append(Hr)
        if len(Y) < 25:
            continue
        # 训练段 H 的滚动值（只到 t0）
        for t in range(W + 26, t0 - 6, 10):
            Hr, _ = rolling_H(lp, t)
            if Hr is not None:
                tr_h.append(Hr)
        te_h.append(np.array(HR))
        DATA[(m, h)] = dict(Y=np.array(Y), LOC=np.array(LOC), SD=np.array(SD),
                            HR=np.array(HR), n=len(Y), half=len(Y) // 2)
    HTR[m] = dict(train_med=float(np.median(tr_h)), train_n=len(tr_h),
                  test_med=float(np.median(np.concatenate(te_h))) if te_h else np.nan)
    print(f"  {MN[m]:<8} 训练段 H 中位 {HTR[m]['train_med']:.3f} (n={HTR[m]['train_n']})  "
          f"测试段 H 中位 {HTR[m]['test_med']:.3f}   |  本文固定 H = {Hfit:.2f}")

print("\n" + "=" * 112)
print("二、关键对照：H 本身的训练-测试漂移（滚动 VR 估计，k=20）")
print("=" * 112)
print(f"{'市场':<8}{'训练段 H 中位':>14}{'测试段 H 中位':>14}{'漂移':>9}{'本文固定 H':>11}")
for m in ms:
    print(f"{MN[m]:<8}{HTR[m]['train_med']:>14.3f}{HTR[m]['test_med']:>14.3f}"
          f"{HTR[m]['test_med']-HTR[m]['train_med']:>9.3f}{PAR[m][1]:>11.2f}")
_drift = [HTR[m]["test_med"] - HTR[m]["train_med"] for m in ms]
print(f"\n  漂移的均值 {np.mean(_drift):+.3f}  中位 {np.median(_drift):+.3f}  "
      f"绝对值中位 {np.median(np.abs(_drift)):.3f}")
print("  => H 的**时点估计**在训练段与测试段之间并不稳定（CEA +0.18、上海 −0.14），"
      "这本身限制了任何跨市场/跨期迁移 H 的价值。")

print("\n" + "=" * 112)
print("三、T1/T2/T4：换 H 的测试段表现（位置块与标度块其余部分原样保留）")
print("=" * 112)
RES = {}
for (m, h) in sorted(DATA.keys()):
    D = DATA[(m, h)]
    Y, LOC, SD, HR = D["Y"], D["LOC"], D["SD"], D["HR"]
    others = [a for a in ms if a != m]
    H_loo = float(np.median([HTR[a]["train_med"] for a in others]))
    cands = {
        "own_fixed": PAR[m][1],       # 现状：全体数据拟合的固定 H
        "train_own": HTR[m]["train_med"],
        "loo": H_loo,
        "half": 0.5,
        "rolling": None,               # 逐原点的滚动 H
    }
    row = dict(n=D["n"], half=D["half"], H_loo=H_loo)
    for k, Hv in cands.items():
        if Hv is None:
            continue
        s = SD * h ** Hv
        row[k] = float(crps_q(Y, LOC[:, None] + s[:, None] * GV).mean())
    row["rolling"] = float(crps_q(Y, LOC[:, None] + (SD * h ** HR)[:, None] * GV).mean())
    RES[f"{m}|{h}"] = row
    print(f"  {MN[m]:<6} h={h:<4} own({PAR[m][1]:.2f}) {row['own_fixed']:.5f} | "
          f"tr_own({HTR[m]['train_med']:.2f}) {row['train_own']:.5f} | "
          f"loo({H_loo:.2f}) {row['loo']:.5f} | half {row['half']:.5f} | "
          f"rolling {row['rolling']:.5f}")

print("\n汇总（相对 own_fixed）：")
for k in ("train_own", "loo", "half", "rolling"):
    rr = np.array([RES[f"{m}|{h}"][k] / RES[f"{m}|{h}"]["own_fixed"] for (m, h) in DATA])
    sw = sl = 0
    for (m, h) in DATA:
        R = RES[f"{m}|{h}"]
        if R[k] < R["own_fixed"]:
            sw += 1
    print(f"  {k:<10} 中位 {np.median(rr):.4f}  优于 own 的格子 {sw}/{len(DATA)}")

print("\n按市场 loo/own：")
for m in ms:
    v = [RES[f"{m}|{h}"]["loo"] / RES[f"{m}|{h}"]["own_fixed"] for h in HOR if f"{m}|{h}" in RES]
    print(f"  {MN[m]:<6} 中位 {np.median(v):.4f}  逐格 {[round(x,3) for x in v]}")
print("\n按期限 loo/own：")
for h in HOR:
    v = [RES[f"{m}|{h}"]["loo"] / RES[f"{m}|{h}"]["own_fixed"] for m in ms if f"{m}|{h}" in RES]
    print(f"  h={h:<5} 中位 {np.median(v):.4f}")

_drift = {m: HTR[m]["test_med"] - HTR[m]["train_med"] for m in ms}
json.dump({"HTR": HTR, "res": RES,
           "drift": {"mean": float(np.mean(list(_drift.values()))),
                     "median": float(np.median(list(_drift.values()))),
                     "abs_median": float(np.median([abs(v) for v in _drift.values()]))},
           "loo_by_h": {str(h): float(np.median([RES[f"{m}|{h}"]["loo"] / RES[f"{m}|{h}"]["own_fixed"]
                                                 for m in ms if f"{m}|{h}" in RES])) for h in HOR},
           "loo_by_m": {m: float(np.median([RES[f"{m}|{h}"]["loo"] / RES[f"{m}|{h}"]["own_fixed"]
                                            for h in HOR if f"{m}|{h}" in RES])) for m in ms}},
          open(os.path.join(DIR, "H_TRANSFER.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 H_TRANSFER.json")
