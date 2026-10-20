# -*- coding: utf-8 -*-
"""
关键诊断：A5 的位置项 λ·μ̂ 在长端是否符号错？
问题：μ̂ = 窗口内已实现的 h 步漂移。均值回复市场里，近窗漂移与远期漂移可能负相关。
若 corr(μ̂_t, y_{t,h}) 在长 h 为负，则位置项必须反号或衰减到 0，而不是"调整 λ"。
输出：
  1) 训练段 corr(μ̂, y) 与 regression slope，逐 (市场, h)
  2) 训练段估计的 slope 用到测试段的效果
  3) 一个结构修正的原型：位置项改为 λ(h)·[μ̂ - c·μ̄] 或直接用 slope 收缩
只读数据，不改任何模型。
"""
import numpy as np, pandas as pd, os, sys, json
from scipy.stats import t as tdist
sys.stdout.reconfigure(encoding="utf-8")
OUT = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(OUT, "PANEL_5MARKET.csv"), parse_dates=["date"])
W = 250; HOR = [5, 20, 40, 60, 120]; STEP = 2
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
PAR = {"CEA": (0.1, 0.60), "GUANGZHOU": (30, 0.35), "SHANGHAI": (0.1, 0.30),
       "HUBEI": (3, 0.35), "EU": (80, 0.45)}
MN = {"CEA": "CEA", "GUANGZHOU": "广州", "SHANGHAI": "上海", "HUBEI": "湖北", "EU": "EU"}


def crps_q(y, Q, TAUS):
    Qs = np.sort(Q, axis=1)
    ind = (y[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (y[:, None] - Qs)).mean(axis=1)


TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
from scipy.stats import norm
GV = norm.ppf(TAUS)

OUTR = {}
print("=" * 126)
print("一、训练段：corr(μ̂, y) 与回归斜率 slope（y = a + b·μ̂）")
print("   若 b < 0，则位置项在长端符号错；若 b≈0，则位置项无信息、应删除")
print("=" * 126)
print(f"{'市场':<8}{'h':>5}{'n_tr':>7}{'corr':>9}{'slope':>10}{'slope t':>10}"
      f"{'a(=均值)':>11}{'μ̂ sd':>9}{'y sd':>9}")
FIT = {}
for m in CUT:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    for h in HOR:
        MU, Y, TT = [], [], []
        for t in range(W + h + 5, t0 - h - 1, STEP):
            win = lp[t - W:t]
            r = np.diff(win)
            sdd = np.std(r, ddof=1)
            if not np.isfinite(sdd) or sdd <= 0:
                continue
            MU.append((win[h:] - win[:-h]).mean())
            Y.append(lp[t + h] - lp[t]); TT.append(t)
        MU, Y, TT = np.array(MU), np.array(Y), np.array(TT)
        n = len(MU)
        if n < 30:
            continue
        c = np.corrcoef(MU, Y)[0, 1]
        X = np.column_stack([np.ones(n), MU])
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        res = Y - X @ beta
        s2 = res @ res / (n - 2)
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * s2)
        tstat = beta[1] / se[1]
        FIT[(m, h)] = dict(corr=float(c), slope=float(beta[1]), a=float(beta[0]),
                           t=float(tstat), n=n, mu_sd=float(MU.std(ddof=1)),
                           y_sd=float(Y.std(ddof=1)))
        print(f"{MN[m]:<8}{h:>5}{n:>7}{c:>9.4f}{beta[1]:>10.4f}{tstat:>10.3f}"
              f"{beta[0]:>11.5f}{MU.std(ddof=1):>9.5f}{Y.std(ddof=1):>9.5f}")

print("\n" + "=" * 126)
print("二、测试段：把训练段的 slope 用到测试段")
print("=" * 126)
print(f"{'市场':<8}{'h':>5}{'n_te':>7}{'corr_te':>9}{'slope_te':>10}{'slope_tr':>10}"
      f"{'方向一致?':>11}")
for (m, h), f in FIT.items():
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    MU, Y = [], []
    for t in range(max(W + h + 5, t0), len(lp) - h, STEP):
        win = lp[t - W:t]; r = np.diff(win)
        sdd = np.std(r, ddof=1)
        if not np.isfinite(sdd) or sdd <= 0:
            continue
        MU.append((win[h:] - win[:-h]).mean()); Y.append(lp[t + h] - lp[t])
    MU, Y = np.array(MU), np.array(Y)
    if len(MU) < 20:
        continue
    c = np.corrcoef(MU, Y)[0, 1]
    X = np.column_stack([np.ones(len(MU)), MU])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    same = "是" if (np.sign(beta[1]) == np.sign(f["slope"])) else "**否**"
    print(f"{MN[m]:<8}{h:>5}{len(MU):>7}{c:>9.4f}{beta[1]:>10.4f}{f['slope']:>10.4f}{same:>11}")

print("\n" + "=" * 126)
print("三、'位置项在长端符号错'的更强证据：标准化残差均值 z̄（来自 SHAPE_DIAG）")
print("   z̄ << 0 表示实际收益系统性低于预测中心；而 λ>0 说明预测中心被抬高")
print("=" * 126)
DG = json.load(open(os.path.join(OUT, "SHAPE_DIAG.json"), encoding="utf-8"))
print(f"{'市场':<8}{'h':>5}{'λ(h)':>9}{'z̄ 测试段':>11}{'z̄/λ':>10}{'解读':>24}")
for m in CUT:
    h0, Hv = PAR[m]
    for h in HOR:
        k = f"{m}|{h}"
        if k not in DG:
            continue
        lam = h0 / (h0 + h)
        zb = DG[k]["mean"]
        ratio = zb / lam if lam > 0 else np.nan
        verdict = "位置项抬高过多" if (lam > 0.05 and zb < -0.2) else \
                  ("位置项无害" if abs(zb) < 0.2 else "需看方向")
        print(f"{MN[m]:<8}{h:>5}{lam:>9.4f}{zb:>11.4f}{ratio:>10.3f}{verdict:>24}")

json.dump({f"{m}|{h}": v for (m, h), v in FIT.items()},
          open(os.path.join(OUT, "MU_SLOPE.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 MU_SLOPE.json")
