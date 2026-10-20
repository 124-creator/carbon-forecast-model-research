# -*- coding: utf-8 -*-
"""
诊断：A5 的高斯形状假设在每个 (市场, 期限) 上错在哪里？
输出 PIT 直方图统计、标准化收益的偏度/峰度/尾概率，供 §24.29 选模型用。
只读数据，不改任何模型。
"""
import numpy as np, pandas as pd, os, sys, json
from scipy.stats import norm, kurtosis, skew, t as tdist
sys.stdout.reconfigure(encoding="utf-8")
OUT = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(OUT, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
W = 250; HOR = [5, 20, 40, 60, 120]; STEP = 2
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
PAR = {"CEA": (0.1, 0.60), "GUANGZHOU": (30, 0.35), "SHANGHAI": (0.1, 0.30),
       "HUBEI": (3, 0.35), "EU": (80, 0.45)}
GV = norm.ppf(TAUS)
MN = {"CEA": "CEA", "GUANGZHOU": "广州", "SHANGHAI": "上海", "HUBEI": "湖北", "EU": "EU"}

RES = {}
print("=" * 120)
print("A. 标准化残差的分布形状（z = (y - λμ̂) / (σ̂_d · h^H)）")
print("   若 A5 的形状假设正确，z 应近似标准正态：偏度 0、峰度 3")
print("=" * 120)
print(f"{'市场':<8}{'h':>5}{'n':>6}{'均值':>9}{'标准差':>9}{'偏度':>9}{'峰度':>9}"
      f"{'P(|z|>1.645)':>13}{'P(|z|>2.326)':>13}{'KS(正态)':>11}{'p':>9}")
ALLZ = {}
for m in CUT:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    h0, Hv = PAR[m]
    for h in HOR:
        ts_te = [t for t in range(max(W + h + 5, t0), len(lp) - h, STEP)]
        Z = []
        for t in ts_te:
            win = lp[t - W:t]; r = np.diff(win)
            sdd = np.std(r, ddof=1)
            if not np.isfinite(sdd) or sdd <= 0:
                continue
            y = lp[t + h] - lp[t]
            mu = (win[h:] - win[:-h]).mean()
            s = sdd * h ** Hv
            Z.append((y - (h0 / (h0 + h)) * mu) / s)
        Z = np.array(Z)
        ALLZ[(m, h)] = Z
        if len(Z) < 20:
            continue
        z1 = float(np.mean(np.abs(Z) > 1.645)); z2 = float(np.mean(np.abs(Z) > 2.326))
        ks = norm.cdf(Z)
        ks = np.sort(ks)
        n = len(ks)
        D = max(np.max(np.arange(1, n + 1) / n - ks), np.max(ks - np.arange(0, n) / n))
        from scipy.stats import kstest
        ksr, kp = kstest(ks, 'uniform')
        RES[f"{m}|{h}"] = dict(n=n, mean=float(Z.mean()), sd=float(Z.std(ddof=1)),
                               skew=float(skew(Z)), kurt=float(kurtosis(Z)),
                               p165=z1, p2326=z2, ks=float(ksr), kp=float(kp))
        print(f"{MN[m]:<8}{h:>5}{n:>6}{Z.mean():>9.4f}{Z.std(ddof=1):>9.4f}"
              f"{skew(Z):>9.3f}{kurtosis(Z):>9.3f}{z1:>13.4f}{z2:>13.4f}{ksr:>11.4f}{kp:>9.4f}")

print("\n" + "=" * 120)
print("B. 归一化后的 PIT 均匀性分段（把 z 反变换成 CDF 值，理想应均匀）")
print("=" * 120)
print(f"{'市场':<8}{'h':>5}{'PIT<0.05':>11}{'0.05-0.95':>11}{'PIT>0.95':>11}{'PIT中位':>10}")
for m in CUT:
    for h in HOR:
        Z = ALLZ.get((m, h))
        if Z is None or len(Z) < 20:
            continue
        u = norm.cdf(Z)
        print(f"{MN[m]:<8}{h:>5}{np.mean(u<0.05):>11.4f}{np.mean((u>=0.05)&(u<=0.95)):>11.4f}"
              f"{np.mean(u>0.95):>11.4f}{np.median(u):>10.4f}")

print("\n" + "=" * 120)
print("C. 尾部厚度的期限依赖（决定 t 分布自由度的方向）")
print("=" * 120)
print(f"{'市场':<8}{'h':>5}{'超额峰度':>10}{'t_nu(标准化后)':>16}{'建议 nu':>10}")
for m in CUT:
    for h in HOR:
        Z = ALLZ.get((m, h))
        if Z is None or len(Z) < 20:
            continue
        Zs = (Z - Z.mean()) / Z.std(ddof=1)
        ex = float(kurtosis(Zs))
        # 峰度与 t 分布 nu 的近似关系： kurt(t_nu) = 6/(nu-4) + 3
        nu = 4 + 6 / (ex - 3) if ex > 3.05 else np.inf
        print(f"{MN[m]:<8}{h:>5}{ex:>10.3f}{(f'{nu:.1f}' if np.isfinite(nu) else 'inf'):>16}"
              f"{(f'{nu:.1f}' if np.isfinite(nu) else 'inf'):>10}")

print("\n" + "=" * 120)
print("D. 训练段 vs 测试段 的 z 分布是否漂移（检验'形状'是否随时间变）")
print("=" * 120)
print(f"{'市场':<8}{'h':>5}{'训练偏度':>10}{'测试偏度':>10}{'训练峰度':>10}{'测试峰度':>10}")
for m in CUT:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    h0, Hv = PAR[m]
    for h in HOR:
        def zs(ts):
            out = []
            for t in ts:
                win = lp[t - W:t]; r = np.diff(win)
                sdd = np.std(r, ddof=1)
                if not np.isfinite(sdd) or sdd <= 0:
                    continue
                y = lp[t + h] - lp[t]
                mu = (win[h:] - win[:-h]).mean()
                out.append((y - (h0 / (h0 + h)) * mu) / (sdd * h ** Hv))
            return np.array(out)
        tr = zs([t for t in range(W + h + 5, t0 - h - 1, STEP)])
        te = zs([t for t in range(max(W + h + 5, t0), len(lp) - h, STEP)])
        if len(tr) < 20 or len(te) < 20:
            continue
        print(f"{MN[m]:<8}{h:>5}{skew(tr):>10.3f}{skew(te):>10.3f}"
              f"{kurtosis(tr):>10.3f}{kurtosis(te):>10.3f}")

json.dump(RES, open(os.path.join(OUT, "SHAPE_DIAG.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
np.save(os.path.join(r"E:\Agent员工\碳价格预测研究\_npzcheck", "allz.npy"),
        np.array([ALLZ[k] for k in sorted(ALLZ)], dtype=object), allow_pickle=True)
print("\n已写出 SHAPE_DIAG.json")
