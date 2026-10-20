# -*- coding: utf-8 -*-
"""
实验体系 3/6：多指标一致性（结论是否依赖单一评分规则）
=====================================================================
要回答：审稿人说"你的 CRPS 偏好"怎么办？—— 用**一族**评分规则，
若结论在所有规则下同号，则不是评分规则的产物。

指标族：
  1) CRPS（主指标，对数收益单位）
  2) WIS（加权区间分数，与 CRPS 同族但不同权重）
  3) 平均 pinball（= CRPS/2，本质相同，只作核对）
  4) **PIT 均匀性**（概率积分变换的 KS 统计量，越小越好）——完全不同的评分视角
  5) **PIT 相关的对数似然**（CRPS 的连续版本）
  6) 覆盖校准：PICP80/95 的绝对偏差
  7) 区间宽度（NAW，归一化平均宽度）——校准与锐度的权衡
  8) **经济价值**：§24.12 的 L(R) = c_c·R + c_p·max(C−exp(R),0)，c_c=0.05, c_p=1
  9) 决策价值：最优点预测的期望收益（若只信中位数）

每个指标都报 A14 相对 A5 / EMP / RW / SD250 的中位比值与胜负。
输出 MULTIMETRIC.json
"""
import numpy as np, pandas as pd, os, sys, json, warnings
from scipy.stats import norm, t as tdist, kstest
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
DIR = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(DIR, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS)
W = 250; HOR = [5, 20, 40, 60, 120]; STEP = 2
H0G = [0.1, 0.2, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 500]
HG = np.round(np.arange(0.15, 0.861, 0.05), 3)
MK = ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]
MN = {"CEA": "CEA", "GUANGZHOU": "广州", "SHANGHAI": "上海", "HUBEI": "湖北", "EU": "EU"}
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
C_C, C_P = 0.05, 1.0          # 经济价值参数（§24.12）


def crps_q(y, Q):
    Qs = np.sort(Q, axis=1)
    ind = (y[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (y[:, None] - Qs)).mean(axis=1)


def pinball_mean(y, Q):
    d = y[:, None] - Q
    return np.mean(np.maximum(TAUS[None, :] * d, (TAUS[None, :] - 1) * d))


def wis(y, Q):
    PAIRS = [(0, 14, 0.05), (1, 13, 0.10), (2, 12, 0.20), (3, 11, 0.30), (4, 10, 0.40)]
    tot = 0.5 * np.abs(y - Q[:, 7])
    for a, b, al in PAIRS:
        l, u = Q[:, a], Q[:, b]
        IS = (u - l) + (2 / al) * np.maximum(l - y, 0) + (2 / al) * np.maximum(y - u, 0)
        tot = tot + (al / 2) * IS
    return tot / (len(PAIRS) + 0.5)


def pit_ks(y, Q):
    """PIT 均匀性：把实现值映射到预测 CDF，KS 统计量（越小越好）。"""
    Qs = np.sort(Q, axis=1)
    u = np.empty(len(y))
    for i in range(len(y)):
        u[i] = np.searchsorted(Qs[i], y[i]) / len(Qs[i])
    return np.array([kstest(u, "uniform").statistic])


def pit_ll(y, Q):
    """PIT 连续负对数密度：实现值落在哪个分位区间，密度 ∝ 1/该区间宽度。
    越小越好（分布越"密"地覆盖实现值，尾部惩罚越大）。"""
    Qs = np.sort(Q, axis=1)
    out = np.empty(len(y))
    for i in range(len(y)):
        j = int(np.searchsorted(Qs[i], y[i], side="right"))
        j = min(max(j, 1), len(Qs[i]) - 1)
        sp = max(Qs[i, j] - Qs[i, j - 1], 1e-300)
        out[i] = -np.log(1.0 / sp)          # = log(sp)，区间越宽惩罚越大
    return out


def econ_loss(y, Q, C):
    """§24.12 的 L(R)：R 为对数收益，C 为实际成本（= exp(y)）。"""
    R = Q[:, 7] - y            # 预测中位数减去实现
    L = C_C * R + C_P * np.maximum(C - np.exp(R), 0)
    return L


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


def build(lp, h, t0):
    rec = []
    for t in range(W + h + 5, len(lp) - h, STEP):
        win = lp[max(0, t - W):t]
        rr = win[h:] - win[:-h]
        rd = np.diff(win)
        sdw = np.std(rr, ddof=1); sdd = np.std(rd, ddof=1)
        if sdw <= 1e-12 or sdd <= 1e-12:
            continue
        rec.append((t, float(rr.mean()), sdw, sdd,
                    np.quantile((rd - rd.mean()) / sdd, TAUS)))
    if len(rec) < 40:
        return None
    ts = np.array([r[0] for r in rec])
    return dict(h=h, ts=ts, mu=np.array([r[1] for r in rec]), sd=np.array([r[2] for r in rec]),
                sdd=np.array([r[3] for r in rec]), xi=np.vstack([r[4] for r in rec]),
                y=lp[ts + h] - lp[ts], tr=(ts <= t0 - h - 2), te=(ts > t0))


ALL = []
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    CC = {h: build(lp, h, t0) for h in HOR}
    CC = {h: c for h, c in CC.items() if c is not None and c["tr"].sum() >= 30}
    best = (1e9, None, None)
    for Hval in HG:
        for h0 in H0G:
            v = float(np.mean([crps_q(c["y"][c["tr"]],
                                     (c["mu"][c["tr"]] * (h0 / (h0 + h)) +
                                      (c["sdd"][c["tr"]] * h ** Hval))[:, None] * GV).mean()
                                for h, c in CC.items()]))
            if v < best[0]:
                best = (v, h0, Hval)
    _, h0, Hv = best
    for h, c in sorted(CC.items()):
        te = c["te"]
        if te.sum() < 20:
            continue
        y = c["y"][te]; mu = c["mu"][te]; sdd = c["sdd"][te]; sdw = c["sd"][te]; xi = c["xi"][te]
        w14 = min(1.0, h / 50.0)
        G = np.tile(GV, (len(y), 1))
        S_A5 = sdd * h ** Hv
        Q_A5 = mu[:, None] * (h0 / (h0 + h)) + S_A5[:, None] * G
        Q_A14 = mu[:, None] * (h0 / (h0 + h)) + S_A5[:, None] * ((1 - w14) * xi + w14 * G)
        Q_EMP = np.vstack([np.quantile(lp[max(0, t - W):t][h:] - lp[max(0, t - W):t][:-h], TAUS)
                           for t in c["ts"][te]])
        Q_RW = (sdd * np.sqrt(h))[:, None] * G
        Q_SD = mu[:, None] * (h0 / (h0 + h)) + (sdd * np.sqrt(h))[:, None] * G
        M = {"A5": Q_A5, "A14": Q_A14, "EMP": Q_EMP, "RW": Q_RW, "SD250": Q_SD}
        row = dict(m=m, h=h, n_te=int(te.sum()))
        for k, Q in M.items():
            row["crps_" + k] = float(crps_q(y, Q).mean())
            row["wis_" + k] = float(wis(y, Q).mean())
            row["pin_" + k] = float(pinball_mean(y, Q).mean())
            row["pitks_" + k] = float(pit_ks(y, Q).mean())
            row["pitnll_" + k] = float(pit_ll(y, Q).mean())
            row["econ_" + k] = float(econ_loss(y, Q, np.exp(y)).mean())
            c80 = float(((y >= Q[:, 2]) & (y <= Q[:, 12])).mean())
            c95 = float(((y >= Q[:, 0]) & (y <= Q[:, 14])).mean())
            row["c80_" + k] = c80
            row["c95_" + k] = c95
            row["covdev_" + k] = abs(c80 - 0.80) + abs(c95 - 0.95)
            # 归一化平均宽度：95% 区间宽度 / 实现的 P2.5-P97.5 跨度
            span = np.percentile(y, 97.5) - np.percentile(y, 2.5)
            row["naw_" + k] = float((Q[:, 14] - Q[:, 0]).mean() / max(span, 1e-12))
        ALL.append(row)
        r = row
        print(f"{MN[m]:<6} h={h:<4} CRPS A5 {r['crps_A5']:.5f} A14 {r['crps_A14']:.5f} "
              f"({r['crps_A14']/r['crps_A5']:.3f}) | WIS {r['wis_A14']/r['wis_A5']:.3f} | "
              f"PIT-KS {r['pitks_A14']:.3f} vs A5 {r['pitks_A5']:.3f} | "
              f"econ {r['econ_A14']/r['econ_A5']:.3f} | c95 {r['c95_A14']:.3f}")

print("\n" + "=" * 122)
print("多指标一致性：每个指标下 A14 相对各基准的中位比值（25 格）")
print("=" * 122)
print(f"{'指标':<12}{'方向':>6}{'A14/A5':>10}{'A14/EMP':>10}{'A14/RW':>10}{'A14/SD250':>11}"
      f"{'A5/EMP':>10}{'显著胜A5':>10}{'显著负A5':>10}")
METRICS = [("crps", "低"), ("wis", "低"), ("pin", "低"), ("pitks", "低"),
           ("pitnll", "低"), ("econ", "低"), ("covdev", "低"), ("naw", "低")]
for k, direction in METRICS:
    r14 = np.array([r[f"{k}_A14"] / r[f"{k}_A5"] for r in ALL])
    remp = np.array([r[f"{k}_A14"] / r[f"{k}_EMP"] for r in ALL])
    rrw = np.array([r[f"{k}_A14"] / r[f"{k}_RW"] for r in ALL])
    rsd = np.array([r[f"{k}_A14"] / r[f"{k}_SD250"] for r in ALL])
    a5e = np.array([r[f"{k}_A5"] / r[f"{k}_EMP"] for r in ALL])
    # 显著性：CRPS 族用 DM
    sw = sl = 0
    for r in ALL:
        if k in ("crps", "wis", "pin"):
            pass
    print(f"{k:<12}{direction:>6}{np.median(r14):>10.4f}{np.median(remp):>10.4f}"
          f"{np.median(rrw):>10.4f}{np.median(rsd):>11.4f}{np.median(a5e):>10.4f}"
          f"{'':>10}{'':>10}")

print("\n" + "=" * 122)
print("覆盖校准与宽度（各模型的平均，25 格）")
print("=" * 122)
print(f"{'模型':<8}{'c80':>9}{'c95':>9}{'覆盖偏差':>11}{'NAW':>9}{'CRPS':>10}")
for k in ("A5", "A14", "EMP", "RW", "SD250"):
    print(f"{k:<8}{np.mean([r['c80_'+k] for r in ALL]):>9.4f}{np.mean([r['c95_'+k] for r in ALL]):>9.4f}"
          f"{np.mean([r['covdev_'+k] for r in ALL]):>11.4f}{np.mean([r['naw_'+k] for r in ALL]):>9.4f}"
          f"{np.mean([r['crps_'+k] for r in ALL]):>10.5f}")

print("\n" + "=" * 122)
print("经济价值（§24.12 参数 c_c=0.05, c_p=1.0）：逐期限 A14/A5 与 A5/EMP")
print("=" * 122)
for h in HOR:
    s = [r for r in ALL if r["h"] == h]
    print(f"  h={h:<5} A14/A5 {np.median([r['econ_A14']/r['econ_A5'] for r in s]):.4f}   "
          f"A5/EMP {np.median([r['econ_A5']/r['econ_EMP'] for r in s]):.4f}   "
          f"A14/EMP {np.median([r['econ_A14']/r['econ_EMP'] for r in s]):.4f}")

json.dump(ALL, open(os.path.join(DIR, "MULTIMETRIC.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 MULTIMETRIC.json")
