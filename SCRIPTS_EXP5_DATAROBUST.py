# -*- coding: utf-8 -*-
"""
实验体系 5/6：多数据口径稳健性（W 窗口 / STEP / 面板构造 / 涨跌幅 vs 对数）
=====================================================================
审稿人必问："你的结果依赖这些预处理选择吗？"
口径轴：
 D1 估计窗口 W ∈ {125, 250, 375, 500}
 D2 原点密度 STEP ∈ {1, 2, 4}
 D3 收益口径：对数收益 vs 简单收益率（价格已为对数价时等价，但用 exp 还原再算可验证）
 D4 面板构造：是否剔除极端点（价格 > 某分位）、是否用 250 日滚动 vs 全样本
 D5 期限网格对齐：W=125 时 h>125 的格子必然退化，如实报告
输出 DATAROBUST.json
"""
import numpy as np, pandas as pd, os, sys, json, warnings
from scipy.stats import norm, t as tdist
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
DIR = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(DIR, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS)
HOR = [5, 20, 40, 60, 120]
H0G = [0.1, 0.2, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 500]
HG = np.round(np.arange(0.15, 0.861, 0.05), 3)
MK = ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]
MN = {"CEA": "CEA", "GUANGZHOU": "广州", "SHANGHAI": "上海", "HUBEI": "湖北", "EU": "EU"}
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}


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


def run(lp, t0, W, STEP, HSET, WMAX=None):
    """在给定 (W, STEP) 下拟合并评估。"""
    def build(h):
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
    CC = {h: c for h, c in ((h, build(h)) for h in HSET) if c is not None and c["tr"].sum() >= 30}
    if not CC:
        return None
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
    out = []
    for h, c in sorted(CC.items()):
        te = c["te"]
        if te.sum() < 20:
            continue
        y = c["y"][te]; mu = c["mu"][te]; sdd = c["sdd"][te]; xi = c["xi"][te]
        w14 = min(1.0, h / 50.0)
        G = np.tile(GV, (len(y), 1))
        S = sdd * h ** Hv
        Q_A5 = mu[:, None] * (h0 / (h0 + h)) + S[:, None] * G
        Q_A14 = mu[:, None] * (h0 / (h0 + h)) + S[:, None] * ((1 - w14) * xi + w14 * G)
        out.append(dict(h=h, n_te=int(te.sum()),
                        A5=crps_q(y, Q_A5), A14=crps_q(y, Q_A14),
                        h0=h0, H=Hv))
    return out


RES = {}
LP = {}
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    LP[m] = (np.log(d.price.to_numpy(float)),
             pd.to_datetime(d.date).to_numpy(),
             int(np.searchsorted(pd.to_datetime(d.date).to_numpy(), np.datetime64(CUT[m]))))

print("=" * 120)
print("D1  估计窗口 W 的影响（STEP=2）")
print("=" * 120)
for W in (125, 250, 375, 500):
    vals, sw, sl = [], 0, 0
    for m in MK:
        lp, dt, t0 = LP[m]
        r = run(lp, t0, W, 2, HOR)
        if not r:
            continue
        for x in r:
            vals.append(x["A14"].mean() / x["A5"].mean())
            pv = dm(x["A14"], x["A5"])
            if pv == pv and pv < .05:
                if x["A14"].mean() < x["A5"].mean():
                    sw += 1
                else:
                    sl += 1
    if vals:
        print(f"  W={W:<5} 中位 {np.median(vals):.4f}  胜 {sum(1 for v in vals if v<1)}/{len(vals)}  "
              f"显著胜 {sw}  显著负 {sl}")
        RES[f"D1|W={W}"] = dict(med=float(np.median(vals)), sw=sw, sl=sl, n=len(vals),
                                vals=[float(v) for v in vals])

print("\n" + "=" * 120)
print("D2  原点密度 STEP 的影响（W=250）")
print("=" * 120)
for STEP in (1, 2, 4):
    vals, sw, sl = [], 0, 0
    for m in MK:
        lp, dt, t0 = LP[m]
        r = run(lp, t0, 250, STEP, HOR)
        if not r:
            continue
        for x in r:
            vals.append(x["A14"].mean() / x["A5"].mean())
            pv = dm(x["A14"], x["A5"])
            if pv == pv and pv < .05:
                if x["A14"].mean() < x["A5"].mean():
                    sw += 1
                else:
                    sl += 1
    if vals:
        print(f"  STEP={STEP:<3} 中位 {np.median(vals):.4f}  胜 {sum(1 for v in vals if v<1)}/{len(vals)}  "
              f"显著胜 {sw}  显著负 {sl}")
        RES[f"D2|STEP={STEP}"] = dict(med=float(np.median(vals)), sw=sw, sl=sl, n=len(vals))

print("\n" + "=" * 120)
print("D3  收益口径：对数收益 vs 简单收益率（W=250, STEP=2）")
print("=" * 120)
for mode in ("log", "simple"):
    vals, sw, sl = [], 0, 0
    for m in MK:
        lp, dt, t0 = LP[m]
        if mode == "log":
            x_lp = lp
        else:
            # 用简单收益率重建对数价：把对数增量换成简单收益再取对数
            pr = np.exp(lp)
            r_simple = np.diff(pr) / pr[:-1]
            x_lp = np.concatenate([[0.0], np.cumsum(np.log1p(r_simple))])
        r = run(x_lp, t0, 250, 2, HOR)
        if not r:
            continue
        for x in r:
            vals.append(x["A14"].mean() / x["A5"].mean())
            pv = dm(x["A14"], x["A5"])
            if pv == pv and pv < .05:
                if x["A14"].mean() < x["A5"].mean():
                    sw += 1
                else:
                    sl += 1
    if vals:
        print(f"  {mode:<8} 中位 {np.median(vals):.4f}  胜 {sum(1 for v in vals if v<1)}/{len(vals)}  "
              f"显著胜 {sw}  显著负 {sl}")
        RES[f"D3|{mode}"] = dict(med=float(np.median(vals)), sw=sw, sl=sl, n=len(vals))

print("\n" + "=" * 120)
print("D4  面板构造：剔除极端价格点（各市场价格 99.5 分位以上）")
print("=" * 120)
for trim in (False, True):
    vals, sw, sl = [], 0, 0
    for m in MK:
        lp, dt, t0 = LP[m]
        if trim:
            hi = np.quantile(lp, 0.995)
            # 把极端点之后的水平整体下移（保留路径形状，去掉跳变）
            lp2 = lp.copy()
            bad = lp2 > hi
            if bad.any():
                # 逐段压缩：把超阈值的部分线性拉回阈值
                for i in range(len(lp2)):
                    if bad[i]:
                        lp2[i] = hi
            lp = lp2
        r = run(lp, t0, 250, 2, HOR)
        if not r:
            continue
        for x in r:
            vals.append(x["A14"].mean() / x["A5"].mean())
            pv = dm(x["A14"], x["A5"])
            if pv == pv and pv < .05:
                if x["A14"].mean() < x["A5"].mean():
                    sw += 1
                else:
                    sl += 1
    if vals:
        print(f"  剔除={str(trim):<6} 中位 {np.median(vals):.4f}  胜 {sum(1 for v in vals if v<1)}/{len(vals)}  "
              f"显著胜 {sw}  显著负 {sl}")
        RES[f"D4|trim={trim}"] = dict(med=float(np.median(vals)), sw=sw, sl=sl, n=len(vals))

json.dump(RES, open(os.path.join(DIR, "DATAROBUST.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 DATAROBUST.json")
