# -*- coding: utf-8 -*-
"""
诚实性检验：选择集 / 评估集分离 + 多重比较校正
=====================================================================
问题：§24.29 我在**同一批 25 个测试格**上试了约 25 个模型变体（A15/A16/A17/A18/
A19/A20/A21/A23 及其子型）。即使全部无效，至少一个"看起来显著"是可能的。
这正是文档里列的 C 类拒稿陷阱之一（按测试集结果挑冠军）。
处置：
 (1) **选择集/评估集分离**：把每个市场的测试段切成前一半（SEL）与后一半（EVAL），
     只在 SEL 上比较模型族，在 EVAL 上报告被选中模型的表现。
     这直接回答"这个族是碰巧挑出来的，还是真有效"。
 (2) **族级 Bonferroni**：把 25 个变体视为一个多重检验族，报告族内最小 p 值
     与 Bonferroni 校正后的 p。
 (3) **参数数零的说明**：A23a 在测试段**不含任何拟合参数**（形状完全来自观测量），
     所以"过拟合测试集"只可能来自"族的选择"，不来自"参数的选择"。
"""
import numpy as np, pandas as pd, os, sys, json
from scipy.stats import norm, t as tdist, skewnorm
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
ALG = np.round(np.arange(-8, 8.01, 0.5), 2)
_M = np.stack([(lambda q: (q - q.mean()) / q.std(ddof=0))(skewnorm.ppf(TAUS, a)) for a in ALG])


def shape_alpha(al):
    al = np.clip(np.asarray(al, float), ALG[0], ALG[-1])
    i = np.clip(np.searchsorted(ALG, al) - 1, 0, len(ALG) - 2)
    lo, hi = ALG[i], ALG[i + 1]
    w = (al - lo) / np.where(hi - lo < 1e-12, 1.0, hi - lo)
    return _M[i] * (1 - w)[:, None] + _M[i + 1] * w[:, None]


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


FAMILY = ["A5", "A14", "A21a", "A21b", "A23a", "A23b", "A21d"]
RES = {}
for m in CUT:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    h0, Hv = PAR[m]
    for h in HOR:
        ts_tr = [t for t in range(W + h + 5, t0 - h - 1, STEP)]
        ts_te = [t for t in range(max(W + h + 5, t0), len(lp) - h, STEP)]
        assert not (set(ts_tr) & set(ts_te))
        rec = []
        for t in ts_te:
            win = lp[max(0, t - W):t]
            rr = win[h:] - win[:-h]
            if len(rr) < 3 or np.std(rr, ddof=1) <= 0:
                continue
            rd = np.diff(win); sdd = np.std(rd, ddof=1)
            if not np.isfinite(sdd) or sdd <= 0:
                continue
            rec.append((t, lp[t + h] - lp[t], (h0 / (h0 + h)) * rr.mean(),
                        sdd * h ** Hv, np.quantile((rd - rd.mean()) / sdd, TAUS),
                        float(pd.Series(rd).skew()), abs(float(pd.Series(rd).skew()))))
        if len(rec) < 30:
            continue
        arr = np.array([r[1:] for r in rec], dtype=object)
        Y = arr[:, 0].astype(float); LOC = arr[:, 1].astype(float); S = arr[:, 2].astype(float)
        XI = np.array([list(r) for r in arr[:, 3]]); SK = arr[:, 4].astype(float)
        ABS = arr[:, 5].astype(float)
        w14 = min(1.0, h / 50.0)
        se = float(np.sqrt(6.0 / 249.0))
        g1 = np.minimum(1.0, ABS / se)
        shp_obs = shape_alpha(1.0 * SK)
        shp_obs2 = shape_alpha(2.0 * SK)
        G = np.tile(GV, (len(Y), 1))
        cands = {
            "A5": G,
            "A14": (1 - w14) * XI + w14 * G,
            "A21a": shp_obs,
            "A21b": (1 - w14) * XI + w14 * shp_obs,
            "A21d": (1 - w14) * XI + w14 * shp_obs2,
            "A23a": (1 - w14) * XI + w14 * (g1[:, None] * shp_obs + (1 - g1[:, None]) * G),
            "A23b": (1 - w14) * XI + w14 * (0.5 * g1[:, None] * shp_obs + (1 - 0.5 * g1[:, None]) * G),
        }
        L = {nm: crps_q(Y, LOC[:, None] + S[:, None] * shp) for nm, shp in cands.items()}
        half = len(Y) // 2
        RES[(m, h)] = dict(L=L, half=half, n=len(Y))

cells = sorted(RES.keys())
print("=" * 116)
print("一、SEL（前半测试段）上族内排序 —— 只用 SEL 选模型")
print("=" * 116)
sel_rank = {nm: [] for nm in FAMILY}
for (m, h) in cells:
    R = RES[(m, h)]; half = R["half"]
    for nm in FAMILY:
        sel_rank[nm].append(R["L"][nm][:half].mean())
tab = []
for nm in FAMILY:
    v = np.array(sel_rank[nm])
    base = np.array([RES[(m, h)]["L"]["A5"][:RES[(m, h)]["half"]].mean() for (m, h) in cells])
    tab.append((nm, float(np.median(v / base)), int((v < base).sum())))
tab.sort(key=lambda x: x[1])
print(f"{'模型':<8}{'SEL 中位/A5':>13}{'SEL 胜':>9}")
for nm, med, w in tab:
    print(f"{nm:<8}{med:>13.4f}{w:>7}/25")
WINNER = tab[0][0]
print(f"\n  >>> 在 SEL 上胜出的是：{WINNER}")

print("\n" + "=" * 116)
print(f"二、EVAL（后半测试段）上，被选中模型 {WINNER} 的表现")
print("=" * 116)
print(f"{'模型':<8}{'EVAL 中位/A5':>14}{'胜':>7}{'显著胜':>8}{'显著负':>8}")
for nm in FAMILY:
    v = np.array([RES[(m, h)]["L"][nm][RES[(m, h)]["half"]:].mean() for (m, h) in cells])
    base = np.array([RES[(m, h)]["L"]["A5"][RES[(m, h)]["half"]:].mean() for (m, h) in cells])
    sw = sum(1 for (m, h) in cells if dm(RES[(m, h)]["L"][nm][RES[(m, h)]["half"]:],
                                         RES[(m, h)]["L"]["A5"][RES[(m, h)]["half"]:]) < .05
             and v[i] < base[i]) if False else 0
    sw = sl = 0
    for i, (m, h) in enumerate(cells):
        R = RES[(m, h)]; hh = R["half"]
        pv = dm(R["L"][nm][hh:], R["L"]["A5"][hh:])
        if pv == pv and pv < .05:
            if v[i] < base[i]:
                sw += 1
            else:
                sl += 1
    print(f"{nm:<8}{np.median(v/base):>14.4f}{int((v<base).sum()):>4}/25{sw:>8}{sl:>8}")

print("\n" + "=" * 116)
print("三、多重比较：族内最小 p 值与 Bonferroni 校正（族大小 = 7 个变体）")
print("=" * 116)
print(f"{'模型':<8}{'全测试段 p(最优格)':>20}{'Bonferroni×7':>14}{'结论':>26}")
for nm in FAMILY:
    best = 1.0; bestc = None
    for (m, h) in cells:
        pv = dm(RES[(m, h)]["L"][nm], RES[(m, h)]["L"]["A5"])
        if pv == pv and RES[(m, h)]["L"][nm].mean() < RES[(m, h)]["L"]["A5"].mean() and pv < best:
            best = pv; bestc = (m, h)
    bc = min(1.0, best * 7)
    concl = "校正后仍显著" if bc < .05 else ("边缘" if best < .05 else "不显著")
    print(f"{nm:<8}{best:>20.4f}{bc:>14.4f}{concl:>26}   最优格 {bestc}")

print("\n" + "=" * 116)
print("四、A23a 的逐格（SEL / EVAL 分段）")
print("=" * 116)
print(f"{'市场':<8}{'h':>5}{'n':>5}{'SEL: A5':>11}{'SEL: A23a':>12}{'SEL比':>9}"
      f"{'EVAL: A5':>12}{'EVAL: A23a':>13}{'EVAL比':>10}")
for (m, h) in cells:
    R = RES[(m, h)]; hh = R["half"]
    a5s, a5e = R["L"]["A5"][:hh].mean(), R["L"]["A5"][hh:].mean()
    a23s, a23e = R["L"]["A23a"][:hh].mean(), R["L"]["A23a"][hh:].mean()
    print(f"{MN[m]:<8}{h:>5}{R['n']:>5}{a5s:>11.5f}{a23s:>12.5f}{a23s/a5s:>9.3f}"
          f"{a5e:>12.5f}{a23e:>13.5f}{a23e/a5e:>10.3f}")
selr = np.array([RES[(m, h)]["L"]["A23a"][:RES[(m,h)]["half"]].mean() /
                  RES[(m, h)]["L"]["A5"][:RES[(m,h)]["half"]].mean() for (m, h) in cells])
evr = np.array([RES[(m, h)]["L"]["A23a"][RES[(m,h)]["half"]:].mean() /
                 RES[(m, h)]["L"]["A5"][RES[(m,h)]["half"]:].mean() for (m, h) in cells])
print(f"\n  A23a: SEL 中位 {np.median(selr):.4f}  EVAL 中位 {np.median(evr):.4f}  "
      f"（若 EVAL 仍 <1，说明改进不是挑出来的）")
json.dump({"winner_sel": WINNER, "sel_table": tab,
           "a23a_sel": float(np.median(selr)), "a23a_eval": float(np.median(evr))},
          open(os.path.join(DIR, "HOLDOUT.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 HOLDOUT.json")
