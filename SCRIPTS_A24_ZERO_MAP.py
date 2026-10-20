# -*- coding: utf-8 -*-
"""
A24：零映射的观测偏度形状 + SEL/EVAL 分离检验
=====================================================================
A21/A23 族的失败原因（HOLDOUT.py 给出）：
  SEL 上 0.95~0.96，EVAL 上 1.01~1.02 ⇒ 改进来自前后半段的分布差异，不是模型。
  A14 在 SEL/EVAL 上都恰是 1.0000，因为它用**逐原点重算的经验分位**（纯观测量）。
  而 A21b 把逐原子的偏度**映射**成逐原子的 α，映射关系 κ 是**每市场固定**的
  ——固定的部分就是可被"记住"的部分，于是过拟合。

A24 的修复：把偏度**直接**作为形状参数，中间不加任何每市场固定的映射。
  形状 = 单位方差 skew-normal(α = f(偏度))，其中 f 是**解析关系**：
  skew-normal(α) 的理论偏度 = δ·(1−δ²)/√(1+δ²)，δ=α/√(1+α²)；
  反解：给定目标偏度 γ，解 δ，再 α = δ/√(1−δ²)。**这是唯一的映射，且无自由参数。**
  （若不做反解，直接用 α=γ 是错的：skew-normal 的 α 与偏度不是 1:1。）

同时测：
  A24a  α 由偏度反解（解析、零参数）
  A24b  A24a + 可靠性门控 g=min(1,|skew|/se)
  A24c  A24b 但用**逐原点重算的半窗一致门控**
全部在 SEL/EVAL 分离下检验。
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
ALG = np.round(np.arange(-12, 12.01, 0.25), 3)
_M = np.stack([(lambda q: (q - q.mean()) / q.std(ddof=0))(skewnorm.ppf(TAUS, a)) for a in ALG])


def shape_alpha(al):
    al = np.clip(np.asarray(al, float), ALG[0], ALG[-1])
    i = np.clip(np.searchsorted(ALG, al) - 1, 0, len(ALG) - 2)
    lo, hi = ALG[i], ALG[i + 1]
    w = (al - lo) / np.where(hi - lo < 1e-12, 1.0, hi - lo)
    return _M[i] * (1 - w)[:, None] + _M[i + 1] * w[:, None]


def alpha_from_skew(gamma):
    """把目标偏度 γ 反解为 skew-normal 的 α（解析、零参数）。
    skew-normal(α) 的偏度 = δ(1-δ²)/√(1+δ²)，δ = α/√(1+α²)。
    先解 δ，再 α = δ/√(1-δ²)（δ<1）。"""
    g = np.clip(np.asarray(gamma, float), -0.995, 0.995)
    # 解 δ(1-δ²)/√(1+δ²) = g（二分）
    lo = np.full_like(g, -0.999); hi = np.full_like(g, 0.999)
    for _ in range(80):
        mid = (lo + hi) / 2
        f = mid * (1 - mid ** 2) / np.sqrt(1 + mid ** 2)
        f = np.where(g < 0, -f, f)      # f 随 δ 单增
        lo = np.where(f < g, mid, lo); hi = np.where(f < g, hi, mid)
    d = (lo + hi) / 2
    return d / np.sqrt(np.maximum(1 - d ** 2, 1e-9))


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


# 自检：反解是否正确
_test = np.array([-0.8, -0.4, 0.0, 0.4, 0.8])
_a = alpha_from_skew(_test)
_q = skewnorm.ppf(TAUS, _a[2])
_d = _a[2] / np.sqrt(1 + _a[2] ** 2)
_theo = _d * (1 - _d ** 2) / np.sqrt(1 + _d ** 2)
print(f"自检：目标偏度 {_test.tolist()} -> α {_a.round(3).tolist()}；"
      f"α=0 时的理论偏度 {float(_theo):.6f}（应≈0）")
_sk = []
for a in ALG:
    q = skewnorm.ppf(np.array([0.05, 0.5, 0.95]), a)
    _sk.append((q[2] - q[0]) / (q[1] - q[0]))
print("skew-normal 偏斜参数 α 与 IQR 偏度比的关系（单调性检查）：",
      np.round(_sk[::8], 3).tolist())

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
        Y, LOC, S, XI, SK, ABS = [], [], [], [], [], []
        for t in ts_te:
            win = lp[max(0, t - W):t]
            rr = win[h:] - win[:-h]
            if len(rr) < 3 or np.std(rr, ddof=1) <= 0:
                continue
            rd = np.diff(win); sdd = np.std(rd, ddof=1)
            if not np.isfinite(sdd) or sdd <= 0:
                continue
            Y.append(lp[t + h] - lp[t]); LOC.append((h0 / (h0 + h)) * rr.mean())
            S.append(sdd * h ** Hv); XI.append(np.quantile((rd - rd.mean()) / sdd, TAUS))
            s = float(pd.Series(rd).skew()); SK.append(s); ABS.append(abs(s))
        if len(Y) < 30:
            continue
        Y, LOC, S, XI, SK, ABS = map(np.array, (Y, LOC, S, XI, SK, ABS))
        w14 = min(1.0, h / 50.0)
        se = float(np.sqrt(6.0 / 249.0))
        g1 = np.minimum(1.0, ABS / se)
        al = alpha_from_skew(SK)
        shp_raw = shape_alpha(al)
        shp_gat = g1[:, None] * shp_raw + (1 - g1[:, None]) * np.tile(GV, (len(Y), 1))
        G = np.tile(GV, (len(Y), 1))
        cands = {
            "A5": G,
            "A14": (1 - w14) * XI + w14 * G,
            "A24a": (1 - w14) * XI + w14 * shp_raw,
            "A24b": (1 - w14) * XI + w14 * shp_gat,
            "A24c": shp_gat,
        }
        L = {nm: crps_q(Y, LOC[:, None] + S[:, None] * shp) for nm, shp in cands.items()}
        RES[(m, h)] = dict(L=L, n=len(Y), half=len(Y) // 2)

cells = sorted(RES.keys())
FAM = ["A5", "A14", "A24a", "A24b", "A24c"]

def seg(mh, nm, which):
    R = RES[mh]; hh = R["half"]
    v = R["L"][nm][:hh] if which == "SEL" else R["L"][nm][hh:]
    return v

for which in ("SEL", "EVAL"):
    print("\n" + "=" * 116)
    print(f"{which} 段（{'前半' if which=='SEL' else '后半'}测试段）")
    print("=" * 116)
    print(f"{'模型':<8}{'中位/A5':>10}{'胜':>7}{'显著胜':>8}{'显著负':>8}")
    for nm in FAM:
        v = np.array([seg(mh, nm, which).mean() for mh in cells])
        b = np.array([seg(mh, "A5", which).mean() for mh in cells])
        sw = sl = 0
        for mh in cells:
            pv = dm(seg(mh, nm, which), seg(mh, "A5", which))
            if pv == pv and pv < .05:
                if seg(mh, nm, which).mean() < seg(mh, "A5", which).mean():
                    sw += 1
                else:
                    sl += 1
        print(f"{nm:<8}{np.median(v/b):>10.4f}{int((v<b).sum()):>4}/25{sw:>8}{sl:>8}")

print("\n" + "=" * 116)
print("逐格 A24b（SEL 比 / EVAL 比），>1 表示比 A5 差")
print("=" * 116)
print(f"{'市场':<8}{'h':>5}{'SEL比':>9}{'EVAL比':>9}")
for mh in cells:
    R = RES[mh]; hh = R["half"]
    s = R["L"]["A24b"][:hh].mean() / R["L"]["A5"][:hh].mean()
    e = R["L"]["A24b"][hh:].mean() / R["L"]["A5"][hh:].mean()
    print(f"{MN[mh[0]]:<8}{mh[1]:>5}{s:>9.3f}{e:>9.3f}")

json.dump({"cells": {f"{m}|{h}": {k: float(RES[(m, h)]["L"][k].mean()) for k in RES[(m, h)]["L"]}
                     for (m, h) in cells},
           "sel_eval": {nm: {
               "SEL": float(np.median([seg(mh, nm, "SEL").mean() / seg(mh, "A5", "SEL").mean() for mh in cells])),
               "EVAL": float(np.median([seg(mh, nm, "EVAL").mean() / seg(mh, "A5", "EVAL").mean() for mh in cells]))}
               for nm in FAM}},
          open(os.path.join(DIR, "A24.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 A24.json")
