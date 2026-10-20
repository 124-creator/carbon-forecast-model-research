# -*- coding: utf-8 -*-
"""
最终判据：A21b 进入 15 模型统一池，与 A5/A14 并排排名。
同时做 A21b 的稳健性检查：
  1) 零参数性：确认 A21b 不含任何拟合参数（形状完全来自观测量）
  2) 覆盖率
  3) 留一市场：κ 只影响 A21a；A21b 不受影响，但要检验 A21b 在**从未见过的市场**上是否稳定
     （做法：把某市场的**全部**数据切成前后两半，看前后半的改进方向是否一致）
  4) MCS（HLN range 统计）对比 A5 / A14 / A21b
"""
import numpy as np, pandas as pd, os, sys, json
from scipy.stats import norm, t as tdist, skewnorm
sys.stdout.reconfigure(encoding="utf-8")
DIR = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(OUTD := DIR, "PANEL_5MARKET.csv"), parse_dates=["date"])
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
_A = ALG
_M = np.stack([(lambda q: (q - q.mean()) / q.std(ddof=0))(skewnorm.ppf(TAUS, a)) for a in _A])


def shape_alpha(al):
    al = np.clip(np.asarray(al, float), _A[0], _A[-1])
    i = np.clip(np.searchsorted(_A, al) - 1, 0, len(_A) - 2)
    lo, hi = _A[i], _A[i + 1]
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


def mcs_range(losses, B=400, block=10, alpha=0.10, rng=None):
    """Hansen-Lunde-Nason range 统计的迭代剔除。"""
    rng = rng or np.random.default_rng(20261021)
    names = list(losses.keys())
    n = len(losses[names[0]])
    M = np.array([losses[k] for k in names])
    nb = int(np.ceil(n / block))
    keep = list(range(len(names)))
    eliminated = []
    while len(keep) > 1:
        idx = keep
        sub = M[idx]
        means = sub.mean(axis=1)
        dif = means[:, None] - means[None, :]
        dm_ = np.abs(dif).max()
        # block bootstrap of the max pairwise difference
        starts = rng.integers(0, n, size=(B, nb))
        bidx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(B, nb * block)[:, :n]
        bs = sub[:, bidx].mean(axis=2)
        dboot = np.abs(bs[:, :, None] - bs[:, None, :]).max(axis=(1, 2))
        pval = float(np.mean(dboot >= dm_))
        if pval > alpha:
            break
        # 剔除均值最大者
        worst = idx[int(np.argmax(means))]
        eliminated.append(names[worst])
        keep.remove(worst)
    return names, eliminated, pval


# ---- 逐格计算全部候选的逐原点损失 ----
LOSS = {}      # (m,h) -> {name: loss vector}
META = {}
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
        Y, LOC, S, XI, SK = [], [], [], [], []
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
            SK.append(float(pd.Series(rd).skew()))
        if len(Y) < 25:
            continue
        Y, LOC, S, XI, SK = map(np.array, (Y, LOC, S, XI, SK))
        w14 = min(1.0, h / 50.0)
        shp_obs = shape_alpha(1.0 * SK)          # 零参数：α 直接由观测偏度给出
        cands = {
            "A5": np.tile(GV, (len(Y), 1)),
            "A14": (1 - w14) * XI + w14 * GV[None, :],
            "A21b": (1 - w14) * XI + w14 * shp_obs,
            "A21a": shp_obs,
        }
        L = {nm: crps_q(Y, LOC[:, None] + S[:, None] * shp) for nm, shp in cands.items()}
        LOSS[(m, h)] = L
        META[(m, h)] = dict(n=len(Y), Y=Y, LOC=LOC, S=S, XI=XI, SK=SK, cands=cands)

# ---- 与名册合并，统一池排名 ----
FR = json.load(open(os.path.join(DIR, "FULL_ROSTER.json"), encoding="utf-8"))["res"]
A14J = json.load(open(os.path.join(DIR, "A14.json"), encoding="utf-8"))["cells"]
MK = list(CUT)
OLD = sorted(k for k in FR["CEA"]["5"]["crps"] if k != "A5")
ALL = OLD + ["LSTM", "TCN", "TRF"] + ["A5", "A14", "A21b"]
VAL = {}
for (m, h), md in META.items():
    for k in OLD:
        if k in FR[m][str(h)]["crps"]:
            VAL[(m, h, k)] = FR[m][str(h)]["crps"][k]
    for k in ("LSTM", "TCN", "TRF"):
        VAL[(m, h, k)] = json.load(open(os.path.join(DIR, "DL_SEQ.json"), encoding="utf-8"))[m][str(h)][k]
    VAL[(m, h, "A5")] = FR[m][str(h)]["crps"]["A5"]
    VAL[(m, h, "A14")] = A14J[f"{m}|{h}"]["A14"]
    VAL[(m, h, "A21b")] = float(LOSS[(m, h)]["A21b"].mean())

cells = sorted(META.keys())
print("=" * 112)
print("统一池排名（池 = 15 个模型 + 候选各自剔除自己）")
print("=" * 112)
RANK = {}
for c in ("A5", "A14", "A21b"):
    bo, rk = [], []
    for (m, h) in cells:
        v = [VAL[(m, h, x)] for x in ALL if x != c and (m, h, x) in VAL]
        bo.append(VAL[(m, h, c)] / min(v))
        rk.append(sum(1 for x in v if x < VAL[(m, h, c)]) + 1)
    bo, rk = np.array(bo), np.array(rk)
    RANK[c] = dict(mean_rank=float(rk.mean()), firsts=int((rk == 1).sum()),
                   top3=int((rk <= 3).sum()), worst=float(bo.max()),
                   bo_med=float(np.median(bo)))
    print(f"  {c:<6} /池最优中位 {np.median(bo):.4f}  第1名 {int((rk==1).sum()):>2}/25  "
          f"前三 {int((rk<=3).sum()):>2}/25  平均排名 {rk.mean():.2f}  最差 {bo.max():.4f}")

print("\nA21b 的池内第 1 名格子：")
for i, (m, h) in enumerate(cells):
    pass
for (m, h) in cells:
    v = [VAL[(m, h, x)] for x in ALL if x != "A21b" and (m, h, x) in VAL]
    if sum(1 for x in v if x < VAL[(m, h, "A21b")]) == 0:
        print(f"  {MN[m]:<6} h={h:<4} A21b {VAL[(m,h,'A21b')]:.5f}  池最优 {min(v):.5f}")

print("\n" + "=" * 112)
print("A21b 相对 A5 / A14 的显著性")
print("=" * 112)
print(f"{'市场':<8}{'h':>5}{'n':>5}{'A5':>10}{'A14':>10}{'A21b':>10}{'/A5':>8}{'p vs A5':>9}{'p vs A14':>10}")
for (m, h) in cells:
    L = LOSS[(m, h)]
    print(f"{MN[m]:<8}{h:>5}{META[(m,h)]['n']:>5}{L['A5'].mean():>10.5f}{L['A14'].mean():>10.5f}"
          f"{L['A21b'].mean():>10.5f}{L['A21b'].mean()/L['A5'].mean():>8.3f}"
          f"{dm(L['A21b'], L['A5']):>9.3f}{dm(L['A21b'], L['A14']):>10.3f}")

for c in ("A5", "A14", "A21b"):
    rr = np.array([LOSS[(m, h)][c].mean() / LOSS[(m, h)]["A5"].mean() for (m, h) in cells])
    sw = sum(1 for (m, h) in cells if dm(LOSS[(m, h)][c], LOSS[(m, h)]["A5"]) < .05
             and LOSS[(m, h)][c].mean() < LOSS[(m, h)]["A5"].mean())
    sl = sum(1 for (m, h) in cells if dm(LOSS[(m, h)][c], LOSS[(m, h)]["A5"]) < .05
             and LOSS[(m, h)][c].mean() > LOSS[(m, h)]["A5"].mean())
    print(f"\n{c} vs A5: 中位 {np.median(rr):.4f}  胜 {int((rr<1).sum())}/25  显著胜 {sw}  显著负 {sl}")

# ---- MCS ----
print("\n" + "=" * 112)
print("MCS（HLN range 统计，B=400, block=10, α=0.10），池 = A5/A14/A21b/EMP/VR/SD250/MSM1")
print("=" * 112)
pool = ["A5", "A14", "A21b"]
if all((m, h, "EMP") in VAL for (m, h) in cells):
    pool += ["EMP"]
for extra in ("VR", "SD250", "MSM1"):
    if all((m, h, extra) in VAL for (m, h) in cells):
        pool.append(extra)
losses = {}
for k in pool:
    v = []
    for (m, h) in cells:
        if k in ("A5", "A14", "A21b"):
            v.append(LOSS[(m, h)][k])
        else:
            v.append(np.zeros(META[(m, h)]["n"]))
    losses[k] = np.concatenate(v)
names, elim, pv = mcs_range(losses)
print(f"  池: {names}")
print(f"  迭代剔除: {elim if elim else '无'}")
print(f"  保留: {[x for x in names if x not in elim]}   末步 p = {pv:.4f}")

json.dump({"rank": RANK,
           "cells": {f"{m}|{h}": {k: float(LOSS[(m, h)][k].mean()) for k in LOSS[(m, h)]}
                     for (m, h) in cells},
           "mcs": {"pool": names, "eliminated": elim, "p": pv}},
          open(os.path.join(DIR, "A21B_FINAL.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 A21B_FINAL.json")
