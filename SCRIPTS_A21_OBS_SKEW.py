# -*- coding: utf-8 -*-
"""
A21：零参数的观测驱动形状 —— 用**窗内日收益的观测偏度**直接构造预测形状
=====================================================================
逻辑链（全部来自已验证的证据，不是猜测）：
 1) §24.29：实现残差偏度在长端达 −1.19~−0.10（**全负**），高斯形状丢掉了它。
 2) §24.29：偏度参数**不可跨样本拟合**（训练段偏好右偏 α=+2，与实现负偏度矛盾）。
 3) 但**窗内日收益的偏度是可观测的、无需估计的**（每个原点都能算）。
 4) A14 的 w(h)=min(1,h/50) 用经验分位形状（自动含偏度）混合高斯，
    有效但权重来自 oracle 常数 50。
 ⇒ 若偏度是真实且可观测的，可以**直接**用观测偏度构造形状，
   权重也由观测偏度的**可靠性**（随窗长/样本量）决定 ⇒ **零拟合参数**。

候选：
  A21a  α_t = κ·skew(窗内日收益) 的映射，κ 由训练段估（1 参数/市场），
        形状 = 单位方差 skew-normal(α_t)
  A21b  A21a + 与高斯按 w(h) 混合（w 仍是 min(1,h/50)，与 A14 同）
  A21c  **纯经验分位形状**（A6 的极端版）作对照
  A21d  A14（w=min(1,h/50)，常数 50）作对照
判据：必须显著优于 A5，且**不劣于 A14**；参数越少越好。
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
# skew-normal 分位缓存（单位方差）
SHC = {}
for a in ALG:
    q = skewnorm.ppf(TAUS, a)
    SHC[a] = (q - q.mean()) / q.std(ddof=0)
_A = np.array(ALG)
_M = np.stack([SHC[a] for a in ALG])          # (n_alpha, 15)


def shape_alpha(alphas):
    """按 alpha 线性插值出单位方差 skew-normal 形状（逐行）。"""
    al = np.clip(np.asarray(alphas, float), _A[0], _A[-1])
    idx = np.clip(np.searchsorted(_A, al) - 1, 0, len(_A) - 2)
    lo = _A[idx]; hi = _A[idx + 1]
    wgt = (al - lo) / np.where(hi - lo < 1e-12, 1.0, hi - lo)
    return _M[idx] * (1 - wgt)[:, None] + _M[idx + 1] * wgt[:, None]


print("skewnorm 缓存", len(SHC), "；α=0 自检 sd =", round(float(SHC[0.0].std(ddof=0)), 8))


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


DATA = {}
for m in CUT:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    h0, Hv = PAR[m]
    for h in HOR:
        def pack(ts):
            Y, LOC, S, XI, SK = [], [], [], [], []
            for t in ts:
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
            return (np.array(Y), np.array(LOC), np.array(S), np.array(XI), np.array(SK))
        tr = pack([t for t in range(W + h + 5, t0 - h - 1, STEP)])
        te = pack([t for t in range(max(W + h + 5, t0), len(lp) - h, STEP)])
        if len(tr[0]) < 30 or len(te[0]) < 25:
            continue
        DATA[(m, h)] = dict(tr=tr, te=te)
cells = sorted(DATA.keys())

# 训练段：估 κ（skew -> alpha 的映射系数），每市场 1 个参数
KGRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
KAP = {}
print("\n训练段估 κ：")
for m in CUT:
    ks = [k for k in cells if k[0] == m]
    if not ks:
        continue
    tot = {}
    for kap in KGRID:
        s = 0.0; nn = 0
        for (mm, h) in ks:
            Y, L, S, XI, SK = DATA[(mm, h)]["tr"]
            al = np.clip(kap * SK, ALG[0], ALG[-1])
            shp = shape_alpha(al)
            s += float(crps_q(Y, L[:, None] + S[:, None] * shp).sum()); nn += len(Y)
        tot[kap] = s / nn
    KAP[m] = min(tot, key=tot.get)
    print(f"  {MN[m]:<8} κ*={KAP[m]}   （窗内日收益偏度中位 "
          f"{np.median([np.median(DATA[(mm,h)]['tr'][4]) for mm,h in ks]):+.3f}）")

RES = {}
for (m, h) in cells:
    Y, L, S, XI, SK = DATA[(m, h)]["te"]
    kap = KAP[m]
    al = np.clip(kap * SK, ALG[0], ALG[-1])
    shp_obs = shape_alpha(al)
    w14 = min(1.0, h / 50.0)
    cand = {
        "A5": np.tile(GV, (len(Y), 1)),
        "A14": (1 - w14) * XI + w14 * GV[None, :],
        "A21a": shp_obs,
        "A21b": (1 - w14) * XI + w14 * shp_obs,
        "A21c": XI,
    }
    Ls = {nm: crps_q(Y, L[:, None] + S[:, None] * shp) for nm, shp in cand.items()}
    row = dict(n=len(Y), kappa=kap, alpha_med=float(np.median(al)))
    for nm in cand:
        row[nm] = float(Ls[nm].mean())
    row["p_A14"] = dm(Ls["A14"], Ls["A5"])
    for nm in ("A21a", "A21b", "A21c"):
        row["p_" + nm] = dm(Ls[nm], Ls["A5"])
        row["p_" + nm + "_vs_A14"] = dm(Ls[nm], Ls["A14"])
    for nm, shp in cand.items():
        Q = L[:, None] + S[:, None] * shp
        row["c95_" + nm] = float(np.mean((Y >= Q[:, 0]) & (Y <= Q[:, 14])))
        row["c80_" + nm] = float(np.mean((Y >= Q[:, 2]) & (Y <= Q[:, 12])))
    RES[f"{m}|{h}"] = row

json.dump({"RES": RES, "KAP": KAP}, open(os.path.join(DIR, "A21.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)

print("\n" + "=" * 118)
print("汇总（相对 A5 / 相对 A14）")
print("=" * 118)
print(f"{'模型':<8}{'参数/市场':>9}{'/A5中位':>10}{'胜':>7}{'显著胜':>8}{'显著负':>8}"
      f"{'/A14中位':>11}{'胜':>7}{'显著胜':>8}{'显著负':>8}{'逐期限/A5':>32}")
for nm, npar in (("A14", 0), ("A21a", 1), ("A21b", 0), ("A21c", 0)):
    rr = np.array([RES[f"{m}|{h}"][nm] / RES[f"{m}|{h}"]["A5"] for m, h in cells])
    sw = sum(1 for m, h in cells if RES[f"{m}|{h}"].get("p_" + nm, 1) < .05 and RES[f"{m}|{h}"][nm] < RES[f"{m}|{h}"]["A5"])
    sl = sum(1 for m, h in cells if RES[f"{m}|{h}"].get("p_" + nm, 1) < .05 and RES[f"{m}|{h}"][nm] > RES[f"{m}|{h}"]["A5"])
    r14 = np.array([RES[f"{m}|{h}"][nm] / RES[f"{m}|{h}"]["A14"] for m, h in cells])
    s14 = sum(1 for m, h in cells if RES[f"{m}|{h}"].get("p_" + nm + "_vs_A14", 1) < .05 and RES[f"{m}|{h}"][nm] < RES[f"{m}|{h}"]["A14"])
    l14 = sum(1 for m, h in cells if RES[f"{m}|{h}"].get("p_" + nm + "_vs_A14", 1) < .05 and RES[f"{m}|{h}"][nm] > RES[f"{m}|{h}"]["A14"])
    per = " / ".join(f"{np.median([RES[f'{m}|{h}'][nm]/RES[f'{m}|{h}']['A5'] for m,_ in cells if h==h]):.3f}"
                     for h in HOR)
    print(f"{nm:<8}{npar:>9}{np.median(rr):>10.4f}{int((rr<1).sum()):>4}/25{sw:>8}{sl:>8}"
          f"{np.median(r14):>11.4f}{int((r14<1).sum()):>4}/25{s14:>8}{l14:>8}{per:>32}")

print("\n逐格：")
print(f"{'市场':<8}{'h':>5}{'A5':>10}{'A14':>10}{'A21a':>10}{'A21b':>10}{'A21c':>10}")
for m, h in cells:
    r = RES[f"{m}|{h}"]
    print(f"{MN[m]:<8}{h:>5}{r['A5']:>10.5f}{r['A14']:>10.5f}{r['A21a']:>10.5f}"
          f"{r['A21b']:>10.5f}{r['A21c']:>10.5f}")
print("\n已写出 A21.json")
