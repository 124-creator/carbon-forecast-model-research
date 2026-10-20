# -*- coding: utf-8 -*-
"""
A18：偏度随期限线性漂移的形状模型（把 A14 的非参数补丁升级为有机制的两参数模型）
=====================================================================
A17 的结构信号：偏度参数 b 的训练段最优值**随期限单调下降**
    CEA  -0.2 -0.2 -0.4 -0.4 -0.8
    上海  0.0 -0.4 -0.8 -0.8 -0.8
    广州  0.0  0.0 -0.2 -0.4 -0.8
    EU   -0.2 -0.2 -0.4  0.4 -0.8
尾参数 a 的最优值则是 0.9 → 0.3~0.6 随期限下降（h=5 厚尾、长端薄尾）。
⇒ 形状应当是 (h) 的函数，不是常数。

A18 规格（每市场 2 参数，与 A5 同量级）：
    Q = λ(h)·μ̂ + σ̂_d·h^H · SA(τ; a(h), b(h))
    b(h) = b0 + b1·(log h − log 5)          偏度的对数期限漂移
    a(h) = a0                            尾参数（常数，备选 a(h)=a0+a1·log h 另测）
形状用 Jones-Pewsey sinh-arcsinh（单位方差）。

检验：
  A18a  (a0, b0, b1) 每市场 3 参数
  A18b  (a0, b0, b1) 每 (市场,期限) 拟合 = 上界
  A18c  用 b(h) 线性漂移 + **A14 的 w(h) 经验形状混合**（检验两者是否互补）
关键判据：**必须优于 A14**（A14 是当前最优），否则这一族没有价值。
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
AGRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.9]
B0G = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2]
B1G = [-0.3, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1]
LH5 = np.log(5.0)


def sa_std(a, b):
    if a < 1e-9:
        return 1.0
    e2 = np.exp(2 * a * a)
    var = 0.5 * (np.cosh(2 * a * b) * e2 - 1.0) - np.sinh(a * b) ** 2 * np.exp(a * a)
    return float(np.sqrt(max(var, 1e-12)) / a)


def sa_shape(a, b):
    u = GV
    if a < 1e-9:
        x = u + b
        return (x - x.mean()) / x.std(ddof=0)
    x = (np.sinh(a * (u + b)) - np.sinh(a * b)) / a
    sd = sa_std(a, b)
    x = x / sd
    return x - x.mean()


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


# 形状缓存
SH = {}
for a in AGRID:
    for b in np.unique(B0G + B1G):
        SH[(a, round(float(b), 4))] = sa_shape(a, b)


def shape_of(a, b):
    k = (a, round(float(b), 4))
    if k not in SH:
        SH[k] = sa_shape(a, b)
    return SH[k]


DATA = {}
for m in CUT:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    h0, Hv = PAR[m]
    for h in HOR:
        def pack(ts):
            Y, LOC, S, XI = [], [], [], []
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
            return np.array(Y), np.array(LOC), np.array(S), np.array(XI)
        tr = pack([t for t in range(W + h + 5, t0 - h - 1, STEP)])
        te = pack([t for t in range(max(W + h + 5, t0), len(lp) - h, STEP)])
        if len(tr[0]) < 30 or len(te[0]) < 25:
            continue
        DATA[(m, h)] = dict(tr=tr, te=te, lh=np.log(h) - LH5)

cells = sorted(DATA.keys())
print(f"有效格 {len(cells)}")

# ---------- 训练段打分 ----------
print("\n" + "=" * 110)
print("训练段：A18a 的 (a0,b0,b1) 搜索（每市场池化，权重=每格样本数）")
print("=" * 110)
BEST = {}
for m in CUT:
    ks = [k for k in cells if k[0] == m]
    if not ks:
        continue
    tot = {}
    for a in AGRID:
        for b0 in B0G:
            for b1 in B1G:
                s = 0.0; nn = 0
                for (mm, h) in ks:
                    dd = DATA[(mm, h)]
                    Y, L, S, _ = dd["tr"]
                    b = b0 + b1 * dd["lh"]
                    sh = shape_of(a, b)
                    s += float(crps_q(Y, L[:, None] + S[:, None] * np.tile(sh, (len(Y), 1))).sum())
                    nn += len(Y)
                tot[(a, b0, b1)] = s / nn
    ka = min(tot, key=tot.get)
    BEST[m] = ka
    per_h = []
    for (mm, h) in ks:
        dd = DATA[(mm, h)]
        Y, L, S, _ = dd["tr"]
        b = ka[1] + ka[2] * dd["lh"]
        v = float(crps_q(Y, L[:, None] + S[:, None] * np.tile(shape_of(ka[0], b), (len(Y), 1))).mean())
        per_h.append((h, round(b, 3), round(v, 5)))
    print(f"  {MN[m]:<8} (a0,b0,b1)=({ka[0]:.1f},{ka[1]:+.1f},{ka[2]:+.2f})  "
          f"逐格 b: {[(h, b) for h, b, _ in per_h]}")

# ---------- 测试段评估 ----------
RES = {}
for m in CUT:
    ka = BEST.get(m)
    if ka is None:
        continue
    a0, b0, b1 = ka
    for (mm, h) in [k for k in cells if k[0] == m]:
        dd = DATA[(mm, h)]
        Y, L, S, XI = dd["te"]
        b = b0 + b1 * dd["lh"]
        sh = shape_of(a0, b)
        L18 = crps_q(Y, L[:, None] + S[:, None] * np.tile(sh, (len(Y), 1)))
        L5 = crps_q(Y, L[:, None] + S[:, None] * np.tile(GV, (len(Y), 1)))
        w14 = min(1.0, h / 50.0)
        sh14 = (1 - w14) * XI + w14 * GV[None, :]
        L14 = crps_q(Y, L[:, None] + S[:, None] * sh14)
        # A18c：SA 形状 + 经验形状混合（两者互补？）
        shc = (1 - w14) * XI + w14 * np.tile(sh, (len(Y), 1))
        L18c = crps_q(Y, L[:, None] + S[:, None] * shc)
        row = dict(n=len(Y), a0=a0, b0=b0, b1=b1, b=float(b),
                   A5=float(L5.mean()), A14=float(L14.mean()),
                   A18=float(L18.mean()), A18c=float(L18c.mean()),
                   p_A14=dm(L14, L5), p_A18=dm(L18, L5), p_A18c=dm(L18c, L5),
                   p_A18_vs_A14=dm(L18, L14), p_A18c_vs_A14=dm(L18c, L14))
        for nm, L, shp in (("A5", L5, np.tile(GV, (len(Y), 1))),
                           ("A14", L14, sh14),
                           ("A18", L18, np.tile(sh, (len(Y), 1))),
                           ("A18c", L18c, shc)):
            Q = L[:, None] + S[:, None] * shp
            row["c95_" + nm] = float(np.mean((Y >= Q[:, 0]) & (Y <= Q[:, 14])))
            row["c80_" + nm] = float(np.mean((Y >= Q[:, 2]) & (Y <= Q[:, 12])))
        RES[f"{m}|{h}"] = row
        print(f"  {MN[m]:<8} h={h:<4} A5 {row['A5']:.5f} | A14 {row['A14']:.5f} "
              f"({row['A14']/row['A5']:.3f},p={row['p_A14']:.3f}) | A18 {row['A18']:.5f} "
              f"({row['A18']/row['A5']:.3f},p={row['p_A18']:.3f}, vsA14 p={row['p_A18_vs_A14']:.3f}) | "
              f"A18c {row['A18c']:.5f} ({row['A18c']/row['A5']:.3f},p={row['p_A18c']:.3f},"
              f" vsA14 p={row['p_A18c_vs_A14']:.3f})")

json.dump({"RES": RES, "BEST": {k: list(v) for k, v in BEST.items()}},
          open(os.path.join(DIR, "A18.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)

cl = [(m, h) for m in CUT for h in HOR if f"{m}|{h}" in RES]
print("\n" + "=" * 110)
print("汇总（相对 A5 / 相对 A14）")
print("=" * 110)
print(f"{'模型':<8}{'/A5 中位':>10}{'胜':>7}{'显著胜':>8}{'显著负':>8}"
      f"{'/A14 中位':>11}{'胜':>7}{'显著胜':>8}{'显著负':>8}{'逐期限/A5 5/20/40/60/120':>36}")
for nm, pk, pav in (("A14", "p_A14", "A5"), ("A18", "p_A18", "A5"), ("A18c", "p_A18c", "A5")):
    rr = np.array([RES[f"{m}|{h}"][nm] / RES[f"{m}|{h}"]["A5"] for m, h in cl])
    sw = sum(1 for m, h in cl if RES[f"{m}|{h}"][pk] < .05 and RES[f"{m}|{h}"][nm] < RES[f"{m}|{h}"]["A5"])
    sl = sum(1 for m, h in cl if RES[f"{m}|{h}"][pk] < .05 and RES[f"{m}|{h}"][nm] > RES[f"{m}|{h}"]["A5"])
    r14 = np.array([RES[f"{m}|{h}"][nm] / RES[f"{m}|{h}"]["A14"] for m, h in cl])
    s14 = sum(1 for m, h in cl if RES[f"{m}|{h}"][pav.replace("A5", "A18_vs_A14") if pav == "A5" else ""] < .05) if False else 0
    s14 = sum(1 for m, h in cl if RES[f"{m}|{h}"]["p_" + nm + "_vs_A14"] < .05
              and RES[f"{m}|{h}"][nm] < RES[f"{m}|{h}"]["A14"])
    l14 = sum(1 for m, h in cl if RES[f"{m}|{h}"]["p_" + nm + "_vs_A14"] < .05
              and RES[f"{m}|{h}"][nm] > RES[f"{m}|{h}"]["A14"])
    per = " / ".join(f"{np.median([RES[f'{m}|{h}'][nm]/RES[f'{m}|{h}']['A5'] for m,_ in cl if h==h]):.3f}"
                     for h in HOR)
    print(f"{nm:<8}{np.median(rr):>10.4f}{int((rr<1).sum()):>4}/25{sw:>8}{sl:>8}"
          f"{np.median(r14):>11.4f}{int((r14<1).sum()):>4}/25{s14:>8}{l14:>8}{per:>36}")

print("\n覆盖率 95%（目标 0.95）：")
for h in HOR:
    ks = [(m, h) for m, h in cl if h == h]
    print(f"  h={h:<5} A5 {np.mean([RES[f'{m}|{h}']['c95_A5'] for m,_ in ks]):.3f}  "
          f"A14 {np.mean([RES[f'{m}|{h}']['c95_A14'] for m,_ in ks]):.3f}  "
          f"A18 {np.mean([RES[f'{m}|{h}']['c95_A18'] for m,_ in ks]):.3f}  "
          f"A18c {np.mean([RES[f'{m}|{h}']['c95_A18c'] for m,_ in ks]):.3f}")
print("\n已写出 A18.json")
