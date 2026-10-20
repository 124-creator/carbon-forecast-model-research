# -*- coding: utf-8 -*-
"""
A17 族：参数化的偏斜-厚尾形状（形状块的新形式）
=====================================================================
诊断（SHAPE_DIAG）：实现残差 z 的形状随期限系统变化
  偏度   : 全期限为负（广州 -1.19~-1.46，上海 -0.10~-1.09，湖北 -0.05~-0.76）
  超额峰度: h=5 时 6.2~10.0（厚尾），h=120 时 -0.3~-1.0（**比高斯还薄**）
A5 用单个高斯 G(τ) 同时承担偏斜与尾，只能折中；A14 的 w(h) 混合是**非参数**补丁。
A17 换用**参数化**的形状族：sinh-arcsinh（Jones & Pewsey 2009）变换的正态：
    X = [sinh(a(Z+b)) - sinh(ab)] / (a·sd(a,b)),  Z ~ N(0,1)
  a = 尾权重（a 大 => 厚尾；a→0 => 正态）
  b = 偏度（b<0 => 左偏，与实现一致）
  单位方差，故标度块 σ̂_d·h^H 保持可比。

候选：
  A17a  (a,b) 每市场一个（跨五期限共享）—— 2 参数/市场，与 A5 同量级
  A17b  (a,b) 每 (市场,期限) 一个 —— 10 参数/市场，上界
  A18   A14 的短端经验形状混合 + A17 的长端参数形状
在**训练段**按 CRPS 网格搜参，测试段评估。全部对 A5 与 A14 做 DM。
"""
import numpy as np, pandas as pd, os, sys, json
from scipy.stats import norm, t as tdist
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
AGRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.9]
BGRID = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4]


def sa_shrink(a, b):
    """X 的标准差因子（闭式）。a=0 时退化为 1。"""
    if a < 1e-9:
        return 1.0
    e2 = np.exp(2 * a * a)
    var = 0.5 * (np.cosh(2 * a * b) * e2 - 1.0) - np.sinh(a * b) ** 2 * np.exp(a * a)
    return float(np.sqrt(max(var, 1e-12)) / a)


def sa_q(a, b):
    """单位方差的 SA 形状分位函数（15 维）。"""
    u = GV
    if a < 1e-9:
        x = u + b
        return (x - x.mean()) / x.std(ddof=0)
    sh = np.sinh(a * (u + b))
    x = (sh - np.sinh(a * b)) / a
    sd = sa_shrink(a, b)
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


# 预生成形状网格
SHAPES = {(a, b): sa_q(a, b) for a in AGRID for b in BGRID}

RES = {}
for m in CUT:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    h0, Hv = PAR[m]
    print(f"\n=== {m} ===")
    for h in HOR:
        ts_tr = [t for t in range(W + h + 5, t0 - h - 1, STEP)]
        ts_te = [t for t in range(max(W + h + 5, t0), len(lp) - h, STEP)]
        assert not (set(ts_tr) & set(ts_te))

        def pack(ts):
            Y, LOC, S, XI, RD = [], [], [], [], []
            for t in ts:
                win = lp[max(0, t - W):t]
                rr = win[h:] - win[:-h]
                if len(rr) < 3 or np.std(rr, ddof=1) <= 0:
                    continue
                rd = np.diff(win); sdd = np.std(rd, ddof=1)
                if not np.isfinite(sdd) or sdd <= 0:
                    continue
                Y.append(lp[t + h] - lp[t])
                LOC.append((h0 / (h0 + h)) * rr.mean())
                S.append(sdd * h ** Hv)
                XI.append(np.quantile((rd - rd.mean()) / sdd, TAUS))
                RD.append(rd)
            return (np.array(Y), np.array(LOC), np.array(S),
                    np.array(XI), RD)

        Ytr, Ltr, Str, Xtr, RDtr = pack(ts_tr)
        Yte, Lte, Ste, Xte, RDte = pack(ts_te)
        if len(Ytr) < 30 or len(Yte) < 25:
            continue
        w14 = min(1.0, h / 50.0)
        SH14 = (1 - w14) * Xte + w14 * GV[None, :]
        SH5 = np.tile(GV, (len(Yte), 1))

        def crps_with(shape):
            return crps_q(Yte, Lte[:, None] + Ste[:, None] * shape)

        # 训练段：每 (a,b) 的 CRPS（用于选参）
        def crps_tr(shape):
            return crps_q(Ytr, Ltr[:, None] + Str[:, None] * shape).mean()

        tr_scores = {}
        for key, sh in SHAPES.items():
            tr_scores[key] = crps_tr(np.tile(sh, (len(Ytr), 1)))
        # A17b：逐格最优
        kb = min(tr_scores, key=tr_scores.get)
        # A17a：每市场跨期限最优（后面统一处理，这里先存逐格分数）
        row = dict(n=len(Yte), n_tr=len(Ytr), kb=list(kb), tr_score_b=tr_scores[kb],
                   tr_scores={f"{k[0]}|{k[1]}": v for k, v in tr_scores.items()})

        L5 = crps_q(Yte, Lte[:, None] + Ste[:, None] * SH5)
        L14 = crps_q(Yte, Lte[:, None] + Ste[:, None] * SH14)
        L17b = crps_q(Yte, Lte[:, None] + Ste[:, None] * np.tile(SHAPES[kb], (len(Yte), 1)))
        row["A5"] = float(L5.mean()); row["A14"] = float(L14.mean()); row["A17b"] = float(L17b.mean())
        row["p_A14_vs_A5"] = dm(L14, L5)
        row["p_A17b_vs_A5"] = dm(L17b, L5)
        row["p_A17b_vs_A14"] = dm(L17b, L14)
        # 覆盖
        for nm, L, shape in (("A5", L5, SH5), ("A14", L14, SH14),
                             ("A17b", L17b, np.tile(SHAPES[kb], (len(Yte), 1)))):
            Q = Lte[:, None] + Ste[:, None] * shape
            row["c95_" + nm] = float(np.mean((Yte >= Q[:, 0]) & (Yte <= Q[:, 14])))
            row["c80_" + nm] = float(np.mean((Yte >= Q[:, 2]) & (Yte <= Q[:, 12])))
        RES[f"{m}|{h}"] = row
        print(f"  h={h:<4} A5 {row['A5']:.5f} | A14 {row['A14']:.5f} "
              f"({row['A14']/row['A5']:.3f},p={row['p_A14_vs_A5']:.3f}) | "
              f"A17b a={kb[0]:.1f} b={kb[1]:+.1f} {row['A17b']:.5f} "
              f"({row['A17b']/row['A5']:.3f},p_vsA5={row['p_A17b_vs_A5']:.3f},"
              f"p_vsA14={row['p_A17b_vs_A14']:.3f})")

# ---------- A17a：每市场跨期限共享 (a,b) ----------
print("\n" + "=" * 118)
print("A17a：每市场跨五期限共享 (a,b)（在训练段池化 CRPS 选参）")
print("=" * 118)
for m in CUT:
    keys = [f"{m}|{h}" for h in HOR if f"{m}|{h}" in RES]
    if not keys:
        continue
    # 池化：用每格训练 CRPS 的和近似（每格样本数不同，用均值池化）
    sc = {}
    for key, sh in SHAPES.items():
        vals = []
        for k in keys:
            r = RES[k]
            sc.setdefault(key, None)
        break
    # 直接重算：每个 (a,b) 在该市场所有格上的训练 CRPS 之和（等权）
    tot = {}
    for key, sh in SHAPES.items():
        s = 0.0
        for k in keys:
            r = RES[k]
            s += r["tr_scores"][f"{key[0]}|{key[1]}"]
        tot[key] = s / len(keys)
    ka = min(tot, key=tot.get)
    RES[m + "|MARKET"] = dict(a=ka[0], b=ka[1])
    print(f"  {MN[m]:<8} 最优 (a,b) = ({ka[0]:.1f}, {ka[1]:+.1f})   "
          f"各格最优 b: {[RES[k]['kb'][1] for k in keys]}")
    print(f"           各格最优 a: {[RES[k]['kb'][0] for k in keys]}")

# 用 A17a 的参数重算测试段
for m in CUT:
    if m + "|MARKET" not in RES:
        continue
    ka = (RES[m + "|MARKET"]["a"], RES[m + "|MARKET"]["b"])
    sh = SHAPES[ka]
    for h in HOR:
        k = f"{m}|{h}"
        if k not in RES:
            continue
        d = p[p.market == m].sort_values("date").reset_index(drop=True)
        lp = np.log(d.price.to_numpy(float)).astype(float)
        dt = pd.to_datetime(d.date).to_numpy()
        t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
        h0, Hv = PAR[m]
        Y, L, S = [], [], []
        for t in range(max(W + h + 5, t0), len(lp) - h, STEP):
            win = lp[max(0, t - W):t]
            rr = win[h:] - win[:-h]
            if len(rr) < 3 or np.std(rr, ddof=1) <= 0:
                continue
            rd = np.diff(win); sdd = np.std(rd, ddof=1)
            if not np.isfinite(sdd) or sdd <= 0:
                continue
            Y.append(lp[t + h] - lp[t]); L.append((h0 / (h0 + h)) * rr.mean())
            S.append(sdd * h ** Hv)
        Y, L, S = map(np.array, (Y, L, S))
        L17a = crps_q(Y, L[:, None] + S[:, None] * np.tile(sh, (len(Y), 1)))
        RES[k]["A17a"] = float(L17a.mean())
        RES[k]["p_A17a_vs_A5"] = dm(L17a, np.array([RES[k]["A5"]] * len(Y)))
        RES[k]["p_A17a_vs_A14"] = dm(L17a, np.array([RES[k]["A14"]] * len(Y)))

json.dump(RES, open(os.path.join(DIR, "A17.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)

cells = [(m, h) for m in CUT for h in HOR]
cells = [(m, h) for m, h in cells if f"{m}|{h}" in RES]
print("\n" + "=" * 118)
print("汇总（相对 A5；A14 为当前最优基准）")
print("=" * 118)
print(f"{'模型':<8}{'中位':>9}{'胜':>7}{'显著胜':>8}{'显著负':>8}{'逐期限中位 5/20/40/60/120':>38}")
for nm, pnm in (("A14", "p_A14_vs_A5"), ("A17a", "p_A17a_vs_A5"), ("A17b", "p_A17b_vs_A5")):
    rr = np.array([RES[f"{m}|{h}"][nm] / RES[f"{m}|{h}"]["A5"] for m, h in cells])
    sw = sum(1 for m, h in cells if RES[f"{m}|{h}"][pnm] < .05 and RES[f"{m}|{h}"][nm] < RES[f"{m}|{h}"]["A5"])
    sl = sum(1 for m, h in cells if RES[f"{m}|{h}"][pnm] < .05 and RES[f"{m}|{h}"][nm] > RES[f"{m}|{h}"]["A5"])
    per = " / ".join(f"{np.median([RES[f'{m}|{h}'][nm]/RES[f'{m}|{h}']['A5'] for m,_ in cells if h==h]):.3f}"
                     for h in HOR)
    print(f"{nm:<8}{np.median(rr):>9.4f}{int((rr<1).sum()):>4}/25{sw:>8}{sl:>8}{per:>38}")

print("\n相对 A14：")
for nm, pnm in (("A17a", "p_A17a_vs_A14"), ("A17b", "p_A17b_vs_A14")):
    rr = np.array([RES[f"{m}|{h}"][nm] / RES[f"{m}|{h}"]["A14"] for m, h in cells])
    sw = sum(1 for m, h in cells if RES[f"{m}|{h}"][pnm] < .05 and RES[f"{m}|{h}"][nm] < RES[f"{m}|{h}"]["A14"])
    sl = sum(1 for m, h in cells if RES[f"{m}|{h}"][pnm] < .05 and RES[f"{m}|{h}"][nm] > RES[f"{m}|{h}"]["A14"])
    print(f"  {nm:<6} 中位 {np.median(rr):.4f}  胜 {int((rr<1).sum())}/25  显著胜 {sw}  显著负 {sl}")

print("\n覆盖率（95%，目标 0.95）：")
for h in HOR:
    ks = [(m, h) for m, h in cells if h == h]
    print(f"  h={h:<5} A5 {np.mean([RES[f'{m}|{h}']['c95_A5'] for m,_ in ks]):.3f}  "
          f"A14 {np.mean([RES[f'{m}|{h}']['c95_A14'] for m,_ in ks]):.3f}  "
          f"A17b {np.mean([RES[f'{m}|{h}']['c95_A17b'] for m,_ in ks]):.3f}")
print("\n已写出 A17.json")
