# -*- coding: utf-8 -*-
"""
A20：A14 的机制化升级 —— 形状混合权重由**有效样本量**决定，而非常数 50
=====================================================================
已确立的事实（§24.27 + §24.29）：
 1) 实现形状（窗内日收益的标准化经验分位）几乎总是优于高斯：
    训练段 CRPS 比值 0.88~0.99（CEA h=120 低至 0.28）。
 2) 但实现形状**不是** CRPS 最优的：A6（纯实现形状）整体更差，
    因为窗内日收益的形状在高期限时不含 h 步聚合的信息。
 3) A14 用 w(h)=min(1, h/50) 混合二者，在 h≤40 拿到 3 格显著胜、0 格显著负。
    但 **50 是从 oracle 剖面读出来的常数，不是拟合值**（§24.27 已披露）。
 4) §24.29 的新证据：形状混合权重在**训练段与测试段的偏度符号相反**
    （训练段偏好右偏 α=+2，实现偏度却是负的）⇒ 形状参数不可跨样本迁移。

A20 的假设：混合权重应当由**有效样本量**决定，而且方向应与 A14 相反：
  - 短期限（h 小）：h 步收益 ≈ 日收益，窗内实现形状**有信息** → w 应大（多用实现）
  - 长期限（h 大）：h 步收益的形状无法从日收益的形状推出 → w 应小（退回高斯）
  ⇒ w(h) = 1 / (1 + (h/h_s)^p)，h_s 与 p 由**训练段**估。
这与 A14 的 w=min(1,h/50) **方向相反**（A14 是 h 越大越用高斯……等等，
  A14 的 w 是"高斯权重"：w=1 时形状是高斯，w=0 时是实现形状。
  所以 A14：h 小 → w=0.1（多用实现形状），h 大 → w=1（全高斯）。
  ⇒ A14 的方向已经是"h 越大越用高斯"，与我的假设一致。
  那 A20 的新意在于：把 w 的形式从 min(1,h/50) 换成**由训练段估计的
  有效样本量函数**，并去掉那个来自 oracle 的常数 50。

候选：
  A20a  w = 1/(1+(h/h_s)^p)，(h_s,p) 每市场由训练段估（2 参数/市场）
  A20b  w = 1/(1+h/h_s)，h_s 每市场由训练段估（1 参数/市场）
  A20c  w = min(1, h/h_s)，h_s 每市场由训练段估（1 参数/市场，A14 的可估版）
  A20d  w = 1/(1+(W/h)/(W/h+κ)) 有效样本量形式，κ 每市场训练段估
关键判据：A20 必须**在同等参数预算下不劣于 A14**，且常数不再来自 oracle。
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
HS = [5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 500, 1000, 1e9]
PS = [0.5, 1.0, 1.5, 2.0, 3.0]
KPS = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1e9]


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
        DATA[(m, h)] = dict(tr=tr, te=te)
cells = sorted(DATA.keys())
print(f"有效格 {len(cells)}")


def wfun(kind, h, par):
    if kind == "A20a":
        hs, pw = par
        return 1.0 / (1.0 + (h / hs) ** pw) if hs < 1e8 else 0.0
    if kind == "A20b":
        hs = par[0] if isinstance(par, tuple) else par
        return 1.0 / (1.0 + h / hs) if hs < 1e8 else 0.0
    if kind == "A20c":
        hs = par[0] if isinstance(par, tuple) else par
        return min(1.0, h / hs) if hs < 1e8 else 0.0
    if kind == "A20d":
        kap = par[0] if isinstance(par, tuple) else par
        nind = max(1.0, W / h)
        return 1.0 / (1.0 + nind / kap) if kap < 1e8 else 0.0
    raise ValueError


GRID = {"A20a": [(hs, pw) for hs in HS for pw in PS],
        "A20b": [(hs,) for hs in HS],
        "A20c": [(hs,) for hs in HS],
        "A20d": [(k,) for k in KPS]}

BEST = {}
print("\n" + "=" * 112)
print("训练段选参（每市场，五期限池化）")
print("=" * 112)
for m in CUT:
    ks = [k for k in cells if k[0] == m]
    if not ks:
        continue
    BEST[m] = {}
    for kind, g in GRID.items():
        tot = {}
        for par in g:
            s = 0.0; nn = 0
            for (mm, h) in ks:
                Y, L, S, XI = DATA[(mm, h)]["tr"]
                w = wfun(kind, h, par)
                shp = (1 - w) * XI + w * GV[None, :]
                s += float(crps_q(Y, L[:, None] + S[:, None] * shp).sum()); nn += len(Y)
            tot[par] = s / nn
        BEST[m][kind] = min(tot, key=tot.get)
    # A14 参照（常数 50）与 A5（全高斯）在训练段的值
    def sc(wfun_fixed):
        s = 0.0; nn = 0
        for (mm, h) in ks:
            Y, L, S, XI = DATA[(mm, h)]["tr"]
            w = wfun_fixed(h)
            shp = (1 - w) * XI + w * GV[None, :]
            s += float(crps_q(Y, L[:, None] + S[:, None] * shp).sum()); nn += len(Y)
        return s / nn
    print(f"  {MN[m]:<8} A20a {BEST[m]['A20a']}  A20b {BEST[m]['A20b']}  "
          f"A20c {BEST[m]['A20c']}  A20d {BEST[m]['A20d']}  |  "
          f"A14(常数50) {sc(lambda h: min(1.0, h/50.0)):.5f}  A5(全高斯) {sc(lambda h: 1.0):.5f}")

RES = {}
for (m, h) in cells:
    Y, L, S, XI = DATA[(m, h)]["te"]
    cand = {"A5": np.tile(GV, (len(Y), 1)),
            "A14": (1 - min(1.0, h / 50.0)) * XI + min(1.0, h / 50.0) * GV[None, :]}
    for kind in GRID:
        par = BEST[m][kind]
        w = wfun(kind, h, par)
        cand[kind] = (1 - w) * XI + w * GV[None, :]
    Ls = {nm: crps_q(Y, L[:, None] + S[:, None] * shp) for nm, shp in cand.items()}
    row = dict(n=len(Y))
    for nm in cand:
        row[nm] = float(Ls[nm].mean())
        row["w_" + nm] = float(1 - (cand[nm][0] - XI[0]).sum() / 0 if False else 0)
    row["p_A14"] = dm(Ls["A14"], Ls["A5"])
    for nm in GRID:
        row["p_" + nm] = dm(Ls[nm], Ls["A5"])
        row["p_" + nm + "_vs_A14"] = dm(Ls[nm], Ls["A14"])
    for nm, shp in cand.items():
        Q = L[:, None] + S[:, None] * shp
        row["c95_" + nm] = float(np.mean((Y >= Q[:, 0]) & (Y <= Q[:, 14])))
        row["c80_" + nm] = float(np.mean((Y >= Q[:, 2]) & (Y <= Q[:, 12])))
    RES[f"{m}|{h}"] = row

json.dump({"RES": RES, "BEST": {m: {k: list(v) for k, v in d.items()} for m, d in BEST.items()}},
          open(os.path.join(DIR, "A20.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)

print("\n" + "=" * 118)
print("汇总（相对 A5 / 相对 A14）")
print("=" * 118)
print(f"{'模型':<8}{'参数/市场':>10}{'/A5中位':>10}{'胜':>7}{'显著胜':>8}{'显著负':>8}"
      f"{'/A14中位':>11}{'胜':>7}{'显著胜':>8}{'显著负':>8}{'逐期限/A5':>32}")
NPAR = {"A14": 0, "A5": 0, "A20a": 2, "A20b": 1, "A20c": 1, "A20d": 1}
for nm in ("A14", "A20a", "A20b", "A20c", "A20d"):
    rr = np.array([RES[f"{m}|{h}"][nm] / RES[f"{m}|{h}"]["A5"] for m, h in cells])
    sw = sum(1 for m, h in cells if RES[f"{m}|{h}"].get("p_" + nm, 1) < .05 and RES[f"{m}|{h}"][nm] < RES[f"{m}|{h}"]["A5"])
    sl = sum(1 for m, h in cells if RES[f"{m}|{h}"].get("p_" + nm, 1) < .05 and RES[f"{m}|{h}"][nm] > RES[f"{m}|{h}"]["A5"])
    r14 = np.array([RES[f"{m}|{h}"][nm] / RES[f"{m}|{h}"]["A14"] for m, h in cells])
    s14 = sum(1 for m, h in cells if RES[f"{m}|{h}"].get("p_" + nm + "_vs_A14", 1) < .05 and RES[f"{m}|{h}"][nm] < RES[f"{m}|{h}"]["A14"])
    l14 = sum(1 for m, h in cells if RES[f"{m}|{h}"].get("p_" + nm + "_vs_A14", 1) < .05 and RES[f"{m}|{h}"][nm] > RES[f"{m}|{h}"]["A14"])
    per = " / ".join(f"{np.median([RES[f'{m}|{h}'][nm]/RES[f'{m}|{h}']['A5'] for m,_ in cells if h==h]):.3f}"
                     for h in HOR)
    print(f"{nm:<8}{NPAR[nm]:>10}{np.median(rr):>10.4f}{int((rr<1).sum()):>4}/25{sw:>8}{sl:>8}"
          f"{np.median(r14):>11.4f}{int((r14<1).sum()):>4}/25{s14:>8}{l14:>8}{per:>32}")
print("\n已写出 A20.json")
