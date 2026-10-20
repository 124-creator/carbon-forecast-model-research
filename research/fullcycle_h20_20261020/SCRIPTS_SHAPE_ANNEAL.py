# -*- coding: utf-8 -*-
"""
形状退火（shape annealing）
=========================
原理：n_eff = W/h 随期限下降，非参数形状估计方差爆炸。
      用一个等效先验样本量 n0 把形状向 Gaussian 极限收缩：

          kappa_h = n0 / (n0 + W/h)
          s(h,p)  = (1 - kappa_h)*s_local(h,p) + kappa_h*Phi^-1(p)

  n0 -> 0   退化为纯本地经验形状
  n0 -> inf 退化为纯 Gaussian

一个参数 n0，在训练段拟合，测试段外推。
对比：本地经验 / 纯 Gaussian / 退火 / 每期限最优 kappa（上界）
"""
import pandas as pd, numpy as np, os, sys, json, warnings
from scipy.stats import norm
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

OUT = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(OUT, "PANEL_5MARKET.csv"), parse_dates=["date"])
MKT = ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]
PFIT = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
GV = norm.ppf(PFIT)
HOR = [2, 3, 5, 10, 20, 40, 60, 120]
W = 250
TEST = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
        "HUBEI": "2023-01-01", "EU": "2023-01-01"}

LP, DT = {}, {}
for m in MKT:
    g = p[p.market == m].sort_values("date").reset_index(drop=True)
    LP[m] = np.log(g.price.to_numpy(float)).astype(float)
    DT[m] = pd.to_datetime(g.date.to_numpy())


def pinball(q, y, ps):
    d = y[:, None] - q
    return np.maximum(ps[None, :] * d, (ps[None, :] - 1) * d).mean()


def build(m, h, t0):
    lv = LP[m]
    recs = []
    lo = max(W + h + 30, t0 - 600)
    for t in range(lo, len(lv) - h):
        win = lv[max(0, t - W):t]
        if len(win) < h + 30:
            continue
        r = win[h:] - win[:-h]
        sd = np.std(r, ddof=1)
        if sd <= 0:
            continue
        loc = np.quantile((r - r.mean()) / sd, PFIT)
        recs.append((float(lv[t + h] - lv[t]), sd, loc, t >= t0))
    if len(recs) < 150:
        return None
    Y = np.array([x[0] for x in recs]); SD = np.array([x[1] for x in recs])
    LO = np.array([x[2] for x in recs]); IS = np.array([x[3] for x in recs])
    if IS.sum() < 80 or (~IS).sum() < 60:
        return None
    return Y, SD, LO, IS


N0G = np.concatenate([[0.0], np.logspace(-0.5, 2.0, 40)])
print("=" * 120)
print("形状退火：kappa_h = n0/(n0+W/h)，n0 在训练段拟合后外推")
print("=" * 120)
ALL = {}
for m in ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]:
    t0 = int(np.searchsorted(DT[m].to_numpy(), np.datetime64(TEST[m])))
    D = {}
    for h in HOR:
        b = build(m, h, t0)
        if b is not None:
            D[h] = b
    if not D:
        continue
    # 拟合单一 n0（跨所有 h 共享）
    los = []
    for n0 in N0G:
        tot = 0.0; cnt = 0
        for h, (Y, SD, LO, IS) in D.items():
            kap = n0 / (n0 + W / h) if n0 > 0 else 0.0
            PS = (1 - kap) * LO + kap * GV[None, :]
            tot += pinball((SD[:, None] * PS)[IS], Y[IS], PFIT) * IS.sum(); cnt += IS.sum()
        los.append(tot / cnt)
    n0b = float(N0G[int(np.argmin(los))])
    print(f"\n{m}   测试起点 {TEST[m]}   拟合 n0 = {n0b:.2f}")
    print(f"{'h':>5}{'W/h':>7}{'kappa':>8}{'n_test':>7}{'本地形状':>11}{'纯Gaussian':>12}"
          f"{'退火':>11}{'最优kappa':>10}{'退火vs本地':>12}{'退火vsGauss':>12}{'退火vs最优':>11}")
    ALL[m] = {"n0": n0b, "rows": {}}
    for h in sorted(D):
        Y, SD, LO, IS = D[h]
        kap = n0b / (n0b + W / h) if n0b > 0 else 0.0
        PS = (1 - kap) * LO + kap * GV[None, :]
        def sc(SC):
            return pinball((SD[:, None] * SC)[~IS], Y[~IS], PFIT)
        v_loc = sc(LO); v_g = sc(np.tile(GV, (len(Y), 1))); v_an = sc(PS)
        # 每期限最优 kappa（训练段）
        kg = np.linspace(0, 1, 21)
        lk = [pinball((SD[:, None] * ((1 - k) * LO + k * GV[None, :]))[IS], Y[IS], PFIT) for k in kg]
        kb = float(kg[int(np.argmin(lk))])
        v_or = sc((1 - kb) * LO + kb * GV[None, :])
        pc = lambda a, b: f"{(a/b-1)*100:+11.2f}%" if b > 0 else f"{'--':>12}"
        print(f"{h:>5}{W/h:>7.1f}{kap:>8.3f}{int((~IS).sum()):>7}{v_loc:>11.5f}{v_g:>12.5f}"
              f"{v_an:>11.5f}{kb:>10.2f}{pc(v_an,v_loc)}{pc(v_an,v_g)}{pc(v_an,v_or)}")
        ALL[m]["rows"][h] = dict(local=v_loc, gauss=v_g, anneal=v_an, oracle=v_or,
                                 kappa=kap, kbest=kb, n=int((~IS).sum()))

print()
print("=" * 120)
print("汇总：退火相对两个端点的变化（负=退火更好）")
print("=" * 120)
print(f"{'市场':<11}{'h':>5}{'vs本地':>10}{'vsGauss':>10}{'哪端更好':>10}")
CNT = {"loc": 0, "g": 0}
for m in ALL:
    for h in sorted(ALL[m]["rows"]):
        r = ALL[m]["rows"][h]
        a = (r["anneal"] / r["local"] - 1) * 100
        b = (r["anneal"] / r["gauss"] - 1) * 100
        better = "本地" if r["local"] < r["gauss"] else "Gaussian"
        CNT["loc" if better == "本地" else "g"] += 1
        print(f"{m:<11}{h:>5}{a:>+10.2f}{b:>+10.2f}{better:>10}")
print(f"\n  端点优劣：本地形状更好 {CNT['loc']} 格，Gaussian 更好 {CNT['g']} 格")

# 关键：按 n_eff 分段统计端点优劣
print()
print("=" * 120)
print("按 W/h 分段：非参数形状何时开始不如 Gaussian？")
print("=" * 120)
seg = [(0, 3), (3, 6), (6, 15), (15, 60), (60, 200)]
for lo, hi in seg:
    dl = []
    for m in ALL:
        for h, r in ALL[m]["rows"].items():
            ne = W / h
            if lo <= ne < hi:
                dl.append(r["local"] / r["gauss"] - 1)
    if dl:
        dl = np.array(dl) * 100
        print(f"  W/h in [{lo:>3},{hi:>3})  n={len(dl):>2}  "
              f"本地形状相对 Gaussian: 中位 {np.median(dl):+6.2f}%  "
              f"（>0 表示非参数更差）  非参数更差的占 {np.mean(dl>0)*100:.0f}%")

json.dump(ALL, open(os.path.join(OUT, "SHAPE_ANNEAL.json"), "w"), indent=1,
          ensure_ascii=False, default=float)
print("\n已写出 SHAPE_ANNEAL.json")
