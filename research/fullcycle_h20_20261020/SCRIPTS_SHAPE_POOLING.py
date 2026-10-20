# -*- coding: utf-8 -*-
"""
形状的双重可分离性检验 v2
========================
v1 的两处修正：
 (1) 标准化必须**先去位置再除尺度**，否则 h=120 的「形状」量到的是漂移
     （v1 中上海中位 +0.653、CEA 0.000，明显是趋势不是形状）
 (2) Part B 的 pinball 广播 bug

位置与形状分离：
  mu   = 0（纯随机游走位置，只让形状不同）
  s_p  = quantile( (r_h - mean(r_h)) / sd(r_h), p )   ← 纯形状
  Q_p  = sigma_h * s_p                               ← sigma_h 用本地经验 h 期标准差
"""
import pandas as pd, numpy as np, os, sys, json, warnings
from scipy.stats import norm
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

OUT = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(OUT, "PANEL_5MARKET.csv"), parse_dates=["date"])
MKT = ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]
P = np.array([0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975])
HOR = [5, 20, 60, 120]
W_SHAPE = 500

LP, DT = {}, {}
for m in MKT:
    g = p[p.market == m].sort_values("date").reset_index(drop=True)
    LP[m] = np.log(g.price.to_numpy(float)).astype(float)
    DT[m] = pd.to_datetime(g.date.to_numpy())


def window_before(m, cutoff, W):
    d = DT[m].to_numpy()
    j = min(int(np.searchsorted(d, np.datetime64(cutoff), side="right")), len(LP[m]))
    return LP[m][max(0, j - W):j], j


def pure_shape(r, Ps=P):
    """去位置、除尺度后的分位函数"""
    if len(r) < 30:
        return None
    s = np.std(r, ddof=1)
    if s <= 0:
        return None
    return np.quantile((r - np.mean(r)) / s, Ps)


rng = np.random.default_rng(11)
print("=" * 118)
print("Part A  去位置后的跨市场形状 s(h,p)（窗口截至 2023-12-31，W=500）")
print("=" * 118)
print(f"{'h':>4}{'p':>7}{'Gauss':>8}" + "".join(f"{m[:8]:>10}" for m in MKT)
      + f"{'跨市场SD':>10}{'vsGauss':>9}")
SHAPE, MATS = {}, {}
for h in HOR:
    rows = {}
    for m in MKT:
        lv, _ = window_before(m, "2023-12-31", W_SHAPE)
        if len(lv) < h + 60:
            continue
        q = pure_shape(lv[h:] - lv[:-h])
        if q is not None:
            rows[m] = q
            SHAPE[(m, h)] = q
    if len(rows) < 3:
        print(f"{h:>4}  可用市场不足")
        continue
    A = np.column_stack([rows[m] for m in MKT if m in rows])
    MATS[h] = (A, [m for m in MKT if m in rows])
    for i, pu in enumerate(P):
        print(f"{h:>4}{pu:>7.3f}{norm.ppf(pu):>8.3f}" + "".join(f"{v:>10.3f}" for v in A[i])
              + f"{np.std(A[i], ddof=1):>10.3f}{np.mean(A[i])-norm.ppf(pu):>+9.3f}")
    print()

print("=" * 118)
print("Part A2 跨市场SD / 市场内自助SD")
print("=" * 118)


def boot_shape(r, nrep=120, bs=20):
    out = np.full((nrep, len(P)), np.nan)
    if len(r) < bs * 4:
        return out
    for k in range(nrep):
        st = rng.integers(0, max(1, len(r) - bs), size=int(np.ceil(len(r) / bs)))
        sx = np.concatenate([r[s:s + bs] for s in st])[:len(r)]
        sd = np.std(sx, ddof=1)
        if sd > 0:
            out[k] = np.quantile((sx - sx.mean()) / sd, P)
    return out


print(f"{'h':>4}{'p':>7}{'跨市场SD':>10}{'抽样SD':>9}{'比值':>8}{'判定':>12}")
RAT = []
for h in HOR:
    if h not in MATS:
        continue
    A, ms = MATS[h]
    for i, pu in enumerate(P):
        csd = float(np.std(A[i], ddof=1))
        ssds = []
        for m in ms:
            lv, _ = window_before(m, "2023-12-31", W_SHAPE)
            bb = boot_shape(lv[h:] - lv[:-h])[:, i]
            bb = bb[np.isfinite(bb)]
            if len(bb) > 20:
                ssds.append(np.std(bb, ddof=1))
        if not ssds:
            continue
        med = float(np.median(ssds))
        rat = csd / med if med > 0 else np.nan
        RAT.append((h, pu, csd, med, rat))
        print(f"{h:>4}{pu:>7.3f}{csd:>10.3f}{med:>9.3f}{rat:>8.2f}"
              f"{('可视为不变' if rat<0.5 else ('边缘' if rat<1.0 else '显著不同')):>12}")

Rv = np.array([r[4] for r in RAT if np.isfinite(r[4])])
print(f"\n  全部 {len(Rv)} 格 比值中位 {np.median(Rv):.2f}；比值<1 占 {np.mean(Rv<1)*100:.0f}%")
print("  按期限：", "  ".join(
    f"h={h}: {np.median([r[4] for r in RAT if r[0]==h and np.isfinite(r[4])]):.2f}" for h in HOR))

# ============================================================
print()
print("=" * 118)
print("Part B  样本外 pinball（位置 mu=0，尺度用本地经验 h 期标准差，只让形状不同）")
print("=" * 118)
PFIT = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
TEST = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01"}
W = 250


def pinball(q, y, ps):
    """q:(n,k) y:(n,) ps:(k,)"""
    d = y[:, None] - q
    return np.maximum(ps[None, :] * d, (ps[None, :] - 1) * d).mean()


def pooled_shape(m, cutoff, h, ps):
    acc = []
    for o in MKT:
        if o == m:
            continue
        lv, _ = window_before(o, cutoff, len(LP[o]))
        if len(lv) < h + 60:
            continue
        r = lv[h:] - lv[:-h]
        s = np.std(r, ddof=1)
        if s > 0:
            acc.append((r - r.mean()) / s)
    if not acc:
        return None
    return np.quantile(np.concatenate(acc), ps)


RES = {}
for m in ["CEA", "GUANGZHOU", "SHANGHAI"]:
    lv_all = LP[m]
    dt = DT[m].to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(TEST[m])))
    print(f"\n{m}   测试起点 {TEST[m]}   n_test={len(lv_all)-t0}")
    print(f"{'h':>5}{'n':>6}{'Gaussian':>11}{'本地形状':>11}{'借用他市场':>12}{'部分池化':>11}"
          f"{'a':>6}{'借用vs本地':>12}{'池化vs本地':>12}")
    RES[m] = {}
    for h in HOR:
        PSH = pooled_shape(m, TEST[m], h, PFIT)
        if PSH is None:
            continue
        recs = []
        lo_t = max(W + h + 30, t0 - 500)
        for t in range(lo_t, len(lv_all) - h):
            win = lv_all[max(0, t - W):t]
            if len(win) < h + 30:
                continue
            r = win[h:] - win[:-h]
            sd = np.std(r, ddof=1)
            if sd <= 0:
                continue
            loc = np.quantile((r - r.mean()) / sd, PFIT)
            recs.append((float(lv_all[t + h] - lv_all[t]), sd, loc, t >= t0))
        if len(recs) < 100:
            continue
        Y = np.array([r[0] for r in recs]); SD = np.array([r[1] for r in recs])
        LOC = np.array([r[2] for r in recs]); IS = np.array([r[3] for r in recs])
        if IS.sum() < 60 or (~IS).sum() < 60:
            continue
        GA = np.tile(norm.ppf(PFIT), (len(Y), 1))
        grid = np.linspace(0, 1, 21)
        los = [pinball((SD[:, None] * ((1 - a) * LOC + a * PSH[None, :]))[IS], Y[IS], PFIT) for a in grid]
        abest = float(grid[int(np.argmin(los))])
        PSB = (1 - abest) * LOC + abest * PSH[None, :]
        out = {}
        for nm, SC in [("gauss", GA), ("local", LOC),
                       ("pool", np.tile(PSH, (len(Y), 1))), ("part", PSB)]:
            out[nm] = pinball((SD[:, None] * SC)[~IS], Y[~IS], PFIT)
        def pc(x, ref):
            return f"{(x/ref-1)*100:+11.2f}%" if ref > 0 else f"{'--':>12}"
        print(f"{h:>5}{int((~IS).sum()):>6}{out['gauss']:>11.5f}{out['local']:>11.5f}"
              f"{out['pool']:>12.5f}{out['part']:>11.5f}{abest:>6.2f}"
              f"{pc(out['pool'],out['local'])}{pc(out['part'],out['local'])}")
        RES[m][h] = dict(n=int((~IS).sum()), gauss=out["gauss"], local=out["local"],
                         pool=out["pool"], part=out["part"], a=abest)
    if RES[m]:
        print("  选出的池化权重 a（0=完全本地，1=完全借用）："
              + "  ".join(f"h{h}:{RES[m][h]['a']:.2f}" for h in sorted(RES[m])))

json.dump({"dispersion": [[float(x) for x in r] for r in RAT],
           "ratio_median": float(np.median(Rv)), "oos": RES},
          open(os.path.join(OUT, "SHAPE_POOLING.json"), "w"), indent=1,
          ensure_ascii=False, default=float)
print("\n已写出 SHAPE_POOLING.json")
