# -*- coding: utf-8 -*-
"""
COMBO v5：可估期限标度指数 —— 最终规格
========================================
v4 的诊断：
  (1) κ 乘子（A5）对 A2 只有 1/25 显著，且在 SH 上反而变差（h=120 +5.05）；
      训练集拟合的 κ 再次在测试集漂移 -> 弃用。
  (2) CEA h=120 的逐格 κ* 打到网格上界 2.50：不是宽度不够，是
      **去掉位置漂移后中心偏了**，只能靠撑宽硬覆盖。
  (3) 更根本的问题：窗口 h 期重叠收益的尺度与形状都只有 n_eff = W/h 个
      有效观测（h=120 时 ≈2 个），抽样误差爆炸 -> 系统性覆盖不足。

最终规格（两块可估 + 一个切换）：
  Q_{t,h}(tau) = lam(h) * mu_{t,h} + sigma_{d,t} * h^H_m * xi_{t,h}(tau)
     位置块 lam(h) = h0_m / (h0_m + h)        —— 逐市场，训练集
     尺度块 H_m                                —— 逐市场，训练集；(h0_m, H_m) 联合网格
              尺度取自**日频**标准差（n_eff = W），替代 h 期重叠标准差（n_eff = W/h）
     形状块 xi：经验分位 X 或 Gaussian G
对照：
  RW   = lam=0 + 窗口 h 期尺度 + 经验形状（随机游走密度基准）
  EMP  = lam=1 + 窗口 h 期尺度 + 经验形状（文献默认：窗口经验分位）
"""
import numpy as np, pandas as pd, os, sys, json, warnings
from scipy.stats import norm, t as tdist
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

OUT = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(OUT, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS); J80 = (2, 12); J95 = (0, 14)
W = 250; HOR = [5, 20, 40, 60, 120]; STEP = 2
H0G = [0.1, 0.2, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 500]
HG = np.round(np.arange(0.15, 0.861, 0.05), 3)
MK = ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]
AB = ["RW", "EMP", "A2 +位置", "A3 +形状", "A4 +H", "A5 完整"]


def crps_q(y, Q):
    Qs = np.sort(Q, axis=1)
    ind = (y[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (y[:, None] - Qs)).mean(axis=1)


def dm(d, L=10):
    n = len(d)
    if n < 20:
        return np.nan, np.nan
    m = d.mean(); dc = d - m
    lrv = dc @ dc / n
    for l in range(1, L + 1):
        lrv += 2 * (1 - l / (L + 1)) * (dc[l:] @ dc[:-l] / n)
    if lrv <= 0:
        return np.nan, np.nan
    s = m / np.sqrt(lrv / n) * np.sqrt((n + 1 - 2 * L + L * (L - 1) / n) / n)
    return float(s), float(2 * (1 - tdist.cdf(abs(s), df=n - 1)))


def precomp(m, h):
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float))
    dt = pd.to_datetime(d.date).to_numpy()
    cut = "2024-01-01" if m in ("CEA", "GUANGZHOU", "SHANGHAI") else "2023-01-01"
    t0 = int(np.searchsorted(dt, np.datetime64(cut)))
    rec = []
    for t in range(W + h + 5, len(lp) - h, STEP):
        win = lp[t - W:t]
        rr = win[h:] - win[:-h]
        rd = np.diff(win)
        sdw = np.std(rr, ddof=1); sdd = np.std(rd, ddof=1)
        if sdw <= 1e-12 or sdd <= 1e-12:
            continue
        z = (rr - rr.mean()) / sdw
        rec.append((t, float(rr.mean()), sdw, sdd, np.quantile(z, TAUS)))
    if len(rec) < 60:
        return None
    ts = np.array([r[0] for r in rec])
    return dict(ts=ts, h=h, m=m,
                mu=np.array([r[1] for r in rec]), sd=np.array([r[2] for r in rec]),
                sdd=np.array([r[3] for r in rec]),
                xi=np.vstack([r[4] for r in rec]),
                y=lp[ts + h] - lp[ts],
                tr=(ts >= W + h + 5) & (ts <= t0 - h - 2),
                te=ts >= max(W + h + 5, t0))


CC = {(m, h): precomp(m, h) for m in MK for h in HOR}
CC = {k: v for k, v in CC.items() if v is not None}
print(f"缓存 {len(CC)} 格；泄漏审计：重叠>0 = "
      f"{sum(int((c['tr'] & c['te']).sum() > 0) for c in CC.values())}（必须 0）")


def Qof(c, idx, lam, Hval, cc, sh):
    mu = c["mu"][idx]
    s = c["sd"][idx] if Hval is None else c["sdd"][idx] * cc ** Hval
    xi = c["xi"][idx] if sh == "X" else np.tile(GV, (len(mu), 1))
    return mu[:, None] * lam + s[:, None] * xi


def loss(c, idx, lam, Hval, cc, sh):
    return crps_q(c["y"][idx], Qof(c, idx, lam, Hval, cc, sh))


def cov(c, idx, lam, Hval, cc, sh, jl, ju):
    Q = Qof(c, idx, lam, Hval, cc, sh); y = c["y"][idx]
    return float(((y >= Q[:, jl]) & (y <= Q[:, ju])).mean())


# ---------- 1. 逐市场联合 (h0, H) ----------
print(f"\n{'='*152}")
print("表 1  逐市场联合拟合 (h0, H)（训练集最小 CRPS，形状块=G）。H=0.5 即 iid sqrt(h)")
print(f"{'='*152}")
PAR = {}
for m in MK:
    hs = [h for h in HOR if (m, h) in CC]
    best = (1e9, None, None)
    for Hval in HG:
        for h0 in H0G:
            v = float(np.mean([loss(CC[(m, h)], CC[(m, h)]["tr"], h0 / (h0 + h), Hval, h, "G").mean()
                               for h in hs]))
            if v < best[0]:
                best = (v, h0, Hval)
    PAR[m] = dict(h0=int(best[1]) if float(best[1]).is_integer() else best[1],
                  H=float(best[2]), train=float(best[0]))
    # 对照：只用训练集的 sqrt(h)（H=0.5）最优 h0
    v05 = float(np.mean([loss(CC[(m, h)], CC[(m, h)]["tr"],
                              PAR[m]["h0"] / (PAR[m]["h0"] + h), 0.5, h, "G").mean() for h in hs]))
    print(f"  {m:<11} h0*={PAR[m]['h0']:<6} H*={PAR[m]['H']:.2f}  训练CRPS={PAR[m]['train']:.5f}"
          f"   (对照 H=0.5: {v05:.5f}, {((v05/PAR[m]['train']-1)*100):+.1f}%)"
          f"   lam(5,20,60,120)="
          + ",".join(f"{PAR[m]['h0']/(PAR[m]['h0']+h):.2f}" for h in (5, 20, 60, 120)))

# ---------- 2. 消融 ----------
print(f"\n{'='*152}")
print("表 2  消融（未校准 CRPS）。RW = 随机游走密度（lam=0, 窗口尺度, 经验形状）")
print(f"{'='*152}")
print(f"{'市场':>10}{'h':>5}{'n_te':>6}" + "".join(f"{n:>10}" for n in AB)
      + f"{'A5vsEMP':>9}{'A5vsRW':>9}" + "".join(f"{'P95:'+n:>10}" for n in ("RW", "EMP", "A5 完整"))
      + f"{'P80:A5':>8}")
CELL = {}


def spec(m, h, nm):
    L = PAR[m]["h0"] / (PAR[m]["h0"] + h)
    return {"RW": (0.0, None, h, "X"), "EMP": (1.0, None, h, "X"),
            "A2 +位置": (L, None, h, "X"), "A3 +形状": (L, None, h, "G"),
            "A4 +H": (L, PAR[m]["H"], h, "X"),
            "A5 完整": (L, PAR[m]["H"], h, "G")}[nm]


for m in MK:
    CELL[m] = {}
    for h in HOR:
        if (m, h) not in CC:
            continue
        c = CC[(m, h)]; te = c["te"]
        v = {nm: float(loss(c, te, *spec(m, h, nm)).mean()) for nm in AB}
        p95 = {nm: cov(c, te, *spec(m, h, nm), *J95) for nm in ("RW", "EMP", "A5 完整")}
        p80 = cov(c, te, *spec(m, h, "A5 完整"), *J80)
        CELL[m][h] = dict(v=v, p95=p95, p80=p80, n_te=int(te.sum()))
        print(f"{m:>10}{h:>5}{int(te.sum()):>6}" + "".join(f"{v[n]:>10.5f}" for n in AB)
              + f"{(v['A5 完整']/v['EMP']-1)*100:>8.1f}%{(v['A5 完整']/v['RW']-1)*100:>8.1f}%"
              + "".join(f"{p95[n]:>10.3f}" for n in ("RW", "EMP", "A5 完整")) + f"{p80:>8.3f}")

print(f"\n{'='*152}")
print("表 3  消融汇总（25 格）")
print(f"{'='*152}")
print(f"{'步骤':>12}{'胜EMP':>8}{'中位Δ':>9}{'均值Δ':>9}{'h>=60中位':>11}"
      f"{'P95>=.90':>10}{'P80>=.75':>10}{'P95中位':>9}{'最差格':>9}")
for nm in AB:
    dd = [CELL[m][h]["v"][nm] / CELL[m][h]["v"]["EMP"] - 1 for m in MK for h in HOR if h in CELL[m]]
    dl = [CELL[m][h]["v"][nm] / CELL[m][h]["v"]["EMP"] - 1 for m in MK for h in HOR
          if h in CELL[m] and h >= 60]
    p95s = [CELL[m][h]["p95"][nm] for m in MK for h in HOR if h in CELL[m] and nm in CELL[m][h]["p95"]]
    p80s = [CELL[m][h]["p80"] for m in MK for h in HOR if h in CELL[m]]
    n95 = sum(1 for x in p95s if x >= 0.90)
    s95 = (str(n95) + "/" + str(len(dd))) if p95s else "-"
    med95 = ("%.3f" % np.median(p95s)) if p95s else "-"
    print(f"{nm:>12}{sum(1 for x in dd if x<0):>6}/{len(dd):<2}{np.median(dd)*100:>8.1f}%"
          f"{np.mean(dd)*100:>8.1f}%{np.median(dl)*100:>10.1f}%"
          f"{s95:>10}{sum(1 for x in p80s if x>=0.75):>8}/{len(dd)}"
          f"{med95:>9}{max(dd)*100:>8.1f}%")

# ---------- 3. DM ----------
print(f"\n{'='*152}")
print("表 4  DM(HLN, HAC L=10)")
print(f"{'='*152}")
print(f"{'市场':>10}{'h':>5}{'A2vsEMP':>9}{'p':>8}{'A5vsEMP':>9}{'p':>8}{'A5vsRW':>9}{'p':>8}")
RAW = []
for m in MK:
    for h in HOR:
        if h not in CELL[m]:
            continue
        c = CC[(m, h)]; te = c["te"]
        le = loss(c, te, *spec(m, h, "EMP"))
        lr = loss(c, te, *spec(m, h, "RW"))
        l2 = loss(c, te, *spec(m, h, "A2 +位置"))
        l5 = loss(c, te, *spec(m, h, "A5 完整"))
        s2, p2 = dm(l2 - le); s5, p5 = dm(l5 - le); s5r, p5r = dm(l5 - lr)
        RAW.append(dict(m=m, h=h, p2=p2, p5=p5, p5r=p5r,
                        d2=float((l2 - le).mean()), d5=float((l5 - le).mean()),
                        d5r=float((l5 - lr).mean())))
        print(f"{m:>10}{h:>5}{s2:>9.2f}{p2:>8.3f}{s5:>9.2f}{p5:>8.3f}{s5r:>9.2f}{p5r:>8.3f}")


def holm(key):
    fin = sorted([r for r in RAW if np.isfinite(r[key])], key=lambda r: r[key])
    k = len(fin)
    return {(r["m"], r["h"]): min(1.0, r[key] * (k - i)) for i, r in enumerate(fin)}


for tag, key, dk, ref in (("A2 vs EMP", "p2", "d2", "EMP"), ("A5 vs EMP", "p5", "d5", "EMP"),
                          ("A5 vs RW", "p5r", "d5r", "RW")):
    H = holm(key)
    print(f"  {tag}：胜 {sum(1 for r in RAW if r[dk]<0)}/{len(RAW)}，"
          f"未校正显著 {sum(1 for r in RAW if np.isfinite(r[key]) and r[key]<0.05)}/{len(RAW)}，"
          f"Holm 后 {sum(1 for v in H.values() if v<0.05)}/{len(RAW)}")

json.dump(dict(params={m: PAR[m] for m in MK},
               cells={f"{m}|{h}": CELL[m][h] for m in MK for h in HOR if h in CELL.get(m, {})},
               protocol="uncalibrated CRPS; raw PICP; joint (h0,H) fitted on train per market"),
          open(os.path.join(OUT, "COMBO_RESULT.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 COMBO_RESULT.json")
