# -*- coding: utf-8 -*-
"""
实验体系 8/6+：DM 检验的**零假设校准**（把 §24.31 的 p 值全部重算）
=====================================================================
§24.31 之后的 MC 检验（exp7）暴露了一个必须修的问题：
  在**打乱序列**上，DM 检验的 p<0.05 比例是 **0.3211**（名义 0.05），
  "显著负"的发生率是 **0.2556**，KS 检验 p=0.0000 明确拒绝均匀性。
  ⇒ **§24.31.1/§24.31.5 里所有基于渐近 t 分布的 p 值都不可信。**
  原因：h 步收益重叠 ⇒ 损失差序列有强自相关，HAC 带宽 L=10 不够。

修法（三选一，本节全部做，并比较）：
  C1 **渐近 t 分布 + 更宽的 HAC 带宽** L ∈ {10, 20, 40, 80}
  C2 **块自举零假设**：在"两模型损失差均值为 0"的原假设下，
     用**中心化 + 块自举**重算 loss differential 统计量，得到经验 p 值
     （Harvey-Leybourne-Newbold 1987 的 studentized bootstrap）
  C3 **置换检验**：在同一原点上交换两个模型的损失，
     保留自相关结构（这是最保守也最干净的做法）
判据：校准后的检验在打乱数据上 p<0.05 比例应接近 0.05，KS 不应拒绝均匀性。
输出 DM_CALIB.json
"""
import numpy as np, pandas as pd, os, sys, json, warnings, time
from scipy.stats import norm, t as tdist, kstest
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
DIR = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(DIR, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS)
W = 250; STEP = 2
HS = [5, 20, 40]
H0G = [0.1, 0.2, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 500]
HG = np.round(np.arange(0.15, 0.861, 0.05), 3)
MK = ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]
MN = {"CEA": "CEA", "GUANGZHOU": "广州", "SHANGHAI": "上海", "HUBEI": "湖北", "EU": "EU"}
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
NSHUF = 40
SEED0 = 777
BOOT = 2000
BLOCKS = 10


def crps_q(y, Q):
    Qs = np.sort(Q, axis=1)
    ind = (y[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (y[:, None] - Qs)).mean(axis=1)


def dm_asym(d, L=10):
    """渐近 t 分布的 DM（HAC 带宽 L）。"""
    n = len(d)
    if n < 20:
        return np.nan, np.nan
    m_ = d.mean(); dc = d - m_
    lrv = dc @ dc / n
    for l in range(1, L + 1):
        lrv += 2 * (1 - l / (L + 1)) * (dc[l:] @ dc[:-l] / n)
    if lrv <= 0:
        return np.nan, np.nan
    s = m_ / np.sqrt(lrv / n) * np.sqrt((n + 1 - 2 * L + L * (L - 1) / n) / n)
    return float(s), float(2 * (1 - tdist.cdf(abs(s), df = n - 1)))


def _dm_stat(d, L):
    n = len(d)
    m_ = d.mean(); dc = d - m_
    lrv = dc @ dc / n
    for l in range(1, L + 1):
        lrv += 2 * (1 - l / (L + 1)) * (dc[l:] @ dc[:-l] / n)
    if lrv <= 0:
        return np.nan, np.nan
    s = m_ / np.sqrt(lrv / n) * np.sqrt((n + 1 - 2 * L + L * (L - 1) / n) / n)
    return s, lrv


def dm_boot(d, L=10, B=BOOT, block=BLOCKS, rng=None):
    """Harvey-Leybourne-Newbold 的 studentized 块自举零假设 p 值。"""
    rng = rng or np.random.default_rng(0)
    n = len(d)
    if n < 20:
        return np.nan
    s_obs, lrv_obs = _dm_stat(d, L)
    if not np.isfinite(s_obs):
        return np.nan
    dbar = d.mean()
    dc = d - dbar
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(B, nb))
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(B, nb * block)[:, :n]
    dbs = dc[idx] + dbar                       # 保持均值为 0 的零假设样本
    sbs = np.empty(B)
    for b in range(B):
        sb, lv = _dm_stat(dbs[b], L)
        sbs[b] = sb
    sbs = sbs[np.isfinite(sbs)]
    if len(sbs) == 0:
        return np.nan
    return float(np.mean(np.abs(sbs) >= abs(s_obs)))


def dm_perm(d, L=10, B=BOOT, rng=None):
    """置换检验：把 d 的符号随机翻转（保持 |d| 的自相关结构），零假设为均值 0。"""
    rng = rng or np.random.default_rng(0)
    n = len(d)
    if n < 20:
        return np.nan
    s_obs, _ = _dm_stat(d, L)
    if not np.isfinite(s_obs):
        return np.nan
    signs = rng.choice([-1.0, 1.0], size=(B, n))
    cnt = 0; tot = 0
    for b in range(B):
        sb, _ = _dm_stat(d * signs[b], L)
        if np.isfinite(sb):
            tot += 1
            if abs(sb) >= abs(s_obs):
                cnt += 1
    return float(cnt / tot) if tot else np.nan


def build(lp, h, t0):
    rec = []
    for t in range(W + h + 5, len(lp) - h, STEP):
        win = lp[max(0, t - W):t]
        rr = win[h:] - win[:-h]
        rd = np.diff(win)
        sdw = np.std(rr, ddof=1); sdd = np.std(rd, ddof=1)
        if sdw <= 1e-12 or sdd <= 1e-12:
            continue
        rec.append((t, float(rr.mean()), sdw, sdd,
                    np.quantile((rd - rd.mean()) / sdd, TAUS)))
    if len(rec) < 40:
        return None
    ts = np.array([r[0] for r in rec])
    return dict(h=h, ts=ts, mu=np.array([r[1] for r in rec]), sd=np.array([r[2] for r in rec]),
                sdd=np.array([r[3] for r in rec]), xi=np.vstack([r[4] for r in rec]),
                y=lp[ts + h] - lp[ts], tr=(ts <= t0 - h - 2), te=(ts > t0))


def losses(lp, t0):
    """返回逐 (h) 的 (loss_A5, loss_A14)，参数独立重估。"""
    CC = {h: c for h, c in ((h, build(lp, h, t0)) for h in HS) if c is not None and c["tr"].sum() >= 30}
    if not CC:
        return None
    best = (1e9, None, None)
    for Hval in HG:
        for h0 in H0G:
            v = float(np.mean([crps_q(c["y"][c["tr"]],
                                     (c["mu"][c["tr"]] * (h0 / (h0 + h)) +
                                      (c["sdd"][c["tr"]] * h ** Hval))[:, None] * GV).mean()
                                for h, c in CC.items()]))
            if v < best[0]:
                best = (v, h0, Hval)
    _, h0, Hv = best
    out = {}
    for h, c in sorted(CC.items()):
        te = c["te"]
        if te.sum() < 20:
            continue
        y = c["y"][te]; mu = c["mu"][te]; sdd = c["sdd"][te]; xi = c["xi"][te]
        w14 = min(1.0, h / 50.0)
        G = np.tile(GV, (len(y), 1))
        S = sdd * h ** Hv
        l5 = crps_q(y, mu[:, None] * (h0 / (h0 + h)) + S[:, None] * G)
        l14 = crps_q(y, mu[:, None] * (h0 / (h0 + h)) + S[:, None] * ((1 - w14) * xi + w14 * G))
        out[h] = (l5, l14)
    return out


DATA = {}
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    DATA[m] = (lp, t0)

print("=" * 116)
print("真实数据：四种 DM 实现的 p 值")
print("=" * 116)
print(f"{'市场':<8}{'h':>5}{'比值':>9}{'渐近L=10':>11}{'渐近L=40':>11}{'渐近L=80':>11}{'块自举':>10}{'置换':>10}")
REALP = []
for m in MK:
    lp, t0 = DATA[m]
    L = losses(lp, t0)
    for h, (l5, l14) in L.items():
        d = l14 - l5
        pa = [dm_asym(d, LL)[1] for LL in (10, 40, 80)]
        rng = np.random.default_rng(SEED0)
        pb = dm_boot(d, 10, BOOT, BLOCKS, rng)
        pp = dm_perm(d, 10, BOOT, np.random.default_rng(SEED0 + 1))
        REALP.append((m, h, float(l14.mean() / l5.mean()), pa[0], pa[1], pa[2], pb, pp))
        print(f"{MN[m]:<8}{h:>5}{l14.mean()/l5.mean():>9.4f}{pa[0]:>11.3f}{pa[1]:>11.3f}"
              f"{pa[2]:>11.3f}{pb:>10.3f}{pp:>10.3f}")

print("\n" + "=" * 116)
print(f"校准检验：打乱序列 × {NSHUF} 次，统计各 DM 实现在零假设下的 p<0.05 比例")
print("=" * 116)
T0 = time.time()
NULLP = {"a10": [], "a40": [], "a80": [], "boot": [], "perm": []}
for it in range(NSHUF):
    rng = np.random.default_rng(SEED0 + 1000 + it)
    for m in MK:
        lp, t0 = DATA[m]
        inc = np.diff(lp)
        lp_s = np.concatenate([[0.0], np.cumsum(inc[rng.permutation(len(inc))])])
        L = losses(lp_s, t0)
        if not L:
            continue
        for h, (l5, l14) in L.items():
            d = l14 - l5
            NULLP["a10"].append(dm_asym(d, 10)[1])
            NULLP["a40"].append(dm_asym(d, 40)[1])
            NULLP["a80"].append(dm_asym(d, 80)[1])
            NULLP["boot"].append(dm_boot(d, 10, BOOT, BLOCKS, np.random.default_rng(it * 7 + 3)))
            NULLP["perm"].append(dm_perm(d, 10, BOOT, np.random.default_rng(it * 7 + 5)))
    if (it + 1) % 8 == 0:
        print(f"  第 {it+1:>3} 次  ({time.time()-T0:.0f} 秒)")

print()
res_cal = {}
for k, lab in (("a10", "渐近 t, L=10"), ("a40", "渐近 t, L=40"), ("a80", "渐近 t, L=80"),
               ("boot", "块自举 (HLN studentized)"), ("perm", "置换检验")):
    v = np.array([x for x in NULLP[k] if x == x])
    ks = kstest(v, "uniform")
    res_cal[k] = dict(lab=lab, n=int(len(v)), frac05=float(np.mean(v < .05)),
                      frac01=float(np.mean(v < .01)), ks=float(ks.statistic), ksp=float(ks.pvalue))
    print(f"  {lab:<28} n={len(v):<5} p<0.05 比例 {np.mean(v<.05):.4f}   "
          f"p<0.01 比例 {np.mean(v<.01):.4f}   KS {ks.statistic:.4f} (p={ks.pvalue:.4f})")

print("\n" + "=" * 116)
print("用校准最好的实现重算真实数据的胜负")
print("=" * 116)
best_key = min(res_cal, key=lambda k: abs(res_cal[k]["frac05"] - 0.05))
print(f"  校准最好的实现：**{res_cal[best_key]['lab']}**（p<0.05 比例 {res_cal[best_key]['frac05']:.4f}）")
idx = {"a10": 3, "a40": 4, "a80": 5, "boot": 6, "perm": 7}[best_key]
sw = sl = 0
for row in REALP:
    p_ = row[idx]
    if p_ < .05:
        if row[2] < 1:
            sw += 1
        else:
            sl += 1
    print(f"  {MN[row[0]]:<8}h={row[1]:<4}比值 {row[2]:.4f}  p={p_:.3f}  "
          f"{'A14 显著胜' if (p_<.05 and row[2]<1) else ('A14 显著负' if (p_<.05 and row[2]>1) else '')}")
print(f"\n  显著胜 {sw}/15   显著负 {sl}/15")

json.dump({"real": [{"m": a, "h": b, "ratio": c, "a10": d, "a40": e, "a80": f, "boot": g, "perm": h}
                     for a, b, c, d, e, f, g, h in REALP],
           "calibration": res_cal, "best": best_key,
           "null_p": {k: [float(x) for x in v if x == x] for k, v in NULLP.items()}},
          open(os.path.join(DIR, "DM_CALIB.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 DM_CALIB.json")
