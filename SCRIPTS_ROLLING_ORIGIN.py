# -*- coding: utf-8 -*-
"""
滚动原点评估（rolling-origin，逐块再估计）
==========================================
现状：$(h_0,H)$ 用整个训练段一次性估计。这是审稿人必挑的一点。
本脚本改成滚动：每 REFIT 个原点用**当时可得的数据**重估一次 (h0,H)，
再对接下来 REFIT 个原点做预测。

关键防泄漏约束：第 k 块的训练原点 t_tr 必须满足 t_tr + h <= t_block_start，
即训练目标日严格早于该块第一个测试目标日。

同时报告三个量，用来判断增益是否只是"全样本调参"的产物：
  EMP      —— 无参数基准（不受影响）
  A5-static—— 一次性全训练段估计（现状）
  A5-roll  —— 滚动再估计（本脚本新增）
"""
import numpy as np, pandas as pd, os, sys, json, warnings
from scipy.stats import norm
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

OUT = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(OUT, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS)
MK = ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]
HOR = [5, 20, 40, 60, 120]
W, STEP, REFIT = 250, 2, 60
HG = np.array([0.1, 0.5, 2, 5, 15, 40, 80, 120, 250, 1e6])
EG = np.array([0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65])


def crps_mean(y, Q):
    Qs = np.sort(Q, axis=1)
    ind = (y[:, None] < Qs).astype(float)
    return float((2 * (TAUS[None, :] - ind) * (y[:, None] - Qs)).mean())


def build(lp, ts, h):
    """对给定原点集合构造 (mu, sd_h, sd_d, xi, y)。"""
    mu = np.empty(len(ts)); sdh = np.empty(len(ts))
    sdd = np.empty(len(ts)); xi = np.empty((len(ts), len(TAUS)))
    ok = np.ones(len(ts), bool)
    for i, t in enumerate(ts):
        win = lp[t - W:t]
        rr = win[h:] - win[:-h]; rd = np.diff(win)
        a, b = np.std(rr, ddof=1), np.std(rd, ddof=1)
        if a <= 1e-12 or b <= 1e-12:
            ok[i] = False; continue
        mu[i] = rr.mean(); sdh[i] = a; sdd[i] = b
        xi[i] = np.quantile((rr - rr.mean()) / a, TAUS)
    y = lp[np.array(ts) + h] - lp[np.array(ts)]
    return mu[ok], sdh[ok], sdd[ok], xi[ok], y[ok]


def fit_h(Hv, h0, h, mu, sdh, sdd, xi, y):
    L = h0 / (h0 + h)
    Q = mu[:, None] * L + (sdd * h ** Hv)[:, None] * np.tile(GV, (len(mu), 1))
    return crps_mean(y, Q)


print("=" * 146)
print("表 1  滚动原点 vs 一次性估计 vs EMP（CRPS，越低越好）")
print(f"      每 {REFIT} 个原点重估一次 (h0,H)，只用当时可得数据；训练目标日严格早于该块测试起点")
print("=" * 146)

RES = {}
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float))
    dt = pd.to_datetime(d.date).to_numpy()
    cut = "2024-01-01" if m in ("CEA", "GUANGZHOU", "SHANGHAI") else "2023-01-01"
    t0 = int(np.searchsorted(dt, np.datetime64(cut)))
    print(f"\n--- {m}  (len={len(lp)}, t0={t0}) ---")
    print(f"{'h':>4}{'n_te':>6}{'EMP':>10}{'A5static':>10}{'A5roll':>10}"
          f"{'roll/stat':>11}{'roll vs EMP':>13}{'n_refit':>9}{'h0 中位':>10}{'H 中位':>9}")
    for h in HOR:
        start = max(W + h + 5, t0)
        ts_te = list(range(start, len(lp) - h, STEP))
        # 一次性估计：全部 t0 之前的原点
        tr_all = [t for t in range(W + h + 5, t0 - h + 1, STEP)]
        if len(tr_all) < 30:
            print(f"{h:>4}  训练原点不足（{len(tr_all)}），跳过"); continue
        mu, sdh, sdd, xi, y = build(lp, tr_all, h)
        best = (np.inf, None, None)
        for h0 in HG:
            for Hv in EG:
                v = fit_h(Hv, h0, h, mu, sdh, sdd, xi, y)
                if v < best[0]:
                    best = (v, h0, Hv)
        _, h0s, Hs = best

        # 滚动
        Qroll_q, Qemp_q, yroll, ylist = [], [], [], []
        picks = []
        for k in range(0, len(ts_te), REFIT):
            blk = ts_te[k:k + REFIT]
            tstart = blk[0]
            tr = [t for t in range(W + h + 5, tstart - h + 1, STEP)]
            if len(tr) < 30:
                h0b, Hb = h0s, Hs          # 冷启动：退回到一次性估计
            else:
                mub, sbh, sbd, xib, yb = build(lp, tr, h)
                bb = (np.inf, h0s, Hs)
                for h0 in HG:
                    for Hv in EG:
                        v = fit_h(Hv, h0, h, mub, sbh, sbd, xib, yb)
                        if v < bb[0]:
                            bb = (v, h0, Hv)
                _, h0b, Hb = bb
            picks.append((tstart, h0b, Hb))
            mut, sh, sd_, xit, yt = build(lp, blk, h)
            Lt = h0b / (h0b + h)
            Qroll_q.append(mut[:, None] * Lt + (sd_ * h ** Hb)[:, None] * np.tile(GV, (len(mut), 1)))
            Qemp_q.append(mut[:, None] + sh[:, None] * xit)
            ylist.append(yt)
        Y = np.concatenate(ylist)
        Qr = np.vstack(Qroll_q); Qe = np.vstack(Qemp_q)
        c_roll, c_emp = crps_mean(Y, Qr), crps_mean(Y, Qe)
        # static 在同一测试集上的值
        mus, shs, sds, xis, ys = build(lp, ts_te, h)
        Ls = h0s / (h0s + h)
        c_static = crps_mean(ys, mus[:, None] * Ls + (sds * h ** Hs)[:, None] * np.tile(GV, (len(mus), 1)))
        assert np.allclose(Y, ys), "测试目标不一致"
        h0med = float(np.median([x[1] for x in picks if x[1] < 1e5]))
        hmed = float(np.median([x[2] for x in picks]))
        RES[f"{m}|{h}"] = dict(n=len(ys), emp=c_emp, stat=c_static, roll=c_roll,
                               h0s=h0s, Hs=Hs, h0med=h0med, Hmed=hmed,
                               ratio=c_roll / c_static, gain=(c_roll / c_emp - 1) * 100)
        print(f"{h:>4}{len(ys):>6}{c_emp:>10.5f}{c_static:>10.5f}{c_roll:>10.5f}"
              f"{(c_roll/c_static-1)*100:>10.1f}%{(c_roll/c_emp-1)*100:>12.1f}%"
              f"{len(picks):>9}{h0med:>10.3g}{hmed:>9.2f}")

print("\n" + "=" * 146)
print("表 2  汇总：滚动再估计是否改变了结论")
print("=" * 146)
for lab, key in (("A5-roll  vs EMP", "gain"), ("A5-static vs EMP", None)):
    rows = []
    for h in HOR:
        ks = [f"{m}|{h}" for m in MK if f"{m}|{h}" in RES]
        if not ks:
            continue
        if key:
            dd = [RES[k][key] for k in ks]
        else:
            dd = [(RES[k]["stat"] / RES[k]["emp"] - 1) * 100 for k in ks]
        rows.append((h, float(np.median(dd)), sum(1 for x in dd if x < 0), len(dd)))
    print(f"\n  {lab}")
    print(f"  {'h':>4}{'中位':>10}{'胜格':>10}")
    for h, md, w, n in rows:
        print(f"  {h:>4}{md:>9.1f}%{w:>7}/{n}")

rr = [RES[k]["ratio"] for k in RES]
print(f"\n  滚动/一次性 比值：中位 {np.median(rr):.4f}，"
      f"滚动更差 {sum(1 for x in rr if x>1)}/{len(rr)}，滚动更好 {sum(1 for x in rr if x<1)}/{len(rr)}")
print(f"  滚动相对 EMP 的胜格：{sum(1 for k in RES if RES[k]['gain']<0)}/{len(RES)}")

json.dump(RES, open(os.path.join(OUT, "ROLLING_ORIGIN.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 ROLLING_ORIGIN.json")
