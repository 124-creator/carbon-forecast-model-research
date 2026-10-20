# -*- coding: utf-8 -*-
"""
Hurst 一致性核验
----------------
目的：解决与 2026-10-06 那篇研报的符号矛盾。
该研报称：广东"显著持续性"(H>0.5)、深圳"显著反持续性"、
全国/湖北/上海"点估计偏离 0.5 但未达显著"。

我用 VR(价格水平) 得到：广州 0.244、上海 0.284（显著反持续）、CEA 0.560（不显著）。

本脚本用 5 种估计量 × 2 个样本期 × 4 个市场，检验：
  (1) 结论是否依赖估计量？
  (2) 是否依赖样本期（试点期 vs 2024+）？
  (3) 收益率的 ACF 符号本身是什么？（这是最不含估计量争议的证据）
"""
import pandas as pd, numpy as np, os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

OUT = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(OUT, "PANEL_5MARKET.csv"), parse_dates=["date"])

FIT = [2, 3, 5, 10, 20, 40]
LG = np.log(np.array(FIT, float))


# ---------- 估计量 1：方差比（价格水平），through-origin 对数回归 ----------
def vr_curve(lv, ks=FIT):
    dv = np.var(np.diff(lv), ddof=1)
    return np.array([np.var(lv[k:] - lv[:-k], ddof=1) / (k * dv) for k in ks])


def h_vr(lv):
    v = vr_curve(lv)
    ok = np.isfinite(v) & (v > 1e-8)
    if ok.sum() < 3:
        return np.nan
    return float((np.sum(LG[ok] * np.log(v[ok])) / np.sum(LG[ok] ** 2) + 1) / 2)


# ---------- 估计量 2：聚合方差法（Aggregated Variance） ----------
def h_av(lv, ks=FIT):
    v = vr_curve(lv, ks)
    ok = np.isfinite(v) & (v > 1e-8)
    if ok.sum() < 3:
        return np.nan
    x = np.log(np.array(ks, float))[ok]
    # log Var(lv[k:]-lv[:-k]) = log(k*dv*VR) 对 log k 回归，斜率 = 2H
    y = np.log(v[ok] * np.array(ks, float)[ok] * np.var(np.diff(lv), ddof=1))
    sl = np.sum(x * (y - y.mean())) / np.sum(x * (x - x.mean()))
    # 允许截距
    A = np.vstack([x, np.ones_like(x)]).T
    sl2 = np.linalg.lstsq(A, y, rcond=None)[0][0]
    return float(sl2 / 2.0)


# ---------- 估计量 3：R/S（重标极差） ----------
def h_rs(x, scales=(8, 16, 32, 64, 128, 256)):
    out_s, out_v = [], []
    for s in scales:
        n = len(x) // s
        if n < 2:
            continue
        vals = []
        for i in range(n):
            seg = x[i * s:(i + 1) * s]
            z = np.cumsum(seg - seg.mean())
            R = z.max() - z.min()
            S = seg.std(ddof=1)
            if S > 1e-12:
                vals.append(R / S)
        if vals:
            out_s.append(s)
            out_v.append(np.mean(vals))
    if len(out_s) < 3:
        return np.nan
    lx, ly = np.log(out_s), np.log(out_v)
    A = np.vstack([lx, np.ones_like(lx)]).T
    return float(np.linalg.lstsq(A, ly, rcond=None)[0][0])


# ---------- 估计量 4：DFA-1 ----------
def h_dfa(x, scales=(8, 16, 32, 64, 128, 256)):
    y = np.cumsum(x - x.mean())
    out_s, out_f = [], []
    for s in scales:
        n = len(y) // s
        if n < 2:
            continue
        seg = y[:n * s].reshape(n, s)
        t = np.arange(s, dtype=float)
        A = np.vstack([t, np.ones(s)]).T
        coef, *_ = np.linalg.lstsq(A, seg.T, rcond=None)
        fit = (A @ coef).T
        out_s.append(s)
        out_f.append(np.sqrt(np.mean((seg - fit) ** 2)))
    if len(out_s) < 3:
        return np.nan
    lx, ly = np.log(out_s), np.log(out_f)
    A2 = np.vstack([lx, np.ones_like(lx)]).T
    return float(np.linalg.lstsq(A2, ly, rcond=None)[0][0])


# ---------- 估计量 5：ACF 和（VR 的解析等价形式） ----------
def acf(x, nlags):
    x = np.asarray(x, float) - np.mean(x)
    v = np.dot(x, x)
    return np.array([np.dot(x[:-k], x[k:]) / v for k in range(1, nlags + 1)])


def acf_sum(r, kmax=40):
    rho = acf(r, kmax)
    return float(rho[1:].sum()), rho


# ==================================================================
print("=" * 108)
print("A. 三种估计量在不同样本期上的一致性 —— 检验是否「结论依赖估计量」")
print("=" * 108)
rows = []
for mk in ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]:
    g = p[p.market == mk].sort_values("date").reset_index(drop=True)
    lp = np.log(g.price.to_numpy(float)).astype(float)
    dt = g.date.to_numpy()
    cut = int(np.searchsorted(dt, np.datetime64("2024-01-01")))
    segs = {"全样本": (0, len(lp)), "试点期(<2024)": (0, cut), "2024+": (cut, len(lp))}
    for nm, (a, b) in segs.items():
        lv = lp[a:b]
        if len(lv) < 80:
            continue
        r = np.diff(lv)
        hv, ha, hr, hd = h_vr(lv), h_av(lv), h_rs(lv), h_dfa(lv)
        rows.append(dict(mk=mk, seg=nm, n=len(lv), H_VR=hv, H_AV=ha, H_RS=hr, H_DFA=hd))
        print(f"{mk:<11} {nm:<14} n={len(lv):>5}  "
              f"H_VR={hv:6.3f}  H_AV={ha:6.3f}  H_RS={hr:6.3f}  H_DFA(水平)={hd:6.3f}  "
              f"{'超扩散' if hv>0.55 else ('次扩散' if hv<0.45 else '≈RW'):>4}")

print()
print("=" * 108)
print("B. 收益率的自相关结构 —— 这是最不含估计量争议的直接证据")
print("   若 sum(rho_1..40) < 0，则收益率反持续、价格次扩散(H<0.5)；>0 则相反")
print("=" * 108)
print(f"{'市场':<11}{'样本期':<14}{'n':>6}{'rho1':>9}{'sum(rho)':>11}{'VR(20)解析':>12}{'VR(20)非参':>12}{'LjungBox p':>12}")
for mk in ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]:
    g = p[p.market == mk].sort_values("date").reset_index(drop=True)
    lp = np.log(g.price.to_numpy(float)).astype(float)
    dt = g.date.to_numpy()
    cut = int(np.searchsorted(dt, np.datetime64("2024-01-01")))
    for nm, (a, b) in {"全样本": (0, len(lp)), "试点期(<2024)": (0, cut), "2024+": (cut, len(lp))}.items():
        lv = lp[a:b]
        if len(lv) < 80:
            continue
        r = np.diff(lv)
        s, rho = acf_sum(r, 40)
        vr_an = 1 + 2 * np.sum((1 - np.arange(1, 21) / 20.0) * rho[:20])
        dv = np.var(r, ddof=1)
        w = lv[20:] - lv[:-20]
        vr_np = np.var(w, ddof=1) / (20 * dv)
        # Ljung-Box Q(20)
        n = len(r)
        Q = n * (n + 2) * np.sum(rho[:20] ** 2 / (n - np.arange(1, 21)))
        from scipy.stats import chi2
        pv = chi2.sf(Q, 20)
        print(f"{mk:<11}{nm:<14}{len(lv):>6}{rho[0]:>9.4f}{s:>11.4f}{vr_an:>12.4f}{vr_np:>12.4f}{pv:>12.4f}")

print()
print("=" * 108)
print("C. 若把 DFA/R-S 作用在收益率上（文献常见做法），H 会变成什么？")
print("   理论关系：H(水平) = H(收益率) + 1")
print("=" * 108)
print(f"{'市场':<11}{'样本期':<14}{'H_DFA(收益率)':>15}{'H_DFA(水平)':>14}{'差+1':>9}")
for mk in ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]:
    g = p[p.market == mk].sort_values("date").reset_index(drop=True)
    lp = np.log(g.price.to_numpy(float)).astype(float)
    dt = g.date.to_numpy()
    cut = int(np.searchsorted(dt, np.datetime64("2024-01-01")))
    for nm, (a, b) in {"全样本": (0, len(lp)), "2024+": (cut, len(lp))}.items():
        lv = lp[a:b]
        if len(lv) < 80:
            continue
        r = np.diff(lv)
        hd_r = h_dfa(r)
        hd_l = h_dfa(lv)
        print(f"{mk:<11}{nm:<14}{hd_r:>15.3f}{hd_l:>14.3f}{hd_r+1-hd_l:>9.3f}")

print()
print("=" * 108)
print("D. 广州/上海 的 H 是否随年份漂移？（滚动 250 日 VR 估计）")
print("=" * 108)
for mk in ["GUANGZHOU", "SHANGHAI", "CEA"]:
    g = p[p.market == mk].sort_values("date").reset_index(drop=True)
    lp = np.log(g.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(g.date.to_numpy())
    W = 250
    yrs, hs = [], []
    for t in range(W, len(lp)):
        w = lp[t - W:t + 1]
        dv = np.var(np.diff(w), ddof=1)
        if dv <= 0:
            continue
        v = np.array([np.var(w[k:] - w[:-k], ddof=1) / (k * dv) for k in FIT])
        ok = np.isfinite(v) & (v > 1e-8)
        if ok.sum() >= 3:
            yrs.append(dt[t].year)
            hs.append((np.sum(LG[ok] * np.log(v[ok])) / np.sum(LG[ok] ** 2) + 1) / 2)
    yrs, hs = np.array(yrs), np.array(hs)
    line = "  ".join(f"{y}:{np.median(hs[yrs==y]):.2f}" for y in sorted(set(yrs)) if (yrs==y).sum() >= 30)
    print(f"{mk:<11} 年度中位H → {line}")

print()
print("=" * 108)
print("E. 结论汇总")
print("=" * 108)
df = pd.DataFrame(rows)
for mk in ["CEA", "GUANGZHOU", "SHANGHAI"]:
    sub = df[df.mk == mk]
    if len(sub) == 0:
        continue
    full = sub[sub.seg == "全样本"]
    if len(full):
        f = full.iloc[0]
        signs = [(">" if f[c] > 0.5 else "<") for c in ["H_VR", "H_AV", "H_RS", "H_DFA"]]
        print(f"{mk:<11} 全样本 4 个估计量符号: {''.join(signs)}   "
              f"值 {f.H_VR:.3f}/{f.H_AV:.3f}/{f.H_RS:.3f}/{f.H_DFA:.3f}")
print()
df.to_csv(os.path.join(OUT, "H_ESTIMATOR_RECONCILE.csv"), index=False, encoding="utf-8-sig")
print("已写出 H_ESTIMATOR_RECONCILE.csv")
