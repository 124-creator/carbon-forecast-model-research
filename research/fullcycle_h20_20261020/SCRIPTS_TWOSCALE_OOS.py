# -*- coding: utf-8 -*-
"""
双分量期限结构的样本外检验
========================
模型族： VR(h) = w*VR_fast(h;phi) + (1-w)*h^(2H-1)
        VR_fast(h;phi) = 1 + 2*sum_{d=1}^{h-1}(1-d/h)*phi^d

对比四个尺度律（位置统一用 RW，mu=0，只让尺度不同）：
  A  根号h          sigma_h = sig*sqrt(h)
  B  单分量幂律     sigma_h = sig*h^H
  C  双分量         sigma_h = sig*sqrt(h*(w*VRfast+(1-w)*h^(2H-1)))
  D  滚动非参数VR   sigma_h = sig*sqrt(h*VR_window(h))

关键：参数全部只用窗口内 h<=40 的 VR 估计（真实可得），
      然后外推到 h=60/120/250。这是真正的期限外推预测。

拟合技巧：给定 (phi,H)，VR 对 w 是线性的 -> w 闭式解，只需二维网格。
"""
import pandas as pd, numpy as np, os, sys, json, warnings
from scipy.stats import norm
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

OUT = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(OUT, "PANEL_5MARKET.csv"), parse_dates=["date"])

W = 250
FITH = np.array([2, 3, 5, 10, 20, 40])
HOR = [5, 20, 40, 60, 120, 250]
LOG_FITH = np.log(FITH)

PHI_G = np.linspace(-0.90, 0.95, 38)
H_G = np.linspace(0.10, 0.90, 33)
PHI_M, H_M = np.meshgrid(PHI_G, H_G, indexing="ij")
PHI_F = PHI_M.ravel(); H_F = H_M.ravel()


def vr_fast_mat(h, phis):
    """VR_fast(h;phi) for a vector of phi"""
    d = np.arange(1, h, dtype=float)
    wig = 1.0 - d / h
    return 1.0 + 2.0 * (wig * (phis[:, None] ** d[None, :])).sum(axis=1)


def crps_n(s, y):
    z = y / s
    return s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def fit_window(win):
    """返回 (H_single, w, phi, H2)。win 为价格水平（对数）"""
    dv = np.var(np.diff(win), ddof=1)
    if dv <= 0:
        return np.nan, np.nan, np.nan, np.nan
    vr = np.array([np.var(win[k:] - win[:-k], ddof=1) / (k * dv) for k in FITH])
    ok = np.isfinite(vr) & (vr > 1e-8)
    if ok.sum() < 4:
        return np.nan, np.nan, np.nan, np.nan
    # 单分量
    Hs = (np.sum(LOG_FITH[ok] * np.log(vr[ok])) / np.sum(LOG_FITH[ok] ** 2) + 1) / 2
    # 双分量：构建 fast 矩阵 (nF, n_ok)
    y = vr[ok]
    fast = np.column_stack([vr_fast_mat(int(h), PHI_F) for h in FITH[ok]])  # (nF, nok)
    slow = np.exp(np.outer(2 * H_F - 1, np.log(FITH[ok].astype(float))))     # (nF, nok)
    d = fast - slow
    num = ((y[None, :] - slow) * d).sum(axis=1)
    den = (d * d).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        wv = np.where(den > 1e-14, num / den, np.nan)
    wv = np.clip(wv, 0.0, 1.0)
    pred = slow + wv[:, None] * d
    rss = ((np.log(y)[None, :] - np.log(np.maximum(pred, 1e-10))) ** 2).sum(axis=1)
    j = int(np.nanargmin(rss))
    return float(Hs), float(wv[j]), float(PHI_F[j]), float(H_F[j])


def run(mk, test_start):
    g = p[p.market == mk].sort_values("date").reset_index(drop=True)
    lp = np.log(g.price.to_numpy(float)).astype(float)
    dt = g.date.to_numpy()
    n = len(lp)
    t0 = int(np.searchsorted(dt, np.datetime64(test_start))) if test_start is not None else int(0.75 * n)
    # 收集所有 origin 的拟合参数（滚动）
    rec = {}
    for t in range(W, n - 1):
        Hs, w, phi, H2 = fit_window(lp[t - W + 1:t + 1])
        rec[t] = (Hs, w, phi, H2)
    test_orig = [t for t in rec if t >= t0]
    if len(test_orig) < 60:
        return None
    res = {}
    for h in HOR:
        rows = []
        for t in test_orig:
            if t + h >= n:
                continue
            Hs, w, phi, H2 = rec[t]
            if not np.isfinite(Hs):
                continue
            win = lp[t - W + 1:t + 1]
            rw = np.diff(win)
            sig = float(np.std(rw, ddof=1))
            if sig <= 0:
                continue
            vrn = float(np.var(win[h:] - win[:-h], ddof=1) / (h * np.var(rw, ddof=1))) if h < W - 5 else np.nan
            y = float(lp[t + h] - lp[t])
            vfast = float(vr_fast_mat(h, np.array([phi]))[0])
            vslow = h ** (2 * H2 - 1)
            vr2 = w * vfast + (1 - w) * vslow
            rows.append((t, y, sig, Hs, vr2, vrn))
        if len(rows) < 40:
            continue
        A = np.array(rows, float)
        ts, ys, sigs, Hss, vr2s, vrns = A.T
        istr = ts < t0
        out = {}
        for nm, sh in [("A_sqrt_h", np.sqrt(h) * sigs),
                       ("B_power", h ** Hss * sigs),
                       ("C_two", np.sqrt(h * np.maximum(vr2s, 1e-6)) * sigs),
                       ("D_rolling", np.where(np.isfinite(vrns), np.sqrt(h * vrns) * sigs, np.nan))]:
            if istr.sum() < 30:
                lam = 1.0
            else:
                grid = np.linspace(0.5, 3.0, 26)
                cs = [np.nanmean(crps_n(np.maximum(gv * sh[istr], 1e-9), ys[istr])) for gv in grid]
                lam = float(grid[int(np.nanargmin(cs))])
            sc = np.maximum(lam * sh, 1e-9)
            testm = np.isfinite(sc)
            if testm.sum() < 40:
                out[nm] = (np.nan, lam, int(testm.sum()), sc, ys, ts)
                continue
            out[nm] = (float(np.mean(crps_n(sc[testm], ys[testm]))), lam, int(testm.sum()), sc, ys, ts)
        res[h] = out
    return dict(rec=rec, test_orig=test_orig, res=res, lp=lp, dt=dt)


CFG = [("CEA", "2024-01-01"), ("GUANGZHOU", "2024-01-01"), ("SHANGHAI", "2024-01-01"),
       ("HUBEI", "2023-01-01"), ("EU", "2023-01-01")]
ALL = {}
for mk, ts in CFG:
    print(f"\n{'='*104}\n{mk}  测试起点 {ts}\n{'='*104}")
    r = run(mk, ts)
    if r is None:
        print("  样本不足，跳过")
        continue
    ALL[mk] = {"test_start": ts, "table": {}}
    print(f"{'h':>5}{'n_test':>8}{'A 根号h':>11}{'B 单分量':>11}{'C 双分量':>11}{'D 滚动VR':>11}"
          f"{'C vs A':>10}{'C vs B':>10}{'C vs D':>10}")
    for h in HOR:
        if h not in r["res"]:
            continue
        o = r["res"][h]
        va = o["A_sqrt_h"][0]; vb = o["B_power"][0]; vc = o["C_two"][0]; vd = o["D_rolling"][0]
        nt = o["C_two"][2]
        def pc(x, ref):
            return f"{(x/ref-1)*100:+8.2f}%" if np.isfinite(x) and np.isfinite(ref) and ref > 0 else f"{'--':>9}"
        print(f"{h:>5}{nt:>8}{va:>11.5f}{vb:>11.5f}{vc:>11.5f}"
              f"{(vd if np.isfinite(vd) else float('nan')):>11.5f}"
              f"{pc(vc,va):>10}{pc(vc,vb):>10}{pc(vc,vd):>10}")
        ALL[mk]["table"][h] = {"A": va, "B": vb, "C": vc, "D": vd, "n": nt,
                               "lam": {k: o[k][1] for k in o}}
    # 参数漂移
    rec = r["rec"]; tot = sorted(rec)
    Hs = np.array([rec[t][0] for t in tot]); ws = np.array([rec[t][1] for t in tot])
    ph = np.array([rec[t][2] for t in tot]); H2 = np.array([rec[t][3] for t in tot])
    print(f"\n  窗口参数（全期滚动中位）：H单={np.nanmedian(Hs):.3f}  w={np.nanmedian(ws):.3f}  "
          f"phi={np.nanmedian(ph):.3f}  H双={np.nanmedian(H2):.3f}")

print(f"\n{'='*104}\n外推检验：只用 h<=40 拟合，预测 VR(250)，与原始估计比较\n{'='*104}")
print(f"{'市场':<11}{'VR(250)原始':>13}{'双分量预测':>12}{'单分量预测':>12}{'自助95%CI':>22}")
EXTR = {}
for mk, ts in CFG:
    g = p[p.market == mk].sort_values("date").reset_index(drop=True)
    lp = np.log(g.price.to_numpy(float)).astype(float)
    if len(lp) < W + 260:
        print(f"{mk:<11}{'样本不足':>13}")
        continue
    r = ALL.get(mk)
    if r is None:
        continue
    h = 250
    sub = [(t, *fit_window(lp[t - W + 1:t + 1])) for t in range(W, len(lp) - h, 5)]
    sub = [s for s in sub if np.isfinite(s[1])]
    if not sub:
        continue
    # 原始 VR(250)
    dv = np.var(np.diff(lp), ddof=1)
    vr_raw = float(np.var(lp[h:] - lp[:-h], ddof=1) / (h * dv))
    rng = np.random.default_rng(7)
    rr = np.diff(lp); bs = []
    for _ in range(400):
        st = rng.integers(0, max(1, len(rr) - 20), size=int(np.ceil(len(rr) / 20)))
        sx = np.concatenate([rr[s:s + 20] for s in st])[:len(rr)]
        lvb = np.concatenate([[0.0], np.cumsum(sx)])
        if len(lvb) > h + 10:
            dd = np.var(np.diff(lvb), ddof=1)
            if dd > 0:
                bs.append(np.var(lvb[h:] - lvb[:-h], ddof=1) / (h * dd))
    lo, hi = (np.quantile(bs, [0.025, 0.975]) if len(bs) > 50 else (np.nan, np.nan))
    mw = np.median([s[2] for s in sub]); mph = np.median([s[3] for s in sub]); mH2 = np.median([s[4] for s in sub])
    pred2 = mw * float(vr_fast_mat(h, np.array([mph]))[0]) + (1 - mw) * h ** (2 * mH2 - 1)
    mHs = np.median([s[1] for s in sub]); pred1 = h ** (2 * mHs - 1)
    EXTR[mk] = dict(vr_raw=vr_raw, pred2=pred2, pred1=pred1, lo=float(lo), hi=float(hi),
                    w=mw, phi=mph, H2=mH2, Hs=mHs)
    inb = "在CI内" if (lo <= pred2 <= hi) else "在CI外"
    print(f"{mk:<11}{vr_raw:>13.3f}{pred2:>12.3f}{pred1:>12.3f}   [{lo:>5.2f},{hi:>5.2f}] {inb}")

json.dump({"oos": ALL, "extrap": EXTR}, open(os.path.join(OUT, "TWOSCALE_OOS.json"), "w"),
          indent=1, ensure_ascii=False, default=float)
print("\n已写出 TWOSCALE_OOS.json")
