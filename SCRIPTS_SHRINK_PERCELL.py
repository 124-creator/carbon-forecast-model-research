# -*- coding: utf-8 -*-
"""
期限依赖的收缩估计量（机制的直接推论）
=====================================
机制 n_eff = W/h 的最锐利预言不是"日频比窗口好"，而是**该用哪一端由 h 决定**：

  sigma2_daily,h = w(h)*s2_cond + (1-w(h))*s2_long,   w(h) = n_eff/(n_eff + c)
  n_eff = W/h,   sigma_h = sqrt(h * sigma2_daily,h)

  c -> inf  退化 SD250（纯长期均值）
  c -> 0    退化 EW97 （纯条件方差）

三个可检验问题：
  Q1 c 的估计值是否在内部（不是端点）？若是，插值有实质内容。
  Q2 w(h) 是否随 h 单调下降？这是 n_eff 预言的方向。
  Q3 SHRINK 是否在 CRPS 上同时优于两个端点？若只优于一个，机制无内容。

同时跑一个自由形式对照 FREE：w = sigmoid(a + b*log(n_eff))，2 个参数，
不预设 w 的函数形式——若 FREE 估出的 b > 0（w 随 n_eff 上升，即随 h 下降）
且拟合出的 w 与 SHRINK 接近，则机制的函数形式不是硬凑的。

每个模型一个校准标量 k，训练期定，测试期固定。
数据：PANEL_5MARKET.csv   输出：SHRINK_NEFF.json
"""
import numpy as np, pandas as pd, os, sys, json, warnings
from scipy.stats import norm, t as tdist
from scipy.optimize import minimize
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

OUT = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(OUT, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS); JM = 7; J05 = 0; J95 = 14
W = 250; HOR = [5, 20, 60, 120]; STEP = 2
KS = np.arange(0.30, 2.01, 0.01)
CVALS = np.array([0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 150, 400, 2000])
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}


def pick_k(y, Q):
    med = Q[:, JM][:, None]
    cov = np.array([((y >= med[:, 0] + k * (Q[:, J05] - med[:, 0])) &
                     (y <= med[:, 0] + k * (Q[:, J95] - med[:, 0]))).mean() for k in KS])
    return float(KS[int(np.argmin(np.abs(cov - 0.95)))])


def apply_k(Q, k):
    med = Q[:, JM][:, None]
    return med + k * (Q - med)


def crps_q(y, Q, taus=TAUS):
    d = y[:, None] - Q
    return float(np.mean(2.0 * (taus[None, :] - (d < 0).astype(float)) * d))


def dm(d, L=10):
    n = len(d)
    m = d.mean(); dc = d - m
    lrv = dc @ dc / n
    for l in range(1, L + 1):
        lrv += 2 * (1 - l / (L + 1)) * (dc[l:] @ dc[:-l] / n)
    if lrv <= 0:
        return np.nan, np.nan
    s = m / np.sqrt(lrv / n) * np.sqrt((n + 1 - 2 * L + L * (L - 1) / n) / n)
    return float(s), float(2 * (1 - tdist.cdf(abs(s), df=n - 1)))


RES = {}
print(f"{'='*140}")
print("Q1/Q2/Q3：期限依赖收缩  sigma2 = w*s2_cond + (1-w)*s2_long, w = n_eff/(n_eff+c)")
print(f"{'='*140}")
for m in CUT:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    st = {}
    for t in range(W + 5, len(lp)):
        win = lp[max(0, t - W):t]; r = np.diff(win)
        if len(r) < 60:
            continue
        v1 = np.var(r, ddof=1)
        if v1 <= 0:
            continue
        v = r[0] ** 2
        for i in range(1, len(r)):
            v = 0.97 * v + 0.03 * r[i - 1] ** 2
        st[t] = (float(v1), float(max(v, 1e-14)))
    RES[m] = {}
    print(f"\n{m}   切分 {CUT[m]}")
    for h in HOR:
        neff = W / h
        ts_tr = [t for t in range(W + h + 5, t0 - h - 1, STEP) if t in st]
        ts_te = [t for t in range(max(W + h + 5, t0), len(lp) - h, STEP) if t in st]
        if len(ts_tr) < 30 or len(ts_te) < 30:
            print(f"  h={h}  SKIP {len(ts_tr)}/{len(ts_te)}"); continue
        y_tr = lp[np.array(ts_tr) + h] - lp[ts_tr]
        y_te = lp[np.array(ts_te) + h] - lp[ts_te]
        s_tr = np.array([st[t] for t in ts_tr]); s_te = np.array([st[t] for t in ts_te])

        def mkQ(sd, hs):
            return np.sqrt(hs * sd[:, None]) * GV[None, :]

        def run(v2_d, y):
            Q = mkQ(v2_d, h)
            k = pick_k(y, Q)
            return Q, k

        cand = {}
        # 两个端点
        Q, k = run(s_tr[:, 0], y_tr); cand["LONG"] = (Q, k, s_te[:, 0])
        Q, k = run(s_tr[:, 1], y_tr); cand["COND"] = (Q, k, s_te[:, 1])
        # 收缩族：c 在训练期 CRPS 上选
        best = (None, 1e18, None)
        for c in CVALS:
            wgt = neff / (neff + c)
            v2 = wgt * s_tr[:, 1] + (1 - wgt) * s_tr[:, 0]
            Q, k = run(v2, y_tr)
            cr = crps_q(y_tr, apply_k(Q, k))
            cand[f"S{c:g}"] = (Q, k, wgt * s_te[:, 1] + (1 - wgt) * s_te[:, 0])
            if cr < best[1]:
                best = (f"S{c:g}", cr, c)
        cbest = best[2]; wbest = neff / (neff + cbest)
        # 自由形式：w = sigmoid(a + b*log(neff))，2 参数，训练期最小化 CRPS
        def obj(th):
            a, b = th
            wgt = 1.0 / (1.0 + np.exp(-(a + b * np.log(neff))))
            v2 = wgt * s_tr[:, 1] + (1 - wgt) * s_tr[:, 0]
            Q, k = run(v2, y_tr)
            return crps_q(y_tr, apply_k(Q, k))
        r_ = minimize(obj, np.array([0.0, 0.5]), method="Nelder-Mead",
                      options=dict(maxiter=300, xatol=1e-3, fatol=1e-9))
        af, bf = r_.x
        wf = 1.0 / (1.0 + np.exp(-(af + bf * np.log(neff))))
        v2f_tr = wf * s_tr[:, 1] + (1 - wf) * s_tr[:, 0]
        Qf, kf = run(v2f_tr, y_tr)
        v2f_te = wf * s_te[:, 1] + (1 - wf) * s_te[:, 0]

        out = {}
        for n, (Q, k, v2_te) in cand.items():
            Qt = np.sqrt(h * v2_te[:, None]) * GV[None, :]
            Qt = apply_k(Qt, k)
            out[n] = crps_q(y_te, Qt)
        Qft = apply_k(np.sqrt(h * v2f_te[:, None]) * GV[None, :], kf)
        out["FREE"] = crps_q(y_te, Qft)
        out["_c"] = cbest; out["_w"] = wbest
        out["_af"] = float(af); out["_bf"] = float(bf); out["_wf"] = float(wf)
        out["_neff"] = neff
        out["_n"] = int(len(y_te))
        RES[m][h] = out
        print(f"  h={h:>3}  n_eff={neff:>5.2f}  c*={cbest:>6g}  w*(h)={wbest:>6.3f}  |  "
              f"FREE b={bf:>+6.3f} w={wf:>6.3f}  |  "
              f"LONG {out['LONG']:.5f}  COND {out['COND']:.5f}  "
              f"SHRINK {out[f'S{cbest:g}']:.5f}  FREE {out['FREE']:.5f}")

print(f"\n{'='*140}")
print("Q1：c* 是否在内部？（端点分别为 LONG=2000 与 COND=0.5）")
print(f"{'市场':<11}{'h':>5}{'n_eff':>7}{'c*':>8}{'w*':>7}   判定")
for m in RES:
    for h in sorted(RES[m]):
        r = RES[m][h]
        if r["_c"] >= 2000:
            tag = "顶到 LONG 端 → 机制无内容"
        elif r["_c"] <= 0.5:
            tag = "顶到 COND 端 → 机制无内容"
        else:
            tag = "**内部解**"
        print(f"{m:<11}{h:>5}{r['_neff']:>7.2f}{r['_c']:>8g}{r['_w']:>7.3f}   {tag}")

print(f"\nQ2：估计的 w(h) 是否随 h 下降？")
print(f"{'市场':<11}" + "".join(f"{'h='+str(h):>10}" for h in HOR) + "   方向")
for m in RES:
    ws = [RES[m][h]["_w"] for h in HOR if h in RES[m]]
    mono = all(ws[i] >= ws[i + 1] - 1e-9 for i in range(len(ws) - 1))
    print(f"{m:<11}" + "".join(f"{v:>10.3f}" for v in ws) +
          ("   **单调下降**" if mono else "   非单调"))

print(f"\nQ3：SHRINK 是否同时优于两个端点？（测试期 CRPS，负=更好）")
print(f"{'市场':<11}{'h':>5}{'c*':>7}{'SHRINK':>10}{'vs LONG':>10}{'vs COND':>10}{'FREE':>10}")
n_both = 0; tot = 0; n_free_b = 0
for m in RES:
    for h in sorted(RES[m]):
        r = RES[m][h]
        s = r[f"S{r['_c']:g}"]
        a = (s / r["LONG"] - 1) * 100; b = (s / r["COND"] - 1) * 100
        tot += 1; n_both += (a < 0 and b < 0)
        if r["_bf"] > 0:
            n_free_b += 1
        print(f"{m:<11}{h:>5}{r['_c']:>7g}{s:>10.5f}{a:>+9.2f}%{b:>+9.2f}%{r['FREE']:>10.5f}")
print(f"\n  同时优于两端：{n_both}/{tot}     FREE 的 b>0：{n_free_b}/{tot}")

json.dump({f"{m}|{h}": v for m in RES for h, v in RES[m].items()},
          open(os.path.join(OUT, "SHRINK_NEFF.json"), "w"), indent=1,
          ensure_ascii=False, default=float)
print("\n已写出 SHRINK_NEFF.json")
