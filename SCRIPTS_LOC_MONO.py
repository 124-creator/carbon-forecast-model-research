# -*- coding: utf-8 -*-
"""
位置块单调性订正：A5 的 lambda(h) 随 h 递减，但数据显示应当递增
================================================================
证据（主消融表，未校准 CRPS）：
  h=5   RW(=无漂移) 赢 A5 四格：CEA .01767<.01830  GZ .02553<.02639
                                  SH .02228<.02283   EU .02466<.02484
  h=120 A5 赢 RW 四格：          CEA .11678<.14229  GZ .09625<.11027
                                  SH .06774<.08203   HUBEI .05215<.06479
=> 短端要 lambda~0，长端要 lambda>0。而 A5 的 h0/(h0+h) 恰好相反。

理论理由：mu_hat 的 t 统计量 ∝ sqrt(h)，精度加权的收缩因子
  lambda = t^2/(1+t^2) = mu^2/(mu^2 + sigma^2/h)
随 h 单调上升。A5 的形式没有这个性质。

四个规格（其余部分与 A5 完全相同：sigma_d 用 250 日、Gaussian 形状、同一切分与原点）：
  A5   lambda = h0/(h0+h)                       递减，2 参数/市场（既有，作为对照）
  A10  lambda = h/(h+h0)                        递增，2 参数/市场
  A9   lambda = h^a/(h^a+h0^a)                  递增，3 参数/市场
  A11  lambda_t = mu^2/(mu^2+sigma_d^2/h)       逐原点精度加权，0 参数/市场

全部参数只在训练段拟合。输出 LOC_MONO.json
"""
import numpy as np, pandas as pd, os, sys, json, warnings
from scipy.stats import norm, t as tdist
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

OUT = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(OUT, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS); JM = 7
W = 250; HOR = [5, 20, 40, 60, 120]; STEP = 2
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
PAR = {"CEA": (0.1, 0.60), "GUANGZHOU": (30, 0.35), "SHANGHAI": (0.1, 0.30),
       "HUBEI": (3, 0.35), "EU": (80, 0.45)}
HG = np.round(np.concatenate([[0.1], np.geomspace(0.5, 400, 11)]), 3)
AG = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
HGRID = np.round(np.arange(0.20, 0.66, 0.05), 2)


def crps_q(y, Q):
    Qs = np.sort(Q, axis=1)
    ind = (y[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (y[:, None] - Qs)).mean(axis=1)


RES = {}
FIT = {}
for m in CUT:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    print(f"\n{'='*150}\n{m}  n_tr={t0} n_te={len(lp)-t0}")

    def bases(ts, h):
        mu, sd, yy, ok = [], [], [], []
        for t in ts:
            win = lp[max(0, t - W):t]
            if len(win) < 120:
                mu.append(np.nan); sd.append(np.nan); yy.append(np.nan)
                ok.append(False); continue
            rr = win[h:] - win[:-h]
            rd = np.diff(win)
            if len(rr) < 40 or np.std(rd, ddof=1) <= 0:
                mu.append(np.nan); sd.append(np.nan); yy.append(np.nan)
                ok.append(False); continue
            mu.append(rr.mean()); sd.append(np.std(rd, ddof=1))
            yy.append(lp[t + h] - lp[t]); ok.append(True)
        return (np.array(mu), np.array(sd), np.array(yy), np.array(ok))

    TR, TE = {}, {}
    for h in HOR:
        ts_tr = [t for t in range(W + h + 5, t0 - h - 1, STEP)]
        ts_te = [t for t in range(max(W + h + 5, t0), len(lp) - h, STEP)]
        assert len(set(ts_tr) & set(ts_te)) == 0
        TR[h] = (ts_tr,) + bases(ts_tr, h)
        TE[h] = (ts_te,) + bases(ts_te, h)

    def pred(B, h, lam, Hv):
        mu, sd = B[1], B[2]
        s = sd * (h ** Hv)
        return lam[:, None] * mu[:, None] + s[:, None] * GV[None, :]

    def fit_forms():
        best = {}
        # A5 既有形式：递减
        for h0 in HG:
            for Hv in HGRID:
                tot = 0.0
                for h in HOR:
                    B = TR[h]
                    lam = np.full(len(B[1]), h0 / (h0 + h))
                    k_ = B[4]; tot += crps_q(B[3][k_], pred(B, h, lam, Hv)[k_]).mean() * k_.sum()
                k = ("A5",)
                if k not in best or tot < best[k][0]:
                    best[k] = (tot, h0, None, Hv)
        # A10 / A9：递增
        for a in AG:
            for h0 in HG:
                for Hv in HGRID:
                    tot = 0.0
                    for h in HOR:
                        B = TR[h]
                        lam = np.full(len(B[1]), h ** a / (h ** a + h0 ** a))
                        k_ = B[4]; tot += crps_q(B[3][k_], pred(B, h, lam, Hv)[k_]).mean() * k_.sum()
                    k = ("A10",) if a == 1.0 else ("A9",)
                    if k not in best or tot < best[k][0]:
                        best[k] = (tot, h0, a, Hv)
        # A11：逐原点精度加权，零参数；只选 H
        for Hv in HGRID:
            tot = 0.0
            for h in HOR:
                B = TR[h]
                mu, sd = B[1], B[2]
                lam = np.clip(mu ** 2 / (mu ** 2 + sd ** 2 / h + 1e-18), 0.0, 1.0)
                k_ = B[4]; tot += crps_q(B[3][k_], pred(B, h, lam, Hv)[k_]).mean() * k_.sum()
            k = ("A11",)
            if k not in best or tot < best[k][0]:
                best[k] = (tot, None, None, Hv)
        return best

    B_ = fit_forms()
    FIT[m] = {k[0]: dict(h0=v[1], a=v[2], H=v[3], train_obj=v[0]) for k, v in B_.items()}
    for k in ("A5", "A10", "A9", "A11"):
        v = FIT[m][k]
        print(f"  {k:<4} h0={str(v['h0']):>6} a={str(v['a']):>4} H={v['H']:.2f}  "
              f"train={v['train_obj']:.4f}")

    RES[m] = {}
    for h in HOR:
        ts_te, mu, sd, y, ok = TE[h]
        lam5 = np.full(len(mu), PAR[m][0] / (PAR[m][0] + h))
        lam10 = np.full(len(mu), h / (h + FIT[m]["A10"]["h0"]))
        lam9 = np.full(len(mu), h ** FIT[m]["A9"]["a"] /
                       (h ** FIT[m]["A9"]["a"] + FIT[m]["A9"]["h0"] ** FIT[m]["A9"]["a"]))
        lam11 = np.clip(mu ** 2 / (mu ** 2 + sd ** 2 / h + 1e-18), 0.0, 1.0)
        lam0 = np.zeros(len(mu))
        out = {}
        for nm, lam, Hv in (("A5", lam5, PAR[m][1]), ("A10", lam10, FIT[m]["A10"]["H"]),
                            ("A9", lam9, FIT[m]["A9"]["H"]),
                            ("A11", lam11, FIT[m]["A11"]["H"]),
                            ("RW", lam0, PAR[m][1])):
            Q = pred((0, mu, sd, y, ok), h, lam, Hv)
            out[nm] = float(crps_q(y[ok], Q[ok]).mean())
        out["n"] = int(ok.sum())
        out["lam_med"] = {nm: float(np.median(lam))
                          for nm, lam in (("A5", lam5), ("A10", lam10), ("A9", lam9), ("A11", lam11))}
        RES[m][h] = out
        print(f"{h:>4}{out['n']:>6}  A5 {out['A5']:.5f}  A10 {out['A10']:.5f}  "
              f"A9 {out['A9']:.5f}  A11 {out['A11']:.5f}  RW {out['RW']:.5f}   "
              f"中位lam A5 {out['lam_med']['A5']:.3f} A10 {out['lam_med']['A10']:.3f} "
              f"A11 {out['lam_med']['A11']:.3f}")

json.dump(dict(res={m: {str(h): RES[m][h] for h in RES[m]} for m in RES}, fit=FIT),
          open(os.path.join(OUT, "LOC_MONO.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 LOC_MONO.json")
