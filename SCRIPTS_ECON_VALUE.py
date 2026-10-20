# -*- coding: utf-8 -*-
"""
经济价值：合规采购的储备决策
=============================
投稿清单第 8 项。不做伪装的覆盖率检验，而是一个真实的决策问题：

  某控排企业须在 t+h 买入 1 单位配额，事后成本 C = exp(L_{t+h})。
  它按预测密度设定资金储备 R（预留现金，占用资本）。
  L(R) = c_c * R + c_p * max(C - R, 0)        c_c 资本占用成本, c_p 短缺惩罚
  最优储备 R* = F^{-1}(1 - c_c/c_p)

取 c_c/c_p = 0.05 → 目标为 95% 分位。指标：
  eco   = L(R_hat) / L(R_oracle) - 1     相对事后最优储备的超额成本（%）
  R_oracle 是测试期内视最优（下界，任何模型都达不到）
  同时报实际超越率（目标 5%）

若 eco 的排序与 CRPS 的排序不同，说明统计精度不等于决策价值——这本身可写。

数据：PANEL_5MARKET.csv   输出：ECON_VALUE.json
"""
import numpy as np, pandas as pd, os, sys, json, warnings
from scipy.stats import norm, t as tdist
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

OUT = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(OUT, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS); JM = 7; J05 = 0; J95 = 14
W = 250; HOR = [20, 60, 120]; STEP = 2
KS = np.arange(0.30, 2.01, 0.01)
C_C, C_P = 0.05, 1.0
MODELS = ["EMP", "EMP0", "SD250", "EW97", "HARh", "MLh"]
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


def daily_feats(lp, t):
    r = np.diff(lp[max(0, t - 60):t + 1])
    if len(r) < 25:
        return None
    v = r ** 2
    return [np.log(v[-1] + 1e-12), np.log(v[-5:].mean() + 1e-12),
            np.log(v[-22:].mean() + 1e-12), np.log(v[-60:].mean() + 1e-12),
            lp[t] - lp[t - 5], lp[t] - lp[t - 22],
            float((np.abs(r) < 1e-12).mean()), np.std(r[-22:], ddof=1)]


def L_of(R, C):
    return C_C * R + C_P * np.maximum(C - R, 0.0)


ALL = {}
for m in CUT:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    cache = {}
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
        cache[t] = dict(sd=float(np.sqrt(v1)), e97=float(max(v, 1e-14)))
    print(f"\n{'='*142}\n{m}   切分 {CUT[m]}  n_te={len(lp)-t0}")
    print(f"{'h':>4}{'n':>6}" + "".join(f"{n:>11}" for n in MODELS)
          + f"{'事后最优':>10}   超越率 5% 目标")
    ALL[m] = {}
    for h in HOR:
        ts_tr = [t for t in range(W + h + 5, t0 - h - 1, STEP) if t in cache]
        ts_te = [t for t in range(max(W + h + 5, t0), len(lp) - h, STEP) if t in cache]
        if len(ts_tr) < 30 or len(ts_te) < 30:
            print(f"{h:>4}  SKIP {len(ts_tr)}/{len(ts_te)}"); continue
        Xh, Yh = [], []
        for t in ts_tr:
            f = daily_feats(lp, t)
            if f is None:
                continue
            Xh.append(f); Yh.append(np.log(max(float(np.sum(np.diff(lp[t:t + h + 1]) ** 2)), 1e-12)))
        Xh = np.array(Xh); Yh = np.array(Yh)
        harh = None; mlh = None
        if len(Yh) > 40:
            harh = np.linalg.lstsq(np.column_stack([np.ones(len(Xh))] + [Xh[:, i] for i in (0, 1, 2)]),
                                   Yh, rcond=None)[0]
            from sklearn.ensemble import HistGradientBoostingRegressor
            mlh = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_depth=3,
                                                min_samples_leaf=15, l2_regularization=1.0, random_state=7)
            mlh.fit(Xh, Yh)

        def mk(ts):
            out = {n: np.full((len(ts), 15), np.nan) for n in MODELS}
            for i, t in enumerate(ts):
                sc = cache[t]; win = lp[max(0, t - W):t]; rr = win[h:] - win[:-h]
                sd = np.std(rr, ddof=1)
                if sd <= 0:
                    continue
                out["EMP"][i] = np.quantile(rr, TAUS)
                out["EMP0"][i] = np.quantile(rr - rr.mean(), TAUS)
                out["SD250"][i] = sc["sd"] * np.sqrt(h) * GV
                out["EW97"][i] = np.sqrt(h * sc["e97"]) * GV
                f = daily_feats(lp, t)
                if f is not None:
                    if harh is not None:
                        lv = harh @ np.array([1.0, f[0], f[1], f[2]])
                        out["HARh"][i] = np.sqrt(np.exp(np.clip(lv, -30, 30))) * GV
                    if mlh is not None:
                        lv = float(mlh.predict(np.array([f]))[0])
                        out["MLh"][i] = np.sqrt(np.exp(np.clip(lv, -30, 30))) * GV
            return out

        A_tr = mk(ts_tr); A_te = mk(ts_te)
        y_tr = lp[np.array(ts_tr) + h] - lp[ts_tr]
        y_te = lp[np.array(ts_te) + h] - lp[ts_te]
        use = [n for n in MODELS if np.isfinite(A_tr[n]).all(1).mean() > 0.95
               and np.isfinite(A_te[n]).all(1).mean() > 0.95]
        gtr = np.all([np.isfinite(A_tr[n]).all(1) for n in use], axis=0)
        gte = np.all([np.isfinite(A_te[n]).all(1) for n in use], axis=0)
        if gtr.sum() < 25 or gte.sum() < 25:
            print(f"{h:>4}  SKIPmask"); continue
        ks = {n: pick_k(y_tr[gtr], A_tr[n][gtr]) for n in use}
        C = np.exp(y_te[gte]); C_tr = np.exp(y_tr[gtr])
        Qk = {n: apply_k(A_te[n][gte], ks[n]) for n in use}
        RH = {n: np.exp(Qk[n][:, 13]) for n in use}       # 各模型 95% 分位储备
        # 两个可实现/理论基准：
        #   朴素 = 训练期 95% 分位作常数储备（无模型、可实现）
        #   floor = 完美预见时只剩资本占用 c_c*E[C]（理论下界，不可达）
        Rbar = float(np.quantile(C_tr, 1 - C_C / C_P))
        L_naive = float(L_of(np.full(len(C), Rbar), C).mean())
        L_floor = float(C_C * C.mean())
        print(f"{h:>4}{int(gte.sum()):>6}" + "".join(
            f"{L_of(RH[n], C).mean():>11.6f}" for n in MODELS) + f"{L_naive:>10.6f}")
        ALL[m][h] = dict(n=int(gte.sum()), naive=L_naive, floor=L_floor,
                         **{n: float(L_of(RH[n], C).mean()) for n in use},
                         **{n + "_cov": float((C > RH[n]).mean()) for n in use},
                         **{n + "_R": float(RH[n].mean()) for n in use})

print(f"\n{'='*142}")
print("经济价值：相对朴素常数储备的成本降幅 eco = L(model)/L(naive) - 1（越低越好）")
print(f"{'市场':<12}{'h':>5}" + "".join(f"{n:>10}" for n in MODELS)
      + f"{'超额率':>9}   实际超越率（目标 5%）")
eco = {n: [] for n in MODELS}; win = {}; CELLS = []
for m in ALL:
    for h in sorted(ALL[m]):
        r = ALL[m][h]
        vals = {n: (r[n] / r["naive"] - 1) * 100 for n in MODELS if n in r}
        if not vals:
            continue
        CELLS.append(vals)
        b = min(vals, key=vals.get); win[b] = win.get(b, 0) + 1
        for n in vals:
            eco[n].append(vals[n])
        line = f"{m:<12}{h:>5}" + "".join(
            f"{vals[n]:>+10.2f}" if n in vals else f"{'-':>10}" for n in MODELS)
        cov = "  ".join(f"{n} {r[n+'_cov']*100:>5.1f}%" for n in vals)
        print(line + "   " + cov)
print(f"\n  获胜计数（经济价值）：{win}")
print(f"  {'模型':<8}{'中位':>10}{'最低':>10}{'最高':>10}{'优于EMP':>10}{'优于SD250':>11}")
for n in MODELS:
    if not eco[n]:
        continue
    v = np.array(eco[n])
    bw = sum(1 for c in CELLS if n in c and "EMP" in c and c[n] < c["EMP"])
    bs = sum(1 for c in CELLS if n in c and "SD250" in c and c[n] < c["SD250"])
    tot = sum(1 for c in CELLS if n in c and "EMP" in c)
    print(f"  {n:<8}{np.median(v):>+9.2f}%{v.min():>+9.2f}%{v.max():>+9.2f}%"
          f"{bw:>7}/{tot:<3}{bs:>8}/{tot}")
json.dump({f"{m}|{h}": v for m in ALL for h, v in ALL[m].items()},
          open(os.path.join(OUT, "ECON_VALUE.json"), "w"), indent=1,
          ensure_ascii=False, default=float)
print("\n已写出 ECON_VALUE.json")
