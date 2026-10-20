# -*- coding: utf-8 -*-
"""
DAT：日频锚定期限结构密度 —— 合并实验 + MCS
============================================
九轮下来各零件分别被验证/否决，但从未组装成一个模型比较：
  尺度  无条件日标准差 * sqrt(h)          (scale2.py: 20/25 胜 EMP, 15/25 显著)
  位置  日均收益 * h   vs   h 期经验均值   (jats3 显示二者在 CEA h=120 差 3.4%)
  形状  长 h 用 Gaussian                  (SHAPE_ANNEAL: h>=40 高斯胜 12/15)
  跳跃  仅在 CEA(0.8%)/EU(0.4%) 存在      (JATS_DECOMP: 形状贡献 -0.06%~-3.2%)

候选模型 DAT:  r_h ~ N(mu_d * h, (sd_d * sqrt(h))^2)
   mu_d, sd_d 为窗口内日收益的均值与标准差 —— 全部参数在日频估计(数千观测)，
   不依赖 h 期重叠窗口(n_eff = W/h)，不估 H，不估 VR 曲线。

基准：EMP(经验分位) / GAU(经验均值+经验sd) / SD_SQ(h期均值+日sd) / EW97(条件日波)
      / SD_H(VR幂律) / JATS(跳跃混合) / QR(线性分位回归)
检验：逐格 CRPS + PICP95 + Diebold-Mariano(HL修正, Bartlett HAC L=10) + MCS(bootstrap)
"""
import numpy as np, pandas as pd, os, sys, json, warnings
from scipy.stats import norm, t as tdist, poisson
from scipy.optimize import linprog
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

OUT = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(OUT, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS); JM = 7; J05 = 0; J95 = 14
W = 250; HOR = [5, 20, 40, 60, 120]; STEP = 2
KS = np.arange(0.30, 2.01, 0.01)
MODELS = ["EMP", "GAU", "SD_SQ", "EW97", "SD_H", "JATS", "DAT"]
XG = np.linspace(-6.0, 9.0, 401); KMAX = 12


def crps_q(yt, Q):
    Qs = np.sort(Q, axis=1)
    ind = (yt[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (yt[:, None] - Qs)).mean(axis=1)


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


def mcs(Lmat, alpha=0.10, B=400, block=10, seed=7):
    """Hansen-Lunde-Nason 范围法 MCS（块自助，bootstrap 已中心化）"""
    rng = np.random.default_rng(seed)
    n, k = Lmat.shape
    inc = np.ones(k, bool)
    nb = int(np.ceil(n / block))
    offs = np.arange(block)
    for _ in range(k - 1):
        cols = np.where(inc)[0]
        Ls = Lmat[:, cols]
        dbar = Ls.mean(0)
        boot = np.empty((B, len(cols)))
        for b in range(B):
            st = rng.integers(0, n, nb)
            take = ((st[:, None] + offs[None, :]) % n).ravel()[:n]
            boot[b] = Ls[take].mean(0)
        se = boot.std(0, ddof=1)
        se[se <= 0] = 1e-12
        Db = boot - dbar[None, :]          # 中心化
        tmax = (dbar - dbar.min()) / se
        tmax_b = (Db - Db.min(1)[:, None]) / se[None, :]
        crit = np.quantile(tmax_b.max(1), 1 - alpha)
        if tmax.max() <= crit:
            break
        inc[cols[int(np.argmax(tmax))]] = False
        if inc.sum() <= 1:
            break
    return np.where(inc)[0].tolist(), inc


def jump_params(win):
    r = np.diff(win)
    if len(r) < 60:
        return None
    bv = (np.pi / 2.0) * np.mean(np.abs(r[1:]) * np.abs(r[:-1]))
    sig = np.sqrt(bv) if bv > 0 else np.sqrt(np.mean(r ** 2))
    if not np.isfinite(sig) or sig <= 1e-12:
        return None
    isj = np.abs(r) > 4.0 * sig
    nj = int(isj.sum()); lam = nj / len(r)
    muJ = float(r[isj].mean()) if nj >= 3 else 0.0
    sigJ = float(r[isj].std(ddof=1)) if nj >= 3 else 0.0
    mu = float(r[~isj].mean()) if (~isj).sum() > 10 else float(r.mean())
    return dict(mu=mu, sig=float(sig), lam=lam, muJ=muJ, sigJ=sigJ)


def mix_q(pr, h):
    lamh = pr["lam"] * h
    kmax = int(min(KMAX, max(4, lamh + 6 * np.sqrt(lamh + 1))))
    k = np.arange(kmax + 1)
    w = poisson.pmf(k, lamh); w = w / max(w.sum(), 1e-300)
    m = pr["mu"] * h + k * pr["muJ"]
    v = np.maximum(pr["sig"] ** 2 * h + k * pr["sigJ"] ** 2, 1e-18)
    cdf = (w[None, :] * norm.cdf((XG[:, None] - m[None, :]) / np.sqrt(v)[None, :])).sum(1)
    cdf = np.maximum.accumulate(np.clip(cdf, 0, 1))
    return np.interp(TAUS, cdf, XG)


RES = {}
for m in ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    cut = "2024-01-01" if m in ("CEA", "GUANGZHOU", "SHANGHAI") else "2023-01-01"
    t0 = int(np.searchsorted(dt, np.datetime64(cut)))
    cache = {}
    for t in range(W + 5, len(lp)):
        win = lp[max(0, t - W):t]
        r = np.diff(win)
        if len(r) < 60:
            continue
        v1 = np.var(r, ddof=1)
        if v1 <= 0:
            continue
        bv = (np.pi / 2.0) * np.mean(np.abs(r[1:]) * np.abs(r[:-1]))
        kk = min(20, len(win) // 3)
        vr = np.var(win[kk:] - win[:-kk], ddof=1) / (kk * v1)
        H = float(np.clip(0.5 + np.log(vr) / (2 * np.log(kk)), 0.05, 0.95)) if vr > 0 else 0.5
        v = r[0] ** 2
        for i in range(1, len(r)):
            v = 0.97 * v + 0.03 * r[i - 1] ** 2
        cache[t] = dict(mu=float(r.mean()), sd=float(np.sqrt(v1)), H=H,
                        e97=float(np.sqrt(max(v, 1e-14))), bv=float(np.sqrt(bv) if bv > 0 else np.sqrt(v1)),
                        jp=jump_params(win))
    print(f"\n{'='*150}\n{m}   切分 {cut}  n_tr={t0} n_te={len(lp)-t0}")
    print(f"{'h':>4}{'n_te':>6}" + "".join(f"{n:>9}" for n in MODELS)
          + f"{'最优':>7}{'MCS(90%)'}")
    RES[m] = {}
    for h in HOR:
        ts_tr = [t for t in range(W + h + 5, t0 - h - 1, STEP) if t in cache]
        ts_te = [t for t in range(W + h + 5, len(lp) - h, STEP) if t in cache]
        if len(ts_tr) < 30 or len(ts_te) < 30:
            print(f"{h:>4}  SKIP {len(ts_tr)}/{len(ts_te)}"); continue

        def mk(ts):
            out = {n: np.full((len(ts), 15), np.nan) for n in MODELS}
            for i, t in enumerate(ts):
                sc = cache[t]
                win = lp[max(0, t - W):t]
                rr = win[h:] - win[:-h]
                sd = np.std(rr, ddof=1)
                if sd <= 0:
                    continue
                sh = np.sqrt(h)
                out["EMP"][i] = np.quantile(rr, TAUS)
                out["GAU"][i] = rr.mean() + sd * GV
                out["SD_SQ"][i] = rr.mean() + sc["sd"] * sh * GV
                out["EW97"][i] = rr.mean() + sc["e97"] * sh * GV
                out["SD_H"][i] = rr.mean() + sc["sd"] * (h ** sc["H"]) * GV
                out["DAT"][i] = sc["mu"] * h + sc["sd"] * sh * GV
                if sc["jp"] is not None:
                    out["JATS"][i] = mix_q(sc["jp"], h)
            return out

        A_tr = mk(ts_tr); A_te = mk(ts_te)
        y_tr = lp[np.array(ts_tr) + h] - lp[ts_tr]
        y_te = lp[np.array(ts_te) + h] - lp[ts_te]
        gtr = np.isfinite(A_tr["EMP"]).all(1); gte = np.isfinite(A_te["EMP"]).all(1)
        use = [n for n in MODELS if np.isfinite(A_tr[n]).all(1).sum() >= 25
               and np.isfinite(A_te[n]).all(1).sum() >= 25]
        gtr = gtr & np.all([np.isfinite(A_tr[n]).all(1) for n in use], axis=0)
        gte = gte & np.all([np.isfinite(A_te[n]).all(1) for n in use], axis=0)
        if gtr.sum() < 25 or gte.sum() < 25:
            print(f"{h:>4}  SKIPmask {gtr.sum()}/{gte.sum()}"); continue
        ks = {n: pick_k(y_tr[gtr], A_tr[n][gtr]) for n in use}
        y = y_te[gte]
        L = np.column_stack([crps_q(y, apply_k(A_te[n][gte], ks[n])) for n in use])
        c = dict(zip(use, L.mean(0)))
        best = min(c, key=c.get)
        inc, mask = mcs(L)
        mset = [use[i] for i in inc]
        pic = {n: float(((y >= apply_k(A_te[n][gte], ks[n])[:, J05]) &
                         (y <= apply_k(A_te[n][gte], ks[n])[:, J95])).mean()) for n in use}
        print(f"{h:>4}{int(gte.sum()):>6}" + "".join(
            f"{c[n]:>9.5f}" if n in c else f"{'-':>9}" for n in MODELS)
              + f"{best:>7}  {' '.join(mset)}")
        RES[m][h] = dict(n=int(gte.sum()), best=best, mcs=mset, picc=pic, **c)

print(f"\n{'='*150}")
print("汇总：CRPS（realistic 校准）与 DAT 相对最优的差距")
print(f"{'市场':<12}{'h':>5}" + "".join(f"{n:>9}" for n in MODELS) + f"{'最优':>7}{'DAT差':>9}{'DAT_PICP':>10}")
Wc = {}; ntot = 0; gaps = []
for m in RES:
    for h in sorted(RES[m]):
        r = RES[m][h]; b = r["best"]
        Wc[b] = Wc.get(b, 0) + 1; ntot += 1
        gp = (r["DAT"] / r[b] - 1) * 100 if "DAT" in r else np.nan
        gaps.append(gp)
        print(f"{m:<12}{h:>5}" + "".join(
            f"{r[n]:>9.5f}" if n in r else f"{'-':>9}" for n in MODELS)
              + f"{b:>7}{gp:>+8.2f}%{r['picc'].get('DAT', float('nan')):>10.3f}")
print(f"\n  获胜计数：{Wc}")
g = np.array(gaps, float); g = g[np.isfinite(g)]
print(f"  DAT 相对该格最优：中位 {np.median(g):+.2f}%  在 MCS 内 {sum(1 for m in RES for h in RES[m] if 'DAT' in RES[m][h]['mcs'])}/{ntot}")
json.dump(RES, open(os.path.join(OUT, "DAT_RESULT.json"), "w"), indent=1,
          ensure_ascii=False, default=float)
print("\n已写出 DAT_RESULT.json")
