# -*- coding: utf-8 -*-
"""
实验体系 4/6：安慰剂检验与负控（negative controls）
=====================================================================
审稿人必问的三类"这会不会只是数据挖掘"：
 N1 **安慰剂期限**：A14 的形状订正针对短端。h=1 是纯噪声期限，
    若模型在 h=1 也"显著更好"，说明改进是噪声产物。
 N2 **打乱时间序列**：把收益率序列**随机打乱**后重跑整套流程。
    真实的时序结构（漂移、波动聚集、偏度）全部消失。
    若 A14 仍"更好"，说明改进来自流程而非数据。
 N3 **特征置零/替换**：把形状块的输入（窗内经验分位 ξ_d）替换成
    (a) 随机分位、(b) 常数高斯、(c) 上一期的分位。
    若 A14 的增益在替换后不消失，说明增益来自"有个形状块"而不是"用了对的形状"。
 N4 **模型成分安慰剂**：给 A5 加上一个**随机**的形状调度 w'(h)，
    检验"只要引入非高斯形状就有效"这一竞争解释。
输出 PLACEBO.json
"""
import numpy as np, pandas as pd, os, sys, json, warnings
from scipy.stats import norm, t as tdist
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
DIR = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(DIR, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS)
W = 250; HOR = [1, 5, 20, 40, 60, 120]; STEP = 2
H0G = [0.1, 0.2, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 500]
HG = np.round(np.arange(0.15, 0.861, 0.05), 3)
MK = ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]
MN = {"CEA": "CEA", "GUANGZHOU": "广州", "SHANGHAI": "上海", "HUBEI": "湖北", "EU": "EU"}
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}


def crps_q(y, Q):
    Qs = np.sort(Q, axis=1)
    ind = (y[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (y[:, None] - Qs)).mean(axis=1)


def dm(a, b, L=10):
    d = np.asarray(a) - np.asarray(b); n = len(d)
    if n < 20:
        return np.nan
    m_ = d.mean(); dc = d - m_
    lrv = dc @ dc / n
    for l in range(1, L + 1):
        lrv += 2 * (1 - l / (L + 1)) * (dc[l:] @ dc[:-l] / n)
    if lrv <= 0:
        return np.nan
    s = m_ / np.sqrt(lrv / n) * np.sqrt((n + 1 - 2 * L + L * (L - 1) / n) / n)
    return float(2 * (1 - tdist.cdf(abs(s), df=n - 1)))


def build(lp, h, t0, rng=None, shuffle=False):
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


def fit_and_eval(lp, t0, hset, xi_mode="real", rng=None, w_sched=None):
    """在给定数据上拟合 (h0,H) 并返回 A5/A14 的测试段损失。"""
    CC = {h: build(lp, h, t0) for h in hset}
    CC = {h: c for h, c in CC.items() if c is not None and c["tr"].sum() >= 30}
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
    out = []
    for h, c in sorted(CC.items()):
        te = c["te"]
        if te.sum() < 20:
            continue
        y = c["y"][te]; mu = c["mu"][te]; sdd = c["sdd"][te]; xi = c["xi"][te]
        G = np.tile(GV, (len(y), 1))
        S = sdd * h ** Hv
        w = (min(1.0, h / 50.0) if w_sched is None else w_sched(h))
        if xi_mode == "real":
            XI = xi
        elif xi_mode == "rand":
            XI = np.vstack([rng.permutation(GV) for _ in range(len(y))])
        elif xi_mode == "const":
            XI = np.tile(GV, (len(y), 1))
        elif xi_mode == "lag":
            XI = np.vstack([xi[(i - 1) % len(xi)] for i in range(len(xi))])
        else:
            XI = xi
        Q_A5 = mu[:, None] * (h0 / (h0 + h)) + S[:, None] * G
        Q_A14 = mu[:, None] * (h0 / (h0 + h)) + S[:, None] * ((1 - w) * XI + w * G)
        out.append(dict(h=h, n_te=int(te.sum()),
                        A5=crps_q(y, Q_A5), A14=crps_q(y, Q_A14)))
    return out


print("=" * 120)
print("N1  安慰剂期限 h=1（纯噪声期限）：A14 的形状订正是否也\"有效\"")
print("=" * 120)
RES = {}
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    for h in [1, 5, 20, 120]:
        r = fit_and_eval(lp, t0, [h])
        if not r:
            continue
        x = r[0]
        rr = x["A14"].mean() / x["A5"].mean()
        pv = dm(x["A14"], x["A5"])
        print(f"  {MN[m]:<6} h={h:<4} A5 {x['A5'].mean():.5f}  A14 {x['A14'].mean():.5f}  "
              f"比 {rr:.4f}  p {pv:.3f}")
        RES[f"N1|{m}|{h}"] = dict(r=float(rr), p=float(pv), n=x["n_te"])
n1 = [v for k, v in RES.items() if k.startswith("N1|")]
for h in [1, 5, 20, 120]:
    s = [RES[f"N1|{m}|{h}"] for m in MK if f"N1|{m}|{h}" in RES]
    if s:
        sw = sum(1 for x in s if x["p"] < .05 and x["r"] < 1)
        print(f"  h={h:<5} 中位比 {np.median([x['r'] for x in s]):.4f}  显著胜 {sw}/{len(s)}")

print("\n" + "=" * 120)
print("N2  打乱时间序列（真实时序结构全部消失）")
print("=" * 120)
rng = np.random.default_rng(20261021)
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    r_real = fit_and_eval(lp, t0, [5, 20, 40, 60, 120])
    # 打乱：把对数价格增量随机置换后重建价格路径
    inc = np.diff(lp)
    perm = rng.permutation(len(inc))
    lp_s = np.concatenate([[0.0], np.cumsum(inc[perm])])
    r_shuf = fit_and_eval(lp_s, t0, [5, 20, 40, 60, 120])
    if not r_real or not r_shuf:
        continue
    rr_real = np.array([x["A14"].mean() / x["A5"].mean() for x in r_real])
    rr_shuf = np.array([x["A14"].mean() / x["A5"].mean() for x in r_shuf])
    sw_real = sum(1 for x in r_real if dm(x["A14"], x["A5"]) < .05 and x["A14"].mean() < x["A5"].mean())
    sw_shuf = sum(1 for x in r_shuf if dm(x["A14"], x["A5"]) < .05 and x["A14"].mean() < x["A5"].mean())
    print(f"  {MN[m]:<6} 真实: 中位 {np.median(rr_real):.4f} 显著胜 {sw_real}/5   |   "
          f"打乱: 中位 {np.median(rr_shuf):.4f} 显著胜 {sw_shuf}/5")
    RES[f"N2|{m}"] = dict(real=float(np.median(rr_real)), shuf=float(np.median(rr_shuf)),
                           sw_real=int(sw_real), sw_shuf=int(sw_shuf))

print("\n" + "=" * 120)
print("N3  形状输入替换（xi_d -> 随机分位 / 常数高斯 / 滞后一期）")
print("=" * 120)
for mode, lab in (("real", "真实 ξ_d"), ("rand", "随机分位"), ("const", "常数高斯"), ("lag", "滞后一期 ξ_d")):
    vals, sws = [], 0
    for m in MK:
        d = p[p.market == m].sort_values("date").reset_index(drop=True)
        lp = np.log(d.price.to_numpy(float)).astype(float)
        dt = pd.to_datetime(d.date).to_numpy()
        t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
        r = fit_and_eval(lp, t0, [5, 20, 40, 60, 120], xi_mode=mode, rng=np.random.default_rng(7))
        if not r:
            continue
        for x in r:
            vals.append(x["A14"].mean() / x["A5"].mean())
            if dm(x["A14"], x["A5"]) < .05 and x["A14"].mean() < x["A5"].mean():
                sws += 1
    print(f"  {lab:<14} 中位 {np.median(vals):.4f}  胜 {sum(1 for v in vals if v<1)}/{len(vals)}  "
          f"显著胜 {sws}")
    RES[f"N3|{mode}"] = dict(med=float(np.median(vals)), sw=sws, n=len(vals))

print("\n" + "=" * 120)
print("N4  模型成分安慰剂：给 A5 加一个**随机**的形状调度 w'(h)")
print("=" * 120)
rng2 = np.random.default_rng(99)
for trial in range(3):
    vals, sws = [], 0
    for m in MK:
        d = p[p.market == m].sort_values("date").reset_index(drop=True)
        lp = np.log(d.price.to_numpy(float)).astype(float)
        dt = pd.to_datetime(d.date).to_numpy()
        t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
        hs = rng2.random(5) * 0.9     # 随机阈值
        sch = {h: (lambda hh, th: min(1.0, hh / th))(h, hs[i]) for i, h in enumerate([5, 20, 40, 60, 120])}
        r = fit_and_eval(lp, t0, [5, 20, 40, 60, 120], w_sched=lambda hh, s=sch: s[hh])
        if not r:
            continue
        for x in r:
            vals.append(x["A14"].mean() / x["A5"].mean())
            if dm(x["A14"], x["A5"]) < .05 and x["A14"].mean() < x["A5"].mean():
                sws += 1
    print(f"  随机调度 {trial+1}: 中位 {np.median(vals):.4f}  胜 {sum(1 for v in vals if v<1)}/{len(vals)}  显著胜 {sws}")
    RES[f"N4|{trial}"] = dict(med=float(np.median(vals)), sw=sws, n=len(vals))

json.dump(RES, open(os.path.join(DIR, "PLACEBO.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 PLACEBO.json")
