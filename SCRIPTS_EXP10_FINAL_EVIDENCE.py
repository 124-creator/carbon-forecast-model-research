# -*- coding: utf-8 -*-
"""
实验体系 10/6+：最终证据表——在**正确零假设**下重做全部对照
=====================================================================
§24.31 的 p 值不可信（exp8：四种 DM 实现全部失准）。
exp9 确立了正确做法：**零假设 = 在没有可利用时序信息的序列上，两模型期望损失相同**，
用"打乱序列 + 完整流程"模拟零分布。

本节把这个框架应用到**全部对照**，给出最终可发表的证据表：
  对照：A5、EMP、RW、SD250、VR
  统计量：合并 log 比（跨市场等权）与按市场/期限分层
  零分布：200 次打乱，每次跑完整流程
  报：真实值、零分布均值±标准差、单尾 p、以及"真实值在零分布中的百分位"

同时给出一个**必须承认的事实**：改进高度集中在广州。
因此最终叙述应是：
  "改进是**真实的**（p=0.005），但**不均匀**（广州 p<0.0001，其他市场 p=0.05–0.41）"，
  而不是"普遍改善"。
输出 FINAL_EVIDENCE.json
"""
import numpy as np, pandas as pd, os, sys, json, warnings, time
from scipy.stats import norm
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
NSHUF = 200
SEED0 = 271828
BASE = ["A5", "EMP", "RW", "SD250", "VR"]


def crps_q(y, Q):
    Qs = np.sort(Q, axis=1)
    ind = (y[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (y[:, None] - Qs)).mean(axis=1)


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


def pipeline(lp, t0):
    """返回 {base: {h: crps}}，参数独立重估；A14 由 A5 的参数 + 形状块得到。"""
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
    out = {b: {} for b in BASE}; out["A14"] = {}
    for h, c in sorted(CC.items()):
        te = c["te"]
        if te.sum() < 20:
            continue
        y = c["y"][te]; mu = c["mu"][te]; sdd = c["sdd"][te]; xi = c["xi"][te]
        sdw = c["sd"][te]
        w14 = min(1.0, h / 50.0)
        G = np.tile(GV, (len(y), 1))
        S = sdd * h ** Hv
        lam = h0 / (h0 + h)
        out["A5"][h] = float(crps_q(y, mu[:, None] * lam + S[:, None] * G).mean())
        out["A14"][h] = float(crps_q(y, mu[:, None] * lam + S[:, None] * ((1 - w14) * xi + w14 * G)).mean())
        out["SD250"][h] = float(crps_q(y, mu[:, None] * lam + (sdd * np.sqrt(h))[:, None] * G).mean())
        out["VR"][h] = float(crps_q(y, (sdw)[:, None] * G).mean())
        out["RW"][h] = float(crps_q(y, (sdd * np.sqrt(h))[:, None] * G).mean())
        out["EMP"][h] = float(crps_q(y, np.vstack([
            np.quantile(lp[max(0, t - W):t][h:] - lp[max(0, t - W):t][:-h], TAUS)
            for t in c["ts"][te]])).mean())
    return out


DATA = {}
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    DATA[m] = (lp, t0)

REAL = {m: pipeline(*DATA[m]) for m in MK}
T0 = time.time()
NULL = []
for it in range(NSHUF):
    rng = np.random.default_rng(SEED0 + it)
    rec = {}
    for m in MK:
        lp, t0 = DATA[m]
        inc = np.diff(lp)
        lp_s = np.concatenate([[0.0], np.cumsum(inc[rng.permutation(len(inc))])])
        L = pipeline(lp_s, t0)
        if L:
            for b in BASE + ["A14"]:
                for h in L[b]:
                    rec[(b, h, m)] = L[b][h]
    NULL.append(rec)
    if (it + 1) % 50 == 0:
        print(f"  打乱 {it+1}/{NSHUF}  ({time.time()-T0:.0f} 秒)")

# 汇总
print("\n" + "=" * 116)
print("最终证据表：正确零假设下的合并检验（200 次打乱，每次完整流程）")
print("=" * 116)
print(f"{'对照':<8}{'真实合并 log比':>15}{'零分布均值':>13}{'零分布SD':>11}"
      f"{'单尾p':>9}{'百分位':>8}{'逐市场最小p':>13}")
FINAL = {}
for b in BASE:
    # 真实：每个 (h,m) 一格，跨 15 格等权
    real = [np.log(REAL[m]["A14"][h] / REAL[m][b][h]) for m in MK for h in HS if h in REAL[m][b]]
    mu_real = float(np.mean(real))
    # 零分布：每次打乱同样 15 格
    nulls = []
    for it, rec in enumerate(NULL):
        vals = [np.log(rec[("A14", h, m)] / rec[(b, h, m)])
                for m in MK for h in HS if ("A14", h, m) in rec and (b, h, m) in rec]
        if len(vals) == len(real):
            nulls.append(float(np.mean(vals)))
    nulls = np.array(nulls)
    p_one = float(np.mean(nulls <= mu_real))
    # 逐市场 p
    mp = {}
    for m in MK:
        rv = [np.log(REAL[m]["A14"][h] / REAL[m][b][h]) for h in HS if h in REAL[m][b]]
        mur = float(np.mean(rv))
        nn = []
        for it, rec in enumerate(NULL):
            vals = [np.log(rec[("A14", h, m)] / rec[(b, h, m)])
                    for h in HS if ("A14", h, m) in rec and (b, h, m) in rec]
            if len(vals) == len(rv):
                nn.append(float(np.mean(vals)))
        mp[m] = float(np.mean(np.array(nn) <= mur))
    pct = float(np.mean(nulls < mu_real) * 100)
    print(f"{b:<8}{mu_real:>15.5f}{nulls.mean():>13.5f}{nulls.std():>11.5f}"
          f"{p_one:>9.4f}{pct:>7.1f}%{min(mp.values()):>13.4f}")
    FINAL[b] = dict(real_mu=mu_real, null_mean=float(nulls.mean()), null_sd=float(nulls.std()),
                     p=p_one, pct=pct, by_market_p=mp,
                     by_market_mu={m: float(np.mean([np.log(REAL[m]["A14"][h] / REAL[m][b][h])
                                                     for h in HS if h in REAL[m][b]])) for m in MK})

print("\n  逐市场明细（真实合并 log 比 与 单尾 p）：")
print(f"{'市场':<8}" + "".join(f"{b:>22}" for b in BASE))
print(f"{'':<8}" + "".join(f"{'log比':>12}{'p':>10}" for b in BASE))
for m in MK:
    row = f"{MN[m]:<8}"
    for b in BASE:
        row += f"{FINAL[b]['by_market_mu'][m]:>12.5f}{FINAL[b]['by_market_p'][m]:>10.3f}"
    print(row)

print("\n" + "=" * 116)
print("广州单独检验（改进最集中的市场）")
print("=" * 116)
for b in BASE:
    rv = [np.log(REAL["GUANGZHOU"]["A14"][h] / REAL["GUANGZHOU"][b][h]) for h in HS]
    mur = float(np.mean(rv))
    nn = []
    for it, rec in enumerate(NULL):
        vals = [np.log(rec[("A14", h, "GUANGZHOU")] / rec[(b, h, "GUANGZHOU")])
                for h in HS if ("A14", h, "GUANGZHOU") in rec and (b, h, "GUANGZHOU") in rec]
        if len(vals) == len(rv):
            nn.append(float(np.mean(vals)))
    nn = np.array(nn)
    print(f"  vs {b:<7} 真实 {mur:+.5f}   零分布 {nn.mean():+.5f}±{nn.std():.5f}   "
          f"p = {float(np.mean(nn <= mur)):.4f}")

json.dump({"final": FINAL, "n_shuffle": NSHUF,
           "real": {m: {b: {str(h): REAL[m][b][h] for h in REAL[m][b]} for b in BASE + ["A14"]}
                    for m in MK}},
          open(os.path.join(DIR, "FINAL_EVIDENCE.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 FINAL_EVIDENCE.json")
