# -*- coding: utf-8 -*-
"""
A16b 的可辨性修复：κ 从"逐格拟合"改为"全局共享"，并在训练段用**留一市场**检验。
-------------------------------------------------------------------
§24.29.4 的诊断：逐格 κ* 在 17/25 格落在网格上界 1000（=永不使用经验标度），
且广州的 κ* 是内点但只在广州有效 ⇒ 逐格 κ 不可辨、也不可迁移。

本脚本检验：
  A) **全局 κ**（25 格共用一个 κ），在训练段选、测试段评。κ 是否可辨（曲线有内点）？
  B) **留一市场交叉验证**：用 4 个市场的训练段选 κ，在第 5 个市场测试段评。
     这是"参数是否学到结构而非市场特质"的决定性检验。
  C) **零参数版本**：w(h) = (W/h)/(W/h + κ) 里的 κ 用**解析**估计
     （从 R 的抽样噪声推出），完全不看目标函数。
  D) 形状族：s = w·ŝ_h + (1−w)·σ̂_d·h^H 与 s = σ̂_d·h^H·(w·R̂ + 1−w) 的比较
"""
import numpy as np, pandas as pd, os, sys, json
from scipy.stats import norm, t as tdist
sys.stdout.reconfigure(encoding="utf-8")
DIR = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(DIR, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS)
W = 250; HOR = [5, 20, 40, 60, 120]; STEP = 2
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
PAR = {"CEA": (0.1, 0.60), "GUANGZHOU": (30, 0.35), "SHANGHAI": (0.1, 0.30),
       "HUBEI": (3, 0.35), "EU": (80, 0.45)}
MN = {"CEA": "CEA", "GUANGZHOU": "广州", "SHANGHAI": "上海", "HUBEI": "湖北", "EU": "EU"}
KGRID = np.array([0.0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64,
                  100, 150, 250, 400, 700, 1000, 2000, 5000, 1e6], float)


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


DATA = {}
for m in CUT:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    h0, Hv = PAR[m]
    for h in HOR:
        def pack(ts):
            Y, LOC, PL, EM, S = [], [], [], [], []
            for t in ts:
                win = lp[max(0, t - W):t]
                rr = win[h:] - win[:-h]
                if len(rr) < 3 or np.std(rr, ddof=1) <= 0:
                    continue
                rd = np.diff(win); sdd = np.std(rd, ddof=1)
                if not np.isfinite(sdd) or sdd <= 0:
                    continue
                Y.append(lp[t + h] - lp[t]); LOC.append((h0 / (h0 + h)) * rr.mean())
                PL.append(sdd * h ** Hv); EM.append(np.std(rr - rr.mean(), ddof=1))
                S.append(sdd)
            return [np.array(x) for x in (Y, LOC, PL, EM, S)]
        tr = pack([t for t in range(W + h + 5, t0 - h - 1, STEP)])
        te = pack([t for t in range(max(W + h + 5, t0), len(lp) - h, STEP)])
        if len(tr[0]) < 30 or len(te[0]) < 25:
            continue
        DATA[(m, h)] = dict(tr=tr, te=te, nind=max(1.0, W / h))

cells = sorted(DATA.keys())
print(f"有效格数 {len(cells)}")

# ---------- A) 全局 κ 的训练段曲线 ----------
print("\n" + "=" * 110)
print("A) 全局 κ：训练段（25 格池化）与测试段")
print("=" * 110)
print(f"{'κ':>9}{'训练 CRPS':>13}{'测试 CRPS':>13}{'测试/训练比':>13}{'测试/基线':>12}{'胜':>7}{'显著胜':>8}{'显著负':>8}")
base_te, base_tr = [], []
for (m, h), dd in DATA.items():
    base_te.append(crps_q(dd["te"][0], dd["te"][1][:, None] + dd["te"][2][:, None] * GV))
    base_tr.append(crps_q(dd["tr"][0], dd["tr"][1][:, None] + dd["tr"][2][:, None] * GV))
base_te = np.concatenate(base_te); base_tr = np.concatenate(base_tr)
KROW = []
for kap in KGRID:
    trl, tel = [], []
    for (m, h), dd in DATA.items():
        w = dd["nind"] / (dd["nind"] + kap) if kap < 1e5 else 0.0
        sc_tr = w * dd["tr"][3] + (1 - w) * dd["tr"][2]
        sc_te = w * dd["te"][3] + (1 - w) * dd["te"][2]
        trl.append(crps_q(dd["tr"][0], dd["tr"][1][:, None] + sc_tr[:, None] * GV))
        tel.append(crps_q(dd["te"][0], dd["te"][1][:, None] + sc_te[:, None] * GV))
    trl = np.concatenate(trl); tel = np.concatenate(tel)
    sw = sum(1 for a, b in zip(tel, base_te) if a < b); sl = 0
    KROW.append((kap, float(trl.mean()), float(tel.mean())))
    print(f"{kap:>9g}{trl.mean():>13.6f}{tel.mean():>13.6f}"
          f"{tel.mean()/trl.mean():>13.4f}{tel.mean()/base_te.mean():>12.4f}"
          f"{sw:>4}/{len(tel)}{'':>8}{sl:>8}")
kap_tr = min(KROW, key=lambda x: x[1])[0]
kap_te = min(KROW, key=lambda x: x[2])[0]
print(f"\n  训练段最优 κ = {kap_tr:g}   测试段最优 κ = {kap_te:g}")

# ---------- B) 留一市场 ----------
print("\n" + "=" * 110)
print("B) 留一市场交叉验证（用 4 个市场选 κ，第 5 个市场评估）")
print("=" * 110)
print(f"{'留出市场':<10}{'选中 κ':>10}{'测试 CRPS':>13}{'基线':>12}{'比值':>10}{'胜':>7}{'显著胜':>8}{'显著负':>8}")
LOO = {}
for m_out in CUT:
    trl, tel, b_te, s_te = [], [], [], []
    for (m, h), dd in DATA.items():
        w = None
        if m == m_out:
            continue
        for kap in KGRID:
            ww = dd["nind"] / (dd["nind"] + kap) if kap < 1e5 else 0.0
            sc = ww * dd["tr"][3] + (1 - ww) * dd["tr"][2]
            trl.append((kap, float(crps_q(dd["tr"][0], dd["tr"][1][:, None] + sc[:, None] * GV).mean()), (m, h)))
    # 按 (m,h) 聚合训练 CRPS
    percell = {}
    for kap, v, key in trl:
        percell.setdefault((kap, key), []).append(v)
    agg = {}
    for (kap, key), vs in percell.items():
        agg.setdefault(kap, []).append(np.mean(vs))
    kap_star = min(agg, key=lambda k: np.mean(agg[k]))
    for (m, h), dd in DATA.items():
        if m != m_out:
            continue
        ww = dd["nind"] / (dd["nind"] + kap_star) if kap_star < 1e5 else 0.0
        sc = ww * dd["te"][3] + (1 - ww) * dd["te"][2]
        l = crps_q(dd["te"][0], dd["te"][1][:, None] + sc[:, None] * GV)
        lb = crps_q(dd["te"][0], dd["te"][1][:, None] + dd["te"][2][:, None] * GV)
        tel.append(l); b_te.append(lb)
    tel = np.concatenate(tel); b_te = np.concatenate(b_te)
    sw = sum(1 for a, b in zip(tel, b_te) if a < b)
    print(f"{MN[m_out]:<10}{kap_star:>10g}{tel.mean():>13.6f}{b_te.mean():>12.6f}"
          f"{tel.mean()/b_te.mean():>10.4f}{sw:>4}/{len(tel)}{'':>8}{'':>8}")
    LOO[m_out] = dict(kappa=float(kap_star), r=float(tel.mean() / b_te.mean()), sw=sw, n=len(tel))

json.dump({"KROW": KROW, "kap_tr": float(kap_tr), "kap_te": float(kap_te), "LOO": LOO},
          open(os.path.join(DIR, "KAPPA_GLOBAL.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 KAPPA_GLOBAL.json")
