# -*- coding: utf-8 -*-
"""
实验体系 1/6：多切分稳健性（每个切分独立重估 (h0,H)，A14 为主角）
=====================================================================
设计要点：
 * 每个切分**独立重估** (h0, H)，只用该切分的训练段 —— 参数绝不跨切分泄漏。
 * A14 的形状块 w(h)=min(1,h/50) 是**零拟合参数**的调度，跨切分照用。
 * 对照集合：A5（上一版规格）、EMP、RW、SD250（固定 H=0.5）、VR（滚动 H）。
 * 每个切分报：CRPS 中位比、显著胜/负（DM，HAC）、WIS、PICP80/95、n_eff。
 * 汇总：跨切分的**一致性**（多少切分里结论方向一致）—— 这是审稿人最看重的。
输出 MULTISPLIT_A14.json
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
W = 250; HOR = [5, 20, 40, 60, 120]; STEP = 2
H0G = [0.1, 0.2, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 500]
HG = np.round(np.arange(0.15, 0.861, 0.05), 3)
MK = ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]
MN = {"CEA": "CEA", "GUANGZHOU": "广州", "SHANGHAI": "上海", "HUBEI": "湖北", "EU": "EU"}
SPL = [0.45, 0.55, 0.65, 0.75, 0.85]


def crps_q(y, Q):
    Qs = np.sort(Q, axis=1)
    ind = (y[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (y[:, None] - Qs)).mean(axis=1)


def wis(y, Q):
    PAIRS = [(0, 14, 0.05), (1, 13, 0.10), (2, 12, 0.20), (3, 11, 0.30), (4, 10, 0.40)]
    tot = 0.5 * np.abs(y - Q[:, 7])
    for a, b, al in PAIRS:
        l, u = Q[:, a], Q[:, b]
        IS = (u - l) + (2 / al) * np.maximum(l - y, 0) + (2 / al) * np.maximum(y - u, 0)
        tot = tot + (al / 2) * IS
    return tot / (len(PAIRS) + 0.5)


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


def rolling_H(lp, t, W=W):
    win = lp[max(0, t - W):t]
    r = np.diff(win)
    if len(r) < 60:
        return None
    v1 = np.var(r, ddof=1)
    if v1 <= 0:
        return None
    kk = min(20, len(win) // 3)
    vr = np.var(win[kk:] - win[:-kk], ddof=1) / (kk * v1)
    return float(np.clip(0.5 + np.log(vr) / (2 * np.log(kk)), 0.05, 0.95)) if vr > 0 else 0.5


def build(lp, h, t0):
    rec = []
    for t in range(W + h + 5, len(lp) - h, STEP):
        win = lp[max(0, t - W):t]
        rr = win[h:] - win[:-h]
        rd = np.diff(win)
        sdw = np.std(rr, ddof=1); sdd = np.std(rd, ddof=1)
        if sdw <= 1e-12 or sdd <= 1e-12:
            continue
        H = rolling_H(lp, t)
        if H is None:
            continue
        rec.append((t, float(rr.mean()), sdw, sdd, H,
                    np.quantile((rd - rd.mean()) / sdd, TAUS)))
    if len(rec) < 40:
        return None
    ts = np.array([r[0] for r in rec])
    return dict(h=h, ts=ts, mu=np.array([r[1] for r in rec]),
                sd=np.array([r[2] for r in rec]), sdd=np.array([r[3] for r in rec]),
                Hroll=np.array([r[4] for r in rec]),
                xi=np.vstack([r[5] for r in rec]), y=lp[ts + h] - lp[ts],
                tr=(ts <= t0 - h - 2), te=(ts > t0))


PANEL = {m: np.log(p[p.market == m].sort_values("date").price.to_numpy(float)) for m in MK}

ALL = []
print("=" * 132)
print("多切分稳健性：每个切分独立重估 (h0, H)；A14 为主角")
print("=" * 132)
print(f"{'切分':>6}{'市场':>7}{'h':>5}{'n_tr':>6}{'n_te':>6}{'h0*':>7}{'H*':>6}"
      f"{'A5':>9}{'A14':>9}{'A14/A5':>9}{'p':>7}{'A14/EMP':>9}{'A14/RW':>9}"
      f"{'A14/SD250':>11}{'c80':>7}{'c95':>7}")
for frac in SPL:
    for m in MK:
        lp = PANEL[m]
        t0 = int(len(lp) * frac)
        CC = {h: build(lp, h, t0) for h in HOR}
        CC = {h: c for h, c in CC.items() if c is not None and c["tr"].sum() >= 30}
        if not CC:
            continue
        # ---- 训练段拟合 (h0, H) ----
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
        for h, c in sorted(CC.items()):
            te = c["te"]
            if te.sum() < 20:
                continue
            y = c["y"][te]; mu = c["mu"][te]; sdd = c["sdd"][te]
            sdw = c["sd"][te]; xi = c["xi"][te]; Hr = c["Hroll"][te]
            w14 = min(1.0, h / 50.0)
            S_A5 = sdd * h ** Hv
            S_SD = sdd * np.sqrt(h)
            S_VR = sdw
            G = np.tile(GV, (len(y), 1))
            Q_A5 = mu[:, None] * (h0 / (h0 + h)) + S_A5[:, None] * G
            Q_A14 = mu[:, None] * (h0 / (h0 + h)) + S_A5[:, None] * ((1 - w14) * xi + w14 * G)
            Q_EMP = np.quantile(np.column_stack([np.full(len(y), np.nan)] * 0 + 0.0), TAUS) if False else None
            # EMP：窗内 h 步收益的经验分位（逐原点）
            Q_EMP = np.empty((len(y), 15))
            for i, t in enumerate(c["ts"][te]):
                win = lp[max(0, t - W):t]
                rr = win[h:] - win[:-h]
                Q_EMP[i] = np.quantile(rr, TAUS)
            Q_RW = (0 + sdd * np.sqrt(h))[:, None] * G
            Q_SD = mu[:, None] * (h0 / (h0 + h)) + S_SD[:, None] * G
            Q_VR = S_VR[:, None] * G
            L = {"A5": crps_q(y, Q_A5), "A14": crps_q(y, Q_A14), "EMP": crps_q(y, Q_EMP),
                 "RW": crps_q(y, Q_RW), "SD250": crps_q(y, Q_SD), "VR": crps_q(y, Q_VR)}
            Qq = Q_A14
            c80 = float(((y >= Qq[:, 2]) & (y <= Qq[:, 12])).mean())
            c95 = float(((y >= Qq[:, 0]) & (y <= Qq[:, 14])).mean())
            row = dict(frac=frac, m=m, h=h, h0=h0, H=Hv, n_tr=int(c["tr"].sum()),
                       n_te=int(te.sum()),
                       **{k: float(v.mean()) for k, v in L.items()},
                       wis_A5=float(wis(y, Q_A5).mean()), wis_A14=float(wis(y, Q_A14).mean()),
                       wis_EMP=float(wis(y, Q_EMP).mean()), wis_RW=float(wis(y, Q_RW).mean()),
                       c80=c80, c95=c95,
                       p_A14_vs_A5=dm(L["A14"], L["A5"]),
                       p_A14_vs_EMP=dm(L["A14"], L["EMP"]),
                       p_A14_vs_RW=dm(L["A14"], L["RW"]),
                       p_A14_vs_SD250=dm(L["A14"], L["SD250"]),
                       p_A5_vs_EMP=dm(L["A5"], L["EMP"]))
            ALL.append(row)
            r = row
            print(f"{frac:>6.2f}{MN[m]:>7}{h:>5}{r['n_tr']:>6}{r['n_te']:>6}{h0:>7}{Hv:>6.2f}"
                  f"{r['A5']:>9.5f}{r['A14']:>9.5f}{r['A14']/r['A5']:>9.3f}{r['p_A14_vs_A5']:>7.3f}"
                  f"{r['A14']/r['EMP']:>9.3f}{r['A14']/r['RW']:>9.3f}{r['A14']/r['SD250']:>11.3f}"
                  f"{c80:>7.3f}{c95:>7.3f}")
    print("-" * 132)

print("\n" + "=" * 132)
print("汇总一：跨切分的一致性（每个切分单独统计）")
print("=" * 132)
print(f"{'切分':>6}{'格数':>6}{'A14/A5 中位':>13}{'胜':>7}{'显著胜':>8}{'显著负':>8}"
      f"{'A14/EMP 中位':>14}{'A14/RW 中位':>13}{'A14/SD250 中位':>16}{'c95 中位':>10}")
for frac in SPL:
    s = [r for r in ALL if r["frac"] == frac]
    if not s:
        continue
    ra = np.array([r["A14"] / r["A5"] for r in s])
    sw = sum(1 for r in s if r["p_A14_vs_A5"] == r["p_A14_vs_A5"] and r["p_A14_vs_A5"] < .05 and r["A14"] < r["A5"])
    sl = sum(1 for r in s if r["p_A14_vs_A5"] == r["p_A14_vs_A5"] and r["p_A14_vs_A5"] < .05 and r["A14"] > r["A5"])
    print(f"{frac:>6.2f}{len(s):>6}{np.median(ra):>13.4f}{int((ra<1).sum()):>4}/{len(s)}{sw:>8}{sl:>8}"
          f"{np.median([r['A14']/r['EMP'] for r in s]):>14.4f}"
          f"{np.median([r['A14']/r['RW'] for r in s]):>13.4f}"
          f"{np.median([r['A14']/r['SD250'] for r in s]):>16.4f}"
          f"{np.median([r['c95'] for r in s]):>10.3f}")

print("\n" + "=" * 132)
print("汇总二：按期限（跨切分合并）")
print("=" * 132)
print(f"{'h':>5}{'格数':>6}{'A14/A5':>10}{'胜':>7}{'显著胜':>8}{'显著负':>8}"
      f"{'A14/EMP':>10}{'A14/RW':>10}{'A14/SD250':>12}{'c80':>8}{'c95':>8}")
for h in HOR:
    s = [r for r in ALL if r["h"] == h]
    ra = np.array([r["A14"] / r["A5"] for r in s])
    sw = sum(1 for r in s if r["p_A14_vs_A5"] == r["p_A14_vs_A5"] and r["p_A14_vs_A5"] < .05 and r["A14"] < r["A5"])
    sl = sum(1 for r in s if r["p_A14_vs_A5"] == r["p_A14_vs_A5"] and r["p_A14_vs_A5"] < .05 and r["A14"] > r["A5"])
    print(f"{h:>5}{len(s):>6}{np.median(ra):>10.4f}{int((ra<1).sum()):>4}/{len(s)}{sw:>8}{sl:>8}"
          f"{np.median([r['A14']/r['EMP'] for r in s]):>10.4f}"
          f"{np.median([r['A14']/r['RW'] for r in s]):>10.4f}"
          f"{np.median([r['A14']/r['SD250'] for r in s]):>12.4f}"
          f"{np.median([r['c80'] for r in s]):>8.3f}{np.median([r['c95'] for r in s]):>8.3f}")

print("\n" + "=" * 132)
print("汇总三：按市场（跨切分合并）")
print("=" * 132)
print(f"{'市场':>8}{'格数':>6}{'A14/A5':>10}{'胜':>7}{'显著胜':>8}{'显著负':>8}{'A14/EMP':>10}{'A14/RW':>10}")
for m in MK:
    s = [r for r in ALL if r["m"] == m]
    if not s:
        continue
    ra = np.array([r["A14"] / r["A5"] for r in s])
    sw = sum(1 for r in s if r["p_A14_vs_A5"] == r["p_A14_vs_A5"] and r["p_A14_vs_A5"] < .05 and r["A14"] < r["A5"])
    sl = sum(1 for r in s if r["p_A14_vs_A5"] == r["p_A14_vs_A5"] and r["p_A14_vs_A5"] < .05 and r["A14"] > r["A5"])
    print(f"{MN[m]:>8}{len(s):>6}{np.median(ra):>10.4f}{int((ra<1).sum()):>4}/{len(s)}{sw:>8}{sl:>8}"
          f"{np.median([r['A14']/r['EMP'] for r in s]):>10.4f}"
          f"{np.median([r['A14']/r['RW'] for r in s]):>10.4f}")

print("\n" + "=" * 132)
print("汇总四：参数稳定性（每个切分独立重估的 (h0, H)）")
print("=" * 132)
for m in MK:
    for h in [5, 120]:
        vs = [(r["frac"], r["h0"], r["H"]) for r in ALL if r["m"] == m and r["h"] == h]
        if vs:
            print(f"  {MN[m]:<6} h={h:<4} " + "  ".join(f"{f:.2f}:({a:g},{b:.2f})" for f, a, b in vs))
print("\n  H* 的跨切分极差（每市场 h=5 与 h=120）：")
for m in MK:
    for h in [5, 120]:
        vs = [r["H"] for r in ALL if r["m"] == m and r["h"] == h]
        if len(vs) >= 2:
            print(f"    {MN[m]:<6} h={h:<4} 极差 {max(vs)-min(vs):.3f}  值 {[round(v,2) for v in vs]}")

json.dump(ALL, open(os.path.join(DIR, "MULTISPLIT_A14.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 MULTISPLIT_A14.json")
