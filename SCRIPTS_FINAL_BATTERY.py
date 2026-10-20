# -*- coding: utf-8 -*-
"""
FINAL：经济价值 + MCS + 覆盖率检验（全部在未校准口径下重算）
==============================================================
经济价值：碳配额是"买不足则罚款"的单边风险。
  C = exp(y) 为到期价格；R 为预购/持有价格。
  L(R) = c_c * R + c_p * max(C - R, 0),  c_c=0.05（资金成本）, c_p=1.0（罚款）
  最优 R* 满足 F(R*) = 1 - c_c/c_p = 0.95 -> R* = Q(0.95)（正好在 TAUS 网格上）
  eco = L(model) / L(naive) - 1，naive = 训练段样本 95% 分位（常数）
  floor = c_c * mean(C)（完美预知的下界）
MCS：Hansen-Lunde-Nason range，block bootstrap B=400, block=10，centred
覆盖率：Kupiec POF（无条件覆盖）+ Christoffersen 独立性
"""
import numpy as np, pandas as pd, os, sys, json, warnings
from scipy.stats import norm, chi2
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

OUT = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(OUT, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS); J95 = (0, 14); I95 = 13
W = 250; HOR = [5, 20, 40, 60, 120]; STEP = 2
MK = ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]
MOD = ["RW", "EMP", "A2 +位置", "A3 +形状", "A4 +H", "A5 完整"]
PAR = {"CEA": (0.1, 0.60), "GUANGZHOU": (30, 0.35), "SHANGHAI": (0.1, 0.30),
       "HUBEI": (3, 0.35), "EU": (80, 0.45)}
C_C, C_P = 0.05, 1.0


def crps_q(y, Q):
    Qs = np.sort(Q, axis=1)
    ind = (y[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (y[:, None] - Qs)).mean(axis=1)


def precomp(m, h):
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float))
    dt = pd.to_datetime(d.date).to_numpy()
    cut = "2024-01-01" if m in ("CEA", "GUANGZHOU", "SHANGHAI") else "2023-01-01"
    t0 = int(np.searchsorted(dt, np.datetime64(cut)))
    rec = []
    for t in range(W + h + 5, len(lp) - h, STEP):
        win = lp[t - W:t]; rr = win[h:] - win[:-h]; rd = np.diff(win)
        sdw = np.std(rr, ddof=1); sdd = np.std(rd, ddof=1)
        if sdw <= 1e-12 or sdd <= 1e-12:
            continue
        rec.append((t, float(rr.mean()), sdw, sdd,
                    np.quantile((rr - rr.mean()) / sdw, TAUS)))
    if len(rec) < 60:
        return None
    ts = np.array([r[0] for r in rec])
    return dict(ts=ts, h=h, m=m, mu=np.array([r[1] for r in rec]),
                sd=np.array([r[2] for r in rec]), sdd=np.array([r[3] for r in rec]),
                xi=np.vstack([r[4] for r in rec]), y=lp[ts + h] - lp[ts],
                tr=(ts >= W + h + 5) & (ts <= t0 - h - 2),
                te=ts >= max(W + h + 5, t0))


CC = {(m, h): precomp(m, h) for m in MK for h in HOR}
CC = {k: v for k, v in CC.items() if v is not None}


def Qof(c, idx, nm):
    h0, Hv = PAR[c["m"]]
    L = h0 / (h0 + c["h"])
    y0 = c["y"][idx]
    if nm == "RW":
        return c["sd"][idx][:, None] * c["xi"][idx]
    if nm == "EMP":
        return c["mu"][idx][:, None] + c["sd"][idx][:, None] * c["xi"][idx]
    s = c["sdd"][idx] * c["h"] ** Hv
    xg = np.tile(GV, (len(y0), 1))
    if nm == "A2 +位置":
        return c["mu"][idx][:, None] * L + c["sd"][idx][:, None] * c["xi"][idx]
    if nm == "A3 +形状":
        return c["mu"][idx][:, None] * L + c["sd"][idx][:, None] * xg
    if nm == "A4 +H":
        return c["mu"][idx][:, None] * L + s[:, None] * c["xi"][idx]
    return c["mu"][idx][:, None] * L + s[:, None] * xg


def loss(c, idx, nm):
    return crps_q(c["y"][idx], Qof(c, idx, nm))


# ---------------- 1. 经济价值 ----------------
print("=" * 150)
print("表 1  经济价值：逐期损失 L_t = 0.05*R_t + max(C_t - exp(R_t), 0)，R_t = Q_t(0.95)")
print("      eco_oracle = mean(L) / mean(L_oracle_R) - 1，oracle_R = 测试段 y 的单一样本 q95（用测试信息，仅作归一化）")
print("=" * 150)
print(f"{'市场':>10}{'h':>5}{'n_te':>6}{'R:A5':>8}{'R:EMP':>8}{'R:naive':>8}"
      f"{'L:A5':>9}{'L:EMP':>9}{'L:RW':>9}{'L:orc':>9}"
      f"{'eco':>8}{'DM vsEMP':>10}{'p':>8}")
ECO = {}


def dml(d, L=10):
    n = len(d)
    if n < 20:
        return np.nan, np.nan
    m = d.mean(); dc = d - m
    lrv = dc @ dc / n
    for l in range(1, L + 1):
        lrv += 2 * (1 - l / (L + 1)) * (dc[l:] @ dc[:-l] / n)
    if lrv <= 0:
        return np.nan, np.nan
    from scipy.stats import t as tdist
    s = m / np.sqrt(lrv / n) * np.sqrt((n + 1 - 2 * L + L * (L - 1) / n) / n)
    return float(s), float(2 * (1 - tdist.cdf(abs(s), df=n - 1)))


for m in MK:
    for h in HOR:
        if (m, h) not in CC:
            continue
        c = CC[(m, h)]; te = c["te"]; y = c["y"][te]
        C = np.exp(y)

        def Lt(R):
            return C_C * R + C_P * np.maximum(C - np.exp(R), 0)

        Ls = {}
        Rs = {}
        for nm in ("A5 完整", "EMP", "RW", "A3 +形状", "A4 +H"):
            R = Qof(c, te, nm)[:, I95]
            Ls[nm] = Lt(R)
            Rs[nm] = R
        Rnv = np.full(len(y), np.quantile(c["y"][c["tr"]], 0.95))
        Ls["naive"] = Lt(Rnv)
        Rorc = np.full(len(y), np.quantile(y, 0.95))
        Lorc = Lt(Rorc)
        eco = float(Ls["A5 完整"].mean() / Lorc.mean() - 1)
        s_, p_ = dml(Ls["A5 完整"] - Ls["EMP"])
        ECO[f"{m}|{h}"] = dict(
            L={k: float(v.mean()) for k, v in Ls.items()}, eco_oracle=eco,
            R={k: float(v.mean()) for k, v in Rs.items()},
            R_naive=float(Rnv[0]), R_oracle=float(Rorc[0]),
            dm_emp=[s_, p_], n_te=int(te.sum()))
        print(f"{m:>10}{h:>5}{int(te.sum()):>6}{Rs['A5 完整'].mean():>8.4f}"
              f"{Rs['EMP'].mean():>8.4f}{Rnv[0]:>8.4f}"
              f"{Ls['A5 完整'].mean()*100:>8.3f}%{Ls['EMP'].mean()*100:>8.3f}%"
              f"{Ls['RW'].mean()*100:>8.3f}%{Lorc.mean()*100:>8.3f}%"
              f"{eco*100:>7.2f}%{s_:>10.2f}{p_:>8.3f}")

print("\n" + "=" * 150)
for nm in ("A5 完整", "A3 +形状", "A4 +H", "EMP", "RW", "naive"):
    rr = [ECO[k]["L"][nm] / ECO[k]["L"]["A5 完整"] - 1 for k in ECO]
    print(f"  相对 A5 的逐期损失：{nm:<9} 高于A5 {sum(1 for x in rr if x>0)}/{len(rr)}，"
          f"中位 {np.median(rr)*100:+.2f}%")
print(f"  A5 平均逐期损失：{np.mean([ECO[k]['L']['A5 完整'] for k in ECO])*100:.3f}%（占名义额）"
      f"，EMP {np.mean([ECO[k]['L']['EMP'] for k in ECO])*100:.3f}%"
      f"，RW {np.mean([ECO[k]['L']['RW'] for k in ECO])*100:.3f}%")
dd = [ECO[k]["L"]["A5 完整"] - ECO[k]["L"]["EMP"] for k in ECO]
print(f"  A5 - EMP 逐期损失差：胜 {sum(1 for x in dd if x<0)}/{len(dd)}，"
      f"中位 {np.median(dd)*100:+.3f} pp，均值 {np.mean(dd)*100:+.3f} pp")
ddr = [ECO[k]["L"]["A5 完整"] - ECO[k]["L"]["RW"] for k in ECO]
print(f"  A5 - RW  逐期损失差：胜 {sum(1 for x in ddr if x<0)}/{len(ddr)}，"
      f"中位 {np.median(ddr)*100:+.3f} pp，均值 {np.mean(ddr)*100:+.3f} pp")
nneg = sum(1 for k in ECO if ECO[k]["dm_emp"][1] < 0.05 and ECO[k]["dm_emp"][0] < 0)
npos = sum(1 for k in ECO if ECO[k]["dm_emp"][1] < 0.05 and ECO[k]["dm_emp"][0] > 0)
print(f"  A5 的逐期损失显著低于 EMP：{nneg}/{len(ECO)}；显著高于 EMP：{npos}/{len(ECO)}")

# ---------------- 2. MCS ----------------
print("\n" + "=" * 150)
print("表 2  MCS（Hansen-Lunde-Nason range，block bootstrap B=400, block=10, centred）")
print("=" * 150)
rng = np.random.default_rng(20261020)
B, BLK = 400, 10
MCS = {}
cnt = {n: 0 for n in MOD}


def boot_dev(Lm):
    n, J = Lm.shape
    base = Lm.mean(axis=0)
    nb = int(np.ceil(n / BLK))
    D = np.empty((B, J))
    for b in range(B):
        st = rng.integers(0, n, nb)
        idx = (st[:, None] * BLK + np.arange(BLK)[None, :]).ravel()[:n] % n
        D[b] = Lm[idx].mean(axis=0) - base
    return D


for m in MK:
    for h in HOR:
        if (m, h) not in CC:
            continue
        c = CC[(m, h)]; te = c["te"]
        Lm = np.column_stack([loss(c, te, n) for n in MOD])
        inc = list(range(len(MOD))); elim = []
        while len(inc) > 1:
            sub = Lm[:, inc]
            d = sub.mean(axis=0)
            D = boot_dev(sub)
            se = D.std(axis=0, ddof=1); se[se <= 0] = np.inf
            dp = d - d.min()
            ts = dp / se
            tb = (D - D.min(axis=1)[:, None]) / se[None, :]
            crit = float(np.quantile(tb.max(axis=1), 0.90))
            if ts.max() > crit:
                w = int(np.argmax(ts)); elim.append(MOD[inc[w]]); inc.pop(w)
            else:
                break
        for k in inc:
            cnt[MOD[k]] += 1
        MCS[f"{m}|{h}"] = dict(inc=[MOD[i] for i in inc], elim=elim)
        print(f"  {m:<11} h={h:>3}  存活 {len(inc)}/6 : " + ", ".join(MOD[i] for i in inc))
print("\n  MCS 存活计数（25 格）：" + "  ".join(f"{k}={v}" for k, v in cnt.items()))

# ---------------- 3. 覆盖率检验 ----------------
print("\n" + "=" * 150)
print("表 3  PICP95 的 Kupiec POF 与 Christoffersen 独立性（p 值）")
print("=" * 150)
print(f"{'市场':>10}{'h':>5}{'PICP:A5':>9}{'POF':>7}{'IND':>7}   {'PICP:EMP':>9}{'POF':>7}{'IND':>7}")
CV = {}


def kupiec(x, n, pr=0.95):
    if x == 0 or x == n:
        return 0.0
    ph = x / n
    lr = -2 * ((n - x) * np.log(1 - pr) + x * np.log(pr)
               - (n - x) * np.log(1 - ph) - x * np.log(ph))
    return float(1 - chi2.cdf(max(lr, 0), 1))


def chris(hit):
    h0, h1 = hit[:-1], hit[1:]
    n00 = int(((h0 == 0) & (h1 == 0)).sum()); n01 = int(((h0 == 0) & (h1 == 1)).sum())
    n10 = int(((h0 == 1) & (h1 == 0)).sum()); n11 = int(((h0 == 1) & (h1 == 1)).sum())
    if (n01 + n11) == 0 or (n00 + n10) == 0:
        return np.nan
    p01 = n01 / max(n01 + n00, 1); p11 = n11 / max(n11 + n10, 1)
    pii = (n01 + n11) / max(n01 + n11 + n00 + n10, 1)
    lnLA = (n01 * np.log(max(p01, 1e-12)) + n00 * np.log(max(1 - p01, 1e-12))
            + n11 * np.log(max(p11, 1e-12)) + n10 * np.log(max(1 - p11, 1e-12)))
    lnL0 = ((n01 + n11) * np.log(max(pii, 1e-12))
            + (n00 + n10) * np.log(max(1 - pii, 1e-12)))
    lr = 2 * (lnLA - lnL0)
    return float(1 - chi2.cdf(max(lr, 0), 1))


pass95 = {"A5 完整": 0, "EMP": 0, "RW": 0}
for m in MK:
    for h in HOR:
        if (m, h) not in CC:
            continue
        c = CC[(m, h)]; te = c["te"]; y = c["y"][te]
        out = {}
        for nm in ("A5 完整", "EMP", "RW"):
            Q = Qof(c, te, nm)
            hit = ((y >= Q[:, J95[0]]) & (y <= Q[:, J95[1]])).astype(int)
            pc = float(hit.mean()); n = len(hit); x = int(hit.sum())
            pf = kupiec(x, n); pi = chris(hit)
            out[nm] = (pc, pf, pi)
            if pf > 0.05 and (np.isnan(pi) or pi > 0.05):
                pass95[nm] += 1
        CV[f"{m}|{h}"] = {k: list(v) for k, v in out.items()}
        a, e = out["A5 完整"], out["EMP"]
        print(f"{m:>10}{h:>5}{a[0]:>9.3f}{a[1]:>7.3f}{a[2] if not np.isnan(a[2]) else -1:>7.3f}"
              f"   {e[0]:>9.3f}{e[1]:>7.3f}{e[2] if not np.isnan(e[2]) else -1:>7.3f}")

print("\n  同时通过 POF 与独立性检验的格数（25 格）："
      + "  ".join(f"{k}={v}" for k, v in pass95.items()))

json.dump(dict(eco=ECO, mcs=MCS, coverage=CV,
               econ_setup=dict(c_c=C_C, c_p=C_P, rule="R*=Q(0.95)", naive="train sample q95 constant"),
               mcs_setup=dict(B=B, block=BLK, alpha=0.10, centred=True, seed=20261020)),
          open(os.path.join(OUT, "FINAL_BATTERY.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 FINAL_BATTERY.json")
