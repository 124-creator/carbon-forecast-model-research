# -*- coding: utf-8 -*-
"""
生成 §24.24 整节 markdown。所有数字由本脚本计算并直接写入，
不经过人工转录，避免抄写错误。
"""
import numpy as np, pandas as pd, os, sys, json, warnings
from scipy.stats import norm
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

OUT = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(OUT, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS)
MK = ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]
PAR = {"CEA": (0.1, 0.60), "GUANGZHOU": (30, 0.35), "SHANGHAI": (0.1, 0.30),
       "HUBEI": (3, 0.35), "EU": (80, 0.45)}
STEP = 2
DATA = {}
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float))
    dt = pd.to_datetime(d.date).to_numpy()
    cut = "2024-01-01" if m in ("CEA", "GUANGZHOU", "SHANGHAI") else "2023-01-01"
    DATA[m] = (lp, int(np.searchsorted(dt, np.datetime64(cut))))


def crps_row(y, Q):
    Qs = np.sort(Q, axis=1)
    ind = (y[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (y[:, None] - Qs)).mean(axis=1)


def dm_hln(a, b, L):
    d = a - b; n = len(d); db = d.mean()
    v = (d * d).mean()
    for l in range(1, min(L, n - 2) + 1):
        v += 2 * (1 - l / (L + 1)) * (d[l:] * d[:-l]).mean()
    if v <= 0:
        return 0.0, 1.0
    t = db / np.sqrt(v / n)
    hln = np.sqrt((n + 1 - 2 * L + L * (L - 1) / n) / n)
    return float(t * hln), float(2 * (1 - norm.cdf(abs(t * hln))))


def cell(m, h, W, want_extra=False):
    if h >= W - 10:
        return None
    lp, t0 = DATA[m]
    h0, Hv = PAR[m]
    lo = max(W + h + 5, t0)
    ts = list(range(lo, len(lp) - h, STEP))
    if len(ts) < 30:
        return None
    Qa, Qe, Qw, y = [], [], [], []
    for t in ts:
        win = lp[t - W:t]
        rr = win[h:] - win[:-h]; rd = np.diff(win)
        mu = rr.mean(); sdh = np.std(rr, ddof=1); sdd = np.std(rd, ddof=1)
        xi = np.quantile((rr - rr.mean()) / sdh, TAUS)
        Qa.append(mu * (h0 / (h0 + h)) + sdd * h ** Hv * GV)
        Qe.append(mu + sdh * xi)
        Qw.append(sdd * np.sqrt(h) * GV)
        y.append(lp[t + h] - lp[t])
    y = np.array(y); Qa = np.array(Qa); Qe = np.array(Qe)
    La, Le, Lw = crps_row(y, Qa), crps_row(y, Qe), crps_row(y, Qw)
    r = dict(n=len(y), neff=len(y) * STEP / h,
             a5=float(La.mean()), emp=float(Le.mean()), rw=float(Lw.mean()),
             cov_a5=float(np.mean((y >= Qa[:, 0]) & (y <= Qa[:, -1]))),
             cov_emp=float(np.mean((y >= Qe[:, 0]) & (y <= Qe[:, -1]))),
             p_emp=dm_hln(La, Le, 10)[1], p_rw=dm_hln(La, Lw, 10)[1])
    if want_extra:
        ylo, ymd, yhi = np.percentile(y, [2.5, 50, 97.5])
        span = yhi - ylo
        def st(Q):
            return (float(np.median(Q[:, -1] - Q[:, 0])),
                    float(np.median(Q[:, 7])),
                    float(np.median((Q[:, -1] + Q[:, 0] - 2 * Q[:, 7])
                                    / np.maximum(Q[:, -1] - Q[:, 0], 1e-12))))
        wa, ma, sa = st(Qa); we, me, se = st(Qe)
        r.update(w_a5=wa / span, w_emp=we / span, m_a5=ma - ymd, m_emp=me - ymd,
                 s_a5=sa, s_emp=se, s_y=(yhi + ylo - 2 * ymd) / span)
    return r


# ---------- 1) 期限剖面 W=250 ----------
PH = [1, 2, 3, 5, 10, 20, 40, 60, 120]
P = {}
for h in PH:
    for m in MK:
        r = cell(m, h, 250, want_extra=True)
        if r:
            P[(m, h)] = r
print("profile done", len(P))

def med(h, k):
    v = [P[(m, h)][k] for m in MK if (m, h) in P]
    return float(np.median(v)) if v else float("nan")

rows1 = []
for h in PH:
    ms = [m for m in MK if (m, h) in P]
    re = np.array([P[(m, h)]["a5"] / P[(m, h)]["emp"] for m in ms])
    rr = np.array([P[(m, h)]["a5"] / P[(m, h)]["rw"] for m in ms])
    rows1.append((h, np.mean([P[(m, h)]["neff"] for m in ms]), float(np.median(re)),
                  int((re < 1).sum()), len(ms), float(np.median(rr)), int((rr < 1).sum()),
                  med(h, "cov_a5"), med(h, "cov_emp")))
T1 = "\n".join(
    f"| {h} | {ne:.1f} | {a:.3f} | {w}/{n} | {r:.3f} | {wr}/{n} | {ca:.3f} | {ce:.3f} |"
    for h, ne, a, w, n, r, wr, ca, ce in rows1)

# ---------- 2) 安慰剂 ----------
T2 = []
for h in (1, 2, 3, 5):
    ms = [m for m in MK if (m, h) in P]
    re = np.array([P[(m, h)]["a5"] / P[(m, h)]["emp"] for m in ms])
    T2.append((h, float(np.median(re)), float(re.min()), float(re.max()), int((re < 1).sum()), len(ms)))
T2s = "\n".join(f"| {h} | {a:.3f} | [{lo:.3f}, {hi:.3f}] | {w}/{n} |"
                for h, a, lo, hi, w, n in T2)

# ---------- 3) W 网格 ----------
WG = [125, 250, 500]
HG = [2, 5, 10, 12, 20, 30, 40, 60, 80, 120, 160, 200]
G = {}
for W in WG:
    for h in HG:
        cc = [cell(m, h, W) for m in MK]
        cc = [c for c in cc if c]
        if len(cc) == len(MK):
            G[(W, h)] = dict(
                re=float(np.median([c["a5"] / c["emp"] for c in cc])),
                rr=float(np.median([c["a5"] / c["rw"] for c in cc])),
                nwin_emp=int(sum(c["a5"] < c["emp"] for c in cc)),
                emp_m=float(np.median([c["emp"] for c in cc])),
                a5_m=float(np.median([c["a5"] for c in cc])))
print("wgrid done", len(G))

T3 = ["| $h$ | " + " | ".join(f"$W$={W}" for W in WG) + " | 三档极差 |",
      "|---:|" + "---:|" * (len(WG) + 1)]
for h in HG:
    v = [G[(W, h)]["re"] for W in WG if (W, h) in G]
    if not v:
        continue
    T3.append(f"| {h} | " + " | ".join(
        (f"{G[(W,h)]['re']:.3f}" if (W, h) in G else "—") for W in WG)
        + f" | {max(v)-min(v):.3f} |")
T3s = "\n".join(T3)

# 持续交叉点：最小的 h 使得该 h 及其后所有 h 的比值都 <1
CROSS = {}
for W in WG:
    hs = sorted([h for (ww, h) in G if ww == W])
    best = None
    for i, h in enumerate(hs):
        if all(G[(W, x)]["re"] < 1 for x in hs[i:]):
            best = h; break
    CROSS[W] = best

# EMP 绝对 CRPS 随 W（中位）
T4 = ["| $h$ | " + " | ".join(f"$W$={W}" for W in WG) + " | $W_{500}/W_{125}$ |",
      "|---:|" + "---:|" * (len(WG) + 1)]
for h in (20, 40, 60, 120):
    if not all((W, h) in G for W in WG):
        continue
    v = [G[(W, h)]["emp_m"] for W in WG]
    T4.append(f"| {h} | " + " | ".join(f"{x:.5f}" for x in v) + f" | {v[2]/v[0]:.2f} |")
T4s = "\n".join(T4)

# ---------- 4) 宽度/位置/形状 ----------
T5 = ["| $h$ | A5 宽度比 | EMP 宽度比 | A5 中位偏差 | EMP 中位偏差 | A5 偏度 | EMP 偏度 | 实际偏度 | A5 覆盖 | EMP 覆盖 |",
      "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for h in PH:
    T5.append(f"| {h} | {med(h,'w_a5'):.3f} | {med(h,'w_emp'):.3f} | "
              f"{abs(med(h,'m_a5')):.4f} | {abs(med(h,'m_emp')):.4f} | "
              f"{med(h,'s_a5'):+.3f} | {med(h,'s_emp'):+.3f} | {med(h,'s_y'):+.3f} | "
              f"{med(h,'cov_a5'):.3f} | {med(h,'cov_emp'):.3f} |")
T5s = "\n".join(T5)

# ---------- 关键数字 ----------
d = dict(
    cross125=CROSS[125], cross250=CROSS[250], cross500=CROSS[500],
    pred125=round(125 / 11, 1), pred250=round(250 / 11, 1), pred500=round(500 / 11, 1),
    w_emp120=med(120, "w_emp"), w_a5120=med(120, "w_a5"),
    cov_emp120=med(120, "cov_emp"), cov_a5120=med(120, "cov_a5"),
    neff120=np.mean([P[(m, 120)]["neff"] for m in MK if (m, 120) in P]),
    ratio120=med(120, "a5") if False else rows1[-1][2],
    ratio1=rows1[0][2],
)
# 对 RW 显著但 EMP 击败 A5 的格子 vs 反之
sig_rw, sig_emp = [], []
for (m, h), r in P.items():
    if r["p_rw"] < 0.05 and r["a5"] < r["rw"]:
        sig_rw.append((m, h, r["p_rw"], r["a5"] / r["emp"]))
    if r["p_emp"] < 0.05 and r["a5"] < r["emp"]:
        sig_emp.append((m, h, r["p_emp"], r["a5"] / r["rw"]))
d["sig_rw"] = sig_rw; d["sig_emp"] = sig_emp

json.dump({f"{k[0]}|{k[1]}": v for k, v in P.items()},
          open(os.path.join(OUT, "WIDTH_DIAG.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
json.dump(d, open(os.path.join(OUT, "PROFILE_KEY.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)

print("\nT1\n" + T1)
print("\nT2\n" + T2s)
print("\nT3\n" + T3s)
print("\nT4\n" + T4s)
print("\nT5\n" + T5s)
print("\nCROSS", CROSS)
print("KEY", json.dumps(d, ensure_ascii=False, default=float)[:1200])

# 保存片段供写文档用
with open(os.path.join(OUT, "_TABLES_24_24.md"), "w", encoding="utf-8", newline="\n") as f:
    f.write("### T1\n" + T1 + "\n\n### T2\n" + T2s + "\n\n### T3\n" + T3s
            + "\n\n### T4\n" + T4s + "\n\n### T5\n" + T5s + "\n")
print("\ntables saved")
