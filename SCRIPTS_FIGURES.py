# -*- coding: utf-8 -*-
"""
论文图表生成（Figure 1–4 + Table 1–3 的数值）
=====================================================================
图 1 期限剖面：A15 / A5 / SD250 / EMP 相对 RW 的 CRPS 随 h 的变化（5 市场平均）
图 2 PIT 校准曲线（真实数据 vs 各模型）
图 3 覆盖校准：c80 / c95 随 h 的变化
图 4 校准零分布：真实统计量在零分布中的位置（多切分）
输出 figures/ 目录
"""
import numpy as np, pandas as pd, os, sys, json, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
DIR = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
FIG = os.path.join(DIR, "figures")
os.makedirs(FIG, exist_ok=True)
J = lambda f: json.load(open(os.path.join(DIR, f), encoding="utf-8"))
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS)
MK = ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]
MN = {"CEA": "CEA", "GUANGZHOU": "Guangzhou", "SHANGHAI": "Shanghai", "HUBEI": "Hubei", "EU": "EU"}
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

# ============ 图 1：期限剖面 ============
HG = J("HORIZON_GRID.json")
HS = sorted({r["h"] for r in HG})
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for key, lab, col, ls in (("A14", "A15 (shape-corrected)", "tab:red", "-"),
                          ("A5", "A5 (Gaussian shape)", "tab:blue", "-"),
                          ("SD250", "SD250 (fixed $\\sqrt{h}$)", "tab:green", "--"),
                          ("EMP", "EMP (empirical h-step)", "tab:gray", ":")):
    ys = []
    for h in HS:
        s = [r for r in HG if r["h"] == h]
        if not s:
            ys.append(np.nan); continue
        # A15（即 JSON 里的 A14，h_sat=50 那版）也以 RW 为分母，保持同轴可比
        ys.append(np.median([r[key] / r["RW"] for r in s]))
    ax[0].plot(HS, ys, marker="o", ms=3.5, label=lab, color=col, ls=ls)
ax[0].axhline(1.0, color="k", lw=0.8, ls="--")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("Horizon $h$ (trading days)"); ax[0].set_ylabel("CRPS ratio vs random walk")
ax[0].set_title("(a) Horizon profile, median over 5 markets")
ax[0].legend(fontsize=8, loc="lower left")

for m in MK:
    s = [r for r in HG if r["m"] == m]
    s = sorted(s, key=lambda r: r["h"])
    ax[1].plot([r["h"] for r in s], [r["A14"] / r["RW"] for r in s], marker="o", ms=3, label=MN[m])
ax[1].axhline(1.0, color="k", lw=0.8, ls="--")
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("Horizon $h$ (trading days)"); ax[1].set_ylabel("CRPS ratio vs random walk")
ax[1].set_title("(b) A15 vs random walk, by market")
ax[1].legend(fontsize=8)
fig.suptitle("Figure 1. Horizon scaling of the predictive distribution", y=1.02)
fig.savefig(os.path.join(FIG, "fig1_horizon_profile.png"))
plt.close(fig)
print("写出 fig1_horizon_profile.png")

# ============ 图 2：PIT 校准 ============
MM = J("MULTIMETRIC.json")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
# 需要逐格 PIT 值；用 PIT-KS 代理 + 覆盖点
ax[0].scatter([r["h"] for r in MM if r["m"] == "GUANGZHOU"],
              [r["pitks_A14"] for r in MM if r["m"] == "GUANGZHOU"], label="A15")
ax[0].scatter([r["h"] for r in MM if r["m"] == "GUANGZHOU"],
              [r["pitks_A5"] for r in MM if r["m"] == "GUANGZHOU"], marker="^", label="A5")
ax[0].scatter([r["h"] for r in MM if r["m"] == "GUANGZHOU"],
              [r["pitks_EMP"] for r in MM if r["m"] == "GUANGZHOU"], marker="s", label="EMP")
ax[0].set_xlabel("Horizon $h$"); ax[0].set_ylabel("KS distance of PIT from uniform")
ax[0].set_title("(a) PIT uniformity, Guangzhou"); ax[0].legend(fontsize=8)
h5 = [r for r in MM if r["h"] == 5]
ax[1].bar(["A5", "A15", "EMP", "RW"],
           [np.mean([r["pitks_" + k] for r in h5]) for k in ("A5", "A14", "EMP", "RW")],
           color=["tab:blue", "tab:red", "tab:gray", "tab:green"])
ax[1].set_ylabel("Mean KS distance of PIT")
ax[1].set_title("(b) Mean PIT KS at $h=5$ (lower is better)")
fig.suptitle("Figure 2. Calibration of the predictive distribution", y=1.02)
fig.savefig(os.path.join(FIG, "fig2_pit_calibration.png"))
plt.close(fig)
print("写出 fig2_pit_calibration.png")

# ============ 图 3：覆盖随 h 变化 ============
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
# HORIZON_GRID 只存了 A14 的覆盖（c80/c95），其它模型用 MULTIMETRIC 的 5 个期限值
MMh = {h: [r for r in MM if r["h"] == h] for h in MM[0].keys() and sorted({r["h"] for r in MM})}
for k, lab, col in (("A14", "A15", "tab:red"), ("A5", "A5", "tab:blue"),
                    ("EMP", "EMP", "tab:gray"), ("RW", "RW", "tab:green")):
    ys = []
    for h in HS:
        s2 = [r for r in HG if r["h"] == h]
        if not s2:
            ys.append(np.nan); continue
        if k == "A14":
            ys.append(np.mean([r["c80"] for r in s2]))
        else:
            v = [r[f"c80_{k}"] for r in MMh.get(h, [])]
            ys.append(np.mean(v) if v else np.nan)
    ax[0].plot(HS, ys, marker="o", ms=3.5, label=lab, color=col)
    ys = []
    for h in HS:
        s2 = [r for r in HG if r["h"] == h]
        if not s2:
            ys.append(np.nan); continue
        if k == "A14":
            ys.append(np.mean([r["c95"] for r in s2]))
        else:
            v = [r[f"c95_{k}"] for r in MMh.get(h, [])]
            ys.append(np.mean(v) if v else np.nan)
    ax[1].plot(HS, ys, marker="o", ms=3.5, label=lab, color=col)
ax[0].axhline(0.80, color="k", lw=0.8, ls="--"); ax[0].set_title("(a) Empirical 80% coverage")
ax[1].axhline(0.95, color="k", lw=0.8, ls="--"); ax[1].set_title("(b) Empirical 95% coverage")
for a in ax:
    a.set_xscale("log"); a.set_xlabel("Horizon $h$"); a.set_ylabel("Coverage")
    a.legend(fontsize=8)
fig.suptitle("Figure 3. Coverage of the central intervals", y=1.02)
fig.savefig(os.path.join(FIG, "fig3_coverage.png"))
plt.close(fig)
print("写出 fig3_coverage.png")

# ============ 图 4：校准零分布 ============
FE = J("FINAL_EVIDENCE.json")
fig, ax = plt.subplots(figsize=(7, 4.2))
bs = ["A5", "EMP", "RW", "SD250", "VR"]
xs = np.arange(len(bs))
ps = [FE["final"][b]["p"] for b in bs]
ax.bar(xs, ps, color=["tab:blue" if p < 0.05 else "tab:gray" for p in ps])
ax.axhline(0.05, color="tab:red", lw=1.2, ls="--", label="5% level")
for x, p in zip(xs, ps):
    ax.text(x, p + 0.012, f"{p:.4f}", ha="center", fontsize=8)
ax.set_xticks(xs); ax.set_xticklabels(bs)
ax.set_ylabel("One-sided calibrated p-value")
ax.set_ylim(0, 1.0)
ax.set_title("Figure 4. A15 vs each benchmark under the calibrated null\n(200 shuffles, full pipeline re-run)")
ax.legend(fontsize=8)
fig.savefig(os.path.join(FIG, "fig4_calibrated_pvalues.png"))
plt.close(fig)
print("写出 fig4_calibrated_pvalues.png")

# ============ 图 5：多切分一致性 ============
MS = J("MULTISPLIT_A14.json")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
fracs = sorted({r["frac"] for r in MS})
for b, lab, col in (("EMP", "vs EMP", "tab:gray"), ("RW", "vs random walk", "tab:green"),
                    ("SD250", "vs fixed $\\sqrt{h}$", "tab:purple")):
    ys = [np.median([r["A14"] / r[b] for r in MS if r["frac"] == f]) for f in fracs]
    ax[0].plot(fracs, ys, marker="o", label=lab)
ys = [np.median([r["A14"] / r["A5"] for r in MS if r["frac"] == f]) for f in fracs]
ax[0].plot(fracs, ys, marker="s", color="tab:red", label="vs A5")
ax[0].axhline(1.0, color="k", lw=0.8, ls="--")
ax[0].set_xlabel("Training fraction (split point)"); ax[0].set_ylabel("Median CRPS ratio")
ax[0].set_title("(a) Consistency across five splits"); ax[0].legend(fontsize=8)
for h in [5, 20, 40, 60, 120]:
    ys = [np.median([r["A14"] / r["A5"] for r in MS if r["frac"] == f and r["h"] == h]) for f in fracs]
    ax[1].plot(fracs, ys, marker="o", label=f"$h={h}$")
ax[1].axhline(1.0, color="k", lw=0.8, ls="--")
ax[1].set_xlabel("Training fraction (split point)"); ax[1].set_ylabel("Median CRPS ratio vs A5")
ax[1].set_title("(b) By horizon"); ax[1].legend(fontsize=8)
fig.suptitle("Figure 5. A15 across five independently refitted splits", y=1.02)
fig.savefig(os.path.join(FIG, "fig5_multisplit.png"))
plt.close(fig)
print("写出 fig5_multisplit.png")

# ============ 表 1：数据描述 ============
p = pd.read_csv(os.path.join(DIR, "PANEL_5MARKET.csv"), parse_dates=["date"])
print("\nTable 1  数据描述")
print(f"{'市场':<12}{'n':>6}{'起':>12}{'止':>12}{'测试起点':>12}{'n_train':>10}{'n_test':>9}")
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
TBL1 = []
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    t0 = int(np.searchsorted(pd.to_datetime(d.date).to_numpy(), np.datetime64(CUT[m])))
    TBL1.append(dict(market=MN[m], n=len(d), start=str(d.date.iloc[0].date()),
                     end=str(d.date.iloc[-1].date()), cut=CUT[m], t0=t0))
    print(f"{MN[m]:<12}{len(d):>6}{str(d.date.iloc[0].date()):>12}{str(d.date.iloc[-1].date()):>12}"
          f"{CUT[m]:>12}{t0:>10}{len(d)-t0:>9}")
json.dump(TBL1, open(os.path.join(DIR, "TABLE1_DATA.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\n写出 TABLE1_DATA.json")
