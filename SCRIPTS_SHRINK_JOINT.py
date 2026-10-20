# -*- coding: utf-8 -*-
"""
单一收缩参数 c 的联合检验（修正 shrink1.py 的设计错误）
=====================================================
shrink1.py 的致命问题：c 逐格选，(market,h) 每格一个自由参数共 20 个，
于是 w(h) 不受约束，"w 随 h 单调下降"这个检验是空的。

机制如果成立，收缩强度应由 n_eff **一个机制**解释，而不是每格一个参数。
故正确的检验是：

  c_cell    : 每格一个 c（20 个参数，非机制的）
  c_market  : 每个市场一个 c，四个期限共用（5 个参数）
  c_global  : 全部 20 格共用一个 c（1 个参数）

若 c_global 的样本外表现接近 c_cell 且优于两个端点，则"期限依赖收缩"
是一个真正的单参数模型改进，机制有内容。
若必须 c_cell 才不输，则所谓机制只是逐格调参。

网格包含端点：c=0.4 近似纯条件方差(EW97)，c=5000 近似纯长期均值(SD250)。
数据：PANEL_5MARKET.csv   输出：SHRINK_JOINT.json
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
W = 250; HOR = [5, 20, 60, 120]; STEP = 2
KS = np.arange(0.30, 2.01, 0.01)
CV = np.array([0.4, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 150, 400, 1000, 5000])
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
MK = list(CUT)


def pick_k(y, Q):
    med = Q[:, JM][:, None]
    cov = np.array([((y >= med[:, 0] + k * (Q[:, J05] - med[:, 0])) &
                     (y <= med[:, 0] + k * (Q[:, J95] - med[:, 0]))).mean() for k in KS])
    return float(KS[int(np.argmin(np.abs(cov - 0.95)))])


def apply_k(Q, k):
    med = Q[:, JM][:, None]
    return med + k * (Q - med)


def crps_q(y, Q):
    d = y[:, None] - Q
    return float(np.mean(2.0 * (TAUS[None, :] - (d < 0).astype(float)) * d))


# ---------- 第一阶段：收集每格每 c 的训练/测试 CRPS ----------
TR = {}; TE = {}; FT = {}
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    st = {}
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
        st[t] = (float(v1), float(max(v, 1e-14)))
    for h in HOR:
        neff = W / h
        ts_tr = [t for t in range(W + h + 5, t0 - h - 1, STEP) if t in st]
        ts_te = [t for t in range(max(W + h + 5, t0), len(lp) - h, STEP) if t in st]
        if len(ts_tr) < 30 or len(ts_te) < 30:
            continue
        y_tr = lp[np.array(ts_tr) + h] - lp[ts_tr]
        y_te = lp[np.array(ts_te) + h] - lp[ts_te]
        s_tr = np.array([st[t] for t in ts_tr]); s_te = np.array([st[t] for t in ts_te])
        ctr = np.zeros(len(CV)); cte = np.zeros(len(CV))
        for i, c in enumerate(CV):
            wgt = neff / (neff + c)
            v2_tr = wgt * s_tr[:, 1] + (1 - wgt) * s_tr[:, 0]
            v2_te = wgt * s_te[:, 1] + (1 - wgt) * s_te[:, 0]
            k = pick_k(y_tr, np.sqrt(h * v2_tr[:, None]) * GV[None, :])
            ctr[i] = crps_q(y_tr, apply_k(np.sqrt(h * v2_tr[:, None]) * GV[None, :], k))
            cte[i] = crps_q(y_te, apply_k(np.sqrt(h * v2_te[:, None]) * GV[None, :], k))
        TR[(m, h)] = ctr; TE[(m, h)] = cte; FT[(m, h)] = neff

CELLS = sorted(TR)
print(f"{'='*150}")
print("第一阶段：每格训练期 CRPS 关于 c 的profile（看最优是否内部、是否平坦）")
print(f"{'格':<18}{'n_eff':>7}{'最优c':>8}{'w':>7}{'训练CRPS':>11}"
      f"{'c=0.4(COND)':>13}{'c=5000(LONG)':>14}   曲率")
for c_ in CELLS:
    tr = TR[c_]; i = int(np.argmin(tr)); neff = FT[c_]
    # 最优相对两端点的改善
    imp = 100 * (tr[i] / min(tr[0], tr[-1]) - 1)
    print(f"{c_[0]+' h='+str(c_[1]):<18}{neff:>7.2f}{CV[i]:>8g}{neff/(neff+CV[i]):>7.3f}"
          f"{tr[i]:>11.5f}{tr[0]:>13.5f}{tr[-1]:>14.5f}   {imp:>+6.2f}%")

# ---------- 第二阶段：三种层级的 c 选择 ----------
print(f"\n{'='*150}")
print("第二阶段：参数层级——逐格 / 逐市场 / 全局 一个 c")
best_cell = {c_: CV[int(np.argmin(TR[c_]))] for c_ in CELLS}
print(f"\n  c_cell（每格一个，共 {len(CELLS)} 个参数）：")
for m in MK:
    print(f"    {m:<10} " + "  ".join(f"h={h}:{best_cell[(m,h)]:g}" for h in HOR if (m, h) in CELLS))

c_market = {}
for m in MK:
    hs = [h for h in HOR if (m, h) in CELLS]
    tot = sum(TR[(m, h)] for h in hs)
    c_market[m] = CV[int(np.argmin(tot))]
print(f"\n  c_market（每市场一个，四期限共用，共 {len(MK)} 个参数）：")
for m in MK:
    print(f"    {m:<10} c*={c_market[m]:<6g}")

tot_all = sum(TR[c_] for c_ in CELLS)
c_gl = CV[int(np.argmin(tot_all))]
print(f"\n  c_global（全部共用，共 1 个参数）：c* = {c_gl:g}")
print(f"\n  {'格':<18}{'c_cell':>8}{'c_market':>10}{'c_global':>10}"
      f"{'测试 CRPS':>12}{'vs 最优端点':>13}")
rows = {}
for c_ in CELLS:
    m, h = c_
    i_gl = list(CV).index(c_gl)
    te_gl = TE[c_][i_gl]
    ref = min(TE[c_][0], TE[c_][-1])
    rows[f"{m}|{h}"] = dict(test=float(te_gl), c_global=float(c_gl),
                            c_market=float(c_market[m]), c_cell=float(best_cell[c_]),
                            ref=float(ref), vs_ref=float(100*(te_gl/ref-1)))
    print(f"{m+' h='+str(h):<18}{best_cell[c_]:>8g}{c_market[m]:>10g}{c_gl:>10g}"
          f"{te_gl:>12.5f}{100*(te_gl/ref-1):>+12.2f}%")

print(f"\n{'='*150}")
print("第三阶段：三种层级各自相对两个端点的测试期表现（汇总 20 格）")
print(f"{'方案':<26}{'参数个数':>9}{'优于两端':>10}{'优于LONG':>10}{'优于COND':>10}{'中位':>10}")
def summarize(name, idx_of, nparam):
    bw = bl = bc = 0; med = []
    for c_ in CELLS:
        te = TE[c_][idx_of(c_)]
        lo, co = TE[c_][-1], TE[c_][0]
        bl += te < lo; bc += te < co; bw += (te < lo and te < co)
        med.append(100 * (te / min(lo, co) - 1))
    print(f"{name:<26}{nparam:>9}{bw:>7}/{len(CELLS):<3}{bl:>7}/{len(CELLS):<3}"
          f"{bc:>7}/{len(CELLS):<3}{np.median(med):>+9.2f}%")
    return bw

summarize("端点 LONG (SD250)", lambda c_: len(CV) - 1, 0)
summarize("端点 COND (EW97)", lambda c_: 0, 0)
summarize("c_cell  逐格", lambda c_: int(np.argmin(TR[c_])), len(CELLS))
summarize("c_market 逐市场", lambda c_: list(CV).index(c_market[c_[0]]), len(MK))
summarize("c_global 全局", lambda c_: list(CV).index(c_gl), 1)

print(f"\n{'='*150}")
print("第四阶段：全局 c 的 profile——若曲线在内部且有明确极小，机制有内容；")
print("          若单调或平台，机制无内容")
print(f"{'c':>8}{'w(h=5)':>9}{'w(h=20)':>9}{'w(h=60)':>9}{'w(h=120)':>10}"
      f"{'训练CRPS合计':>15}{'测试CRPS合计':>15}")
tr_all = np.zeros(len(CV)); te_all = np.zeros(len(CV))
for i, c in enumerate(CV):
    tr_all[i] = sum(TR[c_][i] for c_ in CELLS)
    te_all[i] = sum(TE[c_][i] for c_ in CELLS)
    print(f"{c:>8g}" + "".join(f"{W/h/(W/h+c):>9.3f}" for h in HOR)
          + f"{tr_all[i]:>15.5f}{te_all[i]:>15.5f}")
it = int(np.argmin(tr_all)); ie = int(np.argmin(te_all))
print(f"\n  训练期最优 c = {CV[it]:g}（w 从 {W/5/(W/5+CV[it]):.3f} 降到 "
      f"{W/120/(W/120+CV[it]):.3f}）")
print(f"  测试期最优 c = {CV[ie]:g}  ← 仅作诊断，不可用于选参")
json.dump(dict(cells=rows, c_global=float(c_gl),
               c_market={k: float(v) for k, v in c_market.items()},
               profile_c=list(map(float, CV)),
               profile_train=list(map(float, tr_all)),
               profile_test=list(map(float, te_all))),
          open(os.path.join(OUT, "SHRINK_JOINT.json"), "w"), indent=1, ensure_ascii=False)
print("\n已写出 SHRINK_JOINT.json")
