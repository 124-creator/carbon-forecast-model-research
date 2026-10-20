# -*- coding: utf-8 -*-
"""
实验体系 19/6+：为什么收益集中在北京/上海/湖北？—— 机制诊断
=====================================================================
§24.32.5 的核心发现：改进**不均匀**，广州/上海/湖北显著，CEA/EU 不显著。
§24.38 进一步发现：增益来自"按期限渐进地把高斯形状换成窗内经验分位形状"，
且**形状的内容是必要的**（随机化使模型变差）。

本节要回答：**什么样的市场能从这个机制受益？**
这决定论文能否给出一个**可检验的适用条件**，而不只是"我们的数据上有效"。

候选解释（每个都给可测量的量）：
  E1 **偏度的可重复性**：窗内日收益偏度在相邻窗口之间的一致程度。
     若偏度在不同窗口间反复出现（半窗一致），形状订正才有价值。
  E2 **形状的非高斯程度**：超额峰度 / |偏度| 的量级。
     若分布接近高斯，形状订正无从发挥。
  E3 **短端收益的持续性**：h=5 的自相关是否显著（持续性越强，
     窗内形状对未来越有信息）。
  E4 **波动聚集强度**：GARCH 效应（|r_t| 的自相关）→ 形状订正的价值随期限衰减。

做法：
  1) 在**训练段**上计算每个市场的 E1–E4 指标（不看测试段）。
  2) 用**正确零假设**得到每个市场的改进强度（合并 log 比）。
  3) 跨 5 个市场 + 5 个切分（每切分 5 个市场 = 25 个观测对）做回归/相关。
  4) **留一市场交叉验证**：用 4 个市场拟合关系，第 5 个预测，检验能否预测排序。
     （这一步避免"在 5 个点上拟合 1 个关系"的小样本问题。）

输出 MECHANISM.json
"""
import numpy as np, pandas as pd, os, sys, json, warnings, time
from scipy.stats import norm, skew, kurtosis, pearsonr, spearmanr
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
SPL = [0.55, 0.65, 0.75]
NSHUF = 40
SEED0 = 112358
HSAT = 50.0


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


def shape_diag(lp, t0, n_per=200):
    """训练段上的 E1–E4 诊断量（只用到训练段）。"""
    ts_end = int(np.searchsorted(np.arange(len(lp)), t0))
    tr_lp = lp[:ts_end]
    r = np.diff(tr_lp)
    # E1 偏度的半窗一致率：把训练段分半，两半的偏度同号且比值接近
    h1 = len(r) // 2
    s1 = float(skew(r[:h1])); s2 = float(skew(r[h1:]))
    agree = 1.0 if (s1 * s2 > 0) else 0.0
    # 更细：滚动 125 日窗的偏度序列，计算一阶自相关
    Wd = 125
    sk = np.array([float(skew(r[i - Wd:i])) for i in range(Wd, len(r) + 1, 20)
                   if len(r[i - Wd:i]) > 30])
    if len(sk) > 5 and sk.std() > 0:
        e1_ac = float(np.corrcoef(sk[:-1], sk[1:])[0, 1])
    else:
        e1_ac = 0.0
    e1 = 0.5 * agree + 0.5 * (e1_ac if e1_ac == e1_ac else 0.0)
    # E2 非高斯程度
    e2 = abs(float(skew(r))) + max(0.0, float(kurtosis(r, fisher=True))) / 10.0
    # E3 h=5 收益自相关
    r5 = tr_lp[5:] - tr_lp[:-5]
    e3 = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if len(r) > 30 else 0.0
    ar5 = float(np.corrcoef(r5[:-1], r5[1:])[0, 1]) if len(r5) > 30 else 0.0
    e3 = ar5
    # E4 波动聚集：|r| 的自相关
    e4 = float(np.corrcoef(np.abs(r[:-1]), np.abs(r[1:]))[0, 1]) if len(r) > 30 else 0.0
    return dict(e1_skew_repeat=e1, e1_half_agree=agree, e1_skew_ac=e1_ac,
                e2_nongauss=e2, e3_ar5=e3, e4_vol_cluster=e4,
                e5_skew=abs(float(skew(r))))


def eval_gain(lp, t0):
    """返回 (合并 log 比 vs A5, 是否逐格有效)。"""
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
    out = {}
    for h, c in sorted(CC.items()):
        te = c["te"]
        if te.sum() < 20:
            continue
        y = c["y"][te]; mu = c["mu"][te]; sdd = c["sdd"][te]; xi = c["xi"][te]
        G = np.tile(GV, (len(y), 1))
        S = sdd * h ** Hv
        lam = h0 / (h0 + h)
        w = min(1.0, h / HSAT)
        l5 = crps_q(y, mu[:, None] * lam + S[:, None] * G)
        l14 = crps_q(y, mu[:, None] * lam + S[:, None] * ((1 - w) * xi + w * G))
        out[h] = (float(l5.mean()), float(l14.mean()))
    return out


PANEL = {}
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    PANEL[m] = np.log(d.price.to_numpy(float)).astype(float)

print("=" * 116)
print("一、诊断量（训练段）与改进强度（各切分）")
print("=" * 116)
ROWS = []
for frac in SPL:
    for m in MK:
        lp = PANEL[m]
        t0 = int(len(lp) * frac)
        g = eval_gain(lp, t0)
        if not g:
            continue
        gain = float(np.mean([np.log(b / a) for a, b in g.values()]))
        dg = shape_diag(lp, t0)
        ROWS.append(dict(frac=frac, m=m, gain=gain, **dg))
        if frac == 0.75:
            print(f"  {MN[m]:<6} 改进 {gain:+.5f}   偏度重复 {dg['e1_skew_repeat']:+.3f}  "
                  f"非高斯 {dg['e2_nongauss']:.3f}  AR(5) {dg['e3_ar5']:+.3f}  "
                  f"波动聚集 {dg['e4_vol_cluster']:+.3f}")

print("\n" + "=" * 116)
print("二、诊断量与改进强度的相关性（跨 3 切分 × 5 市场 = 15 个观测）")
print("=" * 116)
gains = np.array([r["gain"] for r in ROWS])
print(f"{'诊断量':<22}{'Pearson':>10}{'p':>9}{'Spearman':>11}{'p':>9}")
for k in ["e1_skew_repeat", "e1_half_agree", "e1_skew_ac", "e2_nongauss",
          "e3_ar5", "e4_vol_cluster", "e5_skew"]:
    x = np.array([r[k] for r in ROWS])
    if x.std() == 0:
        continue
    pr, pp = pearsonr(x, gains)
    sr, sp = spearmanr(x, gains)
    print(f"{k:<22}{pr:>10.3f}{pp:>9.3f}{sr:>11.3f}{sp:>9.3f}")

print("\n" + "=" * 116)
print("三、按市场平均（每个市场 3 个切分）")
print("=" * 116)
print(f"{'市场':<8}{'改进':>10}{'偏度重复':>11}{'非高斯':>9}{'AR(5)':>9}{'波动聚集':>10}{'偏度':>9}")
for m in MK:
    s = [r for r in ROWS if r["m"] == m]
    if not s:
        continue
    f = lambda k: np.mean([r[k] for r in s])
    print(f"{MN[m]:<8}{f('gain'):>10.5f}{f('e1_skew_repeat'):>11.3f}{f('e2_nongauss'):>9.3f}"
          f"{f('e3_ar5'):>9.3f}{f('e4_vol_cluster'):>10.3f}{f('e5_skew'):>9.3f}")

print("\n" + "=" * 116)
print("四、留一市场交叉验证：用 4 个市场拟合关系，预测第 5 个市场的改进强度")
print("=" * 116)
KEY = "e1_skew_repeat"
xs = np.array([r[KEY] for r in ROWS]); ys = gains
pred = np.full(len(ys), np.nan)
for m in MK:
    tr_m = np.array([r["m"] != m for r in ROWS])
    te_m = ~tr_m
    if tr_m.sum() < 4:
        continue
    b, a = np.polyfit(xs[tr_m], ys[tr_m], 1)
    pred[te_m] = a + b * xs[te_m]
ok = ~np.isnan(pred)
if ok.sum() >= 5:
    r2 = 1 - np.sum((ys[ok] - pred[ok]) ** 2) / np.sum((ys[ok] - ys[ok].mean()) ** 2)
    rho = spearmanr(pred[ok], ys[ok])
    print(f"  LOO 预测 vs 实际：R² = {r2:.3f}   Spearman = {rho.statistic:.3f} (p={rho.pvalue:.3f})")
    print(f"  预测值: {np.round(pred[ok],5)}")
    print(f"  实际值: {np.round(ys[ok],5)}")
    # 完全没关系的基线：总均值
    r2_mean = 1 - np.sum((ys[ok] - ys[ok].mean()) ** 2) / np.sum((ys[ok] - ys[ok].mean()) ** 2)
    print(f"  （基线：只用总均值的 R² = {r2_mean:.3f}）")
    LOO = dict(r2=float(r2), spearman=float(rho.statistic), spearman_p=float(rho.pvalue),
               pred=pred[ok].tolist(), actual=ys[ok].tolist())
else:
    LOO = {}
    print("  样本不足，无法做留一市场检验")

json.dump({"rows": ROWS, "loo": LOO,
           "corr": {k: dict(pearson=float(pearsonr(np.array([r[k] for r in ROWS]), gains)[0]),
                            p_pearson=float(pearsonr(np.array([r[k] for r in ROWS]), gains)[1]),
                            spearman=float(spearmanr(np.array([r[k] for r in ROWS]), gains)[0]),
                            p_spearman=float(spearmanr(np.array([r[k] for r in ROWS]), gains)[1]))
                    for k in ["e1_skew_repeat", "e1_half_agree", "e1_skew_ac", "e2_nongauss",
                              "e3_ar5", "e4_vol_cluster", "e5_skew"]}},
          open(os.path.join(DIR, "MECHANISM.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 MECHANISM.json")
