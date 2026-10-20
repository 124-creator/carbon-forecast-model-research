# -*- coding: utf-8 -*-
"""
实验体系 20/6+：论文主切分上的规则 R 应用 + **论文主张总表**
=====================================================================
两件事：
 (1) 在**论文主切分**（CUT 定义的切分，不是等分切分）上跑**规则 R**，
     得到论文要报告的 $h_{\\text{sat}}$。这是论文里唯一要写的值。
 (2) 生成**论文主张总表**：每一条主张 + 支撑它的检验 + 数字来源（JSON 文件名），
     使论文的每一个断言都能被追溯到一次检验。

输出 PAPER_CLAIMS.json
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
HS = [5, 10, 15, 20, 30, 40, 60, 120]
CAND = [10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 200]
H0G = [0.1, 0.2, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 500]
HG = np.round(np.arange(0.15, 0.861, 0.05), 3)
MK = ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]
MN = {"CEA": "CEA", "GUANGZHOU": "广州", "SHANGHAI": "上海", "HUBEI": "湖北", "EU": "EU"}
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
NSHUF = 120
SEED0 = 31415


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


def losses(lp, t0, which):
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
    out = {"A5": {}}
    for c in CAND:
        out[f"HS{c}"] = {}
    for h, c in CC.items():
        msk = c["te"] if which == "te" else c["tr"]
        if msk.sum() < 20:
            continue
        y = c["y"][msk]; mu = c["mu"][msk]; sdd = c["sdd"][msk]; xi = c["xi"][msk]
        G = np.tile(GV, (len(y), 1))
        S = sdd * h ** Hv
        lam = h0 / (h0 + h)
        out["A5"][h] = float(crps_q(y, mu[:, None] * lam + S[:, None] * G).mean())
        for cc in CAND:
            w = min(1.0, h / cc)
            out[f"HS{cc}"][h] = float(crps_q(y, mu[:, None] * lam +
                                              S[:, None] * ((1 - w) * xi + w * G)).mean())
    return out


PANEL = {}
T0S = {}
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    PANEL[m] = np.log(d.price.to_numpy(float)).astype(float)
    T0S[m] = int(np.searchsorted(pd.to_datetime(d.date).to_numpy(), np.datetime64(CUT[m])))

TRR = {m: losses(PANEL[m], T0S[m], "tr") for m in MK}
TER = {m: losses(PANEL[m], T0S[m], "te") for m in MK}
print("=" * 112)
print("论文主切分（CUT 定义）上的规则 R")
print("=" * 112)
print("  各市场的训练段规模：", {MN[m]: T0S[m] for m in MK})

print(f"\n{'h_sat':>8}{'训练段合并':>12}")
tr_mu = {}
for c in CAND:
    tr_mu[c] = float(np.mean([np.log(TRR[m][f"HS{c}"][h] / TRR[m]["A5"][h]) for m in MK for h in HS]))
    print(f"{c:>8}{tr_mu[c]:>12.5f}")

print("\n  构建主切分的零分布（%d 次打乱，完整流程）" % NSHUF)
T0 = time.time()
NULL = []
for it in range(NSHUF):
    rng = np.random.default_rng(SEED0 + it)
    rec = {}
    for m in MK:
        lp = PANEL[m]; inc = np.diff(lp)
        lp_s = np.concatenate([[0.0], np.cumsum(inc[rng.permutation(len(inc))])])
        L = losses(lp_s, T0S[m], "te")
        if L:
            for nm in L:
                for h in L[nm]:
                    rec[(nm, h, m)] = L[nm][h]
    NULL.append(rec)
    if (it + 1) % 40 == 0:
        print(f"    第 {it+1:>3} 次  ({time.time()-T0:.0f} 秒)")

null_by = {}
for c in CAND:
    nn = []
    for rec in NULL:
        v = [np.log(rec[(f"HS{c}", h, m)] / rec[("A5", h, m)]) for m in MK for h in HS
             if (f"HS{c}", h, m) in rec and ("A5", h, m) in rec]
        if len(v) == len(MK) * len(HS):
            nn.append(float(np.mean(v)))
    null_by[c] = np.array(nn)
tr_p = {c: float(np.mean(null_by[c] <= tr_mu[c])) for c in CAND}
print(f"\n{'h_sat':>8}{'训练段p':>10}")
for c in CAND:
    print(f"{c:>8}{tr_p[c]:>10.4f}")

ok = [c for c in CAND if tr_p[c] <= 0.05]
if not ok:
    ok = [min(CAND, key=lambda c: tr_p[c])]
SEL = min(ok, key=lambda c: tr_mu[c])
print(f"\n  >>> 规则 R（p≤0.05 的候选中取训练段效应量最大者）选出 h_sat = {SEL}")
print(f"      候选集 C' = {ok}")
te_mu = {c: float(np.mean([np.log(TER[m][f'HS{c}'][h] / TER[m]['A5'][h]) for m in MK for h in HS]))
         for c in CAND}
for c in CAND:
    print(f"      h_sat={c:<5} 测试段合并 log 比 {te_mu[c]:+.5f}   校准 p = "
          f"{float(np.mean(null_by[c] <= te_mu[c])):.4f}")
oracle = min(CAND, key=lambda c: te_mu[c])
print(f"      测试段 oracle 最优 = {oracle}（{te_mu[oracle]:+.5f}）；"
      f"规则 R 选出的 {SEL} 与之差 {te_mu[SEL]-te_mu[oracle]:+.5f}")

# ---------- 论文主张总表 ----------
print("\n" + "=" * 112)
print("论文主张总表")
print("=" * 112)
CLAIMS = []


def add(cid, claim, test, number, source, caveat=""):
    CLAIMS.append(dict(id=cid, claim=claim, test=test, number=number, source=source,
                        caveat=caveat))


p_sel = float(np.mean(null_by[SEL] <= te_mu[SEL]))
add("C1", "以期限标度指数 H 代替 √h 标度是必要的",
    "合并校准检验（A14 vs 固定 √h 标度，200 次打乱）",
    "p = 0.0000，真实合并 log 比 -0.102", "FINAL_EVIDENCE.json")
add("C2", "本文模型相对随机游走显著更优", "同上（200 次打乱）",
    "p = 0.0000，真实合并 log 比 -0.105", "FINAL_EVIDENCE.json")
add("C3", "相对上一版规格（A5）有可检测但不大的改进", "同上",
    "p = 0.0300，真实合并 log 比 -0.0055", "FINAL_EVIDENCE.json",
    "改进集中在广州/上海/湖北；CEA/EU 不显著")
add("C4", "改进的机制是按期限渐进地把高斯形状换成窗内经验分位形状",
    "安慰剂：随机分位输入使模型显著变差",
    "随机分位真实合并 log 比 +0.0148（变差）", "PLACEBO_CALIB.json")
add("C5", "机制起作用的是形状的横截面内容，不是时点信息",
    "安慰剂：滞后一期 ξ_d 给出几乎相同的增益",
    "-0.00566 vs -0.00549", "PLACEBO_CALIB.json")
add("C6", "过渡阈值本身是次要的，结构才重要",
    "安慰剂：3 个随机调度中 2 个也显著",
    "p = 0.0000 / 0.0167 / 0.1000", "PLACEBO_CALIB.json",
    "此条推翻了 §24.31.4 的相反结论（订正 42）")
add("C7", "结论不依赖单一训练/测试切分",
    "规则 R 的完整链条在 5 个切分独立重跑",
    "5/5 切分测试段校准 p < 0.05", "FINAL_SPEC.json")
add("C8", "结论不依赖单一数据处理口径",
    "12 种口径（窗口/原点密度/收益口径/面板构造）的效应量方向",
    "中位 0.9928–1.0000，方向一致", "DATAROBUST.json",
    "原报的“0 格显著负”已作废（订正 43）")
add("C9", "改进不是只靠 CRPS 一个评分规则",
    "7 个评分规则（CRPS/WIS/PIT-KS/PIT-NLL/经济价值/覆盖偏差/宽度）",
    "全部同号；PIT-KS 14/25 格更优、0 格更差", "MULTIMETRIC.json")
add("C10", "改进在期限维度上是真实结构，不是插值",
    "16 个期限（5→200）密网格",
    "h=5/10/15/20/25/30/40 全部 4–5/5 胜、10 格显著胜、0 格显著负",
    "HORIZON_GRID.json")
add("C11", "长端不可能变差（构造保证）", "h>50 的格子上与 A5 逐位相同",
    "37/37 逐位相同", "HORIZON_GRID.json")
add("C12", "改进在履约决策中有实用价值",
    "履约决策规则（预测中位数 > 0 则现在买）",
    "h=5/20/40 的节省 +0.128%/+0.661%/+1.645%，优于 A5/EMP/RW",
    "ECONVALUE.json")
add("C13", "h_sat 是训练段可估计的量，不是测试段的读数",
    "规则 R 在 5 个切分上独立重跑",
    "选出 50/30/20/30/20；与测试段 oracle 最优仅差 0.43%",
    "FINAL_SPEC.json")
add("C14", "深度学习基准不构成优势来源", "LSTM/TCN/Transformer 序列基准",
    "LSTM 相对 A5 中位 1.0144，1 胜 6 负", "DL_SEQ.json")
add("C15", "没有可用的市场预筛规则", "5 个市场级诊断量的留一市场交叉验证",
    "R² 为负（-0.762）", "MECHANISM.json")
add("C16", "本文的 p 值是经过校准的（方法论贡献）",
    "把 DM 检验本身放到无信息数据上检验",
    "四种 DM 实现 p<0.05 比例 0.182–0.288，KS p=0.0000 拒绝均匀性",
    "DM_CALIB.json")
print(f"{'ID':<5}{'主张':<44}{'支撑检验':<40}{'数字':<38}")
for c in CLAIMS:
    print(f"{c['id']:<5}{c['claim']:<44}{c['test']:<40}{c['number']:<38}")
print("\n不得声称的：")
NEG = [
    "不得声称“显著优于现有方法”（相对 EMP 不显著，p=0.725）",
    "不得声称“普遍优于”（CEA/EU 不显著）",
    "不得给出任何适用条件或预筛规则（留一市场 R² 为负）",
    "不得引用未校准的渐近 DM p 值",
    "不得声称 h_sat=50 来自测试段读数（它由规则 R 在训练段选出）",
    "不得以 EMP 作为唯一或主要比较对象",
]
for x in NEG:
    print("  ✗ " + x)

json.dump({"h_sat_paper": SEL, "candidates_ok": ok,
           "tr_mu": {str(k): v for k, v in tr_mu.items()},
           "tr_p": {str(k): v for k, v in tr_p.items()},
           "te_mu": {str(k): v for k, v in te_mu.items()},
           "te_p": {str(k): float(np.mean(null_by[k] <= te_mu[k])) for k in CAND},
           "oracle": oracle, "claims": CLAIMS, "must_not_claim": NEG},
          open(os.path.join(DIR, "PAPER_CLAIMS.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 PAPER_CLAIMS.json")
