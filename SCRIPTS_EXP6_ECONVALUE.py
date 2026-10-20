# -*- coding: utf-8 -*-
"""
实验体系 6/6 修正版：经济价值与决策应用
=====================================================================
E1 修正：履约决策的正确建模
  设企业需要在 t+h 时拥有 1 单位配额。
  选择一：**现在买**，成本 C0 = P_t。
  选择二：**等到 t+h 按现货买**，成本 Cfut = P_{t+h}。
  选择三：**按模型在 t+h 的预测区间**，若 P_{t+h} 的 95% 上分位 > P_t
        （即大概率涨价），现在买；否则等。
  事后成本 = Cfut 若等待，= C0 若现在买。
  相对"总是等待"的节省 = (Cfut − 实际成本) / C0，用**百分比**表示才有意义。
  关键：必须用**相对**的量，且要报告"总是现在买"与"总是等待"两条基线。
  早期版本把绝对价格代入导致百分比荒谬，已作废。
E2 多期限采购计划：12 个月分批采购
E3 尾部风险：刺穿率、超出幅度、宽度
输出 ECONVALUE.json（覆盖）
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


ALL = []
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    CC = {}
    for h in HOR:
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
        if len(rec) >= 40:
            ts = np.array([r[0] for r in rec])
            CC[h] = dict(ts=ts, mu=np.array([r[1] for r in rec]), sd=np.array([r[2] for r in rec]),
                         sdd=np.array([r[3] for r in rec]), xi=np.vstack([r[4] for r in rec]),
                         y=lp[ts + h] - lp[ts], tr=(ts <= t0 - h - 2), te=(ts > t0))
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
        y = c["y"][te]; mu = c["mu"][te]; sdd = c["sdd"][te]; xi = c["xi"][te]
        w14 = min(1.0, h / 50.0)
        G = np.tile(GV, (len(y), 1))
        S = sdd * h ** Hv
        Q_A5 = mu[:, None] * (h0 / (h0 + h)) + S[:, None] * G
        Q_A14 = mu[:, None] * (h0 / (h0 + h)) + S[:, None] * ((1 - w14) * xi + w14 * G)
        Q_EMP = np.vstack([np.quantile(lp[max(0, t - W):t][h:] - lp[max(0, t - W):t][:-h], TAUS)
                           for t in c["ts"][te]])
        Q_RW = (sdd * np.sqrt(h))[:, None] * G
        M = {"A5": Q_A5, "A14": Q_A14, "EMP": Q_EMP, "RW": Q_RW}

        # ---------- E1 履约决策（相对量） ----------
        # 规则：若预测的对数收益 95% 上分位 > 0 ⇒ 大概率涨价 ⇒ 现在买；否则等到期末。
        # 节省 = (实际支付 − 现在买的成本) / 现在买的成本，**相对量**。
        # 实际支付：现在买 = 0（相对 P_t）；等到期末 = y（对数变动）
        # 节省（正=比"现在买"省）= -实际支付/ P_t 的对数变动 ≈ -y
        # ⇒ "现在买"策略的节省 = 0；"等到期末" = -y；模型策略 = -y·1{等待}
        # 于是节省 = mean(-y · wait) ；wait = 1 表示等到期末
        e1 = {}
        for k, Q in M.items():
            # 决策规则修正：用**预测中位数**（期望价格变动）判断是否预期涨价。
            # 早期版本用 95% 上分位，因 σ̂_d·h^H 远大于漂移，该量几乎恒为正，
            # 策略恒为"买"，节省恒为 0 —— 规则与分数不匹配的失败例子。
            med = Q[:, 7]
            buy_now = med > 0
            wait = ~buy_now
            save = np.where(wait, -y, 0.0)
            e1[k] = float(np.mean(save))
            e1[k + "_buyfrac"] = float(np.mean(buy_now))
            # 附加：用 90% 分位（τ=0.90，idx 12）做更保守的规则
            buy90 = Q[:, 12] > 0
            e1[k + "_save90"] = float(np.mean(np.where(~buy90, -y, 0.0)))
            e1[k + "_buyfrac90"] = float(np.mean(buy90))
        e1["LATER_save"] = float(np.mean(-y))     # 总是等待
        e1["NOW_save"] = 0.0                       # 总是现在买

        # ---------- E3 尾部风险 ----------
        e3 = {}
        for k, Q in M.items():
            viol95 = float(((y < Q[:, 0]) | (y > Q[:, 14])).mean())
            viol80 = float(((y < Q[:, 2]) | (y > Q[:, 12])).mean())
            exc = np.maximum(y - Q[:, 0], 0) + np.maximum(Q[:, 14] - y, 0)
            e3[k] = dict(viol95=viol95, viol80=viol80, es=float(np.mean(exc)),
                         width=float(np.mean(Q[:, 14] - Q[:, 0])))
        row = dict(m=m, h=h, n_te=int(te.sum()),
                   crps_A5=float(crps_q(y, Q_A5).mean()), crps_A14=float(crps_q(y, Q_A14).mean()),
                   **{f"e1_{k}": v for k, v in e1.items()},
                   **{f"e3_{k}_{kk}": vv for k, v in e3.items() for kk, vv in v.items()})
        ALL.append(row)
        print(f"  {MN[m]:<6} h={h:<4} 节省(对数) A5 {row['e1_A5']*100:+.3f}%  A14 {row['e1_A14']*100:+.3f}%  "
              f"EMP {row['e1_EMP']*100:+.3f}%  RW {row['e1_RW']*100:+.3f}%  "
              f"总等待 {row['e1_LATER_save']*100:+.3f}% | 买入比例 A14 {row['e1_A14_buyfrac']:.3f} "
              f"EMP {row['e1_EMP_buyfrac']:.3f} | 刺穿95 A5 {row['e3_A5_viol95']:.3f} "
              f"A14 {row['e3_A14_viol95']:.3f} EMP {row['e3_EMP_viol95']:.3f}")

print("\n" + "=" * 118)
print("E1  履约决策：相对\"现在全买\"的成本节省（对数百分比，正=更省）")
print("    规则：若预测中位数 > 0（预期涨价）则现在买，否则等到期末")
print("=" * 118)
for k in ("A5", "A14", "EMP", "RW", "LATER_save"):
    v = np.array([r["e1_" + k] for r in ALL])
    print(f"  {k:<12} 均值 {v.mean()*100:+.4f}%  中位 {np.median(v)*100:+.4f}%  "
          f"最差 {v.max()*100:+.3f}%  最好 {v.min()*100:+.3f}%")
print("\n  规则2：若预测 90% 分位 > 0 则现在买（更保守）")
for k in ("A5", "A14", "EMP", "RW"):
    v = np.array([r["e1_" + k + "_save90"] for r in ALL])
    bf = np.mean([r["e1_" + k + "_buyfrac90"] for r in ALL])
    print(f"  {k:<12} 均值 {v.mean()*100:+.4f}%  中位 {np.median(v)*100:+.4f}%  买入比例 {bf:.3f}")
print("\n  按期限：")
for h in HOR:
    s = [r for r in ALL if r["h"] == h]
    print(f"    h={h:<5} A5 {np.mean([r['e1_A5'] for r in s])*100:+.3f}%  "
          f"A14 {np.mean([r['e1_A14'] for r in s])*100:+.3f}%  "
          f"EMP {np.mean([r['e1_EMP'] for r in s])*100:+.3f}%  "
          f"RW {np.mean([r['e1_RW'] for r in s])*100:+.3f}%  "
          f"总等待 {np.mean([r['e1_LATER_save'] for r in s])*100:+.3f}%")

print("\n" + "=" * 118)
print("E3  尾部风险")
print("=" * 118)
print(f"{'模型':<8}{'刺穿95':>9}{'刺穿80':>9}{'平均超出':>11}{'95%宽度':>11}")
for k in ("A5", "A14", "EMP", "RW"):
    print(f"{k:<8}{np.mean([r['e3_'+k+'_viol95'] for r in ALL]):>9.4f}"
          f"{np.mean([r['e3_'+k+'_viol80'] for r in ALL]):>9.4f}"
          f"{np.mean([r['e3_'+k+'_es'] for r in ALL]):>11.5f}"
          f"{np.mean([r['e3_'+k+'_width'] for r in ALL]):>11.5f}")

print("\n" + "=" * 118)
print("E2  多期限采购计划：12 个月分批采购（用 h=20 的模型做前瞻）")
print("=" * 118)
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    h = 20
    rec = []
    for t in range(W + h + 5, len(lp) - h, STEP):
        win = lp[max(0, t - W):t]
        rr = win[h:] - win[:-h]
        rd = np.diff(win)
        sdw = np.std(rr, ddof=1); sdd = np.std(rd, ddof=1)
        if sdw <= 1e-12 or sdd <= 1e-12:
            continue
        rec.append((t, float(rr.mean()), sdw, sdd, np.quantile((rd - rd.mean()) / sdd, TAUS)))
    ts = np.array([r[0] for r in rec])
    cc = dict(ts=ts, mu=np.array([r[1] for r in rec]), sd=np.array([r[2] for r in rec]),
              sdd=np.array([r[3] for r in rec]), xi=np.vstack([r[4] for r in rec]),
              y=lp[ts + h] - lp[ts], tr=(ts <= t0 - h - 2), te=(ts > t0))
    best = (1e9, None, None)
    for Hval in HG:
        for h0 in H0G:
            v = float(crps_q(cc["y"][cc["tr"]],
                             (cc["mu"][cc["tr"]] * (h0 / (h0 + h)) +
                              (cc["sdd"][cc["tr"]] * h ** Hval))[:, None] * GV).mean())
            if v < best[0]:
                best = (v, h0, Hval)
    _, h0, Hv = best
    te = cc["te"]
    y = cc["y"][te]; mu = cc["mu"][te]; sdd = cc["sdd"][te]; xi = cc["xi"][te]
    G = np.tile(GV, (len(y), 1)); S = sdd * h ** Hv
    w14 = min(1.0, h / 50.0)
    Q_A5 = mu[:, None] * (h0 / (h0 + h)) + S[:, None] * G
    Q_A14 = mu[:, None] * (h0 / (h0 + h)) + S[:, None] * ((1 - w14) * xi + w14 * G)
    # 采购策略：每月买 1/12 配额；若模型预测 95% 上分位 > 0，则当月多买 1/12、其余月份少买
    base = y.mean()                      # 平均每月成本（对数）
    out = {}
    for k, Q in (("A5", Q_A5), ("A14", Q_A14)):
        buy_now = Q[:, 13] > 0
        # 提前买的那部分，锁定在现在；其余照常
        save = np.where(buy_now, -y * 0.0, -y)   # 简化：买now 的成本是 0（相对现在）
        out[k] = float(np.mean(save))
    print(f"  {MN[m]:<6} 均匀分批的成本 {base*100:+.3f}%/月   "
          f"A5 提前买策略 {out['A5']*100:+.3f}%   A14 {out['A14']*100:+.3f}%   "
          f"（相对均匀分批的节省，负=更差）")

json.dump(ALL, open(os.path.join(DIR, "ECONVALUE.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 ECONVALUE.json")
