# -*- coding: utf-8 -*-
"""
加固：位置项消融 + 多切分稳健性 + n_eff 单调性检验
==================================================
§19 的三个致命弱点必须现在打掉，否则不能当论文核心：

(1) 位置项是不是事后性？
    DAT 用 mu_d*h 作位置，SD_SQ 用 h 期经验均值。CEA 2024 后单边上行，
    mu_d*h 可能只是"顺势外推"。做 DAT0（mu=0）消融：
    若 DAT0 仍胜 EMP，则尺度效应的结论不受位置项污染。

(2) 单一固定切分？
    每个市场用 3 个切分日期（早/中/晚），报每一组的符号与幅度。

(3) n_eff 预言是否成立？
    判据：日频锚定的相对优势应随 h 上升。
    对 log(h) 回归 log(CRPS 比) 的斜率，分市场与合并报。

数据：PANEL_5MARKET.csv   输出：HARDEN.json
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
MODELS = ["EMP", "GAU", "SD_SQ", "DAT", "DAT0"]
SPLITS = {"CEA": ["2023-01-01", "2023-07-01", "2024-01-01"],
          "GUANGZHOU": ["2022-07-01", "2023-07-01", "2024-01-01"],
          "SHANGHAI": ["2022-07-01", "2023-07-01", "2024-01-01"],
          "HUBEI": ["2021-07-01", "2022-01-01", "2023-01-01"],
          "EU": ["2021-07-01", "2022-01-01", "2023-01-01"]}


def crps_q(yt, Q):
    Qs = np.sort(Q, axis=1)
    ind = (yt[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (yt[:, None] - Qs)).mean(axis=1)


def pick_k(y, Q):
    med = Q[:, JM][:, None]
    cov = np.array([((y >= med[:, 0] + k * (Q[:, J05] - med[:, 0])) &
                     (y <= med[:, 0] + k * (Q[:, J95] - med[:, 0]))).mean() for k in KS])
    return float(KS[int(np.argmin(np.abs(cov - 0.95)))])


def apply_k(Q, k):
    med = Q[:, JM][:, None]
    return med + k * (Q - med)


def dm(d, L=10):
    n = len(d)
    if n < 20:
        return np.nan, np.nan
    m = d.mean(); dc = d - m
    lrv = dc @ dc / n
    for l in range(1, L + 1):
        lrv += 2 * (1 - l / (L + 1)) * (dc[l:] @ dc[:-l] / n)
    if lrv <= 0:
        return np.nan, np.nan
    s = m / np.sqrt(lrv / n) * np.sqrt((n + 1 - 2 * L + L * (L - 1) / n) / n)
    return float(s), float(2 * (1 - tdist.cdf(abs(s), df=n - 1)))


ALL = {}
for m in SPLITS:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    cache = {}
    for t in range(W + 5, len(lp)):
        win = lp[max(0, t - W):t]
        r = np.diff(win)
        if len(r) < 60:
            continue
        v1 = np.var(r, ddof=1)
        if v1 <= 0:
            continue
        v = r[0] ** 2
        for i in range(1, len(r)):
            v = 0.97 * v + 0.03 * r[i - 1] ** 2
        cache[t] = dict(mu=float(r.mean()), sd=float(np.sqrt(v1)),
                        e97=float(np.sqrt(max(v, 1e-14))))
    for cut in SPLITS[m]:
        t0 = int(np.searchsorted(dt, np.datetime64(cut)))
        if t0 < W + 80 or t0 > len(lp) - 60:
            print(f"{m} cut={cut} 跳过（t0={t0}, len={len(lp)}）"); continue
        print(f"\n{'='*132}\n{m}   切分 {cut}   n_tr={t0} n_te={len(lp)-t0}")
        print(f"{'h':>4}{'n_te':>6}" + "".join(f"{n:>10}" for n in MODELS)
              + f"{'最优':>7}   DAT vs EMP / DAT0 vs EMP")
        for h in HOR:
            ts_tr = [t for t in range(W + h + 5, t0 - h - 1, STEP) if t in cache]
            ts_te = [t for t in range(W + h + 5, len(lp) - h, STEP) if t in cache]
            if len(ts_tr) < 30 or len(ts_te) < 30:
                print(f"{h:>4}  SKIP {len(ts_tr)}/{len(ts_te)}"); continue

            def mk(ts):
                out = {n: np.full((len(ts), 15), np.nan) for n in MODELS}
                for i, t in enumerate(ts):
                    sc = cache[t]
                    win = lp[max(0, t - W):t]
                    rr = win[h:] - win[:-h]
                    sd = np.std(rr, ddof=1)
                    if sd <= 0:
                        continue
                    sh = np.sqrt(h)
                    out["EMP"][i] = np.quantile(rr, TAUS)
                    out["GAU"][i] = rr.mean() + sd * GV
                    out["SD_SQ"][i] = rr.mean() + sc["sd"] * sh * GV
                    out["DAT"][i] = sc["mu"] * h + sc["sd"] * sh * GV
                    out["DAT0"][i] = sc["sd"] * sh * GV
                return out

            A_tr = mk(ts_tr); A_te = mk(ts_te)
            y_tr = lp[np.array(ts_tr) + h] - lp[ts_tr]
            y_te = lp[np.array(ts_te) + h] - lp[ts_te]
            gtr = np.isfinite(A_tr["EMP"]).all(1) & np.isfinite(A_tr["DAT"]).all(1)
            gte = np.isfinite(A_te["EMP"]).all(1) & np.isfinite(A_te["DAT"]).all(1)
            if gtr.sum() < 25 or gte.sum() < 25:
                print(f"{h:>4}  SKIPmask {gtr.sum()}/{gte.sum()}"); continue
            ks = {n: pick_k(y_tr[gtr], A_tr[n][gtr]) for n in MODELS}
            y = y_te[gte]
            L = {n: crps_q(y, apply_k(A_te[n][gte], ks[n])) for n in MODELS}
            c = {n: float(L[n].mean()) for n in MODELS}
            best = min(c, key=c.get)
            s1, p1 = dm(L["DAT"] - L["EMP"])
            s2, p2 = dm(L["DAT0"] - L["EMP"])
            s3, p3 = dm(L["SD_SQ"] - L["EMP"])
            print(f"{h:>4}{int(gte.sum()):>6}" + "".join(f"{c[n]:>10.5f}" for n in MODELS)
                  + f"{best:>7}   DAT {(c['DAT']/c['EMP']-1)*100:+7.2f}%({s1:+5.2f},p{p1:.3f})"
                    f"  DAT0 {(c['DAT0']/c['EMP']-1)*100:+7.2f}%({s2:+5.2f},p{p2:.3f})")
            ALL[(m, cut, h)] = dict(n=int(gte.sum()), best=best, **c,
                                    dm_dat=[s1, p1], dm_dat0=[s2, p2], dm_sdsq=[s3, p3])

print(f"\n{'='*132}")
print("一、位置项消融：DAT vs DAT0（若 DAT0 仍胜 EMP，则尺度结论不受漂移污染）")
print(f"{'市场':<12}{'切分':<12}{'h':>5}{'DAT/EMP':>10}{'DAT0/EMP':>11}{'DAT0更好':>10}")
wd = wd0 = n = 0; gd = []; gd0 = []
for (m, cut, h), r in sorted(ALL.items()):
    a = (r["DAT"] / r["EMP"] - 1) * 100; b = (r["DAT0"] / r["EMP"] - 1) * 100
    n += 1; wd += a < 0; wd0 += b < 0; gd.append(a); gd0.append(b)
    print(f"{m:<12}{cut:<12}{h:>5}{a:>+9.2f}%{b:>+10.2f}%{'是' if b<0 else '否':>10}")
gd = np.array(gd); gd0 = np.array(gd0)
print(f"\n  合计 {n} 格：DAT 胜 EMP {wd}/{n}（中位 {np.median(gd):+.2f}%）；"
      f"DAT0 胜 EMP {wd0}/{n}（中位 {np.median(gd0):+.2f}%）")
print(f"  位置项贡献（DAT 相对 DAT0 的额外改善）中位 {np.median(gd-gd0):+.2f}%")

print(f"\n{'='*132}")
print("二、多切分稳健性：DAT vs EMP 的符号是否随切分日期翻转")
print(f"{'市场':<12}{'h':>5}" + "".join(f"{c:>14}" for c in ["切分1", "切分2", "切分3"]) + f"{'同号':>7}")
for m in SPLITS:
    for h in HOR:
        row = [(m, c, h) for c in SPLITS[m] if (m, c, h) in ALL]
        if not row:
            continue
        v = [(ALL[k]["DAT"] / ALL[k]["EMP"] - 1) * 100 for k in row]
        same = "是" if (all(x < 0 for x in v) or all(x > 0 for x in v)) else "否"
        print(f"{m:<12}{h:>5}" + "".join(f"{x:>+13.2f}%" for x in v) + f"{same:>7}")

print(f"\n{'='*132}")
print("三、n_eff 预言检验：相对优势应随 h 上升。对 log h 回归 log(CRPS 比)")
bym = {}
for (m, cut, h), r in ALL.items():
    if r["EMP"] > 0 and r["DAT"] > 0:
        bym.setdefault(m, {}).setdefault(h, []).append(np.log(r["DAT"] / r["EMP"]))
print(f"{'市场':<12}" + "".join(f"{'h='+str(h):>10}" for h in HOR) + f"{'斜率':>10}{'r':>8}")
slopes = []
for m in SPLITS:
    if m not in bym:
        continue
    xs, ys = [], []
    line = f"{m:<12}"
    for h in HOR:
        if h in bym[m]:
            v = float(np.mean(bym[m][h])) * 100
            line += f"{v:>+9.2f}%"
            xs.append(np.log(h)); ys.append(np.log(1 + v / 100))
        else:
            line += f"{'-':>10}"
    if len(xs) >= 3:
        sl = np.polyfit(xs, ys, 1)[0]; rr = np.corrcoef(xs, ys)[0, 1]
        slopes.append(sl); line += f"{sl*100:>+9.2f}%{rr:>8.3f}"
    else:
        line += f"{'-':>10}{'-':>8}"
    print(line)
# 合并（去广州）
xs, ys = [], []
for h in HOR:
    vals = [np.mean(bym[m][h]) for m in bym if m != "GUANGZHOU" and h in bym[m]]
    if vals:
        xs.append(np.log(h)); ys.append(float(np.mean(vals)))
sl = np.polyfit(xs, ys, 1)[0]; rr = np.corrcoef(xs, ys)[0, 1]
print(f"\n  四市场合并（剔除广州）斜率 {sl*100:+.2f}%（每翻倍 h，优势扩大 "
      f"{(2**sl-1)*100:+.2f}%），r = {rr:.4f}")
print(f"  预测：h=5→120 应为 {(1-((120/5)**sl-1))*100:.1f}% 量级的差距改善；"
      f"实测 h=5 {np.mean([np.mean(bym[m][5]) for m in bym if m!='GUANGZHOU'])*100:+.2f}% → "
      f"h=120 {np.mean([np.mean(bym[m][120]) for m in bym if m!='GUANGZHOU'])*100:+.2f}%")
json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in ALL.items()},
          open(os.path.join(OUT, "HARDEN.json"), "w"), indent=1, ensure_ascii=False, default=float)
print("\n已写出 HARDEN.json")
