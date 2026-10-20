# -*- coding: utf-8 -*-
"""
DL 序列基准：LSTM / TCN / Transformer（补 §24.22 声明的最大剩余缺口）
====================================================================
名册里已有的 DL 是 ML1/MLh（MLP，无序列结构）。本脚本补三个真序列架构。

预先声明（不看结果再调）：
 1. 输入 = 窗口内最后 K=64 个标准化日频对数收益（除以窗口 sd）+ 静态特征
    [log sigma_d, log h]（用训练段均值/方差标准化）。
 2. 目标 = h 步累计对数收益 / (sigma_d * sqrt(h))。sqrt(h) 是不需拟合的中性
    归一化，偏离 sqrt(h) 的部分由网络借 log h 自己学；A5 只拿到固定的 h^H。
 3. 损失 = 15 个分位的 pinball 均值。该值恒等于同一口径 CRPS/2，故按归一化
    单位训练等价于直接优化 CRPS，且五个期限等权（不会被 h=120 的大数值淹没）。
 4. 分位头 = 累积 softplus，保证单调，不加额外罚项。
 5. 每市场池化五个期限训练；训练/测试原点与 A5/A14 完全一致；验证集取训练
    段按时间排序的最后 20%。
 6. 每架构 3 个随机种子，**先对预测取平均再打分**（同架构内集成），并报告种子
    间离散度，不挑最好种子。
 7. primary = LSTM（预先声明）。TCN / TRF 并列；best-of-3 一并给但标注为事后选择。
输出 DL_SEQ.json，并打印审计三数（训练/测试原点交集必须为 0）。
"""
import numpy as np, pandas as pd, os, sys, json, time, warnings
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import norm, t as tdist
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
torch.set_num_threads(2)  # 小张量上多线程同步开销远大于收益

OUT = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(OUT, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS); NT = len(TAUS)
W = 250; HOR = [5, 20, 40, 60, 120]; STEP = 2; K = 64
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
PAR = {"CEA": (0.1, 0.60), "GUANGZHOU": (30, 0.35), "SHANGHAI": (0.1, 0.30),
       "HUBEI": (3, 0.35), "EU": (80, 0.45)}
SEEDS = [11, 22, 33, 44, 55]


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


class QHead(nn.Module):
    def __init__(self, din, dh=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(din, dh), nn.ReLU(), nn.Linear(dh, NT))

    def forward(self, z):
        o = self.f(z)
        base = o[:, :1]
        return torch.cat([base, base + torch.cumsum(F.softplus(o[:, 1:]), 1)], 1)


class LSTMNet(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.rnn = nn.LSTM(1, d, batch_first=True)
        self.head = QHead(d + 2)

    def forward(self, xs, xc):
        o, _ = self.rnn(xs.unsqueeze(-1))
        return self.head(torch.cat([o[:, -1, :], xc], 1))


class TCNNet(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.c1 = nn.Conv1d(1, d, 3, padding=1, dilation=1)
        self.c2 = nn.Conv1d(d, d, 3, padding=2, dilation=2)
        self.c3 = nn.Conv1d(d, d, 3, padding=4, dilation=4)
        self.head = QHead(d + 2)

    def forward(self, xs, xc):
        h = xs.unsqueeze(1)
        h = F.relu(self.c1(h)); h = F.relu(self.c2(h)); h = F.relu(self.c3(h))
        return self.head(torch.cat([h[:, :, -1], xc], 1))


class TRFNet(nn.Module):
    def __init__(self, d=32, nh=2):
        super().__init__()
        self.inp = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, K, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, nh, 64, dropout=0.0, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, 1)
        self.head = QHead(d + 2)

    def forward(self, xs, xc):
        h = self.tr(self.inp(xs.unsqueeze(-1)) + self.pos)
        return self.head(torch.cat([h[:, -1, :], xc], 1))


ARCH = {"LSTM": LSTMNet, "TCN": TCNNet, "TRF": TRFNet}


def pinball(z, Q, taus):
    dd = z[:, None] - Q
    return torch.mean(torch.maximum(taus * dd, (taus - 1) * dd))


def train_one(kind, Xs, Xc, Z, tr, va, seed, maxep=300, pat=30):
    torch.manual_seed(seed); np.random.seed(seed)
    net = ARCH[kind]()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    taus = torch.tensor(TAUS, dtype=torch.float32)
    Xsv, Xcv, Zv = Xs[va], Xc[va], Z[va]
    Xs, Xc, Z = Xs[tr], Xc[tr], Z[tr]
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    Xs_t, Xc_t, Z_t = t(Xs), t(Xc), t(Z)
    Xs_v, Xc_v, Z_v = t(Xsv), t(Xcv), t(Zv)
    n = len(tr); bs = 512
    best, best_state, bad, ep = 1e18, None, 0, 0
    for ep in range(maxep):
        perm = torch.randperm(n); net.train()
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            pinball(Z_t[j], net(Xs_t[j], Xc_t[j]), taus).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(pinball(Z_v, net(Xs_v, Xc_v), taus))
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= pat:
                break
    net.load_state_dict(best_state); net.eval()
    return net, best, ep + 1


ONLY = os.environ.get("MK_ONLY", "")
RES, T0 = {}, time.time()
for m in ([ONLY] if ONLY else list(CUT)):
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    h0, Hv = PAR[m]

    def rec(h, t):
        win = lp[t - W:t]; r = np.diff(win); sdd = np.std(r, ddof=1)
        if not np.isfinite(sdd) or sdd <= 0 or len(r) < K:
            return None
        y = lp[t + h] - lp[t]
        mu = (win[h:] - win[:-h]).mean()
        return dict(h=h, t=t, y=y, sdd=float(sdd), mu=float(mu),
                    seq=(r[-K:] / sdd).astype(np.float32),
                    s=float(sdd * np.sqrt(h)),
                    s5=float(sdd * h ** Hv))

    TR, TST, tr_ts = [], {}, {}
    for h in HOR:
        tr_ts[h] = [t for t in range(W + h + 5, t0 - h - 1, STEP)]
        te_ts = [t for t in range(max(W + h + 5, t0), len(lp) - h, STEP)]
        TST[h] = [r for r in (rec(h, t) for t in te_ts) if r is not None]
        for t in tr_ts[h]:
            r = rec(h, t)
            if r is not None:
                TR.append(r)
    overlap = sum(len(set(tr_ts[h]) & set(r["t"] for r in TST[h])) for h in HOR)
    print(f"\n=== {m} ===  训练样本 {len(TR)}  测试样本 {sum(len(v) for v in TST.values())}  "
          f"训练/测试原点交集 {overlap}")
    assert overlap == 0, "LEAK"

    def stack(rs):
        Xs = np.stack([r["seq"] for r in rs])
        Xc = np.array([[np.log(r["sdd"]), np.log(r["h"])] for r in rs], np.float32)
        Z = np.array([r["y"] / r["s"] for r in rs], np.float32)
        return Xs, Xc, Z

    Xs, Xc, Z = stack(TR)
    mu_, sd_ = Xc.mean(0), Xc.std(0) + 1e-9
    Xc = (Xc - mu_) / sd_
    order = np.argsort([r["t"] for r in TR], kind="mergesort")
    cut = int(len(TR) * 0.8)
    tr_i, va_i = order[:cut], order[cut:]

    TST_flat = [r for h in HOR for r in TST[h]]
    Xs_T = np.stack([r["seq"] for r in TST_flat])
    Xc_T = (np.array([[np.log(r["sdd"]), np.log(r["h"])] for r in TST_flat], np.float32) - mu_) / sd_
    y_T = np.array([r["y"] for r in TST_flat]); s_T = np.array([r["s"] for r in TST_flat])
    h_T = np.array([r["h"] for r in TST_flat])
    A5_T = np.array([crps_q(np.array([r["y"]]),
                            ((h0 / (h0 + r["h"])) * r["mu"] + r["s5"] * GV)[None, :])[0]
                     for r in TST_flat])
    _FR = json.load(open(os.path.join(OUT, "FULL_ROSTER.json"), encoding="utf-8"))["res"][m]
    dev = max(abs(A5_T[h_T == h].mean() - _FR[str(h)]["crps"]["A5"]) for h in HOR)
    print(f"  A5 复算 vs 名册最大偏差 {dev:.2e}")
    assert dev < 5e-5, "A5 baseline mismatch"

    P = {}
    for kind in ARCH:
        acc, sp = [], []
        for sd in SEEDS:
            net, vl, eps = train_one(kind, Xs, Xc, Z, tr_i, va_i, sd)
            with torch.no_grad():
                Qz = net(torch.tensor(Xs_T), torch.tensor(Xc_T)).numpy()
            acc.append(Qz * s_T[:, None])
            print(f"  {kind:<5} seed={sd} ep={eps:>3} val={vl:.5f}")
        P[kind] = np.mean(acc, 0)
        sp = max(float(np.abs(a - P[kind]).mean() / (np.abs(P[kind]).mean() + 1e-12)) for a in acc)
        P[kind + "_spread"] = sp
        print(f"  {kind:<5} 测试 CRPS {crps_q(y_T, P[kind]).mean():.5f}  A5 {A5_T.mean():.5f}  "
              f"种子离散 {sp:.4f}")

    RES[m] = {}
    for h in HOR:
        k = np.where(h_T == h)[0]
        row = dict(n=int(len(k)), A5=float(A5_T[k].mean()))
        for kind in ARCH:
            l = crps_q(y_T[k], P[kind][k])
            row[kind] = float(l.mean())
            row["p_" + kind] = dm(l, A5_T[k])
            row["spread_" + kind] = P[kind + "_spread"]
        RES[m][h] = row
    json.dump({mm: {str(h): v for h, v in dd.items()} for mm, dd in RES.items()},
              open(os.path.join(OUT, "DL_SEQ.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)

# ---------------- 汇总 ----------------
if ONLY:
    print(f"\n[MK_ONLY={ONLY}] 冒烟测试结束，跳过汇总")
    sys.exit(0)
FR = json.load(open(os.path.join(OUT, "FULL_ROSTER.json"), encoding="utf-8"))["res"]
MK = list(CUT); cells = [(m, h) for m in MK for h in HOR]
med = lambda a: float(np.median(a))
print("\n" + "=" * 118)
print("一、逐格（primary = LSTM）")
print("=" * 118)
print(f"{'市场':<11}{'h':>5}{'n':>5}{'A5':>10}{'LSTM':>10}{'/A5':>8}{'p':>7}"
      f"{'TCN':>10}{'/A5':>8}{'TRF':>10}{'/A5':>8}")
for m, h in cells:
    r = RES[m][h]
    print(f"{m:<11}{h:>5}{r['n']:>5}{r['A5']:>10.5f}{r['LSTM']:>10.5f}"
          f"{r['LSTM']/r['A5']:>8.3f}{r['p_LSTM']:>7.3f}"
          f"{r['TCN']:>10.5f}{r['TCN']/r['A5']:>8.3f}{r['TRF']:>10.5f}{r['TRF']/r['A5']:>8.3f}")

print("\n" + "=" * 118)
print("二、汇总（每格相对 A5；胜 = 该架构 CRPS 低于 A5 的格子数）")
print("=" * 118)
print(f"{'模型':<10}{'相对A5中位':>12}{'胜':>8}{'显著胜':>8}{'显著负':>8}"
      f"{'相对EMP中位':>13}{'相对名册最优中位':>18}{'平均排名':>10}")
BASE = [k for k in FR[MK[0]][str(HOR[0])]["crps"] if k != "A5"]
for kind in ("LSTM", "TCN", "TRF"):
    rr = np.array([RES[m][h][kind] / RES[m][h]["A5"] for m, h in cells])
    sw = sum(1 for m, h in cells if RES[m][h]["p_" + kind] < .05 and RES[m][h][kind] < RES[m][h]["A5"])
    sl = sum(1 for m, h in cells if RES[m][h]["p_" + kind] < .05 and RES[m][h][kind] > RES[m][h]["A5"])
    re = np.array([RES[m][h][kind] / FR[m][str(h)]["crps"]["EMP"] for m, h in cells])
    bo, rk = [], []
    for m, h in cells:
        v = [FR[m][str(h)]["crps"][k] for k in BASE]
        bo.append(RES[m][h][kind] / min(v))
        rk.append(sum(1 for x in v if x < RES[m][h][kind]) + 1)
    print(f"{kind:<10}{med(rr):>12.4f}{int((rr<1).sum()):>5}/25{sw:>8}{sl:>8}"
          f"{med(re):>13.4f}{med(bo):>18.4f}{np.mean(rk):>10.2f}")

print("\nbest-of-3（事后选择，不得当作预声明结论）：")
bb = np.array([min(RES[m][h][k] for k in ("LSTM", "TCN", "TRF")) / RES[m][h]["A5"] for m, h in cells])
bo = []
for m, h in cells:
    best = min(RES[m][h][k] for k in ("LSTM", "TCN", "TRF"))
    v = [FR[m][str(h)]["crps"][k] for k in BASE]
    bo.append((best / min(v), sum(1 for x in v if x < best) + 1))
print(f"  相对 A5 中位 {med(bb):.4f}  胜 {int((bb<1).sum())}/25  "
      f"相对名册最优中位 {med([x[0] for x in bo]):.4f}  平均排名 {np.mean([x[1] for x in bo]):.2f}")

print("\n逐期限（LSTM/A5）")
print(f"{'h':>5}{'LSTM/A5':>10}{'胜':>7}{'显著胜':>8}{'显著负':>8}{'TRF/A5':>10}")
for h in HOR:
    ks = [(m, h) for m in MK]
    print(f"{h:>5}{med([RES[m][h]['LSTM']/RES[m][h]['A5'] for m,_ in ks]):>10.4f}"
          f"{sum(1 for m,_ in ks if RES[m][h]['LSTM']<RES[m][h]['A5']):>4}/5"
          f"{sum(1 for m,_ in ks if RES[m][h]['p_LSTM']<.05 and RES[m][h]['LSTM']<RES[m][h]['A5']):>8}"
          f"{sum(1 for m,_ in ks if RES[m][h]['p_LSTM']<.05 and RES[m][h]['LSTM']>RES[m][h]['A5']):>8}"
          f"{med([RES[m][h]['TRF']/RES[m][h]['A5'] for m,_ in ks]):>10.4f}")

print(f"\n种子离散度（相对预测均值的平均绝对偏差，中位）："
      f"LSTM {med([RES[m][h]['spread_LSTM'] for m,h in cells]):.4f}  "
      f"TCN {med([RES[m][h]['spread_TCN'] for m,h in cells]):.4f}  "
      f"TRF {med([RES[m][h]['spread_TRF'] for m,h in cells]):.4f}")
print(f"总耗时 {(time.time()-T0)/60:.1f} 分钟")
print("已写出 DL_SEQ.json")
