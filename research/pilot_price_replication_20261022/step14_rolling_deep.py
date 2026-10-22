"""Step 14: the deep-model comparison, re-run under rolling origin.

Step 10 used a single forward split, and step 8 already proved that a single split
can manufacture a result that vanishes under rolling origin. So the +1483% figure
has to be re-established the same way everything else here was.

Protocol: expanding training window, network refit every REFIT origins on data up
to that origin only, predictions made forward, persistence scored on the identical
origins. Two variants that differ ONLY in the regression target.

If the level-prediction penalty survives rolling origin, the architecture claim is
safe to write. If it shrinks to nothing, it was a window artefact and must be said.
"""

import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
np.random.seed(20261022)

BASE = os.path.dirname(os.path.abspath(__file__))
PANEL = (r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research"
         r"\research\fullcycle_h20_20261020\PANEL_4MARKET.csv")
START = "2014-04-04"
ORIGIN_FRAC = 0.60
REFIT = 100
WIN = 10
HID = 24
EPOCHS = 250
PATIENCE = 25
WD = 1e-4
LR = 1e-3
MIN_TRAIN = 400


def nw_t(d, lag=0):
    d = np.asarray(d, float)
    n = d.size
    e = d - d.mean()
    v = np.dot(e, e) / n
    for L in range(1, lag + 1):
        v += 2.0 * (1.0 - L / (lag + 1.0)) * np.dot(e[L:], e[:-L]) / n
    se = np.sqrt(v / n)
    return d.mean(), se, (d.mean() / se if se > 0 else np.nan)


class Net(nn.Module):
    def __init__(self, hid):
        super().__init__()
        self.lstm = nn.LSTM(1, hid, batch_first=True)
        self.head = nn.Linear(hid, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1, :]).squeeze(-1)


def build(z, idx):
    X = np.stack([z[i - WIN + 1:i + 1] for i in idx])
    return torch.tensor(X, dtype=torch.float32).unsqueeze(-1)


def train_one(z, fit_idx, mode, mu, sd, seed):
    torch.manual_seed(seed)
    X = build(z, fit_idx)
    Xn = (X - mu) / sd
    if mode == "level":
        y = torch.tensor((z[fit_idx + 1] - mu) / sd, dtype=torch.float32)
    else:
        y = torch.tensor(z[fit_idx + 1] - z[fit_idx], dtype=torch.float32)
    nva = max(60, int(0.15 * fit_idx.size))
    Xtr, ytr = Xn[:-nva], y[:-nva]
    Xva, yva = Xn[-nva:], y[-nva:]
    net = Net(HID)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.MSELoss()
    best, best_state, bad = np.inf, None, 0
    for _ in range(EPOCHS):
        net.train()
        opt.zero_grad()
        lossf(net(Xtr), ytr).backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            v = lossf(net(Xva), yva).item()
        if v < best - 1e-9:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net


df = pd.read_csv(PANEL, parse_dates=["date"])
rows = []
for mkt, g in df.groupby("market"):
    g = g.sort_values("date")
    g = g[g["date"] >= START].reset_index(drop=True)
    y = g["price"].to_numpy(float)
    if y.size < 500:
        continue
    z = np.log(y)
    n = z.size
    t0 = int(n * ORIGIN_FRAC)
    origins = np.arange(t0, n - 1)

    ae = {k: np.full(origins.size, np.nan) for k in ("level", "anchored")}
    ae_p = np.abs(y[origins + 1] - y[origins])

    for bi, lag_start in enumerate(range(0, origins.size, REFIT)):
        t = origins[lag_start]
        fit_idx = np.arange(WIN, t)                 # s+1 <= t
        if fit_idx.size < MIN_TRAIN:
            continue
        mu, sd = float(z[:t].mean()), float(z[:t].std())
        seg = slice(lag_start, min(lag_start + REFIT, origins.size))
        oidx = origins[seg]
        Xte = build(z, oidx)
        for mode in ("level", "anchored"):
            net = train_one(z, fit_idx, mode, mu, sd, 20261022 + bi)
            with torch.no_grad():
                p = net((Xte - mu) / sd).numpy()
            pred = p * sd + mu if mode == "level" else z[oidx] + p
            ae[mode][seg] = np.abs(y[oidx + 1] - np.exp(pred))
        del Xte

    ok = np.isfinite(ae["level"]) & np.isfinite(ae["anchored"])
    for mode in ("level", "anchored"):
        d = ae[mode][ok] - ae_p[ok]
        m, se, tv = nw_t(d)
        rows.append(dict(market=mkt, model=mode, n=int(ok.sum()),
                         MAE=float(ae[mode][ok].mean()),
                         MAE_pers=float(ae_p[ok].mean()),
                         dMAE=float((ae[mode][ok].mean() / ae_p[ok].mean() - 1) * 100),
                         t=tv))
    print("done %s (n=%d)" % (mkt, ok.sum()))

res = pd.DataFrame(rows)
print("\n--- rolling-origin deep models vs persistence, h=1 ---")
print("%-10s %-9s %5s %12s %12s %9s %8s"
      % ("market", "model", "n", "MAE", "MAE_pers", "dMAE%", "NW_t"))
for _, r in res.iterrows():
    print("%-10s %-9s %5d %12.4f %12.4f %+9.2f %+8.2f"
          % (r["market"], r["model"], r["n"], r["MAE"], r["MAE_pers"],
             r["dMAE"], r["t"]))

print("\n--- does the architecture penalty survive? ---")
lev = res[res["model"] == "level"]
anc = res[res["model"] == "anchored"]
print("level    : mean dMAE %+.1f%%, worse on %d/%d markets"
      % (lev["dMAE"].mean(), int((lev["dMAE"] > 0).sum()), len(lev)))
print("anchored : mean dMAE %+.1f%%, worse on %d/%d markets"
      % (anc["dMAE"].mean(), int((anc["dMAE"] > 0).sum()), len(anc)))
print("reduction factor: %.1f x"
      % (lev["dMAE"].mean() / max(anc["dMAE"].mean(), 1e-9)))

res.to_csv(os.path.join(BASE, "STEP14_ROLLING_DEEP.csv"), index=False, encoding="utf-8-sig")
print("\nwrote STEP14_ROLLING_DEEP.csv")
