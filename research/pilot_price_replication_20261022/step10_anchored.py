"""Step 10: level prediction vs anchored-residual prediction.

Claim under test. Carbon allowance prices have AR(1) coefficients of 0.9936-0.9980
(half-lives 108-348 days), i.e. they are near random walks. For such a series the
conditional mean of tomorrow's price is essentially today's price, so a correctly
trained network must output approximately the last observation. Networks that are
asked to predict the LEVEL directly have to reproduce the level's scale through
their output layer; small scale errors and weight decay then translate into
systematic bias, and the literature's reported errors (MAE ~1.4 where persistence
achieves ~0.73) are what that failure looks like.

The fix is architectural, not a hyperparameter: predict the ANCHORED RESIDUAL,
z_{t+1} - z_t, and add the anchor back afterwards. With no signal the network can
output zero and the model is exactly persistence, so it degrades gracefully.

Both variants get identical window, architecture, optimiser, epochs and
early-stopping rule. Only the target differs. Persistence is the reference.

Train on the first 60% only; score on the remaining 40%, forward in time.
"""

import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
torch.manual_seed(20261022)
np.random.seed(20261022)

PANEL = (r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research"
         r"\research\fullcycle_h20_20261020\PANEL_4MARKET.csv")
START = "2014-04-04"
TRAIN_FRAC = 0.60
WIN = 10
HID = 32
EPOCHS = 400
PATIENCE = 40
WD = 1e-4
LR = 1e-3


def met(y, yhat):
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    r = y - yhat
    return dict(n=int(y.size), MAE=float(np.mean(np.abs(r))),
                RMSE=float(np.sqrt(np.mean(r ** 2))),
                MAPE=float(np.mean(np.abs(r / y)) * 100.0),
                R2=float(1 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2)))


class Net(nn.Module):
    def __init__(self, hid):
        super().__init__()
        self.lstm = nn.LSTM(1, hid, batch_first=True)
        self.head = nn.Linear(hid, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1, :]).squeeze(-1)


def make_xy(z, idx, train_mu, train_sd, mode):
    X = np.stack([z[i - WIN + 1:i + 1] for i in idx])
    X = (X - train_mu) / train_sd
    if mode == "level":
        tgt = (z[idx + 1] - train_mu) / train_sd
    else:
        tgt = z[idx + 1] - z[idx]
    return (torch.tensor(X, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(tgt, dtype=torch.float32))


def fit_predict(Xtr, ytr, Xva, yva, Xte, mode, train_mu, train_sd, z_te, idx_te):
    net = Net(HID)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.MSELoss()
    best, best_state, bad = np.inf, None, 0
    for ep in range(EPOCHS):
        net.train()
        opt.zero_grad()
        loss = lossf(net(Xtr), ytr)
        loss.backward()
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
    with torch.no_grad():
        p = net(Xte).numpy()
    if mode == "level":
        return p * train_sd + train_mu
    return z_te[idx_te] + p


df = pd.read_csv(PANEL, parse_dates=["date"])
rows = []
for mkt, g in df.groupby("market"):
    g = g.sort_values("date")
    g = g[g["date"] >= START].reset_index(drop=True)
    y = g["price"].to_numpy(float)
    if y.size < 400:
        continue
    z = np.log(y)
    n = z.size
    cut = int(n * TRAIN_FRAC)

    idx_all = np.arange(WIN, n - 1)
    idx_tr = idx_all[idx_all < cut - 1]
    nva = max(40, int(0.15 * idx_tr.size))
    idx_fit, idx_va = idx_tr[:-nva], idx_tr[-nva:]
    idx_te = idx_all[idx_all >= cut - 1]

    train_mu = float(z[:cut].mean())
    train_sd = float(z[:cut].std())

    for mode in ("level", "anchored"):
        Xtr, ytr = make_xy(z, idx_fit, train_mu, train_sd, mode)
        Xva, yva = make_xy(z, idx_va, train_mu, train_sd, mode)
        Xte, _ = make_xy(z, idx_te, train_mu, train_sd, mode)
        pred = fit_predict(Xtr, ytr, Xva, yva, Xte, mode, train_mu, train_sd, z, idx_te)
        yhat = np.exp(pred)
        m = met(y[idx_te + 1], yhat)
        m0 = met(y[idx_te + 1], y[idx_te])
        rows.append(dict(market=mkt, model=mode, **m,
                         MAE_pers=m0["MAE"], RMSE_pers=m0["RMSE"],
                         R2_pers=m0["R2"],
                         dMAE=(m["MAE"] / m0["MAE"] - 1) * 100,
                         dRMSE=(m["RMSE"] / m0["RMSE"] - 1) * 100))
        print("%-10s %-9s MAE %.4f (pers %.4f, %+.2f%%)  RMSE %.4f (%+.2f%%)  R2 %.4f"
              % (mkt, mode, m["MAE"], m0["MAE"], m["MAE"] / m0["MAE"] * 100 - 100,
                 m["RMSE"], m["RMSE"] / m0["RMSE"] * 100 - 100, m["R2"]))

res = pd.DataFrame(rows)
print("\n--- summary, forward 40%% of each series ---")
print("%-10s %-9s %8s %10s %10s %9s %9s"
      % ("market", "model", "n", "MAE", "RMSE", "dMAE%", "dRMSE%"))
for mkt in res["market"].unique():
    for _, r in res[res["market"] == mkt].iterrows():
        print("%-10s %-9s %8d %10.4f %10.4f %+9.2f %+9.2f"
              % (mkt, r["model"], r["n"], r["MAE"], r["RMSE"], r["dMAE"], r["dRMSE"]))
    print()

print("--- the claim ---")
lev = res[res["model"] == "level"]
anc = res[res["model"] == "anchored"]
print("level-prediction worse than persistence on %d/%d markets (mean %+.1f%%)"
      % (int((lev["dMAE"] > 0).sum()), len(lev), lev["dMAE"].mean()))
print("anchored-residual worse than persistence on %d/%d markets (mean %+.1f%%)"
      % (int((anc["dMAE"] > 0).sum()), len(anc), anc["dMAE"].mean()))

base = os.path.dirname(os.path.abspath(__file__))
res.to_csv(os.path.join(base, "STEP10_ANCHORED.csv"), index=False, encoding="utf-8-sig")
print("\nwrote STEP10_ANCHORED.csv")
