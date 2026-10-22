"""Step 11: shrinkage on the anchored residual.

Step 10 showed anchored-residual prediction cuts the degradation from +1025% to
+60% but still loses to persistence on every market. The residual failure is a
learnt drift that gets extrapolated (Shanghai +213%).

Shrinking the predicted residual toward zero completes the design:

    z_hat(t+1) = z(t) + lambda * delta_hat(t)

lambda = 0 is exactly persistence, lambda = 1 is the unrestricted anchored model.
So the whole family is bounded below by persistence, and lambda is the single knob
that trades bias against variance.

lambda is NOT selected on the held-out window. The full curve is reported as a
sensitivity analysis, and a validation-selected lambda (chosen on a tail slice of
the TRAINING period) is reported separately as the deployable variant.
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
LAMBDAS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.00)


def met(y, yhat):
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    r = y - yhat
    return (float(np.mean(np.abs(r))), float(np.sqrt(np.mean(r ** 2))),
            float(1 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2)))


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


def train_delta(z, idx_fit, idx_va):
    Xtr, Xva = build(z, idx_fit), build(z, idx_va)
    ytr = torch.tensor(z[idx_fit + 1] - z[idx_fit], dtype=torch.float32)
    yva = torch.tensor(z[idx_va + 1] - z[idx_va], dtype=torch.float32)
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
rows, curves = [], {}
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

    net = train_delta(z, idx_fit, idx_va)
    with torch.no_grad():
        p_te = net(build(z, idx_te)).numpy()
        p_va = net(build(z, idx_va)).numpy()

    # deployable lambda: chosen on the validation tail of the TRAINING period
    best_lam, best_mae = None, np.inf
    for lam in LAMBDAS:
        yh = np.exp(z[idx_va] + lam * p_va)
        mae = np.mean(np.abs(y[idx_va + 1] - yh))
        if mae < best_mae:
            best_mae, best_lam = mae, lam
    print("%-10s validation-selected lambda = %.2f" % (mkt, best_lam))

    mae_p, rmse_p, r2_p = met(y[idx_te + 1], y[idx_te])
    curve = []
    for lam in LAMBDAS:
        yh = np.exp(z[idx_te] + lam * p_te)
        mae, rmse, r2 = met(y[idx_te + 1], yh)
        curve.append((lam, mae / mae_p * 100 - 100, rmse / rmse_p * 100 - 100, mae))
        rows.append(dict(market=mkt, lam=lam, MAE=mae, RMSE=rmse, R2=r2,
                         dMAE=(mae / mae_p - 1) * 100,
                         dRMSE=(rmse / rmse_p - 1) * 100,
                         selected=(lam == best_lam)))
    curves[mkt] = (curve, mae_p)

print("\n--- dMAE%% vs persistence, by shrinkage lambda (lambda=0 IS persistence) ---")
print("%-10s" % "market" + "".join(" %10s" % ("lam=%.2f" % l) for l in LAMBDAS))
for mkt, (curve, mae_p) in curves.items():
    print("%-10s" % mkt + "".join(" %+10.2f" % c[1] for c in curve))
print("%-10s" % "persist MAE" + "".join(" %10.4f" % c[3] for c in curve))

print("\n--- dRMSE%% vs persistence ---")
print("%-10s" % "market" + "".join(" %10s" % ("lam=%.2f" % l) for l in LAMBDAS))
for mkt, (curve, mae_p) in curves.items():
    print("%-10s" % mkt + "".join(" %+10.2f" % c[2] for c in curve))

print("\n--- validation-selected lambda applied once to the held-out window ---")
sel = pd.DataFrame([r for r in rows if r["selected"]])
print("%-10s %6s %10s %10s %9s %9s"
      % ("market", "lam", "MAE", "RMSE", "dMAE%", "dRMSE%"))
for _, r in sel.iterrows():
    print("%-10s %6.2f %10.4f %10.4f %+9.2f %+9.2f"
          % (r["market"], r["lam"], r["MAE"], r["RMSE"], r["dMAE"], r["dRMSE"]))
print("\nbeats persistence on %d/%d markets, mean dMAE %+.2f%%"
      % (int((sel["dMAE"] < 0).sum()), len(sel), sel["dMAE"].mean()))

base = os.path.dirname(os.path.abspath(__file__))
pd.DataFrame(rows).to_csv(os.path.join(base, "STEP11_SHRINKAGE.csv"),
                          index=False, encoding="utf-8-sig")
print("\nwrote STEP11_SHRINKAGE.csv")
