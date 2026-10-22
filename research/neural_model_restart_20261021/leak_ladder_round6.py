# -*- coding: utf-8 -*-
"""Leak ladder: how much of the reported carbon-price accuracy is protocol, not model.

One workhorse model (the round-6 QRNN MLP) is retrained under six protocols that
differ only in where information from the test window is allowed to enter:

  A_honest     pre-cut fit, test-window evaluation (project protocol)
  B_norm       feature/target normalisation fitted on the whole series
  C_decomp     decomposition before split: the trend is a two-sided moving average
               evaluated at the target date, i.e. the upper bound of what
               "CEEMDAN/VMD on the full series, then split" can buy
  D_testlag    the input lag is chosen on the test window (Table-13 practice)
  E_stack      B + C + D
  F_stack_lam  E plus a dispersion multiplier swept on the test window

Only leak_ladder_round6* files are written.
"""
from __future__ import annotations
import os, sys
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'
sys.dont_write_bytecode = True
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits
import deep_seq_round4 as DS

T = DS.T
HERE = Path(__file__).resolve().parent
TAU = DS.TAU
SEEDS = [11, 22, 33]
LAGS = [1, 5, 10, 20, 40]
RUNGS = ['A_honest', 'B0_const', 'B_norm', 'C_decomp', 'D_testlag', 'E_stack', 'F_stack_lam', 'G_trainleak']
LAM_GRID = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]


class MLP(nn.Module):
    def __init__(self, nin, width=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nin, width), nn.Tanh(), nn.Dropout(.05), nn.Linear(width, len(TAU)))

    def forward(self, f):
        return torch.sort(self.net(f), dim=1).values


def train_predict(F, tgt, hi, folds, seed, args, pred_idx=None):
    """F: (n,d) float32 features; tgt: normalized target; folds: dict of index arrays."""
    F = torch.tensor(F); hi = torch.tensor(hi, dtype=torch.long); y = torch.tensor(tgt, dtype=torch.float32)
    ix = lambda k: torch.tensor(folds[k])

    def fresh():
        torch.manual_seed(seed)
        net = MLP(F.shape[1]); return net, torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.003)

    def epoch(net, opt, idx):
        net.train(); order = idx[torch.randperm(len(idx))]
        cnt = torch.bincount(hi[idx], minlength=4); w = len(idx) / (4 * cnt.float())
        for b in order.split(512):
            opt.zero_grad(); q = net(F[b])
            loss = (DS.pinball_values(y[b], q).mean(1) * w[hi[b]]).mean()
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True); opt.step()

    def val(net):
        net.eval()
        with torch.no_grad():
            v = ix('va')
            q = torch.cat([net(F[b]) for b in v.split(512)])
            return float(DS.pinball_values(y[v], q).mean())

    net, opt = fresh(); best, bad, best_ep = val(net), 0, 0
    torch.manual_seed(seed + 731)
    for ep in range(1, args.epochs + 1):
        epoch(net, opt, ix('tr')); v = val(net)
        if v < best - 1e-6:
            best, best_ep, bad = v, ep, 0
        else:
            bad += 1
        if ep >= args.min_epochs and bad >= args.patience:
            break
    net, opt = fresh(); torch.manual_seed(seed + 731)
    for _ in range(max(best_ep, 1)):
        epoch(net, opt, ix('rf'))
    net.eval()
    with torch.no_grad():
        pi = ix('te') if pred_idx is None else torch.tensor(pred_idx)
        q = torch.cat([net(F[b]) for b in pi.split(512)]).numpy()
    assert np.isfinite(q).all() and np.all(np.diff(q, axis=1) >= -1e-8)
    return q.astype(np.float64)


def pin(y, q):
    e = y[:, None] - q
    return float(2 * np.maximum(TAU * e, (TAU - 1) * e).mean())


def series(market, d):
    D = DS.build(d)
    lp = np.log(d.price.to_numpy(float))
    tr = pd.Series(lp).rolling(21, center=True, min_periods=1).mean().to_numpy()
    D['lp'] = lp; D['trend'] = tr; D['det'] = lp - tr
    dd = np.diff(lp - tr)
    roll = pd.Series(dd).rolling(120, min_periods=60).std(ddof=1)
    sd_det = np.r_[np.nan, roll.to_numpy()]
    sd_det = np.where(np.isfinite(sd_det), sd_det, np.nanstd(dd))
    D['det_scale'] = np.maximum(sd_det, 1e-8)[D['t']] * np.sqrt(T.HOR[D['hi']])
    return D


def build_input(D, lag, normalise_on, mode):
    """mode: 'level' uses log-price innovations, 'det' uses detrended innovations."""
    t = D['t']; hi = D['hi']; h = T.HOR[hi]
    if mode == 'level':
        x = D['lp']
    else:
        x = D['det']
    step = np.diff(x)
    rows = np.stack([step[t - lag + 1 + k] for k in range(lag - 1, -1, -1)], axis=1) if lag > 1 else step[t][:, None]
    if normalise_on == 'all':
        sd_daily = rows.std(0) + 1e-9
    else:
        pre = D['t'] < D['audit']['validation_start_index']
        sd_daily = rows[pre].std(0) + 1e-9
    seq = rows / sd_daily
    ctx = D['x_aug']
    if normalise_on == 'all':
        cm, cs = ctx.mean(0), ctx.std(0) + 1e-5
    else:
        cm, cs = ctx[D['rf']].mean(0), ctx[D['rf']].std(0) + 1e-5
    F = np.concatenate([np.clip(seq, -8, 8), np.clip((ctx - cm) / cs, -8, 8)], axis=1).astype(np.float32)
    return F


def target_and_scale(D, mode, normalise_on):
    """Trailing scale by default (as in the project protocol); the whole series if leaked."""
    t = D['t']; hi = D['hi']; h = T.HOR[hi]
    if mode == 'level':
        raw = D['lp'][t + h] - D['lp'][t]
        base = np.diff(D['lp']); trail = D['scale']
    else:
        raw = D['det'][t + h] - D['det'][t]
        base = np.diff(D['det']); trail = D['det_scale']
    if normalise_on == 'all':
        return raw, base.std() * np.sqrt(h)
    if normalise_on == 'precut':
        return raw, base[:D['audit']['cut_index']].std() * np.sqrt(h)
    return raw, trail


def run_market(d, market, args):
    """Every rung stores the 15 test log-return quantiles of each seed."""
    D = series(market, d)
    h1 = D['hi'] == 0
    sel = np.flatnonzero(h1)
    pos = -np.ones(len(D['hi']), int); pos[sel] = np.arange(len(sel))
    folds = {k: pos[D[k][D['hi'][D[k]] == 0]] for k in ['tr', 'va', 'rf', 'te']}
    assert all((v >= 0).all() and len(v) for v in folds.values())
    te = folds['te']
    t_h1 = D['t'][sel]
    t = t_h1[te]
    y_h1 = D['lp'][t_h1 + 1] - D['lp'][t_h1]
    y_true = y_h1[te]
    tr_off = D['trend'][t + 1] - D['trend'][t]
    out = {}
    cache = {}

    def fit(mode, lag, normalise_on, seed, fold=None, pred=None):
        fold = folds if fold is None else fold
        pred = fold['te'] if pred is None else pred
        key = (mode, lag, normalise_on, seed, id(fold), id(pred))
        if key not in cache:
            F = build_input(D, lag, normalise_on, mode)[h1]
            raw, s = target_and_scale(D, mode, normalise_on)
            s_all = s[h1]
            q = train_predict(F, raw[h1] / s_all, D['hi'][h1], fold, seed, args, pred_idx=pred) * s_all[pred][:, None]
            cache[key] = q
        return cache[key]

    for seed in args.seeds:
        # A: honest protocol, pre-committed lag 10
        out.setdefault('A_honest', []).append(fit('level', 10, 'rf', seed))
        # B0: a constant pre-cut sd instead of the trailing sd (honest control for B)
        out.setdefault('B0_const', []).append(fit('level', 10, 'precut', seed))
        # B: normalisation fitted on the whole series
        out.setdefault('B_norm', []).append(fit('level', 10, 'all', seed))
        # G: training set contaminated by a random 70% split, evaluated on the honest test window
        if 'gfold' not in out:
            rng = np.random.default_rng(20261021)
            perm = rng.permutation(len(sel))
            n_tr = int(.7 * len(perm)); n_va = int(.1 * len(perm))
            out['gfold'] = {'tr': perm[:n_tr], 'va': perm[n_tr:n_tr + n_va], 'te': perm[n_tr + n_va:]}
            out['gfold']['rf'] = np.concatenate([out['gfold']['tr'], out['gfold']['va']])
        out.setdefault('G_trainleak', []).append(fit('level', 10, 'rf', seed, out['gfold'], te))
        # C: decomposition before split, two-sided trend evaluated at the target date
        out.setdefault('C_decomp', []).append(fit('det', 10, 'rf', seed) + tr_off[:, None])
        # D: input lag chosen on the test window
        best = None
        for lg in args.lags:
            cand = fit('level', lg, 'rf', seed)
            m = pin(y_true, cand)
            if best is None or m < best[0]:
                best = (m, cand, lg)
        out.setdefault('D_testlag', []).append(best[1]); out.setdefault('_testlag_choice', []).append(best[2])
        # E: B + C + D stacked
        best = None
        for lg in args.lags:
            cand = fit('det', lg, 'all', seed) + tr_off[:, None]
            m = pin(y_true, cand)
            if best is None or m < best[0]:
                best = (m, cand, lg)
        out.setdefault('E_stack', []).append(best[1]); out.setdefault('_stack_lag_choice', []).append(best[2])
        # F: E plus a dispersion multiplier swept on the test window
        q = best[1]; mid = np.median(q, axis=1, keepdims=True)
        best = None
        for lam in LAM_GRID:
            cand = mid + lam * (q - mid)
            m = pin(y_true, cand)
            if best is None or m < best[0]:
                best = (m, cand, lam)
        out.setdefault('F_stack_lam', []).append(best[1]); out.setdefault('_lambda_choice', []).append(best[2])
    return t_h1, y_h1, te, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--markets', nargs='+', default=list(T.B.CUT))
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--lags', nargs='+', type=int, default=LAGS)
    ap.add_argument('--epochs', type=int, default=80); ap.add_argument('--min-epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=16); ap.add_argument('--tag', default='leak_ladder_round6')
    args = ap.parse_args(); assert args.tag.startswith('leak_ladder_round6')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    panel = pd.read_csv(T.B.PANEL, parse_dates=['date'])
    t0 = time.perf_counter(); rows = []
    for market in args.markets:
        d = panel[panel.market == market].sort_values('date').reset_index(drop=True)
        t_h1, y_h1, te, out = run_market(d, market, args)
        price = d.price.to_numpy(float)
        p0 = price[t_h1]; level = price[t_h1 + 1]
        row = dict(market=market, n=int(len(t_h1)), n_test=int(len(te)))
        for rung in RUNGS:
            q = np.mean(out[rung], axis=0)
            idx = te
            yy, pp, lv = y_h1[idx], p0[idx], level[idx]
            pred = pp * np.exp(q[:, 7])
            ss = float(((lv - lv.mean()) ** 2).sum())
            row[f'{rung}_MQS'] = pin(yy, q)
            row[f'{rung}_R2'] = float(1 - ((pred - lv) ** 2).sum() / ss)
            row[f'{rung}_MAPE'] = float((np.abs(pred - lv) / lv).mean() * 100)
            row[f'{rung}_RMSE'] = float(np.sqrt(((pred - lv) ** 2).mean()))
            row[f'{rung}_cov80'] = float(((yy >= q[:, 2]) & (yy <= q[:, 12])).mean())
        rows.append(row)
        print(market, {rung: (round(row[f'{rung}_MQS'], 3), round(row[f'{rung}_R2'], 4), round(row[f'{rung}_MAPE'], 3)) for rung in RUNGS},
              'lag', out['_testlag_choice'], 'stack_lag', out['_stack_lag_choice'], 'lam', out['_lambda_choice'],
              'elapsed', round(time.perf_counter() - t0, 1), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / (args.tag + '_cells.csv'), index=False)
    agg = {}
    for rung in RUNGS:
        gain = None if rung == 'G_trainleak' else float(100 * (1 - np.exp(np.log(df[f'{rung}_MQS'] / df['A_honest_MQS']).mean())))
        agg[rung] = dict(MQS=float(df[f'{rung}_MQS'].mean()), R2=float(df[f'{rung}_R2'].mean()),
                         MAPE=float(df[f'{rung}_MAPE'].mean()), RMSE=float(df[f'{rung}_RMSE'].mean()),
                         cov80=float(df[f'{rung}_cov80'].mean()), MQS_gain_vs_A=gain)
    (HERE / (args.tag + '.json')).write_text(json.dumps(dict(config=vars(args), aggregate=agg, cells=rows, elapsed=time.perf_counter() - t0), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(agg, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with threadpool_limits(limits=1):
        main()
