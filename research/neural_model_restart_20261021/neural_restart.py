# -*- coding: utf-8 -*-
"""Exploratory repair of neural quantile experiments; not a publication result.

Run from any directory with the Python environment already used by this project.
The old files are preserved. All three networks below use identical splits,
normalization, anchor quantiles, loss-selection metric and training budget.
`old_symmetric_loss` is a diagnostic, never a competitive baseline.
A calendar/event-time gate is a candidate mechanism, not a novelty claim.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
PANEL = HERE.parent / 'fullcycle_h20_20261020' / 'PANEL_5MARKET.csv'
TAUS = np.array([.025, .05, .10, .15, .20, .30, .40, .50,
                 .60, .70, .80, .85, .90, .95, .975], dtype=np.float32)
HOR = [1, 5, 10, 20]
CUT = {'CEA': '2024-01-01', 'GUANGZHOU': '2024-01-01',
       'SHANGHAI': '2024-01-01', 'HUBEI': '2023-01-01', 'EU': '2023-01-01'}
W, K, KE, STEP = 250, 64, 32, 2
TTAU = torch.tensor(TAUS)


def pinball(y, q):
    d = y[:, None] - q
    return torch.maximum(TTAU * d, (TTAU - 1) * d).mean()


def old_symmetric_loss(y, q):
    return ((q - y[:, None]).abs() * (2 * (TTAU - .5).abs())).mean()


def score(y, q):
    """Mean quantile score (2*mean pinball); not full continuous CRPS."""
    d = y[:, None] - q
    return (2 * np.maximum(TAUS * d, (TAUS - 1) * d)).mean(axis=1)


def wis(y, q):
    out = .5 * np.abs(y - q[:, 7])
    for i, j, alpha in [(0, 14, .05), (1, 13, .10), (2, 12, .20),
                         (3, 11, .30), (4, 10, .40)]:
        lo, hi = q[:, i], q[:, j]
        out += alpha / 2 * ((hi - lo) + 2 / alpha * np.maximum(lo-y, 0)
                           + 2 / alpha * np.maximum(y-hi, 0))
    return out / 5.5


def loss_sanity():
    """One independent known-distribution check for the actual loss defect."""
    rng = np.random.default_rng(10)
    y = torch.tensor(rng.standard_normal(12000), dtype=torch.float32)
    out = {}
    for name, loss in [('old', old_symmetric_loss), ('pinball', pinball)]:
        q = nn.Parameter(torch.zeros(len(TAUS)))
        opt = torch.optim.Adam([q], lr=.035)
        for _ in range(250):
            opt.zero_grad()
            loss(y, q[None].expand(len(y), -1)).backward()
            opt.step()
        fitted = q.detach().numpy()
        out[name] = {str(float(TAUS[j])): float(fitted[j]) for j in [2, 7, 12]}
    expected = np.quantile(y.numpy(), TAUS)
    out['empirical_target'] = {str(float(TAUS[j])): float(expected[j]) for j in [2, 7, 12]}
    assert abs(out['pinball'][str(float(TAUS[2]))] - expected[2]) < .08
    assert abs(out['old'][str(float(TAUS[2]))]) < .08
    return out


def build(d):
    lp = np.log(d.price.to_numpy(float))
    dates = d.date.to_numpy()
    market = str(d.market.iloc[0])
    cut = int(np.searchsorted(dates, np.datetime64(CUT[market])))
    val_start = max(W + 50, int(cut * .75))
    rows = []
    for hidx, h in enumerate(HOR):
        for t in range(W, len(lp)-h, STEP):
            rd = np.diff(lp[t-W:t+1])  # Includes the price known at origin t.
            sd = float(rd.std(ddof=1))
            if not np.isfinite(sd) or sd <= 1e-8:
                continue
            nonzero = np.flatnonzero(np.abs(rd) > 1e-12)
            evt = np.zeros((KE, 3), np.float32)
            if len(nonzero):
                chosen = nonzero[-KE:]
                # Inter-event gaps measured only within the observed history.
                gaps = np.diff(np.r_[-1, nonzero])[-len(chosen):]
                evt[-len(chosen):, 0] = rd[chosen] / sd
                evt[-len(chosen):, 1] = np.log1p(gaps) / np.log1p(W)
                evt[-len(chosen):, 2] = 1.
                age = W-1-int(nonzero[-1])
            else:
                age = W
            raw = rd[-K:] / sd
            cal = np.stack([raw, (np.abs(rd[-K:]) <= 1e-12).astype(float)], axis=1)
            context = np.array([np.log(sd), np.mean(np.abs(rd) <= 1e-12),
                                np.log1p(age), np.log((rd[-20:].std(ddof=1)+1e-8)/sd)], np.float32)
            split = 0 if t+h < val_start else (1 if t >= val_start and t+h < cut
                                               else (2 if t >= cut else -1))
            if split < 0:
                continue
            rows.append((t, hidx, split, cal, evt, context,
                         sd*np.sqrt(h), lp[t+h]-lp[t]))
    cols = list(zip(*rows))
    result = {'t': np.asarray(cols[0]), 'hidx': np.asarray(cols[1]),
              'split': np.asarray(cols[2]), 'cal': np.stack(cols[3]),
              'event': np.stack(cols[4]), 'context': np.stack(cols[5]),
              'scale': np.asarray(cols[6], np.float32), 'y': np.asarray(cols[7], np.float32)}
    result['z'] = result['y'] / result['scale']
    tr, va, te = (np.flatnonzero(result['split'] == s) for s in [0, 1, 2])
    train_last_target = int(max(result['t'][tr] + np.array(HOR)[result['hidx'][tr]]))
    assert train_last_target < val_start
    assert max(result['t'][va] + np.array(HOR)[result['hidx'][va]]) < cut
    assert result['t'][te].min() >= cut
    mean, sd = result['context'][tr].mean(0), result['context'][tr].std(0) + 1e-5
    result['context'] = (result['context'] - mean) / sd
    base = np.stack([np.quantile(result['z'][tr[result['hidx'][tr] == hi]], TAUS)
                     for hi in range(len(HOR))]).astype(np.float32)
    # Prevent a zero inter-quantile gap from freezing an anchored head.
    for hi in range(len(HOR)):
        base[hi] = np.maximum.accumulate(base[hi] - np.arange(len(TAUS))*.001) + np.arange(len(TAUS))*.001
    result['anchor'] = base
    result['audit'] = {'n_rows': int(len(d)), 'cut_index': cut,
                       'validation_start_index': val_start,
                       'last_training_target_index': train_last_target,
                       'counts': {name: {str(h): int(np.sum((result['split'] == si) &
                                                            (result['hidx'] == hi)))
                                         for hi, h in enumerate(HOR)}
                                  for si, name in enumerate(['train', 'validation', 'test'])},
                       'unchanged_price_share': float(np.mean(np.diff(lp) == 0)),
                       'target_clip': False, 'legacy_H_parameters': False}
    return result


class QuantileNet(nn.Module):
    """Shared calendar LSTM with an optional learned event-time branch.

    The gate combines latent encodings, never quantiles from different targets.
    Positive anchored spacings guarantee non-crossing for every learned state.
    """
    def __init__(self, anchor, event=False, width=12):
        super().__init__()
        self.calendar = nn.LSTM(2, width, batch_first=True)
        self.use_event = event
        self.context = nn.Sequential(nn.Linear(5, width), nn.Tanh())
        if event:
            self.event = nn.GRU(3, width, batch_first=True)
            self.gate = nn.Sequential(nn.Linear(width*2, 12), nn.Tanh(), nn.Linear(12, 1), nn.Sigmoid())
        self.head = nn.Sequential(nn.Linear(width*2, 24), nn.Tanh(), nn.Linear(24, len(TAUS)))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer('base', torch.tensor(anchor))

    def forward(self, cal, event, ctx, hi):
        c, _ = self.calendar(cal)
        c = c[:, -1]
        h = torch.log(torch.tensor(HOR, dtype=torch.float32)[hi]) / np.log(max(HOR))
        x = self.context(torch.cat([ctx, h[:, None]], 1))
        gate = torch.zeros((len(cal), 1))
        if self.use_event:
            e, _ = self.event(event)
            gate = self.gate(torch.cat([x, c], 1))
            c = c + gate * e[:, -1]
        a = self.head(torch.cat([c, x], 1))
        base = self.base[hi]
        gaps = (base[:, 1:] - base[:, :-1]) * torch.exp(.8*torch.tanh(a[:, 1:]))
        cum = torch.cat([torch.zeros((len(cal), 1)), gaps.cumsum(1)], 1)
        q = base[:, 7:8] + a[:, :1] + cum - cum[:, 7:8]
        return q, gate


def train_predict(D, seed, model, epochs):
    torch.manual_seed(seed)
    net = QuantileNet(D['anchor'], event=model == 'event_gate')
    opt = torch.optim.AdamW(net.parameters(), lr=.003, weight_decay=1e-3)
    x = {k: torch.tensor(D[k], dtype=torch.long if k == 'hidx' else torch.float32)
         for k in ['cal', 'event', 'context', 'hidx', 'z']}
    tr, va, te = [torch.tensor(np.flatnonzero(D['split'] == s)) for s in [0, 1, 2]]

    def pred(idx):
        return net(x['cal'][idx], x['event'][idx], x['context'][idx], x['hidx'][idx])

    loss = old_symmetric_loss if model == 'old_symmetric_loss' else pinball
    best_state, best_val, bad, best_ep = copy.deepcopy(net.state_dict()), np.inf, 0, 0
    for ep in range(epochs):
        net.train()
        order = tr[torch.randperm(len(tr))]
        for batch in order.split(512):
            opt.zero_grad()
            q, _ = pred(batch)
            train_loss = loss(x['z'][batch], q)
            assert torch.isfinite(train_loss)
            train_loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.)
            opt.step()
        net.eval()
        with torch.no_grad():
            vq, _ = pred(va)
            # Select all models using correct proper loss on ALL horizons.
            vl = float(pinball(x['z'][va], vq))
        if vl < best_val - 1e-5:
            best_val, best_ep, bad = vl, ep+1, 0
            best_state = copy.deepcopy(net.state_dict())
        else:
            bad += 1
        if bad >= 18 and ep >= 25:
            break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        q, gate = pred(te)
    q = q.numpy() * D['scale'][te.numpy(), None]
    assert np.isfinite(q).all() and np.all(np.diff(q, axis=1) >= -1e-6)
    return q, {'seed': seed, 'best_epoch': best_ep, 'epochs_run': ep+1,
               'validation_pinball_normalized': best_val,
               'parameter_count': sum(p.numel() for p in net.parameters()),
               'mean_event_gate': float(gate.mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--markets', nargs='+', default=list(CUT))
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 22])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--models', nargs='+', default=['old_symmetric_loss', 'lstm_pinball', 'event_gate'])
    parser.add_argument('--tag', default='exploratory_repair')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    panel = pd.read_csv(PANEL, parse_dates=['date'])
    report = {'status': 'exploratory; historically reused test periods; no significance or novelty claim',
              'config': vars(args), 'horizons': HOR, 'quantiles': TAUS.tolist(),
              'panel_sha256': hashlib.sha256(PANEL.read_bytes()).hexdigest(),
              'python': platform.python_version(), 'torch': torch.__version__,
              'metric': 'mean quantile score, twice average pinball, not full CRPS',
              'loss_sanity': loss_sanity(), 'markets': {}}
    arrays = {}
    start = time.perf_counter()
    for market in args.markets:
        d = panel[panel.market == market].sort_values('date').reset_index(drop=True)
        D = build(d)
        te = np.flatnonzero(D['split'] == 2)
        y, hid, scale = D['y'][te], D['hidx'][te], D['scale'][te]
        preds = {'anchor': D['anchor'][hid] * scale[:, None]}
        mres = {'audit': D['audit'], 'fits': {}, 'by_horizon': {}}
        for model in args.models:
            qs, fits = [], []
            for seed in args.seeds:
                q, fit = train_predict(D, seed, model, args.epochs)
                qs.append(q)
                fits.append(fit)
                print(market, model, seed, 'best_epoch', fit['best_epoch'],
                      'elapsed_s', round(time.perf_counter()-start, 1), flush=True)
            # Equal-weight seed ensemble of predictions, then scoring.
            preds[model] = np.mean(qs, axis=0)
            mres['fits'][model] = fits
        for hi, h in enumerate(HOR):
            mask = hid == hi
            rows = {}
            for model, q in preds.items():
                yh, qh = y[mask], q[mask]
                rows[model] = {'MQS': float(score(yh, qh).mean()),
                               'WIS': float(wis(yh, qh).mean()),
                               'coverage80': float(np.mean((yh >= qh[:, 2]) & (yh <= qh[:, 12]))),
                               'coverage95': float(np.mean((yh >= qh[:, 0]) & (yh <= qh[:, 14]))),
                               'MAE_median': float(np.abs(yh-qh[:, 7]).mean())}
            mres['by_horizon'][str(h)] = {'n': int(mask.sum()), 'scores': rows}
            print(market, 'h', h, {k: round(v['MQS'], 6) for k, v in rows.items()}, flush=True)
        report['markets'][market] = mres
        arrays[market+'_target'] = y
        arrays[market+'_origin'] = D['t'][te]
        arrays[market+'_horizon'] = np.array(HOR)[hid]
        for model, q in preds.items():
            arrays[market+'_'+model] = q
        (HERE / (args.tag+'.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        np.savez_compressed(HERE / (args.tag+'_predictions.npz'), **arrays)
    summary = {}
    for model in args.models:
        base = 'lstm_pinball' if model == 'event_gate' else 'anchor'
        ratios = [r['scores'][model]['MQS'] / r['scores'][base]['MQS']
                  for m in report['markets'].values() for r in m['by_horizon'].values()]
        summary[model+'_vs_'+base] = {'median_MQS_ratio': float(np.median(ratios)),
                                     'mean_log_MQS_ratio': float(np.mean(np.log(ratios))),
                                     'wins': int(np.sum(np.array(ratios) < 1)), 'cells': len(ratios)}
    report['summary'] = summary
    report['elapsed_seconds'] = time.perf_counter()-start
    (HERE / (args.tag+'.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
