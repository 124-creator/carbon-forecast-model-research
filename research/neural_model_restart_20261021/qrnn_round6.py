# -*- coding: utf-8 -*-
"""QRNN baselines for the round-6 comparison table.

Two arms, identical protocol to deep_seq_round4 (same rows, same target, same
inner-train / val / full-precut refit, 5 seeds, 15 taus, equal-horizon weights):
  qrnn         Taylor (2000) quantile regression neural network, no anchor
  qrnn_anchor  same MLP but initialised as empirical-quantile + zero delta,
               which is the initialisation the round-4 deep arms use
Only qrnn_round6* files are written.
"""
from __future__ import annotations
import os, sys
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'
sys.dont_write_bytecode = True
import argparse, json, time, hashlib
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
MODELS = ['qrnn', 'qrnn_anchor']
SEEDS = [11, 22, 33, 44, 55]
NLAG = 10


class QRNN(nn.Module):
    def __init__(self, model, base, width=32):
        super().__init__()
        self.model = model
        self.net = nn.Sequential(nn.Linear(2 * NLAG + 36, width), nn.Tanh(), nn.Dropout(.05), nn.Linear(width, len(TAU)))
        if model == 'qrnn_anchor':
            nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('base', torch.tensor(base))

    def forward(self, f, hi):
        out = self.net(f)
        if self.model == 'qrnn_anchor':
            out = out + self.base[hi]
        return torch.sort(out, dim=1).values


def features(D):
    seq = D['seq']
    return np.concatenate([seq[:, -NLAG:, 0], seq[:, -NLAG:, 1], D['x_aug']], axis=1).astype(np.float32)


def fit_predict(D, model, seed, args):
    F = torch.tensor(features(D)); hi = torch.tensor(D['hi'], dtype=torch.long); y = torch.tensor(D['z'], dtype=torch.float32)
    tr, va, rf, te = [torch.tensor(D[k]) for k in ['tr', 'va', 'rf', 'te']]

    def setup(idx):
        _, base, meta = DS.prepare(D, np.asarray(idx))
        torch.manual_seed(seed)
        net = QRNN(model, base)
        opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.003)
        original = [p.detach().clone() for p in net.parameters()]
        return net, opt, meta, original

    def delta(net, original):
        return float(sum(((p.detach() - o) ** 2).sum() for p, o in zip(net.parameters(), original)).sqrt())

    def train_epoch(net, opt, idx):
        net.train(); order = idx[torch.randperm(len(idx))]
        count = torch.bincount(hi[idx], minlength=4); weights = len(idx) / (4 * count.float())
        seen = False
        for b in order.split(512):
            opt.zero_grad(); q = net(F[b], hi[b])
            loss = (DS.pinball_values(y[b], q).mean(1) * weights[hi[b]]).mean()
            assert torch.isfinite(loss)
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
            seen = seen or any(p.grad is not None and bool(torch.any(p.grad != 0)) for p in net.parameters())
            opt.step()
        return seen

    def val(net):
        net.eval()
        with torch.no_grad():
            parts = []
            for h in range(4):
                ids = va[hi[va] == h]
                parts.append(torch.cat([DS.pinball_values(y[b], net(F[b], hi[b])).mean(1) for b in ids.split(512)]).mean())
            return float(torch.stack(parts).mean())

    net, opt, inner_meta, orig = setup(tr)
    best = val(net); best_ep = 0; bad = 0; history = [[0, best]]
    torch.manual_seed(seed + 731)
    for ep in range(1, args.epochs + 1):
        seen = train_epoch(net, opt, tr); value = val(net); history.append([ep, value])
        if value < best - 1e-6:
            best, best_ep, bad = value, ep, 0
        else:
            bad += 1
        if ep >= args.min_epochs and bad >= args.patience:
            break
    inner_delta = delta(net, orig); assert seen and inner_delta > 0
    net, opt, refit_meta, orig = setup(rf)
    torch.manual_seed(seed + 731); refit_seen = False
    for _ in range(best_ep):
        refit_seen = train_epoch(net, opt, rf)
    net.eval()
    with torch.no_grad():
        q = torch.cat([net(F[b], hi[b]) for b in te.split(512)]).numpy() * D['scale'][D['te'], None]
    assert np.isfinite(q).all() and np.all(np.diff(q, axis=1) >= -1e-8)
    return q, dict(seed=seed, best_epoch=best_ep, inner_epochs=ep, validation_normalized_macro_pinball=best,
                   validation_history=history, refit_epochs=best_ep, inner_encoder_parameter_L2_change=inner_delta,
                   refit_encoder_nonzero_gradient=refit_seen, allocated_parameters=sum(p.numel() for p in net.parameters()),
                   inner_normalization=inner_meta, refit_normalization=refit_meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--markets', nargs='+', default=list(T.B.CUT))
    ap.add_argument('--models', nargs='+', default=MODELS)
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--epochs', type=int, default=80); ap.add_argument('--min-epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=16); ap.add_argument('--tag', default='qrnn_round6')
    args = ap.parse_args(); assert args.tag.startswith('qrnn_round6')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    panel = pd.read_csv(T.B.PANEL, parse_dates=['date']); start = time.perf_counter()
    prev = np.load(HERE / 'qrf_round4_probability_control_predictions.npz')
    ref = np.load(HERE / 'deep_seq_round4_predictions.npz')
    arrays = {'taus': T.B.TAUS}; rows = []
    out = dict(status='round6 QRNN baselines, same protocol as deep_seq_round4', config=vars(args), markets={})
    for market in args.markets:
        d = panel[panel.market == market].sort_values('date').reset_index(drop=True)
        D = DS.build(d)
        for k, v in [('origin', D['t'][D['te']]), ('horizon', T.HOR[D['hi'][D['te']]]), ('target', D['y'][D['te']]), ('scale', D['scale'][D['te']])]:
            assert np.array_equal(v, ref[market + '_' + k]), k
        arrays[market + '_origin'] = ref[market + '_origin']; arrays[market + '_origin_date'] = ref[market + '_origin_date']
        arrays[market + '_horizon'] = ref[market + '_horizon']; arrays[market + '_target'] = ref[market + '_target']; arrays[market + '_scale'] = ref[market + '_scale']
        fits = {}
        for model in args.models:
            qs = []; fs = []
            for seed in args.seeds:
                q, f = fit_predict(D, model, seed, args)
                qs.append(q); fs.append(f); arrays[f'{market}_{model}_seed{seed}_Q'] = q
                print(market, model, seed, 'best', f['best_epoch'], 'run', f['inner_epochs'], 'elapsed', round(time.perf_counter() - start, 1), flush=True)
            arrays[f'{market}_{model}_Q'] = np.mean(qs, axis=0); fits[model] = fs
        out['markets'][market] = dict(fits=fits)
    np.savez_compressed(HERE / (args.tag + '_predictions.npz'), **arrays)
    out['elapsed_seconds'] = time.perf_counter() - start
    (HERE / (args.tag + '.json')).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('done', out['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with threadpool_limits(limits=1):
        main()
