# -*- coding: utf-8 -*-
"""Finite controlled follow-up; all test outcomes remain exploratory.

Imports the already checked data, loss, network and decoding. The new controls
use the exact same 2,740-parameter architecture, replacing only the input of the
second recurrent branch. The epoch-zero statistical anchor is eligible for
validation selection. No refit on the validation segment is performed here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import neural_restart as B

HERE = Path(__file__).resolve().parent
SEEDS = [11, 22]
EPOCHS = 120
MODES = ['lstm', 'calendar_gate', 'event_gate', 'fixed_event_gate']


class ControlledNet(B.QuantileNet):
    def __init__(self, anchor, mode):
        super().__init__(anchor, event=mode != 'lstm')
        self.mode = mode
        if mode == 'fixed_event_gate':
            # Preserve exactly the same output shape and fuse at a fixed 1/2.
            for p in self.gate.parameters():
                p.requires_grad_(False)

    def forward(self, cal, event, ctx, hi):
        if self.mode == 'calendar_gate':
            evt = torch.zeros_like(event)
            evt[:, :, 0] = cal[:, -B.KE:, 0]
            evt[:, :, 1] = np.log(2.) / np.log1p(B.W)
            evt[:, :, 2] = 1.
            event = evt
        if self.mode == 'fixed_event_gate':
            # An all-zero final gate layer produces sigmoid(0)=0.5.
            # Set once at construction-equivalent time; parameters are frozen.
            with torch.no_grad():
                self.gate[2].weight.zero_()
                self.gate[2].bias.zero_()
        return super().forward(cal, event, ctx, hi)


def fit(D, mode, seed):
    torch.manual_seed(seed)
    net = ControlledNet(D['anchor'], mode)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=.003, weight_decay=1e-3)
    x = {k: torch.tensor(D[k], dtype=torch.long if k == 'hidx' else torch.float32)
         for k in ['cal', 'event', 'context', 'hidx', 'z']}
    inds = [torch.tensor(np.flatnonzero(D['split'] == s)) for s in [0, 1, 2]]
    tr, va, te = inds

    def pred(idx):
        return net(x['cal'][idx], x['event'][idx], x['context'][idx], x['hidx'][idx])

    def vl():
        q, _ = pred(va)
        # Equal horizon weight, all labels available before the outer test cut.
        return float(torch.stack([B.pinball(x['z'][va][x['hidx'][va] == hi],
                                           q[x['hidx'][va] == hi])
                                  for hi in range(len(B.HOR))]).mean())

    net.eval()
    with torch.no_grad():
        best = vl()
    initial = best
    best_state, best_ep, bad = copy.deepcopy(net.state_dict()), 0, 0
    for ep in range(EPOCHS):
        net.train()
        order = tr[torch.randperm(len(tr))]
        for batch in order.split(512):
            q, _ = pred(batch)
            loss = B.pinball(x['z'][batch], q)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.)
            opt.step()
        net.eval()
        with torch.no_grad():
            value = vl()
        if value < best - 1e-5:
            best, best_ep, bad = value, ep+1, 0
            best_state = copy.deepcopy(net.state_dict())
        else:
            bad += 1
        if ep >= 25 and bad >= 18:
            break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        q, gate = pred(te)
    q = q.numpy() * D['scale'][te.numpy(), None]
    assert np.isfinite(q).all() and (np.diff(q, axis=1) >= -1e-6).all()
    return q, {'seed': seed, 'best_epoch': best_ep, 'epochs_run': ep+1,
               'validation_pinball_macro': best, 'epoch0_validation_pinball_macro': initial,
               'parameter_count': sum(p.numel() for p in net.parameters()),
               'trainable_parameters': sum(p.numel() for p in net.parameters() if p.requires_grad),
               'mean_gate': float(gate.mean())}


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    panel = pd.read_csv(B.PANEL, parse_dates=['date'])
    report = {'status': 'exploratory; previously used test periods; no novelty/significance claim',
              'horizons': B.HOR, 'seeds': SEEDS, 'max_epochs': EPOCHS,
              'epoch0_eligible': True, 'validation_selection': 'equal horizon normalized pinball',
              'training_objective': 'pooled normalized pinball', 'refit_after_selection': False,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'dependency_sha256': hashlib.sha256(Path(B.__file__).read_bytes()).hexdigest(),
              'panel_sha256': hashlib.sha256(B.PANEL.read_bytes()).hexdigest(), 'markets': {}}
    arrays = {}
    start = time.perf_counter()
    for market in B.CUT:
        D = B.build(panel[panel.market == market].sort_values('date').reset_index(drop=True))
        te = np.flatnonzero(D['split'] == 2)
        y, hi, scale = D['y'][te], D['hidx'][te], D['scale'][te]
        preds = {'anchor': D['anchor'][hi] * scale[:, None]}
        m = {'audit': D['audit'], 'fits': {}, 'by_horizon': {}}
        for mode in MODES:
            qs, fits = [], []
            for seed in SEEDS:
                q, f = fit(D, mode, seed)
                qs.append(q)
                fits.append(f)
                print(market, mode, seed, f['best_epoch'], round(time.perf_counter()-start, 1), flush=True)
            preds[mode] = np.mean(qs, axis=0)
            m['fits'][mode] = fits
        for hidx, h in enumerate(B.HOR):
            keep = hi == hidx
            m['by_horizon'][str(h)] = {'n': int(keep.sum()), 'scores': {}}
            for mode, q in preds.items():
                qh, yh = q[keep], y[keep]
                m['by_horizon'][str(h)]['scores'][mode] = {
                    'MQS': float(B.score(yh, qh).mean()), 'WIS': float(B.wis(yh, qh).mean()),
                    'coverage80': float(np.mean((yh >= qh[:, 2]) & (yh <= qh[:, 12]))),
                    'coverage95': float(np.mean((yh >= qh[:, 0]) & (yh <= qh[:, 14])))}
        report['markets'][market] = m
        arrays[market+'_target'] = y
        arrays[market+'_origin'] = D['t'][te]
        arrays[market+'_horizon'] = np.array(B.HOR)[hi]
        for mode, q in preds.items():
            arrays[market+'_'+mode] = q
        (HERE/'controlled_followup.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        np.savez_compressed(HERE/'controlled_followup_predictions.npz', **arrays)
    pairs = [('event_gate', 'lstm'), ('event_gate', 'calendar_gate'),
             ('event_gate', 'fixed_event_gate'), ('event_gate', 'anchor')]
    report['summary'] = {}
    for model, base in pairs:
        ratios = [r['scores'][model]['MQS']/r['scores'][base]['MQS']
                  for m in report['markets'].values() for r in m['by_horizon'].values()]
        report['summary'][model+'_vs_'+base] = {'median_MQS_ratio': float(np.median(ratios)),
                                               'mean_log_MQS_ratio': float(np.log(ratios).mean()),
                                               'wins': int(np.sum(np.array(ratios) < 1)),
                                               'cells': len(ratios)}
    report['elapsed_seconds'] = time.perf_counter()-start
    (HERE/'controlled_followup.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report['summary'], indent=2), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
