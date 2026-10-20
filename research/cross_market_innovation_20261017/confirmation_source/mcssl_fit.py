"""Two independent learned completions: an explicitly adapted MCSSL control.

Same GRU, initialization, supervision, shrinkage and optimizer as the matched
augmentation arm. Eq.7 is rectified after a whole-market/horizon average; this
stabilization and the vertex/date adaptation are not the author's benchmark.
"""
from __future__ import annotations

import copy
import time
import numpy as np
import torch

from data_core import group_weights
from fit_core import (configure, network_input, build_network, initialize_head,
                      validation_score, state_arrays, state_hash)
from mcssl_control import sample_tokens


def completed_network_input(raw, data):
    identity = torch.nn.functional.one_hot(
        torch.as_tensor(data['market']), len(data['market_names'])).to(raw.dtype)
    return torch.cat([raw, identity[:, None, :].expand(-1, 20, -1)], dim=-1)


def rectified_two_sample(coarse, a, b, market):
    raw = torch.stack([((coarse[market == m] - a[market == m]) *
                        (coarse[market == m] - b[market == m])).mean(0)
                       for m in torch.unique(market)])
    return raw.clamp_min(0).mean(), float((raw.detach() > 0).double().mean())


def fit_mcssl(train, validation, config, seed, q, max_epochs=60, deadline=None):
    configure(); torch.manual_seed(seed)
    started = time.perf_counter()
    q.eval()
    for p in q.parameters():
        p.requires_grad_(False)
    q_hash_before = state_hash(state_arrays(q))
    weights = torch.as_tensor(group_weights(train))
    y = torch.as_tensor(train['y'])
    mid = torch.as_tensor(train['market'])
    net = build_network(config, len(train['market_names']))
    x = {v: network_input(train, v) for v in ('full', 'short', 'sparse')}
    masks = {v: torch.as_tensor(train[v + '_mask']) for v in x}
    vx = network_input(validation, 'full')
    vm = torch.as_tensor(validation['full_mask'])
    shrink = float(config['shrink'])
    initialize_head(net, x['full'], y, weights, shrink)
    initial_arrays = state_arrays(net)
    with torch.no_grad():
        best = initial_validation = validation_score(validation, net(vx, mask=vm))
    selected, stale = 0, 0
    saved = copy.deepcopy(net.state_dict())
    optimizer = torch.optim.AdamW([
        dict(params=net.gru.parameters(), weight_decay=.001),
        dict(params=net.head.parameters(), weight_decay=0.)], lr=float(config['lr']))
    generator = torch.Generator().manual_seed(int(seed) + 9007)
    trace = []
    for epoch in range(1, max_epochs + 1):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('Frozen fitting deadline reached')
        optimizer.zero_grad(set_to_none=True)
        view = 'short' if epoch % 2 else 'sparse'
        fine = net(x['full'], mask=masks['full'])
        coarse = net(x[view], mask=masks[view])
        raw_a = sample_tokens(train, view, q, generator)
        raw_b = sample_tokens(train, view, q, generator)
        a = net(completed_network_input(raw_a, train), mask=masks['full'])
        b = net(completed_network_input(raw_b, train), mask=masks['full'])
        penalty, positive = rectified_two_sample(coarse, a, b, mid)
        mse = .5 * (weights * ((fine-y).square().mean(1) + (coarse-y).square().mean(1))).mean()
        shrink_loss = .5 * (weights * (fine.square().mean(1) + coarse.square().mean(1))).mean()
        head_l2 = .01 * (net.head.weight.square().sum() + net.head.bias.square().sum()) / 20
        objective = mse + shrink * shrink_loss + head_l2 + float(config['strength']) * penalty
        objective.backward()
        gradient = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            val = validation_score(validation, net(vx, mask=vm))
        row = dict(epoch=epoch, view=view, objective=float(objective.detach()),
                   supervised=float(mse.detach()), prediction_penalty=float(shrink_loss.detach()),
                   mechanism_penalty=float(penalty.detach()), positive_moment_fraction=positive,
                   mean_squared_completion_difference=float((raw_a-raw_b).square().mean()),
                   mean_squared_prediction_difference=float((a-b).square().mean().detach()),
                   preclip_gradient=float(gradient), validation_score=val)
        if not all(np.isfinite(v) for v in row.values() if isinstance(v, (float, int))):
            raise FloatingPointError('Nonfinite MCSSL control trace; no retry')
        trace.append(row)
        if val < best - 1e-6:
            best, selected, stale = val, epoch, 0
            saved = copy.deepcopy(net.state_dict())
        else:
            stale += 1
        if stale >= 12:
            break
    net.load_state_dict(saved)
    q_hash_after = state_hash(state_arrays(q))
    assert q_hash_before == q_hash_after
    receipt = dict(config=dict(config), seed=int(seed),
                   parameter_count=sum(p.numel() for p in net.parameters()),
                   selected_epoch=selected, epochs=len(trace), initial_validation=initial_validation,
                   validation_score=best, trace=trace,
                   state_sha256=state_hash(state_arrays(net)),
                   initial_state_sha256=state_hash(initial_arrays),
                   imputer_unchanged=True, imputer_state_sha256=q_hash_after,
                   imputer_parameter_count=sum(p.numel() for p in q.parameters()),
                   seconds=time.perf_counter()-started,
                   train_counts={name: int((train['market']==i).sum()) for i,name in enumerate(train['market_names'])},
                   mechanism='market/horizon rectified two-independent-completion product',
                   original_paper_unbiasedness_claimed=False,
                   control_is_author_benchmark_reproduction=False,
                   input_includes_market_identity=True, masks_passed_explicitly=True,
                   model_selection_on_confirmation=False)
    return dict(kind='neural', net=net, config=dict(config), market_names=train['market_names']), receipt


def algebra_check():
    configure(); torch.manual_seed(815)
    c, a, b = [torch.randn(8,20,requires_grad=True) for _ in range(3)]
    mid = torch.as_tensor([0,0,0,0,1,1,1,1])
    actual, _ = rectified_two_sample(c,a,b,mid)
    expected = sum(max(0., sum(float((c[i,h]-a[i,h])*(c[i,h]-b[i,h]))
                              for i in range(8) if int(mid[i])==m)/4)
                   for m in range(2) for h in range(20))/40
    assert abs(float(actual)-expected) < 1e-12
    return dict(passed=True, forward_max_error=abs(float(actual)-expected),
                real_market_fits=0, optimizer_steps=0)
