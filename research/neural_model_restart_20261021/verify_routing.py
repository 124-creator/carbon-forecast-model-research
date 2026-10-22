# -*- coding: utf-8 -*-
"""Small mathematical checks for the distribution-routing implementation."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import norm
import multiscale_round1 as M


def main():
    torch.set_num_threads(2)
    torch.manual_seed(45)
    base = np.tile(norm.ppf(M.TAUS), (len(M.HOR), 1)).astype(np.float32)
    net = M.ScaleRouter(base, 'dual_route')
    # Avoid a trivial check in which zero-initialized heads all equal the anchor.
    for head in net.expert:
        torch.nn.init.normal_(head[-1].weight, std=.15)
        torch.nn.init.normal_(head[-1].bias, std=.1)
    seq = torch.randn(40, len(M.SCALES), M.K, 3)
    ctx = torch.randn(40, 8)
    hi = torch.arange(40) % len(M.HOR)
    q, wl, wg = net(seq, ctx, hi)
    assert (torch.diff(q, dim=1) > 0).all()
    assert torch.allclose(wl.sum(1), torch.ones(len(hi)), atol=1e-6)
    assert torch.allclose(wg.sum(1), torch.ones(len(hi)), atol=1e-6)
    assert (wl >= 0).all() and (wg >= 0).all()
    M.losses(torch.randn(len(hi)), q).mean().backward()
    assert net.route_location[-1].weight.grad.abs().sum() > 0
    assert net.route_gaps[-1].weight.grad.abs().sum() > 0
    net.zero_grad()
    shared = M.ScaleRouter(base, 'shared_route')
    shared.load_state_dict(net.state_dict())
    net.route_gaps.load_state_dict(net.route_location.state_dict())
    q_dual = net(seq, ctx, hi)[0]
    q_shared = shared(seq, ctx, hi)[0]
    delta = float((q_dual-q_shared).abs().max().detach())
    assert delta < 1e-6
    result = {'strictly_ordered_quantiles': True, 'route_weights_are_probabilities': True,
              'both_routes_receive_gradients': True,
              'shared_gate_is_nested_special_case': True,
              'maximum_nested_case_difference': delta,
              'nontrivial_randomized_expert_heads': True}
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
