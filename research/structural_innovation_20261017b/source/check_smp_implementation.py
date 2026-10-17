"""Bounded implementation smoke check; synthetic inputs only, no optimizer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from smp_model import SMPModel, VARIANTS, hurdle_nll


def main():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    x = torch.randn(9, 20, 7, dtype=torch.float64)
    state = torch.arange(9)
    delta = torch.tensor([0, -.1, .2, 0, -.3, .4, 0, -.5, .6], dtype=x.dtype)
    event = state % 3 != 0
    records = []
    for variant in VARIANTS:
        torch.manual_seed(17)
        net = SMPModel(7, variant=variant).double()
        applied = net.initialize_head(.8, .35)
        prediction, aux = net(x, state, return_aux=True)
        rw_exact = torch.equal(prediction, torch.zeros_like(prediction))
        assert prediction.shape == (9, 20) and rw_exact
        probability_error = float((aux['q'].sum(-1) - 1.0).abs().max().detach())
        assert probability_error < 1e-14
        assert aux['q'].shape == (9, 9, 3) and aux['b'].shape == (9, 9, 2)
        assert aux['q_current'].shape == (9, 3) and aux['b_current'].shape == (9, 2)
        loss = hurdle_nll(aux, delta, event=event).mean()
        loss.backward()
        gradients_finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in net.parameters())
        assert gradients_finite
        records.append({
            'variant': variant,
            'parameters_at_input_dim7': sum(p.numel() for p in net.parameters()),
            'encoder_parameters': sum(p.numel() for p in net.encoder.parameters()),
            'head_parameters': sum(p.numel() for p in net.q_head.parameters()) + sum(p.numel() for p in net.b_head.parameters()),
            'q_free_groups': net.n_q_groups, 'b_free_groups': net.n_b_groups,
            'strict_RW_initialization': rw_exact,
            'max_probability_sum_error': probability_error,
            'nll': float(loss.detach()),
            'finite_gradients': gradients_finite,
            'initialization': applied,
        })
    root = Path(__file__).resolve().parent.parent
    receipt = {
        'status': 'PASS',
        'scope': 'Synthetic shape, symmetric RW initialization, simplex, and finite NLL backward only; no parameter update.',
        'market_files_read': 0, 'confirmation_files_read': 0,
        'market_fits': 0, 'synthetic_optimizer_steps': 0,
        'torch_version': torch.__version__, 'dtype': 'float64', 'threads': torch.get_num_threads(),
        'model_sha256': hashlib.sha256((root / 'source/smp_model.py').read_bytes()).hexdigest(),
        'check_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'checks': records,
        'limits': [
            'No independent path enumeration or finite-difference check in this smoke receipt; these are assigned to the independent auditor.',
            'Input dimension7 is a synthetic example, not a claim about the final study input channel count.',
            'The common9state contract includes sign0 age0/1 states not reached at normal length20 market origins; counts are nominal total parameters.',
            'At zero head weights the first encoder gradient is zero by construction; this is not a claim of encoder gradient coverage.',
        ],
    }
    destination = root / 'audits/SMP_IMPLEMENTATION_CHECK.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'model_sha256': receipt['model_sha256'], 'receipt': str(destination)}))


if __name__ == '__main__':
    main()
