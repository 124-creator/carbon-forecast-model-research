"""Independent NumPy loss/gradient oracle; no fit_neural invocation or updates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import fit as subject_fit
from models import FAMILIES


ROOT = Path(__file__).resolve().parent.parent


def main():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    source_path = ROOT / 'source/fit.py'
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    n, horizon = 4, 20
    scales = np.array([.01, 1., 10., 100.])
    raw = np.arange(n * horizon).reshape(n, horizon) / 31. - .7
    raw[:, 0] = [0., -2., 4., .5e-10]
    event = np.array([0, 1, 2, 0])  # Explicit raw-price threshold, last row tiny nonzero.
    normalized = raw / scales[:, None]
    mean0 = np.linspace(-.2, .4, n * horizon).reshape(n, horizon)
    denominator = .7 * np.arange(1, 21)
    probabilities = np.array([[.5, .2, .3], [.2, .6, .2], [.25, .25, .5], [.7, .2, .1]])
    magnitude = np.array([[.2, .3], [.8, .9], [.3, .7], [.1, .4]])
    checks, records = [], []

    def check(name, ok, **detail):
        checks.append(dict(name=name, passed=bool(ok), **detail))
        if not ok:
            raise AssertionError(name)

    def expected(family, rho):
        # Hand-written arithmetic oracle, independent from subject objective.
        hh = np.arange(1, 21)
        if family == 'gru_normalized':
            residual = mean0 - normalized
            supervised = np.mean(residual ** 2 / hh)
            penalty = np.mean(mean0 ** 2 / hh)
            derivative = 2 * (residual + rho * mean0) / (n * horizon * hh)
        else:
            count = 1 if family == 'smp_one_step' else horizon
            raw_prediction = mean0[:, :count] * scales[:, None]
            residual = raw_prediction - raw[:, :count]
            supervised = np.mean(residual ** 2 / denominator[:count])
            penalty = np.mean(raw_prediction ** 2 / denominator[:count])
            derivative = np.zeros_like(mean0)
            derivative[:, :count] = 2 * scales[:, None] * (residual + rho * raw_prediction) / (n * count * denominator[:count])
        aw = .05 if (family.startswith('smp_') and family != 'smp_no_aux') or family == 'gru_aux' else 0.
        likelihood = 0.
        dq, db = np.zeros_like(probabilities), np.zeros_like(magnitude)
        if aw:
            individual = []
            for i, e in enumerate(event):
                if e == 0:
                    individual.append(-np.log(probabilities[i, 0]))
                else:
                    k = int(e) - 1
                    individual.append(-np.log(probabilities[i, e]) + np.log(magnitude[i, k]) + abs(normalized[i, 0]) / magnitude[i, k])
                    db[i, k] = aw / n * (1 / magnitude[i, k] - abs(normalized[i, 0]) / magnitude[i, k] ** 2)
                dq[i, e] = -aw / (n * probabilities[i, e])
            likelihood = float(np.mean(individual))
        return np.array([supervised + rho * penalty + aw * likelihood, supervised, penalty, likelihood]), derivative, dq, db

    for family in FAMILIES:
        for rho in (.1, 1.):
            mean = torch.tensor(mean0, requires_grad=True)
            q = torch.tensor(probabilities, requires_grad=True)
            b = torch.tensor(magnitude, requires_grad=True)
            aux = dict(q_current=q, b_current=b, log_q_current=q.log())
            batch = dict(scale=torch.tensor(scales), y=torch.tensor(normalized), delta=torch.tensor(raw), event=torch.tensor(event))
            result = subject_fit.objective(family, mean, aux, batch, torch.tensor(denominator), rho)
            oracle, dm, dq, db = expected(family, rho)
            actual = np.array([float(v.detach()) for v in result])
            check(f'{family}/{rho}/four_loss_components', np.allclose(actual, oracle, rtol=1e-12, atol=1e-12), max_abs_error=float(np.max(abs(actual - oracle))))
            result[0].backward()
            check(f'{family}/{rho}/raw_mean_gradient', np.allclose(mean.grad.numpy(), dm, rtol=1e-12, atol=1e-12), max_abs_error=float(np.max(abs(mean.grad.numpy() - dm))))
            if np.any(dq):
                check(f'{family}/{rho}/auxiliary_q_gradient', np.allclose(q.grad.numpy(), dq, rtol=1e-12, atol=1e-12))
                check(f'{family}/{rho}/auxiliary_b_gradient', np.allclose(b.grad.numpy(), db, rtol=1e-12, atol=1e-12))
            else:
                check(f'{family}/{rho}/no_auxiliary_path', q.grad is None and b.grad is None)
            if family == 'smp_one_step':
                check(f'{family}/{rho}/h2_to_h20_no_direct_gradient', np.array_equal(mean.grad.numpy()[:, 1:], np.zeros((n, 19))))
            records.append(dict(family=family, shrink=rho, loss_components=actual.tolist()))

    # Same raw prediction/truth, changed origin scale: native risk invariant,
    # normalized historical control changes by precisely 1/49 as intended.
    for family in ('gru_native', 'gru_normalized'):
        values = []
        for factor in (1., 7.):
            batch = dict(scale=torch.tensor(scales * factor), y=torch.tensor(normalized / factor), delta=torch.tensor(raw), event=torch.tensor(event))
            result = subject_fit.objective(family, torch.tensor(mean0 / factor), None, batch, torch.tensor(denominator), .1)
            values.append(np.array([float(v) for v in result[:3]]))
        expected_scaled = values[0] if family == 'gru_native' else values[0] / 49.
        check(f'{family}/unit_weight_control', np.allclose(values[1], expected_scaled, rtol=1e-12, atol=1e-12))
    check('fit_source_unchanged', source_hash == hashlib.sha256(source_path.read_bytes()).hexdigest())
    receipt = dict(status='PASS', checks=checks, check_count=len(checks), results=records,
                   fit_source_sha256=source_hash, audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   scope='Independent NumPy objective and analytic gradient oracle only; no fit_neural call.',
                   market_files_read=0, confirmation_files_read=0, market_fits=0, synthetic_fits=0, optimizer_steps=0,
                   limits=[
                       'Early stopping, deadline control, checkpoint restoration, and real scores are read-only static review here, not executed fitting tests.',
                       'one_step has h1 direct training gradients but shared h1/5/20 validation selection; it is not a no-multihorizon-information claim.',
                       'Same supplied auxiliary distributions test likelihood/control identity; representations may subsequently differ during training.',
                   ])
    destination = ROOT / 'audits/FIT_CONTROLS_AUDIT.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', checks=len(checks), fit_source_sha256=source_hash, optimizer_steps=0, receipt=str(destination))))


if __name__ == '__main__':
    main()
