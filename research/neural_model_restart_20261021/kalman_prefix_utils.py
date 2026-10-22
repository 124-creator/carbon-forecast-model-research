"""Differentiable scalar Kalman filtering by parallel doubling prefixes.

This is a computational utility, not a model. Importing it reads no files and
changes no thread settings. The CLI verifies synthetic float64 sequences only.
No frozen model source, market data, or existing experiment result is modified.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import torch


def _inputs(x, R, observed, phi, Q, mu, m0, P0, validate):
    if x.ndim != 1 or R.shape != x.shape:
        raise ValueError('x and R must be one-dimensional tensors of equal shape')
    if not x.is_floating_point() or R.dtype != x.dtype or R.device != x.device:
        raise ValueError('x and R must share a floating dtype and device')
    observed = torch.as_tensor(observed, dtype=torch.bool, device=x.device)
    if observed.shape != x.shape:
        raise ValueError('observed must have the same shape as x')
    phi, Q, mu, m0, P0 = [torch.as_tensor(v, dtype=x.dtype, device=x.device)
                          for v in (phi, Q, mu, m0, P0)]
    if any(v.ndim != 0 for v in (phi, Q, mu, m0, P0)):
        raise ValueError('phi, Q, mu, m0, P0 must be scalar tensors or numbers')
    # Sanitize BEFORE any arithmetic. Missing x/R may be NaN, Inf, or zero.
    safe_x = torch.where(observed, x, torch.zeros_like(x))
    safe_R = torch.where(observed, R, torch.ones_like(R))
    if validate:
        if not bool(torch.isfinite(safe_x).all() and torch.isfinite(safe_R).all()):
            raise ValueError('Observed x/R must be finite')
        if not bool((safe_R > 0).all()):
            raise ValueError('Observed R must be positive')
        scalars = torch.stack([phi, Q, mu, m0, P0])
        if not bool(torch.isfinite(scalars).all()):
            raise ValueError('Scalar inputs must be finite')
        if not bool((phi >= 0) & (phi < 1) & (Q > 0) & (P0 >= 0)):
            raise ValueError('Require 0<=phi<1, Q>0, P0>=0')
    return safe_x, safe_R, observed, phi, Q, mu, m0, P0


def _projective_normalize(matrix):
    # Sum per matrix, never across time. All entries are nonnegative and the
    # positive sum is a removable projective scale, including in autograd.
    return matrix / matrix.sum(dim=(-1, -2), keepdim=True)


def mobius_prefix(matrix):
    """Inclusive products M[t] @ ... @ M[0], with positive scale removal."""
    prefix = _projective_normalize(matrix)
    distance = 1
    while distance < len(prefix):
        composed = prefix[distance:] @ prefix[:-distance]  # NEWER @ OLDER
        prefix = torch.cat([prefix[:distance], _projective_normalize(composed)], dim=0)
        distance *= 2
    return prefix


def affine_prefix(A, B):
    """Inclusive affine compositions; never cumprod followed by division."""
    product, offset = A, B
    distance = 1
    while distance < len(A):
        new_product = product[distance:] * product[:-distance]
        new_offset = product[distance:] * offset[:-distance] + offset[distance:]
        product = torch.cat([product[:distance], new_product])
        offset = torch.cat([offset[:distance], new_offset])
        distance *= 2
    return product, offset


def kalman_prefix(x, R, observed, phi, Q, mu, m0, P0, *, validate=True):
    """Return all posterior (m, P, K) for one scalar Gaussian AR1 chain.

    Parameters phi/Q/mu are constant over this sequence. x/R can vary at every
    row. All seven numerical arguments may require gradients. Missing rows only
    propagate the prior. Float64 CPU equivalence is verified by this module;
    equivalence is within roundoff, not bitwise algebraic identity.
    """
    x, R, observed, phi, Q, mu, m0, P0 = _inputs(x, R, observed, phi, Q, mu, m0, P0, validate)
    n = len(x)
    if not n:
        return x.clone(), x.clone(), x.clone()
    f = phi.square()
    ones, zeros = torch.ones_like(x), torch.zeros_like(x)
    a = torch.where(observed, R * f, ones * f)
    b = torch.where(observed, R * Q, ones * Q)
    c = torch.where(observed, ones * f, zeros)
    d = torch.where(observed, Q + R, ones)
    matrix = torch.stack([torch.stack([a, b], dim=1), torch.stack([c, d], dim=1)], dim=1)
    prefix = mobius_prefix(matrix)
    P = (prefix[:, 0, 0] * P0 + prefix[:, 0, 1]) / (prefix[:, 1, 0] * P0 + prefix[:, 1, 1])
    previous_P = torch.cat([P0.reshape(1), P[:-1]])
    prior_P = f * previous_P + Q
    K = torch.where(observed, prior_P / (prior_P + R), zeros)
    A = (1 - K) * phi
    B = (1 - K) * (1 - phi) * mu + K * x
    product, offset = affine_prefix(A, B)
    m = product * m0 + offset
    return m, P, K


def frozen_serial(x, R, observed, phi, Q, mu, m0, P0, *, validate=True):
    """Synthetic reference: execute the frozen B16.step, without data loading."""
    import bridge_precision_round16 as B16
    x, R, observed, phi, Q, mu, m0, P0 = _inputs(x, R, observed, phi, Q, mu, m0, P0, validate)
    if not len(x):
        return x.clone(), x.clone(), x.clone()
    state = (m0, P0)
    theta = (mu, phi, Q, mu.new_zeros(()))
    means, variances, gains = [], [], []
    for t, flag in enumerate(observed.tolist()):
        # The unbound step needs no object when theta is explicitly supplied.
        state, gain = B16.PrecisionFilter.step(None, state, x[t], flag, R[t], theta)
        means.append(state[0]); variances.append(state[1]); gains.append(gain)
    return torch.stack(means), torch.stack(variances), torch.stack(gains)


def _case(n, phi, seed, pattern='mixed', nonfinite_missing=False):
    generator = torch.Generator().manual_seed(seed)
    dtype = torch.float64
    x = -8 + .45 * torch.randn(n, dtype=dtype, generator=generator)
    R = torch.exp(-1.3 + .6 * torch.randn(n, dtype=dtype, generator=generator))
    if pattern == 'all_missing':
        observed = torch.zeros(n, dtype=torch.bool)
    elif pattern == 'all_observed':
        observed = torch.ones(n, dtype=torch.bool)
    else:
        observed = torch.rand(n, generator=generator) > .27
        if n > 25:
            observed[n // 3:n // 3 + 11] = False
    if nonfinite_missing:
        x[~observed] = float('nan')
        R[~observed] = float('nan')
    scalars = [torch.tensor(v, dtype=dtype) for v in (phi, .02, -8.1, -7.9, .21)]
    return x, R, observed, *scalars


def _clone(case, gradients):
    x, R, observed, *scalar = case
    clone = lambda value: value.detach().clone().requires_grad_(gradients)
    return clone(x), clone(R), observed.clone(), *(clone(value) for value in scalar)


def _objective(outputs):
    m, P, K = outputs
    weights = torch.linspace(.7, 1.3, len(m), dtype=m.dtype, device=m.device)
    return (weights * ((m + 7.8).square() + .2 * P + .3 * K.square()
                       + .05 * torch.log(P + .03) + .02 * m * P)).mean()


def _gradient(fn, case):
    args = _clone(case, True)
    out = fn(*args)
    objective = _objective(out)
    variables = (args[3], args[4], args[5], args[6], args[7], args[0], args[1])
    raw = torch.autograd.grad(objective, variables, allow_unused=True)
    grad = [torch.zeros_like(value) if item is None else item for item, value in zip(raw, variables)]
    return tuple(value.detach() for value in out), grad, float(objective.detach())


def verify_case(n, phi, seed, pattern='mixed', nonfinite_missing=False):
    case = _case(n, phi, seed, pattern, nonfinite_missing)
    reference, gradients_ref, obj_ref = _gradient(frozen_serial, case)
    actual, gradients, obj = _gradient(kalman_prefix, case)
    errors = {}
    for name, value, target in zip(('m', 'P', 'K'), actual, reference):
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value, target, rtol=5e-12, atol=5e-12)
        errors[name] = float((value - target).abs().max())
    gradient_errors = {}
    for name, value, target in zip(('phi', 'Q', 'mu', 'm0', 'P0', 'x', 'R'), gradients, gradients_ref):
        assert torch.isfinite(value).all() and torch.isfinite(target).all()
        torch.testing.assert_close(value, target, rtol=2e-8, atol=2e-10)
        gradient_errors[name] = float((value - target).abs().max())
    missing = ~case[2]
    assert torch.equal(gradients[-2][missing], torch.zeros_like(gradients[-2][missing]))
    assert torch.equal(gradients[-1][missing], torch.zeros_like(gradients[-1][missing]))
    # Future x/R changes cannot influence any earlier prefix; parameters fixed.
    clean = list(_clone(case, False))
    clean[0] = torch.nan_to_num(clean[0], nan=0.)
    clean[1] = torch.nan_to_num(clean[1], nan=1.)
    with torch.no_grad():
        before = kalman_prefix(*clean)
        cut = max(1, n // 2)
        clean[0][cut:] += 6
        clean[1][cut:] *= 4
        after = kalman_prefix(*clean)
        for left, right in zip(before, after):
            torch.testing.assert_close(left[:cut], right[:cut], rtol=0, atol=0)
    return dict(n=n, phi=phi, pattern=pattern, missing=int(missing.sum()),
                nonfinite_missing=nonfinite_missing, state_max_abs_errors=errors,
                gradient_max_abs_errors=gradient_errors, objective_abs_error=abs(obj - obj_ref),
                future_mutation_prefix_exact=True, missing_x_R_gradients_zero=True)


def gradchecks():
    outputs = []
    for pattern in ('mixed', 'all_observed'):
        case = _case(5, .95, 303 if pattern == 'mixed' else 304, pattern)
        if pattern == 'mixed':
            case[2][1] = False
            case[2][2] = True
        x, R, observed, phi, Q, mu, m0, P0 = _clone(case, True)
        def function(phi, Q, mu, m0, P0, x, R):
            return kalman_prefix(x, R, observed, phi, Q, mu, m0, P0, validate=False)
        passed = torch.autograd.gradcheck(function, (phi, Q, mu, m0, P0, x, R),
            eps=1e-6, atol=2e-5, rtol=2e-4, raise_exception=True, fast_mode=False)
        outputs.append(dict(n=5, pattern=pattern, passed=bool(passed)))
    return outputs


def _time_function(fn, case, backward, repeats):
    def one():
        args = _clone(case, backward)
        if backward:
            value = fn(*args, validate=False)
            _objective(value).backward()
        else:
            with torch.no_grad():
                fn(*args, validate=False)
    one()  # warmup excluded
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        one()
        elapsed.append(time.perf_counter() - start)
    return dict(median_seconds=statistics.median(elapsed), minimum_seconds=min(elapsed),
                repeats=repeats, samples_seconds=elapsed)


def benchmark(repeats):
    rows = []
    for n in (128, 1500):
        case = _case(n, .95, 8000 + n)
        for backward in (False, True):
            serial = _time_function(frozen_serial, case, backward, repeats)
            prefix = _time_function(kalman_prefix, case, backward, repeats)
            rows.append(dict(n=n, phi=.95, operation='forward_backward' if backward else 'forward_no_grad',
                             serial=serial, prefix=prefix,
                             median_speedup=serial['median_seconds'] / prefix['median_seconds']))
    return rows


def verification(repeats=3):
    import bridge_precision_round16 as B16
    here = Path(__file__).resolve().parent
    frozen = {'B16': Path(B16.__file__).resolve(), 'R17': here / 'round17_shared_precision.py'}
    fingerprints = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in frozen.items() if path.exists()}
    started = time.perf_counter()
    cases = []
    for n in (128, 1500):
        for j, phi in enumerate((.5, .95, .994)):
            cases.append(verify_case(n, phi, 10000 + n + j, nonfinite_missing=True))
    cases.append(verify_case(128, .994, 20001, pattern='all_missing', nonfinite_missing=True))
    cases.append(verify_case(128, .5, 20002, pattern='all_observed'))
    checks = gradchecks()
    # Scalar n=1 and empty sequences are defined too.
    single = _case(1, .95, 50)
    for a, b in zip(kalman_prefix(*single), frozen_serial(*single)):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    empty = _case(0, .95, 51)
    assert all(value.numel() == 0 for value in kalman_prefix(*empty))
    times = benchmark(repeats)
    for name, expected in fingerprints.items():
        assert hashlib.sha256(frozen[name].read_bytes()).hexdigest() == expected
    max_state = {name: max(case['state_max_abs_errors'][name] for case in cases) for name in ('m', 'P', 'K')}
    max_grad = {name: max(case['gradient_max_abs_errors'][name] for case in cases)
                for name in ('phi', 'Q', 'mu', 'm0', 'P0', 'x', 'R')}
    return dict(status='PURE SYNTHETIC COMPUTATIONAL VERIFICATION; not a forecasting experiment',
        market_files_opened=False, experiment_results_opened=False,
        torch_version=torch.__version__, dtype='float64', cpu_threads=torch.get_num_threads(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frozen_source_sha256_unchanged=fingerprints,
        method='positive-sum-normalized Mobius inclusive doubling, newer@older; then affine-pair inclusive doubling',
        no_cumprod_division=True, missing_sanitized_before_arithmetic=True,
        states_match_frozen_step=True, all_seven_input_gradients_match=True,
        future_mutation_prefix_exact=True, gradchecks=checks,
        maximum_state_absolute_errors=max_state, maximum_gradient_absolute_errors=max_grad,
        cases=cases, benchmarks=times,
        benchmark_note='single-thread CPU wall time, includes equal input cloning; other background jobs can affect timing; no model accuracy claim; frozen inputs were not changed',
        numerical_scope='tested scalar time-constant phi/Q/mu and positive R, float64 CPU; equivalence within roundoff, not bitwise',
        elapsed_seconds=time.perf_counter() - started)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check-only', action='store_true', help='print verification without writing JSON')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--output', type=Path, default=Path(__file__).with_name('kalman_prefix_utils_validation.json'))
    args = ap.parse_args()
    if args.repeats < 1:
        raise ValueError('repeats must be positive')
    if not args.check_only and args.output.exists():
        raise FileExistsError(f'Existing verification output protected: {args.output}')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    result = verification(args.repeats)
    if not args.check_only:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {key: result[key] for key in ('status', 'source_sha256', 'market_files_opened',
        'states_match_frozen_step', 'all_seven_input_gradients_match',
        'maximum_state_absolute_errors', 'maximum_gradient_absolute_errors', 'gradchecks')}
    summary['timings'] = [{key: value for key, value in row.items() if key not in ('serial', 'prefix')} |
                          {'serial_median_seconds': row['serial']['median_seconds'],
                           'prefix_median_seconds': row['prefix']['median_seconds']} for row in result['benchmarks']]
    summary['json_output'] = None if args.check_only else str(args.output.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
