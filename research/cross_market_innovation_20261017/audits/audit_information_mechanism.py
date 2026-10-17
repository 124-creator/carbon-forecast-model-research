"""Independent algebra / synthetic boundary review; no market I/O or fitting.

NumPy implements the penalty and its prediction gradient independently of the
authoring agent's torch implementation. No optimizer is constructed or called.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
MODEL_PATH = WORK / "scratch/innovation_review_20261017/model.py"
DATA_PATH = WORK / "carbon_workspace/research/cross_market_innovation_20261017/source/data_core.py"


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def numpy_penalty(fine, coarse, features, starts, ends, groups, hs=None):
    """Literal ordered-pair definition, independent of torch contractions."""
    n, horizon = fine.shape
    hs = np.ones(horizon) if hs is None else np.asarray(hs)
    d = (fine - coarse) / hs
    distance = ((features[:, None] - features[None, :]) ** 2).mean(-1)
    kernel = sum(np.exp(-distance / (2 * width ** 2)) for width in (.5, 1., 2.)) / 3
    records, gradients, terms = [], [], []
    for group in np.unique(groups):
        weights = np.zeros((n, n))
        count = 0
        for i in range(n):
            for j in range(n):
                eligible = (groups[i] == groups[j] == group and i != j and
                            (ends[i] < starts[j] or ends[j] < starts[i]))
                if eligible:
                    weights[i, j] = kernel[i, j]
                    count += 1
        if count == 0:
            records.append(dict(group=int(group), ordered_pair_count=0, skipped=True))
            continue
        u = np.array([sum(weights[i, j] * d[i, h] * d[j, h]
                          for i in range(n) for j in range(n)) / count
                      for h in range(horizon)])
        grad = 2 * (weights @ d) / count / horizon / hs
        grad[:, u <= 0] = 0
        gradients.append(grad)
        terms.append(np.maximum(u, 0).mean())
        records.append(dict(group=int(group), ordered_pair_count=count,
                            raw_u=u.tolist(), skipped=False))
    value = float(np.mean(terms)) if terms else 0.
    gradient = np.mean(gradients, axis=0) if gradients else np.zeros_like(fine)
    return value, records, gradient


def main():
    torch.set_num_threads(1)
    torch.manual_seed(71219)
    rng = np.random.default_rng(71219)
    model_module = module_from_file("independent_information_model", MODEL_PATH)
    data_module = module_from_file("independent_information_data", DATA_PATH)
    errors = []

    def check(actual, expected, tol=2e-11):
        a, b = np.asarray(actual), np.asarray(expected)
        assert a.shape == b.shape
        error = float(np.max(np.abs(a-b))) if a.size else 0.
        errors.append(error)
        assert error <= tol, (error, tol)

    # Model tensors are synthetic. Groups 0 and 1 contain eligible and excluded
    # pairs; group 2 has no pair because its supports overlap completely.
    groups = np.array([0]*4 + [1]*5 + [2]*2)
    origins = np.array([20, 60, 61, 150, 20, 21, 100, 140, 181, 20, 21])
    starts, ends = origins - 20, origins + 20
    features = rng.normal(size=(11, 7))
    fine = rng.normal(size=(11, 20))
    fine[:4] += .8
    coarse = rng.normal(size=(11, 20))
    hs = np.sqrt(np.arange(1, 21))
    expected, expected_records, gradient = numpy_penalty(
        fine, coarse, features, starts, ends, groups, hs)
    ft = torch.tensor(fine, dtype=torch.float64, requires_grad=True)
    ct = torch.tensor(coarse, dtype=torch.float64, requires_grad=True)
    zt = torch.tensor(features, dtype=torch.float64, requires_grad=True)
    actual, record = model_module.blocked_kernel_moment_penalty(
        ft, ct, zt, torch.tensor(starts), torch.tensor(ends),
        groups=torch.tensor(groups), horizon_scale=torch.tensor(hs))
    actual.backward()
    check(actual.detach().numpy(), expected)
    check(ft.grad.numpy(), gradient)
    check(ct.grad.numpy(), -gradient)
    assert zt.grad is None, "kernel coordinates must be detached"
    for a, e in zip(record["groups"], expected_records):
        assert a["group"] == e["group"] and a["skipped"] == e["skipped"]
        assert a["ordered_pair_count"] == e["ordered_pair_count"]
        if not e["skipped"]:
            check(a["raw_u"], e["raw_u"])

    # Inclusive support [t-20,t+20] requires origin distance > 40, not >= 40.
    assert not (ends[0] < starts[1]) and ends[0] < starts[2]
    assert record["groups"][-1]["skipped"]
    reversed_value, _, _ = numpy_penalty(fine[::-1], coarse[::-1], features[::-1],
                                        starts[::-1], ends[::-1], groups[::-1], hs)
    check(expected, reversed_value)

    # A conditional moment can cancel across markets if they are incorrectly
    # pooled into one group; the supplied grouping must stop that cancellation.
    delta = np.array([1., 1., -1., -1.])[:, None]
    zeros = np.zeros_like(delta)
    s4 = np.arange(4) * 100
    grouped, _, _ = numpy_penalty(delta, zeros, zeros, s4, s4+40, np.array([0,0,1,1]))
    pooled, _, _ = numpy_penalty(delta, zeros, zeros, s4, s4+40, np.zeros(4))
    check(grouped, 1.)
    check(pooled, 0.)
    same_sign_batches = [numpy_penalty(delta[ids], zeros[ids], zeros[ids],
                                     s4[ids], s4[ids]+40, np.zeros(2))[0]
                         for ids in (np.array([0,1]), np.array([2,3]))]
    opposite_batches = [numpy_penalty(delta[ids], zeros[ids], zeros[ids],
                                    s4[ids], s4[ids]+40, np.zeros(2))[0]
                        for ids in (np.array([0,2]), np.array([1,3]))]
    check(np.mean(same_sign_batches), 1.)
    check(np.mean(opposite_batches), 0.)

    # Synthetic prices and dates exercise the production NumPy data constructor.
    days = np.cumsum(np.resize(np.array([1,1,3,1,2]), 520)) + 733000
    date = np.array([data_module.date.fromordinal(int(d)).isoformat() for d in days])
    price = 100 + rng.normal(scale=.4, size=520).cumsum()
    price[85] = price[84]  # a valid unchanged quoted price
    market = data_module.Market("synthetic", date, days, price, "synthetic", "synthetic")
    t = np.array([100, 150, 210])
    xf, mf = data_module.inputs(market, t, 1.7, "full")
    nested = {}
    for view, vertex_ids in data_module.VIEW_VERTICES.items():
        xv, mv = data_module.inputs(market, t, 1.7, view)
        n = len(vertex_ids)-1
        expected_delta = np.stack([np.array([xf[row, a:b, 0].sum()
                                  for a, b in zip(vertex_ids[:-1], vertex_ids[1:])])
                                  for row in range(len(t))])
        check(xv[:, :n, 0], expected_delta)
        check(np.expm1(xv[:, :n, 1]), np.broadcast_to(np.diff(vertex_ids), (len(t), n)))
        fine_days = np.expm1(xf[..., 2])
        expected_days = np.stack([np.array([fine_days[row, a:b].sum()
                                 for a, b in zip(vertex_ids[:-1], vertex_ids[1:])])
                                 for row in range(len(t))])
        check(np.expm1(xv[:, :n, 2]), expected_days)
        assert (mv.sum(1) == n).all()
        changed_price = price.copy()
        for origin in t:
            deleted = np.setdiff1d(np.arange(21), vertex_ids)
            changed_price[origin-20+deleted] += 19.
        changed = data_module.Market("synthetic", date, days, changed_price, "", "")
        changed_view, _ = data_module.inputs(changed, t, 1.7, view)
        check(xv, changed_view, 0.)
        nested[view] = dict(valid_tokens=n, full_input_measurable=True,
                            deleted_price_invariance=True)

    # Appending a known market one-hot requires an explicit original raw mask.
    # All three IDs, including those with zero in the final channel, must work.
    ids = np.eye(3)
    network = model_module.SharedInformationGRU(input_size=6, hidden_size=16).double()
    xm, mask = data_module.inputs(market, t, 1.7, "sparse")
    augmented = np.concatenate([xm, np.broadcast_to(ids[:, None], (3,20,3))
                                * mask[..., None]], axis=2)
    output = network(torch.tensor(augmented), mask=torch.tensor(mask)).detach().numpy()
    assert output.shape == (3,20) and np.isfinite(output).all()
    perturbed = augmented.copy()
    perturbed[~mask] = 10000
    padded_output = network(torch.tensor(perturbed), mask=torch.tensor(mask)).detach().numpy()
    check(output, padded_output, 0.)

    # Counterexample: pooled Bayes predictions need not satisfy per-market tower
    # if known market is omitted from both model input information sets.
    # M=+-1 equally; P(Z=M | M)=.75; Y=M; coarse is constant.
    # f_fine=E[M|Z]=.5 Z, f_coarse=0, E[d|M]=.25 M.
    market_tower = dict(prob_z_matches_market=.75,
                        pooled_coarse_bayes=0., pooled_fine_bayes="0.5 * Z",
                        conditional_difference_by_market={"-1":-.25, "+1":.25},
                        fix="Include known market ID in BOTH fine and coarse information sets.")
    source_hash = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    receipt = dict(status="PASS", scope="independent synthetic formulas and input boundaries only",
                   market_reads=0, market_fits=0, optimizer_steps=0,
                   model_sha256=source_hash,
                   data_core_sha256=hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
                   audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   numpy_version=np.__version__, torch_version=torch.__version__,
                   numerical_comparisons=len(errors), maximum_absolute_error=max(errors),
                   exact_pair_counts=[r["ordered_pair_count"] for r in expected_records],
                   prediction_gradients_independently_verified=True,
                   kernel_feature_gradients_detached=True, nested_views=nested,
                   explicit_mask_with_market_onehot=True,
                   rectification_batch_dependence=dict(full_batch=pooled,
                       same_sign_half_batches=float(np.mean(same_sign_batches)),
                       opposite_sign_half_batches=float(np.mean(opposite_batches)),
                       implication="The training objective depends on batch composition and size."),
                   market_information_counterexample=market_tower,
                   limitations=["This is not a market forecasting result.",
                       "The training runner and final sampler need separate review.",
                       "Nonoverlap is not independence; this receipt proves neither consistency nor novelty."])
    output_path = HERE / f"INFORMATION_PREFLIGHT_{source_hash[:12]}.json"
    output_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(path=str(output_path), status=receipt["status"],
                         max_error=max(errors), model_sha256=source_hash), ensure_ascii=False))


if __name__ == "__main__":
    main()
