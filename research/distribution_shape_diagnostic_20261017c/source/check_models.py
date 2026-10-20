"""Pure synthetic checks and executed upstream-score counterexample.

No market data, archived forecasts, checkpoints or confirmation files are read.
Only code and source manifests are read. The output JSON must not exist.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import torch

import models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite synthetic receipt")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    here = Path(__file__).resolve().parents[1]
    checks, comparisons, maximum = [], 0, 0.0

    def close(name, actual, expected, tol=2e-12):
        nonlocal comparisons, maximum
        a = np.asarray(actual.detach().cpu() if isinstance(actual, torch.Tensor) else actual)
        e = np.asarray(expected.detach().cpu() if isinstance(expected, torch.Tensor) else expected)
        error = float(np.max(np.abs(a-e)))
        comparisons += a.size
        maximum = max(maximum, error)
        if not np.allclose(a, e, rtol=tol, atol=tol):
            raise AssertionError(name)
        checks.append(name)

    def require(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    # Import and execute the original distribution file directly; this test
    # does not depend on the author's constructor-validation adapter.
    upstream = here / "vendor/deep_npts/discrete_distribution.py"
    spec = importlib.util.spec_from_file_location("_original_discrete_counterexample", upstream)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    table = []
    for p in (0.1, 0.5, 0.9):
        atoms = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])
        weights = torch.tensor([[p, 1-p], [p, 1-p]])
        truth = torch.tensor([[-1.0], [1.0]])
        dist = module.DiscreteDistribution(atoms, weights, validate_args=False)
        rps = dist.rps(truth)
        crps = models.exact_crps(atoms, weights, truth[:, 0])
        close(f"upstream_rps_p{p}", rps, [0, p])
        close(f"upstream_negative_log_prob_p{p}", -dist.log_prob(truth), rps)
        close(f"exact_crps_p{p}", crps, [2*(1-p)**2, 2*p*p])
        table.append({"forecast_probability_at_minus_one": p,
                      "true_probabilities": [0.5, 0.5],
                      "author_rps_by_outcome": rps.tolist(),
                      "expected_author_rps": rps.mean().item(),
                      "exact_crps_by_outcome": crps.tolist(),
                      "expected_exact_crps": crps.mean().item()})
    require("author_score_prefers_incorrect_p0p1_over_true_p0p5",
            table[0]["expected_author_rps"] < table[1]["expected_author_rps"])
    require("exact_crps_prefers_true_p0p5", table[1]["expected_exact_crps"] < table[0]["expected_exact_crps"])
    p = torch.tensor(0.5, requires_grad=True)
    weights = torch.stack([p, 1-p])[None].expand(2, -1)
    dist = module.DiscreteDistribution(atoms, weights, validate_args=False)
    gradient_rps = torch.autograd.grad(dist.rps(truth).mean(), p, retain_graph=True)[0]
    gradient_crps = torch.autograd.grad(models.exact_crps(atoms, weights, truth[:, 0]).mean(), p)[0]
    close("upstream_score_gradient_at_true_distribution", gradient_rps, 0.5)
    close("exact_crps_gradient_at_true_distribution", gradient_crps, 0.0)

    torch.manual_seed(20261017)
    b, H, K = 4, 20, 31
    x = torch.randn(b, 20, 3)
    base = torch.linspace(-2, 2, K)[None, None, :].expand(b, H, K).clone()
    # Vary the baseline shape by horizon without changing its first two moments.
    powers = torch.linspace(0.8, 1.2, H)[None, :, None]
    base = base.sign()*base.abs().pow(powers)
    base = models.standardize_measure(base, torch.full_like(base, 1/K))
    x_before, base_before = x.clone(), base.clone()
    target = torch.randn(b, H)*1.3 + 0.2
    family_receipts, author_states, author_outputs = {}, {}, {}
    for family in models.FAMILIES:
        torch.manual_seed(1729)
        start = time.perf_counter()
        model = models.Model(family).double()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="To copy construct from a tensor")
            weights_before = {k: v.clone() for k, v in model.state_dict().items()}
            out, prob = model(x, base)
            require(family+"/shape", out.shape == prob.shape == (b, H, K))
            require(family+"/finite", bool(torch.isfinite(out).all()) and bool(torch.isfinite(prob).all()))
            require(family+"/ordered", bool((torch.diff(out, dim=-1) > 0).all()))
            require(family+"/prob_positive", bool((prob > 0).all()))
            close(family+"/mass", prob.sum(dim=-1), np.ones((b, H)))
            close(family+"/zero_mean", (out*prob).sum(dim=-1), np.zeros((b, H)))
            close(family+"/initial_unit_variance", (out.square()*prob).sum(dim=-1), np.ones((b, H)))
            if not family.startswith("deepnpts_"):
                close(family+"/common_FHS_initialization", out, base)
                close(family+"/uniform_probabilities", prob, np.full((b, H, K), 1/K))
            else:
                require(family+"/probability_floor", bool((prob >= 0.03/K).all()))
                require(family+"/registered_frozen_embedding", "author_dummy_embedding.weight" in weights_before and not model.author_dummy_embedding.weight.requires_grad)
                author_states[family] = weights_before
                author_outputs[family] = (out.detach().clone(), prob.detach().clone())

            loss = model.training_loss(out, prob, target)
            require(family+"/loss_shape_finite", loss.shape == target.shape and bool(torch.isfinite(loss).all()))
            loss.mean().backward()
            require(family+"/finite_gradients", all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None))
            require(family+"/nonzero_gradient", any(bool((p.grad != 0).any()) for p in model.parameters() if p.grad is not None))
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.003)
            optimizer.step()
            after, q = model(x, base)
            close(family+"/post_step_zero_mean", (after*q).sum(dim=-1), np.zeros((b, H)))
            if family in ("global_shape", "conditional_shape", "deepnpts_rps", "deepnpts_crps"):
                close(family+"/post_step_unit_variance", (after.square()*q).sum(dim=-1), np.ones((b, H)))
            if family == "scale_only":
                scale = (after.square()*q).sum(dim=-1).sqrt().unsqueeze(-1)
                close("scale_only/post_step_fixed_shape", after/scale, base)
            torch.manual_seed(9876)
            clone = models.Model(family).double()
            clone.load_state_dict(model.state_dict())
            again, again_q = clone(x, base)
            close(family+"/state_roundtrip_atoms", again, after, tol=0)
            close(family+"/state_roundtrip_probs", again_q, q, tol=0)
        family_receipts[family] = {
            "synthetic_seconds": time.perf_counter()-start,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_registered_parameters": sum(p.numel() for p in model.parameters()),
            "synthetic_optimizer_steps": 1,
        }
    require("author_pair_parameter_keys", set(author_states["deepnpts_rps"]) == set(author_states["deepnpts_crps"]))
    for name in author_states["deepnpts_rps"]:
        close("author_pair/initial_state/"+name, author_states["deepnpts_rps"][name], author_states["deepnpts_crps"][name], tol=0)
    close("author_pair/initial_atoms", author_outputs["deepnpts_rps"][0], author_outputs["deepnpts_crps"][0], tol=0)
    close("author_pair/initial_probs", author_outputs["deepnpts_rps"][1], author_outputs["deepnpts_crps"][1], tol=0)
    close("x_not_mutated", x, x_before, tol=0)
    close("base_not_mutated", base, base_before, tol=0)

    # Independent scalar-loop energy identity on arbitrary finite distributions.
    rng = np.random.default_rng(314)
    a = rng.normal(size=(7, 9)); w = rng.uniform(size=(7, 9)); w /= w.sum(axis=-1, keepdims=True)
    y = rng.normal(size=7)
    oracle = np.array([sum(w[t,i]*abs(a[t,i]-y[t]) for i in range(9)) -
                       0.5*sum(w[t,i]*w[t,j]*abs(a[t,i]-a[t,j]) for i in range(9) for j in range(9))
                       for t in range(7)])
    close("exact_CRPS_independent_scalar_loop", models.exact_crps(torch.tensor(a),torch.tensor(w),torch.tensor(y)), oracle)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "PASS",
        "checks": len(checks), "scalar_comparisons": comparisons,
        "maximum_absolute_difference": maximum, "check_names": checks,
        "environment": {"python":sys.version.split()[0],"numpy":np.__version__,"torch":torch.__version__},
        "source_sha256": {str(p.relative_to(here)):sha(p) for p in [Path(__file__),here/"source/models.py",here/"vendor/deep_npts/runtime.py",upstream,here/"vendor/deep_npts/_network.py",here/"vendor/deep_npts/SOURCE_MANIFEST.json"]},
        "original_score_counterexample": {
            "table":table,"gradient_at_true_p0p5":{"author_rps":gradient_rps.item(),"exact_crps":gradient_crps.item()},
            "conclusion":"For this executed fixed source version, rps/log_prob is not a proper score for the returned finite probability distribution. This is a source-level result; it does not assert every paper formula, version or estimator has the same defect.",
        },
        "family_receipts":family_receipts,
        "market_files_read":0,"confirmation_files_read":0,"market_fits":0,
        "synthetic_optimizer_steps_total":6,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as handle:
        json.dump(receipt,handle,indent=2,ensure_ascii=False,allow_nan=False);handle.write("\n")
    print(json.dumps({key:receipt[key] for key in ("status","checks","scalar_comparisons","maximum_absolute_difference")},ensure_ascii=False))
    print(json.dumps(receipt["original_score_counterexample"]["table"],ensure_ascii=False))


if __name__ == "__main__":
    main()
