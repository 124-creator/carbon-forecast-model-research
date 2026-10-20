#!/usr/bin/env python3
"""Bounded synthetic learner checks; no market files or benchmark fitting."""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

import learning as module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--author-source", type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(180018)
    checks = []
    started = time.monotonic()

    def rand(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float64)

    def check(name, condition, **details):
        if not bool(condition):
            raise AssertionError((name, details))
        checks.append({"name": name, "passed": True, **details})

    def close(name, x, y, atol=1e-10):
        error = float((x.detach() - y.detach()).abs().max())
        check(name, error < atol, maximum_absolute_error=error, tolerance=atol)

    x, y, xv = rand(71, 8), rand(71, 20), rand(19, 8)
    precision = 0.1 * torch.eye(8, dtype=torch.float64)
    base_head = module.ridge_head(x, y, precision)
    a = torch.diag(torch.tensor([0.4, 0.7, 0.9, 1.1, 1.5, 1.8, 2.2, 3.0], dtype=torch.float64))
    a[0, 2], a[3, 6] = 0.6, -0.4
    moved, auxiliary = module.transport_precision(x, x @ a, precision, return_aux=True)
    close("pure_coordinate_A", auxiliary["A"], a)
    close("pure_coordinate_precision", moved, a.T @ precision @ a)
    new_head = module.ridge_head(x @ a, y, moved)
    close("pure_coordinate_head", new_head, torch.linalg.solve(a, base_head))
    close("pure_coordinate_prediction", xv @ a @ new_head, xv @ base_head)
    check("fixed_ridge_not_coordinate_invariant",
          float((xv @ a @ module.ridge_head(x @ a, y, precision) - xv @ base_head).abs().max()) > 1e-4)
    new_x = x @ a + 0.25 * rand(71, 8)
    old_c = module.covariance(x)
    mu = 0.63
    cov_moved = module.transport_precision(x, new_x, mu * old_c, mu)
    close("covariance_transport_exact_collapse", cov_moved, mu * module.covariance(new_x))
    different_residual_coefficient = module.transport_precision(x, new_x, mu * old_c, 0.1)
    check("covariance_collapse_requires_same_coefficient",
          float((different_residual_coefficient - mu * module.covariance(new_x)).abs().max()) > 1e-4)
    calibrated, df_receipt = module.covariance_calibration(x)
    c = module.covariance(x)
    calibrated_df = torch.trace(torch.linalg.solve(c + calibrated*c, c))
    check("initial_covariance_df_matched", abs(float(calibrated_df) - df_receipt["initial_df"]) < 1e-10)
    try:
        module.transport_precision(torch.ones(20, 8, dtype=torch.float64), torch.ones(20, 8, dtype=torch.float64), precision)
    except module.NumericalFailure:
        check("rank_failure_is_explicit_no_jitter", True)
    else:
        check("rank_failure_is_explicit_no_jitter", False)
    try:
        module.transport_precision(x, torch.ones_like(x), precision)
    except module.NumericalFailure:
        check("new_rank_failure_is_explicit", True)
    else:
        check("new_rank_failure_is_explicit", False)
    try:
        module.transport_precision(x, x, -precision)
    except module.NumericalFailure:
        check("indefinite_old_precision_is_rejected", True)
    else:
        check("indefinite_old_precision_is_rejected", False)

    u, uv, yv = rand(71, 8), rand(19, 8), rand(19, 20)
    def outer_scalar(value):
        xx, vv = x + value*u, xv + value*uv
        pp = module.transport_precision(x, xx, precision)
        ww = module.ridge_head(xx, y, pp)
        return (vv @ ww-yv).square().mean()
    scalar = torch.tensor(0.07, dtype=torch.float64, requires_grad=True)
    gradient = torch.autograd.grad(outer_scalar(scalar), scalar)[0]
    eps = 1e-6
    fd = (outer_scalar(float(scalar.detach())+eps)-outer_scalar(float(scalar.detach())-eps))/(2*eps)
    close("full_transport_ridge_gradient_matches_fd", gradient, fd, atol=1e-8)
    def covariance_inner(value):
        xx = x + value*u
        pp = module.covariance_precision(xx, calibrated)
        return module.profile_objective(xx, y, pp)[0]
    cov_gradient = torch.autograd.grad(covariance_inner(scalar), scalar)[0]
    cov_fd = (covariance_inner(float(scalar.detach())+eps)-covariance_inner(float(scalar.detach())-eps))/(2*eps)
    close("covariance_current_gradient_includes_dC", cov_gradient, cov_fd, atol=1e-8)
    xx = x + 0.005*u
    exact = module.ridge_head(xx, y, precision)
    one, info1 = module.head_gradient_steps(xx, y, base_head, old_c, precision, 1, True)
    five, info5 = module.head_gradient_steps(xx, y, base_head, old_c, precision, 5, True)
    many = module.head_gradient_steps(xx, y, base_head, old_c, precision, 200)
    check("five_head_steps_reduce_equation_residual",
          info5["relative_normal_equation_residual"] < info1["relative_normal_equation_residual"])
    close("sufficient_head_steps_converge_to_profile", many, exact)

    if args.author_source is not None:
        sys.path.insert(0, str(args.author_source.resolve()))
        from author_kernel import author_ridge, call_count
        source_ridge = author_ridge
        close("author_source_interface_equals_general_ridge", source_ridge(x, y, .1), base_head)
        calls_before = call_count()
    else:
        source_ridge = module.ridge_head
        calls_before = 0

    task_beta = rand(60, 20) * 0.07
    def batch(n):
        bx = rand(n, 20, 3)
        by = bx.reshape(n, -1) @ task_beta + 0.7*rand(n, 20)
        sigma = torch.exp(0.2*rand(n, 1)) * torch.sqrt(torch.arange(1, 21, dtype=torch.float64))[None, :]
        return {"x": bx, "y": by, "sigma": sigma, "delta": by*sigma}
    case = {"target_support": batch(60), "target_meta": batch(20),
            "target_val": batch(20), "target_test": batch(24),
            "sources": {"A": batch(145), "B": batch(145)}, "source_episodes": []}
    for name in case["sources"]:
        case["sources"][name]["origin"] = torch.arange(145, dtype=torch.long) + 60
    for step in range(3):
        case["source_episodes"].append({name: {"support": list(range(step, step+64)),
                                                "query": list(range(step+84, step+116))}
                                         for name in case["sources"]})
    config = {"max_steps": 3, "val_interval": 1, "patience": 4, "min_delta": 1e-6}
    outcomes = {}
    for name in module.MODEL_NAMES:
        result = module.fit(case, name, 17, config, source_ridge_fn=source_ridge)
        receipt = result["receipt"]
        check("fit_eligible_"+name, receipt["eligible"], status=receipt["status"], failure=receipt["failure"])
        check("fit_test_once_"+name, receipt["test_forward_calls"] == 1 and not receipt["test_target_values_used_in_fit"])
        check("fit_predictions_finite_"+name, result["predictions"].shape == (24,20) and np.isfinite(result["predictions"]).all())
        close("fit_units_"+name, torch.from_numpy(result["raw_predictions"]),
              torch.from_numpy(result["predictions"])*case["target_test"]["sigma"])
        check("fit_has_complete_trace_"+name, len(result["traces"]) == 4 and receipt["steps"] == 3)
        if args.author_source is not None:
            check("fit_author_calls_"+name, receipt["author_ridge_calls"] == 6, author_calls=receipt["author_ridge_calls"])
        outcomes[name] = result
    hashes = {v["receipt"]["initial_encoder_hash"] for v in outcomes.values()}
    check("nine_arms_share_identical_encoder_initialization", len(hashes) == 1)
    for name, result in outcomes.items():
        close("initial_source_weights_"+name,
              torch.tensor(result["traces"][0]["source_weights"]),
              torch.tensor([0.,0.] if name == "local_profile" else [.25,.25]), atol=1e-7)
    check("diagonal_best_state_has_r", outcomes["diagonal_profile"]["state"]["diagonal_log_precision"].shape == (8,))
    check("transport_best_state_has_precision", outcomes["transport"]["state"]["precision"].shape == (8,8))
    poisoned = copy.deepcopy(case)
    poisoned["target_test"]["y"] = torch.full_like(case["target_test"]["y"], float("nan"))
    poisoned["target_test"]["delta"] = torch.full_like(case["target_test"]["delta"], float("nan"))
    rerun = module.fit(poisoned, "transport", 17, config, source_ridge_fn=source_ridge)
    close("test_y_and_delta_poison_does_not_change_prediction",
          torch.from_numpy(rerun["predictions"]), torch.from_numpy(outcomes["transport"]["predictions"]), atol=1e-14)
    check("test_poison_does_not_change_selected_state", rerun["receipt"]["selected_state_hash"] == outcomes["transport"]["receipt"]["selected_state_hash"])
    expired = module.fit(case, "profile", 17, {**config, "deadline": time.monotonic()-1}, source_ridge_fn=source_ridge)
    check("deadline_is_ineligible_and_no_test", expired["receipt"]["status"] == "deadline" and not expired["receipt"]["eligible"] and expired["predictions"] is None)
    invalid = copy.deepcopy(case)
    invalid["sources"]["A"]["origin"] = torch.zeros(145, dtype=torch.long)
    try:
        module._source_indices(invalid, 0, ["A","B"], 20)
    except ValueError:
        check("singular_origin_key_enforces_purge", True)
    else:
        check("singular_origin_key_enforces_purge", False)
    alias_config = {"max_steps":3, "validation_interval":1, "patience":4, "min_delta":1e-6,
                    "learning_rate":.01, "meta_learning_rate":.1, "ridge_lambda":.1}
    alias_run = module.fit(case, "profile", 17, alias_config, source_ridge_fn=source_ridge)
    close("root_config_aliases_same_prediction", torch.from_numpy(alias_run["predictions"]),
          torch.from_numpy(outcomes["profile"]["predictions"]), atol=1e-14)
    check("root_config_aliases_same_selected_state", alias_run["receipt"]["selected_state_hash"] == outcomes["profile"]["receipt"]["selected_state_hash"])
    rank_bad = copy.deepcopy(case)
    rank_bad["target_support"]["x"] = torch.ones_like(rank_bad["target_support"]["x"])
    failed = module.fit(rank_bad, "covariance_profile", 17, config, source_ridge_fn=source_ridge)
    check("initial_covariance_rank_failure_returns_ineligible", not failed["receipt"]["eligible"] and failed["receipt"]["status"] == "numerical_failure" and failed["predictions"] is None)
    synthetic_steps = sum(v["receipt"]["steps"] for v in outcomes.values()) + rerun["receipt"]["steps"] + alias_run["receipt"]["steps"]
    receipt = {"status": "PASS", "checks": checks, "check_count": len(checks),
               "utc": dt.datetime.now(dt.timezone.utc).isoformat(), "seconds": time.monotonic()-started,
               "market_data_reads": 0, "confirmation_file_reads": 0, "market_fits": 0,
               "synthetic_learner_calls": len(outcomes)+4, "synthetic_gradient_steps": synthetic_steps,
               "author_ridge_calls_during_synthetic_fits": call_count()-calls_before if args.author_source else 0,
               "torch_version": torch.__version__, "numpy_version": np.__version__,
               "sources": {str(Path(module.__file__).name): hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                           Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
               "fit_receipts": {k: v["receipt"] for k,v in outcomes.items()},
               "scope": "Pure synthetic tensors, no market or author benchmark execution; optional author ridge kernel only"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(json.dumps({k:receipt[k] for k in ("status","check_count","seconds","synthetic_gradient_steps","sources")}))


if __name__ == "__main__":
    main()
