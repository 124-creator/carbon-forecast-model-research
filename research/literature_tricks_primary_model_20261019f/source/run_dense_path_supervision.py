"""Stage f: dense future-path supervision of the unchanged response model.

Train on the 20 cumulative price changes, select on the original H1/5/20
validation score, and issue the same three evaluated forecasts. Future path
values are labels only. This is a standard multi-output training adaptation,
not teacher distillation, an additional independent sample, or a new model.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import sys
import time
import json

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parent
STUDY = SOURCE.parent
REPO = STUDY.parent.parent
E_SOURCE = STUDY.parent / "signed_response_search_20261019e/source"
sys.path.insert(0, str(E_SOURCE))
import run_signed_response_search as core
from signed_response_models import FitConfig, SignedResponse, solve_readout, ordered_rates

FAMILIES = ("signed_context_response", "static_response", "amplitude_context", "signed_shock_only")
TRAIN_HORIZONS = tuple(range(1, 21))
EVAL_INDICES = np.asarray(core.HORIZONS) - 1
MAX_EPOCHS = 320


def contract():
    return {
        "block": "raw_dense_path_supervision",
        "families": list(FAMILIES),
        "train_horizons": list(TRAIN_HORIZONS),
        "selection_and_evaluation_horizons": list(core.HORIZONS),
        "features": "exact stage-e raw inputs, price scale and volume context",
        "training_loss": "mean over origins and H1..20 of squared cumulative-change error divided by initial-training target second moment, plus original standardized-column ridge",
        "training_second_moment_floor": "original train price RMS squared times 1e-6",
        "validation_loss": "original mean over H1/5/20 with original initial-training variance",
        "alpha_grid": list(core.MEAN_ALPHAS), "learning_rates": list(core.LRS),
        "fit_config_except_alpha_lr": asdict(FitConfig(maximum_epochs=MAX_EPOCHS)),
        "selection_tie_order": ["validation_loss", "alpha", "learning_rate", "selected_epoch"],
        "refit": "fresh original initialization; mature pretest origins; exactly chosen updates",
        "planned_selection_fits": 128, "planned_final_refits": 8,
        "input_information": "history only at all prediction origins; future path used only as matured training labels",
        "new_fitted_parameters": 0,
        "independent_sample_count": "unchanged; 17 extra dependent supervision coordinates per origin",
        "source_sha256": {str(p.relative_to(REPO)): core.digest(p) for p in
                          (Path(__file__), E_SOURCE / "signed_response_models.py", E_SOURCE / "run_signed_response_search.py")},
    }


def dense_data(spec):
    d = core.prepare(spec)
    p, origins = d["price"], d["origins"]
    labels = np.column_stack([p[origins+h] - p[origins] for h in TRAIN_HORIZONS])
    assert np.array_equal(labels[:, EVAL_INDICES], d["y"])
    second = np.maximum(np.mean(labels[d["masks"]["train"]]**2, axis=0), d["scale"]**2 * 1e-6)
    # Use the original exact arithmetic at the three comparison coordinates.
    second[EVAL_INDICES] = d["variance"]
    d["dense_y"], d["dense_variance"] = labels, second
    return d


def fit_dense(family, x_train, y_train, variance, config, *,
              x_val=None, y_val=None, eval_variance=None, fixed_epochs=None):
    if (x_val is None) != (y_val is None):
        raise ValueError("Validation inputs and labels must be paired")
    if x_val is not None and eval_variance is None:
        raise ValueError("Original three-horizon validation variance is required")
    model = SignedResponse(family, horizons=TRAIN_HORIZONS)
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=config.learning_rate)
    max_epochs = config.maximum_epochs if fixed_epochs is None else int(fixed_epochs)
    initial = {k: v.detach().clone() for k, v in model.named_parameters()}
    attempted = {k: v.detach().clone() for k, v in model.named_parameters()}
    best_value, best_epoch, best_state = float("inf"), 0, None
    history, stale, positive_steps = [], 0, 0
    for epoch in range(max_epochs + 1):
        beta, objective, train_loss, residual, column_scale = solve_readout(
            model.basis(x_train), y_train, variance, config.alpha)
        model.set_coefficients(beta, column_scale)
        with torch.no_grad():
            selection = (torch.mean((model(x_val)[:, EVAL_INDICES] - y_val)**2 / eval_variance)
                         if x_val is not None else train_loss)
        value = float(selection.detach())
        if not np.isfinite(value) or not bool(torch.isfinite(objective)):
            raise FloatingPointError("Nonfinite dense-path objective or validation score")
        record = {"epoch": epoch, "training_data_loss": float(train_loss.detach()),
                  "training_profiled_objective": float(objective.detach()),
                  "validation_loss": value if x_val is not None else None,
                  "linear_normal_equation_max_residual": float(residual.detach().abs().max()),
                  "historical_rates": ordered_rates(model.rate_raw).detach().tolist()}
        history.append(record)
        if value < best_value - config.selection_tolerance:
            best_value, best_epoch = value, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == max_epochs:
            break
        if fixed_epochs is None and epoch >= config.minimum_epochs and stale >= config.patience:
            break
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        gradients = [p.grad for p in params if p.grad is not None]
        if len(gradients) != len(params) or not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise FloatingPointError("Invalid nonlinear gradient")
        norm = float(torch.sqrt(sum(torch.sum(g*g) for g in gradients)))
        if norm > 0:
            positive_steps += 1
        record["gradient_norm_before_clip"] = norm
        torch.nn.utils.clip_grad_norm_(params, config.gradient_clip)
        optimizer.step()
        attempted = {k: v.detach().clone() for k, v in model.named_parameters()}
    if fixed_epochs is None:
        model.load_state_dict(best_state)
        selected_epoch = best_epoch
    else:
        selected_epoch = history[-1]["epoch"]
    receipt = {
        "family": family, "config": asdict(config), "deterministic": True,
        "training_horizons": list(TRAIN_HORIZONS), "evaluation_horizons": list(core.HORIZONS),
        "parameter_counts": model.parameter_counts(),
        "selected_epoch": selected_epoch, "epochs_executed": history[-1]["epoch"],
        "positive_gradient_steps": positive_steps,
        "validation_loss": best_value if x_val is not None else None,
        "selected_nonlinear_parameter_max_change": {k: float((v-initial[k]).detach().abs().max())
                                                     for k, v in model.named_parameters()},
        "attempted_last_nonlinear_parameter_max_change": {k: float((v-initial[k]).abs().max())
                                                           for k, v in attempted.items()},
        "linear_readout_fitted_at_epoch0": True,
        "state_dict": {k: v.detach().tolist() for k, v in model.state_dict().items()},
        "history": history,
    }
    return model, receipt


def run_job(spec, family, out):
    torch.set_num_threads(1)
    d = dense_data(spec)
    m = d["masks"]
    x = core.tensor(d["x"])
    y = core.tensor(d["dense_y"] / d["scale"])
    variance = core.tensor(d["dense_variance"] / d["scale"]**2)
    eval_y, eval_variance = core.normalized_labels(d)
    dest = Path(out) / f"{spec['market']}_{core.FOLD}" / family
    trials = []
    for alpha in core.MEAN_ALPHAS:
        for lr in core.LRS:
            config = FitConfig(alpha=alpha, learning_rate=lr, maximum_epochs=MAX_EPOCHS)
            _, receipt = fit_dense(family, x[m["train"]], y[m["train"]], variance, config,
                                   x_val=x[m["val"]], y_val=eval_y[m["val"]], eval_variance=eval_variance)
            name = f"selection_alpha{alpha:g}_lr{lr}.json"
            core.write_json(dest / name, receipt)
            trials.append({"alpha": alpha, "learning_rate": lr, "receipt": name,
                           **{k: receipt[k] for k in ("selected_epoch", "validation_loss", "epochs_executed", "positive_gradient_steps")}})
    choice = min(trials, key=lambda t: (t["validation_loss"], t["alpha"], t["learning_rate"], t["selected_epoch"]))
    config = FitConfig(alpha=choice["alpha"], learning_rate=choice["learning_rate"], maximum_epochs=MAX_EPOCHS)
    model, receipt = fit_dense(family, x[m["final"]], y[m["final"]], variance, config,
                              fixed_epochs=choice["selected_epoch"])
    core.write_json(dest / "SELECTION.json", {"selected": choice, "trials": trials,
                                             "selection_horizons": list(core.HORIZONS), "training_horizons": list(TRAIN_HORIZONS)})
    core.write_json(dest / "REFIT.json", receipt)
    with torch.no_grad():
        path_pred = model(x[m["test"]]).numpy() * d["scale"]
    pred = path_pred[:, EVAL_INDICES]
    np.savez_compressed(dest / "predictions.npz", predictions=pred[None],
                        observed=d["y"][m["test"]], origins=d["origins"][m["test"]],
                        horizons=np.asarray(core.HORIZONS), deterministic=np.array([True]),
                        dense_path_predictions=path_pred[None])
    result = core.summarize(d, family, pred)
    result.update({"enhancement": "raw_dense_path_supervision", "analysis_role": "exposed development",
                   "selected_epoch": choice["selected_epoch"], "parameter_counts": model.parameter_counts(),
                   "selection_positive_gradient_steps": sum(t["positive_gradient_steps"] for t in trials),
                   "refit_positive_gradient_steps": receipt["positive_gradient_steps"],
                   "maximum_epochs": MAX_EPOCHS})
    core.write_json(dest / "SUMMARY.json", result)
    return result


def preflight(specs):
    torch.set_num_threads(1)
    manifests = []
    for spec in specs:
        d = dense_data(spec)
        manifests.append({**d["manifest"], "dense_train_target_second_moment": d["dense_variance"].tolist()})
    # This audit does not evaluate a market model: compare fixed synthetic basis.
    synthetic = torch.zeros((3, 64, 3), dtype=torch.float64)
    synthetic[:, :, 0] = torch.linspace(-.7, .9, 64)
    synthetic[:, -1, 1] = torch.tensor([-.4, 0., .6])
    synthetic[:, -1, 2] = 1.
    gaps = {}
    for family in FAMILIES:
        short = SignedResponse(family)
        dense = SignedResponse(family, horizons=TRAIN_HORIZONS)
        dense.load_state_dict(short.state_dict())
        gaps[family] = float((dense.basis(synthetic)[:, EVAL_INDICES] - short.basis(synthetic)).abs().max().detach())
        assert gaps[family] == 0
        assert dense.parameter_counts() == short.parameter_counts()
    return {"status": "PASS_NO_MARKET_FIT_NO_FORECAST_SCORE", "market_fits": 0, "forecast_scores": 0,
            "contract": contract(), "manifests": manifests, "synthetic_basis_subset_max_gaps": gaps,
            "additional_training_coordinates_are_not_independent_samples": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--expected-plan-sha")
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    specs = core.read_datasets(args.datasets)
    assert len(specs) == 2 and all(s["role"] == "development" for s in specs)
    if args.preflight_report:
        if args.preflight_report.exists():
            raise FileExistsError("Use a fresh preflight receipt")
        core.write_json(args.preflight_report, preflight(specs))
        print("PASS_NO_MARKET_FIT_NO_FORECAST_SCORE", flush=True)
        return
    if args.out is None or args.plan is None or args.expected_plan_sha is None:
        parser.error("A run requires --out, --plan and --expected-plan-sha")
    assert core.digest(args.plan) == args.expected_plan_sha
    plan = json.loads(args.plan.read_text())
    assert plan["dense_path"] == contract()
    assert plan["datasets_sha256"] == core.digest(args.datasets)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("A run requires a new empty output directory")
    if args.out.resolve() == REPO.resolve() or REPO.resolve() in args.out.resolve().parents:
        raise ValueError("Per-origin arrays must stay outside the public repository")
    protocol = {"created_unix": time.time(), "contract": contract(), "runtime": core.runtime(),
                "plan_path": str(args.plan.resolve()), "plan_sha256": args.expected_plan_sha,
                "datasets": specs, "all_results_are_exposed_development": True}
    core.write_json(args.out / "PROTOCOL_BEFORE_RUN.json", protocol)
    for spec in specs:
        d = dense_data(spec)
        core.write_json(args.out / f"{spec['market']}_{core.FOLD}" / "INPUT_MANIFEST.json",
                        {**d["manifest"], "dense_train_target_second_moment": d["dense_variance"].tolist()})
    summaries = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = {pool.submit(run_job, spec, family, args.out): f"{spec['market']}/{family}"
                 for spec in specs for family in FAMILIES}
        for future in as_completed(tasks):
            result = future.result()
            summaries.append(result)
            core.write_json(args.out / "PROGRESS_SUMMARIES.json", summaries)
            print(json.dumps({"completed": tasks[future], "Q": result["mean_mse_over_rw"],
                              "selected_epoch": result["selected_epoch"]}), flush=True)
    core.write_json(args.out / "ALL_SUMMARIES.json", summaries)
    unchanged = contract() == protocol["contract"] and core.digest(args.plan) == args.expected_plan_sha
    core.write_json(args.out / "EXECUTION_COMPLETE.json", {"status": "completed" if unchanged else "IDENTITY_FAILURE",
                    "summary_rows": len(summaries), "source_and_plan_unchanged": unchanged,
                    "selection_fits": 128, "final_refits": 8,
                    "positive_gradient_steps": sum(r["selection_positive_gradient_steps"] + r["refit_positive_gradient_steps"] for r in summaries)})
    assert unchanged


if __name__ == "__main__":
    main()
