"""Frozen low-dimensional identification screen. No market data are read."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize_scalar

from persistence import (PI, HORIZONS, propagate, abs_normal, pair_matrix,
                         population_risk, sample_paths, path_emissions,
                         path_nll_from_emissions)

ARMS = ("conditional_endpoint", "constant_endpoint", "conditional_joint005", "constant_joint005")
SCENARIOS = ("conditional_persistence", "constant_persistence", "misspecified_leverage")
SIZES = (256, 1024, 4096)
SEEDS = tuple(range(1000, 1020))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize_targets(path):
    targets = np.cumsum(path, axis=1)
    skeleton = propagate([.5])
    return {h: abs_normal(targets[:, h-1]/np.sqrt(h), skeleton[h][1]).mean(axis=0) for h in HORIZONS}


def endpoint_objective(rho, first):
    distributions = propagate([rho])
    values = []
    for h in HORIZONS:
        w, s = distributions[h]
        values.append(float(w[0]@first[h]-.5*w[0]@pair_matrix(s)@w[0]))
    return float(np.mean(values))


def bounded_fit(first, emissions, auxiliary):
    calls = 0
    def objective(rho):
        nonlocal calls
        calls += 1
        risk = endpoint_objective(rho, first)
        if auxiliary:
            risk += auxiliary*float(path_nll_from_emissions(emissions, rho).mean())
        return risk
    grid = np.linspace(0., 1., 21)
    grid[-1] = .999
    losses = np.array([objective(float(r)) for r in grid])
    best = int(np.argmin(losses))
    lower, upper = grid[max(0, best-1)], grid[min(len(grid)-1, best+1)]
    fitted = minimize_scalar(objective, bounds=(lower, upper), method="bounded",
                             options={"xatol": 1e-7, "maxiter": 100})
    candidates = [(float(grid[best]), float(losses[best])), (float(fitted.x), float(fitted.fun))]
    rho, loss = min(candidates, key=lambda item: item[1])
    return rho, {"rho": rho, "objective": loss, "success": bool(fitted.success),
                 "message": str(fitted.message), "objective_calls": calls,
                 "refinement_iterations": int(fitted.nit), "grid_best_index": best,
                 "boundary_solution": rho <= 1e-6 or rho >= .999-1e-6}


def evaluation_blocks(local):
    """Separate misspecified truth, 40 independent balanced MC batches."""
    rng = np.random.default_rng(799001)
    group = np.arange(200000) % 2
    truth = np.where(group == 0, .1, .9)
    path = sample_paths(rng, truth, leverage=.25)
    arrays = {}
    for g in range(2):
        batches = np.array_split(path[group == g], 40)
        statistics = [summarize_targets(batch) for batch in batches]
        for h in HORIZONS:
            arrays[f"g{g}_h{h}"] = np.stack([item[h] for item in statistics])
    np.savez_compressed(local/"misspecified_test_statistics.npz", **arrays)
    return {"truth_paths": 200000, "seed": 799001, "independent_batches": 40,
            "group_balance": [100000,100000], "sha256": sha(local/"misspecified_test_statistics.npz")}


def expected_scores(rhos, scenario, local):
    if scenario != "misspecified_leverage":
        truth = [.1, .9] if scenario == "conditional_persistence" else [.5, .5]
        scores = population_risk(rhos, truth)
        return float(scores.mean()), scores.mean(axis=0).tolist(), None
    stats = np.load(local/"misspecified_test_statistics.npz")
    values = []
    pred = propagate(rhos)
    for g in range(2):
        per_h = []
        for h in HORIZONS:
            w, s = pred[h]
            first = stats[f"g{g}_h{h}"]@w[g]
            per_h.append(first-.5*w[g]@pair_matrix(s)@w[g])
        values.append(np.stack(per_h, axis=-1))
    scores = np.mean(values, axis=0)
    return float(scores.mean()), scores.mean(axis=0).tolist(), scores.mean(axis=-1).tolist()


def run_job(args):
    scenario, seed, local_string = args
    started = time.time()
    local = Path(local_string)
    scenario_offset = SCENARIOS.index(scenario)*100000
    rng = np.random.default_rng(seed+scenario_offset)
    group = np.arange(max(SIZES)) % 2
    rhos = np.where(group == 0, .1, .9)
    if scenario == "constant_persistence":
        rhos[:] = .5
    path = sample_paths(rng, rhos, leverage=.25 if scenario == "misspecified_leverage" else 0.)
    path_file = local/"training_paths"/f"{scenario}_seed{seed}.npz"
    np.savez_compressed(path_file, path=path, group=group)
    records = []
    for n in SIZES:
        p, g = path[:n], group[:n]
        first = {k: summarize_targets(p[g == k]) for k in (0,1)}
        first_all = {h: .5*(first[0][h]+first[1][h]) for h in HORIZONS}
        emissions = path_emissions(p, stable=True)
        for arm in ARMS:
            auxiliary = .05 if arm.endswith("joint005") else 0.
            conditional = arm.startswith("conditional")
            if conditional:
                fits = [bounded_fit(first[k], emissions[g == k], auxiliary) for k in (0,1)]
                fitted_rhos = [fit[0] for fit in fits]
            else:
                fits = [bounded_fit(first_all, emissions, auxiliary)]
                fitted_rhos = [fits[0][0], fits[0][0]]
            risk, risk_h, batches = expected_scores(fitted_rhos, scenario, local)
            records.append({"scenario": scenario, "seed": seed, "n": n, "arm": arm,
                            "rho_minus": fitted_rhos[0], "rho_plus": fitted_rhos[1],
                            "population_CRPS": risk, "population_CRPS_h": dict(zip(map(str,HORIZONS),risk_h)),
                            "evaluation": "independent_MC" if batches is not None else "exact",
                            "MC_batch_scores": batches,
                            "scalar_optimizers": [fit[1] for fit in fits],
                            "fitted_parameters": 2 if conditional else 1})
    receipt = {"scenario": scenario, "seed": seed, "max_training_paths": len(path),
               "nested_sizes": list(SIZES), "training_paths_sha256": sha(path_file),
               "elapsed_seconds": time.time()-started, "records": records}
    out = local/"receipts"/f"{scenario}_seed{seed}.json"
    out.write_text(json.dumps(receipt, indent=2)+"\n")
    return receipt


def build_summary(receipts, evaluation):
    all_records = [record for receipt in receipts for record in receipt["records"]]
    groups = []
    for scenario in SCENARIOS:
        for n in SIZES:
            selected = [r for r in all_records if r["scenario"] == scenario and r["n"] == n]
            by_seed = {seed: {r["arm"]:r for r in selected if r["seed"] == seed} for seed in SEEDS}
            arms = {}
            for arm in ARMS:
                row = [v[arm] for v in by_seed.values()]
                arms[arm] = {"mean_population_CRPS": float(np.mean([r["population_CRPS"] for r in row])),
                             "mean_rho_minus": float(np.mean([r["rho_minus"] for r in row])),
                             "mean_rho_plus": float(np.mean([r["rho_plus"] for r in row])),
                             "boundary_scalar_fits": sum(f["boundary_solution"] for r in row for f in r["scalar_optimizers"])}
            ratios = np.array([v["conditional_joint005"]["population_CRPS"]/v["constant_joint005"]["population_CRPS"]-1 for v in by_seed.values()])
            vs_endpoint = np.array([v["conditional_joint005"]["population_CRPS"]/v["constant_endpoint"]["population_CRPS"]-1 for v in by_seed.values()])
            endpoint_ratios = np.array([v["conditional_endpoint"]["population_CRPS"]/v["constant_endpoint"]["population_CRPS"]-1 for v in by_seed.values()])
            groups.append({"scenario": scenario, "n": n, "arms": arms,
                           "primary_relative_mean": float(ratios.mean()),
                           "primary_relative_sd": float(ratios.std(ddof=1)),
                           "primary_wins": int(np.sum(ratios < 0)),
                           "primary_vs_endpoint_constant_relative_mean": float(vs_endpoint.mean()),
                           "endpoint_only_relative_mean": float(endpoint_ratios.mean()),
                           "per_seed_primary_relative": dict(zip(map(str,SEEDS),ratios.tolist()))})
    gate_row = next(r for r in groups if r["scenario"] == "conditional_persistence" and r["n"] == 1024)
    checks = {"at_least_2bps": gate_row["primary_relative_mean"] <= -.0002,
              "at_least_14_of_20_wins": gate_row["primary_wins"] >= 14,
              "no_worse_than_endpoint_constant": gate_row["primary_vs_endpoint_constant_relative_mean"] <= 0}
    return {"decision": "PASS_FINITE_SYNTHETIC_RESOURCE_GATE" if all(checks.values()) else "STOP_PERSISTENCE_CANDIDATE_BEFORE_MARKETS",
            "gate_checks": checks, "groups": groups, "model_fits": len(all_records),
            "scalar_optimizers": sum(len(r["scalar_optimizers"]) for r in all_records),
            "neural_gradient_fits": 0,
            "scalar_objective_calls": sum(f["objective_calls"] for r in all_records for f in r["scalar_optimizers"]),
            "failed_scalar_refinements": sum(not f["success"] for r in all_records for f in r["scalar_optimizers"]),
            "unique_training_paths": len(receipts)*max(SIZES),
            "misspecified_evaluation": evaluation,
            "warning": "The fixed resource gate is not a significance or publication-readiness test; perfect binary features and known pi/scale make this an identification positive control."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    args.local_dir.mkdir(parents=True, exist_ok=True)
    args.public_dir.mkdir(parents=True, exist_ok=True)
    for name in ("training_paths", "receipts"):
        (args.local_dir/name).mkdir(exist_ok=True)
    started = time.time()
    start = {"started_unix": started, "source_sha256": sha(__file__),
             "core_sha256": sha(Path(__file__).with_name("persistence.py")),
             "protocol_sha256": sha(Path(__file__).parents[1]/"provenance"/"PROTOCOL.json"),
             "workers": args.workers}
    (args.public_dir/"START.json").write_text(json.dumps(start, indent=2)+"\n")
    evaluation = evaluation_blocks(args.local_dir)
    print(json.dumps({"stage": "independent_misspecified_truth_ready", **evaluation}), flush=True)
    jobs = [(scenario, seed, str(args.local_dir)) for scenario in SCENARIOS for seed in SEEDS]
    receipts = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run_job, job): job for job in jobs}
        for future in as_completed(pending):
            receipt = future.result()
            receipts.append(receipt)
            if len(receipts) % 3 == 0:
                print(json.dumps({"completed_jobs": len(receipts), "total_jobs": len(jobs),
                                  "elapsed_seconds": time.time()-started}), flush=True)
    receipts.sort(key=lambda r:(r["scenario"], r["seed"]))
    summary = build_summary(receipts, evaluation)
    summary["elapsed_seconds"] = time.time()-started
    (args.public_dir/"SUMMARY.json").write_text(json.dumps(summary, indent=2)+"\n")
    # Synthetic fitted parameters cannot reconstruct any third-party prices.
    (args.public_dir/"ALL_FIT_RECEIPTS.json").write_text(json.dumps(receipts, indent=2)+"\n")
    print(json.dumps({"stage": "complete", "decision": summary["decision"], "gate_checks": summary["gate_checks"],
                      "model_fits": summary["model_fits"], "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
