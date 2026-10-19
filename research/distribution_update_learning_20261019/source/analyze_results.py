"""Aggregate every fixed experiment; no model or budget selection is performed."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from learning import circular_indices

PRIMARY = "path_block_gate3_delta005"
RAW = "raw_block_gate3_same_joint_budget"


def json_write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False))


def bootstrap_comparison(cell_arrays, methods, comparator, seed=219031):
    """Within-fixed-cell paired block diagnostic, not cross-market population inference."""
    rng = np.random.default_rng(seed)
    samples, effects = [], []
    bi = methods.index(comparator)
    for cell, scores in sorted(cell_arrays.items()):
        h = int(cell.rsplit("_h", 1)[1])
        indices = circular_indices(len(scores), 2*h, 1999, rng)
        bootstrap = scores[indices].mean(axis=1)
        samples.append(bootstrap/bootstrap[:, bi:bi+1]-1)
        mean = scores.mean(axis=0)
        effects.append(mean/mean[bi]-1)
    samples, effects = np.mean(samples, axis=0), np.asarray(effects)
    out = {}
    for j, method in enumerate(methods):
        effect = effects[:, j]
        out[method] = {
            "mean_cell_relative_difference": float(effect.mean()),
            "paired_within_cell_block_interval95": np.quantile(samples[:, j], [.025, .975]).tolist(),
            "cells_better": int(np.sum(effect < -1e-10)), "cells_tied": int(np.sum(np.abs(effect) <= 1e-10)),
            "cells_worse": int(np.sum(effect > 1e-10)), "n_cells": len(effects),
            "comparator": comparator,
        }
    return out


def analyze(local, root):
    out, figs = root/"results"/"market", root/"figures"
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    receipts = [json.loads(p.read_text()) for p in sorted((local/"receipts").glob("*.json"))]
    if len(receipts) != 132:
        raise ValueError(f"Expected 132 finished gradient fits, found {len(receipts)}")
    jobs_by_cell = defaultdict(list)
    for receipt in receipts:
        jobs_by_cell[receipt["cell"]].append(receipt)
    if len(jobs_by_cell) != 22 or any(len(v) != 6 for v in jobs_by_cell.values()):
        raise ValueError("Incomplete fixed cell/backbone/seed matrix")
    cell_score, cell_metrics, cell_rows, by_backbone = {}, {}, [], {b: {} for b in ("MLP32_16", "GRU16")}
    methods = None
    n_test_total = 0
    for cell, jobs in sorted(jobs_by_cell.items()):
        loaded = [dict(np.load(local/"predictions"/f'{r["job"]}.npz')) for r in jobs]
        names = loaded[0]["methods"].tolist()
        if methods is None:
            methods = names
        if names != methods or any(p["methods"].tolist() != methods for p in loaded):
            raise ValueError("Method order changed across runs")
        tensor = np.stack([p["metrics"][:, :, :5] for p in loaded])
        means = tensor.mean(axis=0)
        cell_score[cell], cell_metrics[cell] = means[:, :, 0], means
        n_test_total += len(means)
        rw = float(loaded[0]["rw_squared_error"].mean())
        for backbone in by_backbone:
            take = [j for j, r in enumerate(jobs) if r["backbone"] == backbone]
            by_backbone[backbone][cell] = tensor[take, :, :, 0].mean(axis=0)
        base = means[:, methods.index("F0")].mean(axis=0)
        for j, method in enumerate(methods):
            item = means[:, j].mean(axis=0)
            cell_rows.append({"cell": cell, "market": cell.rsplit("_", 2)[0], "year": int(cell.rsplit("_", 2)[1]),
                              "horizon": int(cell.rsplit("_h", 1)[1]), "method": method, "n_origins": len(means),
                              "CRPS_normalized": float(item[0]), "CRPS_ratio_F0": float(item[0]/base[0]),
                              "price_CRPS": float(item[1]), "price_CRPS_ratio_F0": float(item[1]/base[1]),
                              "price_mean_MSE_ratio_RW": float(item[2]/rw), "coverage90": float(item[3]),
                              "interval_width_price_units": float(item[4])})
    comparison_base = bootstrap_comparison(cell_score, methods, "F0")
    comparison_raw = bootstrap_comparison(cell_score, methods, RAW)
    comparison_noprefix = bootstrap_comparison(cell_score, methods, "path_no_prefix_gate3_delta005")
    backbone_summaries = {b: bootstrap_comparison(values, methods, RAW, seed=stable) for b, values, stable in
                          [(b, by_backbone[b], 720+i) for i, b in enumerate(by_backbone)]}
    method_rows = []
    for method in methods:
        rows = [r for r in cell_rows if r["method"] == method]
        summary = {"method": method, **comparison_base[method],
                   "mean_cell_CRPS_ratio_F0": float(np.mean([r["CRPS_ratio_F0"] for r in rows])),
                   "mean_cell_price_CRPS_ratio_F0": float(np.mean([r["price_CRPS_ratio_F0"] for r in rows])),
                   "mean_cell_price_mean_MSE_ratio_RW": float(np.mean([r["price_mean_MSE_ratio_RW"] for r in rows])),
                   "mean_cell_coverage90": float(np.mean([r["coverage90"] for r in rows]))}
        method_rows.append(summary)
    gate_receipts = [r["calibration_gates"][PRIMARY] for r in receipts]
    main_nonzero = sum(max(g["weights"]) > 1e-10 for g in gate_receipts)
    selected_control = sum(any(lam != 0 for lam in g["selected_lambdas"]) for g in gate_receipts)
    control_caused_change = 0
    control_nonzero_changed_jobs = []
    for r in receipts:
        p = np.load(local/"predictions"/f'{r["job"]}.npz')
        ma, ra = p["metrics"][:, methods.index(PRIMARY), 5], p["metrics"][:, methods.index(RAW), 5]
        if np.max(np.abs(ma-ra)) > 1e-10:
            control_caused_change += 1
            control_nonzero_changed_jobs.append(r["job"])
    effects = comparison_raw[PRIMARY]
    gates = {
        "paired_interval_below_zero": effects["paired_within_cell_block_interval95"][1] < 0,
        "at_least_60pct_cells_improve": effects["cells_better"]/22 >= .6,
        "not_worse_than_no_prefix": comparison_noprefix[PRIMARY]["mean_cell_relative_difference"] <= 1e-10,
    }
    proceed = all(gates.values())
    result = {
        "status": "completed", "decision": "ELIGIBLE_TO_PLAN_CONFIRMATION" if proceed else "STOP_FIXED_CANDIDATE_NO_CONFIRMATION_OPENED",
        "primary": PRIMARY, "primary_comparator": RAW, "advance_gates": gates,
        "counts": {"markets": 4, "market_year_horizon_cells": 22, "neural_gradient_fits": 132,
                   "global_mixture_fits": 22, "GARCH_fits": 22, "methods": len(methods),
                   "forecast_score_streams": 132*len(methods), "distinct_within_cell_evaluation_origins": n_test_total,
                   "selected_epoch_zero": sum(r["fit"]["selected_epoch"] == 0 for r in receipts),
                   "optimizer_steps": sum(r["fit"]["gradient_optimizer_steps"] for r in receipts),
                   "nominal_training_epochs": 60*132,
                   "main_nonzero_gate_fits": main_nonzero, "main_nonzero_control_selected_fits": selected_control,
                   "main_control_changes_gate_vs_same_joint_raw": control_caused_change},
        "uncertainty_scope": "Seeds and both backbones are averaged within each cell; their scores are not an ensemble CDF score. Bootstrap resamples overlapping origins in 2*h blocks within the 22 fixed evaluation cells. Intervals are conditional development diagnostics and do not certify independent markets, cross-market population generalization, unknown sign bounds, or arbitrary future shifts.",
        "time_seconds": {"sum_neural_job_runtime": sum(r["elapsed_seconds"] for r in receipts)},
        "primary_vs_raw": effects,
        "primary_vs_no_prefix": comparison_noprefix[PRIMARY],
        "method_results": method_rows,
        "all_vs_same_joint_raw": comparison_raw,
        "per_backbone_vs_same_joint_raw": backbone_summaries,
        "control_changed_jobs": control_nonzero_changed_jobs,
        "new_confirmation_values_opened": False,
    }
    json_write(out/"SUMMARY.json", result)
    json_write(out/"CELL_METRICS.json", cell_rows)
    with (out/"CELL_METRICS.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cell_rows[0]))
        writer.writeheader(); writer.writerows(cell_rows)
    compact_receipts = []
    for r in receipts:
        item = {k: v for k, v in r.items() if k != "metrics"}
        # Validation history is aggregate training evidence, not per-origin data.
        compact_receipts.append(item)
    json_write(out/"FIT_AND_GATE_RECEIPTS.json", compact_receipts)
    make_figures(root, result, cell_rows, figs)
    print(json.dumps({"decision": result["decision"], "counts": result["counts"], "primary_vs_raw": effects}, indent=2))


def make_figures(root, result, cells, figs):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
    chosen = [("F0", "Frozen base"), ("F1", "Neural proposal"), ("empirical_gate3", "Empirical gate"),
              ("raw_block_gate3_same_joint_budget", "Raw update (matched)"), (PRIMARY, "Path budget update"),
              ("path_no_prefix_gate3_delta005", "No-prefix ablation"), ("BOA_CDF_frozen", "BOA CDF frozen"),
              ("BOA_CDF_online", "BOA CDF online"), ("quantile_BOAG_online", "19-atom quantile BOAG"),
              ("FHS", "Filtered historical"), ("GARCH_Normal", "GARCH Normal"), ("EWMA_Normal", "EWMA Normal")]
    lookup = {r["method"]: r for r in result["method_results"]}
    values = [100*(lookup[m]["mean_cell_CRPS_ratio_F0"]-1) for m, _ in chosen]
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#b44630" if m == PRIMARY else "#416c8a" for m, _ in chosen]
    ax.barh(np.arange(len(chosen)), values, color=colors)
    ax.set_yticks(np.arange(len(chosen)), [label for _, label in chosen])
    ax.invert_yaxis(); ax.axvline(0, color="black", lw=.8)
    ax.set_xlabel("Mean cell change in normalized CRPS vs frozen base (%) | lower is better")
    ax.set_title("Development: 4 markets, 22 cells, 2 backbones, 3 seeds\nScores averaged within each cell; no reserved-period confirmation")
    fig.tight_layout(); fig.savefig(figs/"market_method_comparison.png"); fig.savefig(figs/"market_method_comparison.svg"); plt.close(fig)

    syn = json.loads((root/"results"/"synthetic"/"GENERAL_SYNTHETIC.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    names = ["gaussian_h1_negative", "gaussian_h5", "gaussian_h20", "heavy_tail_h5", "leverage_h5"]
    labels = ["Gaussian H1", "Gaussian H5", "Gaussian H20", "Heavy tail H5", "Leverage H5"]
    receipts = {r["scenario"]: r for r in syn["scenarios"]}
    axes[0].barh(labels, [receipts[n]["test_variance_ratio"] for n in names], color="#416c8a")
    axes[0].axvline(1, color="black", ls="--", lw=.8)
    axes[0].set_xlabel("Var(B - C) / Var(B)")
    axes[0].set_title("The control reduces estimation noise")
    differences = []
    for n in names:
        rows = {r["method"]: r for r in syn["results"] if r["scenario"] == n and r["n"] == 2048}
        differences.append(10000*(rows["path_bernstein_valid_budget"]["risk_ratio"]-rows["raw_bernstein"]["risk_ratio"]))
    axes[1].barh(labels, differences, color=["#b44630" if v > 0 else "#3b8263" for v in differences])
    axes[1].axvline(0, color="black", lw=.8)
    axes[1].set_xlabel("Net CRPS change vs raw-safe (basis points of base risk)")
    axes[1].set_title("Confidence and range costs limit net gains (n=2048)")
    fig.tight_layout(); fig.savefig(figs/"synthetic_variance_vs_risk.png"); fig.savefig(figs/"synthetic_variance_vs_risk.svg"); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.local_dir, args.root)
