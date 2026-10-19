"""Independent official Tianjin audit; no experiment imports or fitting.

The final run is guarded by its completion marker. --self-test uses only
synthetic arrays and never opens official market data or prediction files.
"""
from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import independent_v1_statistics as core

STUDY = core.STUDY
MARKET, FOLD = "TIANJIN_OFFICIAL_MEAN", "fixed602020"
PLAN_SHA = "8645aa650d5202852d00700f5bb7a47f907c5c7476db7dd84a923cecc7eae913"
NEURAL = ["liquidity_full", "liquidity_readout", "shock_size_only", "liquidity_unconstrained",
          "allocation_constant", "allocation_readout", "gru", "lstm", "swewt_gru_price", "swewt_gru_full"]
CLASSICAL = ["rw", "drift", "ridge_price", "ridge_full", "hgb_full"]
FAMILIES = NEURAL + CLASSICAL
PARAMETERS = {"liquidity_full": 11, "liquidity_readout": 11, "shock_size_only": 9,
              "liquidity_unconstrained": 11, "allocation_constant": 7, "allocation_readout": 169,
              "gru": 4500, "lstm": 5780, "swewt_gru_price": 4308, "swewt_gru_full": 4788}
ZERO_HEAD = {"gru", "lstm", "swewt_gru_price", "swewt_gru_full"}
CONTRASTS = [
    ("primary_readout_vs_rw", "allocation_readout", "rw"),
    ("primary_readout_vs_constant", "allocation_readout", "allocation_constant"),
    ("primary_readout_vs_swewt_full", "allocation_readout", "swewt_gru_full"),
    ("secondary_full_vs_rw", "liquidity_full", "rw"),
    ("secondary_full_vs_shock_size", "liquidity_full", "shock_size_only")]
SUMMARY_KEYS = ["mean_seed_mse_over_rw", "seed_mse_over_rw_by_horizon", "rw_mse_by_horizon",
                "ensemble_mse_over_rw", "ensemble_mse_over_rw_by_horizon"]


def intervals(observed, losses, length, rng, replications=50000, contrasts=CONTRASTS):
    """Same circular observed-origin weights and RW denominator for all pairs."""
    n = len(observed)
    core.check(0 < length < n, "nondegenerate observed-origin block length")
    weights = core.calendar_weights(rng, replications, n, length)
    denominator = weights @ (observed ** 2)
    valid = np.isfinite(denominator).all(axis=1) & (denominator > 0).all(axis=1)
    invalid = int((~valid).sum())
    result = {}
    for name, candidate, comparator in contrasts:
        if invalid:
            ci95 = ci99 = None
        else:
            numerator = weights @ (losses[comparator] - losses[candidate])
            draws = (numerator / denominator).mean(axis=1)
            core.check(np.isfinite(draws).all(), "finite paired effect draws")
            ci95 = np.quantile(draws, [.025, .975]).tolist()
            ci99 = np.quantile(draws, [.005, .995]).tolist()
        result[name] = {"block_observed_origins": length, "replications": replications,
                        "ci95_marginal": ci95, "ci99_five_contrast_family": ci99,
                        "undefined_rw_denominator_resamples": invalid,
                        "status": "approximate_percentile" if not invalid else "CI_UNAVAILABLE_ZERO_OR_NONFINITE_RW_DENOMINATOR"}
    return result


def self_test():
    """Known effects, seed estimand and undefined-denominator policy; no data."""
    n = 93
    y = (1 + np.arange(n)[:, None] / n) * np.array([[1., 2., 3.]])
    pred = np.stack([np.zeros_like(y), 2 * y])
    metric, _ = core.score(pred, y)
    core.close(metric["mean_seed_mse_over_rw"], 1., "seed-mean loss is not ensemble loss")
    core.close(metric["ensemble_mse_over_rw"], 0., "opposite errors cancel only in ensemble")
    loss = {"candidate": .9 * y ** 2, "rw": y ** 2}
    rng = np.random.default_rng(20261019)
    pair = [("known", "candidate", "rw")]
    for length in [20, 40]:
        result = intervals(y, loss, length, rng, replications=512, contrasts=pair)["known"]
        core.close(result["ci95_marginal"], [.1, .1], "known paired95 effect")
        core.close(result["ci99_five_contrast_family"], [.1, .1], "known paired99 effect")
    sparse = np.zeros((93, 3)); sparse[0] = [1, 2, 3]
    bad = intervals(sparse, {"candidate": .9 * sparse ** 2, "rw": sparse ** 2}, 20,
                    np.random.default_rng(20261019), replications=512, contrasts=pair)["known"]
    core.check(bad["undefined_rw_denominator_resamples"] > 0 and bad["ci95_marginal"] is None
               and bad["ci99_five_contrast_family"] is None, "undefined draws must not be silently dropped")
    return {"status": "PASS_SYNTHETIC_STATISTICS_NO_TRAINING", "model_fits": 0,
            "market_data_reads": 0, "checks": ["mean-seed and ensemble estimands differ as intended",
            "known paired effect under both20/40 circular observed-origin blocks",
            "95 and99 percentile conventions", "undefined ratio draws disable intervals"]}


def reconstruct(manifest, plan, data_file):
    src = manifest["sources"][0]
    identity = plan["data_identity"]
    core.check(core.sha(data_file) == src["sha256"] == identity["prepared_sha256"],
               "official prepared source byte identity")
    df = pd.read_csv(data_file)
    core.check(list(df) == identity["columns"] and len(df) == identity["rows"] == 1077,
               "official schema and all retained rows")
    core.check(df.date.is_unique and df.date.is_monotonic_increasing, "unique increasing source dates")
    core.check([df.date.iloc[0], df.date.iloc[-1]] == identity["date_range"], "full source date range")
    p = df.price.to_numpy(float); volume = df.volume_tons.to_numpy(float)
    turnover = df.turnover_yuan.to_numpy(float)
    core.check(np.isfinite(np.column_stack([p, volume, turnover])).all()
               and (p > 0).all() and (volume > 0).all() and (turnover >= 0).all(), "official finite totals")
    cents_gap = float(np.max(np.abs(100 * p - np.round(100 * p))))
    identity_gap = float(np.max(np.abs(p - turnover / volume)))
    core.check(cents_gap < 1e-9 and identity_gap <= .005000001, "published rounded price check")
    # Deliberately retain p: turnover/volume is only a rounding QC reference.
    dates = pd.to_datetime(df.date).to_numpy(dtype="datetime64[D]")
    n = len(df); a = n * 3 // 5; b = a + n // 5
    core.check(manifest["row_boundaries"] == plan["split"]["row_boundaries"] == [0, a, b, n],
               "fixed sequential60/20/20 boundaries")
    core.check(dates[b] > np.datetime64("2017-12-31"), "later-than2017 launch gate")
    old_checks = []
    for receipt in plan["launch_date_gate"]["independent_old_source_date_checks"]:
        path = STUDY / "data/sciencedb_819_v1/prepared" / (receipt["market"] + ".csv")
        core.check(core.sha(path) == receipt["source_sha256"], "prior source identity for date gate")
        old = pd.to_datetime(pd.read_csv(path).date).to_numpy(dtype="datetime64[D]")
        overlap = int(len(np.intersect1d(dates[b:], old)))
        core.check(overlap == 0, "official new tail and every old V3 source date disjoint")
        old_checks.append({"market": receipt["market"], "tail_date_intersection": overlap,
                           "source_sha256": receipt["source_sha256"]})
    tokens = [np.r_[0., np.diff(p)]]
    energy_checks = []
    core.check(len(manifest["sources"]) == 4 and len(manifest["energy_alignment"]) == 3,
               "three fixed energy inputs")
    for j, name in enumerate(["brent", "wti", "henryhub"]):
        source = manifest["sources"][j + 1]
        path = core.REPO / "data/parsed" / f"{name}_eia_daily.csv"
        core.check(core.sha(path) == source["sha256"], "energy source identity")
        ef = pd.read_csv(path).sort_values("date")
        missing = int(ef.spot_price.isna().sum())
        ef = ef.dropna(subset=["spot_price"])
        ed = pd.to_datetime(ef.date).to_numpy(dtype="datetime64[D]")
        values = ef.spot_price.to_numpy(float)
        ix = np.searchsorted(ed, dates - np.timedelta64(14, "D"), side="right") - 1
        core.check((ix >= 0).all() and np.isfinite(values[ix]).all(), "backward EIA availability")
        age = (dates - ed[ix]).astype(int)
        core.check((age >= 14).all(), "14calendar-day publication buffer")
        alignment = manifest["energy_alignment"][j]
        core.check(alignment["series"] == name and
                   [int(age.min()), int(age.max()), missing] ==
                   [alignment["minimum_age_days"], alignment["maximum_age_days"],
                    alignment["source_missing_price_rows_excluded_before_backward_asof"]], "energy metadata reconstruction")
        tokens.append(np.r_[0., np.diff(values[ix])])
        energy_checks.append({"series": name, "path_resolved": str(path), "sha256": source["sha256"],
                              "minimum_age_days": int(age.min()), "maximum_age_days": int(age.max())})
    tokens = np.column_stack(tokens)
    logs = np.log1p(volume); context = np.zeros((n, 2), dtype="float32")
    for t in range(10, n):
        past = logs[max(0, t - 60):t]
        context[t] = [np.clip((logs[t] - np.median(past)) / max(float(np.std(past)), .25), -5, 5), 1]
    core.check(manifest["volume"]["valid_rows"] == n
               and manifest["volume"]["usable_context_rows"] == n - 10, "past-volume reference counts")
    origins = np.arange(64, n - 20)
    masks = {"train": origins + 20 < a, "val": (origins >= a) & (origins + 20 < b),
             "final": origins + 20 < b, "test": origins >= b}
    counts = {key: int(mask.sum()) for key, mask in masks.items()}
    core.check(counts == manifest["counts"] == plan["split"]["counts"], "mature label counts")
    for key, mask in masks.items():
        span = [int(origins[mask][0]), int(origins[mask][-1])]
        core.check(span == manifest["origin_ranges"][key] == plan["split"]["origin_ranges"][key],
                   "exact fold origin range")
    core.check(origins[masks["train"]][-1] + 20 < origins[masks["val"]][0]
               and origins[masks["val"]][-1] + 20 < origins[masks["test"]][0], "H20 label purge")
    y = np.column_stack([p[origins + h] - p[origins] for h in core.HORIZONS])
    scale = np.maximum(np.sqrt(np.mean(tokens[1:a] ** 2, axis=0)), 1e-7)
    variance = np.maximum(np.mean(y[masks["train"]] ** 2, axis=0), scale[0] ** 2 * 1e-6)
    core.close(scale, manifest["training_rms"], "fixed training-prefix RMS")
    core.close(variance, manifest["train_target_second_moment"], "fixed train-horizon loss normalization")
    check_dates = {"first_train_origin": dates[origins[masks["train"]][0]],
                   "last_train_label": dates[origins[masks["train"]][-1] + 20],
                   "first_validation_origin": dates[origins[masks["val"]][0]],
                   "last_validation_label": dates[origins[masks["val"]][-1] + 20],
                   "first_test_origin": dates[origins[masks["test"]][0]],
                   "last_test_origin": dates[origins[masks["test"]][-1]],
                   "last_test_label": dates[origins[masks["test"]][-1] + 20]}
    for key, value in check_dates.items():
        core.check(str(value) == manifest[key], "manifest date identity")
    core.check([str(check_dates["first_test_origin"]), str(check_dates["last_test_origin"])] ==
               plan["split"]["mature_test_origin_date_range"], "fixed new-tail origin dates")
    rw_val = float(np.mean(y[masks["val"]] ** 2 / variance))
    drift = []
    for window in [20, 60, 120, None]:
        preds = np.array([tokens[max(1, t - window + 1) if window else 1:t + 1, 0].mean() * core.HORIZONS
                          for t in origins[masks["val"]]])
        drift.append((float(np.mean((preds - y[masks["val"]]) ** 2 / variance)), str(window)))
    return {"observed": y[masks["test"]], "origins": origins[masks["test"]],
            "dates": dates[origins[masks["test"]]], "counts": counts,
            "validation_rw": rw_val, "drift_choice": min(drift),
            "source_path_resolved": str(data_file), "source_sha256": source_hash(data_file),
            "published_price_cent_gap": cents_gap, "price_minus_turnover_over_volume_max_abs": identity_gap,
            "training_rms": scale.tolist(), "train_target_second_moment": variance.tolist(),
            "old_date_gate": old_checks, "energy_reconstruction": energy_checks,
            "volume_context_float32_sha256": core.hashlib.sha256(context.tobytes()).hexdigest(),
            "masks_sha256": {key: core.hashlib.sha256(mask.tobytes()).hexdigest() for key, mask in masks.items()}}


def source_hash(path):
    return core.sha(path)


def inspect_extra_traces(rows, data, family):
    for row in rows:
        r = core.read_json(row["path"])
        core.check(r["family"] == family, "family on execution receipt")
        updates = r["epochs_executed"] * math.ceil(data["counts"][r["mode"]] / 128)
        positive = r["gradient_positive_steps"]
        norm = r["max_gradient_norm_before_clip"]
        change = r["optimization_parameter_change_l2_before_selection"]
        core.check(isinstance(positive, int) and 0 <= positive <= updates,
                   "logged gradient counts within actual update budget")
        core.check(np.isfinite(norm) and norm >= 0 and ((positive > 0) == (norm > 0))
                   and np.isfinite(change) and change >= 0, "gradient and parameter-change receipt consistency")
        if r["mode"] == "train":
            if family in ZERO_HEAD:
                core.close(r["initial_validation_loss"], data["validation_rw"], "zero-head initial RW loss")
            best, stale, stop = r["initial_validation_loss"], 0, None
            for epoch, entry in enumerate(r["history"], start=1):
                if entry["validation_loss"] < best - 1e-7:
                    best, stale = entry["validation_loss"], 0
                else:
                    stale += 1
                if epoch >= 15 and stale >= 12:
                    stop = epoch
                    break
            core.check(len(r["history"]) == (stop if stop is not None else 80), "permitted early-stop endpoint")
        else:
            for epoch, entry in enumerate(r["history"], start=1):
                core.check(entry["epoch"] == epoch and np.isfinite(entry["train_loss"])
                           and entry["validation_loss"] is None, "refit fixed-epoch trace")
        row.update({"training_rows": data["counts"][r["mode"]],
                    "optimizer_steps_from_epochs_and_batches": updates, "max_gradient_norm_before_clip": norm})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Completed official Tianjin V4 run directory")
    parser.add_argument("--data-file", type=Path,
                        default=Path("/tmp/carbon-next-market-data-20261019d/TIANJIN_OFFICIAL_MEAN.csv"))
    parser.add_argument("--plan", type=Path, default=STUDY / "audits/V4_STATISTICAL_AUDIT_PLAN.json")
    parser.add_argument("--ewt-module", type=Path, help="Optional relocated EWT module for byte verification")
    parser.add_argument("--out", type=Path, default=STUDY / "audits")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    synthetic = self_test()
    if args.self_test:
        args.out.mkdir(parents=True, exist_ok=True)
        result = {**synthetic, "script_sha256": core.sha(__file__),
                  "utility_script_sha256": core.sha(core.__file__), "frozen_plan_sha256": core.sha(args.plan)}
        core.check(result["frozen_plan_sha256"] == PLAN_SHA, "fixed V4 plan identity")
        (args.out / "V4_STATISTICS_SELF_TEST.json").write_text(core.json.dumps(result, indent=2) + "\n")
        print(core.json.dumps(result, indent=2))
        return
    if args.run is None:
        parser.error("--run is required unless --self-test is supplied")
    args.run = args.run.resolve(); args.data_file = args.data_file.resolve(); args.out = args.out.resolve()
    # No official data/forecast/score files are opened before this completion guard.
    complete_path = args.run / "EXECUTION_COMPLETE.json"
    core.check(complete_path.is_file(), "wait for complete official V4 output")
    complete = core.read_json(complete_path)
    core.check(complete["status"] == "completed" and complete["worker_jobs"] == 11
               and complete["summary_rows"] == 15 and complete["neural_tasks"] == 50
               and complete["selection_tasks"] == 20 and complete["refit_constructions"] == 30,
               "all planned jobs and outputs completed")
    for key in ["source_hashes_unchanged", "statistics_plan_hash_unchanged", "data_hash_unchanged",
                "ewtpy_module_hash_unchanged"]:
        core.check(complete[key], "completed identity check " + key)
    plan = core.read_json(args.plan)
    protocol = core.read_json(args.run / "PROTOCOL_BEFORE_RUN.json")
    core.check(core.sha(args.plan) == PLAN_SHA == protocol["statistics_plan_sha256"], "frozen V4 plan")
    core.check(protocol["fixed_primary_candidate"] == plan["candidate_primary"] == "allocation_readout"
               and protocol["prespecified_secondary_candidate"] == plan["candidate_secondary"] == "liquidity_full",
               "new-stage primary/secondary identities")
    core.check(protocol["market"] == MARKET and protocol["fold"] == FOLD
               and protocol["neural_families"] == plan["neural_families"] == NEURAL
               and protocol["classical_families"] == CLASSICAL, "fixed full family set")
    core.check([(c["id"], c["candidate"], c["comparator"]) for c in plan["five_contrasts_one_joint_family"]] == CONTRASTS,
               "unchanged joint five-comparison family")
    for name, digest in protocol["source_sha256"].items():
        core.check(core.sha(STUDY / "source" / name) == digest, "frozen runner/model/helper byte identity")
    ewt_path = args.ewt_module or Path(protocol["runtime"]["ewtpy_module_path"])
    ewt_verified = ewt_path.is_file()
    if ewt_verified:
        core.check(core.sha(ewt_path) == protocol["runtime"]["ewtpy_module_sha256"], "EWT module byte identity")
    elif args.ewt_module is not None:
        raise FileNotFoundError(ewt_path)
    cd = args.run / f"{MARKET}_{FOLD}"
    manifest = core.read_json(cd / "INPUT_MANIFEST.json")
    ewt_manifest = core.read_json(cd / "SWEWT_INPUT_MANIFEST.json")
    core.check(ewt_manifest["swewt_adaptation"]["original_official_manifest"] == manifest,
               "EWT exact original official manifest")
    core.check({k: v for k, v in ewt_manifest.items() if k != "swewt_adaptation"} == manifest,
               "EWT and ordinary top-level source/label/scale/mask metadata")
    data = reconstruct(manifest, plan, args.data_file)
    stored = core.read_json(args.run / "ALL_SUMMARIES.json")
    summaries = {(r["market"], r["year"], r["family"]): r for r in stored}
    core.check(len(summaries) == len(stored) == 15, "unique summary rows")
    core.PARAMETERS = dict(PARAMETERS)
    ledger, choices, identity, native = [], [], [], []
    validation, metrics, losses = {}, {}, {}
    for family in FAMILIES:
        fd = cd / family
        if family in NEURAL:
            begin = len(ledger)
            val, epoch = core.audit_receipts(fd, family, data, ledger)
            inspect_extra_traces(ledger[begin:], data, family)
            choice = core.read_json(fd / "CHOICE_BEFORE_TEST.json")
            choices.append({"family": family, "selected_epoch": epoch,
                            "selected_lr": choice["learning_rate"], "validation_loss": val})
            validation[family] = val
        else:
            r = core.read_json(fd / "SELECTION.json")
            validation[family] = float(r["validation_loss"])
            if family == "rw":
                core.close(validation[family], data["validation_rw"], "independent RW validation")
            elif family == "drift":
                core.close(validation[family], data["drift_choice"][0], "independent drift validation")
                core.check(r["window"] == data["drift_choice"][1], "selected drift window")
            elif family.startswith("ridge"):
                best = min(r["trials"])
                core.close(validation[family], best[0], "Ridge validation choice")
                core.check(r["alpha"] == best[1], "Ridge selected alpha")
            else:
                best = min(r["trials"])
                core.close(validation[family], best[0], "HGB validation choice")
                core.check([r["max_leaf_nodes"], r["l2_regularization"]] == best[1:], "HGB selected configuration")
        path = fd / "predictions.npz"
        with np.load(path, allow_pickle=False) as packed:
            for key, target in [("origins", data["origins"]), ("observed", data["observed"]),
                                ("horizons", core.HORIZONS), ("seeds", core.SEEDS if family in NEURAL else [-1])]:
                core.check(np.array_equal(packed[key], target), "NPZ exact " + key)
            pred = np.array(packed["predictions"], dtype=float)
        core.check(pred.shape == (3 if family in NEURAL else 1, 196, 3), "all predictions same mature horizons")
        metric, loss = core.score(pred, data["observed"])
        if family == "rw" or (family in ZERO_HEAD and epoch == 0):
            core.check(np.all(pred == 0), "RW or zero-head epoch0 exact zero changes")
        for saved in [core.read_json(fd / "SUMMARY.json"), summaries[(MARKET, FOLD, family)]]:
            core.check(saved["n_origins"] == 196, "summary origin count")
            for key in SUMMARY_KEYS:
                core.close(metric[key], saved[key], "independent NPZ score reconciliation")
            if family in NEURAL:
                core.check(saved["parameters"] == PARAMETERS[family] and saved["selected_epoch"] == epoch
                           and saved["selected_lr"] == choice["learning_rate"], "summary model choice")
        metrics[family], losses[family] = metric, loss
        identity.append({"family": family, "path": str(path), "npz_sha256": core.sha(path), "origins": 196})
        for j, h in enumerate(core.HORIZONS):
            native.append({"market": MARKET, "fold": FOLD, "family": family, "horizon": int(h),
                           "n_origins": 196, "mean_seed_mse_over_rw": float(np.mean(metric["seed_mse_over_rw_by_horizon"], axis=0)[j]),
                           "mean_seed_mse": metric["native_mean_seed_mse_by_horizon"][j],
                           "mean_seed_rmse": metric["native_mean_seed_rmse_by_horizon"][j],
                           "rmse_of_mean_seed_mse": metric["native_rmse_of_mean_seed_mse_by_horizon"][j],
                           "mean_seed_mae": metric["native_mean_seed_mae_by_horizon"][j],
                           "ensemble_mse": metric["native_ensemble_mse_by_horizon"][j]})
    core.check(len(ledger) == 50 and len(identity) == 15 and len(choices) == 10, "all training receipts and NPZ inspected")
    strong = min(core.POOL, key=lambda f: (validation[f], core.POOL.index(f)))
    metrics["validation_selected_strong"] = metrics[strong]
    rng = np.random.default_rng(20261019)
    bootstrap = {"20": intervals(data["observed"], losses, 20, rng),
                 "40": intervals(data["observed"], losses, 40, rng)}
    effects = {}
    for name, candidate, comparator in CONTRASTS:
        pm = metrics[candidate]["mean_seed_mse_over_rw"]; cm = metrics[comparator]["mean_seed_mse_over_rw"]
        per_h = np.mean(metrics[comparator]["seed_mse_over_rw_by_horizon"], axis=0) - np.mean(metrics[candidate]["seed_mse_over_rw_by_horizon"], axis=0)
        effects[name] = {"candidate": candidate, "comparator": comparator,
                         "improvement_rw_units": cm - pm, "relative_risk_reduction": 1 - pm / cm,
                         "improvement_by_horizon_rw_units": per_h.tolist(),
                         "block20": bootstrap["20"][name], "block40": bootstrap["40"][name]}
    refits = [r for r in ledger if r["mode"] == "final"]
    selections = [r for r in ledger if r["mode"] == "train"]
    counts = {"neural_tasks": len(ledger), "selection_tasks": len(selections), "refit_tasks": len(refits),
              "prediction_npz_files": len(identity), "test_origins": 196, "origin_horizon_targets": 588,
              "tasks_with_positive_gradient_steps": sum(r["gradient_positive_steps"] > 0 for r in ledger),
              "selection_tasks_with_positive_gradient_steps": sum(r["gradient_positive_steps"] > 0 for r in selections),
              "refit_tasks_with_positive_gradient_steps": sum(r["gradient_positive_steps"] > 0 for r in refits),
              "zero_epoch_refits": sum(r["epochs_executed"] == 0 for r in refits),
              "selected_epoch_zero_families": sum(r["selected_epoch"] == 0 for r in choices),
              "total_executed_epochs": sum(r["epochs_executed"] for r in ledger),
              "total_logged_positive_gradient_steps": sum(r["gradient_positive_steps"] for r in ledger),
              "optimizer_steps_from_epochs_and_batches": sum(r["optimizer_steps_from_epochs_and_batches"] for r in ledger),
              "auditor_model_fits": 0, "auditor_checkpoint_unpickles": 0}
    core.check(counts["selection_tasks_with_positive_gradient_steps"] == complete["selection_tasks_with_positive_steps"]
               and counts["refit_tasks_with_positive_gradient_steps"] == complete["refit_tasks_with_positive_steps"],
               "independent ledger counts and completion receipt")
    scientific = {"primary_candidate_fixed": "allocation_readout", "secondary_candidate_fixed": "liquidity_full",
                  "single_official_market_later_period_only": True,
                  "primary_block20_99_intervals_exclude_zero_in_positive_direction": {
                      name: bool(e["block20"]["ci99_five_contrast_family"] is not None
                                 and e["block20"]["ci99_five_contrast_family"][0] > 0) for name, e in effects.items()},
                  "note": "Per-contrast numerical interval indicators, not a publication rule or evidence that every market is confirmed."}
    data_identity = {key: value for key, value in data.items() if key not in ["observed", "origins", "dates"]}
    result = {"title": "官方天津 V4：独立统计审查", "status": "PASS_INTEGRITY",
              "created_utc": datetime.now(timezone.utc).isoformat(), "stage": protocol["stage"],
              "design_timing": plan["design_timing"], "script_sha256": core.sha(__file__),
              "utility_script_sha256": core.sha(core.__file__), "frozen_plan_sha256": PLAN_SHA,
              "source_sha256": protocol["source_sha256"], "protocol_sha256": core.sha(args.run / "PROTOCOL_BEFORE_RUN.json"),
              "completion_sha256": core.sha(complete_path),
              "resolved_arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "runtime_ewt_module": {"path": str(ewt_path), "current_bytes_verified": ewt_verified,
                                     "recorded_sha256": protocol["runtime"]["ewtpy_module_sha256"]},
              "self_test": synthetic, "data_reconstruction": data_identity, "execution_counts": counts,
              "single_market_metrics": metrics, "five_prespecified_effects": effects,
              "validation_selected_strong": {"family": strong, "pool_validation_losses": {f: validation[f] for f in core.POOL}},
              "selected_epochs": choices, "training_receipt_ledger": ledger, "npz_identity": identity,
              "scientific_assessment": scientific,
              "limitations": [
                  "Primary candidate selected after old V3/SWEWT outcomes, before this official new-tail forecast evaluation; old V3 identities unchanged.",
                  "One official Tianjin later period, not new-market population confirmation. Test dates became exposed after this run.",
                  "All official rounded prices retained. Current historical snapshot lacks certified daily vintages; no author ScienceDB monthly cleaning is newly imposed.",
                  "196 irregular observed origins, overlapping H20 labels, circular-block and percentile approximation;20 primary and40 sensitivity both retained.",
                  "Three primary and two secondary contrasts jointly comprise the five-family Bonferroni set; no seeds treated as independent observations and no horizon-specific tests added.",
                  "Condition on fitted models and recorded validation choices; does not retroactively correct every old development search.",
                  "Gradient counts reconstructed from hash-matched execution logs and optimizer-call bounds, not independent backward-pass replay.",
                  "Checkpoint bytes checked without unpickling; trained checkpoint validation predictions not replayed by this audit.",
                  "Useful fixed initialization differs from learned improvement; softplus and affine controls differ in optimization geometry as well as sign freedom.",
                  "No arbitrary1% gate and no publication-success guarantee."
              ]}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "V4_INDEPENDENT_STATISTICAL_AUDIT_ZH.json").write_text(core.json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (args.out / "V4_NATIVE_METRICS.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(native[0])); writer.writeheader(); writer.writerows(native)
    def ci(x):
        return "不可用（零/非有限RW分母抽样）" if x is None else f"[{100*x[0]:+.6f}, {100*x[1]:+.6f}]"
    lines = ["# 官方天津 V4：独立统计审查", "",
             "**完整性 PASS。** 主候选 allocation_readout、次候选 liquidity_full 保持本阶段事先指定身份。两者身份是在旧 V3/SWEWT 结果之后、这份官方较近时期尾段模型评价之前固定；旧 V3 结论和主次身份不变。本审查没有训练模型。", "",
             "## 单市场较近时期指标", "",
             "官方1077条记录保留其公布的两位小数成交均价；60/20/20行分界646/861，训练562、验证195、最终训练777、测试196个共同H20成熟原点。测试原点2023-04-19至2026-07-02，末标签2026-09-30。新尾段与旧V3三个市场所有来源日期均无交集。H1/H5/H20为观测记录步长。", "",
             "先平均种子平方误差、每步长除以同一目标RW MSE，再等权平均三个步长。集成先平均预测，仅作次要结果。以下越小越好。", "",
             "| 模型 | H1 MSE/RW | H5 MSE/RW | H20 MSE/RW | 主 mean-seed MSE/RW | 次要 ensemble MSE/RW |",
             "|---|---:|---:|---:|---:|---:|"]
    for family in FAMILIES + ["validation_selected_strong"]:
        m = metrics[family]; h = np.mean(m["seed_mse_over_rw_by_horizon"], axis=0)
        lines.append(f"| {family} | {h[0]:.9f} | {h[1]:.9f} | {h[2]:.9f} | {m['mean_seed_mse_over_rw']:.9f} | {m['ensemble_mse_over_rw']:.9f} |")
    lines += ["", "## 五项事先指定比较", "",
              "改善为对照−候选，以RW归一化MSE百分点呈现，正数有利于候选。三项主readout比较与两项次full比较共同构成一个家族；每项99%分位区间给出五项约同时95%覆盖，边际95%另列。", "",
              "| 比较 | 改善RW百分点 | 相对对照降低率 | 20块95% | 20块99%（五项） | 40块95% | 40块99%（敏感性） |",
              "|---|---:|---:|---|---|---|---|"]
    for name, _, _ in CONTRASTS:
        e = effects[name]
        lines.append(f"| {name} | {100*e['improvement_rw_units']:+.6f} | {100*e['relative_risk_reduction']:+.6f}% | {ci(e['block20']['ci95_marginal'])} | {ci(e['block20']['ci99_five_contrast_family'])} | {ci(e['block40']['ci95_marginal'])} | {ci(e['block40']['ci99_five_contrast_family'])} |")
    lines += ["",
              "按观测原点循环移动块，20为主、40为预指定敏感性，各50000次；随机数default_rng(20261019)，顺序20再40。全部模型、步长和RW分母同步配对，抽样内重算比率。种子先折叠为逐原点损失，没有用作独立样本。若出现未定义RW比率抽样，该块长区间标不可用，不静默删抽样。", "",
              "196个原点约容纳9个完整不重叠20原点块、4个40原点块；这不是正式有效样本量。所有区间均依赖短且不规则交易记录序列上的近似，条件于已拟合模型与选择结果；不能由单市场推称所有碳市场已确认。40结果不能替代不利的20主结果。", "",
              "## 验证选择与执行核数", "",
              "| 模型 | LR | epoch | 验证损失 |", "|---|---:|---:|---:|"]
    for row in choices:
        lines.append(f"| {row['family']} | {row['selected_lr']} | {row['selected_epoch']} | {row['validation_loss']:.9f} |")
    lines += ["",
              f"50项神经任务=20次选择+30次重训；正梯度任务{counts['tasks_with_positive_gradient_steps']}项，最终重训中{counts['refit_tasks_with_positive_gradient_steps']}项；{counts['selected_epoch_zero_families']}个模型选择epoch0，对应{counts['zero_epoch_refits']}份零epoch重训。共记录{counts['total_executed_epochs']}个实际epoch、{counts['total_logged_positive_gradient_steps']}个正梯度步骤；按实际batch/epoch核算{counts['optimizer_steps_from_epochs_and_batches']}次优化器调用。",
              "",
              "完整验证trace和LR/epoch选择已核对，全部checkpoint字节SHA通过；梯度次数是哈希匹配执行收据的核数和一致性检查，不是本审查重新反向传播。原始15份NPZ的目标、原点、步长、种子精确匹配源价格重建，所有SUMMARY均复算。主计划、原/新模型源码及现存EWT模块字节一致。", "",
              "公开历史快照不是认证的逐日vintage。当前均价与成交量假定在原点报告后已知；EIA沿用14日延后，成交量参考仅使用此前60条记录。固定初始化的预测能力不能自动说成学习收益；约束和无约束映射的局部优化尺度差别继续保留。", "",
              "没有任意1%门槛。区间跨零表示在当前样本与方案下仍不确定，不能直接证明没有价值；对constant、SWEWT强结构及RW的效应应一起呈现。数值结果和所有原单位指标见同名JSON与V4_NATIVE_METRICS.csv。", ""]
    (args.out / "V4_INDEPENDENT_STATISTICAL_AUDIT_ZH.md").write_text("\n".join(lines))
    print(core.json.dumps({"status": result["status"], "counts": counts,
                           "scores": {f: m["mean_seed_mse_over_rw"] for f, m in metrics.items()},
                           "effects": effects, "choices": choices}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
