"""Synthetic checks of independent D1/D2 selection and calendar uncertainty."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import tempfile

import numpy as np

from independent_results import (
    BASELINE_FAMILIES, NEURAL_FAMILIES, RIDGE, SHRINK, SCORE_INDEX,
    audit_stage, calendar_day_counts, d2_uncertainty, evaluate_d2, expected_jobs,
    mean_seed_squared_loss, paired_calendar_bootstrap, recompute_metrics,
    select_d1, sha, write_json,
)


def run_checks():
    checks = []
    comparisons = 0
    max_error = 0.0
    def truth(name, condition):
        nonlocal comparisons
        if not bool(condition):
            raise AssertionError(name)
        comparisons += 1
        checks.append({"name": name, "status": "PASS", "comparisons": 1})
    def close(name, actual, expected, atol=1e-11):
        nonlocal comparisons, max_error
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.shape != expected.shape:
            raise AssertionError(name + " shape")
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=1e-11, err_msg=name)
        error = float(np.max(np.abs(actual-expected))) if actual.size else 0.0
        max_error = max(max_error, error)
        comparisons += actual.size
        checks.append({"name": name, "status": "PASS",
                       "comparisons": actual.size, "max_abs_error": error})
    def rejects(name, function):
        try:
            function()
        except ValueError:
            truth(name, True)
        else:
            raise AssertionError(name + " did not reject")

    markets = ["EEX_SYNTHETIC", "A", "B", "C", "D"]
    protocol = {
        "markets": [{"name": m, "sha256": "synthetic-source-only"} for m in markets],
        "neural_families": list(NEURAL_FAMILIES),
        "primary_baseline_families": list(BASELINE_FAMILIES),
        "shrink_candidates": list(SHRINK), "ridge_alpha_candidates": list(RIDGE),
        "D1_seeds": [17], "D2_seeds": [17, 29, 43], "validation_size": 80,
    }
    def synthetic_rows(phase, d1=None):
        rows = []
        for job in expected_jobs(protocol, phase, d1):
            f, v = job["family"], job["value"]
            if phase == "D1":
                if f == "smp_full": risk = {0.0: 0.985, 0.1: 0.95}[v]
                elif f == "smp_iid_state": risk = {0.0: 0.94, 0.1: 0.97}[v]
                elif f == "smp_one_step": risk = {0.0: 0.93, 0.1: 0.96}[v]
                elif f == "gru_native": risk = {0.0: 0.94, 0.1: 0.98}[v]
                elif f in NEURAL_FAMILIES: risk = 0.98 + v/10
                elif f == "ridge": risk = 0.97 + list(RIDGE).index(v)/100
                elif f == "markov_mle": risk = 0.995
                elif f == "rw": risk = 1.0
                else: risk = 1.05
            else:
                if f == "smp_full": risk = 0.94
                elif f == "smp_iid_state": risk = {0.0: 0.92, 0.1: 0.96}[v]
                elif f == "smp_one_step": risk = {0.0: 0.91, 0.1: 0.97}[v]
                elif f == "gru_native": risk = {0.0: 0.97, 0.1: 0.90}[v]
                elif f in NEURAL_FAMILIES: risk = 0.99 + v/10
                elif f == "ridge": risk = 0.98 + list(RIDGE).index(v)/100
                elif f == "markov_mle": risk = 0.995
                elif f == "rw": risk = 1.0
                else: risk = 1.05
                if job["kind"] == "neural":
                    risk += {17: -0.01, 29: 0.0, 43: 0.01}[job["seed"]]
            rows.append({**job, "relative_mse": [risk]*20, "score": risk,
                         "selected_validation_score":
                         (5-list(RIDGE).index(v) if f == "ridge" else 1.0)})
        return rows

    d1_rows = synthetic_rows("D1")
    d1 = select_d1(d1_rows, protocol)
    truth("D1_exact_160_jobs_150_new_predictor_fits_10_unfitted_references",
          len(d1_rows) == 160
          and sum(r["kind"] not in ("rw", "project_reference") for r in d1_rows) == 150)
    truth("D1_selects_full_alpha_0p1", d1["selected"]["smp_full"]["value"] == 0.1)
    truth("D1_matches_full_alpha_not_other_families_best_alpha",
          d1["selected"]["smp_iid_state"]["value"] == 0
          and d1["advance_to_D2"])
    truth("D1_ridge_risk_choice_and_preyear_validation_choice_are_distinct",
          d1["selected"]["ridge"]["value"] == 0.01
          and all(x["value"] == 100
                  for x in d1["selected"]["ridge_val"]["pre_year_validation_choices"].values()))
    truth("D1_fixed_strongest_baseline", d1["best_baseline"]["family"] == "gru_native")
    tied = deepcopy(d1_rows)
    for row in tied:
        if row["family"] == "smp_full":
            row["relative_mse"] = [0.95]*20
        if row["family"] in ("gru_native", "gru_aux"):
            row["relative_mse"] = [0.94]*20
        if row["family"] == "ridge":
            row["selected_validation_score"] = 1.0
    tied_selection = select_d1(tied, protocol)
    truth("exact_family_score_tie_prefers_larger_value",
          tied_selection["selected"]["smp_full"]["value"] == 0.1
          and tied_selection["selected"]["gru_native"]["value"] == 0.1)
    truth("exact_baseline_tie_prefers_alphabetical_family",
          tied_selection["best_baseline"]["family"] == "gru_aux")
    truth("exact_validation_tie_prefers_larger_alpha",
          all(x["value"] == 100 for x in
              tied_selection["selected"]["ridge_val"]["pre_year_validation_choices"].values()))
    rejects("missing_D1_seed_or_series_rejected",
            lambda: select_d1(d1_rows[:-1], protocol))

    d2_rows = synthetic_rows("D2", d1)
    d2 = evaluate_d2(d2_rows, protocol, d1)
    truth("D2_passes_fixed_synthetic_point_gate", d2["point_gate_pass"])
    truth("D2_does_not_reselect_better_matching_alpha_baseline_on_2023",
          d2["selected"]["gru_native"]["value"] == 0
          and d2["matched_smp_alpha"]["gru_native"]["score"]
          < d2["selected"]["gru_native"]["score"])
    truth("D2_keeps_D1_best_baseline_identity_for_uncertainty",
          d2["d1_fixed_strongest_baseline_family"] == "gru_native")
    close("D2_average_seed_risk", d2["selected"]["smp_full"]["score"], 0.94)
    def changed_full(by_market=None, by_seed=None):
        rows = deepcopy(d2_rows)
        for row in rows:
            if row["family"] == "smp_full":
                risk = by_market[row["market"]] if by_market else by_seed[row["seed"]]
                row["relative_mse"] = [risk]*20
        return evaluate_d2(rows, protocol, d1)
    bad_eex = changed_full(by_market=dict(zip(markets, [1.01, .90, .90, .90, .90])))
    truth("EEX_gate_is_required_despite_good_overall_score",
          not bad_eex["gates"]["eex_below_rw"] and not bad_eex["point_gate_pass"])
    bad_breadth = changed_full(by_market=dict(zip(markets, [.8, .8, 1.02, 1.02, 1.02])))
    truth("three_series_breadth_gate_is_required",
          not bad_breadth["gates"]["at_least_three_series_below_rw"])
    bad_seeds = changed_full(by_seed={17: .8, 29: 1.0, 43: 1.0})
    truth("two_of_three_seed_risks_required",
          not bad_seeds["gates"]["at_least_two_seeds_below_rw"])
    no_bootstrap = d2_uncertainty(bad_seeds, {}, {})
    truth("point_gate_failure_returns_before_reading_arrays_or_bootstrapping",
          no_bootstrap["bootstrap_repetitions_executed"] == 0
          and no_bootstrap["status"] == "not_run_point_gate_failed")

    target = np.arange(1, 101, dtype=float).reshape(5, 20) / 10
    predicted = target * .25 + np.arange(20)/30
    metric = recompute_metrics(predicted, target)
    manual_mse = np.array([sum((predicted[i,h]-target[i,h])**2 for i in range(5))/5
                           for h in range(20)])
    manual_rw = np.array([sum(target[i,h]**2 for i in range(5))/5 for h in range(20)])
    close("all_20_metrics_against_scalar_loops", metric["mse"], manual_mse)
    close("all_20_RW_denominators_against_scalar_loops", metric["rw_mse"], manual_rw)
    close("score_is_equal_h1_h5_h20", metric["score"],
          np.mean((manual_mse/manual_rw)[[0,4,19]]))
    close("raw_price_unit_change_preserves_relative_risk",
          recompute_metrics(predicted*7, target*7)["relative_mse"], metric["relative_mse"])
    seed_predictions = np.stack([target+1, target-1])
    close("seed_losses_averaged_instead_of_ensemble_prediction",
          mean_seed_squared_loss(seed_predictions, target), np.ones((5,20)))
    truth("ensemble_counterexample_is_detectable",
          np.max(np.abs(np.mean(seed_predictions,axis=0) - target)) < 1e-12)

    # Calendar draw oracle with explicit prescribed starts, including wrap.
    class FixedStarts:
        def integers(self, low, high, size):
            return np.array([363, 100, 0])[:size]
    counts = calendar_day_counts(365, 150, FixedStarts())
    drawn = []
    for start in (363,100,0):
        for offset in range(150):
            drawn.append((start+offset) % 365)
    manual_counts = [drawn[:365].count(d) for d in range(365)]
    close("circular_blocks_wrap_and_truncate_to_natural_year", counts, manual_counts, atol=0)

    dates = np.datetime64("2023-01-01") + np.arange(365)
    # Identical ratios across all days imply exact point and bootstrap differences,
    # even when absolute RW losses vary sharply and some series have sparse days.
    series = {}
    for k, name in enumerate(markets):
        indices = np.arange(k, 365, k+1)
        rw = np.tile((1+indices**2/50)[:,None], (1,20)) * (k+1)
        series[name] = {"date": dates[indices], "loss_a": rw*.8,
                        "loss_b": rw*1.1, "rw_loss": rw}
    boot = paired_calendar_bootstrap(series, year=2023, block_days=28,
                                     repetitions=40, max_attempts=200, return_samples=True)
    close("each_replicate_recomputes_heterogeneous_RW_denominators",
          boot["differences"], np.full(40, -.3))
    close("percentile_interval_for_fixed_risk_ratio", boot["percentile_95_interval"], [-.3,-.3])
    truth("full_calendar_not_observation_span_and_40_valid_replicates",
          boot["calendar_days"] == 365 and boot["valid_repetitions"] == 40)
    leap = {name: {**item, "date": item["date"] + np.timedelta64(365,"D")}
            for name,item in series.items()}
    leap_boot = paired_calendar_bootstrap(leap, year=2024, block_days=14,
                                          repetitions=5, max_attempts=100)
    truth("leap_year_uses_366_calendar_days", leap_boot["calendar_days"] == 366)

    # Choose a synthetic date absent in the first fixed random draw, to guarantee
    # an invalid replicate without relying on a chance outcome in the test.
    first_counts = calendar_day_counts(365, 28, np.random.default_rng(314159))
    missing_day = int(np.flatnonzero(first_counts == 0)[0])
    sparse_rw = np.zeros((365,20))
    sparse_rw[missing_day] = 1
    invalid_series = {name: {"date": dates, "loss_a": sparse_rw*.8,
                             "loss_b": sparse_rw, "rw_loss": sparse_rw}
                      for name in markets}
    exhausted = paired_calendar_bootstrap(invalid_series, year=2023,
                                           repetitions=1, max_attempts=1)
    truth("zero_RW_denominator_is_counted_invalid_and_attempt_cap_is_honored",
          exhausted["invalid_repetitions"] == 1 and exhausted["attempts"] == 1
          and exhausted["status"] == "insufficient_valid_replicates"
          and exhausted["percentile_95_interval"] is None)
    one_origin = {name: {"date": dates[[missing_day]], "loss_a": np.full((1,20),.8),
                         "loss_b": np.ones((1,20)), "rw_loss": np.ones((1,20))}
                  for name in markets}
    empty_draw = paired_calendar_bootstrap(one_origin, year=2023,
                                           repetitions=1, max_attempts=1)
    truth("no_weighted_origin_is_counted_invalid",
          empty_draw["invalid_repetitions"] == 1)

    # Full file-schema fixture: 160 synthetic predictions, no price CSV or fits.
    with tempfile.TemporaryDirectory(prefix="smp_independent_synthetic_") as temp:
        root = Path(temp)
        protocol_path = root/"provenance/PROTOCOL.json"
        write_json(protocol_path, protocol)
        protocol_sha = sha(protocol_path)
        jobs = expected_jobs(protocol, "D1")
        kinds = {}
        for job in jobs:
            kinds[job["kind"]] = kinds.get(job["kind"],0)+1
        stage = root/"results/D1"
        write_json(stage/"STARTED.json", {
            "job_ids":[j["id"] for j in jobs], "total_jobs":len(jobs),
            "job_counts":kinds, "protocol_sha256":protocol_sha})
        write_json(stage/"COMPLETED.json", {
            "failed_jobs":0, "completed_jobs":len(jobs), "attempted_jobs":len(jobs),
            "jobs":[{"id":j["id"],"status":"completed"} for j in jobs],
            "actual_counts":kinds, "protocol_sha256":protocol_sha})
        for k,job in enumerate(jobs):
            n = 8
            all_prices = 10 + np.arange(n+20,dtype=float)/10
            base = all_prices[:n]
            future = np.stack([all_prices[h:h+n] for h in range(1,21)],axis=1)
            delta = future-base[:,None]
            pred = np.zeros_like(delta) if job["family"]=="rw" else delta*.1
            origin = np.arange(200,208)
            date = np.datetime64("2022-01-04")+np.arange(n)
            public = stage/"jobs"/job["id"]
            private = root/"local_only/D1"/job["id"]/"forecasts.npz"
            private.parent.mkdir(parents=True)
            np.savez_compressed(private,origin=origin,date=date,base=base,
                                scale=np.ones(n),state=np.full(n,6),
                                predicted_change=pred,target_change=delta)
            mse = [sum((pred[i,h]-delta[i,h])**2 for i in range(n))/n for h in range(20)]
            rw = [sum(delta[i,h]**2 for i in range(n))/n for h in range(20)]
            rel = [a/b for a,b in zip(mse,rw)]
            mae = [sum(abs(pred[i,h]-delta[i,h]) for i in range(n))/n for h in range(20)]
            record = {**job, "status":"completed","phase":"D1","n":n,
                      "mse":mse,"rw_mse":rw,"relative_mse":rel,"mae":mae,
                      "score":sum(rel[h] for h in (0,4,19))/3,"score_horizons":[1,5,20],
                      "selected_validation_score":1.0,
                      "private_forecast_sha256":sha(private),"protocol_sha256":protocol_sha}
            write_json(public/"METRICS.json",record)
            write_json(public/"STARTED.json",{
                "market":job["market"],"protocol_sha256":protocol_sha,
                "job":{k:v for k,v in job.items() if k!="market"}})
            if job["kind"] not in ("rw","project_reference"):
                write_json(public/"FIT.json",{
                    "eligible_for_selection":True,"selected_validation_score":1.0})
            write_json(public/"SUPPORT.json",{
                "market":job["market"],"year":2022,"source_sha256":"synthetic-source-only",
                "evaluation_n":n,"evaluation_first_origin":200,"evaluation_last_origin":207,
                "evaluation_last_label":227,"evaluation_first_date":str(date[0]),
                "evaluation_last_date":str(date[-1]),"train_first_origin":60,
                "train_last_origin":79,"train_n":20,"max_train_label":99,
                "first_validation_origin":100,"last_validation_origin":179,
                "validation_n":80,"max_validation_label":199,
                "statistics_price_prefix_end_exclusive":100,"source_rows":228,
                "current_state_from_full_observed_prefix":True,
                "future_context_frozen":True,"quote_imputation":False})
        fixture_rows, fixture_arrays, fixture_audit = audit_stage(
            root, protocol_path, "D1")
        truth("full_D1_synthetic_files_all_hashes_metrics_and_common_support",
              fixture_audit["status"]=="PASS" and fixture_audit["jobs"]==160
              and fixture_audit["horizon_metric_cells"]==3200)
        # Corrupt metadata only to verify fail-closed support identity; no source file read.
        job_path = stage/"jobs"/jobs[0]["id"]/"SUPPORT.json"
        broken = json.loads(job_path.read_text())
        broken["source_sha256"] = "wrong-source"
        job_path.write_text(json.dumps(broken))
        rejects("admitted_source_identity_mismatch_is_rejected",
                lambda:audit_stage(root, protocol_path, "D1"))

    return {
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "status":"PASS","checks":checks,"check_count":len(checks),
        "scalar_comparisons":int(comparisons),"max_abs_error":max_error,
        "python":platform.python_version(),"numpy":np.__version__,
        "real_market_file_reads":0,"confirmation_reads":0,"market_fits":0,
        "synthetic_forecast_files_checked":160,
        "fixture_metric_comparisons":fixture_audit["scalar_comparisons"],
        "source_sha256":sha(Path(__file__).with_name("independent_results.py")),
        "checker_sha256":sha(Path(__file__)),
        "scope":"Hand-built selection risks, scalar loss oracles, calendar fixtures and temporary synthetic NPZ/receipt files only.",
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    receipt=run_checks()
    if args.output:
        write_json(args.output,receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!="checks"},
                     ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()

