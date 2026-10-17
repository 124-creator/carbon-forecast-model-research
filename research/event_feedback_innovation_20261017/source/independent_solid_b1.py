"""Independent SOLID B1 selection and prediction-to-metric verification.

Only NumPy and the standard library are used. The sole imported study helper is
the already-frozen independent D1 metric implementation, never a model/runner.
The check command reads no market outcomes. The run command requires all 60 B1
executed streams and 20 equivalent derived predictions to be complete, verifies
every forecast, then selects each family using
2022 only. All output is aggregate evidence; EEX confirmation is never opened.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import datetime as dt
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

PROTOCOL_SHA256 = "b44357f30ce5c7f50f2023ebb0fd3fcc17e1598038724738b0b64b56c7ca76fd"
D1_PROTOCOL_SHA256 = "404c14867dcc5b43f456075d6b9edd4a5cd75cd99ad943a034e8da7172ef8a70"
D1_METRICS_SHA256 = "85612d7365034ae85b34cbbf94e04b1bb2f5e2a649980f52c579604d295428e8"
METRIC_HELPER_SHA256 = "22134f7e74467f7f1af8191351ca6a0044d4d52c8d9400693e6347d3362ce261"
FAMILIES = ("solid_distance", "solid_recency")
BASELINES = ("static_gru", "rw")
YEARS = (2022, 2023)
HORIZONS = (1, 5, 20)
HERE = Path(__file__).resolve().parent


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


if sha((HERE / "independent_metric_audit.py").read_bytes()) != METRIC_HELPER_SHA256:
    raise ValueError("Frozen independent metric helper changed")
sys.path.insert(0, str(HERE))
from independent_metric_audit import (  # noqa: E402
    approved_market, compare_table, metric_rows, prediction_digest, table_rows,
    synthetic as metric_synthetic,
)


def load_protocol(path):
    blob = Path(path).read_bytes()
    if sha(blob) != PROTOCOL_SHA256:
        raise ValueError("SOLID B1 protocol identity differs from the frozen plan")
    cfg = json.loads(blob)
    if (cfg["phase"] != "SOLID_B1" or cfg["timing_only"] or cfg["confirmation_access"]
            or tuple(cfg["scored_years"]) != YEARS or cfg["neighbors"] != 10):
        raise ValueError("Unexpected phase or information boundary")
    rule = cfg["selection_rule"]
    if (rule["selection_year"] != 2022 or rule["evaluation_year"] != 2023
            or tuple(rule["families"]) != FAMILIES):
        raise ValueError("The only allowed selection is within-family 2022 selection")
    jobs = {(e["market"], int(e["seed"])) for e in cfg["jobs"]}
    names = {m for m, _ in jobs}
    if len(cfg["jobs"]) != 10 or len(names) != 5 or jobs != {(m, s) for m in names for s in (17, 29)}:
        raise ValueError("Unexpected fixed series/seed job grid")
    wanted = {(r, c, lr) for r in (False, True) for c in (128, 512) for lr in (0.001, 0.01)}
    actual = {(s["remove_distance"], s["context_samples"], s["learning_rate"]) for s in cfg["configs"]}
    if len(cfg["configs"]) != 8 or actual != wanted or len({s["id"] for s in cfg["configs"]}) != 8:
        raise ValueError("Unexpected fixed eight-configuration grid")
    if (cfg["counts"]["streams"] != 60 or cfg["counts"]["metric_cells"] != 480
            or cfg["counts"]["prediction_series"] != 80 or cfg["counts"]["equivalent_derived_prediction_series"] != 20):
        raise ValueError("Unexpected formal run count")
    aliases = {s["id"]: s["equivalent_prediction_from"] for s in cfg["configs"] if "equivalent_prediction_from" in s}
    if aliases != {"recency_c512_lr0p001": "recency_c128_lr0p001", "recency_c512_lr0p01": "recency_c128_lr0p01"}:
        raise ValueError("Unexpected predeclared recency equivalence")
    return cfg


def registry(cfg):
    result = {s["id"]: {**s, "family": FAMILIES[int(s["remove_distance"])]} for s in cfg["configs"]}
    result.update({c: {"family": c} for c in BASELINES})
    return result


def rational(value):
    if not math.isfinite(float(value)):
        raise ValueError("Nonfinite metric value")
    return Fraction(str(value))


def average(values):
    return sum(values, Fraction(0)) / len(values)


def validate_cube(rows, cfg):
    configs = registry(cfg)
    jobs = {(e["market"], int(e["seed"])) for e in cfg["jobs"]}
    expected = {(c, y, m, s, h) for c in configs for y in YEARS for m, s in jobs for h in HORIZONS}
    cells, supports, denominators = {}, {}, {}
    for row in rows:
        c, y, m, s, h = row["config_id"], int(row["year"]), row["market"], int(row["seed"]), int(row["horizon"])
        key = c, y, m, s, h
        if key not in expected or key in cells or row["family"] != configs[c]["family"]:
            raise ValueError("Missing/duplicate/unplanned or mislabelled metric cell")
        count = int(row["origins"])
        support_key = y, m, s
        if count <= 0 or (support_key in supports and supports[support_key] != count):
            raise ValueError("Different or empty full-H20 support")
        supports[support_key] = count
        values = {f: rational(row[f]) for f in ("mse", "rw_mse", "relative_mse", "mae", "directional_accuracy")}
        if (any(values[f] < 0 for f in ("mse", "relative_mse", "mae")) or values["rw_mse"] <= 0
                or not 0 <= values["directional_accuracy"] <= 1):
            raise ValueError("Invalid metric or RW denominator")
        if not math.isclose(float(values["mse"] / values["rw_mse"]), float(values["relative_mse"]), rel_tol=5e-12, abs_tol=1e-12):
            raise ValueError("MSE / RW MSE differs from reported relative MSE")
        denominator_key = y, m, h
        if denominator_key in denominators and not math.isclose(
                float(denominators[denominator_key]), float(values["rw_mse"]), rel_tol=5e-12, abs_tol=1e-12):
            raise ValueError("SOLID and D1 do not share the identical RW denominator")
        denominators[denominator_key] = values["rw_mse"]
        if c == "rw" and values["relative_mse"] != 1:
            raise ValueError("RW relative MSE must equal one")
        cells[key] = values["relative_mse"]
    if set(cells) != expected:
        raise ValueError("Incomplete balanced SOLID + fixed-baseline cube")
    for year in YEARS:
        for market in {m for m, _ in jobs}:
            if supports[year, market, 17] != supports[year, market, 29]:
                raise ValueError("Different origin supports across seeds")
    for spec in cfg["configs"]:
        if "equivalent_prediction_from" in spec:
            for year in YEARS:
                for market, seed in jobs:
                    for horizon in HORIZONS:
                        if cells[spec["id"], year, market, seed, horizon] != cells[spec["equivalent_prediction_from"], year, market, seed, horizon]:
                            raise ValueError("The predeclared equivalent recency configurations differ")
    return cells, supports


def summarize(rows, cfg):
    cells, supports = validate_cube(rows, cfg)
    configs = registry(cfg)
    names = list(dict.fromkeys(e["market"] for e in cfg["jobs"]))
    score, series, horizon = {}, {}, {}
    for c in configs:
        for y in YEARS:
            for m in names:
                for h in HORIZONS:
                    horizon[c, y, m, h] = average([cells[c, y, m, s, h] for s in (17, 29)])
                series[c, y, m] = average([horizon[c, y, m, h] for h in HORIZONS])
            score[c, y] = average([series[c, y, m] for m in names])
    selected = {f: min((c for c, meta in configs.items() if meta["family"] == f),
                       key=lambda c: (score[c, 2022], c)) for f in FAMILIES}
    family_rows = []
    for family, c in selected.items():
        family_rows.append({"family": family, "selected_config_2022": c,
            "selection_score_2022": float(score[c, 2022]), "unchanged_config_score_2023": float(score[c, 2023]),
            "static_gru_score_2023": float(score["static_gru", 2023]), "rw_score_2023": float(score["rw", 2023]),
            "relative_improvement_vs_static_gru_2023": float(1 - score[c, 2023] / score["static_gru", 2023])
                if score["static_gru", 2023] else None,
            "relative_improvement_vs_rw_2023": float(1 - score[c, 2023]),
            "series_wins_vs_static_gru_2023": sum(series[c, 2023, m] < series["static_gru", 2023, m] for m in names),
            "series_wins_vs_rw_2023": sum(series[c, 2023, m] < 1 for m in names)})
    overall = [{"config_id": c, "family": meta["family"], "year": y,
                "equal_series_seed_horizon_score": float(score[c, y]), "cells": 30,
                "selected_in_family_on_2022": c in selected.values()}
               for c, meta in configs.items() for y in YEARS]
    by_series = [{"config_id": c, "family": meta["family"], "year": y, "market": m,
                  "equal_seed_horizon_score": float(series[c, y, m]), "cells": 6,
                  "origins_per_seed": supports[y, m, 17]}
                 for c, meta in configs.items() for y in YEARS for m in names]
    by_horizon = [{"config_id": c, "family": meta["family"], "year": y, "market": m,
                   "horizon": h, "mean_seed_relative_mse": float(horizon[c, y, m, h]),
                   "origins_per_seed": supports[y, m, 17]}
                  for c, meta in configs.items() for y in YEARS for m in names for h in HORIZONS]
    matched, matched_series = [], []
    for c, spec in configs.items():
        if spec["family"] != "solid_distance":
            continue
        counterparts = [other for other, v in configs.items() if v["family"] == "solid_recency"
                        and v["context_samples"] == spec["context_samples"] and v["learning_rate"] == spec["learning_rate"]]
        if len(counterparts) != 1:
            raise ValueError("Missing unique context/LR matched recency control")
        other = counterparts[0]
        for y in YEARS:
            common = {"distance_config": c, "recency_config": other, "context_samples": spec["context_samples"],
                      "learning_rate": spec["learning_rate"], "year": y}
            matched.append({**common, "distance_score": float(score[c, y]), "recency_score": float(score[other, y]),
                            "recency_minus_distance": float(score[other, y] - score[c, y])})
            for m in names:
                matched_series.append({**common, "market": m, "distance_score": float(series[c, y, m]),
                                       "recency_score": float(series[other, y, m]),
                                       "recency_minus_distance": float(series[other, y, m] - series[c, y, m])})
    summary = {"selection_year": 2022, "evaluation_year": 2023, "selected": selected,
        "selected_family_evaluations": family_rows, "SOLID_metric_cells": 480, "fixed_D1_baseline_cells": 120,
        "configurations": 8, "executed_streams": 60, "derived_equivalent_prediction_series": 20,
        "saved_B1_prediction_series": 80, "baseline_configurations": 2, "selection_arithmetic":
        "Exact rational mean of finite CSV decimal relative-MSE values; exact ties by ascending configuration id.",
        "averaging": "Ratio per series/year/seed/horizon, then equal weights; no price-weighted pooling or seed forecast ensemble.",
        "context_lr_matched_pairs": 4, "confirmation_opening_authorized": False, "RRC_stop_reversed": False,
        "new_algorithm_claim": False, "publication_readiness_established": False,
        "interpretation": "Post-RRC exploratory suitability of an existing published method on development years already used for research. No new gate, significance test or confirmatory claim.",
        "dependence": "Five series cover four related market systems; EEX and EU_REPORTED share EU ETS exposure. Seeds/horizons are not independent markets."}
    return summary, {"FAMILY_SELECTION_2022.csv": family_rows, "CONFIG_YEAR_SCORES.csv": overall,
                     "CONFIG_YEAR_SERIES_SCORES.csv": by_series, "CONFIG_YEAR_SERIES_HORIZON_SCORES.csv": by_horizon,
                     "MATCHED_DISTANCE_RECENCY.csv": matched, "MATCHED_DISTANCE_RECENCY_SERIES.csv": matched_series}


def query_origins(dates):
    """Enumerate only origins with all 20 targets inside the same scored year."""
    values = []
    for t, date in enumerate(dates):
        if t + 20 >= len(dates):
            continue
        for year in YEARS:
            if f"{year}-01-01" <= date < f"{year+1}-01-01" and dates[t+20] < f"{year+1}-01-01":
                values.append(t)
    return np.asarray(values, dtype=np.int64)


def require_complete(record, cfg):
    if (record["protocol_sha256"] != PROTOCOL_SHA256 or record["jobs"] != 10 or record["streams"] != 60
            or record["prediction_series"] != 80 or record["equivalent_derived_series"] != 20
            or record["metrics"] != 480 or record["failures"] or record["timing_only"]
            or record["confirmation_data_read"] or record["new_backbone_pretraining_fits"] != 0
            or record["query_local_adaptations"] != cfg["counts"]["query_local_head_adaptations"]):
        raise ValueError("B1 is incomplete, failed, or violates the frozen execution scope")


def read_forecast(path, expected_origins, expected_digest):
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"origins", "predictions"}:
            raise ValueError("Unexpected forecast NPZ schema")
        origins, predictions = payload["origins"], payload["predictions"]
    if (origins.dtype.kind not in "iu" or not np.array_equal(origins, expected_origins)
            or predictions.dtype != np.dtype("float64") or predictions.shape != (len(origins), 20)
            or not np.isfinite(predictions).all() or prediction_digest(predictions) != expected_digest):
        raise ValueError("Forecast origins, dtype, shape, finiteness or digest differs")
    return origins, predictions


def verify_and_select(cfg, protocol_path, root):
    freeze_path = root / "provenance/SOLID_B1_INDEPENDENT_FREEZE.json"
    frozen = json.loads(freeze_path.read_bytes())
    if (frozen["script_sha256"] != sha(Path(__file__).read_bytes()) or frozen["protocol_sha256"] != PROTOCOL_SHA256
            or frozen["metric_helper_sha256"] != METRIC_HELPER_SHA256):
        raise ValueError("Independent audit was not frozen at these source identities")
    for rel, expected in cfg["source_sha256"].items():
        if sha((root / rel).read_bytes()) != expected:
            raise ValueError("A frozen experimental source changed")
    output = root / "results/SOLID_B1"
    complete_path = output / "COMPLETE.json"
    complete = json.loads(complete_path.read_bytes())
    require_complete(complete, cfg)
    if (output / "FAILURES.json").exists():
        raise ValueError("A failure record exists; do not silently treat a retry as this run")
    d1_protocol = root / "provenance/D1_PROTOCOL.json"
    d1_table = root / "results/D1/ALL_METRICS.csv"
    if sha(d1_protocol.read_bytes()) != D1_PROTOCOL_SHA256 or sha(d1_table.read_bytes()) != D1_METRICS_SHA256:
        raise ValueError("Frozen D1 comparator identity changed")
    # Establish completion of every B1 job before opening market/forecast arrays.
    jobs = []
    for entry in cfg["jobs"]:
        local = root / "local_only/SOLID_B1" / f"{entry['market']}_s{entry['seed']}"
        record = json.loads((local / "COMPLETE.json").read_bytes())
        started = json.loads((local / "STARTED.json").read_bytes())
        public = json.loads((output / f"{entry['market']}_s{entry['seed']}_SUMMARY.json").read_bytes())
        if (record != public or record["status"] != "complete" or started["protocol_sha256"] != PROTOCOL_SHA256
                or record["market"] != entry["market"] or record["seed"] != entry["seed"] or record["streams"] != 6
                or record["prediction_series"] != 8 or record["equivalent_derived_series"] != 2
                or record["new_backbone_pretraining_fits"] != 0 or record["timing_only"]
                or not record["forecast_metrics_computed"] or record["confirmation_data_read"]
                or record["data_sha256"] != entry["data_sha256"]
                or record["initial_teacher_state_sha256"] != entry["teacher_initial_state_sha256"]):
            raise ValueError("Job receipt/scope or initial state mismatch")
        runs = {v["config"]["id"]: v for v in record["runs"]}
        if len(record["runs"]) != 8 or set(runs) != {s["id"] for s in cfg["configs"]}:
            raise ValueError("A planned B1 stream is missing or duplicated")
        jobs.append((entry, local, record, runs))
    cache, all_cells, baseline_cells, forecasts, checks, queries, backwards = {}, [], [], [], 0, 0, 0
    derived_checks, prediction_queries = 0, 0
    for entry, local, record, runs in jobs:
        market, seed = entry["market"], entry["seed"]
        if market not in cache:
            path = Path(entry["data_path"])
            if "development" not in path.name.lower() or any("sealed" in p.lower() for p in path.parts):
                raise ValueError("Only the frozen development CSV is permitted")
            dates, prices = approved_market({"path": str(path), "sha256": entry["data_sha256"]}, "2024-01-01")
            if any(d >= "2024-01-01" for d in dates):
                raise ValueError("Post-2023 source boundary violated")
            cache[market] = dates, prices
        dates, prices = cache[market]
        origins_expected = query_origins(dates)
        count = len(origins_expected)
        if record["query_count_per_config"] != count or record["query_local_adaptations"] != count * 6:
            raise ValueError("Job query count differs from independent full-H20 support")
        cells = []
        recency_predictions = {}
        for spec in cfg["configs"]:
            config = spec["id"]
            run = runs[config]
            adapter = run["adapter"]
            derived_from = spec.get("equivalent_prediction_from")
            if (run["config"] != spec or run["query_count"] != count
                    or run["actual_author_stream_executed"] != (derived_from is None)
                    or run["equivalent_prediction_from"] != derived_from
                    or adapter["queries"] != count or adapter["neighbors"] != 10
                    or adapter["context_samples"] != spec["context_samples"]
                    or adapter["learning_rate"] != spec["learning_rate"] or adapter["remove_distance"] != spec["remove_distance"]
                    or adapter["initial_offline_state_sha256"] != entry["teacher_initial_state_sha256"]
                    or adapter["final_offline_state_sha256"] != entry["teacher_initial_state_sha256"]
                    or adapter["query_future_labels_supplied"]):
                raise ValueError("Published adapter receipt violates the frozen per-query update scope")
            if derived_from is None:
                if (adapter["status"] != "COMPLETE" or adapter["head_updates"] != count
                        or adapter["neighbor_backward_calls"] != count * 10 or not adapter["author_query_scoring_disabled"]):
                    raise ValueError("Executed author stream update count/scope differs")
            elif (adapter["status"] != "EXACT_RECENCY_EQUIVALENCE_REUSE" or adapter["executed"]
                    or adapter["derived_from"] != derived_from or adapter["head_updates"] != 0
                    or adapter["neighbor_backward_calls"] != 0):
                raise ValueError("Derived prediction was counted as an executed author stream")
            npz = local / f"forecast_{config}.npz"
            origins, predictions = read_forecast(npz, origins_expected, run["prediction_sha256"])
            if derived_from is None:
                if adapter["raw_prediction_sha256"] != run["prediction_sha256"]:
                    raise ValueError("Prediction identity differs across adapter and run receipts")
                if spec["remove_distance"]:
                    recency_predictions[config] = predictions.copy()
            else:
                if predictions.tobytes() != recency_predictions[derived_from].tobytes():
                    raise ValueError("Derived recency forecast is not bitwise identical")
                alias = json.loads((local / f"adaptations_{config}_ALIAS.json").read_bytes())
                if alias != {"exact_equivalence_source": f"adaptations_{derived_from}.npz", "new_head_updates": 0}:
                    raise ValueError("Derived adaptation alias differs from the explicit fixed source")
                derived_checks += 1
            recomputed = metric_rows(market, dates, prices, origins, predictions, YEARS)
            for row in recomputed:
                row.update(seed=seed, config_id=config, family=FAMILIES[int(spec["remove_distance"])])
            cells.extend(recomputed)
            forecasts.append({"phase": "SOLID_B1", "market": market, "seed": seed, "config_id": config,
                              "queries": count, "actual_author_stream_executed": derived_from is None,
                              "equivalent_prediction_from": derived_from, "npz_sha256": sha(npz.read_bytes()),
                              "prediction_digest": run["prediction_sha256"]})
            queries += count if derived_from is None else 0
            prediction_queries += count
            backwards += adapter["neighbor_backward_calls"]
        checks += compare_table(table_rows(local / "METRICS.csv"), cells)
        all_cells.extend(cells)
        # Independently reconstruct the two fixed D1 comparators on exactly the
        # same B1 query set. Their 2021 stream/tail is neither scored nor pooled.
        old_local = root / "local_only/D1" / f"{market}_s{seed}"
        old = json.loads((old_local / "COMPLETE.json").read_bytes())
        old_started = json.loads((old_local / "STARTED.json").read_bytes())
        if old["status"] != "complete" or old_started["protocol_sha256"] != D1_PROTOCOL_SHA256:
            raise ValueError("D1 comparator did not complete under its fixed protocol")
        old_runs = {r["config_id"]: r for r in old["runs"]}
        old_origins = np.arange(int(np.searchsorted(dates, "2021-01-01")), len(dates), dtype=int)
        positions = np.searchsorted(old_origins, origins_expected)
        if not np.array_equal(old_origins[positions], origins_expected):
            raise ValueError("B1 support absent from the D1 stream")
        for config in BASELINES:
            npz = old_local / f"forecast_{config}.npz"
            _, predictions = read_forecast(npz, old_origins, old_runs[config]["predictions_sha256"])
            recomputed = metric_rows(market, dates, prices, origins_expected, predictions[positions], YEARS)
            for row in recomputed:
                row.update(seed=seed, config_id=config, family=config)
            baseline_cells.extend(recomputed)
            forecasts.append({"phase": "D1_fixed_comparator", "market": market, "seed": seed, "config_id": config,
                              "B1_common_queries": count, "npz_sha256": sha(npz.read_bytes()),
                              "prediction_digest": old_runs[config]["predictions_sha256"]})
    b1_table = output / "ALL_METRICS.csv"
    observed_b1 = table_rows(b1_table)
    observed_d1 = [r for r in table_rows(d1_table) if r["config_id"] in BASELINES]
    checks += compare_table(observed_b1, all_cells)
    checks += compare_table(observed_d1, baseline_cells)
    if (len(all_cells) != 480 or len(baseline_cells) != 120
            or queries != cfg["counts"]["query_local_head_adaptations"]
            or backwards != cfg["counts"]["neighbor_backward_calls"]):
        raise ValueError("Observed complete count differs from the frozen formal run")
    summary, tables = summarize(observed_b1 + observed_d1, cfg)
    # Selection uses the emitted finite decimal scores, as frozen above. Verify
    # that independently recomputed values yield the same two identities too.
    recomputed_selection = summarize(all_cells + baseline_cells, cfg)[0]["selected"]
    if summary["selected"] != recomputed_selection:
        raise ValueError("Selection changed under independently recomputed metrics")
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    identities = {"created_utc": now, "script_sha256": sha(Path(__file__).read_bytes()),
        "metric_helper_sha256": METRIC_HELPER_SHA256, "protocol_sha256": PROTOCOL_SHA256,
        "D1_protocol_sha256": D1_PROTOCOL_SHA256, "B1_metric_table_sha256": sha(b1_table.read_bytes()),
        "D1_metric_table_sha256": D1_METRICS_SHA256, "B1_complete_sha256": sha(complete_path.read_bytes()),
        "independent_freeze_sha256": sha(freeze_path.read_bytes())}
    summary.update(identities)
    summary.update(market_fit=False, new_confirmation_data_read=False, training_code_imported=False)
    audit = {**identities, "passed": True, "B1_forecasts_recomputed": 80, "D1_comparator_forecasts_recomputed": 20,
        "B1_metric_cells_recomputed": 480, "D1_comparator_metric_cells_recomputed": 120,
        "reported_metric_scalar_and_label_comparisons": checks, "query_local_head_adaptations": queries,
        "actual_author_streams": 60, "bitwise_equivalence_checks": derived_checks,
        "derived_equivalent_prediction_series": 20, "saved_prediction_query_rows": prediction_queries,
        "neighbor_backward_calls": backwards, "market_files_read": 5, "market_fit": False,
        "training_code_imported": False, "new_confirmation_data_read": False,
        "selection_same_from_recomputed_metrics": True, "forecasts": forecasts,
        "scope": "Prediction-to-metric and support/receipt identity audit. Mathematical adaptation/backprop correctness is a separate independent audit; this script does not claim to verify training gradients."}
    return summary, tables, audit


def synthetic(cfg):
    generic = metric_synthetic()
    configs = registry(cfg)
    rows = []
    family_configs = {f: sorted(c for c, meta in configs.items() if meta["family"] == f) for f in FAMILIES}
    for c, meta in configs.items():
        for year in YEARS:
            value = "1.1"
            if c == "rw": value = "1"
            if c == "static_gru": value = "1.05"
            if meta["family"] in FAMILIES:
                order = family_configs[meta["family"]].index(c)
                if meta["family"] == "solid_recency":
                    order %= 2
                value = str(Fraction(6 + order, 10)) if year == 2022 else str(Fraction(9 - order * 2, 10))
                value = str(float(Fraction(value)))
            for i, job in enumerate(cfg["jobs"]):
                for h in HORIZONS:
                    rows.append({"market": job["market"], "seed": job["seed"], "config_id": c,
                        "family": meta["family"], "year": year, "horizon": h,
                        "origins": 17 + 5 * (i // 2), "mse": value, "rw_mse": "1", "relative_mse": value,
                        "mae": "0.2", "directional_accuracy": "0.5"})
    first, tables = summarize(rows, cfg)
    assert first["selected"] == {f: cs[0] for f, cs in family_configs.items()}
    assert all(v["unchanged_config_score_2023"] == 0.9 for v in first["selected_family_evaluations"])
    assert summarize(list(reversed(rows)), cfg) == (first, tables)
    checks = ["2022 independently selects both families despite tempting better 2023 alternatives",
              "row-order invariant selection and all-configuration tables"]
    weighted = [dict(r) for r in rows]
    names = list(dict.fromkeys(e["market"] for e in cfg["jobs"]))
    for row in weighted:
        if row["config_id"] != "rw":
            value = rational(row["relative_mse"]) + Fraction(names.index(row["market"]), 100)
            row["mse"] = row["relative_mse"] = str(float(value))
    weighted_summary = summarize(weighted, cfg)[0]
    assert all(v["unchanged_config_score_2023"] == 0.92 for v in weighted_summary["selected_family_evaluations"])
    pooled = sum((17 + 5*i) * (0.9 + i/100) for i in range(5)) / sum(17 + 5*i for i in range(5))
    assert not math.isclose(pooled, 0.92)
    checks.append("known unequal-series scores average equally despite unequal origin counts; pooled score rejected")
    tied = [dict(r) for r in rows]
    for row in tied:
        if row["year"] == 2022 and row["family"] in FAMILIES:
            row["mse"] = row["relative_mse"] = "0.6"
    assert summarize(tied, cfg)[0]["selected"] == first["selected"]
    checks.append("exact ties use ascending config id within family")
    assert len(tables["MATCHED_DISTANCE_RECENCY.csv"]) == 8
    assert len(tables["MATCHED_DISTANCE_RECENCY_SERIES.csv"]) == 40
    checks.append("all four context/LR-matched pairs preserved in both years")
    bad_variants = [rows[:-1], rows + [rows[0]]]
    for field, value in (("relative_mse", "nan"), ("origins", 0), ("family", "unknown"),
                         ("mse", "1.23"), ("rw_mse", "2"), ("directional_accuracy", "1.1")):
        bad = [dict(r) for r in rows]
        bad[0][field] = value
        bad_variants.append(bad)
    for bad in bad_variants:
        try:
            summarize(bad, cfg)
        except ValueError:
            continue
        raise AssertionError("Invalid/incomplete/comparator-inconsistent cube accepted")
    checks.append("missing, duplicate, nonfinite, support, family, ratio and directional errors refused")
    bad = [dict(r) for r in rows]
    row = next(r for r in bad if r["config_id"] == "recency_c512_lr0p001")
    row["mse"] = row["relative_mse"] = "0.61"
    try:
        summarize(bad, cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("Non-equivalent declared recency alias accepted")
    checks.append("predeclared recency equivalence cannot silently change selected scores")
    dates = np.asarray([f"2022-12-{d:02}" for d in range(1, 32)] + [f"2023-01-{d:02}" for d in range(1, 32)])
    assert np.array_equal(query_origins(dates), np.asarray(list(range(11)) + list(range(31, 42))))
    checks.append("independent enumeration excludes every cross-year and incomplete H20 origin")
    valid_complete = {"protocol_sha256": PROTOCOL_SHA256, "jobs": 10, "streams": 60, "metrics": 480,
                      "prediction_series": 80, "equivalent_derived_series": 20,
                      "failures": [], "timing_only": False, "confirmation_data_read": False,
                      "new_backbone_pretraining_fits": 0,
                      "query_local_adaptations": cfg["counts"]["query_local_head_adaptations"]}
    require_complete(valid_complete, cfg)
    for field, value in (("streams", 79), ("confirmation_data_read", True), ("timing_only", True), ("failures", ["synthetic"])):
        try:
            require_complete({**valid_complete, field: value}, cfg)
        except ValueError:
            continue
        raise AssertionError("Partial/failed/confirmation run accepted")
    checks.append("partial, failed, timing-only or confirmation-reading phase cannot be scored")
    return {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "passed": True,
        "script_sha256": sha(Path(__file__).read_bytes()), "protocol_sha256": PROTOCOL_SHA256,
        "metric_helper_sha256": METRIC_HELPER_SHA256, "new_checks": checks,
        "inherited_independent_metric_checks": generic["checks"], "synthetic_metric_cells": 600,
        "market_files_read": False, "B1_results_read": False, "market_fit": False, "confirmation_read": False,
        "training_code_imported": False}


def write_json(path, obj):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study-root", type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Independent output exists; refusing overwrite or hidden reselection")
    cfg = load_protocol(args.protocol)
    if args.action == "check":
        result = synthetic(cfg)
        write_json(args.output, result)
    else:
        result, tables, audit = verify_and_select(cfg, args.protocol, Path(args.study_root or cfg["study_root"]))
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "INDEPENDENT_SELECTION.json", result)
        write_json(args.output / "INDEPENDENT_METRIC_AUDIT.json", audit)
        for name, rows in tables.items():
            write_csv(args.output / name, rows)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
