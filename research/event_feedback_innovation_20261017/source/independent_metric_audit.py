"""D1 score-audit draft: independent metrics, no training-code import or fit.

The check command uses only synthetic arrays. The run command requires an
explicit frozen protocol and a fully completed D1. It never searches for data,
opens sealed confirmation directories, or proposes a new model-selection grid.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

HORIZONS = (1, 5, 20)
FULL_HORIZON = 20
SCALARS = ("mse", "rw_mse", "relative_mse", "mae", "directional_accuracy")


def sha256(blob):
    return hashlib.sha256(blob).hexdigest()


def mean(values):
    return math.fsum(float(x) for x in values) / len(values)


def metric_rows(name, dates, prices, origins, predictions, years):
    dates = np.asarray(dates)
    prices = np.asarray(prices, dtype=np.float64)
    origins = np.asarray(origins)
    predictions = np.asarray(predictions, dtype=np.float64)
    if origins.ndim != 1 or origins.dtype.kind not in "iu" or np.any(np.diff(origins) <= 0):
        raise ValueError("Invalid ordered origin identifiers")
    if predictions.shape != (len(origins), FULL_HORIZON) or not np.isfinite(predictions).all():
        raise ValueError("Invalid full-H20 prediction array")
    if len(prices) != len(dates) or not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("Invalid source prices")
    if origins.size == 0 or origins[0] < 0 or origins[-1] >= len(prices):
        raise ValueError("Origin outside observed source")
    rows = []
    for year in years:
        start, end = f"{year}-01-01", f"{year+1}-01-01"
        valid = [j for j, origin in enumerate(origins)
                 if start <= dates[origin] < end and origin + FULL_HORIZON < len(prices)
                 and dates[origin + FULL_HORIZON] < end]
        if not valid:
            raise ValueError("No complete H20 origins for a planned year")
        selected = origins[valid]
        base = prices[selected]
        for horizon in HORIZONS:
            target = prices[selected + horizon]
            pred = predictions[valid, horizon - 1]
            error = [float(p) - float(y) for p, y in zip(pred, target)]
            reference_error = [float(b) - float(y) for b, y in zip(base, target)]
            mse = mean([e * e for e in error])
            rw_mse = mean([e * e for e in reference_error])
            if rw_mse <= 0:
                raise ValueError("Zero RW denominator; relative MSE undefined")
            directions = [((p > b) - (p < b)) == ((y > b) - (y < b))
                          for p, y, b in zip(map(float, pred), map(float, target), map(float, base))]
            rows.append({"market": name, "year": int(year), "horizon": horizon, "origins": len(selected),
                         "mse": mse, "rw_mse": rw_mse, "relative_mse": mse / rw_mse,
                         "mae": mean([abs(e) for e in error]), "directional_accuracy": mean(directions)})
    return rows


def config_families(cfg):
    result = {"rw": "rw", "static_gru": "static_gru",
              "static_teacher_student": "static_teacher_student", "online_gru": "online_gru"}
    for cal in cfg["calibrators"]:
        if not cal["old_target"]:
            key = "output_" + cal["id"]
            if key in result:
                raise ValueError("Duplicate planned config identifier")
            result[key] = "output_only"
    for spec in cfg["student_configs"]:
        if spec["id"] in result:
            raise ValueError("Duplicate planned config identifier")
        result[spec["id"]] = spec["family"]
    return result


def key(row):
    return (row["market"], int(row["seed"]), row["config_id"], int(row["year"]), int(row["horizon"]))


def table_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def compare_table(actual, expected):
    observed = {key(row): row for row in actual}
    reference = {key(row): row for row in expected}
    if len(observed) != len(actual) or set(observed) != set(reference):
        raise ValueError("Metric table has missing, duplicate, or extra planned cells")
    comparisons = 0
    for ident, row in reference.items():
        reported = observed[ident]
        if int(reported["origins"]) != row["origins"] or reported["family"] != row["family"]:
            raise ValueError("Cell origin count or family label differs")
        comparisons += 2
        for field in SCALARS:
            if not math.isclose(float(reported[field]), row[field], rel_tol=5e-12, abs_tol=1e-12):
                raise ValueError(f"Independent metric mismatch at {ident}, field {field}")
            comparisons += 1
    return comparisons


def prediction_digest(predictions):
    a = np.asarray(predictions)
    digest = hashlib.sha256()
    digest.update(b"predictions")
    digest.update(str(a.shape).encode())
    digest.update(str(a.dtype).encode())
    digest.update(a.tobytes())
    return digest.hexdigest()


def approved_market(entry, end_exclusive):
    path = Path(entry["path"])
    if "sealed_confirmation" in path.parts or end_exclusive > "2024-01-01":
        raise ValueError("This D1 draft cannot open new EEX confirmation data")
    blob = path.read_bytes()
    if sha256(blob) != entry["sha256"]:
        raise ValueError("Approved development source byte identity changed")
    records = list(csv.DictReader(io.StringIO(blob.decode("utf-8-sig"))))
    dates = [r["date"][:10] for r in records]
    if not dates or any(b <= a for a, b in zip(dates, dates[1:])):
        raise ValueError("Source dates not strictly ordered")
    selected = [r for r in records if r["date"][:10] < end_exclusive]
    return np.asarray([r["date"][:10] for r in selected]), np.asarray([float(r["price"]) for r in selected])


def verify(protocol_path, override_root=None):
    protocol_blob = protocol_path.read_bytes()
    cfg = json.loads(protocol_blob)
    protocol_hash = sha256(protocol_blob)
    root = Path(override_root or cfg["study_root"])
    if cfg["data_end_exclusive"] > "2024-01-01" or any(int(y) >= 2024 for y in cfg["scored_years"]):
        raise ValueError("D1-only boundary guard")
    for rel, expected_hash in cfg["source_sha256"].items():
        if sha256((root / rel).read_bytes()) != expected_hash:
            raise ValueError("Frozen experimental source changed")
    complete_path = root / "results/D1/COMPLETE.json"
    complete = json.loads(complete_path.read_bytes())
    expected_jobs = len(cfg["markets"]) * len(cfg["seeds"])
    if complete["jobs"] != expected_jobs or complete["failures"] or complete["protocol_sha256"] != protocol_hash:
        raise ValueError("D1 incomplete or protocol mismatch; no partial score audit")
    families = config_families(cfg)
    # Verify every job is complete before opening any market or forecast file.
    jobs = []
    for entry in cfg["markets"]:
        for seed in cfg["seeds"]:
            local = root / "local_only/D1" / f"{entry['name']}_s{seed}"
            receipt = json.loads((local / "COMPLETE.json").read_bytes())
            started = json.loads((local / "STARTED.json").read_bytes())
            if receipt["status"] != "complete" or started["protocol_sha256"] != protocol_hash:
                raise ValueError("A planned job is not complete under this protocol")
            runs = {r["config_id"]: r for r in receipt["runs"]}
            if len(runs) != len(receipt["runs"]) or set(runs) != set(families):
                raise ValueError("A planned predictor is missing or duplicated")
            jobs.append((entry, seed, local, receipt, runs))
    market_cache, expected_all, comparisons = {}, [], 0
    for entry, seed, local, receipt, runs in jobs:
        if entry["name"] not in market_cache:
            market_cache[entry["name"]] = approved_market(entry, cfg["data_end_exclusive"])
        dates, prices = market_cache[entry["name"]]
        start = int(np.searchsorted(dates, cfg["online_start"]))
        expected_origins = np.arange(start, len(prices), dtype=int)
        if receipt["source_sha256"] != entry["sha256"]:
            raise ValueError("Job source identity differs from protocol")
        cells = []
        for config, family in families.items():
            with np.load(local / f"forecast_{config}.npz", allow_pickle=False) as payload:
                if set(payload.files) != {"origins", "predictions"}:
                    raise ValueError("Unexpected forecast schema")
                origins, predictions = payload["origins"], payload["predictions"]
            if not np.array_equal(origins, expected_origins) or predictions.dtype != np.dtype("float64"):
                raise ValueError("Forecast origin chronology or dtype differs")
            if prediction_digest(predictions) != runs[config]["predictions_sha256"]:
                raise ValueError("Forecast digest differs from completion receipt")
            recomputed = metric_rows(entry["name"], dates, prices, origins, predictions, cfg["scored_years"])
            for row in recomputed:
                row.update(seed=int(seed), config_id=config, family=family)
            config_score = mean([r["relative_mse"] for r in recomputed])
            if not math.isclose(config_score, runs[config]["score"], rel_tol=5e-12, abs_tol=1e-12):
                raise ValueError("Per-job equal-cell score differs")
            comparisons += 1
            cells.extend(recomputed)
        comparisons += compare_table(table_rows(local / "METRIC_CELLS.csv"), cells)
        expected_all.extend(cells)
    comparisons += compare_table(table_rows(root / "results/D1/ALL_METRICS.csv"), expected_all)
    if complete["metric_rows"] != len(expected_all):
        raise ValueError("Completion metric count mismatch")
    cube_size = len(cfg["markets"]) * len(cfg["seeds"]) * len(cfg["scored_years"]) * len(HORIZONS)
    macro = []
    for config, family in families.items():
        cells = [r for r in expected_all if r["config_id"] == config]
        if len(cells) != cube_size:
            raise ValueError("Unbalanced market/seed/year/horizon cube")
        macro.append({"config_id": config, "family": family, "cells": len(cells),
                      "equal_cell_relative_mse": mean([r["relative_mse"] for r in cells])})
    return {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "passed": True,
            "protocol_sha256": protocol_hash, "audit_script_sha256": sha256(Path(__file__).read_bytes()),
            "completed_jobs": expected_jobs, "predictors_per_job": len(families),
            "recomputed_metric_cells": len(expected_all), "reported_value_comparisons": comparisons,
            "development_only": True, "new_confirmation_data_read": False,
            "training_code_imported": False, "market_fit": False,
            "equal_cell_aggregates_in_protocol_order": macro,
            "inference_limit": "These are development diagnostics. No confirmatory p-value, bootstrap grid, model ranking rule or selected candidate is introduced by this audit."}


def synthetic():
    dates = np.asarray([(dt.date(2019, 12, 1) + dt.timedelta(days=i)).isoformat() for i in range(90)])
    prices = np.arange(1., 91.)
    origins = np.arange(90)
    rw = np.repeat(prices[:, None], 20, axis=1)
    perfect = rw.copy()
    for origin in origins:
        for horizon in range(1, 21):
            if origin + horizon < len(prices):
                perfect[origin, horizon-1] = prices[origin+horizon]
    rw_rows = metric_rows("SYNTHETIC", dates, prices, origins, rw, (2019, 2020))
    perfect_rows = metric_rows("SYNTHETIC", dates, prices, origins, perfect, (2019, 2020))
    assert [r["origins"] for r in rw_rows] == [11, 11, 11, 39, 39, 39]
    for row in rw_rows:
        assert row["mse"] == row["horizon"] ** 2 and row["relative_mse"] == 1.
        assert row["mae"] == row["horizon"] and row["directional_accuracy"] == 0.
    for row in perfect_rows:
        assert row["mse"] == 0. and row["relative_mse"] == 0. and row["directional_accuracy"] == 1.
    enriched = [{**r, "seed": 7, "config_id": "rw", "family": "rw"} for r in rw_rows]
    assert compare_table(enriched, enriched) == 42
    for invalid in (enriched[:-1], enriched + [enriched[0]]):
        try:
            compare_table(invalid, enriched)
            raise AssertionError("Missing/duplicate cell accepted")
        except ValueError:
            pass
    corrupt = [dict(r) for r in enriched]
    corrupt[0]["mse"] += 1.
    try:
        compare_table(corrupt, enriched)
        raise AssertionError("Altered metric accepted")
    except ValueError:
        pass
    return {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "passed": True,
            "audit_script_sha256": sha256(Path(__file__).read_bytes()),
            "checks": ["full-H20 within-year common origins", "RW exact relative MSE=1",
                       "perfect known signal and direction", "all metric cells compared",
                       "missing/duplicate table refusal", "altered numeric metric refusal"],
            "market_files_read": False, "rrc_results_read": False, "market_fit": False,
            "status": "Draft API; real protocol/result locations require parent handoff before run."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    check = commands.add_parser("check")
    check.add_argument("--output", required=True, type=Path)
    run = commands.add_parser("run")
    run.add_argument("--protocol", required=True, type=Path)
    run.add_argument("--study-root", type=Path)
    run.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Audit receipt exists; refusing overwrite")
    receipt = synthetic() if args.action == "check" else verify(args.protocol, args.study_root)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
