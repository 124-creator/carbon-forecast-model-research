"""Verify the public confirmation CSV against already audited saved forecasts.

This is an export identity check, not another forecast fit or significance test.
It reads no original market price file and performs no model inference.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
from collections import Counter
from pathlib import Path

import numpy as np

from verify_d1 import ROOT, HERE, digest


def main():
    audit_dir = HERE / "confirmation_audit"
    prior_path = audit_dir / "AUDIT_CONFIRMATION_COMPLETE.json"
    replay_path = audit_dir / "RECOMPUTED_CONFIRMATION_COMPLETE.json"
    prior = json.loads(prior_path.read_text())
    replay = json.loads(replay_path.read_text())
    assert prior["status"] == "PASS" and prior["completed_predictor_fits"] == 62
    complete = json.loads((ROOT / "CONFIRMATION_RUN_COMPLETE.json").read_text())
    assert complete["all_completed"] and complete["completed_predictor_fits"] == 62

    expected = {}
    references = {}
    authenticated_predictions = {}
    for fit in replay["fits"]:
        task = fit["task"]
        path = ROOT / "local_only/confirmation" / (fit["id"] + "_predictions.npz")
        sha = digest(path)
        assert sha == fit["predictions_file_sha256"], fit["id"]
        authenticated_predictions[fit["id"]] = sha
        with np.load(path, allow_pickle=False) as data:
            key = task["cutoff"][:4]
            ref = {k: data[k].copy() for k in ("raw_y", "origin", "dates", "stat_raw")}
            if key in references:
                for field in ref:
                    assert np.array_equal(ref[field], references[key][field]), (key, field)
            else:
                references[key] = ref
            name = task["scope"] + ":" + task["config"]["id"]
            for i, origin in enumerate(data["origin"]):
                for h in (1, 5, 20):
                    cell = (name, task["seed"], int(origin), h)
                    assert cell not in expected, cell
                    expected[cell] = (str(data["dates"][i]), float(data["raw_y"][i, h-1]),
                                      float(data["prediction_full"][i, h-1]))

    for ref in references.values():
        for i, origin in enumerate(ref["origin"]):
            for h in (1, 5, 20):
                for name, forecast in (("rw", 0.0), ("stat", float(ref["stat_raw"][i, h-1]))):
                    key = ("deterministic:" + name, 17, int(origin), h)
                    assert key not in expected
                    expected[key] = (str(ref["dates"][i]), float(ref["raw_y"][i, h-1]), forecast)

    path = ROOT / "results/CONFIRMATION_FORECASTS_H1_H5_H20.csv"
    seen, counts, max_error = set(), Counter(), 0.0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["model", "seed", "date", "origin", "horizon",
                                     "raw_future_change", "predicted_change"]
        for row in reader:
            key = (row["model"], int(row["seed"]), int(row["origin"]), int(row["horizon"]))
            assert key in expected and key not in seen, key
            seen.add(key)
            date, truth, forecast = expected[key]
            assert row["date"] == date, key
            actual = (float(row["raw_future_change"]), float(row["predicted_change"]))
            assert actual == (truth, forecast), key
            max_error = max(max_error, abs(actual[0]-truth), abs(actual[1]-forecast))
            counts[row["model"]] += 1
    assert seen == set(expected) and len(seen) == 21879
    assert len(counts) == 13
    result = {
        "status": "PASS",
        "audit_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "Public CSV identity against previously independently replayed prediction artifacts",
        "rows": len(seen),
        "models": len(counts),
        "rows_by_model": dict(sorted(counts.items())),
        "maximum_absolute_float_roundtrip_difference": max_error,
        "duplicate_keys": 0,
        "missing_keys": 0,
        "extra_keys": 0,
        "csv_sha256": digest(path),
        "prior_complete_audit_sha256": digest(prior_path),
        "prior_complete_recomputed_sha256": digest(replay_path),
        "prediction_files_sha256": authenticated_predictions,
        "audit_script_sha256": digest(Path(__file__)),
        "numpy_version": np.__version__,
        "market_price_file_reads": 0,
        "model_forward_calls": 0,
        "model_fits": 0,
        "optimizer_steps": 0,
        "additional_statistical_tests": 0,
        "seed_column_note": "RW, statistical reference and deterministic ridge are exported once with seed label 17; no stochastic seed replication is claimed.",
    }
    output = audit_dir / "AUDIT_CONFIRMATION_EXPORT.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "rows", "models",
                                            "maximum_absolute_float_roundtrip_difference")}))


if __name__ == "__main__":
    main()
