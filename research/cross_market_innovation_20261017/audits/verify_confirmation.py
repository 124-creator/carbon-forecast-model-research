"""Independent completed-confirmation audit. No training or selection changes.

Unlike the preflight modules, this script is authorized to open the frozen
Shanghai confirmation partition, but only after all 62 formal fits complete.
Every prediction, label and selected-state validation is replayed with NumPy.
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import csv
import datetime as dt
import hashlib
import importlib.util
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from verify_d1 import (ROOT, HERE, H, Comparisons, cell_metrics, digest, load_sources,
                       metric_check, pack, predict_normalized, state_hash, validate_score)
from verify_d2 import check_epoch, support_check, with_levels, MECHANISM_FAMILIES
from numpy_patchtst import predict_raw as patch_raw
from verify_development_selection import compare_tree

OUT = HERE / "confirmation_audit"


def load_completed_markets(plan, complete):
    assert complete["all_completed"] and complete["completed_predictor_fits"] == 62
    markets, metadata = load_sources({"registry_sha256": plan["data_registry_sha256"]})
    # The parent authorized this new price-file read after all formal fits.
    raw = Path(plan["sealed_path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == plan["sealed_sha256"]
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    assert len(rows) == 261
    dates = np.array([row["date"] for row in rows])
    assert np.all(dates >= "2024-01-01") and np.all(dates < "2026-01-01")
    assert np.all(dates[1:] > dates[:-1])
    price = np.array([float(row["price"]) for row in rows])
    assert np.isfinite(price).all() and np.all(price > 0)
    day = np.array([dt.date.fromisoformat(x).toordinal() for x in dates])
    old = markets["SHANGHAI_D4"]
    assert old["dates"][-1] < dates[0]
    markets["SHANGHAI_D4"] = dict(price=np.r_[old["price"], price], dates=np.r_[old["dates"], dates], days=np.r_[old["days"], day])
    return markets, metadata, {"sha256": plan["sealed_sha256"], "rows": len(rows),
                               "first_date": str(dates[0]), "last_date": str(dates[-1]),
                               "raw_observations_by_year": {y: int(sum(x[:4] == y for x in dates)) for y in ("2024", "2025")}}


def model_summary(group, reference, arrays, cmp, published):
    """Independently reduce verified point losses to the public seed/cell table."""
    years = np.array([int(x[:4]) for x in reference["dates"]])
    truth = reference["raw_y"]
    predicted = arrays["full"]
    seeds = sorted({row["task"]["seed"] for row in group})
    cells = []
    for year in (2024, 2025):
        use = years == year
        for h in (1, 5, 20):
            losses = (predicted[:, use, h - 1] - truth[None, use, h - 1]) ** 2
            rw = float(np.mean(truth[use, h - 1] ** 2))
            cells.append(dict(year=year, horizon=h, origins=int(use.sum()), mse=float(losses.mean()), rw_mse=rw,
                              relative_mse=float(losses.mean() / rw),
                              mae=float(np.mean(np.abs(predicted[:, use, h - 1] - truth[None, use, h - 1])))))
    expected = dict(
        name=published["name"], scope=group[0]["task"]["scope"], config_id=group[0]["task"]["config"]["id"],
        config=group[0]["task"]["config"], score=float(np.mean([c["relative_mse"] for c in cells])),
        seeds=seeds, fits=len(group), cells=cells,
        per_seed={str(s): float(np.mean([np.mean((predicted[j, years == y, h] - truth[years == y, h]) ** 2)
                                        / np.mean(truth[years == y, h] ** 2) for y in (2024, 2025) for h in (0, 4, 19)]))
                  for j, s in enumerate(seeds)},
        per_year={str(y): float(np.mean([c["relative_mse"] for c in cells if c["year"] == y])) for y in (2024, 2025)},
        per_horizon={str(h): float(np.mean([c["relative_mse"] for c in cells if c["horizon"] == h])) for h in (1, 5, 20)},
        views={view: float(np.mean([np.mean((p[:, years == y, h] - truth[None, years == y, h]) ** 2)
                                   / np.mean(truth[years == y, h] ** 2) for y in (2024, 2025) for h in (0, 4, 19)])) for view, p in arrays.items()},
    )
    compare_tree(cmp, expected, published, "summary/" + published["name"])
    return expected


def main():
    OUT.mkdir(exist_ok=True)
    plan_path = ROOT / "CONFIRMATION_PRE_RUN.json"
    plan = json.loads(plan_path.read_text())
    complete = json.loads((ROOT / "CONFIRMATION_RUN_COMPLETE.json").read_text())
    started = json.loads((ROOT / "CONFIRMATION_STARTED.json").read_text())
    assert complete["all_completed"] and complete["completed_predictor_fits"] == len(plan["tasks"]) == 62
    assert complete["completed_imputer_fits"] == 6 and complete["completed_imputer_steps"] == 3000
    assert not complete["failures"] and complete["additional_retry_fits"] == 0
    assert not complete["confirmation_scores_used_for_configuration_changes"]
    assert complete["elapsed_seconds"] <= plan["deadline_seconds"]
    assert started["plan_sha256"] == digest(plan_path)
    assert plan["created_utc"] < started["utc"] < complete["completed_utc"]
    for file, sha in plan["source_sha256"].items():
        assert digest(ROOT / "confirmation_source" / file) == sha, file
    for file, key in (("CONFIRMATION_DESIGN_RULES.json", "rules_sha256"), ("DEVELOPMENT_AGGREGATE.json", "selection_sha256")):
        assert digest(ROOT / file) == plan[key]
    selection_audit = json.loads((HERE / "d2_audit/AUDIT_DEVELOPMENT_SELECTION_AND_LEARNING.json").read_text())
    assert selection_audit["status"] == "PASS" and selection_audit["development_aggregate_sha256"] == plan["selection_sha256"]
    assert plan["selected_candidate"]["id"] == selection_audit["selected_candidate"]["config_id"]
    assert plan["selected_baseline"]["config_id"] == selection_audit["selected_baseline"]["config_id"]
    assert not plan["prices_opened_before_freeze"] and not plan["confirmation_scored_before_freeze"]
    cmp = Comparisons()
    markets, dev_meta, confirmation_meta = load_completed_markets(plan, complete)
    public = json.loads((ROOT / "CONFIRMATION_RESULTS.json").read_text())
    assert len(public["results"]) == 62
    caches, initial, audits, q_audits, q_states = {}, {}, [], [], {}
    prediction_cache = {}
    for task, public_row in zip(plan["tasks"], public["results"]):
        tid, family, config = task["id"], task["config"]["family"], task["config"]
        assert public_row["task"] == task
        rp = ROOT / "confirmation_receipts" / (tid + ".json")
        sp = ROOT / "local_only/confirmation" / (tid + "_state.npz")
        pp = ROOT / "local_only/confirmation" / (tid + "_predictions.npz")
        receipt = json.loads(rp.read_text())
        assert receipt["task"] == task and receipt["seed"] == task["seed"]
        assert receipt["hyperparameters_selected_by_development_only"] and receipt["other_market_prices_end_before_2024"]
        for name, value in config.items():
            assert receipt["config"][name] == value
        key = (tuple(task["market_names"]), task["cutoff"])
        if key not in caches:
            train, supports = pack(markets, *key, "train")
            valid, _ = pack(markets, *key, "validation")
            evaluate, _ = pack(markets, ("SHANGHAI_D4",), task["cutoff"], "evaluation")
            evaluate["market_names"] = list(key[0])
            evaluate["market"] = np.full(len(evaluate["origin"]), key[0].index("SHANGHAI_D4"), dtype=np.int64)
            caches[key] = (train, with_levels(valid, markets), with_levels(evaluate, markets), supports)
        train, valid, evaluate, supports = caches[key]
        support_check(cmp, supports, receipt["supports"], tid + "/support")
        with np.load(sp, allow_pickle=False) as stream:
            state = {k: stream[k] for k in stream.files}
        assert state_hash(state) == receipt["state_sha256"]
        buffers = [k for k in state if any(s in k for s in ("running_mean", "running_var", "num_batches_tracked"))]
        assert sum(v.size for k, v in state.items() if k not in buffers) == receipt["parameter_count"]
        if family == "patchtst":
            assert set(buffers) == set(receipt["batchnorm_state_entries"]) and len(buffers) == 6
            assert receipt["trainable_parameter_count"] == receipt["parameter_count"] - 1
            for name in buffers:
                if "num_batches_tracked" in name:
                    assert int(state[name]) == receipt["selected_epoch"]
        with np.load(pp, allow_pickle=False) as stream:
            saved = {k: stream[k] for k in stream.files}
        prediction_cache[tid] = saved
        for name in ("origin", "dates", "market", "market_names"):
            cmp.equal(saved[name], evaluate[name], tid + "/" + name)
        for name in ("raw_y", "scale", "stat_raw"):
            cmp.close(saved[name], evaluate[name], tid + "/" + name, atol=1e-13)
        views = ("full",) if family == "patchtst" else ("full", "short", "sparse")
        computed, max_forward = {}, 0.0
        for view in views:
            if family == "patchtst":
                forecast = patch_raw(state, evaluate["levels"], receipt["model_metadata"])
            else:
                forecast = predict_normalized(state, config, evaluate, view) * evaluate["scale"][:, None] * np.sqrt(H)
            max_forward = max(max_forward, float(np.max(np.abs(forecast - saved["prediction_" + view]))))
            cmp.close(forecast, saved["prediction_" + view], tid + "/numpy_forward/" + view)
            computed[view] = cell_metrics(evaluate, forecast, view)
            metric_check(cmp, computed[view], receipt["confirmation_metrics"][view], tid + "/receipt/" + view)
            metric_check(cmp, computed[view], public_row["metrics"][view], tid + "/public/" + view)
        if family == "patchtst":
            val_prediction = patch_raw(state, valid["levels"], receipt["model_metadata"]) / valid["scale"][:, None] / np.sqrt(H)
        else:
            val_prediction = predict_normalized(state, config, valid, "full")
        validation = validate_score(valid, val_prediction)
        cmp.close(validation, receipt["validation_score"], tid + "/selected_validation")
        if family == "ridge":
            assert receipt["epochs"] == receipt["selected_epoch"] == 0 and not receipt["trace"]
        else:
            check_epoch(cmp, receipt, family, tid)
        for field in ("epochs", "selected_epoch", "seconds"):
            assert public_row[field] == receipt[field]
        if family in MECHANISM_FAMILIES:
            initial_key = key + (task["seed"], config["hidden"], config["shrink"])
            assert initial.setdefault(initial_key, receipt["initial_state_sha256"]) == receipt["initial_state_sha256"]
        if family in ("kcm", "kcm_unblocked"):
            for row in receipt["kernel_audit"]:
                n = len(supports[task["market_names"][row["market"]]]["train"])
                expected_count = (n - 41) * (n - 40) if family == "kcm" else n * (n - 1)
                assert row["ordered_pairs"] == expected_count
        if family == "mcssl":
            qid = f'imputer_{task["cutoff"][:4]}_s{task["seed"]}'
            qpath = ROOT / "local_only/confirmation" / (qid + "_state.npz")
            qrpath = ROOT / "confirmation_receipts" / (qid + ".json")
            qr = json.loads(qrpath.read_text())
            with np.load(qpath, allow_pickle=False) as stream:
                qs = {k: stream[k] for k in stream.files}
            q_states[qid] = qs
            qhash = hashlib.sha256()
            for name, value in qs.items():
                qhash.update(name.encode()); qhash.update(value.tobytes())
            assert qhash.hexdigest() == qr["state_sha256"]
            assert receipt["imputer_state_sha256"] == state_hash(qs) and receipt["imputer_unchanged"]
            assert qr["maxsteps"] == qr["optimizer_steps"] == len(qr["training_loss_trace"]) == 500
            assert qr["seed"] == task["seed"] + 10000 and qr["forecaster_seed"] == task["seed"]
            assert qr["training_origins"] == len(train["origin"]) and sum(v.size for v in qs.values()) == 19733
            assert np.isfinite(qr["training_loss_trace"]).all()
            support_check(cmp, supports, qr["supports"], qid + "/support")
            q_audits.append(dict(id=qid, steps=500, receipt_sha256=digest(qrpath), state_file_sha256=digest(qpath)))
        audits.append(dict(id=tid, task=task, metrics=computed, selected_epoch=receipt["selected_epoch"],
                           epochs=receipt["epochs"], validation_score=validation, maximum_forward_difference=max_forward,
                           receipt_sha256=digest(rp), state_file_sha256=digest(sp), predictions_file_sha256=digest(pp)))
    attempts = [json.loads(line) for line in (ROOT / "CONFIRMATION_ATTEMPTS.jsonl").read_text().splitlines()]
    assert [r["task_id"] for r in attempts] == [r["id"] for r in plan["tasks"]]
    assert len(q_states) == 6 and len(q_audits) == 6
    assert len(list((ROOT / "confirmation_receipts").glob("imputer_*_STARTED.json"))) == 6
    groups = defaultdict(list)
    for row in audits:
        groups[row["task"]["scope"] + ":" + row["task"]["config"]["id"]].append(row)
    published = json.loads((ROOT / "CONFIRMATION_SUMMARY.json").read_text())
    published_models = {r["name"]: r for r in published["models"]}
    reference, model_arrays, model_summaries = None, {}, []
    for name, group in groups.items():
        seeds = sorted({r["task"]["seed"] for r in group})
        arrays_by_view = defaultdict(list)
        for seed in seeds:
            ordered = sorted([r for r in group if r["task"]["seed"] == seed], key=lambda r: r["task"]["cutoff"])
            assert len(ordered) == 2 and [r["task"]["cutoff"] for r in ordered] == ["2024-01-01", "2025-01-01"]
            packs = [prediction_cache[r["id"]] for r in ordered]
            metadata = {k: np.concatenate([p[k] for p in packs]) for k in ("origin", "dates", "raw_y", "stat_raw")}
            if reference is None:
                reference = metadata
            for k in metadata:
                cmp.equal(metadata[k], reference[k], name + "/paired/" + k)
            for view in ordered[0]["metrics"]:
                arrays_by_view[view].append(np.concatenate([p["prediction_" + view] for p in packs]))
        arrays_by_view = {v: np.stack(p) for v, p in arrays_by_view.items()}
        model_summaries.append(model_summary(group, reference, arrays_by_view, cmp, published_models[name]))
        full = arrays_by_view["full"]
        model_arrays[name] = np.repeat(full, 3, axis=0) if len(seeds) == 1 else full
    years = np.array([int(d[:4]) for d in reference["dates"]])
    assert len(years) == 221 and published["origins"] == 221
    observed_counts = {str(y): int(np.sum(years == y)) for y in (2024, 2025)}
    assert observed_counts == published["origins_by_year"]
    for name, pred in (("rw", np.zeros_like(reference["raw_y"])), ("stat", reference["stat_raw"])):
        key = "deterministic:" + name
        model_arrays[key] = np.repeat(pred[None], 3, axis=0)
        values = []
        for year in (2024, 2025):
            ids = years == year
            data = dict(raw_y=reference["raw_y"][ids], dates=reference["dates"][ids],
                        market=np.zeros(int(ids.sum()), dtype=int), market_names=["SHANGHAI_D4"])
            metric = cell_metrics(data, pred[ids], "full")
            metric_check(cmp, metric, public["controls"][str(year) + "-01-01"][name], key + "/control")
            values.extend([{k: v for k, v in cell.items() if k not in ("market", "view")} for cell in metric["cells"]])
        expected = dict(name=key, scope="deterministic", config_id=name,
                        score=float(np.mean([c["relative_mse"] for c in values])), seeds=[], fits=0, cells=values,
                        per_year={str(y): float(np.mean([c["relative_mse"] for c in values if c["year"] == y])) for y in (2024, 2025)},
                        per_horizon={str(h): float(np.mean([c["relative_mse"] for c in values if c["horizon"] == h])) for h in (1, 5, 20)})
        compare_tree(cmp, expected, published_models[key], "summary/" + key)
        model_summaries.append(expected)
    candidate = "pooled:" + plan["selected_candidate"]["id"]
    baseline = plan["selected_baseline"]["scope"] + ":" + plan["selected_baseline"]["config_id"]
    augmentation = "pooled:augment_h32_s1_l0"
    assert published["primary_candidate"] == candidate and published["primary_baseline"] == baseline and published["matched_augmentation"] == augmentation
    statpath = ROOT / "confirmation_source/paired_statistics.py"
    assert digest(statpath) == "c0003775786ad89aead6a9007d3bf28dc7b01aa647df5be6eaece7b91d4bcd81"
    spec = importlib.util.spec_from_file_location("frozen_confirmation_statistics", statpath)
    stats = importlib.util.module_from_spec(spec); spec.loader.exec_module(stats)
    inference = stats.confirmation_inference({name: model_arrays[name] for name in (candidate, baseline, augmentation)},
                                            reference["raw_y"], years, reference["origin"], candidate=candidate,
                                            strongest_development_baseline=baseline, matched_augmentation=augmentation,
                                            seed_ids=(17, 29, 43))
    published_inference = json.loads((ROOT / "CONFIRMATION_INFERENCE.json").read_text())
    numerical_replay, numerical_published = dict(inference), dict(published_inference)
    replay_numpy = numerical_replay.pop("numpy_version")
    published_numpy = numerical_published.pop("numpy_version")
    # Metadata faithfully records different fitting/auditing vs postprocessing
    # environments; compare every numerical field, never rewrite that metadata.
    compare_tree(cmp, numerical_replay, numerical_published, "registered_inference")
    assert published["both_primary_comparisons_pass"] == inference["both_primary_comparisons_pass"]
    with np.load(ROOT / "results/CONFIRMATION_PRIMARY_INPUTS.npz", allow_pickle=False) as stream:
        for key, data in (("raw_y", reference["raw_y"]), ("years", years), ("origins", reference["origin"]),
                          ("dates", reference["dates"]), ("candidate", model_arrays[candidate]),
                          ("baseline", model_arrays[baseline]), ("augmentation", model_arrays[augmentation]), ("seed_ids", np.array([17, 29, 43]))):
            cmp.equal(stream[key], data, "public_statistics_input/" + key)
    correction_path = ROOT / "POSTPROCESSING_CORRECTION.json"
    correction = json.loads(correction_path.read_text()) if correction_path.exists() else None
    result = dict(status="PASS", audit_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  completed_predictor_fits=62, completed_imputer_fits=6, imputer_steps_accounted=3000,
                  independent_comparisons=cmp.elements, maximum_absolute_difference=cmp.maximum_error,
                  largest_difference_location=cmp.largest,
                  maximum_forward_difference_by_family={family: max(r["maximum_forward_difference"] for r in audits if r["task"]["config"]["family"] == family)
                                                       for family in sorted({r["task"]["config"]["family"] for r in audits})},
                  plan_sha256=digest(plan_path), source_sha256=plan["source_sha256"],
                  audit_script_sha256=digest(Path(__file__)), numpy_patchtst_sha256=digest(HERE / "numpy_patchtst.py"),
                  environment=dict(training_numpy=started["numpy"], training_torch=started["torch"],
                                   original_postprocessing_numpy=published_numpy, audit_numpy=replay_numpy,
                                   statistical_numerics_match_across_recorded_versions=True),
                  selected_models=dict(candidate=candidate, baseline=baseline, augmentation=augmentation),
                  scores={m["name"]: m["score"] for m in model_summaries},
                  primary_inference=inference["blocks"]["40"], both_primary_comparisons_pass=inference["both_primary_comparisons_pass"],
                  raw_confirmation_metadata=confirmation_meta, evaluation_origins_by_year=observed_counts,
                  auditor_model_fits=0, auditor_optimizer_steps=0,
                  confirmation_partition_first_read_by_this_script="after CONFIRMATION_RUN_COMPLETE all_completed=true; parent explicitly authorized post-run audit",
                  postprocessing_correction=correction,
                  checks=["all 62 mature supports and annual training-only scalers", "4-market vs local Shanghai ID correspondence",
                          "all recurrent/ridge/PatchTST full and available coarse predictions", "selected-state validation and epoch-selection logic",
                          "all PatchTST BatchNorm buffers and author core state", "6 frozen historical-only imputers and 3000 steps",
                          "all seed/year/horizon point-loss summaries", "exact three fixed comparator arrays and registered bootstrap/Holm outputs",
                          "public primary-input artifact matches independently aligned saved predictions"],
                  limitations=["Inference is re-executed from independently aligned and prediction-verified arrays with the frozen statistics implementation; a separate scalar synthetic test and a second agent reviewed its numerical and mathematical implementation.",
                               "Intermediate optimizer states, all epoch weights, and imputer stochastic training trajectories are not independently replayed.",
                               "Approximate fixed-forecast bootstrap inference does not repeat training or prove stationarity/independence; a failed primary comparison is not repaired with secondary raw-MSE or alternate blocks."])
    (OUT / "AUDIT_CONFIRMATION_COMPLETE.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (OUT / "RECOMPUTED_CONFIRMATION_COMPLETE.json").write_text(json.dumps(dict(fits=audits, imputers=q_audits, models=model_summaries, inference=inference), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "completed_predictor_fits", "completed_imputer_fits", "independent_comparisons", "maximum_absolute_difference", "largest_difference_location", "evaluation_origins_by_year", "both_primary_comparisons_pass")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
