"""Independent D2 replay: NumPy/math only, approved development files only.

All recurrent and small PatchTST saved-state predictions and final validation
scores are independently recomputed. Historical imputer state, boundaries and
step accounting are verified without re-running its stochastic optimizer.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from verify_d1 import (ROOT, HERE, H, Comparisons, cell_metrics, digest,
                       load_sources, metric_check, pack, predict_normalized,
                       state_hash, validate_score)
from numpy_patchtst import predict_raw as patch_raw

OUT = HERE / "d2_audit"
MECHANISM_FAMILIES = {"gru", "augment", "kcm", "pointwise", "kcm_unblocked", "mcssl"}


def with_levels(data, markets):
    out = dict(data)
    levels = np.empty((len(data["origin"]), 21))
    for mid, name in enumerate(data["market_names"]):
        use = data["market"] == mid
        levels[use] = markets[name]["price"][data["origin"][use, None] + np.arange(-20, 1)]
    out["levels"] = levels
    return out


def crop_train(train, supports, cap):
    if cap is None:
        return train, supports
    ids = np.concatenate([np.flatnonzero(train["market"] == mid)[-cap:] for mid in np.unique(train["market"])])
    new = {k: v[ids] if isinstance(v, np.ndarray) else v for k, v in train.items()}
    changed = {name: dict(row) for name, row in supports.items()}
    for mid, name in enumerate(train["market_names"]):
        changed[name]["train"] = new["origin"][new["market"] == mid]
        changed[name]["cap"] = cap
        changed[name]["scaler_uses_common_uncapped_training_history"] = True
    return new, changed


def support_check(cmp, computed, saved, prefix):
    assert set(computed) == set(saved)
    for name, expected in computed.items():
        observed = saved[name]
        for field in ("train", "validation"):
            cmp.equal(expected[field], observed[field], prefix + "/" + name + "/" + field)
        for field in ("end", "scale_start", "scale_stop", "cutoff", "cap"):
            assert expected[field] == observed[field], (prefix, name, field)
        if expected["cap"] is not None:
            assert observed["scaler_uses_common_uncapped_training_history"]
        cmp.close(expected["scale"], observed["scale"], prefix + "/" + name + "/scale", atol=1e-14)


def check_epoch(cmp, receipt, family, prefix):
    selected, epochs = receipt["selected_epoch"], receipt["epochs"]
    assert epochs == len(receipt["trace"]) <= 60 and 0 <= selected <= epochs
    best, found, stale = receipt["initial_validation"], 0, 0
    augmented = family in MECHANISM_FAMILIES - {"gru"}
    for epoch, row in enumerate(receipt["trace"], 1):
        assert row["epoch"] == epoch
        assert row["view"] == (("short" if epoch % 2 else "sparse") if augmented else "full")
        values = [v for v in row.values() if isinstance(v, (int, float))]
        assert np.isfinite(values).all()
        if "mechanism_penalty" in row:
            assert row["mechanism_penalty"] >= 0
        if family == "patchtst":
            assert row["training_mode"] == "train" and row["validation_mode"] == "eval" and row["status"] == "validated"
        if row["validation_score"] < best - 1e-6:
            best, found, stale = row["validation_score"], epoch, 0
        else:
            stale += 1
        assert stale < 12 or epoch == epochs
    assert found == selected and (epochs == 60 or stale == 12)
    cmp.close(best, receipt["validation_score"], prefix + "/epoch_selection")
    if family == "patchtst":
        assert receipt["eligible_for_selection"] and receipt["optimizer_steps"] == epochs


def audit():
    OUT.mkdir(exist_ok=True)
    plan_path = ROOT / "D2_PRE_RUN.json"
    plan = json.loads(plan_path.read_text())
    completed = json.loads((ROOT / "D2_RUN_COMPLETE.json").read_text())
    started = json.loads((ROOT / "D2_STARTED.json").read_text())
    assert completed["all_completed"] and completed["completed_predictor_fits"] == len(plan["tasks"]) == 158
    assert completed["completed_imputer_fits"] == 9 and completed["completed_imputer_steps"] == 4500
    assert not completed["failures"] and completed["additional_retry_fits"] == 0 and not completed["confirmation_used"]
    assert completed["elapsed_seconds"] <= plan["deadline_seconds"]
    assert started["plan_sha256"] == digest(plan_path)
    assert not plan["confirmation_files_allowed"]
    for name, sha in plan["source_sha256"].items():
        assert digest(ROOT / "d2_source" / name) == sha, name
    assert digest(ROOT / "D1_RUN_COMPLETE.json") == plan["d1_complete_sha256"]
    markets, metadata = load_sources(plan)
    cmp = Comparisons()
    caches, audited, initial_groups = {}, [], {}
    q_states, q_receipts, q_audits = {}, {}, []
    public = json.loads((ROOT / "D2_RESULTS.json").read_text())["results"]
    assert len(public) == 158
    # Extend D1's matching checks across stage boundaries, without replaying D1.
    d1 = json.loads((HERE / "d1_audit/RECOMPUTED_D1_PARTIAL_141.json").read_text())
    for row in d1:
        task, family = row["task"], row["task"]["config"]["family"]
        if family in MECHANISM_FAMILIES:
            receipt = json.loads((ROOT / "d1_receipts" / (task["id"] + ".json")).read_text())
            assert digest(ROOT / "d1_receipts" / (task["id"] + ".json")) == row["receipt_sha256"]
            key = (tuple(task["market_names"]), task["cutoff"], task["seed"], task["config"]["hidden"], task["config"]["shrink"], None)
            if key in initial_groups:
                assert initial_groups[key] == receipt["initial_state_sha256"]
            initial_groups[key] = receipt["initial_state_sha256"]
    for task, public_row in zip(plan["tasks"], public):
        tid, config, family = task["id"], task["config"], task["config"]["family"]
        assert public_row["task"] == task
        rp = ROOT / "d2_receipts" / (tid + ".json")
        sp = ROOT / "local_only/d2" / (tid + "_state.npz")
        pp = ROOT / "local_only/d2" / (tid + "_predictions.npz")
        receipt = json.loads(rp.read_text())
        assert receipt["task"] == task and receipt["seed"] == task["seed"]
        assert receipt["fitting_code_sources"] == plan["source_sha256"] and not receipt["confirmation_data_used"]
        for key, value in config.items():
            assert receipt["config"][key] == value, (tid, key)
        cache_key = (tuple(task["market_names"]), task["cutoff"])
        if cache_key not in caches:
            full_train, supports = pack(markets, *cache_key, "train")
            valid, _ = pack(markets, *cache_key, "validation")
            evaluate, _ = pack(markets, *cache_key, "evaluation")
            caches[cache_key] = (with_levels(full_train, markets), with_levels(valid, markets),
                                 with_levels(evaluate, markets), supports)
        full_train, valid, evaluate, full_supports = caches[cache_key]
        train, supports = crop_train(full_train, full_supports, task["train_cap"])
        support_check(cmp, supports, receipt["supports"], tid + "/support")
        with np.load(sp, allow_pickle=False) as stream:
            state = {k: stream[k] for k in stream.files}
        assert state_hash(state) == receipt["state_sha256"]
        buffers = [k for k in state if any(s in k for s in ("running_mean", "running_var", "num_batches_tracked"))]
        assert sum(v.size for k, v in state.items() if k not in buffers) == receipt["parameter_count"]
        if family == "patchtst":
            assert set(buffers) == set(receipt["batchnorm_state_entries"]) and len(buffers) == 6
            assert receipt["trainable_parameter_count"] == receipt["parameter_count"] - 1
            for key in buffers:
                if "num_batches_tracked" in key:
                    assert int(state[key]) == receipt["selected_epoch"]
                elif "running_var" in key:
                    assert np.all(state[key] >= 0)
            assert receipt["model_metadata"]["author_commit"] == "204c21efe0b39603ad6e2ca640ef5896646ab1a9"
        with np.load(pp, allow_pickle=False) as stream:
            saved = {k: stream[k] for k in stream.files}
        cmp.equal(saved["market_names"], evaluate["market_names"], tid + "/market_names")
        for field in ("origin", "dates", "market"):
            cmp.equal(saved[field], evaluate[field], tid + "/" + field)
        for field in ("raw_y", "scale", "stat_raw"):
            cmp.close(saved[field], evaluate[field], tid + "/" + field, atol=1e-13)
        views = ("full",) if family == "patchtst" else ("full", "short", "sparse")
        computed, forward_error = {}, 0.0
        for view in views:
            if family == "patchtst":
                forecast = patch_raw(state, evaluate["levels"], receipt["model_metadata"])
            else:
                forecast = predict_normalized(state, config, evaluate, view) * evaluate["scale"][:, None] * np.sqrt(H)
            forward_error = max(forward_error, float(np.max(np.abs(forecast - saved["prediction_" + view]))))
            cmp.close(forecast, saved["prediction_" + view], tid + "/numpy_forward/" + view)
            computed[view] = cell_metrics(evaluate, forecast, view)
            metric_check(cmp, computed[view], receipt["development_metrics"][view], tid + "/receipt/" + view)
            metric_check(cmp, computed[view], public_row["metrics"][view], tid + "/public/" + view)
        if family == "patchtst":
            vpred = patch_raw(state, valid["levels"], receipt["model_metadata"]) / valid["scale"][:, None] / np.sqrt(H)
        else:
            vpred = predict_normalized(state, config, valid, "full")
        score = validate_score(valid, vpred)
        cmp.close(score, receipt["validation_score"], tid + "/selected_validation")
        check_epoch(cmp, receipt, family, tid)
        for field in ("epochs", "selected_epoch", "seconds"):
            assert public_row[field] == receipt[field]
        if family in MECHANISM_FAMILIES:
            key = cache_key + (task["seed"], config["hidden"], config["shrink"], task["train_cap"])
            previous = initial_groups.setdefault(key, receipt["initial_state_sha256"])
            assert previous == receipt["initial_state_sha256"], tid
        if family in ("kcm", "kcm_unblocked"):
            for row in receipt["kernel_audit"]:
                origins = train["origin"][train["market"] == row["market"]]
                assert np.all(np.diff(origins) == 1)
                n = len(origins)
                expected = (n - 41) * (n - 40) if family == "kcm" else n * (n - 1)
                assert row["ordered_pairs"] == expected and row["fixed_feature_dimension"] == 60
                assert row["minimum_origin_separation"] == (41 if family == "kcm" else 1)
        if family == "mcssl":
            qid = f'imputer_{task["cutoff"][:4]}_s{task["seed"]}'
            qpath = ROOT / "local_only/d2" / (qid + "_state.npz")
            qrpath = ROOT / "d2_receipts" / (qid + ".json")
            if qid not in q_states:
                with np.load(qpath, allow_pickle=False) as stream:
                    q_states[qid] = {k: stream[k] for k in stream.files}
                qr = q_receipts[qid] = json.loads(qrpath.read_text())
                support_check(cmp, supports, qr["supports"], qid + "/support")
                qhash = hashlib.sha256()
                for key, value in q_states[qid].items():
                    qhash.update(key.encode()); qhash.update(value.tobytes())
                assert qhash.hexdigest() == qr["state_sha256"]
                assert qr["seed"] == task["seed"] + 10000 and qr["forecaster_seed"] == task["seed"]
                assert qr["optimizer_steps"] == qr["maxsteps"] == len(qr["training_loss_trace"]) == 500
                assert np.isfinite(qr["training_loss_trace"]).all() and min(qr["training_loss_trace"]) >= 0
                assert qr["training_origins"] == len(train["origin"])
                assert sum(v.size for v in q_states[qid].values()) == qr["parameter_count"] == 19733
                assert not any(qr[k] for k in ("future_target_access", "raw_y_access", "real_missing_date_access", "validation_selection", "conditional_law_correctness_claimed"))
                q_audits.append(dict(id=qid, steps=500, receipt_sha256=digest(qrpath), state_file_sha256=digest(qpath),
                                     training_origins=len(train["origin"])))
            assert receipt["imputer_unchanged"] and receipt["imputer_state_sha256"] == state_hash(q_states[qid])
            assert receipt["imputer_parameter_count"] == 19733
        audited.append(dict(id=tid, task=task, metrics=computed, selected_epoch=receipt["selected_epoch"],
                            epochs=receipt["epochs"], validation_score=score, maximum_forward_difference=forward_error,
                            receipt_sha256=digest(rp), state_file_sha256=digest(sp), predictions_file_sha256=digest(pp)))
    attempts = [json.loads(x) for x in (ROOT / "D2_ATTEMPTS.jsonl").read_text().splitlines()]
    assert [x["task_id"] for x in attempts] == [t["id"] for t in plan["tasks"]]
    assert len(q_states) == len(q_receipts) == 9
    assert len(list((ROOT / "d2_receipts").glob("imputer_*_STARTED.json"))) == 9
    result = dict(status="PASS", audit_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  audited_predictor_fits=len(audited), audited_imputer_fits=len(q_states),
                  imputer_optimizer_steps_accounted=4500, independent_comparisons=cmp.elements,
                  maximum_absolute_difference=cmp.maximum_error, largest_difference_location=cmp.largest,
                  family_counts=dict(Counter(row["task"]["config"]["family"] for row in audited)),
                  maximum_forward_difference_by_family={family: max(row["maximum_forward_difference"] for row in audited if row["task"]["config"]["family"] == family)
                                                       for family in sorted({row["task"]["config"]["family"] for row in audited})},
                  plan_sha256=digest(plan_path), source_sha256=plan["source_sha256"],
                  audit_script_sha256=digest(Path(__file__)), numpy_patchtst_sha256=digest(HERE / "numpy_patchtst.py"),
                  numpy_version=np.__version__, completed_receipt=completed, development_sources=metadata,
                  confirmation_file_reads=0, auditor_model_fits=0, auditor_optimizer_steps=0,
                  checks=["source and data identities", "fully mature support and same-year evaluation",
                          "common uncapped scaler with capped label windows", "all saved recurrent and PatchTST predictions",
                          "all final-state validation predictions and epoch selection logic",
                          "PatchTST RevIN/attention/GELU/BatchNorm/flatten head via NumPy/math",
                          "all published point-loss summaries", "matched initial hashes across D1/D2 mechanisms",
                          "kernel pair-count boundaries", "imputer state hash preserved through forecaster training",
                          "9 imputers x 500 updates, 158 predictors, no retries"],
                  limitations=["Intermediate optimizer states and epoch weights are not independently reconstructed.",
                               "Initial random-encoder/head values were hashed, not saved; matching hashes are checked without reconstructing random initialization.",
                               "Imputer training losses and per-epoch stochastic completion diversity are not independently re-simulated; source information boundaries, finite traces, support and saved/frozen parameter hashes are checked.",
                               "This implementation and score audit does not establish method novelty or confirmation significance."])
    (OUT / "AUDIT_D2_COMPLETE.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (OUT / "RECOMPUTED_D2_COMPLETE.json").write_text(json.dumps(dict(fits=audited, imputers=q_audits), indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "audited_predictor_fits", "audited_imputer_fits", "independent_comparisons", "maximum_absolute_difference", "largest_difference_location", "maximum_forward_difference_by_family")}, ensure_ascii=False))


if __name__ == "__main__":
    audit()
