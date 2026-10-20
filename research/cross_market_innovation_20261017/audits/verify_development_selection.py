"""Independent aggregation/selection and 24-fit learning diagnostic replay."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

from verify_d1 import (ROOT, HERE, H, Comparisons, cell_metrics, digest, load_sources,
                       metric_check, pack, predict_normalized, state_hash, validate_score)
from verify_d2 import crop_train, support_check, check_epoch

OUT = HERE / "d2_audit"


def compare_tree(cmp, a, b, label):
    if isinstance(a, dict):
        assert set(a) == set(b), (label, set(a), set(b))
        for k in a:
            compare_tree(cmp, a[k], b[k], label + "/" + str(k))
    elif isinstance(a, list):
        assert len(a) == len(b), label
        for i, (left, right) in enumerate(zip(a, b)):
            compare_tree(cmp, left, right, label + "/" + str(i))
    elif isinstance(a, float):
        cmp.close(a, b, label)
    else:
        assert a == b, (label, a, b)


def aggregate_replays(cmp):
    d1 = json.loads((HERE / "d1_audit/RECOMPUTED_D1_PARTIAL_141.json").read_text())
    d2 = json.loads((OUT / "RECOMPUTED_D2_COMPLETE.json").read_text())["fits"]
    rows = [("D1", r) for r in d1] + [("D2", r) for r in d2]
    groups = defaultdict(list)
    for stage, r in rows:
        task = r["task"]
        groups[(task["scope"], task["config"]["id"], task.get("train_cap"))].append((stage, r))
    expected = {}
    for (scope, config_id, cap), members in groups.items():
        seeds = sorted({row["task"]["seed"] for _, row in members})
        cells = [dict(c, seed=row["task"]["seed"]) for _, row in members for c in row["metrics"]["full"]["cells"]]
        # Balanced cells make averaging seed-specific cell MSE exactly the
        # registered seed-average point-loss estimand, without an ensemble.
        keys = [(c["seed"], c["market"], c["year"], c["horizon"]) for c in cells]
        assert len(keys) == len(set(keys))
        coverage = [{key[1:] for key in keys if key[0] == s} for s in seeds]
        assert all(cover == coverage[0] for cover in coverage)
        views = sorted({v for _, r in members for v in r["metrics"]})
        expected[(scope, config_id, cap)] = dict(
            scope=scope, config_id=config_id, config=members[0][1]["task"]["config"], train_cap=cap,
            fits=len(members), seeds=seeds, score=float(np.mean([c["relative_mse"] for c in cells])),
            per_seed={str(s): float(np.mean([c["relative_mse"] for c in cells if c["seed"] == s])) for s in seeds},
            per_market={m: float(np.mean([c["relative_mse"] for c in cells if c["market"] == m])) for m in sorted({c["market"] for c in cells})},
            per_year={str(y): float(np.mean([c["relative_mse"] for c in cells if c["year"] == y])) for y in sorted({c["year"] for c in cells})},
            per_horizon={str(h): float(np.mean([c["relative_mse"] for c in cells if c["horizon"] == h])) for h in (1, 5, 20)},
            views={v: float(np.mean([c["relative_mse"] for _, r in members if v in r["metrics"] for c in r["metrics"][v]["cells"]])) for v in views},
            selected_epoch_zero=sum(r["selected_epoch"] == 0 for _, r in members),
            task_ids=[dict(stage=stage, task_id=r["task"]["id"]) for stage, r in members],
        )
    published = json.loads((ROOT / "DEVELOPMENT_AGGREGATE.json").read_text())
    assert len(published["development_summary"]) == len(expected)
    for group in published["development_summary"]:
        key = (group["scope"], group["config_id"], group["train_cap"])
        compare_tree(cmp, expected[key], group, "aggregate/" + str(key))
    controls_saved = json.loads((HERE / "d1_audit/RECOMPUTED_D1_COMPLETE.json").read_text())["controls"]
    controls = {}
    for name in ("rw", "stat"):
        cells = [c for key, value in controls_saved.items() if key.startswith("(('GUANGZHOU', 'HUBEI', 'EU_REPORTED', 'SHANGHAI_D4')") for c in value[name]["cells"]]
        assert len(cells) == 36
        controls[name] = dict(score=float(np.mean([c["relative_mse"] for c in cells])), cells=cells)
    compare_tree(cmp, controls, published["controls"], "aggregate/controls")
    candidate = min([expected[("pooled", f"kcm_h32_s1_l{strength}", None)] for strength in (1, 100)],
                    key=lambda r: (r["score"], r["config"]["strength"]))
    assert candidate["seeds"] == [17, 29, 43] and candidate["fits"] == 9
    patch = min([r for r in expected.values() if r["config"]["family"] == "patchtst" and r["train_cap"] is None], key=lambda r: r["score"])
    ridge = min([r for r in expected.values() if r["config"]["family"] == "ridge"], key=lambda r: r["score"])
    baseline_pool = [expected[("pooled", "gru_price_h16_s0.1_l0", None)],
                     expected[("pooled", "lstm_price_h32_s1_l0", None)],
                     expected[("local", "gru_price_h16_s1_l0", None)], patch, ridge]
    baseline_pool += [dict(config_id=k, scope="deterministic", config=dict(id=k, family=k), score=v["score"]) for k, v in controls.items()]
    baseline = min(baseline_pool, key=lambda r: r["score"])
    for key, selected in (("selected_candidate", candidate), ("selected_patchtst", patch), ("selected_ridge", ridge), ("selected_best_baseline", baseline)):
        compare_tree(cmp, selected, published[key], "aggregate/" + key)
    curves = []
    for label, cid, scope in (("shared_price_GRU", "gru_price_h16_s0.1_l0", "pooled"),
                              ("KCM_lambda1", "kcm_h32_s1_l1", "pooled"),
                              ("local_price_GRU", "gru_price_h16_s1_l0", "local")):
        for cap in (100, 400, None):
            chosen = [r for _, r in rows if r["task"]["config"]["id"] == cid and r["task"]["scope"] == scope
                      and r["task"].get("train_cap") == cap and r["task"]["cutoff"] == "2023-01-01"
                      and (scope != "local" or r["task"]["seed"] == 17)]
            seeds = sorted({r["task"]["seed"] for r in chosen})
            scores = {str(s): float(np.mean([c["relative_mse"] for r in chosen if r["task"]["seed"] == s for c in r["metrics"]["full"]["cells"]])) for s in seeds}
            curves.append(dict(model=label, per_market_train_cap=cap, seeds=seeds, score=float(np.mean(list(scores.values()))),
                               per_seed=scores, period="2023", common_uncapped_training_scale=True))
    compare_tree(cmp, curves, published["learning_curve"], "aggregate/learning_curve")
    assert published["no_confirmation_data_read"] and published["model_selection_is_development"]
    return dict(groups=len(expected), candidate=candidate, baseline=baseline, patchtst=patch,
                all_groups=list(expected.values()), curves=curves)


def audit_learning(cmp):
    plan = json.loads((ROOT / "LEARNING_DIAGNOSTIC_PRE_RUN.json").read_text())
    complete = json.loads((ROOT / "LEARNING_DIAGNOSTIC_COMPLETE.json").read_text())
    d2_plan = json.loads((ROOT / "D2_PRE_RUN.json").read_text())
    assert digest(ROOT / "learning_diagnostic.py") == plan["source_sha256"]
    assert digest(ROOT / "D2_PRE_RUN.json") == plan["d2_plan_sha256"]
    for name, sha in d2_plan["source_sha256"].items():
        assert digest(ROOT / "d2_source" / name) == sha
    assert complete["all_completed"] and complete["completed_fits"] == len(plan["tasks"]) == 24
    assert not complete["failures"] and not complete["confirmation_used"]
    assert complete["seconds"] < plan["deadline_seconds"] and not plan["confirmation_data_allowed"]
    markets, _ = load_sources(d2_plan)
    names = plan["tasks"][0]["market_names"]
    train, supports = pack(markets, names, "2023-01-01", "train")
    valid, _ = pack(markets, names, "2023-01-01", "validation")
    evaluate, _ = pack(markets, names, "2023-01-01", "evaluation")
    cached, initial, checked = {}, {}, []
    public = json.loads((ROOT / "LEARNING_DIAGNOSTIC_RESULTS.json").read_text())["results"]
    for task, saved_result in zip(plan["tasks"], public):
        mode, cap = task["selection"], task["cap"]
        assert saved_result["task"] == task
        if (mode, cap) not in cached:
            if mode == "recent":
                chosen, ss = crop_train(train, supports, cap)
            else:
                selected = []
                for mid in range(len(names)):
                    rows = np.flatnonzero(train["market"] == mid)
                    assert len(rows) >= 400
                    # Independent rational-index construction, without the
                    # production np.linspace/rint helper.
                    grid = [round(k * (len(rows) - 1) / 399) for k in range(400)]
                    use = grid if cap == 400 else [grid[round(k * 399 / 99)] for k in range(100)]
                    assert len(set(use)) == cap and use[0] == 0 and use[-1] == len(rows) - 1
                    selected.extend(rows[use])
                chosen = {k: v[selected] if isinstance(v, np.ndarray) else v for k, v in train.items()}
                ss = {name: dict(row) for name, row in supports.items()}
                for mid, name in enumerate(names):
                    ss[name].update(train=chosen["origin"][chosen["market"] == mid], cap=cap,
                                    support_selection="uniform temporal lattice;100nestedin400",
                                    scaler_uses_common_uncapped_training_history=True)
            cached[(mode, cap)] = chosen, ss
        chosen, ss = cached[(mode, cap)]
        tid, family = task["id"], task["config"]["family"]
        rp = ROOT / "learning_receipts" / (tid + ".json")
        sp = ROOT / "local_only/learning" / (tid + "_state.npz")
        pp = ROOT / "local_only/learning" / (tid + "_predictions.npz")
        receipt = json.loads(rp.read_text())
        assert receipt["task"] == task and not receipt["confirmation_used"]
        support_check(cmp, ss, receipt["supports"], tid + "/support")
        with np.load(sp, allow_pickle=False) as stream:
            state = {k: stream[k] for k in stream.files}
        assert state_hash(state) == receipt["state_sha256"]
        with np.load(pp, allow_pickle=False) as stream:
            p = {k: stream[k] for k in stream.files}
        for key in ("origin", "dates", "market", "market_names"):
            cmp.equal(p[key], evaluate[key], tid + "/" + key)
        for key in ("raw_y", "scale"):
            cmp.close(p[key], evaluate[key], tid + "/" + key)
        forecast = predict_normalized(state, task["config"], evaluate, "full") * evaluate["scale"][:, None] * np.sqrt(H)
        cmp.close(forecast, p["prediction_full"], tid + "/numpy_forward")
        metric = cell_metrics(evaluate, forecast, "full")
        metric_check(cmp, metric, receipt["development_metrics"]["full"], tid + "/receipt_metric")
        metric_check(cmp, metric, saved_result["metrics"]["full"], tid + "/public_metric")
        score = validate_score(valid, predict_normalized(state, task["config"], valid, "full"))
        cmp.close(score, receipt["validation_score"], tid + "/selected_validation")
        check_epoch(cmp, receipt, family, tid)
        if family in ("kcm", "augment"):
            key = (mode, cap, task["seed"])
            assert initial.setdefault(key, receipt["initial_state_sha256"]) == receipt["initial_state_sha256"]
        if family == "kcm":
            for row in receipt["kernel_audit"]:
                origins = chosen["origin"][chosen["market"] == row["market"]]
                expected = int(np.sum(np.abs(origins[:, None] - origins[None]) > 40))
                assert row["ordered_pairs"] == expected
        checked.append(dict(task=task, metrics={"full": metric}, selected_epoch=receipt["selected_epoch"],
                            receipt_sha256=digest(rp), state_sha256=digest(sp), predictions_sha256=digest(pp)))
    assert len(checked) == len(public) == 24
    for name in names:
        assert set(cached[("spread", 100)][1][name]["train"]).issubset(set(cached[("spread", 400)][1][name]["train"]))
    summary = []
    for mode, family, cap in sorted({(r["task"]["selection"], r["task"]["config"]["family"], r["task"]["cap"]) for r in checked}):
        rows = [r for r in checked if (r["task"]["selection"], r["task"]["config"]["family"], r["task"]["cap"]) == (mode, family, cap)]
        summary.append(dict(selection=mode, family=family, cap=cap,
                            score=float(np.mean([r["metrics"]["full"]["score"] for r in rows])),
                            per_seed={str(r["task"]["seed"]): r["metrics"]["full"]["score"] for r in rows}))
    return dict(fits=checked, summary=summary, complete=complete)


def main():
    cmp = Comparisons()
    selection = aggregate_replays(cmp)
    learning = audit_learning(cmp)
    result = dict(status="PASS", audit_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  aggregate_groups=selection["groups"], labelled_window_diagnostic_fits=len(learning["fits"]),
                  independent_comparisons=cmp.elements, maximum_absolute_difference=cmp.maximum_error,
                  largest_difference_location=cmp.largest,
                  selected_candidate={k: selection["candidate"][k] for k in ("config_id", "score", "per_seed")},
                  selected_baseline={k: selection["baseline"][k] for k in ("scope", "config_id", "score")},
                  selected_patchtst={k: selection["patchtst"][k] for k in ("config_id", "score")},
                  learning_diagnostic_summary=learning["summary"],
                  development_aggregate_sha256=digest(ROOT / "DEVELOPMENT_AGGREGATE.json"),
                  learning_plan_sha256=digest(ROOT / "LEARNING_DIAGNOSTIC_PRE_RUN.json"),
                  audit_script_sha256=digest(Path(__file__)),
                  confirmation_file_reads=0, auditor_model_fits=0, auditor_optimizer_steps=0,
                  limitations=["The learning diagnostic has complete final receipts and a single-start guard, but no separate per-fit start journal.",
                               "Recent-tail caps jointly alter labelled-window count and history recency; spread caps preserve the same history endpoints but do not create independent observations.",
                               "The 24 diagnostic fits are development exploration and are excluded from frozen confirmation model selection."])
    (OUT / "AUDIT_DEVELOPMENT_SELECTION_AND_LEARNING.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (OUT / "RECOMPUTED_DEVELOPMENT_SELECTION_AND_LEARNING.json").write_text(json.dumps(dict(selection=selection, learning=learning), indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
