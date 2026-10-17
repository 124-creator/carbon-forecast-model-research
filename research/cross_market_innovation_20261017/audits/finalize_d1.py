"""Finalize the completed independent D1 replay without repeating model forwards.

Reads frozen development receipts and saved development predictions only. No
confirmation data, training framework, optimizer, or new model fit is used.
"""
from __future__ import annotations

import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from verify_d1 import ROOT, OUT, Comparisons, cell_metrics, digest, metric_check


def main():
    partial_path = OUT / "AUDIT_D1_PARTIAL_141.json"
    replay_path = OUT / "RECOMPUTED_D1_PARTIAL_141.json"
    previous = json.loads(partial_path.read_text())
    replay = json.loads(replay_path.read_text())
    plan = json.loads((ROOT / "D1_PRE_RUN.json").read_text())
    done = json.loads((ROOT / "D1_RUN_COMPLETE.json").read_text())
    public = json.loads((ROOT / "D1_RESULTS.json").read_text())
    summary = json.loads((ROOT / "D1_SUMMARY.json").read_text())
    attempts = [json.loads(x) for x in (ROOT / "D1_ATTEMPTS.jsonl").read_text().splitlines()]
    assert previous["audited_fits"] == len(replay) == len(public["results"]) == 141
    assert done["all_completed"] and done["completed_fits"] == done["expected_fits"] == 141
    assert not done["failures"] and done["additional_retry_fits"] == 0
    assert not done["confirmation_used"] and done["elapsed_seconds"] <= plan["deadline_seconds"]
    assert len(attempts) == 141 and [x["task_id"] for x in attempts] == [x["id"] for x in plan["tasks"]]
    assert previous["plan_sha256"] == digest(ROOT / "D1_PRE_RUN.json")
    assert previous["audit_script_sha256"] == digest(Path(__file__).with_name("verify_d1.py"))
    for name, sha in previous["source_sha256"].items():
        assert digest(ROOT / "frozen_d1_source" / name) == sha
    assert digest(ROOT / "provenance/DEVELOPMENT_REGISTRY.json") == plan["registry_sha256"]
    cmp = Comparisons()
    grouped, controls = defaultdict(list), {}
    for checked, result in zip(replay, public["results"]):
        tid, task = checked["id"], checked["task"]
        assert result["task"] == task
        paths = {
            "receipt_sha256": ROOT / "d1_receipts" / (tid + ".json"),
            "state_file_sha256": ROOT / "local_only/d1" / (tid + "_state.npz"),
            "predictions_file_sha256": ROOT / "local_only/d1" / (tid + "_predictions.npz"),
        }
        for field, path in paths.items():
            assert digest(path) == checked[field], (tid, field)
        receipt = json.loads(paths["receipt_sha256"].read_text())
        for field in ("epochs", "selected_epoch"):
            assert checked[field] == result[field] == receipt[field]
        cmp.close(receipt["seconds"], result["seconds"], tid + "/public_seconds")
        for view, metrics in checked["metrics"].items():
            metric_check(cmp, metrics, result["metrics"][view], tid + "/public/" + view)
        grouped[(task["scope"], task["config"]["id"])].append(checked)
        key = str((tuple(task["market_names"]), task["cutoff"]))
        if key not in controls:
            with np.load(paths["predictions_file_sha256"], allow_pickle=False) as pack:
                data = {k: pack[k] for k in ("market_names", "market", "dates", "raw_y", "stat_raw")}
            assert np.all(data["dates"] < "2024-01-01")
            controls[key] = {
                "rw": cell_metrics(data, np.zeros_like(data["raw_y"]), "full"),
                "stat": cell_metrics(data, data["stat_raw"], "full"),
            }
            for name, metric in controls[key].items():
                metric_check(cmp, metric, public["controls"][key][name], key + "/" + name)
    assert set(controls) == set(public["controls"])
    recomputed_summary = []
    for (scope, config_id), rows in grouped.items():
        cells = [c for row in rows for c in row["metrics"]["full"]["cells"]]
        assert len(cells) == 36
        assert len({(c["market"], c["year"], c["horizon"]) for c in cells}) == 36
        recomputed_summary.append({
            "scope": scope, "config_id": config_id,
            "family": rows[0]["task"]["config"]["family"],
            "fits": len(rows), "cells": len(cells),
            "score": float(np.mean([c["relative_mse"] for c in cells])),
            "market_scores": {m: float(np.mean([c["relative_mse"] for c in cells if c["market"] == m]))
                              for m in sorted({c["market"] for c in cells})},
            "selected_epoch_zero": sum(row["selected_epoch"] == 0 for row in rows),
        })
    recomputed_summary.sort(key=lambda x: x["score"])
    assert len(summary) == len(recomputed_summary) == 35
    for expected, saved in zip(recomputed_summary, summary):
        for field in ("scope", "config_id", "family", "fits", "cells", "selected_epoch_zero"):
            assert expected[field] == saved[field], (field, expected, saved)
        cmp.close(expected["score"], saved["score"], "summary/" + expected["config_id"])
        for name, value in expected["market_scores"].items():
            cmp.close(value, saved["market_scores"][name], "summary/" + expected["config_id"] + "/" + name)
    pooled_keys = [key for key in controls if key.startswith("(('GUANGZHOU', 'HUBEI', 'EU_REPORTED', 'SHANGHAI_D4')")]
    assert len(pooled_keys) == 3
    control_scores = {
        name: float(np.mean([c["relative_mse"] for key in pooled_keys for c in controls[key][name]["cells"]]))
        for name in ("rw", "stat")
    }
    scores = {(r["scope"], r["config_id"]): r["score"] for r in recomputed_summary}
    kcm = scores[("pooled", "kcm_h32_s1_l1")]
    augment = scores[("pooled", "augment_h32_s1_l0")]
    strongest = min(recomputed_summary, key=lambda x: x["score"])
    out = dict(previous)
    out.update({
        "status": "PASS", "audit_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "completed_receipt": done,
        "finalization_method": "Authenticate all 141 already-replayed files, then independently aggregate published results and controls; no repeat model forwards.",
        "earlier_replay_report_sha256": digest(partial_path),
        "earlier_recomputed_results_sha256": digest(replay_path),
        "finalizer_sha256": digest(Path(__file__)),
        "finalization_additional_numeric_checks": cmp.elements,
        "independent_checks": previous["independent_checks"] + cmp.elements,
        "finalization_maximum_absolute_difference": cmp.maximum_error,
        "public_results_sha256": digest(ROOT / "D1_RESULTS.json"),
        "public_summary_sha256": digest(ROOT / "D1_SUMMARY.json"),
        "public_summary_groups": len(summary),
        "public_control_supports": len(controls),
        "control_scores": control_scores,
        "descriptive_findings": {
            "strongest_trained_development_baseline": strongest,
            "selected_kcm_score": kcm, "matched_augmentation_score": augment,
            "kcm_minus_matched_augmentation": kcm - augment,
            "kcm_minus_strongest_trained_baseline": kcm - strongest["score"],
            "single_seed_screen_is_confirmatory": False,
        },
        "documentation_errata": [
            "Frozen data_core.py top docstring says market IDs are not neural inputs. Actual fit_core appends known market one-hot IDs at every valid token for all neural methods; this actual behavior was independently replayed. Frozen source is preserved; D2 may correct documentation explicitly."
        ],
    })
    out["checks"] += ["all public D1 results", "35 public model summaries", "15 RW/statistical control supports", "141 completed attempts within frozen time budget"]
    (OUT / "AUDIT_D1_COMPLETE.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    (OUT / "RECOMPUTED_D1_COMPLETE.json").write_text(json.dumps({"fits": replay, "summary": recomputed_summary, "controls": controls}, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: out[k] for k in ("status", "audited_fits", "independent_checks", "maximum_absolute_difference", "control_scores", "descriptive_findings")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
