"""Frozen development selection; no test-year hindsight choice or rescue grid."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from models import FAMILIES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, obj):
    if path.exists():
        raise RuntimeError("Refuse replacing an existing selection: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def load_results(root, phase):
    final = json.loads((root / "results" / phase / "COMPLETED.json").read_text())
    if final["failed_jobs"] or final["completed_jobs"] != final["attempted_jobs"]:
        raise RuntimeError("An incomplete stage cannot be selected")
    files = sorted((root / "results" / phase / "jobs").glob("*/METRICS.json"))
    rows = [json.loads(p.read_text()) for p in files]
    if len(rows) != final["completed_jobs"] or any(r["status"] != "completed" for r in rows):
        raise RuntimeError("Metrics inventory does not match actual completed jobs")
    return rows


def aggregate(rows, markets):
    groups = defaultdict(list)
    for row in rows:
        groups[row["config_id"]].append(row)
    output = []
    for config_id, members in sorted(groups.items()):
        by_market = {}
        for market in markets:
            sub = [r for r in members if r["market"] == market]
            if not sub:
                raise RuntimeError("Missing market in configuration " + config_id)
            relative = np.mean([r["relative_mse"] for r in sub], axis=0)
            by_market[market] = dict(score=float(relative[[0, 4, 19]].mean()), relative_mse=relative.tolist(),
                                     n=sub[0]["n"], seeds=sorted(r["seed"] for r in sub),
                                     job_ids=sorted(r["id"] for r in sub))
        output.append(dict(config_id=config_id, family=members[0]["family"], value=members[0]["value"],
                           score=float(np.mean([r["score"] for r in by_market.values()])),
                           by_market=by_market, aggregation="equal markets; equal h1/5/20; average seed risks, not predictions"))
    # A deployable local choice based exclusively on each year's pre-cutoff val.
    ridge_members = [r for r in rows if r["family"] == "ridge"]
    chosen = []
    for market in markets:
        sub = [r for r in ridge_members if r["market"] == market]
        chosen.append(min(sub, key=lambda r:(r["selected_validation_score"], -r["value"])))
    by_market = {r["market"]:dict(score=r["score"], relative_mse=r["relative_mse"], n=r["n"],
                                      seeds=[0], job_ids=[r["id"]], selected_value=r["value"],
                                      selected_validation_score=r["selected_validation_score"]) for r in chosen}
    output.append(dict(config_id="ridge_val", family="ridge_val", value=None,
                       score=float(np.mean([r["score"] for r in chosen])), by_market=by_market,
                       aggregation="local pre-cutoff validation selection; equal markets and score horizons"))
    return output


def choose_d1(root, protocol_path):
    protocol = json.loads(protocol_path.read_text())
    rows = load_results(root, "D1")
    if any(r["year"] != 2022 for r in rows):
        raise RuntimeError("D1 selection may use 2022 only")
    markets = [m["name"] for m in protocol["markets"]]
    table = aggregate(rows, markets)
    chosen = {}
    for family in list(FAMILIES) + ["ridge", "ridge_val", "markov_mle", "project_reference", "rw"]:
        candidates = [r for r in table if r["family"] == family]
        chosen[family] = min(candidates, key=lambda r:(r["score"], -(r["value"] or 0.)))
    full = chosen["smp_full"]
    matched = {family:next(r for r in table if r["family"] == family and r["value"] == full["value"])
               for family in FAMILIES}
    strong = min((chosen[f] for f in protocol["primary_baseline_families"]), key=lambda r:(r["score"],r["family"]))
    gates = dict(better_than_rw=full["score"] < 1.,
                 better_than_matched_iid=full["score"] < matched["smp_iid_state"]["score"],
                 better_than_matched_one_step=full["score"] < matched["smp_one_step"]["score"],
                 within_practical_margin_of_best_baseline=full["score"] <= protocol["D1_baseline_max_ratio"] * strong["score"])
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), stage="D1", year=2022,
                   protocol_sha256=sha(protocol_path), selected=chosen, matched_to_smp=matched,
                   fixed_best_non_smp_baseline=strong, gate_checks=gates,
                   advance_to_D2=all(gates.values()),
                   gate_interpretation="Prespecified resource screen, not journal acceptance or statistical proof",
                   confirmation_access=False, selection_ties="larger regularization; alphabetical family for baseline tie")
    write_new(root / "results" / "D1_ALL_CONFIGS.json", {"rows":table,"protocol_sha256":sha(protocol_path)})
    write_new(root / "results" / "D1_SELECTION.json", result)
    print(json.dumps({"stage":"D1","smp":full["score"],"selected_shrink":full["value"],
                      "best_baseline":strong["family"],"best_baseline_score":strong["score"],
                      "gates":gates,"advance_to_D2":result["advance_to_D2"]}))


def assess_d2(root, protocol_path, d1_path):
    protocol = json.loads(protocol_path.read_text())
    d1 = json.loads(d1_path.read_text())
    if not d1["advance_to_D2"] or d1["protocol_sha256"] != sha(protocol_path):
        raise RuntimeError("D2 lacks an eligible unchanged D1 selection")
    rows = load_results(root, "D2")
    if any(r["year"] != 2023 for r in rows):
        raise RuntimeError("D2 assessment requires 2023 only")
    markets = [m["name"] for m in protocol["markets"]]
    table = aggregate(rows, markets)
    by_id = {r["config_id"]:r for r in table}
    chosen = {family:by_id[row["config_id"]] for family,row in d1["selected"].items()}
    matched = {family:by_id[row["config_id"]] for family,row in d1["matched_to_smp"].items()}
    full = chosen["smp_full"]
    best = min((chosen[f] for f in protocol["primary_baseline_families"]), key=lambda r:(r["score"],r["family"]))
    seed_scores = {str(seed):float(np.mean([r["score"] for r in rows if r["family"] == "smp_full" and r["value"] == full["value"] and r["seed"] == seed])) for seed in protocol["D2_seeds"]}
    gate = dict(practical_rw_improvement=full["score"] <= protocol["D2_rw_max_score"],
                better_than_all_selected_primary_baselines=full["score"] < best["score"],
                EEX_better_than_rw=full["by_market"]["EEX_EUA_T3PA"]["score"] < 1.,
                better_than_matched_iid=full["score"] < matched["smp_iid_state"]["score"],
                better_than_matched_one_step=full["score"] < matched["smp_one_step"]["score"],
                at_least_three_series_better_than_rw=sum(v["score"] < 1. for v in full["by_market"].values()) >= 3,
                at_least_two_seeds_better_than_rw=sum(v < 1. for v in seed_scores.values()) >= 2)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), stage="D2", year=2023,
                   protocol_sha256=sha(protocol_path), d1_selection_sha256=sha(d1_path),
                   selected=chosen, matched_to_smp=matched, seed_scores=seed_scores,
                   gate_checks=gate, point_gate_pass=all(gate.values()),
                   fixed_best_non_smp_baseline=chosen[d1["fixed_best_non_smp_baseline"]["family"]],
                   best_baseline_in_D2_for_descriptive_gate=best,
                   confirmation_ready=False,
                   confirmation_note="Requires independent metric audit, prespecified block uncertainty checks, and separately frozen C protocol before any C use",
                   exposure_note="2022/2023 are already exposed development periods; D2 is not an untouched confirmation test")
    write_new(root / "results" / "D2_ALL_CONFIGS.json", {"rows":table,"protocol_sha256":sha(protocol_path)})
    write_new(root / "results" / "D2_ASSESSMENT.json", result)
    print(json.dumps({"stage":"D2","smp":full["score"],"best_baseline":best["family"],
                      "best_baseline_score":best["score"],"gates":gate,"point_gate_pass":result["point_gate_pass"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--phase", choices=("D1", "D2"), required=True)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    root = args.protocol.resolve().parent.parent
    if args.phase == "D1":
        choose_d1(root, args.protocol.resolve())
    else:
        if args.selection is None:
            raise ValueError("D2 requires --selection")
        assess_d2(root, args.protocol.resolve(), args.selection.resolve())
