"""Frozen real-data experiment runner. One process, saved predictions and states."""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time
import traceback

import numpy as np
import torch

from data_core import load_market, pack, evaluation_pack, metrics, boundary_checks
from fit_core import fit, predict, serialize_state, moment_implementation_check
from statistical_reference import anchor_arrays


ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def get_markets():
    registry = json.loads((ROOT / "provenance" / "DEVELOPMENT_REGISTRY.json").read_text())
    markets = []
    for rec in registry:
        data = Path(rec["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != rec["sha256"]:
            raise ValueError("Changed development data: " + rec["name"])
        if "confirmation" in rec["path"]:
            raise ValueError("Confirmation data cannot be routed to this runner")
        markets.append(load_market(rec["path"], rec["name"], rec["price_column"], "2024-01-01"))
    return markets


def with_anchor(data, markets):
    raw = anchor_arrays(data, markets)
    data["stat_raw"] = raw
    data["stat_normalized"] = raw / data["scale"][:, None] / np.sqrt(np.arange(1, 21))
    return data


def init():
    plan_path = ROOT / "D1_PRE_RUN.json"
    if plan_path.exists():
        raise FileExistsError("Frozen D1 plan already exists")
    markets = get_markets()
    config_list = []
    families = ["gru_price", "gru", "gru_stat_price", "augment", "kcm", "lstm_price"]
    for family in families:
        for hidden in (16, 32):
            for shrink in (.1, 1.):
                strengths = (1., 10.) if family == "kcm" else (0.,)
                for strength in strengths:
                    config_list.append(dict(id=f"{family}_h{hidden}_s{shrink:g}_l{strength:g}",
                                            family=family, hidden=hidden, shrink=shrink,
                                            strength=strength, lr=.003))
    configs = list(config_list)
    for alpha in (.01, .1, 1.):
        configs.append(dict(id=f"ridge_a{alpha:g}", family="ridge", alpha=alpha))
    tasks = []
    for cutoff in ("2021-01-01", "2022-01-01", "2023-01-01"):
        for c in configs:
            tasks.append(dict(scope="pooled", market_names=[m.name for m in markets], cutoff=cutoff,
                              config=c, seed=17))
        for c in config_list:
            if c["family"] != "gru_price":
                continue
            for market in markets:
                tasks.append(dict(scope="local", market_names=[market.name], cutoff=cutoff,
                                  config=c, seed=17))
    # All candidates/year cuts are intermixed; a deadline cannot systematically
    # privilege a whole family by giving it all early time slots.
    random.Random(20261017).shuffle(tasks)
    for i, task in enumerate(tasks):
        task["id"] = f"d1_{i:03d}"
    plan = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), stage="D1",
                purpose="Bounded development, not independent confirmation",
                formal_predictor_fits=len(tasks), tasks=tasks, maximum_epochs=60,
                early_stopping_patience=12, initializer="training-only ridge head, all neural arms identical",
                validation_origins_per_market=80, train_validation_purge=20, train_cap=None,
                observation_horizons=[1, 5, 20], complete_same_calendar_year_H20=True,
                full_batch=True, process_count=1, math_threads=1, dtype="float64",
                deadline_seconds=2400, retry_failed_fits=False,
                development_end_exclusive="2024-01-01", confirmation_files_allowed=False,
                model_selection_score="equal market/year/horizon MSE relative to RW; all configured cells shown",
                mechanism_selection="compare KCM with its exactly matched augmentation arm and ERM; also compare with best strong baseline",
                no_publication_claim_from_D1=True,
                source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "source").glob("*.py")},
                registry_sha256=hashlib.sha256((ROOT / "provenance" / "DEVELOPMENT_REGISTRY.json").read_bytes()).hexdigest(),
                timing_plan_sha256=hashlib.sha256((ROOT / "TIMING_PLAN.json").read_bytes()).hexdigest())
    dump(plan_path, plan)
    print(json.dumps({"created_plan": str(plan_path), "fits": len(tasks), "seconds": 2400}), flush=True)


def run():
    plan_path = ROOT / "D1_PRE_RUN.json"
    plan = json.loads(plan_path.read_text())
    if (ROOT / "D1_STARTED.json").exists():
        raise FileExistsError("D1 is non-resumable; no silent reruns")
    for name, sha in plan["source_sha256"].items():
        if hashlib.sha256((ROOT / "source" / name).read_bytes()).hexdigest() != sha:
            raise ValueError("Frozen source changed: " + name)
    markets = get_markets()
    by_name = {m.name: m for m in markets}
    started = time.monotonic(); deadline = started + plan["deadline_seconds"]
    dump(ROOT / "D1_STARTED.json", dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),
         plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(), python=platform.python_version(),
         numpy=np.__version__, torch=torch.__version__, pid=os.getpid()))
    receipts = ROOT / "d1_receipts"; receipts.mkdir()
    saved = ROOT / "local_only" / "d1"; saved.mkdir(parents=True)
    caches = {}; results = []; controls = {}
    completed, failed = 0, []
    for task in plan["tasks"]:
        if time.monotonic() >= deadline:
            failed.append(dict(task=task["id"], reason="deadline before fit")); break
        key = (tuple(task["market_names"]), task["cutoff"])
        if key not in caches:
            chosen = [by_name[n] for n in task["market_names"]]
            train = with_anchor(pack(chosen, task["cutoff"], "train"), chosen)
            val = with_anchor(pack(chosen, task["cutoff"], "validation"), chosen)
            end = str(int(task["cutoff"][:4]) + 1) + "-01-01"
            evaluate = with_anchor(evaluation_pack(chosen, task["cutoff"], end, train["supports"]), chosen)
            caches[key] = (train, val, evaluate)
            controls[str(key)] = dict(rw=metrics(evaluate, np.zeros_like(evaluate["raw_y"])),
                                      stat=metrics(evaluate, evaluate["stat_raw"]))
        train, val, evaluate = caches[key]
        attempt = dict(task_id=task["id"], started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                       config=task["config"], scope=task["scope"], markets=task["market_names"], cutoff=task["cutoff"])
        with (ROOT / "D1_ATTEMPTS.jsonl").open("a") as f:
            f.write(json.dumps(attempt) + "\n")
        try:
            model, receipt = fit(train, val, task["config"], task["seed"],
                                 max_epochs=plan["maximum_epochs"], deadline=deadline)
            views = ["full"] if task["config"]["family"] == "gru_stat_price" else ["full", "short", "sparse"]
            forecasts = {v: predict(model, evaluate, v) for v in views}
            metric_set = {v: metrics(evaluate, pred, v) for v, pred in forecasts.items()}
            receipt.update(task=task, supports=train["supports"], development_metrics=metric_set,
                           fitting_code_sources=plan["source_sha256"],
                           confirmation_data_used=False, labels_accessed_only_after_predicted_origin_for_scoring=True)
            dump(receipts / (task["id"] + ".json"), receipt)
            np.savez_compressed(saved / (task["id"] + "_state.npz"), **serialize_state(model))
            np.savez_compressed(saved / (task["id"] + "_predictions.npz"),
                                **{"prediction_" + v: p for v, p in forecasts.items()},
                                raw_y=evaluate["raw_y"], origin=evaluate["origin"], dates=evaluate["dates"],
                                market=evaluate["market"], scale=evaluate["scale"],
                                market_names=np.array(evaluate["market_names"]), stat_raw=evaluate["stat_raw"])
            results.append(dict(task=task, metrics=metric_set, epochs=receipt["epochs"],
                                selected_epoch=receipt["selected_epoch"], seconds=receipt["seconds"]))
            completed += 1
            print(json.dumps({"done": completed, "total": len(plan["tasks"]), "id": task["id"],
                              "family": task["config"]["family"], "scope": task["scope"],
                              "cutoff": task["cutoff"], "score": metric_set["full"]["score"],
                              "epochs": receipt["epochs"], "seconds": receipt["seconds"]}), flush=True)
        except Exception as exc:
            error = dict(task=task, exception=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
            dump(receipts / (task["id"] + "_FAILED.json"), error)
            failed.append(error)
            print(json.dumps({"failed": task["id"], "error": str(exc)}), flush=True)
            break
    dump(ROOT / "D1_RESULTS.json", dict(results=results, controls=controls))
    dump(ROOT / "D1_RUN_COMPLETE.json", dict(completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
         expected_fits=len(plan["tasks"]), completed_fits=completed, failures=failed,
         elapsed_seconds=time.monotonic() - started, all_completed=(completed == len(plan["tasks"])),
         confirmation_used=False, additional_retry_fits=0))
    if failed:
        raise SystemExit(2)


def summarize():
    bundle = json.loads((ROOT / "D1_RESULTS.json").read_text())
    rows = bundle["results"]
    groups = {}
    for r in rows:
        task = r["task"]
        key = (task["scope"], task["config"]["id"])
        groups.setdefault(key, []).append(r)
    summary = []
    for (scope, config_id), members in groups.items():
        cells = [c for r in members for c in r["metrics"]["full"]["cells"]]
        summary.append(dict(scope=scope, config_id=config_id, family=members[0]["task"]["config"]["family"],
                            fits=len(members), cells=len(cells), score=float(np.mean([c["relative_mse"] for c in cells])),
                            market_scores={m: float(np.mean([c["relative_mse"] for c in cells if c["market"] == m]))
                                           for m in sorted({c["market"] for c in cells})},
                            selected_epoch_zero=sum(r["selected_epoch"] == 0 for r in members)))
    summary.sort(key=lambda r: r["score"])
    dump(ROOT / "D1_SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["init", "run", "summarize", "check"])
    args = parser.parse_args()
    if args.command == "init": init()
    elif args.command == "run": run()
    elif args.command == "summarize": summarize()
    else:
        out = dict(data=boundary_checks(), moments=moment_implementation_check())
        dump(ROOT / "IMPLEMENTATION_CHECK.json", out); print(json.dumps(out))
