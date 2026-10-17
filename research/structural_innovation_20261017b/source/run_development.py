"""Execute exactly the frozen, bounded SMP development job list.

Every attempted fit has a public start/failure/complete receipt. Point arrays
and weights are always local_only. D1 selection must be frozen before D2.
No confirmation file is discovered or opened by this program.
"""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from data import AnnualData, load_market, metrics
from fit import fit_neural, predict, state_arrays
from models import FAMILIES
from stat_baselines import fit_ridge, predict_ridge, fit_markov_mle, predict_markov_mle
from statistical_reference import reference


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError("Refuse overwriting an experiment receipt: " + str(path))
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def config_name(family, value=None):
    return family if value is None else family + "__a" + str(float(value)).replace(".", "p")


def jobs_from_protocol(protocol, phase, selected=None):
    jobs = []
    year = 2022 if phase == "D1" else 2023
    if phase == "D2" and (selected is None or not selected["advance_to_D2"]):
        raise RuntimeError("D2 requires an eligible fixed D1 selection")
    for market in protocol["markets"]:
        for family in FAMILIES:
            if phase == "D1":
                values = protocol["shrink_candidates"]
            else:
                values = sorted(set([selected["selected"][family]["value"], selected["selected"]["smp_full"]["value"]]))
            for value in values:
                for seed in protocol["D1_seeds"] if phase == "D1" else protocol["D2_seeds"]:
                    jobs.append(dict(kind="neural", family=family, value=value, seed=seed, year=year, market=market))
        # The training-validation-selected ridge is a separate, causal stronger
        # reference. All five values are fitted both years, with no D2 reselection
        # on evaluation errors; the global D1 choice is also retained separately.
        for value in protocol["ridge_alpha_candidates"]:
            jobs.append(dict(kind="ridge", family="ridge", value=value, seed=0, year=year, market=market))
        for family in ("markov_mle", "project_reference", "rw"):
            jobs.append(dict(kind=family, family=family, value=None, seed=0, year=year, market=market))
    for job in jobs:
        job["config_id"] = config_name(job["family"], job["value"])
        job["id"] = f"{job['market']['name']}__{job['config_id']}__s{job['seed']}"
    return jobs


def verify_sources(root, protocol):
    for relative, expected in protocol["source_sha256"].items():
        if sha(root / relative) != expected:
            raise RuntimeError("Frozen executable bytes changed: " + relative)


def execute_job(job, root_string, phase, protocol_path, deadline):
    root = Path(root_string)
    protocol = json.loads(Path(protocol_path).read_text())
    verify_sources(root, protocol)
    public = root / "results" / phase / "jobs" / job["id"]
    private = root / "local_only" / phase / job["id"]
    started = time.perf_counter()
    save_json(public / "STARTED.json", dict(created_utc=utc(), job={k:v for k,v in job.items() if k != "market"},
                                            market=job["market"]["name"], protocol_sha256=sha(protocol_path)))
    try:
        if time.monotonic() >= deadline:
            raise TimeoutError("Stage deadline reached before job")
        market = load_market(job["market"], protocol["data_end_exclusive"])
        data = AnnualData(market, job["year"], protocol["validation_size"])
        train = data.batch(data.train, data.start - 1)
        val = data.batch(data.validation, data.start - 1)
        # This input view contains no target array. Test truth is constructed only
        # after the frozen predictor has issued its complete forecast vector.
        query = data.batch(data.test, data.end - 1, labels=False)
        save_json(public / "SUPPORT.json", data.support_receipt())
        private.mkdir(parents=True, exist_ok=True)
        if job["kind"] == "neural":
            net, fit_receipt = fit_neural(train, val, job["family"], job["value"], job["seed"],
                                           max_epochs=protocol["max_epochs"], patience=protocol["patience"],
                                           batch_size=protocol["batch_size"], deadline=deadline)
            save_json(public / "FIT.json", fit_receipt)
            if not fit_receipt["eligible_for_selection"]:
                raise TimeoutError("Incomplete fit cannot enter selection")
            normalized = predict(net, query)
            np.savez_compressed(private / "weights.npz", **state_arrays(net))
            validation_score = fit_receipt["selected_validation_score"]
        elif job["kind"] == "ridge":
            model = fit_ridge(train, job["value"])
            normalized = predict_ridge(model, query)
            validation_score = metrics(predict_ridge(model, val) * val["scale"][:, None], val)["score"]
            np.savez_compressed(private / "weights.npz", feature_mean=model.feature_mean,
                                feature_std=model.feature_std, coefficients=model.coefficients)
            save_json(public / "FIT.json", dict(status="completed", eligible_for_selection=True,
                                                 kind="closed_form_weighted_ridge", alpha=job["value"],
                                                 selected_validation_score=validation_score,
                                                 parameters=int(model.coefficients.size), optimizer_steps=0))
        elif job["kind"] == "markov_mle":
            model = fit_markov_mle(train)
            normalized = predict_markov_mle(model, query)
            validation_score = metrics(predict_markov_mle(model, val) * val["scale"][:, None], val)["score"]
            np.savez_compressed(private / "weights.npz", q=model[0], b=model[1])
            save_json(public / "FIT.json", dict(status="completed", eligible_for_selection=True,
                                                 kind="smoothed_Markov_counts_and_marks", parameters=45,
                                                 selected_validation_score=validation_score, optimizer_steps=0))
        elif job["kind"] == "project_reference":
            # Existing causal filter; no new optimized fit. Its longer historical
            # state/window is explicitly supplemental, not a one-factor control.
            changes, _ = reference(market.price[:int(data.test[-1]) + 1])
            normalized = changes[data.test] / query["scale"][:, None]
            validation_score = None
        elif job["kind"] == "rw":
            normalized = np.zeros((len(data.test), 20))
            validation_score = 1.
        else:
            raise ValueError("Unknown kind")
        raw_forecast = normalized * query["scale"][:, None]
        # Evaluation labels are opened only here, after all training and issuance.
        truth = data.batch(data.test, data.end - 1, labels=True)
        result = metrics(raw_forecast, truth)
        np.savez_compressed(private / "forecasts.npz", origin=query["origin"],
                            date=market.dates[query["origin"]], base=query["base"],
                            scale=query["scale"], state=query["state"],
                            predicted_change=raw_forecast, target_change=truth["delta"])
        result.update({k:job[k] for k in ("id", "config_id", "family", "kind", "seed", "year", "value")})
        result.update(market=market.name, phase=phase, selected_validation_score=validation_score,
                      created_utc=utc(), protocol_sha256=sha(protocol_path),
                      private_forecast_sha256=sha(private / "forecasts.npz"),
                      wall_seconds=time.perf_counter() - started, status="completed")
        save_json(public / "METRICS.json", result)
        return dict(id=job["id"], status="completed", seconds=result["wall_seconds"])
    except Exception as exc:
        failure = dict(id=job["id"], status="failed", created_utc=utc(),
                       error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(),
                       wall_seconds=time.perf_counter() - started)
        save_json(public / "FAILED.json", failure)
        return failure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--phase", choices=("D1", "D2"), required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--data-map", type=Path, help="JSON {market_name: local_csv_path}; bytes must still match the frozen SHA256")
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    root = protocol_path.parent.parent
    protocol = json.loads(protocol_path.read_text())
    verify_sources(root, protocol)
    if args.data_map:
        mapping = json.loads(args.data_map.read_text())
        expected_names = {m["name"] for m in protocol["markets"]}
        if set(mapping) != expected_names or any(not isinstance(p, str) for p in mapping.values()):
            raise ValueError("Data map must name exactly the five frozen series")
        protocol["markets"] = [dict(m, path=str(Path(mapping[m["name"]]).expanduser().resolve())) for m in protocol["markets"]]
    selected = json.loads(args.selection.read_text()) if args.selection else None
    if selected is not None and selected["protocol_sha256"] != sha(protocol_path):
        raise RuntimeError("Selection belongs to a different protocol")
    jobs = jobs_from_protocol(protocol, args.phase, selected)
    counts = {kind:sum(j["kind"] == kind for j in jobs) for kind in sorted({j["kind"] for j in jobs})}
    expected = protocol["D1_exact_job_counts"] if args.phase == "D1" else protocol["D2_max_job_counts"]
    if any((counts.get(k, 0) != v if args.phase == "D1" else counts.get(k, 0) > v) for k,v in expected.items()):
        raise RuntimeError("Generated jobs differ from the frozen budget")
    output = root / "results" / args.phase
    started = time.perf_counter()
    start_record = dict(created_utc=utc(), phase=args.phase, protocol_sha256=sha(protocol_path),
                        selection_sha256=sha(args.selection) if args.selection else None,
                        data_map_sha256=sha(args.data_map) if args.data_map else None,
                        job_counts=counts, total_jobs=len(jobs), processes=protocol["cpu_processes"],
                        threads_per_process=1, deadline_seconds=protocol[args.phase + "_deadline_seconds"],
                        job_ids=[j["id"] for j in jobs])
    save_json(output / "STARTED.json", start_record)
    deadline = time.monotonic() + protocol[args.phase + "_deadline_seconds"]
    rows = []
    with ProcessPoolExecutor(max_workers=protocol["cpu_processes"]) as pool:
        pending = {pool.submit(execute_job, job, str(root), args.phase, str(protocol_path), deadline):job for job in jobs}
        for future in as_completed(pending):
            row = future.result()
            rows.append(row)
            if row["status"] != "completed" or len(rows) % 10 == 0:
                print(json.dumps({"phase":args.phase,"completed_attempts":len(rows),"total":len(jobs),
                                  "last_id":row["id"],"last_status":row["status"],
                                  "wall_seconds":time.perf_counter()-started}), flush=True)
    summary = dict(created_utc=utc(), phase=args.phase, protocol_sha256=sha(protocol_path),
                   wall_seconds=time.perf_counter()-started, attempted_jobs=len(rows),
                   completed_jobs=sum(r["status"] == "completed" for r in rows),
                   failed_jobs=sum(r["status"] != "completed" for r in rows),
                   actual_counts=counts, jobs=sorted(rows,key=lambda r:r["id"]))
    save_json(output / "COMPLETED.json", summary)
    print(json.dumps({k:v for k,v in summary.items() if k != "jobs"}), flush=True)
    if summary["failed_jobs"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
