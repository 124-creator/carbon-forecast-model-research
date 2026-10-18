"""Replay all completed neural checkpoints without computing market scores.

Only hashes, counts, identifiers, and maximum numerical discrepancies are
published. Prices, forecasts, residuals, samples, and origin-level values are
never written to the public audit. Checkpoints are the locally generated,
receipt-verified files from this experiment, not arbitrary external pickles.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for _thread_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_name] = "1"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pipeline as P
import model as M
import run as R

SPECS = {"quantum": (("Q1",1000,1250),("Q2",1250,1500),("Q3",1500,1785)),
         "ding": (("D1",386,516),("D2",516,644),("D3",644,752))}
EXPECTED_NEURAL_CHECKPOINTS = 126
EXPECTED_FORECAST_FILES = 156
STATE_KEYS = ("xmean","xstd","cmean","cstd","denominator")
ABSOLUTE_TOLERANCE = 1e-10
RELATIVE_TOLERANCE = 2e-12


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def comparison(actual, expected, label):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape, label + ": shape mismatch")
    require(np.isfinite(a).all() and np.isfinite(b).all(), label + ": nonfinite value")
    maximum = float(np.abs(a-b).max()) if a.size else 0.
    require(np.allclose(a,b,atol=ABSOLUTE_TOLERANCE,rtol=RELATIVE_TOLERANCE),
            label + ": numerical mismatch, maximum absolute difference=" + str(maximum))
    return maximum


def completed_run_gate(run_root, fingerprint):
    """Require every task/file/manifest before any observation is loaded."""
    plan = read_json(run_root / "plan.json")
    require(plan["code_fingerprint"] == fingerprint, "Run plan code fingerprint mismatch")
    require(tuple(plan["neural_families"]) == R.NEURAL, "Neural family manifest mismatch")
    require(tuple(plan["classical_families"]) == R.CLASSICAL, "Classical family manifest mismatch")
    require(tuple(plan["seeds"]) == R.SEEDS, "Seed manifest mismatch")
    receipts = read_json(run_root / "all_fit_receipts.json")
    require(len(receipts) == 72, "All 72 task receipts must exist before replay")
    identities = {(r["source"],r["fold"],r["family"]) for r in receipts}
    expected = {(source,fold,family) for source,specs in SPECS.items()
                for fold,_,_ in specs for family in R.NEURAL + R.CLASSICAL}
    require(identities == expected, "Task completion identities differ from full plan")
    for source,specs in SPECS.items():
        for fold,start,end in specs:
            manifest = read_json(run_root/source/fold/"manifest.json")
            for key,value in {"source":source,"fold":fold,"test_start":start,"test_end":end,
                              "code_fingerprint":fingerprint}.items():
                require(manifest[key] == value, "Fold completion manifest mismatch: " + key)
            for family in R.NEURAL + R.CLASSICAL:
                folder = run_root/source/fold/family
                complete = read_json(folder/"complete.json")
                require(complete["code_fingerprint"] == fingerprint, "Task code mismatch")
                for seed in R.SEEDS if family in R.NEURAL else (0,):
                    require((folder/f"seed_{seed}.npz").is_file(), "Missing completed forecast")
                    require((folder/f"seed_{seed}.json").is_file(), "Missing forecast receipt")
                    if family in R.NEURAL:
                        require((folder/f"seed_{seed}.pt").is_file(), "Missing completed checkpoint")
    require(len(list(run_root.glob("*/*/*/seed_*.pt"))) == EXPECTED_NEURAL_CHECKPOINTS,
            "Unexpected checkpoint count")
    require(len(list(run_root.glob("*/*/*/seed_*.npz"))) == EXPECTED_FORECAST_FILES,
            "Unexpected forecast file count")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",type=Path,default=Path("/tmp/carbon-path-local"))
    parser.add_argument("--run-dir",type=Path,default=Path("/tmp/carbon-path-run"))
    parser.add_argument("--output",type=Path,default=ROOT/"audits"/"REPLAY_AUDIT.json")
    args = parser.parse_args()
    start_time = time.perf_counter()
    fingerprint = R.fingerprint()
    completed_run_gate(args.run_dir,fingerprint)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows = []
    source_hashes = {}
    for source,specs in SPECS.items():
        source_path = args.data_dir/f"{source}.csv"
        source_hashes[source] = digest(source_path)
        frame = P.load_source(source_path,source)
        require(tuple(P.folds_for(frame,source)) == specs, "Recomputed fold definitions differ")
        data = P.OriginData(frame)
        for fold,a,b in specs:
            final = data.positions(P.FIRST_ORIGIN,a-20)
            require(int(data.origin[final].max())+20 < a, "Immature training labels")
            prep = P.Prepared.fit(data,final)
            expected_state = prep.state()
            offsets = np.array([0,(b-a)//2,b-a-1],dtype=int)
            probes = data.positions(a,b)[offsets]
            require(len(np.unique(probes)) == 3, "Replay probes are not unique")
            for family in R.NEURAL:
                folder = args.run_dir/source/fold/family
                complete = read_json(folder/"complete.json")
                require(complete["data_sha256"] == source_hashes[source], "Task source hash mismatch")
                for seed in R.SEEDS:
                    checkpoint_path = folder/f"seed_{seed}.pt"
                    forecast_path = folder/f"seed_{seed}.npz"
                    receipt_path = folder/f"seed_{seed}.json"
                    metadata = read_json(receipt_path)
                    checkpoint_hash = digest(checkpoint_path)
                    forecast_hash = digest(forecast_path)
                    require(checkpoint_hash == metadata["model_sha256"], "Checkpoint hash differs from receipt")
                    require(forecast_hash == metadata["forecast_sha256"], "Forecast hash differs from receipt")
                    expected_metadata = {"source":source,"fold":fold,"family":family,"seed":seed,
                        "test_start":a,"test_end":b,"data_sha256":source_hashes[source],
                        "code_fingerprint":fingerprint,"train_origin_start":P.FIRST_ORIGIN,
                        "train_origin_end_exclusive":a-20,"train_max_label_index":a-1,
                        "numerical_clip_count":0,"forecast_rows":b-a}
                    for key,value in expected_metadata.items():
                        require(metadata.get(key) == value, "Checkpoint metadata mismatch: " + key)
                    # Loading is restricted to our hash-verified local experiment files.
                    checkpoint = torch.load(checkpoint_path,map_location="cpu",weights_only=False)
                    for key,value in {"family":family,"data_sha256":source_hashes[source],
                                      "code_fingerprint":fingerprint}.items():
                        require(checkpoint.get(key) == value, "Checkpoint payload mismatch: " + key)
                    require(set(checkpoint["preprocessing"]) == set(STATE_KEYS), "Unexpected preprocessing state")
                    state_differences = {key:comparison(checkpoint["preprocessing"][key],expected_state[key],
                                            "Training-only preprocessing "+key) for key in STATE_KEYS}
                    replay_model = M.make_model(family)
                    replay_model.load_state_dict(checkpoint["model_state"],strict=True)
                    replay_model.eval()
                    with torch.no_grad():
                        output = R.forward(replay_model,prep.tensors(probes),M.sobol_normal(128,908113))
                    R.ensure_finite(output)
                    replay_mean = output["mean"].numpy()
                    replay_samples = (output["samples"].numpy() if output["samples"] is not None
                                      else np.empty((3,0,3,3)))
                    clip_count = int(output["clip_count"])
                    require(clip_count == 0, "Numerical guard triggered during replay")
                    with np.load(forecast_path,allow_pickle=False) as archive:
                        require(np.array_equal(archive["origin"][offsets],data.origin[probes]),
                                "First/middle/last replay origins differ from saved origins")
                        mean_difference = comparison(replay_mean,archive["mean"][offsets],"Saved analytical mean")
                        sample_difference = comparison(replay_samples,archive["samples"][offsets],"Saved 128-point summary sample")
                        den_difference = comparison(expected_state["denominator"],archive["denominator"],"Saved denominator")
                    rows.append({"source":source,"fold":fold,"family":family,"seed":seed,
                        "probe_count":3,"sample_count_per_probe":int(replay_samples.shape[1]),
                        "analytical_mean_values_compared":int(replay_mean.size),
                        "summary_sample_values_compared":int(replay_samples.size),
                        "maximum_absolute_mean_difference":mean_difference,
                        "maximum_absolute_sample_difference":sample_difference,
                        "maximum_absolute_preprocessing_difference":max(state_differences.values()),
                        "maximum_absolute_denominator_difference":den_difference,
                        "numerical_clip_count":clip_count,"checkpoint_sha256":checkpoint_hash,
                        "forecast_sha256":forecast_hash,"receipt_sha256":digest(receipt_path)})
            print(json.dumps({"fold_replayed":source+"/"+fold,"checkpoints_completed":len(rows)}),flush=True)
    require(len(rows) == EXPECTED_NEURAL_CHECKPOINTS, "Incomplete checkpoint replay")
    report = {"status":"PASS","market_scores_computed":False,"code_fingerprint":fingerprint,
        "audit_script_sha256":digest(__file__),"source_sha256":source_hashes,
        "full_run_gate":{"task_receipts":72,"forecast_files":EXPECTED_FORECAST_FILES,
                         "checkpoint_files":EXPECTED_NEURAL_CHECKPOINTS},
        "probe_rule":"First, floor(N/2), and last origin within each fixed test fold",
        "checkpoint_count":len(rows),"origin_replays":sum(r["probe_count"] for r in rows),
        "analytical_mean_values_compared":sum(r["analytical_mean_values_compared"] for r in rows),
        "summary_sample_values_compared":sum(r["summary_sample_values_compared"] for r in rows),
        "numerical_clip_count":sum(r["numerical_clip_count"] for r in rows),
        "maximum_absolute_mean_difference":max(r["maximum_absolute_mean_difference"] for r in rows),
        "maximum_absolute_sample_difference":max(r["maximum_absolute_sample_difference"] for r in rows),
        "maximum_absolute_preprocessing_difference":max(r["maximum_absolute_preprocessing_difference"] for r in rows),
        "tolerances":{"absolute":ABSOLUTE_TOLERANCE,"relative":RELATIVE_TOLERANCE},
        "preprocessing_recomputed_from_matured_final_training_pool":True,
        "checkpoints":rows,"elapsed_seconds":time.perf_counter()-start_time,
        "scope":"Artifact and numerical replay only; does not score market outcomes or establish predictive validity."}
    R.write_json(args.output,report)
    print(json.dumps({key:report[key] for key in ("status","checkpoint_count","origin_replays",
        "maximum_absolute_mean_difference","maximum_absolute_sample_difference",
        "maximum_absolute_preprocessing_difference","numerical_clip_count","elapsed_seconds")}))


if __name__ == "__main__":
    main()
