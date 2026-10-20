"""Finite, matched SOLID reference execution using frozen D1 GRU checkpoints.

No new backbone pretraining; all real query-local head adaptations are counted.
The original RRC stop decision and the sealed confirmation data remain intact.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import traceback
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch
from rrc_core import configure, load_market, CausalData, Teacher, arrays, digest_arrays, metric_cells
from solid_author_adapter import run_solid_author


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def state_and_data(cfg, entry):
    configure()
    if sha(entry["data_path"]) != entry["data_sha256"]:
        raise ValueError("Development file identity differs")
    if sha(entry["checkpoint_path"]) != entry["checkpoint_sha256"]:
        raise ValueError("D1 pretrained state identity differs")
    market = load_market(entry["data_path"], entry["market"], entry["with_auction"], "2024-01-01")
    data = CausalData(market, "2021-01-01")
    teacher = Teacher(data.at(data.start)[0].shape[1])
    with np.load(entry["checkpoint_path"], allow_pickle=False) as archive:
        state = {key[2:]: torch.from_numpy(archive[key].copy()) for key in archive.files if key.startswith("T.")}
    teacher.load_state_dict(state)
    if digest_arrays(arrays(teacher)) != entry["teacher_initial_state_sha256"]:
        raise ValueError("Teacher state differs from the common D1 initialization")
    teacher.eval()
    return data, teacher


def origins_for(data, years):
    dates = data.market.dates
    origins = np.arange(data.start, len(dates) - 20)
    kept = []
    for year in years:
        use = origins[(dates[origins] >= f"{year}-01-01") &
                      (dates[origins] < f"{year+1}-01-01") &
                      (dates[origins+20] < f"{year+1}-01-01")]
        kept.extend(use.tolist())
    return np.array(kept, dtype=int)


def worker(protocol_path, entry):
    pp = Path(protocol_path)
    cfg = json.loads(pp.read_bytes())
    root = Path(cfg["study_root"])
    private = root / "local_only" / cfg["phase"] / f"{entry['market']}_s{entry['seed']}"
    if private.exists():
        raise FileExistsError("Refuse overwrite or silent rerun")
    private.mkdir(parents=True)
    deadline = time.monotonic() + max(0., (datetime.fromisoformat(cfg["deadline_utc"]) -
                                         datetime.now(timezone.utc)).total_seconds())
    started = datetime.now(timezone.utc).isoformat()
    write_json(private / "STARTED.json", {"started_utc": started, "protocol_sha256": sha(pp)})
    data, teacher = state_and_data(cfg, entry)
    origins = origins_for(data, cfg["scored_years"])
    if cfg["timing_only"]:
        origins = origins[:cfg["pilot_query_count"]]
    cells, summaries = [], []
    for spec in cfg["configs"]:
        artifacts = []
        pred, receipt = run_solid_author(data, teacher, origins,
            author_root=root / "backup_review/author_solid",
            context_samples=spec["context_samples"], neighbors=cfg["neighbors"],
            learning_rate=spec["learning_rate"], remove_distance=spec["remove_distance"],
            cache_latents=True, deadline=deadline, artifact_sink=artifacts.append)
        if pred.shape != (len(origins), 20) or not np.isfinite(pred).all():
            raise ValueError("Nonfinite or incomplete SOLID outcome; no automatic retry")
        if digest_arrays(arrays(teacher)) != entry["teacher_initial_state_sha256"]:
            raise ValueError("Query adaptation modified the shared initial model")
        np.savez_compressed(private / f"forecast_{spec['id']}.npz", origins=origins, predictions=pred)
        if len(artifacts) != len(origins):
            raise ValueError("Incomplete per-query state archive")
        np.savez_compressed(private / f"adaptations_{spec['id']}.npz",
                            **{key: np.stack([row[key] for row in artifacts]) for key in artifacts[0]})
        write_json(private / f"receipt_{spec['id']}.json", receipt)
        if not cfg["timing_only"]:
            rows = metric_cells(data, [{"origin": int(t)} for t in origins], pred, cfg["scored_years"])
            for row in rows:
                row.update(seed=entry["seed"], config_id=spec["id"],
                           family="solid_recency" if spec["remove_distance"] else "solid_distance")
            cells.extend(rows)
        # The adapter's public receipt allowlist is set in the frozen protocol.
        public_receipt = {key: receipt[key] for key in cfg["public_adapter_receipt_keys"] if key in receipt}
        summaries.append({"config": spec, "query_count": len(origins),
                          "prediction_sha256": digest_arrays({"predictions": pred}),
                          "adapter": public_receipt})
    if cells:
        write_csv(private / "METRICS.csv", cells)
    record = {"market": entry["market"], "seed": entry["seed"], "started_utc": started,
              "finished_utc": datetime.now(timezone.utc).isoformat(),
              "query_count_per_config": len(origins), "streams": len(cfg["configs"]),
              "query_local_adaptations": len(origins)*len(cfg["configs"]),
              "new_backbone_pretraining_fits": 0, "timing_only": cfg["timing_only"],
              "forecast_metrics_computed": not cfg["timing_only"],
              "initial_teacher_state_sha256": entry["teacher_initial_state_sha256"],
              "data_sha256": entry["data_sha256"], "runs": summaries,
              "confirmation_data_read": False, "status": "complete"}
    write_json(private / "COMPLETE.json", record)
    return record, cells


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", type=Path, required=True)
    args = ap.parse_args()
    pp = args.protocol.resolve()
    cfg = json.loads(pp.read_bytes())
    root = Path(cfg["study_root"])
    for rel, expected in cfg["source_sha256"].items():
        if sha(root / rel) != expected:
            raise ValueError(f"Frozen source changed: {rel}")
    output = root / "results" / cfg["phase"]
    if output.exists():
        raise FileExistsError("Phase output exists; refuse repeat execution")
    output.mkdir(parents=True)
    write_json(output / "STARTED.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
                                        "protocol_sha256": sha(pp), "jobs": len(cfg["jobs"])})
    records, cells, failed = [], [], []
    with ProcessPoolExecutor(max_workers=cfg["cpu_processes"]) as pool:
        jobs = {pool.submit(worker, str(pp), entry): entry for entry in cfg["jobs"]}
        for future in as_completed(jobs):
            entry = jobs[future]
            try:
                record, rows = future.result()
                records.append(record); cells.extend(rows)
                write_json(output / f"{entry['market']}_s{entry['seed']}_SUMMARY.json", record)
                print(json.dumps({"phase": cfg["phase"], "completed_jobs": len(records),
                                  "market": entry["market"], "seed": entry["seed"]}), flush=True)
            except BaseException as exc:
                failed.append({"market": entry["market"], "seed": entry["seed"],
                               "error": repr(exc), "traceback": traceback.format_exc()})
                write_json(output / "FAILURES.json", failed)
                for other in jobs:
                    other.cancel()
                raise
    if cells:
        write_csv(output / "ALL_METRICS.csv", cells)
    write_json(output / "COMPLETE.json", {"finished_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha(pp), "jobs": len(records), "failures": failed,
        "streams": sum(r["streams"] for r in records),
        "query_local_adaptations": sum(r["query_local_adaptations"] for r in records),
        "new_backbone_pretraining_fits": 0, "metrics": len(cells),
        "timing_only": cfg["timing_only"], "confirmation_data_read": False})


if __name__ == "__main__":
    main()
