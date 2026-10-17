"""Read only frozen protocol and source text; never import a training module."""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study",type=Path,default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parent)
    args=ap.parse_args()
    root=args.study.resolve()
    protocol=root/"provenance/D1_PROTOCOL.json"
    cfg=json.loads(protocol.read_text())
    source_hashes={}
    for rel,expected in cfg["source_sha256"].items():
        actual=sha(root/rel)
        assert actual==expected,(rel,actual,expected)
        ast.parse((root/rel).read_text())
        source_hashes[rel]=actual
    jobs=len(cfg["markets"])*len(cfg["seeds"])
    cal=cfg["calibrators"]
    students=cfg["student_configs"]
    assert len({m["name"] for m in cfg["markets"]})==len(cfg["markets"])
    assert len({c["id"] for c in cal})==len(cal)
    assert len({s["id"] for s in students})==len(students)
    assert all(s.get("calibrator") is None or s["calibrator"] in {c["id"] for c in cal} for s in students)
    actual_counts=dict(jobs=jobs,
                       joint_teacher_student_offline_fits=jobs*len(cfg["pretrain_shrink"]),
                       selected_teacher_online_streams=jobs,
                       student_online_streams=jobs*len(students),
                       ridge_calibration_online_streams=jobs*len(cal))
    for key,value in actual_counts.items():
        assert value==cfg["counts"][key],(key,value,cfg["counts"][key])
    scored_configs=4+sum(not c["old_target"] for c in cal)+len(students)
    expected_rows=jobs*scored_configs*len(cfg["scored_years"])*len(cfg["score_horizons"])
    assert cfg["confirmation_access"] is False
    assert cfg["data_end_exclusive"]=="2024-01-01"
    assert cfg["online_start"]=="2021-01-01"
    assert cfg["warmup_year"]==2021 and cfg["scored_years"]==[2022,2023]
    assert cfg["selection_year"]==2022 and cfg["advancement_year"]==2023
    assert cfg["L"]==cfg["H"]==20 and cfg["td_k"]==[1,2,4,8,16]
    assert cfg["validation_size"]==80 and cfg["pretrain_lr"]==.003
    assert cfg["dtype"]=="float64" and cfg["threads_per_process"]==1
    for market in cfg["markets"]:
        path=Path(market["path"])
        assert "development" in path.name.lower()
        assert "confirmation" not in str(path).lower()
        assert len(market["sha256"])==64
    result=dict(
        status="PASS_WITH_COOPERATIVE_TIMEOUT_SCOPE",
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        method="Independent human static review plus AST parse, frozen hash and protocol-count assertions; no training-module imports.",
        protocol_sha256=sha(protocol),source_sha256=source_hashes,
        auditor_sha256=sha(Path(__file__)),actual_counts=actual_counts,
        scored_configurations_per_job=scored_configs,expected_metric_rows=expected_rows,
        processes=cfg["cpu_processes"],threads_per_process=cfg["threads_per_process"],
        data_paths_reviewed=[dict(name=m["name"],path=m["path"],with_auction=m["with_auction"]) for m in cfg["markets"]],
        data_file_contents_read=False,market_numeric_reads=0,model_fits=0,
        result_or_forecast_files_read=False,
        reviewed_semantics=[
            "Only the five explicitly configured development paths are passed to load_market; numeric price conversion follows the date<2024 filter.",
            "The worker refuses an existing per-series/per-seed directory; the parent refuses an existing D1 STARTED marker.",
            "Each worker selects shrink only from pre-online validation; exact ties prefer the larger shrink.",
            "A single selected teacher stream is shared by every student intervention; each student_stream independently copies the same initial student.",
            "Four direct forecasts, eight correct-target output calibrations and 26 student branches yield 38 configurations per job.",
            "The runner saves every score without selecting on 2023; the separately specified selection function is outside this review.",
            "Public worker summaries omit raw support/local paths and numerical arrays; per-origin forecasts and numerical calibration/weights are written under local_only.",
            "Any caught worker failure is recorded and unstarted futures are cancelled; there is no failed-fit retry loop."
        ],
        timeout_scope=[
            "One shared UTC deadline is converted to a local monotonic deadline in every worker, so late-starting jobs do not receive a fresh 40-minute budget.",
            "Deadline checks occur at each pretrain epoch and teacher/calibrator/student event, not during an individual optimizer or linear-solve call.",
            "Static forecast assembly, metrics, serialization and file writes do not have independent hard-stop checks.",
            "ProcessPoolExecutor context exit waits for already-running workers; cancellation does not forcibly terminate running work.",
            "Describe the budget as cooperative. It does not guarantee an exact wall-clock kill at the deadline."
        ],
        limitations=[
            "No actual market row counts, market file hashes, runtime result completeness or scores verified here.",
            "Score arithmetic and 2022-only selection are assigned to a separate independent audit.",
            "EEX and EU_REPORTED are related EU ETS targets, not independent replications."
        ])
    output=args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=True)
    (output/"RRC_RUNNER_STATIC_AUDIT.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("status","protocol_sha256","actual_counts","scored_configurations_per_job","expected_metric_rows","processes")},indent=2))


if __name__=="__main__":
    main()
