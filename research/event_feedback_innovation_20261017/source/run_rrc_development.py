"""Execute only the hash-frozen, finite development protocol.

Full price-valued forecasts, cached events, calibrated targets and states remain
local. Public output is restricted to aggregate losses, settings and provenance.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_k]="1"
import argparse,copy,csv,hashlib,json,time,traceback
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from rrc_core import (load_market,CausalData,pretrain,teacher_stream,calibration_stream,
                      student_stream,metric_cells,arrays,digest_arrays,SQRT_H)


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def worker(protocol,entry,seed):
    cfg=json.loads(Path(protocol).read_text())
    root=Path(cfg["study_root"])
    local=root/"local_only"/"D1"/f"{entry['name']}_s{seed}"
    if local.exists():
        raise FileExistsError(f"Refuse overwrite or hidden rerun: {local}")
    local.mkdir(parents=True)
    deadline=time.monotonic()+max(0.,(datetime.fromisoformat(cfg["deadline_utc"])-datetime.now(timezone.utc)).total_seconds())
    started=datetime.now(timezone.utc).isoformat()
    write_json(local/"STARTED.json",dict(started_utc=started,seed=seed,market=entry["name"],
                                       protocol_sha256=hashlib.sha256(Path(protocol).read_bytes()).hexdigest()))
    market=load_market(entry["path"],entry["name"],entry["with_auction"],cfg["data_end_exclusive"])
    if market.source_sha256!=entry["sha256"]:
        raise ValueError("Data identity changed")
    data=CausalData(market,cfg["online_start"])
    pretrained=[]
    for shrink in cfg["pretrain_shrink"]:
        teacher,student,receipt=pretrain(data,seed,shrink,max_epochs=cfg["max_epochs"],
                                         patience=cfg["patience"],deadline=deadline)
        tag=f"shrink{shrink}"
        write_json(local/f"pretrain_{tag}.json",receipt)
        np.savez_compressed(local/f"pretrain_{tag}.npz",**{"T."+k:v for k,v in arrays(teacher).items()},
                            **{"S."+k:v for k,v in arrays(student).items()})
        pretrained.append((receipt["validation_score"],shrink,teacher,student,receipt))
    _,shrink,teacher,student,pre=sorted(pretrained,key=lambda x:(x[0],-x[1]))[0]
    write_json(local/"PRETRAIN_SELECTION.json",dict(shrink=shrink,score=pre["validation_score"],
                rule="Minimum pre-online validation score; exact ties prefer larger shrink"))
    events,final_teacher,tr=teacher_stream(data,teacher,seed=seed,shrink=shrink,
                    teacher_lr=cfg["teacher_lr"],replay_size=cfg["replay_size"],replay_batch=cfg["replay_batch"],deadline=deadline)
    write_json(local/"teacher_stream.json",tr)
    # A local exact cache permits independent chronological audit. It is not
    # included in the public deliverable because it contains derived quotes.
    serial=[]
    for e in events:
        serial.append({k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in e.items()})
    write_json(local/"teacher_events.json",serial)
    np.savez_compressed(local/"teacher_final.npz",**arrays(final_teacher))
    corrections={};cal_receipts={}
    for cc in cfg["calibrators"]:
        corr,beta,receipt=calibration_stream(data,events,alpha=cc["alpha"],mode=cc["mode"],
                              old_target=cc["old_target"],event_window=cfg["calibration_window"],
                              min_events=cfg["calibration_min_events"],refit_every=cfg["calibration_refit_every"],deadline=deadline)
        corrections[cc["id"]]=corr;cal_receipts[cc["id"]]=receipt
        write_json(local/f"cal_{cc['id']}.json",receipt)
        np.savez_compressed(local/f"cal_{cc['id']}.npz",origins=np.array(list(corr)),
                            corrections=np.stack(list(corr.values())),beta=beta)
    all_cells=[];runs=[]
    def score_and_save(config_id,family,pred,extra=None):
        cells=metric_cells(data,events,pred,cfg["scored_years"])
        for c in cells:
            c.update(seed=seed,config_id=config_id,family=family)
        all_cells.extend(cells)
        np.savez_compressed(local/f"forecast_{config_id}.npz",origins=np.array([e["origin"] for e in events]),predictions=pred)
        runs.append(dict(market=entry["name"],seed=seed,config_id=config_id,family=family,
                         score=float(np.mean([c["relative_mse"] for c in cells])),
                         predictions_sha256=digest_arrays({"predictions":pred}),
                         extra=extra or {}))
    static_t=[];static_ts=[];online_t=[];rw=[]
    with torch.no_grad():
        for e in events:
            st=torch.tensor(e["static"][None]);x=torch.tensor(e["x"][None])
            ss=student(x,st).numpy()[0]
            static_t.append(e["base"]+e["scale"]*SQRT_H*e["static"])
            static_ts.append(e["base"]+e["scale"]*SQRT_H*(e["static"]+ss))
            online_t.append(e["base"]+e["scale"]*SQRT_H*e["current"])
            rw.append(np.repeat(e["base"],20))
    score_and_save("rw","rw",np.stack(rw))
    score_and_save("static_gru","static_gru",np.stack(static_t))
    score_and_save("static_teacher_student","static_teacher_student",np.stack(static_ts))
    score_and_save("online_gru","online_gru",np.stack(online_t))
    for cc in cfg["calibrators"]:
        if cc["old_target"]:
            continue
        pp=[]
        for e in events:
            norm=corrections[cc["id"]].get(e["origin"],np.zeros((5,20))).mean(0)
            pp.append(e["base"]+e["scale"]*SQRT_H*(e["current"]+norm))
        score_and_save("output_"+cc["id"],"output_only",np.stack(pp),cc)
    for spec in cfg["student_configs"]:
        cc=spec.get("calibrator")
        corr=None if cc is None else corrections[cc]
        pred,final,receipt=student_stream(data,events,student,variant=spec["variant"],
                              shrink=shrink,lr=spec["lr"],corrections=corr,gamma=cfg["gamma"],deadline=deadline)
        write_json(local/f"student_{spec['id']}.json",receipt)
        np.savez_compressed(local/f"student_{spec['id']}.npz",**arrays(final))
        score_and_save(spec["id"],spec["family"],pred,spec)
    write_csv(local/"METRIC_CELLS.csv",all_cells)
    record=dict(market=entry["name"],seed=seed,started_utc=started,
                finished_utc=datetime.now(timezone.utc).isoformat(),selected_shrink=shrink,
                pretraining_fits=len(pretrained),teacher_online_streams=1,
                student_online_streams=len(cfg["student_configs"]),calibration_online_streams=len(cfg["calibrators"]),
                source_sha256=market.source_sha256,support=data.support_receipt(),
                teacher_receipt=tr,calibrators={k:{kk:vv for kk,vv in v.items() if kk!="updates"} for k,v in cal_receipts.items()},
                runs=runs,local_directory=str(local),status="complete")
    write_json(local/"COMPLETE.json",record)
    # Omit raw supports, local paths and derived quotes from public summary.
    public={k:v for k,v in record.items() if k not in ("support","local_directory")}
    return public,all_cells


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--protocol",required=True);args=ap.parse_args()
    p=Path(args.protocol).resolve();cfg=json.loads(p.read_text());root=Path(cfg["study_root"])
    for rel,expected in cfg["source_sha256"].items():
        if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=expected:
            raise ValueError(f"Frozen source changed: {rel}")
    output=root/"results"/"D1";output.mkdir(parents=True,exist_ok=True)
    if (output/"STARTED.json").exists():
        raise FileExistsError("D1 has already been started; refuse silent rerun")
    write_json(output/"STARTED.json",dict(started_utc=datetime.now(timezone.utc).isoformat(),
               protocol_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),planned_jobs=len(cfg["markets"])*len(cfg["seeds"])))
    records=[];cells=[];failures=[]
    with ProcessPoolExecutor(max_workers=cfg["cpu_processes"]) as pool:
        jobs={pool.submit(worker,str(p),m,s):(m["name"],s) for m in cfg["markets"] for s in cfg["seeds"]}
        for fut in as_completed(jobs):
            key=jobs[fut]
            try:
                record,metrics=fut.result();records.append(record);cells.extend(metrics)
                write_json(output/f"{key[0]}_s{key[1]}_SUMMARY.json",record)
                print(json.dumps(dict(completed_market=key[0],seed=key[1],completed_jobs=len(records),status="complete")),flush=True)
            except BaseException as exc:
                failures.append(dict(market=key[0],seed=key[1],error=repr(exc),traceback=traceback.format_exc()))
                write_json(output/"FAILURES.json",failures)
                for other in jobs:other.cancel()
                raise
    write_csv(output/"ALL_METRICS.csv",cells)
    write_json(output/"COMPLETE.json",dict(finished_utc=datetime.now(timezone.utc).isoformat(),jobs=len(records),
               joint_pretraining_fits=sum(r["pretraining_fits"] for r in records),
               teacher_online_streams=sum(r["teacher_online_streams"] for r in records),
               student_online_streams=sum(r["student_online_streams"] for r in records),
               calibration_online_streams=sum(r["calibration_online_streams"] for r in records),
               failures=failures,metric_rows=len(cells),protocol_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))


if __name__=="__main__":main()
