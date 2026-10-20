"""One prespecified market timing pilot; never computes evaluation scores."""
from __future__ import annotations
import argparse,json,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
from rrc_core import load_market,CausalData,pretrain,teacher_stream,calibration_stream,student_stream,arrays
import numpy as np

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--protocol",required=True);args=ap.parse_args()
    protocol=Path(args.protocol);cfg=json.loads(protocol.read_text())
    source=Path(__file__).with_name("rrc_core.py")
    assert hashlib.sha256(source.read_bytes()).hexdigest()==cfg["source_sha256"]
    out=Path(cfg["output_directory"]);out.mkdir(parents=True,exist_ok=True)
    started=datetime.now(timezone.utc).isoformat()
    market=load_market(cfg["data_path"],"EEX_EUA_T3PA",True)
    assert market.source_sha256==cfg["data_sha256"]
    data=CausalData(market,"2021-01-01")
    teacher,student,fit=pretrain(data,998,.1,max_epochs=3)
    events,final_teacher,ts=teacher_stream(data,teacher,seed=998,shrink=.1,stop_index=data.start+80)
    corr,beta,cal=calibration_stream(data,events,alpha=.1)
    pred,final_student,ss=student_stream(data,events,student,variant="rrc",shrink=.1,corrections=corr)
    np.savez_compressed(out/"pilot_states_and_predictions.npz",predictions=pred,**{"T_"+k:v for k,v in arrays(final_teacher).items()},**{"S_"+k:v for k,v in arrays(final_student).items()})
    record=dict(started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),source_sha256=cfg["source_sha256"],kind="one market timing pilot only",seed=998,pretraining=fit,teacher_stream=ts,calibration=cal,student_stream=ss,forecast_metrics_computed=False)
    (out/"TIMING_PILOT.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(dict(offline_seconds=fit["seconds"],offline_epochs=fit["epochs"],training_origins=len(data.train),teacher_80_event_seconds=ts["seconds"],calibrator_80_event_seconds=cal["seconds"],student_80_event_seconds=ss["seconds"],formal_market_fits=0,timing_market_fits=1,scored_forecasts=0)))
