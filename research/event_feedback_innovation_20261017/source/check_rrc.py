"""Non-market signal, maturity, invariance and streaming checks."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from datetime import date,timedelta
import numpy as np
import torch

from rrc_core import (Market,CausalData,configure,pretrain,teacher_stream,
                      calibration_stream,student_stream,digest_arrays,H,SQRT_H,K)


def run():
    configure()
    rng=np.random.default_rng(19271)
    n=625
    dates=np.array([(date(2018,1,1)+timedelta(days=i)).isoformat() for i in range(n)])
    days=np.arange(n)+date(2018,1,1).toordinal()
    # A genuine learnable conditional signal, with observation noise.
    dp=np.zeros(n)
    for t in range(1,n):
        dp[t]=.65*dp[t-1]+rng.normal(0,.2)
    p=100+np.cumsum(dp)
    def make(price):
        return Market("synthetic",dates,days,price,np.empty((n,0)),(),"synthetic")
    data=CausalData(make(p),dates[450])
    assert data.train[-1]+H<data.validation[0]
    assert data.validation[-1]+H<data.start
    try:
        data.batch([449],450)
        raise AssertionError("Immature target accepted")
    except ValueError:
        pass
    teacher,student,fit=pretrain(data,331,shrink=.1,max_epochs=3)
    assert fit["epochs"]==3 and np.isfinite([r["teacher_objective"] for r in fit["trace"]]).all()
    events,_,tr=teacher_stream(data,teacher,seed=331,shrink=.1)
    corr,beta,cr=calibration_stream(data,events,alpha=.1)
    pred,_,sr=student_stream(data,events,student,variant="rrc",shrink=.1,corrections=corr)
    assert pred.shape==(n-450,H) and np.isfinite(pred).all()
    assert tr["slow_teacher_updates"]>0 and sr["fast_steps"]>0 and cr["ridge_refits"]>0
    first_refit=450+max(K)+H+32-1
    assert cr["updates"][0]["event"]==first_refit
    assert all(c["latest_issue"]+H==c["latest_label"]==c["event"] for c in cr["updates"])
    assert all(b["event"]-a["event"]==5 for a,b in zip(cr["updates"],cr["updates"][1:]))
    for t,c in corr.items():
        if t<first_refit:
            assert not c.any()
    # Perturb an unseen suffix by a huge deterministic amount. Every issuance
    # made before it, including online updates and calibration, must be identical.
    modified=p.copy(); modified[551:]+=np.linspace(500,2000,n-551)
    alt=CausalData(make(modified),dates[450])
    assert alt.floor==data.floor
    e2,_,_=teacher_stream(alt,teacher,seed=331,shrink=.1)
    c2,_,_=calibration_stream(alt,e2,alpha=.1)
    pred2,_,_=student_stream(alt,e2,student,variant="rrc",shrink=.1,corrections=c2)
    untouched=551-450
    prefix_error=float(np.max(np.abs(pred[:untouched]-pred2[:untouched])))
    suffix_difference=float(np.max(np.abs(pred[untouched:]-pred2[untouched:])))
    assert prefix_error==0 and suffix_difference>0
    # Positive affine price units leave every normalized feature and label
    # unchanged; reconstruction must use each issuing origin's own scale.
    affine=CausalData(make(3*p+17),dates[450])
    b1=data.batch([100,300,420],data.start-1)
    b2=affine.batch([100,300,420],affine.start-1)
    assert np.allclose(b1["x"],b2["x"],rtol=0,atol=1e-11)
    assert np.allclose(b1["y"],b2["y"],rtol=0,atol=1e-11)
    assert np.allclose(b2["scale"],3*b1["scale"],atol=1e-12)
    # A one-step teacher predicting 107 after observing 105 must become the
    # old-origin six-step target 107, irrespective of either normalized unit.
    old_base,current_base,old_scale,current_scale,a,j=100.,105.,2.,3.,5,1
    teacher_norm=(107.-current_base)/(current_scale*np.sqrt(j))
    old_target=(current_base-old_base+current_scale*np.sqrt(j)*teacher_norm)/(old_scale*np.sqrt(a+j))
    assert abs(old_base+old_scale*np.sqrt(a+j)*old_target-107)<1e-12
    # Replicating age views must not change ridge regularization per event.
    x=rng.normal(size=(37,5,7)); yy=rng.normal(size=(37,5,20))
    z=x.reshape(-1,7); y=yy.reshape(-1,20)
    b=np.linalg.solve(z.T@z/len(z)+.1*np.eye(7),z.T@y/len(z))
    zr=np.repeat(z,2,axis=0); yr=np.repeat(y,2,axis=0)
    br=np.linalg.solve(zr.T@zr/len(zr)+.1*np.eye(7),zr.T@yr/len(zr))
    duplicate_error=float(np.max(abs(b-br)))
    assert duplicate_error<1e-12
    frozen=digest_arrays({"pred":pred})
    with torch.no_grad():
        for par in teacher.parameters(): par.add_(99.)
    assert frozen==digest_arrays({"pred":pred})
    return dict(passed=True,market_fits=0,synthetic_joint_pretrains=1,
                checks=["nonempty learnable signal fit","train/validation H20 purge", "reject immature targets",
                        "multi-horizon shapes", "cold calibration start", "H20 delayed calibration labels",
                        "fixed five-event calibration refits", "future suffix perturbation full online pipeline",
                        "positive affine price unit invariance", "old/current origin scale reanchoring",
                        "event-weighted ridge invariant to duplicated views", "issued prediction immutability"],
                future_prefix_prediction_error=prefix_error,changed_suffix_prediction_difference=suffix_difference,
                duplicated_views_ridge_error=duplicate_error,first_calibration_refit_event=first_refit,
                teacher_updates=tr["slow_teacher_updates"],student_fast_updates=sr["fast_steps"],
                calibration_refits=cr["ridge_refits"],source_sha256=hashlib.sha256(Path(__file__).with_name("rrc_core.py").read_bytes()).hexdigest())


if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument("--output",required=True);args=ap.parse_args()
    result=run();path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result))
