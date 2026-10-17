"""Independent synthetic audit of current RRC implementation; no market reads."""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
SOURCE = WORKSPACE / "carbon_workspace/research/event_feedback_innovation_20261017/source/rrc_core.py"
CHECKS = []
COMPARISONS = 0
MAX_ERROR = 0.


def check(name, value, detail=None):
    assert bool(value), (name,detail)
    CHECKS.append(dict(name=name,status="PASS",detail=detail))


def close(name, a, b, tolerance=2e-11):
    global COMPARISONS, MAX_ERROR
    a,b=np.asarray(a),np.asarray(b)
    assert a.shape == b.shape, (name,a.shape,b.shape)
    error=float(np.max(np.abs(a-b))) if a.size else 0.
    COMPARISONS += a.size
    MAX_ERROR=max(MAX_ERROR,error)
    check(name,np.isfinite(a).all() and np.isfinite(b).all() and error<=tolerance,
          dict(values=int(a.size),max_absolute_error=error,tolerance=tolerance))


def load_core():
    spec=importlib.util.spec_from_file_location("rrc_independent_synthetic",SOURCE)
    core=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=core
    spec.loader.exec_module(core)
    core.configure()
    core.load_market=lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError("Market loading forbidden"))
    return core


def make_data(core, future_cut=None):
    rng=np.random.default_rng(202610178)
    n,start=490,300
    increments=.015*np.sin(np.arange(n-1)/21)+rng.normal(0,.2,n-1)
    price=100+np.r_[0.,np.cumsum(increments)]
    days=dt.date(2018,1,1).toordinal()+np.r_[0,np.cumsum(rng.integers(1,4,n-1))]
    dates=np.array([dt.date.fromordinal(int(d)).isoformat() for d in days])
    if future_cut is not None:
        price[future_cut+1:]+=np.linspace(10.,30.,n-future_cut-1)
    market=core.Market("SYNTHETIC",dates,days,price,np.empty((n,0)),(),"synthetic:no-file")
    return core.CausalData(market,str(dates[start]),validation_size=80)


class GuardedArray:
    def __init__(self, a, clock, name):
        self.a,self.clock,self.name=np.asarray(a),clock,name
        self.reads=0
    def __len__(self):
        return len(self.a)
    def __getitem__(self,key):
        selected=np.arange(len(self.a))[key]
        if np.size(selected) and int(np.max(selected))>self.clock[0]:
            raise AssertionError(f"{self.name} future access through {int(np.max(selected))} at {self.clock[0]}")
        self.reads+=int(np.size(selected))
        return self.a[key]


class ClockedEvents:
    def __init__(self,events,clock):
        self.events,self.clock=events,clock
    def __len__(self):
        return len(self.events)
    def __iter__(self):
        for event in self.events:
            self.clock[0]=event["origin"]
            yield event


def guard_data(data):
    copied=copy.deepcopy(data)
    clock=[data.start-1]
    price=GuardedArray(copied.market.price,clock,"price")
    days=GuardedArray(copied.market.days,clock,"day")
    copied.market.price,copied.market.days=price,days
    # Do not let already cached features hide an otherwise invalid fresh read.
    copied._features={}
    return copied,clock,price,days


def event_hash(events):
    h=hashlib.sha256()
    for e in events:
        for key,value in sorted(e.items()):
            h.update(key.encode())
            if isinstance(value,np.ndarray):
                h.update(str(value.shape).encode());h.update(value.tobytes())
            else:
                h.update(repr(value).encode())
    return h.hexdigest()


def independent_update(loss,opt,module):
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(module.parameters(),1.)
    opt.step()


def chronological_joint(core,data,initial_teacher,initial_student,seed,shrink):
    """Separate event loop, with teacher/student slow losses built before both steps."""
    teacher,student=copy.deepcopy(initial_teacher),copy.deepcopy(initial_student)
    topt=torch.optim.AdamW(teacher.parameters(),lr=.0001,weight_decay=.001)
    sslow=torch.optim.AdamW(student.parameters(),lr=.0003,weight_decay=.001)
    sfast=torch.optim.AdamW(student.parameters(),lr=.0003,weight_decay=.001)
    rng=np.random.default_rng(seed+20261017)
    issued,teacher_issued,replay=[],[],[]
    steps=dict(teacher_slow=0,student_slow=0,student_fast=0)
    for t in range(data.start,len(data.market.price)):
        if t-core.H>=data.start:
            replay.append(t-core.H)
            replay=replay[-128:]
        if len(replay)>=16:
            ids=np.array(replay)[rng.choice(len(replay),16,replace=False)]
            batch=data.batch(ids,t)
            x,y=torch.tensor(batch["x"]),torch.tensor(batch["y"])
            pt=teacher(x)
            fs=pt.detach()+student(x,pt)
            tloss=(pt-y).square().mean()+shrink*pt.square().mean()
            sloss=(fs-y).square().mean()+shrink*fs.square().mean()
            independent_update(tloss,topt,teacher)
            independent_update(sloss,sslow,student)
            steps["teacher_slow"]+=1;steps["student_slow"]+=1
        now_x,now_scale=data.at(t)
        with torch.no_grad():
            now_t=teacher(torch.tensor(now_x[None]))
        if t-max(core.K)>=data.start:
            old_ids=np.array([t-k for k in core.K])
            old=data.batch(old_ids,t,labels=False)
            ox=torch.tensor(old["x"])
            with torch.no_grad():
                old_t=teacher(ox)
            old_s=old_t+student(ox,old_t)
            # Independent price construction by old absolute target index.
            target=np.empty((len(core.K),core.H))
            weight=np.empty_like(target)
            for i,age in enumerate(core.K):
                o=t-age
                for h in range(1,core.H+1):
                    if h<=age:
                        target_price=data.market.price[o+h]
                        weight[i,h-1]=1.
                    else:
                        j=h-age
                        target_price=data.market.price[t]+now_scale*np.sqrt(j)*float(now_t[0,j-1])
                        weight[i,h-1]=.9**j
                    target[i,h-1]=(target_price-data.market.price[o])/(old["scale"][i]*np.sqrt(h))
            fastloss=((old_s-torch.tensor(target)).square()*torch.tensor(weight)).mean()
            independent_update(fastloss,sfast,student)
            steps["student_fast"]+=1
        with torch.no_grad():
            now_s=now_t+student(torch.tensor(now_x[None]),now_t)
        issued.append(data.market.price[t]+now_scale*np.sqrt(np.arange(1,core.H+1))*now_s.numpy()[0])
        teacher_issued.append(now_t.numpy()[0].copy())
    return np.stack(issued),np.stack(teacher_issued),teacher,student,steps


def independent_phi(core,data,events,t):
    current=events[t-data.start]
    scale=current["scale"]
    delta=np.diff(data.market.price[t-10:t+1])
    rms=max(np.sqrt(np.mean(delta**2)),data.floor)
    v=np.tanh(np.r_[current["current"][[0,4,19]],delta.mean()/scale,
                      np.log(rms/scale),
                      np.log1p(data.market.days[t]-data.market.days[t-1])/np.log(8.)])
    features=[]
    for age in core.K:
        past=events[t-age-data.start]
        teacher_prices=past["base"]+past["scale"]*np.sqrt(np.arange(1,21))*past["current"]
        err=data.market.price[t-age+1:t+1]-teacher_prices[:age]
        scaled=err/(scale*np.sqrt(np.arange(1,age+1)))
        u=np.r_[np.tanh(scaled[-1]),np.tanh(scaled.mean()),
                np.tanh((err[-1]-err[0])/(scale*np.sqrt(age))),age/20]
        features.append(np.r_[1.,u,v,(u[:,None]*v[None,:]).reshape(-1)])
    return np.stack(features)


def main():
    started=time.perf_counter()
    sha_before=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    core=load_core()
    data=make_data(core)
    check("training_validation_purge",data.train[-1]+core.H<data.validation[0])
    check("validation_mature_before_online",data.validation[-1]+core.H<data.start)
    try:
        data.batch([data.start-5],data.start)
        raise AssertionError("Immature label accepted")
    except ValueError as error:
        check("batch_rejects_immature_H20","immature" in str(error))
    base=data.batch([data.start],data.start,labels=False)
    future=np.arange(1,21)*.03+base["base"][0]
    norm=(future-base["base"][0])/base["scale"][0]/core.SQRT_H
    close("price_scale_roundtrip",base["base"][0]+base["scale"][0]*core.SQRT_H*norm,future)
    # Verify actual pretrain updates, including detach, not merely a restatement.
    old_update=core._update
    last_teacher=None
    detach_checks=[]
    def observed_update(loss,opt,parameters):
        nonlocal last_teacher
        pp=list(parameters)
        is_teacher=any(p.ndim==2 and p.shape[0]==96 for p in pp)
        before=None if is_teacher or last_teacher is None else [None if p.grad is None else p.grad.clone() for p in last_teacher]
        value=old_update(loss,opt,pp)
        if is_teacher:
            last_teacher=pp
        else:
            detach_checks.append(all((a is None and p.grad is None) or
                                     (a is not None and p.grad is not None and torch.equal(a,p.grad))
                                     for a,p in zip(before,last_teacher)))
        return value
    core._update=observed_update
    try:
        _,_,pretrain_receipt=core.pretrain(data,113,.1,max_epochs=2,patience=2)
    finally:
        core._update=old_update
    check("actual_pretrain_student_backward_does_not_change_teacher_gradients",
          len(detach_checks)==2 and all(detach_checks),detach_checks)
    torch.manual_seed(114)
    teacher=core.Teacher(2,hidden=4)
    student=core.Student(2,width=3)
    with torch.no_grad():
        teacher.head.weight.normal_(0,.03);teacher.head.bias.normal_(0,.01)
        student.net[-1].weight.normal_(0,.02);student.net[-1].bias.normal_(0,.01)
    seed,shrink=113,.1
    events,final_teacher,teacher_receipt=core.teacher_stream(data,teacher,seed=seed,shrink=shrink)
    before=event_hash(events)
    guarded,clock,price_guard,day_guard=guard_data(data)
    stream=ClockedEvents(events,clock)
    raw,final_student,student_receipt=core.student_stream(guarded,stream,student,variant="dsof",shrink=shrink)
    joint,joint_teacher_values,joint_teacher,joint_student,joint_steps=chronological_joint(core,data,teacher,student,seed,shrink)
    close("cached_teacher_vs_joint_teacher_every_event",
          np.stack([e["current"] for e in events]),joint_teacher_values)
    close("cached_student_vs_joint_execution_every_forecast",raw,joint)
    for name,value in core.arrays(final_teacher).items():
        close("teacher_final_state_"+name,value,core.arrays(joint_teacher)[name])
    for name,value in core.arrays(final_student).items():
        close("student_final_state_"+name,value,core.arrays(joint_student)[name])
    check("joint_update_count_teacher",teacher_receipt["slow_teacher_updates"]==joint_steps["teacher_slow"])
    check("joint_update_count_student",
          student_receipt["slow_steps"]==joint_steps["student_slow"] and student_receipt["fast_steps"]==joint_steps["student_fast"])
    # Actual calibrator with reads guarded at each current event.
    corrections,beta,cal_receipt=core.calibration_stream(guarded,stream,alpha=.1)
    expected_first=data.start+max(core.K)+core.H+32-1
    expected_refits=list(range(expected_first,len(data.market.price),5))
    check("refit_every_five_even_after_window_saturation",
          [v["event"] for v in cal_receipt["updates"]]==expected_refits,expected_refits)
    check("window_saturation_reached",cal_receipt["updates"][-1]["unique_issuances"]==128)
    check("cal_empty_until_first_full_mature_minimum",
          all(np.array_equal(v,np.zeros_like(v)) for t,v in corrections.items() if t<expected_first))
    check("full_H20_has_training_target",beta.shape==(35,20) and np.any(beta[:,-1]!=0.))
    last=cal_receipt["updates"][-1]
    origins=np.arange(last["earliest_issue"],last["latest_issue"]+1)
    gram=np.zeros((35,35));cross=np.zeros((35,20))
    for t in origins:
        phi=independent_phi(core,data,events,int(t))
        issued=events[t-data.start]
        target=(data.market.price[t+np.arange(1,21)]-issued["base"])/(issued["scale"]*core.SQRT_H)-issued["current"]
        gram+=phi.T@phi/len(core.K)/len(origins)
        cross+=phi.T@np.broadcast_to(target,(len(core.K),20))/len(core.K)/len(origins)
    beta_independent=np.linalg.solve(gram+.1*np.eye(35),cross)
    close("event_averaged_ridge_independent_normal_equations",beta,beta_independent,8e-11)
    close("current_revision_label_and_feature_application",
          corrections[last["event"]],independent_phi(core,data,events,last["event"])@beta_independent,8e-11)
    # Same views repeated: per-issuance risk/regularization must stay identical.
    original_k=core.K
    try:
        core.K=original_k+original_k
        repeated,beta_repeated,repeated_receipt=core.calibration_stream(guarded,stream,alpha=.1)
    finally:
        core.K=original_k
    close("duplicate_K_views_do_not_change_ridge",beta,beta_repeated,8e-11)
    close("duplicate_K_views_do_not_change_issued_correction",
          corrections[last["event"]],repeated[last["event"]][:5],8e-11)
    check("duplicate_K_unique_event_count_unchanged",
          [u["unique_issuances"] for u in cal_receipt["updates"]]==
          [u["unique_issuances"] for u in repeated_receipt["updates"]])
    # Zero correction must be exactly the original TD branch.
    zeros={t:np.zeros_like(v) for t,v in corrections.items()}
    zero_pred,_,_=core.student_stream(guarded,stream,student,variant="rrc",shrink=shrink,corrections=zeros)
    close("zero_RRC_equals_original_TD",raw,zero_pred)
    check("teacher_and_issued_event_cache_never_mutated",before==event_hash(events))
    # Alter all future prices and separately truncate execution at a past cut.
    cut=410
    changed=make_data(core,future_cut=cut)
    prefix,_,_=core.teacher_stream(changed,teacher,seed=seed,shrink=shrink,stop_index=cut+1)
    for key in ("current","fast_teacher","er_teacher","x"):
        for i,e in enumerate(prefix):
            close("teacher_future_suffix_invariance_"+key+"_"+str(i),e[key],events[i][key])
    changed_guard,changed_clock,_,_=guard_data(changed)
    changed_stream=ClockedEvents(events,changed_clock)
    changed_corrections,_,_=core.calibration_stream(changed_guard,changed_stream,alpha=.1)
    for t in corrections:
        if t<=cut:
            close("calibrator_future_suffix_invariance_"+str(t),corrections[t],changed_corrections[t])
    prefix_student,_,_=core.student_stream(changed,events[:len(prefix)],student,variant="dsof",shrink=shrink)
    close("student_future_suffix_and_truncation_invariance",prefix_student,raw[:len(prefix)])
    sha_after=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    check("audited_source_unchanged_during_execution",sha_before==sha_after)
    result=dict(status="PASS",created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                audited_source=str(SOURCE),audited_source_sha256=sha_after,
                auditor_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                numpy_version=np.__version__,torch_version=torch.__version__,
                market_reads=0,market_fits=0,synthetic_pretrain_epochs=pretrain_receipt["epochs"],
                synthetic_events=len(events),cached_teacher_updates=teacher_receipt["slow_teacher_updates"],
                cached_student_slow_steps=student_receipt["slow_steps"],cached_student_fast_steps=student_receipt["fast_steps"],
                independent_joint_steps=joint_steps,
                guarded_price_elements_read=price_guard.reads,guarded_day_elements_read=day_guard.reads,
                exact_refit_event_indices=expected_refits,comparison_values=int(COMPARISONS),
                max_absolute_error=MAX_ERROR,checks=CHECKS,
                seconds=time.perf_counter()-started,
                scope="Static core review plus synthetic causality/normalization/cache/ridge checks; no market data, author benchmark, runner selection, or independent future confirmation audited.")
    (HERE/"RRC_CORE_SYNTHETIC_AUDIT.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="checks"},indent=2))


if __name__=="__main__":
    main()
