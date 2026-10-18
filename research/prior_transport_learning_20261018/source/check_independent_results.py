"""Synthetic fixtures for the independent D1 metrics auditor: no model fits."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import numpy as np


def load_auditor(path):
    spec=importlib.util.spec_from_file_location("prior_transport_independent_metrics",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dump(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def iso(origin):
    return (dt.date(2019,1,1)+dt.timedelta(days=int(origin))).isoformat()


def role(origin):
    return dict(n=len(origin),first_origin=int(origin[0]),last_origin=int(origin[-1]),
                first_date=iso(origin[0]),last_date=iso(origin[-1]),last_label_date=iso(origin[-1]+20))


def arrays(origins,market):
    h=np.arange(1,21,dtype=float)[None,:]
    t=np.arange(len(origins),dtype=float)[:,None]
    delta=(.9+.07*t)*np.sin(.31*h)+.06*(market+1)*np.cos(.17*h)+.01*t
    sigma=(.5+.07*market+.02*t)*np.sqrt(h)
    return delta,sigma


def analytic_metrics(delta,score):
    # Truth is generated so error = -sqrt(score)*delta. This oracle uses that
    # algebra; it never calls the auditor's scoring function.
    rw=np.mean(delta**2,axis=0)
    mse=score*rw
    rows=[dict(h=h,mse=float(mse[h-1]),rw_mse=float(rw[h-1]),
               rmse=float(np.sqrt(mse[h-1])),relative_mse=float(score),
               mae=float(np.sqrt(score)*np.mean(np.abs(delta[:,h-1])))) for h in (1,5,20)]
    return dict(score=float(score),horizons=rows,n_origins=len(delta),
                all20_relative_mse=[float(score)]*20)


def build(module,folder):
    public,private=folder/"public",folder/"private"
    public.mkdir();private.mkdir()
    protocol=dict(year=2022,models=list(module.MODELS),seeds=[17,29,43],static_seed=17,
                  config=dict(seed=17,max_steps=20,episode_seed=17018,validation_interval=10,
                              min_delta=1e-6,patience=10),synthetic_fixture=True)
    tasks,_,_=module.planned(protocol)
    protocol["tasks"]=tasks
    dump(folder/"PROTOCOL.json",protocol)
    means=dict(local_profile=1.03,uniform_profile=1.01,aux_one=1.02,aux_five=.99,
               profile=.96,transport=.93,transport_uniform=.97,covariance_profile=.95,
               diagonal_profile=.98,local_gru=.965,pooled_gru=.945,ridge=.985,rw=1.)
    modelscores={m:[] for m in module.MODELS}
    for task in tasks:
        target,budget,model,seed=(task[k] for k in ("target","budget","model","seed"))
        mid=module.MARKETS.index(target)
        support=np.arange(60,300);meta=np.arange(320,400);val=np.arange(420,500)
        if budget==100:
            support=support[[round(j*239/59) for j in range(60)]]
            meta=meta[[round(j*79/19) for j in range(20)]]
            val=val[[round(j*79/19) for j in range(20)]]
        origins=np.arange(1100,1108)
        # Nested validation labels must come from the same underlying function,
        # not a row-number function that changes when its sample is thinned.
        full_val=np.arange(420,500)
        vd_full,vs_full=arrays(full_val,mid)
        val_idx=val-420
        vd,vs=vd_full[val_idx],vs_full[val_idx]
        delta,sigma=arrays(origins,mid)
        offset=0. if model in module.STATIC else ({17:-.005,29:0.,43:.005}[seed])
        score=means[model]+offset
        val_score=1. if model=="rw" else max(.01,score-.015)
        prediction=(1.-math.sqrt(score))*delta/sigma
        val_prediction=(1.-math.sqrt(val_score))*vd/vs
        path=private/(task["job"]+"_PREDICTIONS.npz")
        np.savez_compressed(path,prediction=prediction,delta=delta,sigma=sigma,
                            origin=origins,date=np.array([iso(o) for o in origins],dtype="U10"),
                            last_label_date=np.array([iso(o+20) for o in origins],dtype="U10"),
                            val_prediction=val_prediction,val_delta=vd,val_sigma=vs,
                            val_origin=val,val_date=np.array([iso(o) for o in val],dtype="U10"),
                            val_last_label_date=np.array([iso(o+20) for o in val],dtype="U10"))
        names=[n for n in module.MARKETS if n!=target and
               {n,target}!={"EEX_EUA_T3PA","EU_REPORTED"}]
        sources={}
        for n in names:
            count=180+module.MARKETS.index(n)*7
            sources[n]=dict(n=count,sha256=hashlib.sha256(n.encode()).hexdigest(),
                            first_origin=60,last_origin=60+count-1,first_date="2017-01-01",
                            last_label_date="2018-12-31")
        data_receipt=dict(target=target,budget=budget,year=2022,
            roles={key:role(value) for key,value in
                   dict(support=support,meta=meta,val=val,test=origins).items()},
            source_cutoff_exclusive=iso(meta[0]),sources=sources,episode_seed=17018,
            episode_schedule_sha256=module.schedule_hash(sources,20,17018))
        if model in module.PROFILE:
            fit=dict(eligible=True,steps=20,selected_step=20,selected_validation_score=val_score,
                     initial_encoder_hash=hashlib.sha256(f"profile-{seed}".encode()).hexdigest(),
                     source_ridge_calls_expected_completed_steps=20*len(names),
                     author_ridge_calls=20*len(names),test_target_values_used_in_fit=False,
                     test_forward_calls=1)
            traces=[dict(step=0,validation_score=val_score+.05,selected=True),
                    dict(step=10,validation_score=val_score+.01,selected=True),
                    dict(step=20,validation_score=val_score,selected=True)]
        elif model in module.GRU:
            fit=dict(steps_completed=20,best_step=20,best_validation_score=val_score,
                     initial_encoder_hash=hashlib.sha256(f"gru-{seed}".encode()).hexdigest(),
                     test_labels_accessed=False,initial_predictions_equal_rw=True,test_forward_calls=1)
            active=[target]+(sorted(sources) if model=='pooled_gru' else [])
            sampler_seeds={};sampler_digests={}
            for name in active:
                sampler_seed=(17018 if name==target else int.from_bytes(
                    hashlib.sha256(f'17018|baseline_source|{name}'.encode()).digest()[:8],'big'))
                n=len(support)+len(meta) if name==target else sources[name]['n']
                generator=np.random.default_rng(sampler_seed);hasher=hashlib.sha256()
                for _ in range(20):
                    hasher.update(np.asarray(generator.choice(n,size=64,replace=n<64),dtype='<i8').tobytes())
                sampler_seeds[name]=sampler_seed;sampler_digests[name]=hasher.hexdigest()
            fit['sampler_seeds']=sampler_seeds;fit['sampled_indices_sha256']=sampler_digests
            traces=[dict(step=0,validation=dict(score=1.),selected=True),
                    dict(step=10,validation=dict(score=val_score+.005),selected=True),
                    dict(step=20,validation=dict(score=val_score),selected=True)]
        elif model=="ridge":
            fit=dict(ridge_fits=5,selected_alpha=.1,validation_score=val_score,test_forward_calls=1)
            traces=[dict(alpha=a,validation_score=val_score+(0. if a==.1 else .02))
                    for a in (.01,.1,1.,10.,100.)]
        else:
            fit=dict(fits=0,test_forward_calls=0);traces=[]
        summary=dict(**task,status="COMPLETE",data_receipt=data_receipt,fit_receipt=fit,
                     validation_traces=traces,private_prediction_sha256=checksum(path),
                     evaluation=analytic_metrics(delta,score),validation=analytic_metrics(vd,val_score))
        dump(public/(task["job"]+"_SUMMARY.json"),summary)
        modelscores[model].append((budget,score))
    # Root-like aggregation oracle: expected analytic scores, no production
    # scorer and no use of the independent audit's output.
    ranked=[]
    for model,values in modelscores.items():
        ranked.append(dict(model=model,score=float(np.mean([s for _,s in values])),
                           budget100=float(np.mean([s for b,s in values if b==100])),
                           budget400=float(np.mean([s for b,s in values if b==400])),jobs=len(values)))
    lookup={r["model"]:r["score"] for r in ranked}
    gate=dict(all_jobs_completed=True,
              beats_profile_by_half_percent=lookup["transport"]<.995*lookup["profile"],
              beats_covariance_by_half_percent=lookup["transport"]<.995*lookup["covariance_profile"],
              does_not_lose_to_rw=lookup["transport"]<=1.)
    gate["advance"]=all(gate.values())
    dump(public/"AGGREGATE.json",dict(completed_jobs=len(tasks),failed_jobs=0,
                                     ranking=ranked,resource_gate=gate))
    return tasks,means


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--auditor",type=Path,default=Path(__file__).with_name("independent_results.py"))
    p.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[1]/"audits/INDEPENDENT_METRICS_SYNTHETIC.json")
    args=p.parse_args()
    module=load_auditor(args.auditor)
    tests=[]
    def require(value,label):
        if not bool(value):raise AssertionError(label)
        tests.append(label)
    def rejects(label,fn):
        try:fn()
        except (AssertionError,ValueError):tests.append(label)
        else:raise AssertionError(label+" was not rejected")
    with tempfile.TemporaryDirectory(prefix="prior_transport_metric_synthetic_") as tmp:
        folder=Path(tmp)
        tasks,means=build(module,folder)
        receipt,selection=module.audit(folder/"PROTOCOL.json",folder/"public",folder/"private",folder/"out")
        require(receipt["completed_prediction_flows"]==350,"all350syntheticflows")
        require(receipt["all20_split_cell_rows"]==14000,"all20_test_and_validation_rows")
        require(receipt["seed_averaged_cell_rows"]==2600,"2600_seed_averaged_cells")
        require(receipt["resource_counts"]["neural_fits_completed"]==330,"330_declared_fixture_neural_receipts")
        require(receipt["resource_counts"]["ridge_candidate_solves"]==50,"50_static_candidate_solves")
        require(selection["gates"]["advance"],"analytic_gate_pass")
        for row in selection["ranking"]:
            require(abs(row["score"]-means[row["model"]])<2e-13,"analytic_mean_"+row["model"])

        # A paired sigma mutation is made invisible to raw-score calculations by
        # reciprocally scaling the forecast. Pairing must still catch it.
        job=tasks[0]["job"]
        npz=folder/"private"/(job+"_PREDICTIONS.npz")
        summary_path=folder/"public"/(job+"_SUMMARY.json")
        saved_npz,saved_summary=npz.read_bytes(),summary_path.read_bytes()
        with np.load(npz,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
        data["sigma"]*=1.01;data["prediction"]/=1.01
        np.savez_compressed(npz,**data)
        summary=json.loads(saved_summary);summary["private_prediction_sha256"]=checksum(npz)
        dump(summary_path,summary)
        rejects("paired_sigma_tampering_detected",
                lambda:module.audit(folder/"PROTOCOL.json",folder/"public",folder/"private",folder/"bad"))
        npz.write_bytes(saved_npz);summary_path.write_bytes(saved_summary)
        missing=folder/"public"/(tasks[-1]["job"]+"_SUMMARY.json")
        saved=missing.read_bytes();missing.unlink()
        rejects("missing_frozen_flow_detected",
                lambda:module.audit(folder/"PROTOCOL.json",folder/"public",folder/"private",folder/"bad"))
        missing.write_bytes(saved)

        # The learner fits are only labels on analytic fixtures. No actual
        # predictor, optimizer or head solve is executed by this check.
        synthetic_comparisons=receipt["comparisons"]
        synthetic_max_error=receipt["maximum_absolute_error"]
    ones=np.ones((3,20))
    plus=module.raw_score(2*ones,ones,ones)["score"]
    minus=module.raw_score(np.zeros((3,20)),ones,ones)["score"]
    ensemble=module.raw_score(ones,ones,ones)["score"]
    require((plus+minus)/2==1. and ensemble==0.,"seed_loss_is_not_ensemble_loss")
    require(abs((2.+.5)/2-(2.+50.)/(1.+100.))>.5,"equal_cell_ratio_is_not_pooled_raw_ratio")
    rejects("zero_RW_denominator",lambda:module.raw_score(ones,np.zeros((3,20)),ones))
    def gates(candidate,profile=1.,covariance=1.,complete=True):
        rows=[dict(model=m,score=s) for m,s in
              [("transport",candidate),("profile",profile),("covariance_profile",covariance),("rw",1.)]]
        return module.development_selection(rows,complete)
    require(not gates(.995)["advance"],"exact_half_percent_is_not_strict_pass")
    require(gates(float(np.nextafter(.995,0.)))["advance"],"strictly_better_than_half_percent")
    require(not gates(float(np.nextafter(.995,1.)))["advance"],"just_worse_than_half_percent")
    require(gates(1.,2.,2.)["advance"],"RW_equality_allowed")
    require(not gates(float(np.nextafter(1.,2.)),2.,2.)["advance"],"RW_loss_disallowed")
    require(not gates(.9,complete=False)["advance"],"resource_failure_blocks_advance")
    result=dict(status="PASS_SYNTHETIC_INDEPENDENT_METRICS",
                completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                independent_results_sha256=checksum(args.auditor),
                check_source_sha256=checksum(__file__),
                analytic_fixture_flows=350,synthetic_comparisons=synthetic_comparisons,
                maximum_absolute_error=synthetic_max_error,tests=tests,
                runtime=dict(numpy=np.__version__),
                actual_actions=dict(market_reads=0,formal_result_reads=0,confirmation_reads=0,
                                    model_fits=0,head_solves=0,optimizer_steps=0),
                scope="Analytic fixture receipts emulate350flows; they are not actual model fits. No formal D1 outputs were accessed.")
    dump(args.output,result)
    print(json.dumps({k:result[k] for k in ("status","independent_results_sha256","synthetic_comparisons",
                                          "maximum_absolute_error","actual_actions")},ensure_ascii=False))


if __name__=="__main__":
    main()
