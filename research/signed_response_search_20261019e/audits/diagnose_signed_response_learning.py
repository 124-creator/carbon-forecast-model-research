"""Post-completion stage-e diagnostics: NumPy forward and linear-optimality replay.

No alpha/epoch/model selection, no gradient optimization, no new market split.
The two extra learned-rate, zero-gate readout solves are descriptive specified
post-fit diagnostics, not additional arms in the frozen comparison family.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY/"source"))
import run_signed_response_search as run

EXPECTED = {"signed_response_models.py": "5ceb7f04c567be32f1a399cbf01dafdfc6894d3840a158b6faae4153f09eb2f8",
            "run_signed_response_search.py": "905da0ca33607799b1ca63d9efe6710df173ec33c5c44e137071e024906d990c"}


def read(path):
    return json.loads(Path(path).read_text())


def rates(raw):
    raw = np.asarray(raw, float)
    lo = .98*expit(raw[...,0])
    hi = lo+(.98-lo)*expit(raw[...,1])
    return np.stack((lo,hi), axis=-1)


def basis(x, state, family, hs=(1,5,20)):
    rho = rates(state["rate_raw"])
    weights = (1-rho)[None]*rho[None]**np.arange(63,-1,-1)[:,None]
    v = x[...,0]@weights
    z = np.column_stack((np.log1p(np.abs(x[:,-1,0])),x[:,-1,1]*x[:,-1,2]))
    inp = z[:,:1] if family == "signed_shock_only" else z
    q = 2*np.tanh(inp@np.asarray(state["gate"]).T)
    lam = rates(np.asarray(state["rate_raw"])[None]+q) if family in (
        "signed_context_response","free_horizon_context","signed_shock_only") else np.broadcast_to(rho,(len(x),2))
    amp = np.exp(q) if family == "amplitude_context" else np.ones_like(q)
    ss = np.cumsum(lam[:,None,:]**np.arange(1,max(hs)+1)[None,:,None],axis=1)
    base = np.concatenate((np.broadcast_to(np.asarray(hs)[None,:,None],(len(x),len(hs),1)),
                           ss[:,np.asarray(hs)-1,:]*(v*amp)[:,None,:]),axis=-1)
    if family == "free_horizon_context":
        phi=(base[:,:,None,:]*np.eye(len(hs))[None,:,:,None]).reshape(len(x),len(hs),-1)
    else:
        phi=base
    return phi,{"states":v,"context":z,"future_rates":lam,"historical_rates":rho,"gate_shift":q,"amplitude":amp}


def independent_readout(phi,y,variance,alpha,stored=None):
    a=(phi/np.sqrt(variance)[None,:,None]).reshape(-1,phi.shape[-1])
    b=(y/np.sqrt(variance)[None]).reshape(-1)
    scale=np.maximum(np.sqrt(np.mean(a*a,axis=0)),1e-8)
    aa=a/scale
    aug_a=np.vstack((aa/np.sqrt(len(a)),np.sqrt(alpha)*np.eye(a.shape[-1])))
    aug_b=np.r_[b/np.sqrt(len(a)),np.zeros(a.shape[-1])]
    gamma=np.linalg.lstsq(aug_a,aug_b,rcond=None)[0]
    beta=gamma/scale
    record={"independent_residual_max":float(np.max(np.abs(aa.T@(aa@gamma-b)/len(a)+alpha*gamma)))}
    if stored is not None:
        stored_beta=np.asarray(stored["coefficients"])
        stored_gamma=stored_beta*scale
        kkt=aa.T@(aa@stored_gamma-b)/len(a)+alpha*stored_gamma
        rhs=aa.T@b/len(a)
        record.update({"coefficient_max_gap":float(np.max(np.abs(beta-stored_beta))),
                       "column_scale_max_gap":float(np.max(np.abs(scale-np.asarray(stored["head_column_scale"])))),
                       "stored_scaled_kkt_max":float(np.max(np.abs(kkt))),
                       "stored_scaled_kkt_relative":float(np.max(np.abs(kkt))/(1+np.max(np.abs(rhs))))})
    return beta,scale,record


def predict(phi,state,scale):
    return (phi@np.asarray(state["coefficients"]))*scale


def risk(pred,observed):
    rw=np.mean(observed**2,axis=0)
    ratios=np.mean((pred-observed)**2,axis=0)/rw
    return {"risk":float(np.mean(ratios)),"mse_over_rw_by_horizon":ratios.tolist(),
            "normalized_signed_alignment_A":float(np.mean(np.mean(pred*observed,axis=0)/rw)),
            "normalized_prediction_energy_B":float(np.mean(np.mean(pred**2,axis=0)/rw))}


def quantiles(a):
    return {"min":np.min(a,axis=0).tolist(),"q25":np.quantile(a,.25,axis=0).tolist(),
            "median":np.median(a,axis=0).tolist(),"q75":np.quantile(a,.75,axis=0).tolist(),"max":np.max(a,axis=0).tolist()}


def state_from_pt(path, family):
    obj=torch.load(path,map_location="cpu",weights_only=True)
    state=obj["state_dict"]
    return {k:np.asarray(v.detach().numpy() if torch.is_tensor(v) else v).tolist() for k,v in state.items()}


def verify_one(receipt,d,subset,expected_prediction=None):
    state=receipt["state_dict"]
    family=receipt["family"]
    phi,_=basis(d["x"],state,family)
    mask=d["masks"][subset]
    yn=d["y"]/d["scale"]
    vn=d["variance"]/d["scale"]**2
    _,_,optimality=independent_readout(phi[mask],yn[mask],vn,receipt["config"]["alpha"],state)
    pred=predict(phi,state,d["scale"])
    val_gap=None
    if subset == "train":
        val=d["masks"]["val"]
        val_loss=float(np.mean((pred[val]-d["y"][val])**2/d["variance"]))
        val_gap=abs(val_loss-receipt["validation_loss"])
    gap=None
    if expected_prediction is not None:
        gap=float(np.max(np.abs(pred[d["masks"]["test"]]-expected_prediction)))
    rec={"family":family,"subset":subset,"selected_epoch":receipt["selected_epoch"],
         "epochs_executed":receipt["epochs_executed"],"positive_gradient_steps":receipt["positive_gradient_steps"],
         "exact_head_solves_in_fit":len(receipt["history"]),"validation_loss_max_gap":val_gap,
         "saved_native_prediction_max_gap":gap,**optimality}
    assert optimality["stored_scaled_kkt_relative"] < 1e-9
    assert optimality["column_scale_max_gap"] < 1e-9
    assert val_gap is None or val_gap < 1e-9
    assert gap is None or gap < 1e-8
    assert len(receipt["history"]) == receipt["epochs_executed"]+1
    return rec,pred,phi


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path("/tmp/carbon-signed-response-runs-20261019e/core"))
    p.add_argument("--report",type=Path,default=STUDY/"methods/SIGNED_RESPONSE_LEARNING_DIAGNOSIS.json")
    args=p.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve previous diagnostic receipts")
    torch.set_num_threads(1)
    complete=read(args.out/"EXECUTION_COMPLETE.json")
    assert complete["status"] == "completed" and complete["source_hashes_unchanged"] and complete["plan_hash_unchanged"]
    protocol=read(args.out/"PROTOCOL_BEFORE_RUN.json")
    assert protocol["source_sha256"] == EXPECTED == run.source_hashes()
    run.check_old_sources()
    checks=[]
    markets=[]
    for spec in protocol["datasets"]:
        d=run.prepare(spec)
        folder=args.out/f"{spec['market']}_{run.FOLD}"
        rows=[]
        all_risks={}
        for family in run.ALL_FAMILIES:
            npz=np.load(folder/family/"predictions.npz")
            all_risks[family]=risk(npz["predictions"][0],npz["observed"])
        for family in run.RESPONSE_FAMILIES:
            dest=folder/family
            selection=read(dest/"SELECTION.json")
            for trial in selection["trials"]:
                receipt=read(dest/trial["receipt"])
                check,_,_=verify_one(receipt,d,"train")
                pt=state_from_pt(dest/Path(trial["receipt"]).with_suffix(".pt"),family)
                check["json_pt_state_max_gap"]=max(float(np.max(np.abs(np.asarray(pt[k])-np.asarray(v)))) for k,v in receipt["state_dict"].items())
                assert check["json_pt_state_max_gap"] == 0
                check["receipt"]=str(dest/trial["receipt"])
                checks.append(check)
            receipt=read(dest/"REFIT.json")
            npz=np.load(dest/"predictions.npz")
            expected=npz["predictions"][0]
            check,pred,phi=verify_one(receipt,d,"final",expected)
            state=receipt["state_dict"]
            pt=state_from_pt(dest/"refit.pt",family)
            check["json_pt_state_max_gap"]=max(float(np.max(np.abs(np.asarray(pt[k])-np.asarray(v)))) for k,v in state.items())
            assert check["json_pt_state_max_gap"] == 0
            check["receipt"]=str(dest/"REFIT.json")
            checks.append(check)
            final,test=d["masks"]["final"],d["masks"]["test"]
            yn,vn=d["y"]/d["scale"],d["variance"]/d["scale"]**2
            alpha=selection["selected"]["alpha"]
            initial=run.SignedResponse(family)
            initstate={k:v.detach().numpy().tolist() for k,v in initial.state_dict().items()}
            initphi,_=basis(d["x"],initstate,family)
            initbeta,initscale,_=independent_readout(initphi[final],yn[final],vn,alpha)
            initstate["coefficients"]=initbeta.tolist()
            initstate["head_column_scale"]=initscale.tolist()
            initpred=predict(initphi,initstate,d["scale"])[test]
            initgap=float(np.max(np.abs(initpred-npz["fixed_dynamics_refit_readout_predictions"][0])))
            assert initgap < 1e-8
            finalrisk=risk(pred[test],d["y"][test])
            initrisk=risk(initpred,d["y"][test])
            selected_receipt=read(dest/selection["selected"]["receipt"])
            first,last=selected_receipt["history"][0],selected_receipt["history"][selected_receipt["selected_epoch"]]
            row={"family":family,"selected_configuration":selection["selected"],
                 "final_state_dict":state,"initial_state_dict":initstate,
                 "selected_trial_training_objective_epoch0":first["training_profiled_objective"],
                 "selected_trial_training_objective_selected":last["training_profiled_objective"],
                 "selected_trial_validation_epoch0":first["validation_loss"],
                 "selected_trial_validation_selected":last["validation_loss"],
                 "all_selection_trials_positive_gradient_steps":sum(read(dest/t["receipt"])["positive_gradient_steps"] for t in selection["trials"]),
                 "final_positive_gradient_steps":receipt["positive_gradient_steps"],
                 "final_nonlinear_parameter_changes":receipt["selected_nonlinear_parameter_max_change"],
                 "final":finalrisk,"same_alpha_initial_dynamics_final_readout":initrisk,
                 "nonlinear_learning_risk_gain":initrisk["risk"]-finalrisk["risk"],
                 "initial_dynamics_npz_replay_native_max_gap":initgap}
            _,comp=basis(d["x"][test],state,family)
            lam=comp["future_rates"]
            shape5=np.sum(lam[:,None,:]**np.arange(1,6)[None,:,None],axis=1)/lam
            shape20=np.sum(lam[:,None,:]**np.arange(1,21)[None,:,None],axis=1)/lam
            row["test_context_response"]={"historical_rates":comp["historical_rates"].tolist(),
                "future_rates":quantiles(lam),"gate_shift":quantiles(comp["gate_shift"]),
                "single_mode_S5_over_S1":quantiles(shape5),"single_mode_S20_over_S1":quantiles(shape20)}
            if family != "free_horizon_context":
                beta=np.asarray(state["coefficients"])
                pathphi,_=basis(d["x"][test],state,family,tuple(range(1,21)))
                path=pathphi@beta
                delta=np.diff(np.column_stack((np.zeros(len(path)),path)),axis=1)
                row["sign_structure"]={"drift":float(beta[0]),"two_mode_coefficients":beta[1:].tolist(),
                    "coefficients_mixed_sign":bool(beta[1]*beta[2]<0),
                    "future_increment_sign_change_fraction":float(np.mean((delta.min(1)<0)&(delta.max(1)>0))),
                    "cumulative_future_change_sign_change_fraction":float(np.mean((path.min(1)<0)&(path.max(1)>0))),
                    "positive_cumulative_fraction_h1_h5_h20":np.mean(path[:,[0,4,19]]>0,axis=0).tolist()}
            if family == "signed_context_response":
                offstate=copy.deepcopy(state)
                offstate["gate"]=np.zeros_like(np.asarray(offstate["gate"])).tolist()
                offphi,_=basis(d["x"],offstate,family)
                plugin=risk(predict(offphi,offstate,d["scale"])[test],d["y"][test])
                offbeta,offscale,offkkt=independent_readout(offphi[final],yn[final],vn,alpha)
                offstate["coefficients"],offstate["head_column_scale"]=offbeta.tolist(),offscale.tolist()
                rehead=risk(predict(offphi,offstate,d["scale"])[test],d["y"][test])
                row["postfit_zero_gate_diagnostics"]={"same_alpha":alpha,"global_rates_kept_at_full_final_values":True,
                    "same_head_plugin":plugin,"training_only_exactly_refitted_head":rehead,
                    "refitted_head_coefficients":offbeta.tolist(),"refitted_head_optimality":offkkt,
                    "gate_effect_risk_gain_with_head_refit":rehead["risk"]-finalrisk["risk"],
                    "no_alpha_epoch_period_selection_or_gradient_optimization":True,
                    "scope": "post-fit descriptive counterfactual, not the separately validation-selected static_response or fixed_response arm"}
            rows.append(row)
        markets.append({"market":spec["market"],"role":spec["role"],"counts":d["manifest"]["counts"],
                        "all_frozen_arm_risks":all_risks,"response_families":rows})
    selectionchecks=[r for r in checks if r["subset"]=="train"]
    finalchecks=[r for r in checks if r["subset"]=="final"]
    output={"status":"PASS_READ_ONLY_REPLAY_AND_POSTFIT_DESCRIPTION","audit_author_role":"model/runner implementer; this is not a fully external blind audit",
            "new_nonlinear_fits":0,"new_gradient_steps":0,"new_hyperparameter_or_candidate_selection":0,
            "additional_descriptive_zero_gate_linear_readout_solves":len(markets),
            "no_new_p_values_or_period_filters":True,"source_sha256":EXPECTED,
            "frozen_protocol_sha256":run.digest(args.out/"PROTOCOL_BEFORE_RUN.json"),
            "diagnostic_script_sha256":run.digest(__file__),
            "counts":{"selection_checkpoints":len(selectionchecks),"final_checkpoints":len(finalchecks),
                "all_checkpoints":len(checks),"selection_nonlinear_optimizer_steps":sum(r["epochs_executed"] for r in selectionchecks),
                "selection_positive_gradient_steps":sum(r["positive_gradient_steps"] for r in selectionchecks),
                "final_nonlinear_optimizer_steps":sum(r["epochs_executed"] for r in finalchecks),
                "final_positive_gradient_steps":sum(r["positive_gradient_steps"] for r in finalchecks),
                "response_training_exact_head_solves":sum(r["exact_head_solves_in_fit"] for r in checks),
                "runner_fixed_dynamics_diagnostic_head_solves":len(finalchecks),
                "ridge_interaction_fits":len(markets)*9,"sklearn_price_ridge_fits":len(markets)*9},
            "maximum_gaps":{"validation_loss":max(r["validation_loss_max_gap"] for r in selectionchecks),
                "saved_native_prediction":max(r["saved_native_prediction_max_gap"] for r in finalchecks),
                "independent_linear_coefficients":max(r["coefficient_max_gap"] for r in checks),
                "column_scale":max(r["column_scale_max_gap"] for r in checks),
                "stored_scaled_kkt":max(r["stored_scaled_kkt_max"] for r in checks),
                "stored_scaled_kkt_relative":max(r["stored_scaled_kkt_relative"] for r in checks),
                "json_pt_state":max(r["json_pt_state_max_gap"] for r in checks)},
            "markets":markets,"checkpoint_checks":checks}
    assert len(checks)==188
    run.write_json(args.report,output)
    brief={"status":output["status"],"report":str(args.report),"counts":output["counts"],"maximum_gaps":output["maximum_gaps"],
           "main": [{"market":m["market"],**{k:r[k] for k in ["final","same_alpha_initial_dynamics_final_readout","nonlinear_learning_risk_gain","sign_structure","postfit_zero_gate_diagnostics"]}}
                    for m in markets for r in m["response_families"] if r["family"]=="signed_context_response"]}
    print(json.dumps(brief,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
