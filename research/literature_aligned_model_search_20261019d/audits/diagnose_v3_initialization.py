"""Post-result v3 initialization/learning replay of six fixed structure arms.

No fitting, optimizer update, candidate selection, significance test, or
checkpoint rewrite. Numeric self-developed state_dict values are exported
under the project owner's explicit authorization; binary checkpoints stay
at their original runtime locations.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import torch

STUDY=Path(__file__).resolve().parents[1]
SOURCE=STUDY/"source"
sys.path.insert(0,str(SOURCE))
import diagnose_learning_mechanism as helpers  # noqa: E402
import run_new_market_replication as replication  # noqa: E402

common=replication.common
FAMILIES=("liquidity_full","liquidity_readout","shock_size_only",
          "liquidity_unconstrained","allocation_constant","allocation_readout")


def core_and_input(model,x):
    if isinstance(model,replication.PriceEnergyView):
        return model.core,x[...,:4]
    return model,x


def parameter_record(model,initial_state):
    core=model.core if isinstance(model,replication.PriceEnergyView) else model
    groups={}
    for name,parameter in model.named_parameters():
        now=parameter.detach().double()
        initial=initial_state[name].double()
        delta=now-initial
        groups[name]={"n_parameters":now.numel(),"initial":initial.tolist(),"value":now.tolist(),
                      "change_l2":float(delta.norm()),"n_changed_exact":int(torch.count_nonzero(delta))}
    with torch.no_grad():
        rho=core.rates()
        w=core._log_transient_allocation(core.base_logits).exp()
        result={"parameters":sum(p.numel() for p in model.parameters()),"groups":groups,
                "parameter_change_l2":sum(z["change_l2"]**2 for z in groups.values())**.5,
                "rates":rho.tolist(),"base_allocation":[float(1-w.sum()),*w.tolist()],
                "beta":core.beta.tolist(),"state_dict":{k:v.tolist() for k,v in model.state_dict().items()}}
        if core.gate is not None and hasattr(core.gate,"coefficients"):
            gamma,delta=core.gate.coefficients()
            result.update({"effective_gamma":gamma.tolist(),"effective_delta":delta.tolist(),
                           "n_negative_gamma":int((gamma<0).sum()),"n_negative_delta":int((delta<0).sum())})
    return result


def diagnostics(model,x):
    core,view=core_and_input(model,x)
    return helpers.forward_diagnostics(core,view)


def outcome_scale(d,role):
    mask=d["masks"][role]
    y=d["y"][mask]
    moment=np.mean(y*y,axis=0)
    return {"n_origins":len(y),"mean_actual_change_native":np.mean(y,axis=0).tolist(),
            "actual_change_second_moment_native":moment.tolist(),
            "actual_change_rms_native":np.sqrt(moment).tolist(),
            "actual_change_centered_variance_native":np.var(y,axis=0).tolist(),
            "rw_loss_in_fixed_training_risk_units_by_horizon":(moment/d["variance"]).tolist(),
            "rw_loss_in_fixed_training_risk_units_mean":float(np.mean(moment/d["variance"]))}


def macro(cells,family):
    initial=np.array([c["families"][family]["initial_risk"]["risk_over_rw"] for c in cells])
    fitted=np.array([c["families"][family]["fitted_risk"]["risk_over_rw"] for c in cells])
    gain=initial-fitted
    return {"markets":[c["market"] for c in cells],"n_market_cells":len(cells),
            "initial_mean_mse_over_rw":float(initial.mean()),"fitted_mean_seed_mse_over_rw":float(fitted.mean()),
            "initial_prior_improvement_from_rw_ratio_points":float(1-initial.mean()),
            "learning_improvement_from_initial_ratio_points":float(gain.mean()),
            "total_improvement_from_rw_ratio_points":float(1-fitted.mean()),
            "n_markets_improved_from_initial":int((gain>1e-12).sum()),
            "n_markets_worsened_from_initial":int((gain<-1e-12).sum()),
            "n_markets_unchanged":int((np.abs(gain)<=1e-12).sum()),
            "learning_net_gain_contributions":{c["market"]:{"gain_ratio_points":float(g),"fraction_of_net_gain":float(g/gain.sum()) if abs(gain.sum())>1e-15 else None} for c,g in zip(cells,gain)},
            "initial_prediction_energy":float(np.mean([c["families"][family]["initial_risk"]["forecast_energy_penalty"] for c in cells])),
            "fitted_prediction_energy":float(np.mean([c["families"][family]["fitted_risk"]["forecast_energy_penalty"] for c in cells])),
            "initial_outcome_alignment_credit":float(np.mean([c["families"][family]["initial_risk"]["outcome_alignment_credit"] for c in cells])),
            "fitted_outcome_alignment_credit":float(np.mean([c["families"][family]["fitted_risk"]["outcome_alignment_credit"] for c in cells]))}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-dir",type=Path,default=Path("/tmp/carbon-literature-aligned-runs-20261019d/new_market_v3"))
    parser.add_argument("--out",type=Path,default=STUDY/"methods/V3_INITIALIZATION_AND_LEARNING_ZH.json")
    args=parser.parse_args()
    torch.set_num_threads(1)
    protocol_path=args.run_dir/"PROTOCOL_BEFORE_RUN.json"
    protocol=json.loads(protocol_path.read_text())
    frozen=protocol["source_sha256"]
    assert all(helpers.digest(SOURCE/k)==v for k,v in frozen.items())
    files_read={"PROTOCOL_BEFORE_RUN.json":helpers.digest(protocol_path)}
    cells,checkpoints=[],[]
    validation_gap=prediction_gap=initial_seed_gap=0.0
    for market in protocol["markets"]:
        d=replication.prepare_replication(market)
        cell=f"{market}_{replication.FOLD}"
        directory=args.run_dir/cell
        manifest_path=directory/"INPUT_MANIFEST.json"
        assert json.loads(manifest_path.read_text())==d["manifest"]
        files_read[str(manifest_path.relative_to(args.run_dir))]=helpers.digest(manifest_path)
        mask=d["masks"]["test"]
        x_test,y_test=d["x"][mask],d["y"][mask]
        cell_record={"cell":cell,"market":market,"role":d["manifest"]["role"],
                     "counts":d["manifest"]["counts"],"training_rms":d["scale"].tolist(),
                     "training_loss_second_moment_native":d["variance"].tolist(),
                     "outcome_scale_by_role":{role:outcome_scale(d,role) for role in d["masks"]},"families":{}}
        initial_predictions={}
        for family in FAMILIES:
            directory_f=directory/family
            choice_path=directory_f/"CHOICE_BEFORE_TEST.json"
            choice=json.loads(choice_path.read_text())
            files_read[str(choice_path.relative_to(args.run_dir))]=helpers.digest(choice_path)
            initial_model=replication.make_replication_model(family,17)
            init_state=copy.deepcopy(initial_model.state_dict())
            initial_test,initial_diag=diagnostics(initial_model,x_test)
            initial_predictions[family]=initial_test
            for seed in [29,43]:
                other=common.predict(replication.make_replication_model(family,seed),x_test)
                gap=float(np.max(np.abs(initial_test-other)))
                initial_seed_gap=max(initial_seed_gap,gap)
                assert gap==0
            initial_risk=helpers.risk_components(initial_test*d["scale"][0],y_test)
            initial_parameters=parameter_record(initial_model,init_state)
            predictions_path=directory_f/"predictions.npz"
            with np.load(predictions_path) as saved:
                original=saved["predictions"].copy()
                assert np.array_equal(saved["observed"],y_test)
                assert np.array_equal(saved["origins"],d["origins"][mask])
                assert np.array_equal(saved["horizons"],common.HORIZONS)
                assert np.array_equal(saved["seeds"],common.SEEDS)
            files_read[str(predictions_path.relative_to(args.run_dir))]=helpers.digest(predictions_path)
            final_risk=helpers.risk_components(original,y_test)
            family_records=[]
            paths=sorted(directory_f.glob("selection_lr*.json"))+sorted(directory_f.glob("refit_seed*.json"))
            assert len(paths)==5
            for path in paths:
                receipt=json.loads(path.read_text())
                checkpoint=path.with_suffix(".pt")
                assert helpers.digest(checkpoint)==receipt["checkpoint_sha256"]
                files_read[str(path.relative_to(args.run_dir))]=helpers.digest(path)
                files_read[str(checkpoint.relative_to(args.run_dir))]=helpers.digest(checkpoint)
                model=replication.make_replication_model(family,receipt["seed"])
                same_seed_initial=copy.deepcopy(model.state_dict())
                role="train" if receipt["mode"]=="train" else "final"
                fit_mask=d["masks"][role]
                fit_x=d["x"][fit_mask]
                initial_fit_loss=helpers.exact_loss(common.predict(model,fit_x),d,fit_mask)
                model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True))
                model.eval()
                params=parameter_record(model,same_seed_initial)
                fitted_loss=helpers.exact_loss(common.predict(model,fit_x),d,fit_mask)
                history=receipt["history"]
                record={"market":market,"cell":cell,"family":family,"file":str(path.relative_to(args.run_dir)),
                        "mode":receipt["mode"],"seed":receipt["seed"],"learning_rate":receipt["learning_rate"],
                        "selected_epoch":receipt["selected_epoch"],"epochs_executed":receipt["epochs_executed"],
                        "is_original_selected_lr":receipt["learning_rate"]==choice["learning_rate"],
                        "selected_at_maximum_epoch_budget":receipt["selected_epoch"]==protocol["maximum_epochs"],
                        "gradient_positive_steps":receipt["gradient_positive_steps"],
                        "max_gradient_norm_before_clip":receipt["max_gradient_norm_before_clip"],
                        "optimization_parameter_change_before_selection_restore":receipt["optimization_parameter_change_l2_before_selection"],
                        "initial_exact_fitting_loss":initial_fit_loss,"checkpoint_exact_fitting_loss":fitted_loss,
                        "fitting_loss_relative_change_percent":100*(fitted_loss/initial_fit_loss-1),
                        "logged_first_epoch_minibatch_loss":history[0]["train_loss"] if history else None,
                        "logged_last_epoch_minibatch_loss":history[-1]["train_loss"] if history else None,
                        "parameters":params,"checkpoint_sha256":receipt["checkpoint_sha256"]}
                if receipt["mode"]=="train":
                    val_mask=d["masks"]["val"]
                    val_pred=common.predict(model,d["x"][val_mask])
                    val_loss=helpers.exact_loss(val_pred,d,val_mask)
                    gap=abs(val_loss-receipt["selected_validation_loss"])
                    validation_gap=max(validation_gap,gap)
                    assert gap<1e-6
                    record.update({"initial_validation_loss":receipt["initial_validation_loss"],
                                   "checkpoint_validation_loss":val_loss,"validation_replay_absolute_gap":gap,
                                   "validation_loss_relative_change_percent":100*(val_loss/receipt["initial_validation_loss"]-1)})
                else:
                    prediction,diag=diagnostics(model,x_test)
                    native=prediction*d["scale"][0]
                    seed_index=common.SEEDS.index(receipt["seed"])
                    gap=float(np.max(np.abs(native-original[seed_index])))
                    prediction_gap=max(prediction_gap,gap)
                    assert gap<1e-6
                    _,fit_diag=diagnostics(model,fit_x)
                    record.update({"original_prediction_replay_max_abs_gap_native_units":gap,
                                   "test_risk":helpers.risk_components(original[seed_index],y_test),
                                   "fitting_state_and_gate":fit_diag,"test_state_and_gate":diag,
                                   "forecast_shift_from_initial_rms_in_training_risk_units":float(np.sqrt(np.mean((prediction.astype(float)-initial_test)**2/(d["variance"]/d["scale"][0]**2))))})
                if receipt["selected_epoch"]==0:
                    assert params["parameter_change_l2"]==0
                checkpoints.append(record)
                family_records.append(record)
            selected=[r for r in family_records if r["mode"]=="train" and r["is_original_selected_lr"]][0]
            refits=[r for r in family_records if r["mode"]=="final"]
            assert abs(final_risk["risk_over_rw"]-json.loads((directory_f/"SUMMARY.json").read_text())["mean_seed_mse_over_rw"])<1e-12
            fr={"initial_risk":initial_risk,"fitted_risk":final_risk,
                "initial_parameters_seed17":initial_parameters,"initial_test_state_and_gate":initial_diag,
                "learning_improvement_from_initial_ratio_points":initial_risk["risk_over_rw"]-final_risk["risk_over_rw"],
                "selected_lr":choice["learning_rate"],"selected_epoch":choice["selected_epoch"],
                "selected_at_maximum_epoch_budget":choice["selected_epoch"]==protocol["maximum_epochs"],
                "selected_training_loss_relative_change_percent":selected["fitting_loss_relative_change_percent"],
                "selected_validation_loss_relative_change_percent":selected["validation_loss_relative_change_percent"],
                "refit_training_loss_relative_change_percent_mean":float(np.mean([r["fitting_loss_relative_change_percent"] for r in refits])),
                "refit_rates_mean":np.mean([r["parameters"]["rates"] for r in refits],axis=0).tolist(),
                "refit_base_allocation_mean":np.mean([r["parameters"]["base_allocation"] for r in refits],axis=0).tolist(),
                "refit_beta_mean":np.mean([r["parameters"]["beta"] for r in refits],axis=0).tolist(),
                "refit_gate_adjustment_endpoint_mean":np.mean([r["test_state_and_gate"]["endpoint_gate_adjustment"]["mean"] for r in refits],axis=0).tolist(),
                "refit_gate_adjustment_endpoint_std_mean":np.mean([r["test_state_and_gate"]["endpoint_gate_adjustment"]["std"] for r in refits],axis=0).tolist(),
                "refit_endpoint_actual_allocation_mean":np.mean([r["test_state_and_gate"]["endpoint_actual_allocation"]["mean"] for r in refits],axis=0).tolist(),
                "refit_readout_multiplier_mean":np.mean([r["test_state_and_gate"]["readout_multiplier"]["mean"] for r in refits],axis=0).tolist(),
                "n_refits_with_positive_steps":sum(r["gradient_positive_steps"]>0 for r in refits),
                "n_refits_with_all_parameter_groups_changed":sum(all(g["change_l2"]>0 for g in r["parameters"]["groups"].values()) for r in refits)}
            if "effective_gamma" in refits[0]["parameters"]:
                fr["refit_gamma_mean"]=np.mean([r["parameters"]["effective_gamma"] for r in refits],axis=0).tolist()
                fr["refit_delta_mean"]=np.mean([r["parameters"]["effective_delta"] for r in refits],axis=0).tolist()
            cell_record["families"][family]=fr
        assert np.array_equal(initial_predictions["liquidity_full"],initial_predictions["liquidity_unconstrained"])
        assert np.array_equal(initial_predictions["allocation_constant"],initial_predictions["allocation_readout"])
        cell_record["initial_full_vs_readout_forecast_max_abs_gap_normalized"]=float(np.max(np.abs(initial_predictions["liquidity_full"]-initial_predictions["liquidity_readout"])))
        cells.append(cell_record)
        print(f"READ_ONLY_V3_DIAGNOSED {market}",flush=True)
    primary=[c for c in cells if c["market"] in protocol["primary_markets"]]
    stress=[c for c in cells if c["market"] not in protocol["primary_markets"]]
    assert len(primary)==2 and len(stress)==1
    assert all(helpers.digest(SOURCE/k)==v for k,v in frozen.items())
    learning_counts={}
    for family in FAMILIES:
        records=[r for r in checkpoints if r["family"]==family]
        selection=[r for r in records if r["mode"]=="train"]
        refits=[r for r in records if r["mode"]=="final"]
        learning_counts[family]={"selection_tasks":len(selection),"refit_constructions":len(refits),
            "selection_tasks_with_positive_steps":sum(r["gradient_positive_steps"]>0 for r in selection),
            "refit_tasks_with_positive_steps":sum(r["gradient_positive_steps"]>0 for r in refits),
            "selection_positive_steps":sum(r["gradient_positive_steps"] for r in selection),
            "refit_positive_steps":sum(r["gradient_positive_steps"] for r in refits),
            "refits_all_parameter_groups_changed":sum(all(g["change_l2"]>0 for g in r["parameters"]["groups"].values()) for r in refits),
            "selected_cells_epoch_zero":sum(c["families"][family]["selected_epoch"]==0 for c in cells),
            "selected_cells_at_maximum_budget":sum(c["families"][family]["selected_at_maximum_epoch_budget"] for c in cells)}
    report={"status":"PASS_READ_ONLY_REPLAY","created_utc":datetime.now(timezone.utc).isoformat(),
            "scope":"Post-result v3 initialization/learning diagnostic, separate from the frozen v3 primary statistical analysis.",
            "authorship":"Model implementer performed this checkpoint replay; not external blind review.",
            "authorization":"Project owner authorized public numeric self-developed state_dict export; original binary checkpoints are read only and remain in runtime storage.",
            "new_fit_count":0,"new_optimizer_step_count":0,"new_candidate_selection_count":0,"new_significance_test_count":0,
            "aggregation":"Equal Beijing/Tianjin primary-market and H1/H5/H20 mean of individual-seed losses; Chongqing is a separate stress test. No pooled three-market primary statistic.",
            "risk_decomposition":"risk/RW = 1 + E[f^2]/E[y^2] - 2E[fy]/E[y^2] within each seed and horizon; then equal means.",
            "scale_interpretation":"All exact training/validation risks reuse fixed training horizon second moments. Raw validation losses can differ greatly between markets because outcome scales shift; they are not comparable native-currency errors.",
            "initialization_interpretation":"Each family has its own fixed initial function, verified identical across three seeds; allocation_readout hidden initialization is seed-specific although zero output gate makes initial forecasts identical.",
            "source_sha256":frozen,"source_hashes_unchanged":True,
            "diagnostic_script_sha256":helpers.digest(__file__),"helper_script_sha256":helpers.digest(Path(helpers.__file__)),
            "runtime":{"python_executable":sys.executable,"python":platform.python_version(),"numpy":np.__version__,"torch":torch.__version__},
            "checks":{"checkpoint_count":len(checkpoints),"maximum_validation_loss_replay_gap":validation_gap,
                      "maximum_original_prediction_replay_gap_native_units":prediction_gap,"maximum_initial_prediction_gap_across_seeds":initial_seed_gap,
                      "full_and_unconstrained_initial_predictions_equal":True,"constant_and_v1_readout_initial_predictions_equal":True},
            "primary_macro":{f:macro(primary,f) for f in FAMILIES},
            "chongqing_stress_only":{f:macro(stress,f) for f in FAMILIES},
            "learning_counts_all_three_markets":learning_counts,"cells":cells,"checkpoints":checkpoints,
            "source_artifact_hashes_relative_to_original_run_dir":files_read}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps({"out":str(args.out),"checks":report["checks"],"primary_macro":report["primary_macro"],
                      "chongqing_stress_only":report["chongqing_stress_only"],"learning_counts":learning_counts},indent=2,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
