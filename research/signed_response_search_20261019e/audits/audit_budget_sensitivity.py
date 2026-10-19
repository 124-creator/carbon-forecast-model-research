"""Read-only audit of the frozen max320 development sensitivity.

Uses independently written NumPy forward/augmented linear least-squares
helpers. No model optimizer is called. Stored selection is reconstructed only
to verify the frozen rule; it is never changed or used to launch another fit.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

import diagnose_signed_response_learning as diagnostic

run = diagnostic.run
STUDY = Path(__file__).resolve().parents[1]
PLAN_SHA = "ac1117d24f395cb7b3331ffd2e4de3d6e936f9b9891748204c4aa8df394de72c"
FAMILIES = tuple(f for f in run.RESPONSE_FAMILIES if f != "fixed_response")


def history_rule(receipt):
    config = receipt["config"]
    history = receipt["history"]
    best, best_epoch, stale = float("inf"), 0, 0
    for position, item in enumerate(history):
        assert item["epoch"] == position
        loss = item["validation_loss"]
        if loss < best-config["selection_tolerance"]:
            best, best_epoch, stale = loss, position, 0
        else:
            stale += 1
        if position < len(history)-1:
            assert not (position >= config["minimum_epochs"] and stale >= config["patience"])
    assert best_epoch == receipt["selected_epoch"]
    assert best == receipt["validation_loss"]
    assert len(history)-1 == receipt["epochs_executed"]
    assert receipt["epochs_executed"] <= config["maximum_epochs"]
    assert receipt["epochs_executed"] == config["maximum_epochs"] or (
        receipt["epochs_executed"] >= config["minimum_epochs"] and stale >= config["patience"])
    gradients = [item["gradient_norm_before_clip"] for item in history if "gradient_norm_before_clip" in item]
    assert len(gradients) == receipt["epochs_executed"]
    assert np.isfinite(gradients).all()
    assert sum(g > 0 for g in gradients) == receipt["positive_gradient_steps"]
    return {"best_epoch": best_epoch, "best_validation_loss": best,
            "termination": "cap" if receipt["epochs_executed"] == config["maximum_epochs"] else "patience"}


def prefix_check(old, new):
    differences = {k: [old["config"][k], new["config"][k]] for k in old["config"]
                   if old["config"][k] != new["config"][k]}
    assert differences == {"maximum_epochs": [80, 320]}
    length = min(len(old["history"]), len(new["history"]))
    fields = ["training_data_loss", "training_profiled_objective", "validation_loss",
              "historical_rates", "gradient_norm_before_clip", "linear_normal_equation_max_residual"]
    gaps = {}
    for field in fields:
        pairs = [(a[field], b[field]) for a,b in zip(old["history"][:length],new["history"][:length])
                 if field in a and field in b]
        gaps[field] = max((float(np.max(np.abs(np.asarray(a)-np.asarray(b)))) for a,b in pairs), default=0.)
    assert max(gaps.values()) < 1e-9
    if old["epochs_executed"] < 80:
        assert new["epochs_executed"] == old["epochs_executed"]
    return {"epochs_compared_including_epoch0": length, "fit_config_differences": differences,
            "maximum_gaps": gaps, "original_epochs": old["epochs_executed"],
            "supplementary_epochs": new["epochs_executed"]}


def point_metrics(npz):
    return diagnostic.risk(npz["predictions"][0],npz["observed"])


def checkpoint_identity(path, receipt):
    state = diagnostic.state_from_pt(path,receipt["family"])
    gap = max(float(np.max(np.abs(np.asarray(v)-np.asarray(receipt["state_dict"][k])))) for k,v in state.items())
    assert gap == 0
    return gap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original",type=Path,default=Path("/tmp/carbon-signed-response-runs-20261019e/core"))
    parser.add_argument("--budget",type=Path,default=Path("/tmp/carbon-signed-response-runs-20261019e/budget320"))
    parser.add_argument("--report",type=Path,default=STUDY/"audits/BUDGET_SENSITIVITY_AUDIT.json")
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Use a new audit destination")
    torch.set_num_threads(1)
    addendum_path = STUDY/"audits/BUDGET_SENSITIVITY_BEFORE_RUN.json"
    assert run.digest(addendum_path) == PLAN_SHA
    addendum = diagnostic.read(addendum_path)
    actual_sources = {name:run.digest(STUDY/"source"/name) for name in addendum["source_sha256"]}
    assert actual_sources == addendum["source_sha256"]
    run.check_old_sources()
    completion = diagnostic.read(args.budget/"EXECUTION_COMPLETE.json")
    assert completion["status"] == "completed" and completion["source_hashes_unchanged"] and completion["plan_hash_unchanged"]
    assert completion["summary_rows"] == 10 and completion["development_sensitivity_only"]
    protocol = diagnostic.read(args.budget/"PROTOCOL_BEFORE_RUN.json")
    original = diagnostic.read(args.original/"PROTOCOL_BEFORE_RUN.json")
    assert protocol["source_sha256"] == actual_sources and protocol["addendum_sha256"] == PLAN_SHA
    assert run.digest(args.original/"PROTOCOL_BEFORE_RUN.json") == addendum["original_core_protocol_sha256"]
    assert protocol["families"] == list(FAMILIES)
    original_specs = {spec["market"]:spec for spec in original["datasets"]}
    checkpoints, prefixes, rows = [], [], []
    for spec in protocol["datasets"]:
        assert spec["role"] == "development"
        oldspec = original_specs[spec["market"]]
        assert all(spec[k] == v for k,v in oldspec.items() if k != "role")
        assert spec["original_role"] == oldspec["role"]
        d, oldd = run.prepare(spec), run.prepare(oldspec)
        for key in ("x","y","origins","scale","variance","dates","price","volume"):
            assert np.array_equal(d[key],oldd[key],equal_nan=True) if np.asarray(d[key]).dtype.kind not in "Mm" else np.array_equal(d[key],oldd[key])
        assert all(np.array_equal(d["masks"][k],oldd["masks"][k]) for k in d["masks"])
        folder = args.budget/f"{spec['market']}_{run.FOLD}"
        oldfolder = args.original/f"{spec['market']}_{run.FOLD}"
        for family in FAMILIES:
            dest, olddest = folder/family, oldfolder/family
            selected = diagnostic.read(dest/"SELECTION.json")
            oldselected = diagnostic.read(olddest/"SELECTION.json")
            assert len(selected["trials"]) == 16
            pairs = {(r["alpha"],r["learning_rate"]) for r in selected["trials"]}
            assert pairs == {(a,l) for a in run.MEAN_ALPHAS for l in run.LRS}
            trials_from_receipts = []
            for trial in selected["trials"]:
                receipt = diagnostic.read(dest/trial["receipt"])
                prior = diagnostic.read(olddest/trial["receipt"])
                history_rule(receipt)
                pref = prefix_check(prior,receipt)
                pref.update({"market":spec["market"],"family":family,"alpha":trial["alpha"],"lr":trial["learning_rate"]})
                prefixes.append(pref)
                assert receipt["config"]["alpha"] == trial["alpha"] and receipt["config"]["learning_rate"] == trial["learning_rate"]
                for key in ("selected_epoch","validation_loss","epochs_executed","positive_gradient_steps"):
                    assert trial[key] == receipt[key]
                trials_from_receipts.append(trial)
                check,_,_ = diagnostic.verify_one(receipt,d,"train")
                check["json_pt_state_max_gap"] = checkpoint_identity(dest/Path(trial["receipt"]).with_suffix(".pt"),receipt)
                check.update({"market":spec["market"],"receipt":str(dest/trial["receipt"])})
                checkpoints.append(check)
            reconstructed = min(trials_from_receipts,key=lambda r:(r["validation_loss"],r["alpha"],r["learning_rate"],r["selected_epoch"]))
            assert reconstructed == selected["selected"]
            receipt = diagnostic.read(dest/"REFIT.json")
            previous_refit = diagnostic.read(olddest/"REFIT.json")
            assert receipt["selected_epoch"] == receipt["epochs_executed"] == selected["selected"]["selected_epoch"]
            assert receipt["config"]["alpha"] == selected["selected"]["alpha"] and receipt["config"]["learning_rate"] == selected["selected"]["learning_rate"]
            gradient_records=[r["gradient_norm_before_clip"] for r in receipt["history"] if "gradient_norm_before_clip" in r]
            assert len(gradient_records) == receipt["epochs_executed"]
            assert receipt["positive_gradient_steps"] == sum(g>0 for g in gradient_records)
            new_npz, old_npz = np.load(dest/"predictions.npz"), np.load(olddest/"predictions.npz")
            for key in ("observed","origins","horizons"):
                assert np.array_equal(new_npz[key],old_npz[key])
            assert np.array_equal(new_npz["observed"],d["y"][d["masks"]["test"]])
            check,pred,phi = diagnostic.verify_one(receipt,d,"final",new_npz["predictions"][0])
            check["json_pt_state_max_gap"] = checkpoint_identity(dest/"refit.pt",receipt)
            check.update({"market":spec["market"],"receipt":str(dest/"REFIT.json")})
            checkpoints.append(check)
            oldrisk, newrisk = point_metrics(old_npz), point_metrics(new_npz)
            summary = diagnostic.read(dest/"SUMMARY.json")
            assert abs(summary["mean_mse_over_rw"]-newrisk["risk"]) < 1e-12 and summary["role"] == "development"
            initial = run.SignedResponse(family)
            initial_state = {k:v.detach().numpy().tolist() for k,v in initial.state_dict().items()}
            initial_phi,_ = diagnostic.basis(d["x"],initial_state,family)
            m = d["masks"]
            yn,vn = d["y"]/d["scale"],d["variance"]/d["scale"]**2
            alpha = selected["selected"]["alpha"]
            beta,ss,_ = diagnostic.independent_readout(initial_phi[m["final"]],yn[m["final"]],vn,alpha)
            initial_state["coefficients"],initial_state["head_column_scale"] = beta.tolist(),ss.tolist()
            ip = diagnostic.predict(initial_phi,initial_state,d["scale"])[m["test"]]
            initgap = float(np.max(np.abs(ip-new_npz["fixed_dynamics_refit_readout_predictions"][0])))
            assert initgap < 1e-8
            initrisk = diagnostic.risk(ip,new_npz["observed"])
            state = receipt["state_dict"]
            _,comp = diagnostic.basis(d["x"][m["test"]],state,family)
            final_beta = np.asarray(state["coefficients"])
            row = {"market":spec["market"],"role":"post-exposure development sensitivity","family":family,
                   "old_choice":oldselected["selected"],"budget_choice":selected["selected"],
                   "old_risk":oldrisk,"budget_risk":newrisk,
                   "budget_risk_improvement_pp":100*(oldrisk["risk"]-newrisk["risk"]),
                   "same_alpha_initial_dynamics_final_readout_risk":initrisk,
                   "budget_nonlinear_learning_gain_pp":100*(initrisk["risk"]-newrisk["risk"]),
                   "initial_reference_replay_max_gap":initgap,
                   "final_positive_gradient_steps":receipt["positive_gradient_steps"],
                   "final_state_dict":state,"old_final_state_dict":previous_refit["state_dict"],
                   "final_parameter_change_from_initial":receipt["selected_nonlinear_parameter_max_change"],
                   "old_global_rates":diagnostic.rates(previous_refit["state_dict"]["rate_raw"]).tolist(),
                   "budget_global_rates":comp["historical_rates"].tolist(),
                   "budget_future_rates":diagnostic.quantiles(comp["future_rates"]),
                   "still_selected_cap320":selected["selected"]["selected_epoch"]==320}
            if family != "free_horizon_context":
                row["mode_coefficients"] = final_beta[1:].tolist()
                row["mixed_sign_mode_coefficients"] = bool(final_beta[1]*final_beta[2]<0)
            if family == "signed_context_response":
                off = copy.deepcopy(state)
                off["gate"] = np.zeros_like(off["gate"]).tolist()
                offphi,_ = diagnostic.basis(d["x"],off,family)
                plugin = diagnostic.risk(diagnostic.predict(offphi,off,d["scale"])[m["test"]],new_npz["observed"])
                offbeta,offss,optimality = diagnostic.independent_readout(offphi[m["final"]],yn[m["final"]],vn,alpha)
                off["coefficients"],off["head_column_scale"] = offbeta.tolist(),offss.tolist()
                rehead = diagnostic.risk(diagnostic.predict(offphi,off,d["scale"])[m["test"]],new_npz["observed"])
                row["postfit_same_alpha_zero_gate"] = {"same_head":plugin,"training_only_refitted_head":rehead,
                    "gate_gain_pp":100*(rehead["risk"]-newrisk["risk"]),"refitted_head":offbeta.tolist(),
                    "optimality":optimality,"additional_linear_solves":1,
                    "scope":"post-fit descriptive; no new hyperparameter choice, gradient fit or formal comparison"}
            rows.append(row)
    assert len(checkpoints)==170 and len(prefixes)==160 and len(rows)==10
    selection_checks=[r for r in checkpoints if r["subset"]=="train"]
    final_checks=[r for r in checkpoints if r["subset"]=="final"]
    maxprefix={key:max(r["maximum_gaps"][key] for r in prefixes) for key in prefixes[0]["maximum_gaps"]}
    output={"status":"PASS_READ_ONLY_BUDGET_SENSITIVITY_AUDIT",
            "analysis_role":"Post-exposure development sensitivity for both markets; original confirmation unchanged",
            "audit_author":"Model and wrapper implementer; independent numerical formula but not a fully external blind review",
            "source_sha256":actual_sources,"addendum_sha256":PLAN_SHA,
            "original_protocol_sha256":run.digest(args.original/"PROTOCOL_BEFORE_RUN.json"),
            "budget_protocol_sha256":run.digest(args.budget/"PROTOCOL_BEFORE_RUN.json"),
            "audit_script_sha256":run.digest(__file__),
            "formula_helper_sha256":run.digest(diagnostic.__file__),
            "new_gradient_fits":0,"new_gradient_steps":0,"new_parameter_searches":0,
            "new_p_values":0,"postfit_zero_gate_descriptive_linear_solves":2,
            "same_data_labels_origins_scales_variance":True,"only_configuration_change":"maximum_epochs80to320",
            "counts":{"selection_checkpoints":160,"final_checkpoints":10,"all_checkpoints":170,
                      "matched_history_prefixes":160,"total_common_prefix_epoch_records":sum(r["epochs_compared_including_epoch0"] for r in prefixes),
                      "selection_optimizer_steps":sum(r["epochs_executed"] for r in selection_checks),
                      "selection_positive_gradient_steps":sum(r["positive_gradient_steps"] for r in selection_checks),
                      "final_optimizer_steps":sum(r["epochs_executed"] for r in final_checks),
                      "final_positive_gradient_steps":sum(r["positive_gradient_steps"] for r in final_checks),
                      "training_exact_head_solves":sum(r["exact_head_solves_in_fit"] for r in checkpoints),
                      "runner_initial_dynamics_diagnostic_solves":10},
            "maximum_gaps":{"common_history_prefix":maxprefix,
                "validation_loss":max(r["validation_loss_max_gap"] for r in selection_checks),
                "saved_native_prediction":max(r["saved_native_prediction_max_gap"] for r in final_checks),
                "linear_coefficients":max(r["coefficient_max_gap"] for r in checkpoints),
                "column_scale":max(r["column_scale_max_gap"] for r in checkpoints),
                "stored_scaled_kkt":max(r["stored_scaled_kkt_max"] for r in checkpoints),
                "json_pt_state":max(r["json_pt_state_max_gap"] for r in checkpoints)},
            "still_selected_cap320":[{"market":r["market"],"family":r["family"]} for r in rows if r["still_selected_cap320"]],
            "results":rows,"prefix_checks":prefixes,"checkpoint_checks":checkpoints,
            "interpretation_limit":"This one finite extension cannot eliminate all possible optimization inadequacy. Existing addendum stop_rule forbids another budget expansion; no original claim is rewritten."}
    run.write_json(args.report,output)
    print(json.dumps({"status":output["status"],"report":str(args.report),"counts":output["counts"],
                      "maximum_gaps":output["maximum_gaps"],"still_selected_cap320":output["still_selected_cap320"],
                      "results":[{k:r[k] for k in ("market","family","old_risk","budget_risk","budget_risk_improvement_pp","budget_choice","budget_nonlinear_learning_gain_pp","still_selected_cap320")} for r in rows]},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
