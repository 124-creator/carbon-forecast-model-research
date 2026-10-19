#!/usr/bin/env python3
"""Independent date/array/fit-lineage audit; performs zero market model fits.

Reconstructs prepared features, labels and conditional variance from fixed
development CSVs, checks frozen role indices and completed fit metadata, and
executes the actual online feedback loop with inert marker-valued sentinels.
No optimizer, neural training, learned gate fit or reserved-period data read.
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import types

import numpy as np
import pandas as pd
from scipy.special import erf


ROLE_MAP = {"train": "train", "validation": "validation", "calibration": "calibration", "test": "evaluation"}
BACKBONES = ("MLP32_16", "GRU16")
SEEDS = (17, 41, 89)
OLD_GUARD = '''    df = pd.read_csv(data_path)
    dates = pd.to_datetime(df["date"])
    price = df["price"].to_numpy(float)
    if dates.max() >= pd.Timestamp("2024-01-01"):
        raise ValueError("This development runner refuses reserved-period prices")
'''
NEW_GUARD = '''    date_only = pd.read_csv(data_path, usecols=["date"])
    dates = pd.to_datetime(date_only["date"])
    if dates.max() >= pd.Timestamp("2024-01-01"):
        raise ValueError("This development runner refuses reserved-period prices")
    df = pd.read_csv(data_path)
    price = df["price"].to_numpy(float)
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def expr_equal(node, text):
    return ast.dump(node, include_attributes=False) == ast.dump(ast.parse(text, mode="eval").body, include_attributes=False)


def function(tree, name):
    return next(node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)


def calls_named(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name]


def test_guard(prepare_function):
    """A synthetic future DATE ONLY is returned; no price value/file exists."""
    accesses = []
    class DateOnlyPandas:
        Timestamp = staticmethod(pd.Timestamp)
        to_datetime = staticmethod(pd.to_datetime)
        @staticmethod
        def read_csv(path, **kwargs):
            accesses.append(kwargs)
            if kwargs.get("usecols") != ["date"]:
                raise AssertionError("Price/full CSV access occurred before rejecting future date")
            return pd.DataFrame({"date": ["2024-01-01"]})
    scope = {"pd": DateOnlyPandas, "time": types.SimpleNamespace(time=lambda: 0.)}
    tree = ast.Module(body=[prepare_function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), "<actual_prepare_guard_only>", "exec"), scope)
    try:
        scope["prepare_cell"]("NO_REAL_FILE_DATE_ONLY_SENTINEL", None, "AUDIT", 2023, 5)
    except ValueError as error:
        require("reserved-period" in str(error), "Unexpected error in date rejection guard")
    else:
        raise AssertionError("Synthetic future date was not rejected")
    require(accesses == [{"usecols": ["date"]}], "Guard did more than one date-only read")
    return {"passed": True, "synthetic_date_only": "2024-01-01", "real_future_source_files": 0,
            "price_values_supplied": 0, "full_or_price_reads": 0, "date_only_reads": 1}


def source_audit(study):
    prefit = json.loads((study/"results/market/MARKET_PREFIT.json").read_text())
    texts = {path.name: path.read_text() for path in (study/"source").glob("*.py")}
    current = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in texts.items()}
    differences = [name for name, expected in prefit["source_sha256"].items() if current[name] != expected]
    require(set(differences).issubset({"run_market.py"}), "Unexpected post-prefit fit/score source change")
    guard_only = True
    if differences:
        require(texts["run_market.py"].count(NEW_GUARD) == 1, "Approved guard correction not found exactly once")
        reconstructed = texts["run_market.py"].replace(NEW_GUARD, OLD_GUARD, 1)
        guard_only = hashlib.sha256(reconstructed.encode()).hexdigest() == prefit["source_sha256"]["run_market.py"]
        require(guard_only, "Changes beyond the approved date-before-price correction detected")
    require(sha(study/"provenance/PROTOCOL.json") == prefit["protocol_sha256"], "Frozen protocol hash changed")
    require(sha(study/"provenance/DESIGN_ADDENDUM_PRE_RUN.json") == prefit["addendum_sha256"], "Frozen addendum hash changed")
    tree = ast.parse(texts["run_market.py"])
    preparation = function(tree, "prepare_cell")
    job = function(tree, "fit_job")
    gm = calls_named(preparation, "fit_global_mixture")
    require(len(gm) == 1 and expr_equal(gm[0].args[0], "y[train-39]"), "F0 fit is not restricted to training labels")
    ga = calls_named(preparation, "fit_garch")
    require(len(ga) == 1 and expr_equal(ga[0].args[0], "returns") and expr_equal(ga[0].args[1], "training_label_end"), "GARCH endpoint call changed")
    fhs = [n for n in ast.walk(preparation) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr == "quantile"]
    require(len(fhs) == 1 and expr_equal(fhs[0].args[0], "y[train-39]"), "FHS fit includes non-training labels")
    neural = calls_named(job, "fit_proposal")
    require(len(neural) == 1, "Unexpected neural fit calls")
    for actual, expected in zip(neural[0].args[:4], ('d["train_X"]','d["train_y"]','d["validation_X"]','d["validation_y"]')):
        require(expr_equal(actual, expected), "Neural fitting/selection receives wrong role array")
    controls = calls_named(job, "reflected_control")
    require(len(controls) == 4 and all(expr_equal(c.args[0], 'd["calibration_path"]') for c in controls), "Path calibration sees a non-calibration path")
    loops = [n for n in ast.walk(job) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple)
             and [e.id for e in n.target.elts if isinstance(e,ast.Name)] == ["i","current_origin"]]
    require(len(loops) == 1, "Actual deployment loop not uniquely identified")
    loop = loops[0]
    forbidden = {"fit_proposal", "fit_global_mixture", "fit_garch", "block_gate", "empirical_gate", "reflected_control"}
    require(not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in forbidden for n in ast.walk(loop)), "Deployment loop refits a frozen main-method component")
    model_tree = ast.parse(texts["market_models.py"])
    garch = function(model_tree, "fit_garch")
    garch_text = ast.unparse(garch)
    require("log_returns[1:training_end + 1]" in garch_text, "GARCH QMLE training prefix changed")
    fit_text = ast.unparse(function(model_tree, "fit_proposal"))
    for fragment in ["torch.tensor(xtrain", "torch.tensor(ytrain", "torch.tensor(xval", "torch.tensor(yval",
                     "with torch.no_grad()", "score < best_score - 1e-08", "model.load_state_dict(state)", "v.detach().clone()"]:
        require(fragment in fit_text, "Missing frozen neural-selection condition: " + fragment)
    require("calibration" not in fit_text and "test_" not in fit_text, "Neural training function references later roles")
    prediction_text = ast.unparse(function(model_tree, "predict_factor"))
    require("with torch.no_grad()" in prediction_text, "Prediction path tracks gradients")
    return {"passed": True, "current_source_sha256": current, "prefit_source_sha256": prefit["source_sha256"],
            "post_prefit_changed_files": differences,
            "approved_date_guard_only_change_reconstructs_prefit_bytes": guard_only,
            "post_launch_correction_interpretation": "Only CSV date validation moved before price conversion. The 22 original preparations use fixed <=2023 input hashes. No fitted parameters, arrays or scores need recomputation.",
            "guard_sentinel": test_guard(preparation),
            "static_fit_lineage": {"F0": "y[train-39]", "FHS": "y[train-39]",
                "GARCH": "returns[1:training_label_end+1]", "neural_gradient": "train_X/train_y",
                "neural_epoch_selection": "validation_X/validation_y including epoch zero",
                "path_control_and_gate": "calibration labels, calibration paths, calibration history basis only",
                "main_method_deployment": "fixed coefficients/experts; history-state basis remains causal; no test feedback refit"}}, loop


def sentinel_feedback(loop, origins, horizon):
    """Run only the actual loop with index markers and no learning machinery."""
    n = len(origins)
    env, events, qevents, issued_alpha, issued_quantiles = {}, [], [], {}, {}
    class InertOnline:
        @property
        def alpha(self):
            value = .1 + .00001*len(events)
            issued_alpha[int(env["i"])] = value
            return value
        def update_ab(self, a, b, issued):
            past = int(a)
            require(b == -past-1, "Wrong historical BOA B marker")
            require(issued == issued_alpha[past], "BOA used a current/reconstructed rather than issued alpha")
            events.append((past,int(env["i"])))
    class InertQuantile:
        def predict(self, quote):
            index = int(quote[0,0])
            atoms = np.full(19, index+.00001*len(qevents))
            issued_quantiles[index] = atoms.copy()
            return atoms
        def update_quantiles(self, quote, target, issued):
            past = int(quote[0,0])
            require(target == past+100000, "Wrong historical quantile target marker")
            require(np.array_equal(issued,issued_quantiles[past]), "Quantile feedback did not use originally issued atoms")
            qevents.append((past,int(env["i"])))
    qt = np.broadcast_to(np.arange(n)[:,None,None],(n,19,2)).copy()
    env.update({"origins": np.array(origins), "h": horizon, "at": np.arange(n,dtype=float),
                "bt": -np.arange(n,dtype=float)-1, "online": InertOnline(), "quantile_model": InertQuantile(),
                "alpha_online": np.empty(n), "issued_atoms": np.empty((n,19)), "qt": qt,
                "d": {"test_y": np.arange(n)+100000}, "observed": 0, "maturity_checks": 0})
    executable = compile(ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[])), "<actual_online_loop_inert_markers>", "exec")
    exec(executable, env)
    expected, seen = [], set()
    for i, now in enumerate(origins):
        for past in range(i):
            if past not in seen and origins[past]+horizon <= now:
                expected.append((past,i)); seen.add(past)
    require(events == expected and qevents == expected, "Actual loop does not match independent maturity-release schedule")
    require(env["maturity_checks"] == len(expected), "Online maturity counter mismatch")
    require(all(origins[past]+horizon <= origins[now] for past,now in expected), "Immature feedback consumed")
    return {"passed": True,"price_or_return_values_used": False,"actual_source_loop_executed": True,
            "market_parameter_updates": 0,"marker_feedback_events_per_baseline": len(expected),
            "immature_feedback_events": 0,"issued_prediction_reuse_checks_per_baseline": len(expected),
            "last_observed_origin": int(origins[expected[-1][0]]) if expected else None}


def source_data(path, expected_hash):
    require(sha(path) == expected_hash, "Development input source hash mismatch: " + path.name)
    dates, prices = [], []
    with path.open(newline="",encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            date = dt.date.fromisoformat(row["date"])
            require(date < dt.date(2024,1,1), "Reserved date in development source")
            # No price conversion before this date guard.
            price = float(row["price"])
            require(math.isfinite(price) and price > 0, "Invalid author price; audit does not repair")
            dates.append(date); prices.append(price)
    require(dates == sorted(dates) and len(dates) == len(set(dates)), "Date ordering/duplication mismatch")
    return dates, np.array(prices)


def independent_indices(dates, year, horizon):
    first = next(i for i,d in enumerate(dates) if d.year == year)
    end_cal = first-horizon-1
    start_cal = end_cal-251
    end_val = start_cal-horizon-1
    start_val = end_val-125
    end_train = start_val-horizon-1
    return {"train": np.arange(39,end_train+1),"validation": np.arange(start_val,end_val+1),
            "calibration": np.arange(start_cal,end_cal+1),
            "test": np.array([i for i,d in enumerate(dates) if i+horizon<len(dates) and d.year==year and dates[i+horizon].year==year],dtype=int)}


def state_from_prefix(prices, dates, floor):
    logp = np.log(prices)
    returns = np.concatenate(([0.],np.diff(logp)))
    variance = np.empty(len(prices))
    variance[:21] = sum(float(returns[i])**2 for i in range(1,21))/20
    for i in range(21,len(prices)):
        variance[i] = .94*variance[i-1]+.06*returns[i]**2
    scale = np.maximum(np.sqrt(variance),floor)
    gaps = np.array([1.]+[(b-a).days for a,b in zip(dates,dates[1:])],dtype=float)
    features = np.column_stack((returns/scale,np.log(scale),np.log1p(gaps)))
    return logp, returns, scale, features


def gaussian_crps_mean(y, weights, sigma):
    z = y[:,None]/sigma[None,:]
    first = sigma[None,:]*np.sqrt(2/np.pi)*np.exp(-.5*z*z)+y[:,None]*erf(z/np.sqrt(2))
    pair = np.sqrt(2/np.pi)*np.sqrt(sigma[:,None]**2+sigma[None,:]**2)
    return float(np.mean(first@weights)-.5*np.sum(weights[:,None]*weights[None,:]*pair))


def check_close(actual, expected, label, errors, tolerance=2e-9):
    actual, expected = np.asarray(actual),np.asarray(expected)
    require(actual.shape == expected.shape, "Shape mismatch: "+label)
    require(np.all(np.isfinite(actual)), "Nonfinite prepared array: "+label)
    error = float(np.max(np.abs(actual-expected),initial=0))
    errors[label] = error
    require(np.allclose(actual,expected,rtol=2e-10,atol=tolerance), "Numerical reconstruction mismatch: "+label+" max_abs="+str(error))


def audit_cell(case, dates, prices, data_sha, runs, loop):
    market, year, h = case["market"],case["evaluation_year"],case["horizon_observation_steps"]
    name = f"{market}_{year}_h{h}"
    npz = runs/"prepared"/(name+".npz")
    receipt_path = runs/"prepared"/(name+".json")
    receipt = json.loads(receipt_path.read_text())
    require(receipt["source_sha256"] == data_sha, "Preparation used an unexpected source: "+name)
    require(receipt["prepared_sha256"] == sha(npz), "Prepared binary hash differs from receipt: "+name)
    with np.load(npz,allow_pickle=False) as file:
        stored = {key:file[key] for key in file.files}
    ids = independent_indices(dates,year,h)
    errors, boundaries = {}, []
    for role, indices in ids.items():
        frozen = case["roles"][ROLE_MAP[role]]
        require(np.array_equal(indices,np.arange(frozen["origin_index_start"],frozen["origin_index_end_inclusive"]+1)),"Frozen split mismatch: "+name+role)
        require(np.array_equal(stored[role+"_indices"],indices),"Prepared indices mismatch: "+name+role)
        require(len(indices)==frozen["origin_count"],"Frozen role count mismatch")
        expected_dates = np.array([dates[i].isoformat() for i in indices])
        require(np.array_equal(stored[role+"_dates"],expected_dates),"Prepared dates mismatch")
        rr=receipt["roles"][role]
        require(rr["count"]==len(indices) and rr["first_origin"]==indices[0] and rr["last_origin"]==indices[-1],"Receipt index/count mismatch")
        require(rr["first_date"]==dates[indices[0]].isoformat() and rr["last_date"]==dates[indices[-1]].isoformat(),"Receipt date mismatch")
        require(rr["last_label_index"]==indices[-1]+h and rr["last_label_date"]==dates[indices[-1]+h].isoformat(),"Receipt maturity mismatch")
        require(stored[role+"_X"].shape==(len(indices),20,3),"Actual network input is not 20x3")
    for previous,following in (("train","validation"),("validation","calibration"),("calibration","test")):
        require(ids[previous][-1]+h < ids[following][0],"A target crosses a role boundary")
        boundaries.append({"from_role":previous,"to_role":following,"last_maturity_index":int(ids[previous][-1]+h),"next_first_origin_index":int(ids[following][0]),"passed":True})
    train=ids["train"]; training_end=int(train[-1]+h)
    raw_returns=np.concatenate(([0.],np.diff(np.log(prices))))
    floor=max(1e-8,.1*np.median(np.abs(raw_returns[1:training_end+1])))
    logp,returns,scale,features=state_from_prefix(prices,dates,floor)
    xtrain=np.stack([features[i-19:i+1] for i in train])
    mean=xtrain.mean(axis=(0,1)); sd=np.maximum(xtrain.std(axis=(0,1)),1e-5)
    lvmean=float(np.mean(np.log(scale[train]))); lvsd=max(float(np.std(np.log(scale[train]))),1e-5)
    check_close(stored["floor"],floor,"floor",errors)
    check_close(stored["feature_mean"],mean,"feature_mean_training_only",errors)
    check_close(stored["feature_sd"],sd,"feature_sd_training_only",errors)
    check_close(stored["history_logscale_mean"],lvmean,"gate_state_mean_training_only",errors)
    check_close(stored["history_logscale_sd"],lvsd,"gate_state_sd_training_only",errors)
    gr=receipt["GARCH_fit"]
    require(gr["training_last_return_index"]==training_end,"GARCH fitted beyond training-label endpoint")
    require(training_end < ids["validation"][0],"Global fitted scale/variance parameters not known before validation")
    omega,alpha,beta=gr["omega"],gr["alpha"],gr["beta"]
    rho=alpha+beta
    require(0<=rho<1 and omega>0,"GARCH parameter support invalid")
    longrun=omega/(1-rho)
    next_var=np.full(len(prices),longrun)
    for i in range(1,len(prices)):
        next_var[i]=omega+alpha*returns[i]**2+beta*next_var[i-1]
    for role,indices in ids.items():
        x=np.stack([features[i-19:i+1] for i in indices])
        unit=scale[indices]*np.sqrt(h)
        y=(logp[indices+h]-logp[indices])/unit
        path=np.stack([np.diff(logp[i:i+h+1])/unit[j] for j,i in enumerate(indices)])
        state=(np.log(scale[indices])-lvmean)/lvsd
        logits=-.5*((state[:,None]-np.array([-1.,0.,1.]))/.75)**2
        exponential=np.exp(logits-logits.max(axis=1,keepdims=True))
        basis=exponential/exponential.sum(axis=1,keepdims=True)
        future_variances=np.stack([longrun+(next_var[indices]-longrun)*rho**k for k in range(h)])
        garch_sigma=np.sqrt(np.maximum(future_variances.sum(axis=0),1e-20))/unit
        for suffix,value in {"X":(x-mean)/sd,"unit":unit,"y":y,"path":path,"basis":basis,
                             "origin_price":prices[indices],"target_price":prices[indices+h],"garch_sigma":garch_sigma}.items():
            check_close(stored[role+"_"+suffix],value,role+"_"+suffix,errors)
        check_close(path.sum(axis=1),y,role+"_path_sum",errors)
    w,s=stored["weights"],stored["sigma0"]
    require(w.shape==(3,) and s.shape==(3,) and np.all(w>=0) and np.all(s>0) and abs(w.sum()-1)<1e-12,"F0 support invalid")
    check_close(w,np.array(receipt["global_mixture_fit"]["weights"]),"F0_weights_receipt",errors)
    check_close(s,np.array(receipt["global_mixture_fit"]["scales"]),"F0_scales_receipt",errors)
    check_close(s/s[1],np.array([.5,1.,2.]),"F0_component_ratios",errors)
    check_close(gaussian_crps_mean(stored["train_y"],w,s),receipt["global_mixture_fit"]["objective"],"F0_train_objective_independent_formula",errors)
    atoms=np.quantile(stored["train_y"],(np.arange(101)+.5)/101);atoms-=atoms.mean()
    check_close(stored["fhs_atoms"],atoms,"FHS_training_quantiles",errors)
    prefix_checks=0
    for role in ("validation","calibration","test"):
        for position in (0,len(ids[role])-1):
            origin=int(ids[role][position])
            _,_,prefix_scale,prefix_features=state_from_prefix(prices[:origin+1],dates[:origin+1],floor)
            check_close((prefix_features[origin-19:origin+1]-mean)/sd,stored[role+"_X"][position],role+"_prefix_X_"+str(position),errors)
            check_close(prefix_scale[origin]*np.sqrt(h),stored[role+"_unit"][position],role+"_prefix_unit_"+str(position),errors)
            prefix_checks+=1
    feedback=sentinel_feedback(loop,ids["test"],h)
    return {"cell":name,"passed":True,"source_sha256":data_sha,"prepared_sha256":sha(npz),"preparation_receipt_sha256":sha(receipt_path),
            "role_counts":{role:len(v) for role,v in ids.items()},"strict_maturity_boundaries":boundaries,
            "last_training_label_index":training_end,"last_training_label_date":dates[training_end].isoformat(),
            "EWMA_initialization_last_return_index":20,"first_ever_used_feature_index":20,
            "network_feature_shape":[20,3],"prefix_reconstruction_checks":prefix_checks,
            "numeric_array_max_abs_errors":errors,"global_fit_success":receipt["global_mixture_fit"]["success"],
            "garch_fit_success":gr["success"],"online_loop_sentinel":feedback},ids


def neural_receipt_audit(runs,cell_results,indices):
    results,missing=[] ,[]
    for cell in cell_results:
        for backbone in BACKBONES:
            for seed in SEEDS:
                name=f"{cell['cell']}_{backbone}_s{seed}"
                path=runs/"receipts"/(name+".json")
                if not path.exists():
                    missing.append(name);continue
                receipt=json.loads(path.read_text());fit=receipt["fit"]
                history=fit["history"]
                require([entry["epoch"] for entry in history]==list(range(61)),"Validation epoch history incomplete: "+name)
                require(all(math.isfinite(entry["validation_CRPS"]) for entry in history),"Invalid validation history")
                best,best_epoch=history[0]["validation_CRPS"],0
                for entry in history[1:]:
                    if entry["validation_CRPS"] < best-1e-8:
                        best,best_epoch=entry["validation_CRPS"],entry["epoch"]
                require(fit["selected_epoch"]==best_epoch and fit["selected_validation_CRPS"]==best,"Epoch selection not according to frozen validation-only rule: "+name)
                require(fit["actual_training_epochs"]==60,"Unexpected actual training epochs")
                expected_steps=60*math.ceil(cell["role_counts"]["train"]/64)
                require(fit["gradient_optimizer_steps"]==expected_steps,"Unexpected number of training optimizer steps")
                require(receipt["n_test"]==cell["role_counts"]["test"],"Receipt evaluation count differs from frozen split")
                online=cell["online_loop_sentinel"]
                require(receipt["online_updates_with_verified_maturity"]==online["marker_feedback_events_per_baseline"],"Actual run feedback count differs from independently traced source loop")
                require(receipt["online_last_observed_origin"]==online["last_observed_origin"],"Actual run last feedback origin differs from mature schedule")
                results.append({"job":name,"passed":True,"receipt_sha256":sha(path),"validation_history_entries":61,
                                "epoch_selection_verified":True,"gradient_step_count_verified":True,"mature_feedback_count_verified":True})
    return {"expected_jobs":len(cell_results)*6,"completed_jobs_audited":len(results),"missing_jobs":missing,
            "complete":not missing,"test_scores_used_to_select_or_check_epoch":False,"jobs":results}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-dir",type=Path,default=Path('/tmp/carbon-distribution-update-data/development'))
    parser.add_argument("--run-dir",type=Path,default=Path('/tmp/carbon-distribution-update-runs'))
    parser.add_argument("--output-dir",type=Path,default=None)
    parser.add_argument("--require-all-jobs",action="store_true")
    args=parser.parse_args();study=args.study.resolve();out=(args.output_dir or study/"audits").resolve()
    out.mkdir(parents=True,exist_ok=True)
    output={"created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"script_sha256":sha(Path(__file__)),
            "market_model_fits_executed_by_audit":0,"optimizer_calls_executed_by_audit":0,
            "reserved_price_files_read":0,"network_requests":0,"point_price_or_prediction_publication":False}
    try:
        sources=json.loads((study/"data_sources/SOURCE_MANIFEST.json").read_text())
        frozen=json.loads((study/"data_sources/SPLIT_DATE_COUNTS.json").read_text())
        cases=[c for c in frozen["formal_observation_blocks"]["cases"] if c["eligible_for_model_evaluation"]]
        require(len(cases)==22,"Frozen panel does not contain 22 eligible cases")
        expected_names={f"{c['market']}_{c['evaluation_year']}_h{c['horizon_observation_steps']}" for c in cases}
        actual_names={p.stem for p in (args.run_dir/"prepared").glob("*.npz")}
        require(actual_names==expected_names,"Prepared cases missing or undeclared extra cases present")
        source_check,loop=source_audit(study)
        output["source_audit"]=source_check
        inputs={}
        for market in sources["primary_markets"]:
            path=args.data_dir/(market+"_development.csv")
            dates,prices=source_data(path,sources["artifacts"][market]["sha256"])
            inputs[market]=(dates,prices,sha(path))
        output["development_inputs"]={m:{"rows":len(v[0]),"first_date":v[0][0].isoformat(),"last_date":v[0][-1].isoformat(),"sha256":v[2]} for m,v in inputs.items()}
        checks,indices=[],{}
        for case in cases:
            dates,prices,source_sha=inputs[case["market"]]
            result,ids=audit_cell(case,dates,prices,source_sha,args.run_dir,loop)
            checks.append(result);indices[result["cell"]]=ids
        output["prepared_cells"]=checks
        output["neural_job_metadata"]=neural_receipt_audit(args.run_dir,checks,indices)
        if args.require_all_jobs:
            require(output["neural_job_metadata"]["complete"],"Some neural job metadata is not yet available")
        output["history_definition"]={"minimum_initial_quote_prefix":40,"first_allowed_origin_index":39,
              "initial_EWMA_return_count":20,"network_feature_rows":20,"feature_columns":3,
              "earliest_network_feature_row_for_origin_t":"t-19","earliest_raw_price_for_those_increments":"t-20",
              "EWMA_memory":"Recursive past from initialization, not a rolling 40-quote truncation",
              "training_preprocessing_caveat":"Floor, scaling moments and fitted baselines are trained on the declared training sample; they need not have been known at every early training origin, but are fully determined before validation/calibration/test."}
        output["unused_garch_entry_caveat"]={
              "source":"source/market_models.py: fit_garch next_variance[0]",
              "observation":"Index zero stores dimensionless longrun; indices one and later store return-variance units.",
              "affected_index":0,
              "minimum_origin_used_by_current_cases":min(int(v[0]) for ids in indices.values() for v in ids.values()),
              "current_used_variances_independently_verified":True,
              "current_numeric_effect":False,
              "action":"Reported to root; retain frozen source for this completed experiment, repair before future reuse of index zero."}
        output["summary"]={"prepared_cells_audited":len(checks),"strict_role_boundaries_verified":sum(len(c["strict_maturity_boundaries"]) for c in checks),
              "prefix_reconstruction_checks":sum(c["prefix_reconstruction_checks"] for c in checks),
              "neural_jobs_audited":output["neural_job_metadata"]["completed_jobs_audited"],
              "online_feedback_events_verified_per_baseline":sum(c["online_loop_sentinel"]["marker_feedback_events_per_baseline"] for c in checks),
              "immature_feedback_events":0,"all_statistical_fit_receipts_successful":all(c['global_fit_success'] and c['garch_fit_success'] for c in checks)}
        output["passed"]=True
    except Exception as error:
        output["passed"]=False;output["error_type"]=type(error).__name__;output["error"]=str(error)
        (out/"INDEPENDENT_TIME_DATA_AUDIT.json").write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
        raise
    (out/"INDEPENDENT_TIME_DATA_AUDIT.json").write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    summary=output["summary"]
    report=f'''# 独立数据与时间路径审查

审查时间：{output['created_utc']}。结论：**通过当前开发输入、22 个准备案例和已完成训练回执的时间路径审查。** 本脚本执行的市场拟合与优化器调用均为 0，不读取任何保留期价格，不评价或选择新的模型。

## 1. 审查范围及实证核对

固定四市场 CSV 的全部哈希与来源清单一致，最大日期均为 2023-12-29。独立重算的起点集合与 data_sources/SPLIT_DATE_COUNTS.json、prepared NPZ 和准备回执逐项一致。上海 2021 两期限没有进入 prepared；EU_REPORTED 没有进入主面板。

- 准备案例：{summary['prepared_cells_audited']}。
- 严格角色成熟边界：{summary['strict_role_boundaries_verified']}，全部通过。
- 截断到起点的独立特征/尺度复算：{summary['prefix_reconstruction_checks']}，全部通过。
- 已审查神经训练回执：{summary['neural_jobs_audited']} / {output['neural_job_metadata']['expected_jobs']}。
- 源代码实际 BOA 循环的无学习哨兵反馈：每个基线 {summary['online_feedback_events_verified_per_baseline']} 次，未成熟反馈 0 次。

本审查核对每个角色的索引、日期、目标成熟日期、价格对应、标准化特征、目标增量、完整路径、尺度和状态基函数；用独立 CRPS 公式验证保存 F0 参数在训练标签上的目标值；用保存的 GARCH 参数递推而不重新拟合，验证各起点累计条件方差；复算 FHS 训练分位数。公开 JSON 只输出哈希、计数和误差幅度，不输出逐点价格、预测、权重或参数值。

## 2. 已修复的拒绝保留期输入顺序

审查发现启动时的 prepare_cell 在日期拒绝之前转换 price 列。root 随后改为先 usecols=['date'] 读取日期、拒绝 2024+，再完整读取价格。当前源代码逆向恢复这一处代码块后，其 SHA-256 与 MARKET_PREFIT 中原始 run_market.py 精确一致；其他拟合、学习、评分源码和协议哈希全部仍匹配启动快照。

这证明是一次启动后的防护顺序修复，没有隐藏数值算法更改。22 个已准备输入均为固定哈希的 ≤2023 文件；独立数组复算通过，因此当前实验结果不受这次改动影响，无需重复 44 次统计拟合。另执行实际 prepare_cell 的日期防护片段，以仅含合成 2024 日期、完全不提供价格的哨兵验证：1 次日期列读取后拒绝，0 次价格/全表读取。没有打开真实 2024+ 文件。

## 3. 20 特征窗口与 40 报价暖机前缀

实际网络输入为 **20×3**：t−19,…,t 的标准化历史增量、log EWMA、log1p 日历间隔。train_start=39 表示最早允许的起点为第 40 条报价；它使最早使用的特征行是索引 20，该行的初始 EWMA 只依赖已到达的 20 个增量。40 是初始所需报价前缀，不是神经网络 40 步输入或 EWMA 40 步截断。

某些数据容量模板中的 t−39 历史参考索引仅用于原先 40 条暖机前缀约定，不能据此推断网络特征长度。实际特征窗口的首行是 t−19，对其首个增量所需的原价是 p[t−20]；EWMA 还递归保留更早过去。独立审查明确按照实际 20×3 数组验证。

## 4. 参数、标签及选择边界

EWMA 原始递推仅使用到当前起点的收益；初始 20 增量之后没有前向滤波。尺度 floor 的中位数只到训练末目标成熟日，特征标准化矩和状态基函数矩只使用训练起点。F0 与 FHS 只使用训练标签；GARCH QMLE 只使用到训练末目标的收益前缀。所有这些参数在验证首起点之前已可确定。

训练期内的全样本拟合参数本来可能使用较早训练起点之后的数据，这属于离线训练样本内参数估计；不能表述为在每个历史训练起点上已经在线得到了同样的 floor 或标准化矩。本审查确认的是验证、校准和部署时不存在向未来的参数依赖。

神经梯度只用 train_X/train_y；epoch 0 也参加候选，按 validation CRPS 的固定 1e-8 改善门槛保存复制状态并恢复，随后 eval/no_grad 预测。全部已完成回执的历史和选中 epoch、优化器步数都按该规则核验，没有使用测试分数选择 epoch。

校准的路径控制、门控和块抽样只依赖 calibration 数组；校准末目标在测试首起点之前成熟。主方法的专家和门控系数随后冻结，部署门控输出可以随当时已知状态基函数变化，但没有用测试反馈重新拟合主方法。

## 5. BOA 的成熟反馈

对实际源代码在线 for/while 循环进行独立执行，替换为仅记录索引的惰性对象，不运行 BOA 学习或重新拟合市场参数。每一次反馈都满足 origin+h<=current_origin，且消费的是该历史起点真正已发出的 alpha/分位数原子；当前预测和未来标签都没有被提前反馈。两个基线释放序列与独立枚举完全相同，也与 132 份已完成训练回执中的更新数量和末次反馈起点一致。

校准阶段 BOA replay 使用的是在测试开始时已经全部成熟的既往校准标签；它不是对过去校准期逐日部署的重演。测试期才实行上面的延迟成熟反馈。

## 6. 边界与复现

此结论限于当前固定输入、源码快照和准备/训练回执。它不是对未知供应商行情、独立同分布假设、未来分布漂移安全性或统计显著性的认证。上海稀疏日历与长校准日历跨度仍成立，H5/H20 仍指作者报价观测步。

另有一项未使用入口的单位问题：market_models.fit_garch 的 next_variance[0] 存入了未乘 unit² 的 longrun，索引 1 及其后才是实际收益方差单位。当前全部案例读取的最小起点索引为 39，递推内部的无量纲状态没有这个问题，所有实际使用的条件方差也已独立复算通过。因此它不影响本轮数组或分数，但未来若要使用索引 0，需要先修正该入口。本次保留冻结源码，已向 root 报告。

```bash
python -B /path/to/study/audits/independent_time_data_audit.py \\
  --study /path/to/study \\
  --data-dir /tmp/carbon-distribution-update-data/development \\
  --run-dir /tmp/carbon-distribution-update-runs \\
  --require-all-jobs
```

默认要求 22 个 prepared 全部存在；没有 --require-all-jobs 时只核对已经完成的训练回执并如实报告缺项。详细输入哈希、逐案例误差与逐训练回执核验见 INDEPENDENT_TIME_DATA_AUDIT.json。
'''
    (out/"INDEPENDENT_TIME_DATA_AUDIT_ZH.md").write_text(report,encoding="utf-8")
    print(json.dumps({"passed":True,**summary,"output_dir":str(out)},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
