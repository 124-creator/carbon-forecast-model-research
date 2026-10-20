"""Independent development-run audit. No runner import, fitting, or checkpoint loading.

Reconstructs targets, metric units, scale/alignment, and score aggregation from
source CSV and NPZ. Selection receipts are audited without replaying training.
Calendar paired bootstrap averages seed losses before time resampling.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
FAMILIES = ["allocation_full", "allocation_constant", "allocation_readout",
            "allocation_price", "gru", "lstm",
            "rw", "drift", "ridge_price", "ridge_full", "hgb_full"]
NEURAL = FAMILIES[:6]
POOL = ["rw", "drift", "ridge_price", "ridge_full", "hgb_full", "gru", "lstm"]
COMPARATORS = ["rw", "allocation_constant", "allocation_readout", "validation_selected_strong"]
HORIZONS = np.array([1, 5, 20])
SEEDS = [17, 29, 43]
PARAMETERS = {"allocation_full": 169, "allocation_constant": 7,
              "allocation_readout": 169, "allocation_price": 118, "gru": 4308, "lstm": 5524}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def check(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def close(a, b, message, rtol=1e-10, atol=1e-11):
    check(np.allclose(a, b, rtol=rtol, atol=atol), message)


def reconstruct(manifest):
    """Only arithmetic reconstruction of existing input files; no fitted models."""
    arrays = []
    dates = None
    hashes = []
    for i, src in enumerate(manifest["sources"]):
        path = REPO / src["path"]
        got = sha(path)
        check(got == src["sha256"], f"source hash: {path}")
        hashes.append({"path": src["path"], "sha256": got})
        frame = pd.read_csv(path).sort_values("date").reset_index(drop=True)
        dd = pd.to_datetime(frame.date).to_numpy(dtype="datetime64[D]")
        if i == 0:
            check(frame.date.is_unique, "nonunique target dates")
            dates = dd
            p = frame.price.to_numpy(float)
            check(np.isfinite(p).all(), "nonfinite source target")
            arrays.append(np.r_[0.0, np.diff(p)])
        else:
            finite = frame.spot_price.notna().to_numpy()
            missing = int((~finite).sum())
            ep = frame.spot_price.to_numpy(float)[finite]
            ed = dd[finite]
            indices = np.searchsorted(ed, dates - np.timedelta64(14, "D"), side="right") - 1
            check((indices >= 0).all(), "missing past energy")
            ages = (dates - ed[indices]).astype(int)
            check((ages >= 14).all(), "energy lookahead")
            arrays.append(np.r_[0.0, np.diff(ep[indices])])
            saved = manifest["energy_alignment"][i-1]
            check(int(ages.min()) == saved["minimum_age_days"], "minimum energy age")
            check(int(ages.max()) == saved["maximum_age_days"], "maximum energy age")
            check(missing == saved["source_missing_price_rows_excluded_before_backward_asof"], "energy missing count")
    tokens = np.column_stack(arrays)
    years = pd.DatetimeIndex(dates).year.to_numpy()
    year = int(manifest["year"])
    first = int(np.flatnonzero(years == year)[0])
    val_first, val_end = first-100, first-20
    origins = np.arange(64, len(p)-20)
    masks = {
        "train": origins+20 < val_first,
        "val": (origins >= val_first) & (origins < val_end),
        "final": origins+20 < first,
        "test": (years[origins] == year) & (years[origins+20] == year)
    }
    target = np.column_stack([p[origins+h]-p[origins] for h in HORIZONS])
    rms = np.maximum(np.sqrt((tokens[1:val_first]**2).mean(axis=0)), 1e-7)
    variance = np.maximum((target[masks["train"]]**2).mean(axis=0), rms[0]**2*1e-6)
    close(rms, manifest["training_rms"], "training RMS")
    close(variance, manifest["train_target_second_moment"], "training target weights")
    for name, mask in masks.items():
        check(int(mask.sum()) == manifest["counts"][name], f"count {name}")
    expected_dates = {
        "last_train_label": dates[origins[masks["train"]][-1]+20],
        "first_validation_origin": dates[origins[masks["val"]][0]],
        "last_validation_label": dates[origins[masks["val"]][-1]+20],
        "first_test_origin": dates[origins[masks["test"]][0]],
        "last_test_label": dates[origins[masks["test"]][-1]+20]
    }
    for key, value in expected_dates.items():
        check(str(value) == manifest[key], f"date {key}")
    check(origins[masks["train"]][-1]+20 < origins[masks["val"]][0], "train-val purge")
    check(origins[masks["val"]][-1]+20 < origins[masks["test"]][0], "val-test purge")
    validation_rw = float(np.mean(target[masks["val"]]**2/variance))
    drift_trials = []
    for window in [20, 60, 120, None]:
        pp = np.array([tokens[max(1,t-window+1) if window else 1:t+1,0].mean()*HORIZONS
                       for t in origins[masks["val"]]])
        drift_trials.append((float(np.mean((pp-target[masks["val"]])**2/variance)), str(window)))
    return {
        "observed": target[masks["test"]],
        "origins": origins[masks["test"]],
        "dates": dates[origins[masks["test"]]],
        "validation_rw": validation_rw,
        "drift_choice": min(drift_trials),
        "variance": variance,
        "sources": hashes,
        "counts": manifest["counts"]
    }


def score(predictions, observed):
    error = predictions-observed[None]
    mse_seed_h = (error**2).mean(axis=1)
    rw_h = (observed**2).mean(axis=0)
    check(np.isfinite(error).all() and (rw_h > 0).all(), "finite score and positive RW")
    ratio_seed_h = mse_seed_h/rw_h[None]
    ensemble = predictions.mean(axis=0)
    ensemble_mse_h = ((ensemble-observed)**2).mean(axis=0)
    close(mse_seed_h.mean(axis=0)-ensemble_mse_h,
          ((predictions-ensemble[None])**2).mean(axis=(0,1)),
          "bias-variance seed identity")
    return {
        "mean_seed_mse_over_rw": float(ratio_seed_h.mean()),
        "seed_mse_over_rw_by_horizon": ratio_seed_h.tolist(),
        "rw_mse_by_horizon": rw_h.tolist(),
        "ensemble_mse_over_rw": float((ensemble_mse_h/rw_h).mean()),
        "ensemble_mse_over_rw_by_horizon": (ensemble_mse_h/rw_h).tolist(),
        "native_mean_seed_mse_by_horizon": mse_seed_h.mean(axis=0).tolist(),
        "native_mean_seed_rmse_by_horizon": np.sqrt(mse_seed_h).mean(axis=0).tolist(),
        "native_rmse_of_mean_seed_mse_by_horizon": np.sqrt(mse_seed_h.mean(axis=0)).tolist(),
        "native_mean_seed_mae_by_horizon": np.abs(error).mean(axis=(0,1)).tolist(),
        "native_ensemble_mse_by_horizon": ensemble_mse_h.tolist(),
        "native_ensemble_mae_by_horizon": np.abs(ensemble-observed).mean(axis=0).tolist()
    }, (error**2).mean(axis=0)


def audit_receipts(fd, family, source, ledger):
    """Inspect executed receipts and saved state bytes, never unpickle checkpoints."""
    records = []
    for lr in [0.003, 0.01]:
        rp = fd / f"selection_lr{lr}.json"
        r = read_json(rp)
        check(r["seed"] == 17 and r["mode"] == "train", "selection seed/mode")
        check(r["learning_rate"] == lr and r["parameters"] == PARAMETERS[family], "selection LR/params")
        check(len(r["history"]) == r["epochs_executed"], "selection epoch count")
        check(15 <= r["epochs_executed"] <= 80, "selection epoch bounds")
        best, epoch = r["initial_validation_loss"], 0
        for step, line in enumerate(r["history"], start=1):
            check(line["epoch"] == step and np.isfinite(line["train_loss"]), "epoch trace")
            val = line["validation_loss"]
            check(np.isfinite(val), "finite validation")
            if val < best-1e-7:
                best, epoch = val, step
        close(best, r["selected_validation_loss"], "selected validation replay")
        check(epoch == r["selected_epoch"], "selected epoch replay")
        cp = rp.with_suffix(".pt")
        check(sha(cp) == r["checkpoint_sha256"], "selection checkpoint hash")
        ledger.append({**{key:r[key] for key in
            ["family","seed","mode","learning_rate","epochs_executed","selected_epoch",
             "gradient_positive_steps","optimization_parameter_change_l2_before_selection"]},
             "path": str(rp), "receipt_sha256": sha(rp), "checkpoint_sha256": r["checkpoint_sha256"]})
        records.append(r)
    best = min(records, key=lambda x:(x["selected_validation_loss"],x["learning_rate"]))
    choice = read_json(fd/"CHOICE_BEFORE_TEST.json")
    for key in ["selected_validation_loss","selected_epoch","learning_rate","seed","checkpoint_sha256"]:
        check(choice[key] == best[key], f"choice mismatch {key}")
    for seed in SEEDS:
        rp = fd/f"refit_seed{seed}.json"
        r = read_json(rp)
        check(r["mode"] == "final" and r["seed"] == seed, "refit seed/mode")
        check(r["parameters"] == PARAMETERS[family], "refit parameters")
        check(r["learning_rate"] == choice["learning_rate"], "refit LR")
        check(r["epochs_executed"] == choice["selected_epoch"] == r["selected_epoch"], "refit epoch")
        check(len(r["history"]) == r["epochs_executed"], "refit history length")
        check(r["initial_validation_loss"] is None and r["selected_validation_loss"] is None,
              "refit must not select validation epoch")
        if r["epochs_executed"] == 0:
            check(r["gradient_positive_steps"] == 0, "epoch0 gradient")
            check(r["optimization_parameter_change_l2_before_selection"] == 0, "epoch0 parameters")
        else:
            check(r["gradient_positive_steps"] > 0, "refit no gradients")
        check(sha(rp.with_suffix(".pt")) == r["checkpoint_sha256"], "refit checkpoint hash")
        ledger.append({**{key:r[key] for key in
            ["family","seed","mode","learning_rate","epochs_executed","selected_epoch",
             "gradient_positive_steps","optimization_parameter_change_l2_before_selection"]},
             "path": str(rp), "receipt_sha256": sha(rp), "checkpoint_sha256": r["checkpoint_sha256"]})
    return float(choice["selected_validation_loss"]), int(choice["selected_epoch"])


def calendar_weights(rng, replications, n_days, length):
    starts = rng.integers(0, n_days, size=(replications, math.ceil(n_days/length)))
    sampled = ((starts[:,:,None] + np.arange(length)[None,None,:]) % n_days).reshape(replications,-1)[:, :n_days]
    flat = sampled + np.arange(replications)[:,None]*n_days
    weights = np.bincount(flat.ravel(), minlength=replications*n_days).reshape(replications,n_days)
    check(np.all(weights.sum(axis=1) == n_days), "bootstrap calendar length")
    return weights.astype(float)


def paired_calendar_bootstrap(cells, length, replications, seed):
    rng = np.random.default_rng(seed)
    draws = np.zeros((replications, len(COMPARATORS)), dtype=float)
    for year in sorted({c["year"] for c in cells}):
        start = np.datetime64(f"{year}-01-01")
        n_days = int((np.datetime64(f"{year+1}-01-01")-start).astype(int))
        weights = calendar_weights(rng, replications, n_days, length)
        for c in cells:
            if c["year"] != year:
                continue
            offsets = (c["dates"]-start).astype(int)
            check((offsets >= 0).all() and (offsets < n_days).all(), "calendar origin year")
            w = weights[:,offsets]
            den = w @ (c["observed"]**2)
            check((den > 0).all(), "bootstrap has zero RW denominator")
            for j, comparator in enumerate(COMPARATORS):
                difference = c["loss"][comparator]-c["loss"]["allocation_full"]
                draws[:,j] += ((w @ difference)/den).mean(axis=1)/len(cells)
    return {name:{
        "improvement_ci95": np.quantile(draws[:,j],[0.025,0.975]).tolist(),
        "improvement_ci98_75_bonferroni4": np.quantile(draws[:,j],[0.00625,0.99375]).tolist(),
        "bootstrap_positive_fraction_not_p_value": float((draws[:,j] > 0).mean()),
        "replications": replications,
        "block_calendar_days": length
    } for j,name in enumerate(COMPARATORS)}


def self_test():
    # Equal and opposite seed errors: seed mean MSE stays positive, ensemble MSE is zero.
    truth = np.column_stack([np.arange(1,9),np.arange(2,10),np.arange(3,11)]).astype(float)
    preds = np.stack([truth+2,truth-2])
    metrics, losses = score(preds,truth)
    close(losses, np.full_like(truth,4), "seed loss must precede averaging")
    check(metrics["ensemble_mse_over_rw"] == 0, "ensemble distinct")
    check(metrics["mean_seed_mse_over_rw"] > 0, "meanseed positive")
    # Constant contrast = .25 RW squared loss gives exactly .25 in every paired replicate.
    dates = np.arange(np.datetime64("2022-01-01"),np.datetime64("2023-01-01"))
    y = np.column_stack([np.ones(len(dates)),2*np.ones(len(dates)),3*np.ones(len(dates))])
    base = y*y
    cells=[]
    for market in ["A","B"]:
        loss={name:base.copy() for name in COMPARATORS}
        loss["allocation_full"]=0.75*base
        cells.append({"year":2022,"market":market,"dates":dates,"observed":y,"loss":loss})
    result = paired_calendar_bootstrap(cells,42,100,123)
    for contrast in result.values():
        close(contrast["improvement_ci95"],[.25,.25],"paired bootstrap known effect")
    return {"status":"PASS","checks":["seed-loss-before-ensemble distinction",
            "paired calendar resampling known constant effect","calendar multiplicity length"]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,default=Path("/tmp/carbon-literature-aligned-runs-20261019d/main_v1"))
    parser.add_argument("--out",type=Path,default=STUDY/"audits")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    toy=self_test()
    if args.self_test:
        print(json.dumps(toy,ensure_ascii=False)); return
    run=args.run
    protocol=read_json(run/"PROTOCOL_BEFORE_RUN.json")
    complete=read_json(run/"EXECUTION_COMPLETE.json")
    check(complete["status"]=="completed" and complete["source_hashes_unchanged"],"run complete/hash flag")
    check(complete["jobs"]==56 and complete["summary_rows"]==88,"completed job counts")
    check(protocol["stage"]=="development_search","stage")
    check("mean_seed_mse_over_rw" in protocol["primary_seed_summary"],"primary metric declaration")
    for name,digest in protocol["source_sha256"].items():
        check(sha(STUDY/"source"/name)==digest,"frozen source hash")
    ledger,cells,metrics_rows,identity,selected=[],[],[],[],[]
    selected_epoch_rows=[]
    all_summary=read_json(run/"ALL_SUMMARIES.json")
    all_summary_index={(r["market"],r["year"],r["family"]):r for r in all_summary}
    check(len(all_summary_index)==88,"unique summary rows")
    for market in protocol["markets"]:
        for year in protocol["years"]:
            cd=run/f"{market}_{year}"
            manifest=read_json(cd/"INPUT_MANIFEST.json")
            source=reconstruct(manifest)
            validation_scores={}
            for family in FAMILIES:
                fd=cd/family
                if family in NEURAL:
                    vs, epoch = audit_receipts(fd,family,source,ledger)
                    validation_scores[family]=vs
                    selected_epoch_rows.append({"market":market,"year":year,"family":family,"selected_epoch":epoch})
                else:
                    r=read_json(fd/"SELECTION.json")
                    validation_scores[family]=float(r["validation_loss"])
                    if family=="rw":
                        close(r["validation_loss"],source["validation_rw"],"RW validation reconstruction")
                    elif family=="drift":
                        close(r["validation_loss"],source["drift_choice"][0],"drift validation reconstruction")
                        check(r["window"]==source["drift_choice"][1],"drift validation choice")
                    elif family.startswith("ridge"):
                        best=min(r["trials"])
                        close(r["validation_loss"],best[0],"ridge validation minimum")
                        check(r["alpha"]==best[1],"ridge alpha")
                    elif family=="hgb_full":
                        best=min(r["trials"])
                        close(r["validation_loss"],best[0],"HGB validation minimum")
                        check(r["max_leaf_nodes"]==best[1] and r["l2_regularization"]==best[2],"HGB choice")
            strong=min(POOL,key=lambda family:(validation_scores[family],POOL.index(family)))
            selected.append({"market":market,"year":year,"family":strong,
                             "validation_loss":validation_scores[strong],
                             "pool_validation_losses":{f:validation_scores[f] for f in POOL}})
            # The strong family above is fixed before any NPZ in this cell is opened.
            cell={"market":market,"year":year,**source,"loss":{},"metrics":{},"strong":strong}
            for family in FAMILIES:
                fd=cd/family; pp=fd/"predictions.npz"
                with np.load(pp,allow_pickle=False) as z:
                    required={"predictions","observed","origins","horizons","seeds"}
                    check(set(z.files)==required,"NPZ fields")
                    pred=np.array(z["predictions"],dtype=float)
                    obs=np.array(z["observed"],dtype=float)
                    check(np.array_equal(z["origins"],source["origins"]),"NPZ/source origins")
                    check(np.array_equal(z["horizons"],HORIZONS),"NPZ horizons")
                    check(np.array_equal(z["seeds"],SEEDS if family in NEURAL else [-1]),"NPZ seed labels")
                    close(obs,source["observed"],"NPZ/source target",rtol=0,atol=0)
                check(pred.shape==(3 if family in NEURAL else 1,len(obs),3),"prediction shape")
                metrics,loss=score(pred,obs)
                if family=="rw":
                    check(np.all(pred==0),"RW prediction zero")
                if family in ["gru","lstm"]:
                    epoch=next(x["selected_epoch"] for x in selected_epoch_rows
                               if (x["market"],x["year"],x["family"])==(market,year,family))
                    if epoch==0:
                        check(np.all(pred==0),"recurrent epoch0 exact RW")
                for stored in [read_json(fd/"SUMMARY.json"),all_summary_index[(market,year,family)]]:
                    check(stored["n_origins"]==len(obs),"stored origin count")
                    for key in ["mean_seed_mse_over_rw","seed_mse_over_rw_by_horizon",
                                "rw_mse_by_horizon","ensemble_mse_over_rw",
                                "ensemble_mse_over_rw_by_horizon"]:
                        close(metrics[key],stored[key],f"stored summary mismatch {family}/{key}")
                cell["metrics"][family]=metrics; cell["loss"][family]=loss
                identity.append({"market":market,"year":year,"family":family,
                                 "npz_sha256":sha(pp),"n_origins":len(obs),
                                 "first_origin_date":str(source["dates"][0]),
                                 "last_origin_date":str(source["dates"][-1])})
                for j,h in enumerate(HORIZONS):
                    metrics_rows.append({"market":market,"year":year,"family":family,"horizon":int(h),
                        "n_origins":len(obs),"mean_seed_mse_over_rw":float(np.mean(
                            np.array(metrics["seed_mse_over_rw_by_horizon"])[:,j])),
                        "mean_seed_mse":metrics["native_mean_seed_mse_by_horizon"][j],
                        "mean_seed_rmse":metrics["native_mean_seed_rmse_by_horizon"][j],
                        "rmse_of_mean_seed_mse":metrics["native_rmse_of_mean_seed_mse_by_horizon"][j],
                        "mean_seed_mae":metrics["native_mean_seed_mae_by_horizon"][j],
                        "ensemble_mse":metrics["native_ensemble_mse_by_horizon"][j],
                        "ensemble_mae":metrics["native_ensemble_mae_by_horizon"][j]})
            cell["metrics"]["validation_selected_strong"]=cell["metrics"][strong]
            cell["loss"]["validation_selected_strong"]=cell["loss"][strong]
            cells.append(cell)
    check(len(cells)==8 and len(identity)==88 and len(ledger)==240,"audit counts")
    families=FAMILIES+["validation_selected_strong"]
    macro={f:{
        "primary_mean_seed_mse_over_rw":float(np.mean([c["metrics"][f]["mean_seed_mse_over_rw"] for c in cells])),
        "secondary_ensemble_mse_over_rw":float(np.mean([c["metrics"][f]["ensemble_mse_over_rw"] for c in cells]))
    } for f in families}
    seed_macro={f:np.mean([np.array(c["metrics"][f]["seed_mse_over_rw_by_horizon"]).mean(axis=1)
                           for c in cells],axis=0).tolist() for f in NEURAL}
    effects={}
    full=macro["allocation_full"]["primary_mean_seed_mse_over_rw"]
    for comp in COMPARATORS:
        control=macro[comp]["primary_mean_seed_mse_over_rw"]
        per_cell=[c["metrics"][comp]["mean_seed_mse_over_rw"]-c["metrics"]["allocation_full"]["mean_seed_mse_over_rw"] for c in cells]
        per_h=[(c["loss"][comp]-c["loss"]["allocation_full"]).mean(axis=0)/(c["observed"]**2).mean(axis=0) for c in cells]
        effects[comp]={"positive_improvement_rw_units":control-full,
                       "relative_reduction_in_macro_ratio":1-full/control,
                       "cell_wins_of_8":int(np.sum(np.array(per_cell)>0)),
                       "horizon_cell_wins_of_24":int(np.sum(np.array(per_h)>0)),
                       "cell_effects":[{"market":c["market"],"year":c["year"],"effect":v}
                                       for c,v in zip(cells,per_cell)],
                       "leave_one_market_out":{m:float(np.mean([v for c,v in zip(cells,per_cell) if c["market"]!=m]))
                                                for m in protocol["markets"]}}
    boot42=paired_calendar_bootstrap(cells,42,20000,20261019)
    boot84=paired_calendar_bootstrap(cells,84,20000,20261019)
    for comp in COMPARATORS:
        effects[comp]["bootstrap_primary_42_calendar_days"]=boot42[comp]
        effects[comp]["bootstrap_sensitivity_84_calendar_days"]=boot84[comp]
    selections=[r for r in ledger if r["mode"]=="train"]
    refits=[r for r in ledger if r["mode"]=="final"]
    counts={"logged_neural_tasks":len(ledger),"selection_tasks":len(selections),"refit_tasks":len(refits),
            "tasks_with_positive_gradient_steps":sum(r["gradient_positive_steps"]>0 for r in ledger),
            "selection_with_positive_gradient_steps":sum(r["gradient_positive_steps"]>0 for r in selections),
            "refits_with_positive_gradient_steps":sum(r["gradient_positive_steps"]>0 for r in refits),
            "zero_epoch_refits":sum(r["epochs_executed"]==0 for r in refits),
            "total_executed_epochs":sum(r["epochs_executed"] for r in ledger),
            "total_logged_positive_gradient_steps":sum(r["gradient_positive_steps"] for r in ledger),
            "selected_family_cells":len(selected_epoch_rows),
            "selected_epoch_zero_family_cells":sum(r["selected_epoch"]==0 for r in selected_epoch_rows),
            "classical_predictions":40,"neural_prediction_files":48,"total_prediction_files":88,
            "unique_origins_across_8_cells":sum(len(c["observed"]) for c in cells),
            "origin_horizon_targets":sum(c["observed"].size for c in cells)}
    result={
        "title":"V1 独立统计审查","status":"PASS_INTEGRITY","stage":"development_only",
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "run_root":str(run),"source_sha256":protocol["source_sha256"],
        "audit_script_sha256":sha(__file__),"audit_plan_sha256":sha(STUDY/"audits/V1_STATISTICAL_AUDIT_PLAN.json"),
        "protocol_sha256":sha(run/"PROTOCOL_BEFORE_RUN.json"),
        "self_test":toy,"execution_counts":counts,"macro_scores":macro,
        "seed_primary_scores_order_17_29_43":seed_macro,"core_effects":effects,
        "validation_selected_strong":selected,
        "selected_epochs":selected_epoch_rows,"npz_identity":identity,"training_receipt_ledger":ledger,
        "cell_metrics":[{"market":c["market"],"year":c["year"],"counts":c["counts"],
                        "first_test_origin":str(c["dates"][0]),"last_test_origin":str(c["dates"][-1]),
                        "strong_baseline":c["strong"],"metrics":c["metrics"]} for c in cells],
        "limitations":[
            "All 2022/2023 outcomes are already exposed development evidence; uncertainty intervals do not account for prior research searches.",
            "No market model was fitted or optimization replayed by this audit. Stored checkpoint bytes and receipt consistency were checked.",
            "RW and drift validation scores were independently reconstructed; ridge/HGB/neural validation scores rely on frozen runner plus internally consistent selection receipts.",
            "Bootstrap fixes trained models and selected family identities. Seeds are averaged as losses within origins and are not independent samples.",
            "Synchronous 42/84 calendar-day circular blocks retain shared date fluctuations across markets. Approximation and boundary wrap assumptions remain.",
            "95% intervals are marginal. 98.75% intervals are approximate Bonferroni simultaneous 95% intervals for these four contrasts only, not an earlier-search correction.",
            "The positive bootstrap fraction is not a p-value or posterior probability.",
            "Public historical EIA data lagged 14 calendar days are not certified historical vintages.",
            "Native loss units differ by market and are reported by cell/horizon, not pooled as one economically commensurate MSE."
        ]}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"V1_INDEPENDENT_STATISTICAL_AUDIT_ZH.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    with (args.out/"V1_NATIVE_METRICS.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics_rows[0]));writer.writeheader();writer.writerows(metrics_rows)
    lines=["# V1 独立统计审查","",
       "**完整性结论：PASS。科学结论仍限定为 development。** 本审查没有重新训练模型；从源 CSV 重建目标、日期、分割、RMS、训练二阶矩和能源日期对齐，再从 88 个 NPZ 的逐 origin、逐 horizon、逐 seed 预测独立复算。原始 SUMMARY 仅用于最后一致性比较。",
       "",
       "## 主指标","",
       "各市场/年度/horizon 先对三个 seed 的平方误差分别评分，再平均；除以同一单元的 RW MSE，最后对 4 市场 × 2 年 × 3 horizon 等权。它是单次训练误差的 seed 平均，不是三个预测先平均的集成。",
       "",
       "| 家族 | 主指标：mean-seed MSE/RW | 次要：ensemble MSE/RW |",
       "|---|---:|---:|"]
    for f in sorted(families,key=lambda name:macro[name]["primary_mean_seed_mse_over_rw"]):
        lines.append(f"| {f} | {macro[f]['primary_mean_seed_mse_over_rw']:.9f} | {macro[f]['secondary_ensemble_mse_over_rw']:.9f} |")
    lines += ["","## 四个核心对比","",
       "正值表示 full 改善。下表“改善”以 RW 归一化 MSE 的百分点计（100 × comparator−full）；相对降低另除以 comparator 的宏观指标。不要把两种百分数混写。",
       "",
       "| 对照 | 改善（RW 百分点） | 相对降低 | 95% 区间（42 日） | 98.75% 区间（42 日） | 95% 区间（84 日） | 获胜单元 / 8 |",
       "|---|---:|---:|---|---|---|---:|"]
    def ci(value):
        return f"[{100*value[0]:+.4f}, {100*value[1]:+.4f}]"
    for comp,e in effects.items():
        b=e["bootstrap_primary_42_calendar_days"];s=e["bootstrap_sensitivity_84_calendar_days"]
        lines.append(f"| {comp} | {100*e['positive_improvement_rw_units']:+.5f} | {100*e['relative_reduction_in_macro_ratio']:+.5f}% | {ci(b['improvement_ci95'])} | {ci(b['improvement_ci98_75_bonferroni4'])} | {ci(s['improvement_ci95'])} | {e['cell_wins_of_8']} |")
    lines += ["",
       "Bootstrap 在每年完整日历上抽取循环连续块，并用同一日历日期重复次数同步加权所有市场。每次都重算各单元的误差与 RW 分母，保持配对和单元等权。种子先在 origin 内折为平均损失；没有把 seeds 当作独立样本。预先写入本次审查方案的块长是 42 日、84 日，均为 20,000 次，随机种子 20261019。该审查方案形成于实验完成后，不冒充研究的事前注册。",
       "",
       "区间条件于已经训练的模型及已选家族，不包含重训误差、此前研究筛选、或全新的市场抽样。98.75% 是四对比 Bonferroni 的近似同时区间，仅处理本表四项。所有 2022/2023 数据已经暴露，不能把区间转述为 fresh confirmation。",
       "",
       "## 强基线的验证选择","",
       "每单元先按过去验证损失从 RW、drift、ridge_price、ridge_full、hgb_full、GRU、LSTM 选择，再读取该单元测试 NPZ。严格相等的数值按此列举顺序处理。RW、drift 的验证损失由源数据独立重建；其余家族核对冻结源码和选择日志，未重放拟合。",
       "",
       "| 市场 | 年 | 选中家族 | 验证损失 | 测试主指标 | full 主指标 |",
       "|---|---:|---|---:|---:|---:|"]
    for c,s in zip(cells,selected):
        lines.append(f"| {c['market']} | {c['year']} | {s['family']} | {s['validation_loss']:.8f} | {c['metrics'][s['family']]['mean_seed_mse_over_rw']:.8f} | {c['metrics']['allocation_full']['mean_seed_mse_over_rw']:.8f} |")
    lines += ["","## 训练执行与完整性","",
       f"共核对 {counts['logged_neural_tasks']} 个神经网络任务日志：{counts['selection_tasks']} 次选择训练、{counts['refit_tasks']} 次最终重训。其中记录正梯度步骤的任务为 {counts['tasks_with_positive_gradient_steps']}，最终重训中为 {counts['refits_with_positive_gradient_steps']}；{counts['zero_epoch_refits']} 次最终重训选择 epoch 0，没有执行梯度更新。48 个家族/单元选择中，{counts['selected_epoch_zero_family_cells']} 个选择 epoch 0。",
       "",
       f"共 {counts['unique_origins_across_8_cells']} 个市场/年度预测起点、{counts['origin_horizon_targets']} 个起点/horizon 目标，88 个预测文件。文件间 origins、targets、horizons 均一致，源码与输入哈希一致，目标按源价格精确复原，现有 SUMMARY 的主指标与集成指标均通过独立复算。每个 checkpoint 的字节哈希与日志一致；本审查未加载或执行 checkpoint。",
       "",
       "## 局部稳定性","",
       "| 对照 | 去掉 GZ | 去掉 HB | 去掉 SH | 去掉 EU |",
       "|---|---:|---:|---:|---:|"]
    for comp,e in effects.items():
        values=[100*e["leave_one_market_out"][m] for m in protocol["markets"]]
        lines.append("| "+comp+" | "+" | ".join(f"{v:+.5f}" for v in values)+" |")
    lines += ["",
       "上表为同样 RW 百分点的留一市场效应；它用于观察某一市场是否主导均值，不是额外独立试验。按市场/horizon 的原单位 MSE、RMSE、MAE 已保存到 V1_NATIVE_METRICS.csv；不将不同市场的原单位损失直接混为一个平均数。",
       "",
       "## 审查证据与限制","",
       "- 机器记录：V1_INDEPENDENT_STATISTICAL_AUDIT_ZH.json，含 88 个 NPZ 哈希、240 个训练日志核对、逐单元主指标、种子分数、核心区间与强基线选择。",
       "- 独立脚本：independent_v1_statistics.py；设计记录：V1_STATISTICAL_AUDIT_PLAN.json。",
       "- 纯合成检查验证先平均 seed 损失与先平均预测的区别，以及已知恒定配对效应在日历块重采样后的精确保持。",
       "- 实际训练更新以执行日志及其 checkpoint 字节一致性为依据，未重放优化。该完整性 PASS 不等于模型有效性或创新性 PASS。",
       "- EIA 14 日滞后使用公共历史版本；不是 vintage 认证。",
       ""]
    (args.out/"V1_INDEPENDENT_STATISTICAL_AUDIT_ZH.md").write_text("\n".join(lines))
    print(json.dumps({"status":result["status"],"counts":counts,"macro":macro,
                      "core_effects":{name:{key:value for key,value in effect.items()
                                           if key not in ["cell_effects","leave_one_market_out"]}
                                      for name,effect in effects.items()}},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()

