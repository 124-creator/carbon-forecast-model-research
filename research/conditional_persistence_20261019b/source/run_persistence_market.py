"""Execute the frozen persistence market study after independent source review.

No preparation, download, or source-price parsing occurs here. Only explicitly
prepared private NPZ files and their producer receipts are accepted. All model
weights, individual forecasts and scores are private. Public output is limited
to hash receipts, learning curves, model parameter counts and cell aggregates.

The --smoke mode never reads prepared data and is accounted separately from the
312 planned market fits. The implementer is not this runner's independent auditor.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

try:
    from .persistence_market import (ARMS, SEEDS, EPOCHS, HORIZONS, PersistenceMarketModel,
        fit_references, fit_neural, predict, predict_parameters, statistical_prediction, finite, reference_mode)
    from .persistence import PI, sample_paths
except ImportError:
    from persistence_market import (ARMS, SEEDS, EPOCHS, HORIZONS, PersistenceMarketModel,
        fit_references, fit_neural, predict, predict_parameters, statistical_prediction, finite, reference_mode)
    from persistence import PI, sample_paths


STUDY = Path(__file__).resolve().parents[1]
REPOSITORY = STUDY.parents[1]
DEVELOPMENT_PANELS = tuple(f"{market}_{year}"
    for market in ("EUA_LONG","GUANGZHOU","HUBEI","SHANGHAI")
    for year in (2021,2022,2023) if (market,year) != ("SHANGHAI",2021))
ADDITIONAL_PANELS = ("SHENZHEN_PLOS_2020","SHENZHEN_PLOS_2021")
ALLOWED_PANELS = DEVELOPMENT_PANELS+ADDITIONAL_PANELS
ROLES = ("train","validation","test")
KNOWN_SOURCE_HASHES = {
    "EUA_LONG":"142f391305c482eb3bd58dcf8df477db90cf54402495527749a53e126cdf0e7f",
    "GUANGZHOU":"8d78a4f13c5a799519546fd3bfa0e0ac5eaa52b0441a983a95ef06f5e5e34051",
    "HUBEI":"04655d59c1b585f6c35c02a43469a965fe3589eb613d7d8d8e46996751b7d797",
    "SHANGHAI":"425424353bb3ac09e3d96b291680b4179e86f180da626a5d19e5c1e58ed960a1",
    "SHENZHEN_PLOS":"90c49501afa9beb7370c4409084a28729daa4904a7e7e5de310d0f6bd7a3ba10",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def private_directory(path):
    path = Path(path).resolve()
    if path == REPOSITORY or REPOSITORY in path.parents:
        raise ValueError("Private model/data files may not be stored in the Git repository")
    path.mkdir(parents=True,exist_ok=True)
    return path


def write_json_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as handle:
        json.dump(value,handle,indent=2,ensure_ascii=False,allow_nan=False)
        handle.write("\n")


def write_npz_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle,**value)


def validate_protocol(path):
    p = json.loads(Path(path).read_text())
    if p["synthetic_decision"] != "PASS_FINITE_SYNTHETIC_RESOURCE_GATE":
        raise ValueError("Market protocol does not record the required synthetic gate")
    if p["model"]["arms_GRU16"] != list(ARMS["GRU16"]):
        raise ValueError("GRU arms differ from the frozen protocol")
    if p["model"]["arms_MLP16"] != list(ARMS["MLP16"]):
        raise ValueError("MLP arms differ from the frozen protocol")
    if p["training"]["seeds"] != list(SEEDS) or p["training"]["epochs"] != EPOCHS:
        raise ValueError("Training budget differs from the frozen protocol")
    return p


def code_identity(protocol_path):
    files = [Path(__file__),STUDY/"source"/"persistence_market.py",
             STUDY/"source"/"persistence.py",STUDY/"source"/"mixture_metrics.py",
             STUDY/"data_sources"/"prepare_data.py"]
    return {"protocol_sha256":sha(protocol_path),
            "source_sha256":{str(p.relative_to(STUDY)):sha(p) for p in files}}


def read_prepared(directory,panel):
    if panel not in ALLOWED_PANELS:
        raise ValueError("Panel is outside the fixed 13-panel universe")
    path = Path(directory)/f"{panel}.npz"
    receipt_path = path.with_suffix(".json")
    if path.is_symlink() or receipt_path.is_symlink():
        raise ValueError("Prepared artifacts must be regular files")
    receipt = json.loads(receipt_path.read_text())
    digest = sha(path)
    if digest != receipt["prepared_sha256"] or receipt["cell"] != panel:
        raise ValueError("Prepared artifact hash or panel identity mismatch")
    market,year_text = panel.rsplit("_",1)
    if receipt["source"]["sha256"] != KNOWN_SOURCE_HASHES[market]:
        raise ValueError("Producer receipt is outside the frozen five-source whitelist")
    with np.load(path,allow_pickle=False) as content:
        # Inspect the small date/source fields before materializing any price,
        # return, feature or target member of this NPZ.
        dates = content["all_dates"].copy()
        cutoff = "2022-01-01" if market == "SHENZHEN_PLOS" else "2024-01-01"
        if not len(dates) or max(dates.tolist()) >= cutoff:
            raise ValueError("Reserved dates rejected before prepared numeric arrays are loaded")
        if str(content["source_sha256"].item()) != KNOWN_SOURCE_HASHES[market]:
            raise ValueError("Prepared source outside fixed hash whitelist")
        arrays = {key:content[key].copy() for key in content.files}
    if list(arrays["horizons"]) != list(HORIZONS):
        raise ValueError("Prepared horizon mismatch")
    if max(arrays["all_dates"].tolist()) >= "2024-01-01":
        raise ValueError("2024+ dates are outside the market protocol")
    if str(arrays["source_sha256"].item()) != receipt["source"]["sha256"]:
        raise ValueError("Prepared source identity disagrees with producer receipt")
    year = int(year_text)
    for role in ROLES:
        n = len(arrays[f"{role}_indices"])
        if arrays[f"{role}_X"].shape != (n,20,3) or arrays[f"{role}_y"].shape != (n,3):
            raise ValueError("Prepared feature or target shape mismatch")
        if arrays[f"{role}_future_normalized"].shape != (n,20):
            raise ValueError("Prepared future path shape mismatch")
        expected_y = np.cumsum(arrays[f"{role}_future_normalized"],axis=1)[:,np.array(HORIZONS)-1]/np.sqrt(HORIZONS)
        np.testing.assert_allclose(arrays[f"{role}_y"],expected_y,rtol=2e-12,atol=2e-11)
        np.testing.assert_array_equal(arrays[f"{role}_target_indices"],
                                      arrays[f"{role}_indices"][:,None]+np.array(HORIZONS))
        if np.any(arrays[f"{role}_daily_scale"] <= 0) or np.any(arrays[f"{role}_origin_price"] <= 0):
            raise ValueError("Nonpositive prepared scale/price")
        for suffix in ("X","y","future_normalized","unit","origin_price","target_price"):
            finite(arrays[f"{role}_{suffix}"],f"prepared {role}_{suffix}")
    train_end = int(arrays["train_indices"][-1])
    val_first,val_last = int(arrays["validation_indices"][0]),int(arrays["validation_indices"][-1])
    test_first = int(arrays["test_indices"][0])
    if train_end+20 >= val_first or val_last+20 >= test_first:
        raise ValueError("Maximum-horizon maturity crosses train/validation/test boundary")
    if int(arrays["train_label_end"]) != train_end+20:
        raise ValueError("Training label cutoff mismatch")
    all_years = np.array([str(date)[:4] for date in arrays["all_dates"].tolist()])
    year_ids = np.flatnonzero(all_years == str(year))
    if not len(year_ids):
        raise ValueError("Missing fixed annual panel")
    first = int(year_ids[0])
    expected_train = np.arange(39,first-167+1)
    expected_validation = np.arange(first-146,first-21+1)
    expected_test = np.flatnonzero((all_years[:-20] == str(year)) & (all_years[20:] == str(year)))
    if len(expected_train)<128 or len(expected_validation)!=126:
        raise ValueError("Fixed sample-size gate not satisfied")
    for role,expected in (("train",expected_train),("validation",expected_validation),("test",expected_test)):
        np.testing.assert_array_equal(arrays[f"{role}_indices"],expected)
    if any(str(value)[:4] != str(year) for value in arrays["test_dates"].tolist()):
        raise ValueError("Wrong test origin year")
    if any(str(value)[:4] != str(year) for value in arrays["test_target_dates"].reshape(-1).tolist()):
        raise ValueError("Wrong test maturity year")
    return arrays,{"panel":panel,"prepared_sha256":digest,
                   "producer_receipt_sha256":sha(receipt_path),
                   "source_sha256":receipt["source"]["sha256"],
                   "role_counts":{r:len(arrays[f"{r}_indices"]) for r in ROLES},
                   "producer":receipt}


def mixture_metrics_function():
    try:
        from .mixture_metrics import metrics
    except ImportError:
        from mixture_metrics import metrics
    return metrics


def score_role(arrays,role,prediction):
    metrics = mixture_metrics_function()
    individual,aggregate = {},{}
    for column,h in enumerate(HORIZONS):
        parts = {}
        n = len(arrays[f"{role}_indices"])
        for first in range(0,n,128):
            end = min(n,first+128)
            w = prediction[f"weights_h{h}"][first:end]
            value = metrics(arrays[f"{role}_y"][first:end,column],
                unit=arrays[f"{role}_unit"][first:end,column],
                price0=arrays[f"{role}_origin_price"][first:end],
                pricetrue=arrays[f"{role}_target_price"][first:end,column],
                w=w,means=np.zeros_like(w),
                sigmas=prediction[f"sigmas_h{h}"][first:end])
            for key,score in value.items():
                finite(score,f"{role} h{h} {key}")
                parts.setdefault(key,[]).append(np.asarray(score))
        values = {key:np.concatenate(blocks) for key,blocks in parts.items()}
        individual.update({f"score_h{h}_{key}":value for key,value in values.items()})
        row = {key:float(value.mean()) for key,value in values.items()}
        rw = row["rw_mean_price_squared_error"]
        row["mean_price_MSE_relative_RW"] = row["mean_price_squared_error"]/rw if rw>0 else None
        row["n"] = n
        aggregate[str(h)] = row
    return individual,aggregate


def forecast_artifact(arrays,model=None,reference=None,mode=None):
    private,aggregate = {},{}
    for role in ROLES:
        n = len(arrays[f"{role}_indices"])
        if role == "train":
            # Training loss is fully recorded at epochs0..40. Archive its final
            # f/rho only; complete densities and price scores can be reproduced
            # from the selected checkpoint and hash-bound prepared data.
            if model is not None:
                parameters = predict_parameters(model,arrays["train_X"])
            else:
                parameters = {"f":np.full(n,reference["f0"]),
                    "rho_h5":np.full(n,reference["rho_"+mode]),
                    "rho_h20":np.full(n,reference["rho_"+mode])}
            for key,value in parameters.items():
                finite(value,f"training {key}")
                private[f"train_{key}"]=value
            private["train_indices"]=arrays["train_indices"]
            private["train_dates"]=arrays["train_dates"]
            aggregate["train_parameters"] = {
                key:{"mean":float(value.mean()),"std":float(value.std()),
                     "minimum":float(value.min()),"maximum":float(value.max())}
                for key,value in parameters.items()}
            continue
        prediction = predict(model,arrays[f"{role}_X"]) if model is not None else statistical_prediction(reference,mode,n)
        scores,role_aggregate = score_role(arrays,role,prediction)
        for key,value in {**prediction,**scores}.items():
            private[f"{role}_{key}"]=value
        for h in HORIZONS:
            private[f"{role}_means_h{h}"]=np.zeros_like(prediction[f"weights_h{h}"])
        for suffix in ("indices","dates","target_indices","target_dates","y","unit","origin_price","target_price"):
            private[f"{role}_{suffix}"]=arrays[f"{role}_{suffix}"]
        aggregate[role] = role_aggregate
        aggregate[role+"_parameters"] = {
            key:{"mean":float(prediction[key].mean()),"std":float(prediction[key].std()),
                 "minimum":float(prediction[key].min()),"maximum":float(prediction[key].max())}
            for key in ("f","rho_h5","rho_h20")}
    return private,aggregate


def validate_resume(receipt,identity,prepared,private_dir):
    if receipt["identity"] != identity or receipt["prepared"]["prepared_sha256"] != prepared["prepared_sha256"]:
        raise ValueError("Resume refused: source/protocol or prepared hash changed")
    for name,digest in receipt["private_artifacts"].items():
        if sha(private_dir/name) != digest:
            raise ValueError("Resume refused: private artifact hash changed")


def reference_job(arrays,prepared,identity,private_dir,public_dir,resume):
    panel = prepared["panel"]
    path = private_dir/"references"/f"{panel}.json"
    receipt_path = public_dir/"references"/f"{panel}.json"
    if receipt_path.exists():
        if not resume:
            raise FileExistsError(f"Reference receipt already exists: {panel}")
        receipt = json.loads(receipt_path.read_text())
        validate_resume(receipt,identity,prepared,private_dir)
        return json.loads(path.read_text())
    if path.exists():
        raise FileExistsError("Incomplete reference job exists; no silent overwrite")
    reference = fit_references(arrays["train_future_normalized"],arrays["train_y"])
    write_json_new(path,reference)
    artifacts = {str(path.relative_to(private_dir)):sha(path)}
    aggregates = {}
    for mode in ("endpoint","joint005"):
        prediction,aggregate = forecast_artifact(arrays,reference=reference,mode=mode)
        forecast_path = private_dir/"forecasts"/f"{panel}__statistical_{mode}.npz"
        write_npz_new(forecast_path,prediction)
        artifacts[str(forecast_path.relative_to(private_dir))]=sha(forecast_path)
        aggregates[mode]=aggregate
    receipt = {"panel":panel,"completed_utc":now(),"identity":identity,
               "prepared":prepared,"private_artifacts":artifacts,
               "statistical_fits":3,"fitting_scope":"train only",
               "reference":reference,"metrics":aggregates}
    write_json_new(receipt_path,receipt)
    return reference


def neural_job(arrays,prepared,reference,identity,private_dir,public_dir,
               backbone,arm,seed,resume):
    panel = prepared["panel"]
    key = f"{panel}__{backbone}__{arm}__s{seed}"
    checkpoint_path = private_dir/"checkpoints"/f"{key}.pt"
    forecast_path = private_dir/"forecasts"/f"{key}.npz"
    receipt_path = public_dir/"fits"/f"{key}.json"
    if receipt_path.exists():
        if not resume:
            raise FileExistsError(f"Neural receipt already exists: {key}")
        receipt = json.loads(receipt_path.read_text())
        validate_resume(receipt,identity,prepared,private_dir)
        return receipt
    if checkpoint_path.exists() or forecast_path.exists():
        raise FileExistsError("Incomplete neural job exists; no silent overwrite")
    def progress(epoch,steps,selected_epoch):
        if epoch%10 == 0:
            print(json.dumps({"stage":"training","job":key,"epoch":epoch,
                              "optimizer_steps":steps,"selected_epoch":selected_epoch}),flush=True)
    model,fit = fit_neural(arrays,reference,backbone,arm,seed,progress=progress)
    checkpoint = {"schema_version":1,"panel":panel,"backbone":backbone,"arm":arm,
        "seed":seed,"selected_epoch":fit["selected_epoch"],"reference":reference,
        "model_state_dict":model.state_dict(),
        "selected_optimizer_state_dict":model.selected_optimizer_state,
        "final_optimizer_state_dict":model.final_optimizer_state,
        "final_minibatch_generator_state":model.final_minibatch_generator_state,
        "completed_optimizer_steps":fit["optimizer_steps"],
        "identity":identity,"prepared_sha256":prepared["prepared_sha256"],
        "source_sha256":prepared["source_sha256"],
        "feature_mean":arrays["feature_mean"].tolist(),
        "feature_sd":arrays["feature_sd"].tolist(),"floor":float(arrays["floor"]),
        "train_label_end":int(arrays["train_label_end"]),
        "train_indices":arrays["train_indices"].tolist(),
        "validation_indices":arrays["validation_indices"].tolist(),
        "fit_receipt":fit,"note":"Selected model state is for inference; final optimizer is separately labeled, not paired to selected weights."}
    checkpoint_path.parent.mkdir(parents=True,exist_ok=True)
    with checkpoint_path.open("xb") as handle:
        torch.save(checkpoint,handle)
    prediction,aggregate = forecast_artifact(arrays,model=model)
    write_npz_new(forecast_path,prediction)
    receipt = {"job":key,"panel":panel,"completed_utc":now(),"identity":identity,
               "prepared":prepared,"fit":fit,"metrics":aggregate,
               "private_artifacts":{
                   str(checkpoint_path.relative_to(private_dir)):sha(checkpoint_path),
                   str(forecast_path.relative_to(private_dir)):sha(forecast_path)},
               "market_neural_fits":1,"test_parameter_updates":0,
               "shared_path_distribution":not model.independent}
    write_json_new(receipt_path,receipt)
    return receipt


def implementation_smoke(public_dir):
    """Eight architectures x two steps. Pure synthetic implementation check."""
    started = time.perf_counter()
    rng = np.random.default_rng(261019)
    x = rng.normal(size=(32,20,3))
    path = sample_paths(rng,np.full(32,.5))
    y = np.cumsum(path,axis=1)[:,np.array(HORIZONS)-1]/np.sqrt(HORIZONS)
    reference = {"pi":PI.tolist(),"f0":1.,"rho_endpoint":.5,"rho_joint005":.5}
    results = []
    for backbone,arms in ARMS.items():
        for arm in arms:
            torch.manual_seed(17)
            model = PersistenceMarketModel(backbone,arm,reference)
            optimizer = torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.001)
            tensors = (torch.as_tensor(x,dtype=torch.float32),torch.as_tensor(y,dtype=torch.float64),
                       torch.as_tensor(path,dtype=torch.float64))
            initial = predict(model,x[:4])
            np.testing.assert_allclose(initial["f"],1.,rtol=0,atol=1e-14)
            np.testing.assert_allclose(initial["rho_h5"],.5,rtol=0,atol=1e-14)
            losses=[]
            for step in range(2):
                values = model.loss_components(*tensors)
                loss = values["objective"].mean()
                finite(loss,"synthetic smoke objective")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
            forecast = predict(model,x[:4])
            for value in forecast.values():
                finite(value,"smoke forecast")
            results.append({"backbone":backbone,"arm":arm,"optimizer_steps":2,
                            "parameters":sum(p.numel() for p in model.parameters()),
                            "objective_before_steps":losses,
                            "last_gradient_norm_before_clip":float(gradient_norm)})
    receipt = {"passed":True,"scope":"Implementation smoke only; no market arrays or parameter selection",
        "created_utc":now(),"synthetic_origin_paths":32,"architectures":8,
        "synthetic_gradient_steps":16,"market_fits":0,"reference_optimizer_fits":0,
        "models":results,"elapsed_seconds":time.perf_counter()-started,
        "source_sha256":{p.name:sha(p) for p in [Path(__file__),STUDY/"source"/"persistence_market.py"]}}
    target = public_dir/"PERSISTENCE_IMPLEMENTATION_SMOKE.json"
    write_json_new(target,receipt)
    print(json.dumps({"stage":"implementation_smoke_complete","passed":True,
                      "synthetic_gradient_steps":16,"market_fits":0,"receipt":str(target)}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir",type=Path)
    parser.add_argument("--private-dir",type=Path)
    parser.add_argument("--public-dir",type=Path,required=True)
    parser.add_argument("--protocol",type=Path,default=STUDY/"provenance"/"MARKET_PROTOCOL.json")
    parser.add_argument("--panels",nargs="*")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--threads",type=int,default=1)
    parser.add_argument("--smoke",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    args.public_dir.mkdir(parents=True,exist_ok=True)
    if args.smoke:
        implementation_smoke(args.public_dir)
        return
    if args.prepared_dir is None or args.private_dir is None:
        parser.error("Market execution requires --prepared-dir and --private-dir")
    validate_protocol(args.protocol)
    identity = code_identity(args.protocol)
    private_dir = private_directory(args.private_dir)
    panels = args.panels or [p for p in ALLOWED_PANELS if (args.prepared_dir/f"{p}.npz").exists()]
    if not panels or len(set(panels)) != len(panels) or any(p not in ALLOWED_PANELS for p in panels):
        raise ValueError("Invalid or empty prepared panel selection")
    run_key = hashlib.sha256(json.dumps(sorted(panels)).encode()).hexdigest()[:12]
    start_path = args.public_dir/f"START_{run_key}.json"
    start = {"started_utc":now(),"identity":identity,"panels":panels,
             "planned_neural_fits":len(panels)*24,"planned_statistical_fits":len(panels)*3,
             "threads":args.threads,"source_implementer_role":"not independent audit"}
    if start_path.exists():
        previous = json.loads(start_path.read_text())
        if not args.resume or previous["identity"] != identity or previous["panels"] != panels:
            raise FileExistsError("Existing run start receipt disagrees; no overwrite")
    else:
        write_json_new(start_path,start)
    started = time.perf_counter()
    receipts=[]
    for panel in panels:
        arrays,prepared = read_prepared(args.prepared_dir,panel)
        reference = reference_job(arrays,prepared,identity,private_dir,args.public_dir,args.resume)
        print(json.dumps({"stage":"references_ready","panel":panel,"static_fits":3}),flush=True)
        for backbone,arms in ARMS.items():
            for arm in arms:
                for seed in SEEDS:
                    result = neural_job(arrays,prepared,reference,identity,private_dir,args.public_dir,
                                        backbone,arm,seed,args.resume)
                    receipts.append(result)
                    print(json.dumps({"stage":"job_complete","completed":len(receipts),
                        "planned":len(panels)*24,"job":result["job"],
                        "elapsed_seconds":time.perf_counter()-started}),flush=True)
    manifest = {"completed_utc":now(),"identity":identity,"panels":panels,
        "neural_fits_completed":len(receipts),"static_fits_completed":3*len(panels),
        "optimizer_steps":sum(r["fit"]["optimizer_steps"] for r in receipts),
        "epochs_completed":sum(r["fit"]["epochs_run"] for r in receipts),
        "epoch_zero_selected":sum(r["fit"]["epoch_zero_selected"] for r in receipts),
        "jobs":[{"job":r["job"],"receipt_sha256":sha(args.public_dir/"fits"/f"{r['job']}.json")}
                for r in receipts],"elapsed_seconds":time.perf_counter()-started,
        "no_test_parameter_updates":True}
    final_path = args.public_dir/f"COMPLETE_{run_key}.json"
    if final_path.exists():
        if not args.resume:
            raise FileExistsError("Completed manifest already exists")
        old = json.loads(final_path.read_text())
        if old["identity"] != identity or old["jobs"] != manifest["jobs"]:
            raise ValueError("Completed manifest differs on resume")
    else:
        write_json_new(final_path,manifest)
    print(json.dumps({"stage":"complete","neural_fits":len(receipts),
                      "static_fits":3*len(panels),"optimizer_steps":manifest["optimizer_steps"]}),flush=True)


if __name__ == "__main__":
    main()
