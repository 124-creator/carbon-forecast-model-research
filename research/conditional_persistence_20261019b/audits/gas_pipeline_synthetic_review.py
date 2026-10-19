"""Full synthetic-only pipeline check; no development file is opened."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"source"))
import run_gas_benchmark as run


def main():
    torch.set_num_threads(1)
    rng = np.random.default_rng(20261019)
    dates = pd.bdate_range("2018-01-01",periods=680).strftime("%Y-%m-%d").to_numpy(dtype="U10")
    returns = np.r_[0.,rng.normal(0,.012,679)]
    price = 80*np.exp(np.cumsum(returns))
    first = int(np.flatnonzero(np.char.startswith(dates,"2020"))[0])
    val_end = first-21; val_start = val_end-125; train_end = val_start-21
    roles = {"train":np.arange(39,train_end+1),"validation":np.arange(val_start,val_end+1),
             "test":np.arange(first,len(dates)-20)}
    horizons = np.array([1,5,20])
    base = {"horizons":horizons,"all_dates":dates,"all_returns":returns,
            "train_label_end":np.array(train_end+20),"source_sha256":np.array("SYNTHETIC_NOT_MARKET")}
    for role,ids in roles.items():
        unit = np.broadcast_to(.012*np.sqrt(horizons),(len(ids),3)).copy()
        targets = ids[:,None]+horizons
        base.update({f"{role}_indices":ids,f"{role}_dates":dates[ids],
                     f"{role}_origin_price":price[ids],f"{role}_target_price":price[targets],
                     f"{role}_target_indices":targets,f"{role}_target_dates":dates[targets],
                     f"{role}_unit":unit,f"{role}_y":np.log(price[targets]/price[ids,None])/unit})
    private = Path(tempfile.mkdtemp(prefix="carbon-gas-pipeline-synthetic-"))
    fingerprint = "SYNTHETIC_PIPELINE_REVIEW_NOT_A_FORMAL_MARKET_FIT"
    receipts = []
    for h in (5,20):
        arrays,norm = run.build_prepared(base,h)
        cell = f"SYNTHETIC_REVIEW_h{h}"
        path = private/"prepared"/f"{cell}.npz"
        run.write_npz(path,**arrays)
        run.write_json(path.with_suffix(".json"),{"source_fingerprint":fingerprint,
            "prepared_sha256":run.sha(path),"normalizers":norm})
        run.save_all_controls(private,cell,arrays,fingerprint)
        for name in run.NORMALIZERS:
            assert arrays[f"train_X_{name}"].shape == (len(roles["train"]),20,1)
            torch.manual_seed(17)
            model = run.GRUMixture()
            w,m,s = run.predict(model,arrays,"test",name)
            with np.load(private/"predictions"/cell/f"{name}_no_neural"/"seed_0.npz",allow_pickle=False) as baseline:
                np.testing.assert_array_equal(w,baseline["weights"])
                np.testing.assert_array_equal(m,baseline["means"])
                np.testing.assert_array_equal(s,baseline["sigmas"])
            receipt = run.train_one(path,name,17,private,fingerprint)
            assert receipt["epochs_executed"] == 40 and len(receipt["history"]) == 41
            assert receipt["neural_parameters"] == 1065
            assert receipt["gradient_updates"] == 40*int(np.ceil(len(roles["train"])/64))
            again = run.train_one(path,name,17,private,fingerprint)
            assert again == receipt  # complete resume performs no new fitting
            receipts.append({"h":h,"normalizer":name,"epochs":40,"history_rows":41,
                             "gradient_updates":receipt["gradient_updates"],
                             "neural_parameters":receipt["neural_parameters"],
                             "selected_epoch":receipt["selected_epoch"]})
    public = {"synthetic_only":True,"market_values_opened":False,
              "synthetic_static_GAS_fits":2,"synthetic_full_40_epoch_neural_smoke_fits":6,
              "epoch0_exact_forecast_ties":True,"complete_resume_no_refit":True,
              "roles_have_correct_common_max20_maturity":True,"fits":receipts,
              "private_synthetic_archive":str(private),
              "source_sha256":{name:hashlib.sha256((ROOT/"source"/name).read_bytes()).hexdigest()
                                for name in ("gas_filter.py","mixture_metrics.py","run_gas_benchmark.py")}}
    run.write_json(ROOT/"audits"/"GAS_PIPELINE_SYNTHETIC_REVIEW.json",public)
    print(json.dumps(public,indent=2))


if __name__ == "__main__":
    main()
