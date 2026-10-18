"""Independent synthetic-only integration/causality audit of the frozen runner.

Both regimes test execution, finite predictions, chronology, and artifact
integrity. Relative errors are descriptive only: no finite-sample improvement
is a required assertion or evidence for a forecasting contribution.
"""
from __future__ import annotations
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import model as M
import pipeline as P
import run as R


def synthetic(regime, n=1400):
    rng = np.random.default_rng(9381 if regime == "persistent" else 9382)
    z = np.zeros(n)
    state_noise = rng.normal(size=n)
    for j in range(1, n):
        z[j] = .95 * z[j-1] + np.sqrt(1-.95**2) * state_noise[j]
    ret = .01 * rng.normal(size=n)
    if regime == "persistent":
        ret[1:] += .003 * z[:-1]
    else:
        # The primary target is P, so use an exact geometric price martingale.
        # Zero Gaussian log drift alone would give positive Jensen price drift.
        ret -= .5 * .01**2
    ret[0] = 0
    price = 50 * np.exp(np.cumsum(ret))
    d = .01 * z
    q = .012 + .002 / (1 + np.exp(-z))
    return pd.DataFrame({"source_index": np.arange(n), "date": [None] * n,
                         "price": price, "high": price*np.exp(d+q),
                         "low": price*np.exp(d-q), "open": np.nan,
                         "volume": np.nan})


def causal_and_scaler_audit():
    frame = synthetic("persistent")
    original = P.OriginData(frame)
    altered = frame.copy()
    origin = 1100
    # Change a future suffix and all non-price history; keep historical P exact.
    altered.loc[origin:, ["price", "high", "low"]] *= 1.4
    changed = P.OriginData(altered)
    tr, va, final, te, selection, prepared = R.fold_setup(original, origin, 1380)
    tr2, va2, final2, te2, selection2, prepared2 = R.fold_setup(changed, origin, 1380)
    for idx1, idx2 in [(tr,tr2), (va,va2), (final,final2), (te,te2)]:
        np.testing.assert_array_equal(idx1, idx2)
    assert original.origin[tr].max()+20 < original.origin[va].min()
    assert original.origin[final].max()+20 < original.origin[te].min()
    for obj1, obj2 in [(selection, selection2), (prepared, prepared2)]:
        for name in obj1.state():
            np.testing.assert_array_equal(obj1.state()[name], obj2.state()[name])
    prefix = np.flatnonzero(original.origin < origin)
    for name in ("x", "context", "p0", "scale", "basis0", "radius0"):
        np.testing.assert_array_equal(getattr(original,name)[prefix], getattr(changed,name)[prefix])
    rangechanged = frame.copy()
    rangechanged.high *= 1.3
    rangechanged.low *= .7
    rangechanged.open = frame.price * 1.2
    rangechanged.volume = np.linspace(1, 10000, len(frame))
    rangedata = P.OriginData(rangechanged)
    rangeprep = P.Prepared.fit(rangedata, final)
    for name in ("context", "scale", "p0"):
        np.testing.assert_array_equal(getattr(original,name), getattr(rangedata,name))
    np.testing.assert_array_equal(prepared.classical(te,True), rangeprep.classical(te,True))
    torch.manual_seed(722)
    price_model = M.make_model("gru_price")
    with torch.no_grad():
        price_model.head.weight.normal_(0,.1)
        p1 = R.forward(price_model,prepared.tensors(te),None)["mean"]
        p2 = R.forward(price_model,rangeprep.tensors(te),None)["mean"]
    torch.testing.assert_close(p1,p2,atol=0,rtol=0)
    # The runner must reject numerical clipping as well as nonfinite outputs.
    try:
        R.ensure_finite({"mean": torch.ones(1,3), "samples": None,
                         "clip_count": torch.tensor(1)})
    except RuntimeError:
        clip_rejected = True
    else:
        raise AssertionError("Numerical clipping was accepted")
    return {"future_suffix_cannot_change_past_inputs_or_fitted_transforms": True,
            "range_open_volume_changes_cannot_change_price_only_inputs_or_predictions": True,
            "selection_max_label": int(original.origin[tr].max()+20),
            "validation_first_origin": int(original.origin[va].min()),
            "final_max_label": int(original.origin[final].max()+20),
            "test_first_origin": int(original.origin[te].min()),
            "clip_is_hard_failure": clip_rejected}


def learning_smoke():
    results = []
    for regime in ("persistent", "martingale"):
        data = P.OriginData(synthetic(regime))
        tr, va, final, te, selection, prep = R.fold_setup(data,1100,1380)
        # Fixed ten epochs; no validation selection or family-dependent stopping.
        for family in ("summary_state", "state_price", "gru_full", "gru_price"):
            model, receipt = R.fit_neural(family,selection,tr,np.array([],int),
                                          seed=717,lr=.001,epochs=10,select=False)
            mean, samples = R.predict(model,selection,te)
            assert np.isfinite(mean).all() and np.isfinite(samples).all()
            assert mean.shape == (len(te),3)
            assert receipt["selected_epoch"] == 10
            assert receipt["optimizer_steps"] == 10 * int(np.ceil(len(tr)/R.BATCH))
            with torch.no_grad():
                direct = R.forward(model,selection.tensors(te),M.sobol_normal(128,908113))
            np.testing.assert_allclose(mean,direct["mean"].numpy(),rtol=1e-13,atol=1e-13)
            with tempfile.TemporaryDirectory(prefix="carbon-synthetic-audit-") as directory:
                target = Path(directory) / "forecast.npz"
                R.save_forecast(target,selection,te,mean,samples,{"synthetic":True})
                with np.load(target,allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["mean"],mean)
                    np.testing.assert_array_equal(saved["truth"],data.y[te])
                    np.testing.assert_array_equal(saved["target_index"],data.origin[te,None]+np.array([1,5,20])[None])
                metadata = json.loads(target.with_suffix(".json").read_text())
                assert metadata["forecast_sha256"] == P.sha256(target)
            mse = np.mean((mean-data.y[te,:,0])**2,axis=0)
            rw_mse = np.mean((data.p0[te,None]-data.y[te,:,0])**2,axis=0)
            result = {"regime":regime,"family":family,"epochs":10,
                      "optimizer_steps":receipt["optimizer_steps"],
                      "finite_forward_backward_and_saved_analytic_mean":True,
                      "first_epoch_training_loss":receipt["history"][0]["train_loss"],
                      "last_epoch_training_loss":receipt["history"][-1]["train_loss"],
                      "descriptive_mse_ratio_to_rw_by_horizon":(mse/rw_mse).tolist(),
                      "elapsed_seconds":receipt["elapsed_seconds"]}
            print(json.dumps(result),flush=True)
            results.append(result)
    return results


if __name__ == "__main__":
    result = {"synthetic_only":True,"forecasting_success_claim":False,
              "martingale_definition":"P_t=P_(t-1)*exp(-0.5*0.01^2+0.01*epsilon_t); range state is independent of returns",
              "code_fingerprint":R.fingerprint(),
              "hashes":{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ("model.py","pipeline.py","run.py")},
              "causality":causal_and_scaler_audit(),
              "fit_smoke":learning_smoke()}
    Path(__file__).with_suffix(".json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":"PASS","audit_path":str(Path(__file__).with_suffix('.json'))}))
