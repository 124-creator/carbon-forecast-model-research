"""Post-screen descriptive failure decomposition; no fitting or re-selection."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    selection_path = root / "results/D1_SELECTION.json"
    scope_path = root / "provenance/POST_D1_DIAGNOSTIC_SCOPE.json"
    scope = json.loads(scope_path.read_text())
    if hashlib.sha256(selection_path.read_bytes()).hexdigest() != scope["selection_sha256"]:
        raise RuntimeError("Frozen selection identity differs")
    selection = json.loads(selection_path.read_text())
    if selection["advance_to_D2"]:
        raise RuntimeError("This diagnostic is for the failed D1 screen")
    out = root / "results/D1_FAILURE_DECOMPOSITION.json"
    if out.exists():
        raise RuntimeError("Refuse overwriting diagnostic")
    rows = []
    max_identity_error = 0.
    for family, config in selection["selected"].items():
        for market, cell in config["by_market"].items():
            if len(cell["job_ids"]) != 1:
                raise ValueError("D1 has one risk realization per selected series")
            job_id = cell["job_ids"][0]
            with np.load(root / "local_only/D1" / job_id / "forecasts.npz", allow_pickle=False) as pack:
                pred = pack["predicted_change"].copy()
                truth = pack["target_change"].copy()
            denominator = np.mean(truth ** 2, axis=0)
            pred_term = np.mean(pred ** 2, axis=0) / denominator
            cross = 2 * np.mean(pred * truth, axis=0) / denominator
            relative = np.mean((pred - truth) ** 2, axis=0) / denominator
            difference = float(np.max(abs(1 + pred_term - cross - relative)))
            max_identity_error = max(max_identity_error, difference)
            if difference > 1e-10:
                raise AssertionError("MSE decomposition does not agree")
            corr = []
            for h in range(20):
                corr.append(float(np.corrcoef(pred[:, h], truth[:, h])[0, 1])
                            if np.std(pred[:, h]) > 1e-14 and np.std(truth[:, h]) > 1e-14 else None)
            rms = np.sqrt(denominator)
            row = dict(family=family, config_id=config["config_id"], market=market,
                       n=len(pred), job_id=job_id, relative_mse=relative.tolist(),
                       prediction_energy_over_rw=pred_term.tolist(), twice_alignment_over_rw=cross.tolist(),
                       prediction_rms_over_rw=np.sqrt(pred_term).tolist(),
                       mean_prediction_over_rw_rms=(pred.mean(axis=0) / rms).tolist(),
                       mean_truth_over_rw_rms=(truth.mean(axis=0) / rms).tolist(),
                       mean_error_over_rw_rms=((pred - truth).mean(axis=0) / rms).tolist(),
                       correlation=corr)
            fit_path = root / "results/D1/jobs" / job_id / "FIT.json"
            if fit_path.exists():
                fit = json.loads(fit_path.read_text())
                row["fit"] = {k:fit[k] for k in ("selected_epoch", "executed_epochs", "optimizer_steps",
                                                 "initial_validation_score", "selected_validation_score",
                                                 "initial_state_sha256", "selected_state_sha256") if k in fit}
            rows.append(row)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope_sha256=hashlib.sha256(scope_path.read_bytes()).hexdigest(),
                  selection_sha256=scope["selection_sha256"], new_fits=0, confirmation_access=False,
                  interpretation="Post-selection development description; decomposition is exact algebra, not a causal economic explanation",
                  maximum_decomposition_identity_error=max_identity_error, rows=rows)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    eex = next(r for r in rows if r["family"] == "smp_full" and r["market"] == "EEX_EUA_T3PA")
    print(json.dumps({"checked_series_models":len(rows),"max_identity_error":max_identity_error,
                      "EEX_SMP":{k:([v[i] for i in (0,4,19)] if isinstance(v,list) and len(v)==20 else v)
                                 for k,v in eex.items() if k not in ("correlation",)}}))


if __name__ == "__main__":
    main()
