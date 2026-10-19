"""Controlled adaptation of the information-boundary comparison in SWEWT (2024).

This is NOT the paper's author-code reproduction.  In particular we use a
residual ridge predictor, fixed EWT settings and different, already exposed
development series.  FULL_SERIES_RETROSPECTIVE intentionally sees future data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from ewtpy import EWT1D
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


MARKETS = ["GUANGZHOU", "HUBEI", "SHANGHAI_D4", "EU_REPORTED"]
WINDOW, LOOKBACK, MODES, HORIZON = 64, 17, 3, 1
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decompose(p: np.ndarray) -> np.ndarray:
    components, _, _ = EWT1D(
        p, N=MODES, log=0, detect="locmax", completion=1,
        reg="average", lengthFilter=10, sigmaFilter=5,
    )
    if components.shape != (len(p), MODES) or not np.isfinite(components).all():
        raise ValueError(f"Invalid EWT output {components.shape}")
    return components


def features(p: np.ndarray, t: int, mode: str, full=None) -> np.ndarray:
    raw = p[t-LOOKBACK+1:t+1] - p[t]
    if mode == "RAW":
        return raw.copy()
    if mode == "SLIDING_PAST_ONLY":
        e = decompose(p[t-WINDOW+1:t+1])[-LOOKBACK:].copy()
    elif mode == "FULL_SERIES_RETROSPECTIVE":
        e = full[t-LOOKBACK+1:t+1].copy()
    else:
        raise ValueError(mode)
    # Translation-equivariant features.  EWT's low band carries the DC level;
    # all higher bands have zero DC response. No target enters this centering.
    e[:, 0] -= p[t]
    return np.column_stack([raw, e]).ravel()


def run(repo: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "development_information_diagnostic_not_confirmation",
        "paper": "10.3390/en17174358",
        "adaptation": "fixed ewtpy + residual Ridge; not author-code reproduction",
        "window": WINDOW, "lookback": LOOKBACK, "modes": MODES,
        "horizon_observation_steps": HORIZON,
        "split": "price-prefix 70/10/20; purge labels at each boundary",
        "predictor": "Ridge; intercept; feature scaler fit only on training rows",
        "alpha_grid": ALPHAS, "selection": "minimum validation native-price MSE",
        "full_series_branch": "intentional retrospective full-series decomposition, not feasible forecasting",
        "source_sha256": sha(Path(__file__)),
        "python": platform.python_version(), "numpy": np.__version__,
    }
    (out / "PROTOCOL_BEFORE_RUN.json").write_text(json.dumps(protocol, indent=2))
    rows, audits, manifests, outputs = [], [], [], []
    for market in MARKETS:
        path = repo / "research/cross_market_innovation_20261017/data" / f"{market}_development.csv"
        df = pd.read_csv(path).sort_values("date").reset_index(drop=True)
        p = df.price.to_numpy(dtype=float)
        assert df.date.is_unique and np.isfinite(p).all()
        n = len(p)
        train_end, val_end = int(0.7*n), int(0.8*n)
        origins = np.arange(WINDOW-1, n-HORIZON)
        masks = {
            "train": origins+HORIZON < train_end,
            "val": (origins >= train_end) & (origins+HORIZON < val_end),
            "test": origins >= val_end,
        }
        y = p[origins+HORIZON]-p[origins]
        full = decompose(p)
        t = train_end
        perturbed = p.copy()
        j = np.arange(n-t-1)
        perturbed[t+1:] += np.std(np.diff(p[:train_end]))*(5.0+np.sin(j/5))
        causal_difference = float(np.max(np.abs(
            features(p, t, "SLIDING_PAST_ONLY") - features(perturbed, t, "SLIDING_PAST_ONLY")
        )))
        retrospective_difference = float(np.max(np.abs(
            features(p, t, "FULL_SERIES_RETROSPECTIVE", full) -
            features(perturbed, t, "FULL_SERIES_RETROSPECTIVE", decompose(perturbed))
        )))
        assert causal_difference == 0.0 and retrospective_difference > 1e-8
        audits.append({"market": market, "cutoff_date": str(df.date.iloc[t]),
                       "future_perturbation_causal_max_abs": causal_difference,
                       "future_perturbation_retrospective_max_abs": retrospective_difference})
        manifests.append({"market": market, "path": str(path.relative_to(repo)),
                          "sha256": sha(path), "rows": n,
                          "start": str(df.date.iloc[0]), "end": str(df.date.iloc[-1]),
                          "last_training_label": str(df.date.iloc[train_end-1]),
                          "first_validation_origin": str(df.date.iloc[train_end]),
                          "last_validation_label": str(df.date.iloc[val_end-1]),
                          "first_test_origin": str(df.date.iloc[val_end]),
                          "counts": {k: int(v.sum()) for k, v in masks.items()}})
        rw_mse = float(np.mean(y[masks["test"]]**2))
        for mode in ["RAW", "SLIDING_PAST_ONLY", "FULL_SERIES_RETROSPECTIVE"]:
            x = np.stack([features(p, int(t), mode, full) for t in origins])
            scaler = StandardScaler().fit(x[masks["train"]])
            x = scaler.transform(x)
            choices = []
            for alpha in ALPHAS:
                model = Ridge(alpha=alpha).fit(x[masks["train"]], y[masks["train"]])
                vmse = float(np.mean((model.predict(x[masks["val"]])-y[masks["val"]])**2))
                choices.append((vmse, alpha, model))
            vmse, alpha, model = min(choices, key=lambda z: (z[0], z[1]))
            pred = model.predict(x[masks["test"]])
            err = pred-y[masks["test"]]
            mse = float(np.mean(err**2))
            row = {"market": market, "mode": mode, "alpha": alpha,
                   "validation_mse": vmse, "validation_rw_mse": float(np.mean(y[masks["val"]]**2)),
                   "test_mse": mse, "test_rw_mse": rw_mse,
                   "test_mse_over_rw": mse/rw_mse, "test_rmse": mse**0.5,
                   "test_mae": float(np.mean(np.abs(err))), "test_n": len(pred),
                   "tuning_trials": len(ALPHAS)}
            rows.append(row)
            for origin, yp, yt in zip(origins[masks["test"]], pred, y[masks["test"]]):
                outputs.append({"market": market, "mode": mode, "origin_row": int(origin),
                                "origin_date": str(df.date.iloc[origin]),
                                "predicted_change": float(yp), "observed_change": float(yt)})
            print(json.dumps(row), flush=True)
    pd.DataFrame(rows).to_csv(out / "METRICS.csv", index=False)
    pd.DataFrame(outputs).to_csv(out / "PREDICTIONS.csv", index=False)
    (out / "FUTURE_PERTURBATION_AUDIT.json").write_text(json.dumps(audits, indent=2))
    (out / "INPUT_MANIFEST.json").write_text(json.dumps(manifests, indent=2))
    aggregate = pd.DataFrame(rows).groupby("mode").test_mse_over_rw.mean().to_dict()
    (out / "SUMMARY.json").write_text(json.dumps({
        "equal_market_mean_mse_over_rw": aggregate,
        "fitted_models": len(MARKETS)*3*len(ALPHAS),
        "selected_forecasters": len(rows), "causal_audit_pass": True,
        "claims": "Information-protocol sensitivity only; all periods already exposed development data.",
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.repo.resolve(), args.out.resolve())
