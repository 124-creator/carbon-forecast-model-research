"""Corrected daily Parkinson HAR experiment (exploratory reanalysis).

Each asset uses its own sorted source trading dates. At close t, all predictors
include t and forecasts target the next source trading date t+1. No pooled panel,
return-square replacement, positive-target deletion, or test-selected settings.

Target: log(high / low)^2 / (4 log 2), including exact zeros.
Test: target date >= 2024-01-01. Expanding estimation, every 22 test origins.
Epsilon = max(1e-12, 1e-4 * median(positive pre-2024 target)); also forecast floor.
Log HAR uses averages of log(y+epsilon) over exactly 1/5/22 days through t.
Log-OLS corrections use only estimation residuals; forecasts subtract epsilon
before applying the positive floor. Level HAR uses averages of y itself.
QLIKE loglink uses the same log-HAR regressors but fits conditional E[y] directly
by log(f)+y/f, with no offset subtraction or smearing needed.

All models are scored on exactly the same targets, including zero. Report loss
difference MODEL-MA22 (negative better); HAC t uses MA22-MODEL (positive better).
Traditional QLIKE is secondary and uses one identical positive-target subset.
No percentage comparisons of log(f)+y/f: the additive level is scale-dependent.
RW at a zero current target is floored, so its scores are epsilon-dependent.
"""

from pathlib import Path
import hashlib
import json
import math
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

SRC = Path(r"E:\Agent员工\碳价格预测研究\独立数据归档\_legacy_202605_archive"
           r"\主要实验推进_审计加固版_CLEAN_20260528\08-新实验"
           r"\22_structural_ohlc_q2_route_20260609\03_raw_data\normalized_yahoo_ohlc")
OUT = Path(__file__).resolve().parent
SYMS = ("KRBN", "KEUA", "KCCA")
HAR_LAGS = (1, 5, 22)
TEST_START = pd.Timestamp("2024-01-01")
REFIT_EVERY = 22
MIN_TRAIN = 150
MODELS = ("RW", "MA22", "HAR_raw", "HAR_ln", "HAR_sm",
          "HAR_level_OLS", "HAR_QLIKE_loglink")


def load(symbol):
    path = SRC / f"{symbol}_ohlc.csv"
    d = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    assert not d.date.isna().any() and not d.date.duplicated().any(), symbol
    h, l = d.high.to_numpy(float), d.low.to_numpy(float)
    valid = np.isfinite(h) & np.isfinite(l) & (h >= l) & (l > 0)
    if not valid.all():
        raise ValueError(f"{symbol}: invalid H/L rows {d.loc[~valid, 'date'].tolist()}")
    y = np.log(h / l) ** 2 / (4 * np.log(2.0))
    assert np.isfinite(y).all() and (y >= 0).all()
    pre = (d.date < TEST_START).to_numpy()
    positive = y[pre & (y > 0)]
    if len(positive) == 0:
        raise ValueError(f"{symbol}: no positive pre-2024 observations")
    eps = max(1e-12, float(np.median(positive)) * 1e-4)
    info = {"source": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_rows": len(d), "source_start": str(d.date.iloc[0].date()),
            "source_end": str(d.date.iloc[-1].date()), "source_zero_targets": int((y == 0).sum()),
            "pre2024_rows": int(pre.sum()), "epsilon": eps,
            "pre2024_positive_median": float(np.median(positive))}
    return d.date.to_numpy(), y, eps, info


def har_row(x, t):
    """Inclusive t: precisely L asset observations, never L+1."""
    assert t >= max(HAR_LAGS) - 1
    return np.array([np.mean(x[t - lag + 1:t + 1]) for lag in HAR_LAGS])


def qlike_series(y, forecast):
    y, forecast = np.asarray(y, float), np.asarray(forecast, float)
    assert np.isfinite(y).all() and (y >= 0).all()
    assert np.isfinite(forecast).all() and (forecast > 0).all()
    return np.log(forecast) + y / forecast


def nw_t(improvement, lag=5):
    """Positive means improvement > 0, i.e. the named model beats MA22."""
    d = np.asarray(improvement, float)
    assert np.isfinite(d).all()
    n = len(d)
    if n < 3:
        return np.nan
    e = d - d.mean()
    var = float(e @ e / n)
    for k in range(1, min(lag, n - 1) + 1):
        var += 2 * (1 - k / (lag + 1)) * float(e[k:] @ e[:-k] / n)
    return float(d.mean() / np.sqrt(var / n)) if var > 0 else np.nan


def standardize(x):
    mean, sd = x.mean(axis=0), x.std(axis=0)
    sd = np.where(sd > 1e-14, sd, 1.0)
    return np.column_stack((np.ones(len(x)), (x - mean) / sd)), mean, sd


def fit_models(xlog, xlevel, target, eps):
    a, mean_log, sd_log = standardize(xlog)
    b_log = np.linalg.lstsq(a, np.log(target + eps), rcond=None)[0]
    residual = np.log(target + eps) - a @ b_log
    smear = float(np.mean(np.exp(residual)))
    s2 = float(np.sum(residual ** 2) / (len(target) - a.shape[1]))
    al, mean_level, sd_level = standardize(xlevel)
    scale = float(target.mean())
    assert scale > 0
    b_level = np.linalg.lstsq(al, target / scale, rcond=None)[0]
    scaled_y = target / scale

    def objective(beta):
        eta = a @ beta
        # Convex QLIKE objective, expressed relative to training target mean.
        # Reject overflow proposals rather than silently clipping the fit.
        if (-eta).max() > 700:
            return np.inf, np.zeros_like(beta)
        ratio = scaled_y * np.exp(-eta)
        return float(np.mean(eta + ratio)), a.T @ (1 - ratio) / len(target)

    init = np.zeros(a.shape[1])
    opt = minimize(objective, init, method="BFGS", jac=True,
                   options={"gtol": 1e-8, "maxiter": 1000})
    gradient = float(np.max(np.abs(objective(opt.x)[1])))
    if not np.isfinite(opt.fun) or gradient > 1e-6:
        opt = minimize(objective, opt.x, method="L-BFGS-B", jac=True,
                       options={"gtol": 1e-9, "ftol": 1e-14, "maxiter": 2000})
        gradient = float(np.max(np.abs(objective(opt.x)[1])))
    if not np.isfinite(opt.fun) or gradient > 1e-5:
        raise RuntimeError(f"QLIKE fit did not converge: {opt.message}; max gradient {gradient}")
    return {"b_log": b_log, "smear": smear, "s2": s2,
            "mean_log": mean_log, "sd_log": sd_log,
            "mean_level": mean_level, "sd_level": sd_level,
            "b_level": b_level, "scale": scale, "b_qlike": opt.x,
            "qlike_gradient": gradient, "qlike_optimizer_success": bool(opt.success)}


def model_predictions(fit, xlog, xlevel, eps):
    z = np.r_[1.0, (xlog - fit["mean_log"]) / fit["sd_log"]]
    zl = np.r_[1.0, (xlevel - fit["mean_level"]) / fit["sd_level"]]
    base = float(z @ fit["b_log"])
    forecasts = {"HAR_raw": np.exp(base) - eps,
                 "HAR_ln": np.exp(base + fit["s2"] / 2) - eps,
                 "HAR_sm": np.exp(base) * fit["smear"] - eps,
                 "HAR_level_OLS": float(zl @ fit["b_level"]) * fit["scale"],
                 "HAR_QLIKE_loglink": np.exp(float(z @ fit["b_qlike"])) * fit["scale"]}
    if not all(np.isfinite(v) for v in forecasts.values()):
        raise FloatingPointError("Nonfinite forecast; do not silently change scored sample")
    return forecasts


def run(symbol):
    dates, y, eps, info = load(symbol)
    # Each row j has origin t and label t+1; all earlier rows have labels <= t.
    origins = np.arange(21, len(y) - 1)
    xlog = np.array([har_row(np.log(y + eps), t) for t in origins])
    xlevel = np.array([har_row(y, t) for t in origins])
    target = y[origins + 1]
    eligible = np.flatnonzero(dates[origins + 1] >= np.datetime64(TEST_START))
    if not len(eligible) or eligible[0] < MIN_TRAIN:
        raise ValueError(f"{symbol}: inadequate pre-test training")
    results, fit, fit_j = [], None, None
    for counter, j in enumerate(eligible):
        t = origins[j]
        refit = counter % REFIT_EVERY == 0
        if refit:
            fit = fit_models(xlog[:j], xlevel[:j], target[:j], eps)
            fit_j = j
        training_end = dates[origins[fit_j - 1] + 1]
        assert training_end <= dates[t] < dates[t + 1]
        assert len(y[t - 21:t + 1]) == 22
        raw = model_predictions(fit, xlog[j], xlevel[j], eps)
        raw.update({"RW": float(y[t]), "MA22": float(y[t - 21:t + 1].mean())})
        common = {"asset": symbol, "origin_date": pd.Timestamp(dates[t]),
                  "target_date": pd.Timestamp(dates[t + 1]), "origin_index": int(t),
                  "target_index": int(t + 1), "target": float(y[t + 1]),
                  "origin_target": float(y[t]), "epsilon": eps,
                  "ma22_window_start": pd.Timestamp(dates[t - 21]),
                  "ma22_window_end": pd.Timestamp(dates[t]), "ma22_n": 22,
                  "train_n": int(fit_j), "train_target_end": pd.Timestamp(training_end),
                  "refit": refit, "qlike_gradient": fit["qlike_gradient"],
                  "qlike_optimizer_success": fit["qlike_optimizer_success"]}
        baseline = max(eps, raw["MA22"])
        baseline_loss = float(qlike_series([y[t + 1]], [baseline])[0])
        for model in MODELS:
            forecast = max(eps, raw[model])
            loss = float(qlike_series([y[t + 1]], [forecast])[0])
            ratio = y[t + 1] / forecast
            traditional = ratio - np.log(ratio) - 1 if y[t + 1] > 0 else np.nan
            results.append({**common, "model": model, "raw_forecast": raw[model],
                            "forecast": forecast, "floor_applied": raw[model] < eps,
                            "loss_logf_y_over_f": loss, "loss_diff_vs_MA22": loss - baseline_loss,
                            "traditional_qlike_positive_only": traditional,
                            "squared_error": (y[t + 1] - forecast) ** 2})
    info.update({"test_origins": len(eligible), "test_start": str(pd.Timestamp(dates[origins[eligible[0]] + 1]).date()),
                 "test_end": str(pd.Timestamp(dates[origins[eligible[-1]] + 1]).date()),
                 "refits": int((len(eligible) - 1) // REFIT_EVERY + 1)})
    return pd.DataFrame(results), info


def summarize(predictions):
    records = []
    for asset, group in predictions.groupby("asset", sort=False):
        periods = [("ALL", group)] + [(str(year), group[group.target_date.dt.year == year])
                                      for year in (2024, 2025, 2026)]
        for period, frame in periods:
            if frame.empty:
                continue
            counts = frame.groupby("model").size()
            assert counts.nunique() == 1 and len(counts) == len(MODELS)
            for model in MODELS:
                g = frame[frame.model == model].sort_values("target_date")
                delta = g.loss_diff_vs_MA22.to_numpy()
                statistic = nw_t(-delta)
                p = math.erfc(abs(statistic) / np.sqrt(2)) if np.isfinite(statistic) else np.nan
                pos = g.target > 0
                records.append({"asset": asset, "period": period, "model": model,
                                "n": len(g), "n_zero_targets": int((~pos).sum()),
                                "n_positive_targets": int(pos.sum()),
                                "target_start": str(g.target_date.min().date()),
                                "target_end": str(g.target_date.max().date()),
                                "mean_loss_logf_y_over_f": float(g.loss_logf_y_over_f.mean()),
                                "loss_diff_vs_MA22_negative_better": float(delta.mean()),
                                "HAC_t_MA22_minus_model_positive_better": statistic,
                                "HAC_normal_p_two_sided": p, "HAC_lag": 5,
                                "MSE": float(g.squared_error.mean()),
                                "MAE": float(np.mean(np.abs(g.target - g.forecast))),
                                "mean_target": float(g.target.mean()),
                                "mean_forecast": float(g.forecast.mean()),
                                "traditional_QLIKE_common_positive_subset": float(g.loc[pos, "traditional_qlike_positive_only"].mean()),
                                "forecast_floor_count": int(g.floor_applied.sum()),
                                "epsilon_fixed_pre2024": float(g.epsilon.iloc[0]),
                                "RW_floor_dependent": model == "RW"})
    return pd.DataFrame(records)


def validate(predictions):
    """Meaningful invariants for the old off-by-one and sample selection defects."""
    fixture = np.arange(1., 51.)
    assert np.allclose(har_row(fixture, 30), [31, 29, 20.5])
    assert np.isfinite(qlike_series([0, 1], [0.1, 1])).all()
    assert nw_t(np.array([1., 2., 3., 2., 1., 2., 3., 2.]), lag=1) > 0
    for asset, frame in predictions.groupby("asset"):
        sizes = frame.groupby("target_date").size()
        assert (sizes == len(MODELS)).all()
        assert (frame.target_index == frame.origin_index + 1).all()
        assert (frame.train_target_end <= frame.origin_date).all()
        assert (frame.origin_date < frame.target_date).all()
        assert (frame.ma22_window_end == frame.origin_date).all()
        assert (frame.ma22_n == 22).all()
        assert np.isfinite(frame.forecast).all() and (frame.forecast > 0).all()
        assert (frame.groupby("target_date").target.nunique() == 1).all()
        ma = frame[frame.model == "MA22"]
        assert np.allclose(ma.loss_diff_vs_MA22, 0)
        dates, y, eps, _ = load(asset)
        for row in frame[frame.model == "RW"].itertuples():
            assert row.target == y[row.target_index]
            assert row.forecast == max(eps, y[row.origin_index])
        for row in ma.itertuples():
            assert row.forecast == max(eps, y[row.origin_index - 21:row.origin_index + 1].mean())
        expected = dates[22:][dates[22:] >= np.datetime64(TEST_START)]
        assert np.array_equal(ma.target_date.to_numpy(), expected)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    frames, sources = [], {}
    for symbol in SYMS:
        frame, info = run(symbol)
        frames.append(frame)
        sources[symbol] = info
        print(f"{symbol}: {info['test_origins']} origins, epsilon={info['epsilon']:.12g}, "
              f"refits={info['refits']}", flush=True)
    predictions = pd.concat(frames, ignore_index=True)
    validate(predictions)
    summary = summarize(predictions)
    pred_path = OUT / "CORRECTED_RANGE_HAR_PREDICTIONS.csv"
    sum_path = OUT / "CORRECTED_RANGE_HAR_SUMMARY.csv"
    predictions.to_csv(pred_path, index=False, encoding="utf-8-sig", float_format="%.17g")
    summary.to_csv(sum_path, index=False, encoding="utf-8-sig", float_format="%.17g")
    manifest = {"protocol": __doc__, "sources": sources,
                "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                "script": str(Path(__file__).resolve()),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (pred_path, sum_path)},
                "validation": "passed: native trading dates, target t+1, exact 22-day mean, shared samples, zero targets retained, HAC sign",
                "inference": "HAC normal approximation, lag 5; exploratory unadjusted comparisons; no post-test model selection",
                "forecast_floor": "fixed epsilon for every model; RW on zero current range is explicitly epsilon-dependent",
                "qlike_loglink_features": "1/5/22 averages of log(y+epsilon), same as log-OLS HAR; forecast E[y] directly",
                "target": "pure daily Parkinson including zero; no return-square substitution"}
    (OUT / "CORRECTED_RANGE_HAR_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    cols = ["asset", "period", "model", "n", "n_zero_targets",
            "loss_diff_vs_MA22_negative_better", "HAC_t_MA22_minus_model_positive_better", "forecast_floor_count"]
    print(summary[cols].to_string(index=False))
    print("Saved", sum_path, "and", pred_path)


if __name__ == "__main__":
    main()
