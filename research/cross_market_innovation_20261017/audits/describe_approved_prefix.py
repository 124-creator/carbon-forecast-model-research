"""Description only: approved 950 observations, no fitting and no later bytes."""
from pathlib import Path
import sys
import json
import math
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
FAIR = WORK / "carbon_workspace/research/recurrent_response_20261014/fair_recurrent_tuning_20261016"
sys.path.insert(0, str(FAIR))
from fair_data import load_prefix, support


def correlation(x, lag):
    x = np.asarray(x, dtype=float)
    if len(x) <= lag + 2 or np.std(x[:-lag]) == 0 or np.std(x[lag:]) == 0:
        return None
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def hac_mean_description(x, bandwidth):
    """Stationary mean diagnostic; never interpreted as neural training ESS."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    z = x - x.mean()
    variance = float(z @ z / n)
    bandwidth = min(int(bandwidth), n - 2)
    long_run = variance
    for lag in range(1, bandwidth + 1):
        gamma = float(z[:-lag] @ z[lag:] / n)
        long_run += 2 * (1 - lag / (bandwidth + 1)) * gamma
    return dict(n=n, Bartlett_bandwidth=bandwidth, variance=variance,
                long_run_variance=long_run, variance_inflation=long_run / variance if variance else None,
                equivalent_n_for_mean=n * variance / long_run if long_run > 0 else None,
                assumption="stationary weak dependence; descriptive estimate, not independent training examples")


def nonoverlap_geometry(origins, horizon):
    origins = np.asarray(origins)
    coverage = {}
    for origin in origins:
        for index in range(int(origin) + 1, int(origin) + horizon + 1):
            coverage[index] = coverage.get(index, 0) + 1
    n = len(origins)
    # Exact hypothetical variance factors under iid zero-mean increments.
    factor_mean = 1 + 2 * sum((1 - lag / n) * (1 - lag / horizon)
                              for lag in range(1, min(horizon, n)))
    factor_squared = 1 + 2 * sum((1 - lag / n) * (1 - lag / horizon) ** 2
                                 for lag in range(1, min(horizon, n)))
    return dict(origins=n, horizon=horizon, cumulative_target_cells=n * horizon,
                distinct_future_observation_indices=len(coverage),
                average_target_index_reuse=float(np.mean(list(coverage.values()))),
                maximum_target_index_reuse=max(coverage.values()),
                maximum_disjoint_horizon_windows=len(origins[::horizon]),
                adjacent_target_window_overlap=(horizon - 1) / horizon,
                iid_increment_equivalent_n_for_target_mean=n / factor_mean,
                iid_gaussian_increment_equivalent_n_for_squared_target_mean=n / factor_squared,
                caveat="geometry and hypothetical iid calculation; not estimated number of independent neural samples")


def describe():
    price, dates, manifest = load_prefix()
    delta = np.diff(price)
    logreturn = np.diff(np.log(price))
    yearly = []
    for year in ("2018", "2019", "2020", "2021"):
        index = np.flatnonzero(np.char.startswith(dates, year))
        target_positions = index[index > 0]
        change = price[target_positions] - price[target_positions - 1]
        logchange = np.log(price[target_positions] / price[target_positions - 1])
        day_gaps = np.diff(dates[index].astype("datetime64[D]")).astype(int)
        yearly.append(dict(year=year, observations=len(index), first_date=str(dates[index[0]]),
            last_date=str(dates[index[-1]]), mean_price=float(price[index].mean()),
            minimum_price=float(price[index].min()), maximum_price=float(price[index].max()),
            return_observations=len(change), zero_price_change_count=int(np.count_nonzero(change == 0)),
            zero_price_change_fraction=float(np.mean(change == 0)),
            mean_price_change=float(change.mean()), std_price_change=float(change.std(ddof=1)),
            mean_log_return=float(logchange.mean()), std_log_return=float(logchange.std(ddof=1)),
            price_change_acf={str(lag):correlation(change, lag) for lag in (1, 2, 5, 10, 20)},
            mean_calendar_gap=float(day_gaps.mean()), maximum_calendar_gap=int(day_gaps.max()),
            median_absolute_change=float(np.median(np.abs(change))),
            p95_absolute_change=float(np.quantile(np.abs(change), .95)),
            maximum_absolute_change=float(np.max(np.abs(change)))))
    folds = []
    for cut in (459, 703):
        ids, train, valid, scale, target = support(price[:cut], cut)
        fold = dict(cut=cut, train_first_origin=int(ids[train[0]]),
                    train_last_origin=int(ids[train[-1]]), train_n=len(train),
                    validation_n=len(valid), train_geometry=nonoverlap_geometry(ids[train], 20),
                    validation_geometry=nonoverlap_geometry(ids[valid], 20),
                    train_increment_scale=scale, targets=[])
        for h in (1, 5, 20):
            values = target[train, h - 1]
            fold["targets"].append(dict(horizon=h, mean=float(values.mean()), sd=float(values.std(ddof=1)),
                acf={str(lag):correlation(values,lag) for lag in (1, 2, 5, 10, 20)},
                squared_target_acf={str(lag):correlation(values**2,lag) for lag in (1, 2, 5, 10, 20)},
                mean_hac20=hac_mean_description(values,20), mean_hac60=hac_mean_description(values,60),
                squared_mean_hac20=hac_mean_description(values**2,20),
                squared_mean_hac60=hac_mean_description(values**2,60)))
        folds.append(fold)
    results = dict(scope="950 approved observations only; description, no model fitting or new data access",
                   manifest=manifest, software=dict(numpy=np.__version__),
                   market_fits=0, additional_market_online_updates=0,
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   yearly=yearly, folds=folds,
                   full_prefix_price_change_acf={str(lag):correlation(delta, lag) for lag in (1, 2, 5, 10, 20)},
                   full_prefix_logreturn_acf={str(lag):correlation(logreturn, lag) for lag in (1, 2, 5, 10, 20)})
    path = ROOT / "APPROVED_PREFIX_DESCRIPTION.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(dict(yearly=yearly, folds=[dict(cut=r["cut"], train_n=r["train_n"],
                      geometry=r["train_geometry"], h20=r["targets"][-1]) for r in folds]),ensure_ascii=False))


if __name__ == "__main__":
    describe()
