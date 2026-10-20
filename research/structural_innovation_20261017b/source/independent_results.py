"""Independent development metrics, selection and calendar-block uncertainty.

Uses only standard library and NumPy. Never imports production data/metrics,
training models, or the root selector. The audit CLI accepts D1/D2 only and
reads completed development receipts/forecast NPZs, never original price CSVs
or confirmation directories. Pure functions support synthetic verification.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

H = 20
SCORE_INDEX = np.array([0, 4, 19], dtype=np.int64)
NEURAL_FAMILIES = (
    "smp_full", "smp_no_clock", "smp_no_sign", "smp_pooled_mark",
    "smp_iid_state", "smp_one_step", "smp_no_aux", "gru_native",
    "gru_normalized", "gru_aux", "lstm_native", "patchtst_native",
)
BASELINE_FAMILIES = (
    "gru_native", "gru_normalized", "gru_aux", "lstm_native",
    "patchtst_native", "ridge", "ridge_val", "markov_mle",
)
STATISTICAL_FAMILIES = ("markov_mle", "project_reference", "rw")
SHRINK = (0.0, 0.1)
RIDGE = (0.01, 0.1, 1.0, 10.0, 100.0)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _matrix(value: Any, name: str) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    require(a.ndim == 2 and a.shape[0] > 0 and a.shape[1] == H,
            name + " must have shape (N>0,20)")
    require(np.isfinite(a).all(), name + " is non-finite")
    return a


def recompute_metrics(predicted_change: Any, target_change: Any) -> dict:
    predicted = _matrix(predicted_change, "predicted_change")
    target = _matrix(target_change, "target_change")
    require(predicted.shape == target.shape, "forecast/target shape mismatch")
    error = predicted - target
    squared_error = np.square(error)
    squared_rw = np.square(target)
    mse = squared_error.sum(axis=0) / len(target)
    rw = squared_rw.sum(axis=0) / len(target)
    require(np.isfinite(mse).all() and np.isfinite(rw).all(), "overflowed metrics")
    require(np.all(rw > 0), "nonpositive evaluation RW denominator")
    relative = mse / rw
    return {
        "n": len(target),
        "mse": mse.tolist(),
        "rw_mse": rw.tolist(),
        "relative_mse": relative.tolist(),
        "mae": (np.abs(error).sum(axis=0) / len(target)).tolist(),
        "score": float(relative[SCORE_INDEX].sum() / 3),
        "score_horizons": [1, 5, 20],
    }


def mean_seed_squared_loss(predictions: Any, target: Any) -> np.ndarray:
    """Average risks over seeds, never square an ensemble-mean forecast error."""
    target = _matrix(target, "target")
    predictions = np.asarray(predictions, dtype=np.float64)
    require(predictions.ndim == 3 and predictions.shape[0] > 0
            and predictions.shape[1:] == target.shape, "seed forecast shape mismatch")
    require(np.isfinite(predictions).all(), "non-finite seed predictions")
    return np.mean(np.square(predictions - target[None, :, :]), axis=0)


def _value_key(value: Any) -> float | None:
    return None if value is None else float(value)


def _config(family: str, value: Any) -> str:
    return family if value is None else family + "__a" + str(float(value)).replace(".", "p")


def _markets(protocol: dict) -> list[str]:
    markets = [x["name"] for x in protocol["markets"]]
    require(len(markets) == 5 and len(set(markets)) == 5, "exactly five series required")
    require(all(re.fullmatch(r"[A-Za-z0-9_.-]+", x) is not None
                and ".." not in x for x in markets), "unsafe series name")
    return markets


def _summary(rows: list[dict], markets: list[str], family: str,
             value: Any, seeds: list[int]) -> dict:
    indexed = {}
    for row in rows:
        key = (row["market"], int(row["seed"]))
        require(key not in indexed, "duplicate series/seed in a family summary")
        indexed[key] = row
    require(set(indexed) == {(m, s) for m in markets for s in seeds},
            "incomplete or unmatched series/seed summary")
    risks = np.array([[indexed[(m, s)]["relative_mse"] for s in seeds] for m in markets],
                     dtype=np.float64)
    require(risks.shape == (len(markets), len(seeds), H)
            and np.isfinite(risks).all(), "invalid relative-risk summary")
    cells = risks.mean(axis=1)
    return {
        "family": family, "value": _value_key(value),
        "score": float(cells[:, SCORE_INDEX].mean()),
        "by_market": {
            m: {"score": float(cells[i, SCORE_INDEX].mean()),
                "relative_mse": cells[i].tolist()}
            for i, m in enumerate(markets)
        },
        "by_seed": {
            str(s): float(risks[:, i, :][:, SCORE_INDEX].mean())
            for i, s in enumerate(seeds)
        },
        "seeds": seeds,
        "job_ids": [indexed[(m, s)]["id"] for m in markets for s in seeds],
    }


def _family(rows: list[dict], markets: list[str], family: str,
            value: Any, seeds: list[int]) -> dict:
    wanted = _value_key(value)
    chosen = [r for r in rows if r["family"] == family
              and _value_key(r["value"]) == wanted]
    return _summary(chosen, markets, family, wanted, seeds)


def _ridge_validation(rows: list[dict], markets: list[str]) -> dict:
    chosen, decisions = [], {}
    for market in markets:
        available = [r for r in rows if r["market"] == market and r["family"] == "ridge"]
        require(len(available) == 5 and {float(r["value"]) for r in available} == set(RIDGE),
                "ridge validation candidates incomplete")
        require(all(r["selected_validation_score"] is not None
                    and math.isfinite(float(r["selected_validation_score"])) for r in available),
                "missing finite pre-year ridge validation score")
        winner = min(available,
                     key=lambda r: (float(r["selected_validation_score"]), -float(r["value"])))
        chosen.append(winner)
        decisions[market] = {
            "value": float(winner["value"]),
            "validation_score": float(winner["selected_validation_score"]),
            "job_id": winner["id"],
        }
    result = _summary(chosen, markets, "ridge_val", None, [0])
    result["pre_year_validation_choices"] = decisions
    return result


def select_d1(rows: list[dict], protocol: dict) -> dict:
    """Pure D1 selection: exact score ties prefer the larger numerical value."""
    markets = _markets(protocol)
    if "neural_families" in protocol:
        require(tuple(protocol["neural_families"]) == NEURAL_FAMILIES,
                "neural family list changed")
    if "primary_baseline_families" in protocol:
        require(tuple(protocol["primary_baseline_families"]) == BASELINE_FAMILIES,
                "primary baseline family list changed")
    seeds = list(map(int, protocol["D1_seeds"]))
    require(seeds == [17], "D1 seed differs from the fixed single-seed scope")
    selected, candidates = {}, []
    for family in NEURAL_FAMILIES:
        options = [_family(rows, markets, family, value, seeds) for value in SHRINK]
        candidates.extend(options)
        selected[family] = min(options, key=lambda r: (r["score"], -r["value"]))
    options = [_family(rows, markets, "ridge", alpha, [0]) for alpha in RIDGE]
    candidates.extend(options)
    selected["ridge"] = min(options, key=lambda r: (r["score"], -r["value"]))
    selected["ridge_val"] = _ridge_validation(rows, markets)
    candidates.append(selected["ridge_val"])
    for family in STATISTICAL_FAMILIES:
        selected[family] = _family(rows, markets, family, None, [0])
        candidates.append(selected[family])
    best = min((selected[f] for f in BASELINE_FAMILIES),
               key=lambda r: (r["score"], r["family"]))
    full = selected["smp_full"]
    matched = {
        family: _family(rows, markets, family, full["value"], seeds)
        for family in NEURAL_FAMILIES
    }
    gates = {
        "full_below_rw": full["score"] < 1.0,
        "full_below_same_alpha_iid": full["score"] < matched["smp_iid_state"]["score"],
        "full_below_same_alpha_one_step": full["score"] < matched["smp_one_step"]["score"],
        "full_within_2pct_of_best_baseline": full["score"] <= 1.02 * best["score"],
    }
    return {
        "phase": "D1", "selection_year": 2022,
        "selected": selected, "candidate_summaries": candidates,
        "best_baseline": best, "matched_smp_alpha": matched,
        "gate_values": {
            "full": full["score"], "same_alpha_iid": matched["smp_iid_state"]["score"],
            "same_alpha_one_step": matched["smp_one_step"]["score"],
            "best_baseline": best["score"], "best_baseline_times_1p02": 1.02 * best["score"],
        },
        "gates": gates, "advance_to_D2": all(gates.values()),
        "tie_rules": "Exact score equality: larger value within family; alphabetical baseline family.",
        "interpretation": "Resource screen only. No bootstrap, p-value, confirmation or publication claim.",
    }


def evaluate_d2(rows: list[dict], protocol: dict, d1: dict) -> dict:
    """Evaluate fixed D1 choices; never choose a neural value from D2 risk."""
    require(d1["advance_to_D2"], "D1 did not authorize D2")
    markets = _markets(protocol)
    seeds = list(map(int, protocol["D2_seeds"]))
    require(len(seeds) == 3 and len(set(seeds)) == 3, "D2 requires three fixed seeds")
    selected = {}
    for family in NEURAL_FAMILIES:
        value = d1["selected"][family]["value"]
        selected[family] = _family(rows, markets, family, value, seeds)
    selected["ridge"] = _family(rows, markets, "ridge",
                                d1["selected"]["ridge"]["value"], [0])
    selected["ridge_val"] = _ridge_validation(rows, markets)
    for family in STATISTICAL_FAMILIES:
        selected[family] = _family(rows, markets, family, None, [0])
    full = selected["smp_full"]
    matched = {family: _family(rows, markets, family, full["value"], seeds)
               for family in NEURAL_FAMILIES}
    best_2023 = min((selected[f] for f in BASELINE_FAMILIES),
                    key=lambda r: (r["score"], r["family"]))
    eex = [m for m in markets if m.upper().startswith("EEX")]
    require(len(eex) == 1, "exactly one explicitly named EEX series required")
    market_wins = sum(full["by_market"][m]["score"] < 1 for m in markets)
    seed_wins = sum(score < 1 for score in full["by_seed"].values())
    gates = {
        "full_at_least_1pct_better_than_rw": full["score"] <= 0.99,
        "full_below_all_eight_fixed_baseline_families": full["score"] < best_2023["score"],
        "eex_below_rw": full["by_market"][eex[0]]["score"] < 1.0,
        "full_below_same_alpha_iid": full["score"] < matched["smp_iid_state"]["score"],
        "full_below_same_alpha_one_step": full["score"] < matched["smp_one_step"]["score"],
        "at_least_three_series_below_rw": market_wins >= 3,
        "at_least_two_seeds_below_rw": seed_wins >= 2,
    }
    fixed_best = d1["best_baseline"]["family"]
    return {
        "phase": "D2", "evaluation_year": 2023,
        "selected": selected, "matched_smp_alpha": matched,
        "lowest_2023_score_among_eight_fixed_families": best_2023,
        "d1_fixed_strongest_baseline_family": fixed_best,
        "d1_fixed_strongest_baseline_2023": selected[fixed_best],
        "gate_values": {
            "full": full["score"], "lowest_fixed_baseline": best_2023["score"],
            "same_alpha_iid": matched["smp_iid_state"]["score"],
            "same_alpha_one_step": matched["smp_one_step"]["score"],
            "eex": full["by_market"][eex[0]]["score"],
            "series_below_rw": market_wins, "seeds_below_rw": seed_wins,
        },
        "gates": gates, "point_gate_pass": all(gates.values()),
        "seed_aggregation": "Mean squared losses/risks across seeds, not an ensemble forecast.",
        "confirmation_opened": False,
        "interpretation": "Development gate only; the main paired intervals must also pass before a separate confirmation freeze.",
    }


def calendar_day_counts(n_days: int, block_days: int, rng: Any) -> np.ndarray:
    """Circular moving blocks on the complete natural-year calendar."""
    require(n_days > 0 and 1 <= block_days <= n_days, "invalid calendar/block size")
    blocks = math.ceil(n_days / block_days)
    starts = rng.integers(0, n_days, size=blocks)
    offsets = np.arange(block_days)
    days = ((starts[:, None] + offsets[None, :]) % n_days).reshape(-1)[:n_days]
    return np.bincount(days, minlength=n_days).astype(np.float64)


def paired_calendar_bootstrap(
    series: dict[str, dict], *, year: int, block_days: int = 28,
    repetitions: int = 2000, seed: int = 314159, max_attempts: int = 10000,
    return_samples: bool = False,
) -> dict:
    """Paired 15-cell score differences from pre-averaged seed squared losses.

    Every series receives the SAME sampled calendar-day multiplicities. Each
    replicate recomputes both the model MSE and its own RW MSE denominator.
    Invalid replicates count toward max_attempts and are explicitly reported.
    """
    require(len(series) == 5, "bootstrap requires exactly five named series")
    require(0 < repetitions <= max_attempts, "invalid replicate/attempt budget")
    start = np.datetime64(f"{year}-01-01", "D")
    end = np.datetime64(f"{year+1}-01-01", "D")
    n_days = int((end - start).astype(int))
    names = sorted(series)
    accum_a = np.zeros((5, n_days, 3), dtype=np.float64)
    accum_b = np.zeros_like(accum_a)
    accum_rw = np.zeros_like(accum_a)
    counts = np.zeros((5, n_days), dtype=np.float64)
    for k, name in enumerate(names):
        item = series[name]
        dates = np.asarray(item["date"], dtype="datetime64[D]")
        require(dates.ndim == 1 and len(dates) > 0 and not np.isnat(dates).any(),
                "invalid bootstrap dates")
        index = (dates - start).astype(np.int64)
        require(np.all((index >= 0) & (index < n_days)), "origin outside declared calendar year")
        losses = []
        for label in ("loss_a", "loss_b", "rw_loss"):
            loss = _matrix(item[label], label)
            require(loss.shape[0] == len(dates) and np.all(loss >= 0),
                    "invalid date/loss alignment")
            losses.append(loss[:, SCORE_INDEX])
        for target, loss in zip((accum_a, accum_b, accum_rw), losses):
            np.add.at(target[k], index, loss)
        np.add.at(counts[k], index, 1.0)

    def scores(weights):
        origin_weight = counts @ weights
        if np.any(origin_weight <= 0):
            return None
        total_a = np.einsum("mdh,d->mh", accum_a, weights)
        total_b = np.einsum("mdh,d->mh", accum_b, weights)
        total_rw = np.einsum("mdh,d->mh", accum_rw, weights)
        mse_a = total_a / origin_weight[:, None]
        mse_b = total_b / origin_weight[:, None]
        rw = total_rw / origin_weight[:, None]
        if np.any(rw <= 0) or not all(np.isfinite(x).all() for x in (mse_a, mse_b, rw)):
            return None
        return float(np.mean(mse_a / rw)), float(np.mean(mse_b / rw))

    point = scores(np.ones(n_days))
    require(point is not None, "undefined point estimate RW denominator")
    rng = np.random.default_rng(seed)
    differences, score_a, score_b = [], [], []
    attempts = invalid = 0
    while len(differences) < repetitions and attempts < max_attempts:
        attempts += 1
        estimate = scores(calendar_day_counts(n_days, block_days, rng))
        if estimate is None:
            invalid += 1
            continue
        a, b = estimate
        score_a.append(a)
        score_b.append(b)
        differences.append(a - b)
    sufficient = len(differences) == repetitions
    interval = (
        np.quantile(differences, [0.025, 0.975], method="linear").tolist()
        if sufficient else None
    )
    result = {
        "status": "completed" if sufficient else "insufficient_valid_replicates",
        "year": int(year), "calendar_start": str(start), "calendar_end_exclusive": str(end),
        "calendar_days": n_days, "block_days": block_days,
        "requested_repetitions": repetitions, "valid_repetitions": len(differences),
        "attempts": attempts, "invalid_repetitions": invalid, "max_attempts": max_attempts,
        "seed": seed, "point_score_a": point[0], "point_score_b": point[1],
        "point_difference_a_minus_b": point[0] - point[1],
        "percentile_95_interval": interval,
        "upper_below_zero": bool(sufficient and interval[1] < 0),
        "horizons": [1, 5, 20], "equal_cells": 15,
        "common_calendar_weights_across_series": True,
        "rw_denominator_recomputed_each_rep": True,
        "seed_risks_averaged_before_resampling": True,
        "p_value_computed": False,
    }
    if return_samples:
        result["differences"] = differences
        result["score_a_samples"] = score_a
        result["score_b_samples"] = score_b
    return result


def expected_jobs(protocol: dict, phase: str, d1: dict | None = None) -> list[dict]:
    require(phase in ("D1", "D2"), "only development phases D1/D2 are allowed")
    markets = _markets(protocol)
    require(tuple(map(float, protocol["shrink_candidates"])) == SHRINK,
            "shrink candidates changed")
    require(tuple(map(float, protocol["ridge_alpha_candidates"])) == RIDGE,
            "ridge candidates changed")
    if phase == "D2":
        require(d1 is not None and d1["advance_to_D2"], "eligible D1 selection required")
    jobs = []
    for market in markets:
        for family in NEURAL_FAMILIES:
            values = SHRINK if phase == "D1" else sorted({
                float(d1["selected"][family]["value"]),
                float(d1["selected"]["smp_full"]["value"]),
            })
            for value in values:
                for seed in protocol[phase + "_seeds"]:
                    jobs.append(dict(market=market, family=family, kind="neural",
                                     value=float(value), seed=int(seed)))
        for value in RIDGE:
            jobs.append(dict(market=market, family="ridge", kind="ridge",
                             value=value, seed=0))
        for family in STATISTICAL_FAMILIES:
            jobs.append(dict(market=market, family=family, kind=family, value=None, seed=0))
    for job in jobs:
        job["config_id"] = _config(job["family"], job["value"])
        job["id"] = f"{job['market']}__{job['config_id']}__s{job['seed']}"
        job["year"] = 2022 if phase == "D1" else 2023
    return jobs


def audit_stage(root: Path, protocol_path: Path, phase: str,
                d1: dict | None = None) -> tuple[list[dict], dict, dict]:
    """Read only complete authorized D1/D2 outputs; no source-price reads."""
    protocol = json.loads(protocol_path.read_text())
    protocol_sha = sha(protocol_path)
    jobs = expected_jobs(protocol, phase, d1)
    admitted_source_hashes = {m["name"]: m["sha256"] for m in protocol["markets"]}
    expected_ids = {j["id"] for j in jobs}
    stage = root / "results" / phase
    complete = json.loads((stage / "COMPLETED.json").read_text())
    require(complete["failed_jobs"] == 0 and complete["completed_jobs"] == len(jobs)
            and complete["attempted_jobs"] == len(jobs), "stage not fully complete")
    require({j["id"] for j in complete["jobs"]} == expected_ids
            and all(j["status"] == "completed" for j in complete["jobs"]),
            "completed job scope mismatch")
    require(complete["protocol_sha256"] == protocol_sha, "completion protocol mismatch")
    start = json.loads((stage / "STARTED.json").read_text())
    require(set(start["job_ids"]) == expected_ids and start["total_jobs"] == len(jobs),
            "started job scope mismatch")
    require(start["protocol_sha256"] == protocol_sha, "start protocol mismatch")
    kind_counts = dict(Counter(j["kind"] for j in jobs))
    require(start["job_counts"] == kind_counts and complete["actual_counts"] == kind_counts,
            "job-kind counts mismatch")

    rows, arrays, common, support_common = [], {}, {}, {}
    scalar_comparisons = 0
    metric_max_error = 0.0
    artifact_hashes = []
    for job in jobs:
        public = stage / "jobs" / job["id"]
        private = root / "local_only" / phase / job["id"] / "forecasts.npz"
        record = json.loads((public / "METRICS.json").read_text())
        require(record["status"] == "completed" and record["phase"] == phase,
                "metric completion/phase mismatch")
        for key in ("id", "config_id", "family", "kind", "seed", "year", "value", "market"):
            require(record[key] == job[key], "metric job identity mismatch: " + key)
        require(record["protocol_sha256"] == protocol_sha, "metric protocol mismatch")
        forecast_hash = sha(private)
        require(forecast_hash == record["private_forecast_sha256"], "forecast byte hash mismatch")
        started = json.loads((public / "STARTED.json").read_text())
        require(started["market"] == job["market"]
                and started["protocol_sha256"] == protocol_sha, "job start identity mismatch")
        for key in ("id", "config_id", "family", "kind", "seed", "year", "value"):
            require(started["job"][key] == job[key], "started job identity mismatch")
        if job["kind"] in ("neural", "ridge", "markov_mle"):
            fit = json.loads((public / "FIT.json").read_text())
            require(fit["eligible_for_selection"] is True, "ineligible fit")
            require(fit["selected_validation_score"] == record["selected_validation_score"],
                    "fit/metric validation score mismatch")
        with np.load(private, allow_pickle=False) as archive:
            require(set(archive.files) == {
                "date", "origin", "base", "scale", "state", "predicted_change", "target_change"
            }, "unexpected forecast NPZ schema")
            data = {key: archive[key].copy() for key in archive.files}
        n = len(data["origin"])
        origin = data["origin"]
        dates = np.asarray(data["date"], dtype="datetime64[D]")
        require(n > 0 and origin.shape == (n,) and np.issubdtype(origin.dtype, np.integer),
                "invalid origin vector")
        require(np.array_equal(origin, np.arange(int(origin[0]), int(origin[-1]) + 1)),
                "evaluation origin support is not contiguous")
        require(dates.shape == (n,) and not np.isnat(dates).any()
                and np.all(np.diff(dates.astype(np.int64)) > 0), "invalid date ordering")
        require(np.all((dates >= np.datetime64(f"{job['year']}-01-01"))
                       & (dates < np.datetime64(f"{job['year']+1}-01-01"))),
                "evaluation dates escape declared year")
        for key in ("base", "scale", "state"):
            require(data[key].shape == (n,) and np.isfinite(data[key]).all(),
                    "invalid common input vector: " + key)
        require(np.all(data["scale"] > 0) and np.all(data["base"] > 0), "invalid scale/base")
        require(np.all((data["state"] >= 0) & (data["state"] < 9))
                and np.all(data["state"] == np.floor(data["state"])), "invalid observed state")
        computed = recompute_metrics(data["predicted_change"], data["target_change"])
        require(computed["n"] == n and record["n"] == n
                and record["score_horizons"] == [1, 5, 20],
                "metric count/horizon mismatch")
        for field in ("mse", "rw_mse", "relative_mse", "mae", "score"):
            actual = np.asarray(computed[field])
            reported = np.asarray(record[field])
            require(actual.shape == reported.shape, "metric shape mismatch")
            require(np.allclose(actual, reported, atol=1e-11, rtol=1e-10),
                    "independent metric mismatch: " + field)
            metric_max_error = max(metric_max_error, float(np.max(np.abs(actual-reported))))
            scalar_comparisons += actual.size
        if job["family"] == "rw":
            require(np.all(data["predicted_change"] == 0), "RW forecast is not zero change")
        support = json.loads((public / "SUPPORT.json").read_text())
        require(support["market"] == job["market"] and support["year"] == job["year"],
                "support identity mismatch")
        require(support["source_sha256"] == admitted_source_hashes[job["market"]],
                "support source identity differs from admitted protocol metadata")
        require(support["evaluation_n"] == n
                and support["evaluation_first_origin"] == int(origin[0])
                and support["evaluation_last_origin"] == int(origin[-1])
                and support["evaluation_last_label"] == int(origin[-1]) + H,
                "support evaluation bounds mismatch")
        require(support["evaluation_first_date"] == str(dates[0])
                and support["evaluation_last_date"] == str(dates[-1]),
                "support date bounds mismatch")
        require(support["train_first_origin"] == 60
                and support["train_last_origin"] + H == support["max_train_label"]
                and support["max_train_label"] == support["first_validation_origin"] - 1
                and support["train_n"] == support["train_last_origin"] - 59,
                "training maturity/purge mismatch")
        require(support["validation_n"] == protocol["validation_size"]
                and support["validation_n"] == (support["last_validation_origin"]
                                                 - support["first_validation_origin"] + 1)
                and support["last_validation_origin"] + H == support["max_validation_label"]
                and support["max_validation_label"] == int(origin[0]) - 1,
                "pre-year validation maturity mismatch")
        require(support["statistics_price_prefix_end_exclusive"]
                == support["first_validation_origin"], "training statistic boundary mismatch")
        require(support["evaluation_last_label"] < support["source_rows"],
                "evaluation target exceeds admitted source")
        require(support["current_state_from_full_observed_prefix"] is True
                and support["future_context_frozen"] is True
                and support["quote_imputation"] is False, "support behavior flags mismatch")
        market = job["market"]
        if market in common:
            for key in ("date", "origin", "base", "scale", "state", "target_change"):
                require(np.array_equal(data[key], common[market][key]),
                        "unequal common support/input/target arrays: " + key)
                scalar_comparisons += data[key].size
            require(support == support_common[market], "family/seed support receipts differ")
        else:
            common[market] = {key: data[key] for key in (
                "date", "origin", "base", "scale", "state", "target_change")}
            support_common[market] = support
            # Target overlap where future origins are themselves saved is an
            # additional internal coherence check; it does not replace a source audit.
            for h in range(1, H+1):
                if n > h:
                    recovered = data["base"][:-h] + data["target_change"][:-h, h-1]
                    require(np.allclose(recovered, data["base"][h:], atol=1e-9, rtol=1e-12),
                            "overlapping target/base inconsistency")
                    scalar_comparisons += n-h
        row = {**record, **computed}
        rows.append(row)
        arrays[job["id"]] = data
        artifact_hashes.append({
            "id": job["id"], "forecast_sha256": forecast_hash,
            "metrics_sha256": sha(public / "METRICS.json"),
            "support_sha256": sha(public / "SUPPORT.json"),
        })
    receipt = {
        "created_utc": utc(), "status": "PASS", "phase": phase,
        "protocol_sha256": protocol_sha, "auditor_sha256": sha(Path(__file__)),
        "jobs": len(jobs), "job_counts": kind_counts, "horizon_metric_cells": len(jobs)*H,
        "scalar_comparisons": int(scalar_comparisons),
        "max_metric_abs_error": metric_max_error,
        "common_support_checked_across_all_configurations_and_seeds": True,
        "source_price_files_read": 0, "confirmation_files_read": 0, "market_fits": 0,
        "original_source_truth_independently_reloaded": False,
        "scope": "Independent forecast-file metrics, identities, common support and internal overlapping-label coherence; no claim to re-audit all training weights.",
        "artifact_hashes": artifact_hashes,
    }
    return rows, arrays, receipt


def _loss_inputs(summary_a: dict, summary_b: dict, arrays: dict, markets: list[str]) -> dict:
    result = {}
    for market in markets:
        ids_a = [job for job in summary_a["job_ids"] if job.startswith(market + "__")]
        ids_b = [job for job in summary_b["job_ids"] if job.startswith(market + "__")]
        require(ids_a and ids_b, "missing paired series forecast groups")
        reference = arrays[ids_a[0]]
        for job in ids_a + ids_b:
            for key in ("date", "origin", "target_change"):
                require(np.array_equal(arrays[job][key], reference[key]),
                        "unmatched paired bootstrap forecast support")
        truth = reference["target_change"]
        result[market] = {
            "date": reference["date"],
            "loss_a": mean_seed_squared_loss(
                np.stack([arrays[j]["predicted_change"] for j in ids_a]), truth),
            "loss_b": mean_seed_squared_loss(
                np.stack([arrays[j]["predicted_change"] for j in ids_b]), truth),
            "rw_loss": np.square(truth),
        }
    return result


def d2_uncertainty(d2: dict, arrays: dict, protocol: dict) -> dict:
    if not d2["point_gate_pass"]:
        return {
            "status": "not_run_point_gate_failed", "bootstrap_repetitions_executed": 0,
            "eligible_for_separate_confirmation_freeze": False,
            "confirmation_opened": False,
        }
    markets = _markets(protocol)
    full = d2["selected"]["smp_full"]
    comparators = {
        "rw": d2["selected"]["rw"],
        "d1_fixed_strongest_baseline": d2["d1_fixed_strongest_baseline_2023"],
    }
    comparisons = {}
    for name, comparator in comparators.items():
        loss_inputs = _loss_inputs(full, comparator, arrays, markets)
        comparisons[name] = {
            str(length): paired_calendar_bootstrap(
                loss_inputs, year=2023, block_days=length,
                repetitions=2000, seed=314159, max_attempts=10000)
            for length in (28, 14, 56)
        }
    main_pass = all(comparisons[name]["28"]["upper_below_zero"] for name in comparators)
    sensitivity_pass = {
        str(length): all(comparisons[name][str(length)]["upper_below_zero"]
                         for name in comparators)
        for length in (14, 56)
    }
    return {
        "status": "completed", "comparisons": comparisons,
        "primary_block_days": 28, "primary_two_intervals_upper_below_zero": main_pass,
        "sensitivity_joint_pass": sensitivity_pass,
        "sensitivity_disagreement_with_primary": any(
            value != main_pass for value in sensitivity_pass.values()),
        "eligible_for_separate_confirmation_freeze": main_pass,
        "confirmation_opened": False,
        "interpretation": "Only the prespecified 28-day pair of intervals gates resources; 14/56 are all reported and cannot replace it. Development uncertainty is not a confirmation p-value.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--phase", choices=("D1", "D2"), required=True)
    parser.add_argument("--d1-selection", type=Path,
                        help="This auditor's frozen independent D1 selection, required for D2.")
    parser.add_argument("--output", type=Path, required=True,
                        help="New public aggregate-only audit directory.")
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    root = protocol_path.parent.parent
    require(not args.output.exists(), "refuse overwriting an independent audit")
    args.output.mkdir(parents=True)
    write_json(args.output / "STARTED.json", {
        "created_utc": utc(), "phase": args.phase,
        "protocol_sha256": sha(protocol_path), "auditor_sha256": sha(Path(__file__)),
        "source_price_or_confirmation_read_planned": False,
    })
    try:
        protocol = json.loads(protocol_path.read_text())
        d1 = json.loads(args.d1_selection.read_text()) if args.d1_selection else None
        if args.phase == "D2":
            require(d1 is not None and d1["protocol_sha256"] == sha(protocol_path),
                    "independent D1 selection protocol mismatch")
        rows, arrays, audit = audit_stage(root, protocol_path, args.phase, d1)
        selection = select_d1(rows, protocol) if args.phase == "D1" else evaluate_d2(rows, protocol, d1)
        selection["protocol_sha256"] = sha(protocol_path)
        selection["auditor_sha256"] = sha(Path(__file__))
        selection["created_utc"] = utc()
        write_json(args.output / "METRIC_AUDIT.json", audit)
        write_json(args.output / "RECOMPUTED_METRICS.json", rows)
        write_json(args.output / "SELECTION.json", selection)
        if args.phase == "D2":
            uncertainty = d2_uncertainty(selection, arrays, protocol)
            write_json(args.output / "PAIRED_UNCERTAINTY.json", uncertainty)
        print(json.dumps({
            "phase": args.phase, "audit_status": audit["status"], "jobs": audit["jobs"],
            "horizon_metric_cells": audit["horizon_metric_cells"],
            "advance": selection.get("advance_to_D2", selection.get("point_gate_pass")),
        }, ensure_ascii=False))
    except Exception as exc:
        write_json(args.output / "FAILED.json", {
            "created_utc": utc(), "error_type": type(exc).__name__, "error": str(exc),
            "confirmation_opened": False,
        })
        raise


if __name__ == "__main__":
    main()

