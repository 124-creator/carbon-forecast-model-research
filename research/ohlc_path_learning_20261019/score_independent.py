#!/usr/bin/env python3
"""Independent, frozen scorer for the reported-price/range experiment.

This module deliberately imports neither pipeline.py, run.py, nor model.py.
It reads LOCAL observations and forecasts, verifies their receipts, reconstructs
labels/scales, and publishes only aggregate CSV/JSON. No per-origin predictions,
observations, losses, samples, or bootstrap draws are written to disk.

Run after ALL 156 forecast files and six fold manifests are complete:
    python score_independent.py
    python score_independent.py --self-test

Primary estimates average analytical/deterministic means over the three neural
seeds before evaluating MSE. Finite-ensemble energy scores instead evaluate each
seed's 128-member distribution separately, then average scores over seeds.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_thread_variable, "1")

import numpy as np
from scipy.spatial.distance import pdist


ROOT = Path(__file__).resolve().parent
HORIZONS = np.array([1, 5, 20], dtype=np.int64)
NEURAL = ("summary_state", "state_price", "state_marginal", "state_step1",
          "free_path", "gru_price", "gru_full")
CLASSICAL = ("rw", "drift", "ridge_price", "ridge_full", "hgb_full")
FAMILIES = NEURAL + CLASSICAL
GENERATIVE = frozenset(NEURAL[:5])
NEURAL_SEEDS = (17, 29, 43)
POINT_CONTROLS = ("gru_full", "ridge_full", "hgb_full")
CANDIDATE = "summary_state"
SELECTED = "validation_selected_point"
COMPARATORS = tuple(f for f in FAMILIES if f != CANDIDATE) + (SELECTED,)
BOOTSTRAP_SEED = 20261019
BOOTSTRAP_REPETITIONS = 2000
BLOCK_LENGTHS = (20, 10, 40)
FIRST_ORIGIN = 60
SCHEMA_VERSION = 1


class AuditError(RuntimeError):
    """A failure of the declared data, forecast, or scoring contract."""


def require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise AuditError(message)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path: Path) -> dict:
    require(path.is_file(), f"Required JSON receipt is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        raise AuditError(f"Cannot read JSON receipt: {path}") from error
    require(isinstance(value, dict), f"Receipt must be a JSON object: {path}")
    return value


def close(a: Any, b: Any, message: str, rtol: float = 2e-12, atol: float = 1e-10) -> None:
    aa, bb = np.asarray(a), np.asarray(b)
    require(aa.shape == bb.shape, message + " (shape)")
    require(np.allclose(aa, bb, rtol=rtol, atol=atol, equal_nan=False), message)


def outside_git(path: Path) -> None:
    probe = subprocess.run(["git", "-C", str(path), "rev-parse", "--show-toplevel"],
                           capture_output=True, text=True, timeout=10, check=False)
    require(probe.returncode != 0,
            f"Observation/forecast inputs must stay outside a Git worktree: {path}")


@dataclass
class Source:
    name: str
    path: Path
    dates: np.ndarray
    price: np.ndarray
    high: np.ndarray
    low: np.ndarray
    sha256: str

    def labels(self, origins: np.ndarray) -> np.ndarray:
        """An independent literal definition; future window excludes origin."""
        result = np.empty((len(origins), 3, 3), dtype=np.float64)
        for row, origin in enumerate(origins):
            for j, horizon in enumerate(HORIZONS):
                future = slice(int(origin) + 1, int(origin + horizon) + 1)
                result[row, j] = (self.price[origin + horizon],
                                  np.max(self.high[future]), np.min(self.low[future]))
        return result

    def denominator(self, start: int, end: int) -> np.ndarray:
        origins = np.arange(start, end, dtype=np.int64)
        require(len(origins) > 0, "Empty training range for fixed scale")
        deviations = self.labels(origins) - self.price[origins, None, None]
        rms = np.sqrt(np.mean(deviations * deviations, axis=0))
        return np.maximum(rms, float(np.mean(self.price[origins])) * 0.001)


def load_source(directory: Path, name: str) -> Source:
    path = directory / f"{name}.csv"
    require(path.is_file(), f"Reconstructed source is missing: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ["source_index", "date", "price", "open", "high", "low", "volume"],
                f"Unexpected reconstructed source schema: {name}")
        rows = list(reader)
    require(len(rows) == {"quantum": 1805, "ding": 772}[name], f"Unexpected source length: {name}")
    dates = np.asarray([row["date"] for row in rows], dtype=str)
    if name == "quantum":
        require(np.all(dates == ""), "Quantum calendar dates must remain absent")
    else:
        # Check dates before converting numeric values from any admitted row.
        try:
            parsed = [date.fromisoformat(item) for item in dates]
        except ValueError as error:
            raise AuditError("Ding dates must be ISO calendar dates") from error
        require(all(date(2021, 1, 1) <= item < date(2024, 1, 1) for item in parsed),
                "Ding contains dates outside the admitted 2021--2023 interval")
        require(all(a < b for a, b in zip(parsed, parsed[1:])), "Ding dates must be unique and ordered")
    try:
        indexes = np.asarray([int(row["source_index"]) for row in rows])
        fields = {key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
                  for key in ("price", "high", "low")}
    except (TypeError, ValueError) as error:
        raise AuditError(f"Invalid source index or admitted price/range field: {name}") from error
    require(np.array_equal(indexes, np.arange(len(rows))), f"Source row order/index changed: {name}")
    require(all(np.isfinite(values).all() and (values > 0).all() for values in fields.values()),
            f"The two primary sources must have complete positive P/H/L: {name}")
    require(np.all(fields["high"] >= fields["low"]), f"Reported H/L order is invalid: {name}")
    return Source(name, path, dates, fields["price"], fields["high"], fields["low"], digest(path))


def fold_specs(source: Source) -> list[tuple[str, int, int]]:
    if source.name == "quantum":
        return [("Q1", 1000, 1250), ("Q2", 1250, 1500), ("Q3", 1500, 1785)]
    starts = [int(np.searchsorted(source.dates, cutoff))
              for cutoff in ("2022-07-01", "2023-01-01", "2023-07-01")]
    return [(f"D{i + 1}", start, end) for i, (start, end)
            in enumerate(zip(starts, starts[1:] + [len(source.price) - 20]))]


def energy_v(samples: np.ndarray, truth: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Exact ES of each equal-weight finite measure, divided by sqrt(9).

    pdist sums each unordered pair once. Thus sum(pdist)/S**2 equals
    0.5 * sum_{i,j} distance(x_i,x_j)/S**2, including the zero diagonal.
    No U-statistic or unbiased-continuous-distribution claim is made.
    """
    count = samples.shape[1]
    require(count > 0, "Energy score requires a nonempty sample forecast")
    x = (samples / denominator[None, None]).reshape(len(truth), count, 9)
    y = (truth / denominator[None]).reshape(len(truth), 9)
    scores = np.empty(len(truth), dtype=np.float64)
    for i in range(len(truth)):
        scores[i] = (np.linalg.norm(x[i] - y[i], axis=1).mean()
                     - pdist(x[i], metric="euclidean").sum() / count**2) / 3.0
    require(np.isfinite(scores).all(), "Energy scores are nonfinite")
    return scores


def code_fingerprint() -> str:
    return hashlib.sha256(b"".join((ROOT / name).read_bytes()
                                  for name in ("model.py", "pipeline.py", "run.py"))).hexdigest()


def check_receipt(metadata: dict, path: Path, source: Source, fold: str, family: str,
                  seed: int, start: int, end: int, fingerprint: str) -> None:
    expected = {
        "source": source.name, "fold": fold, "family": family, "seed": seed,
        "test_start": start, "test_end": end,
        "train_origin_start": FIRST_ORIGIN, "train_origin_end_exclusive": start - 20,
        "train_max_label_index": start - 1,
        "validation_origin_start": start - 120, "validation_origin_end_exclusive": start - 20,
        "selection_train_origin_end_exclusive": start - 140,
        "selection_train_max_label_index": start - 121,
        "data_sha256": source.sha256, "code_fingerprint": fingerprint,
        "horizons": HORIZONS.tolist(), "labels_count_observed_records": True,
        "numerical_clip_count": 0, "forecast_rows": end - start,
    }
    for key, value in expected.items():
        require(metadata.get(key) == value, f"Receipt field mismatch ({key}): {path}")
    require(metadata.get("forecast_sha256") == digest(path), f"Forecast byte hash mismatch: {path}")
    require(metadata["train_max_label_index"] < metadata["test_start"], f"Immature final labels: {path}")
    require(metadata["selection_train_max_label_index"] < metadata["validation_origin_start"],
            f"Immature selection labels: {path}")
    require(np.isfinite(metadata.get("validation_mse", np.nan)) and metadata["validation_mse"] >= 0,
            f"Invalid validation score: {path}")
    if family in NEURAL:
        checkpoint = path.with_suffix(".pt")
        require(checkpoint.is_file(), f"Neural checkpoint is missing: {checkpoint}")
        require(metadata.get("model_sha256") == digest(checkpoint), f"Neural checkpoint hash mismatch: {path}")


@dataclass
class Fold:
    source: str
    name: str
    start: int
    end: int
    selected: str
    losses: np.ndarray  # LOCAL only: origin x family x horizon
    mean_losses: np.ndarray  # family x horizon


def metric_row(source: str, fold: str, horizon: int, family: str, estimate: str,
               seed: int | str, errors: np.ndarray, rw_mse: float) -> dict:
    mse = float(np.mean(errors * errors))
    return {"source": source, "fold": fold, "horizon": int(horizon), "family": family,
            "estimate": estimate, "seed": seed, "origins": len(errors), "mse": mse,
            "rmse": math.sqrt(mse), "mae": float(np.mean(np.abs(errors))),
            "mse_ratio_to_rw": mse / rw_mse}


def seed_mean(values: list[np.ndarray]) -> np.ndarray:
    """Equal arithmetic mean with exact preservation of identical forecasts.

    Centering avoids roundoff from summing three identical origin prices and
    dividing by three. A zero-epoch RW forecast must remain exactly RW.
    """
    stacked = np.stack(values, axis=0)
    reference = stacked[0]
    return reference + np.mean(stacked - reference[None], axis=0)


def read_fold(source: Source, name: str, start: int, end: int, run_root: Path,
              fingerprint: str) -> tuple[Fold, list[dict], list[dict], dict]:
    folder = run_root / source.name / name
    manifest = read_json(folder / "manifest.json")
    for key, value in {"source": source.name, "fold": name, "test_start": start, "test_end": end,
                       "data_sha256": source.sha256, "code_fingerprint": fingerprint}.items():
        require(manifest.get(key) == value, f"Fold manifest mismatch ({key}): {source.name}/{name}")
    selected = manifest.get("validation_selected_baseline")
    scores = manifest.get("validation_baseline_scores", {})
    require(selected in POINT_CONTROLS and set(scores) == set(POINT_CONTROLS),
            f"Validation-selected point baseline is incomplete: {source.name}/{name}")
    require(all(np.isfinite(value) and value >= 0 for value in scores.values()),
            f"Invalid baseline validation score: {source.name}/{name}")
    close(scores[selected], min(scores.values()), "Selected baseline does not minimize validation MSE")
    expected_selected = min(POINT_CONTROLS, key=lambda family: (scores[family], family))
    require(selected == expected_selected, "Validation baseline violates the fixed exact-tie name ordering")

    origins = np.arange(start, end, dtype=np.int64)
    truth = source.labels(origins)
    denominator = source.denominator(FIRST_ORIGIN, start - 20)
    rw_errors = source.price[origins, None] - truth[:, :, 0]
    rw_mse = np.mean(rw_errors**2, axis=0)
    require(np.all(rw_mse > 0), f"Undefined relative MSE: zero RW loss in {source.name}/{name}")
    means, metric_rows, energy_rows, receipts = [], [], [], []
    for family in FAMILIES:
        family_dir = folder / family
        seeds = NEURAL_SEEDS if family in NEURAL else (0,)
        expected_files = {f"seed_{seed}.npz" for seed in seeds}
        require({item.name for item in family_dir.glob("seed_*.npz")} == expected_files,
                f"Missing or unexpected seed forecasts: {source.name}/{name}/{family}")
        complete = read_json(family_dir / "complete.json")
        for key, value in {"source": source.name, "fold": name, "family": family,
                           "data_sha256": source.sha256, "code_fingerprint": fingerprint}.items():
            require(complete.get(key) == value, f"Family completion receipt mismatch ({key}): {family_dir}")
        if family in POINT_CONTROLS:
            close(complete.get("validation_mse", np.nan), scores[family],
                  f"Fold baseline score differs from family validation receipt: {family_dir}")
        trials = complete.get("selection_fits", [])
        if trials:
            close(complete.get("validation_mse", np.nan), min(t["validation_mse"] for t in trials),
                  f"Family receipt does not select minimum validation MSE: {family_dir}")
        seed_means, seed_es = [], []
        for seed in seeds:
            path = family_dir / f"seed_{seed}.npz"
            metadata = read_json(path.with_suffix(".json"))
            check_receipt(metadata, path, source, name, family, seed, start, end, fingerprint)
            close(metadata["validation_mse"], complete.get("validation_mse", np.nan),
                  f"Prediction and selection validation scores differ: {path}")
            if family in NEURAL:
                for key in ("selected_epoch", "selected_learning_rate"):
                    require(metadata.get(key) == complete.get(key), f"Selected neural setting differs ({key}): {path}")
            with np.load(path, allow_pickle=False) as archive:
                keys = {"origin", "target_index", "truth", "valid", "p0", "mean", "samples", "denominator", "date"}
                require(keys.issubset(archive.files), f"Incomplete NPZ schema: {path}")
                values = {key: archive[key] for key in keys}
            require(values["origin"].dtype.kind in "iu" and np.array_equal(values["origin"], origins),
                    f"Forecast origins differ from the frozen common sequence: {path}")
            require(values["target_index"].dtype.kind in "iu" and
                    np.array_equal(values["target_index"], origins[:, None] + HORIZONS[None]),
                    f"Forecast endpoint indices are incorrect: {path}")
            close(values["truth"], truth, f"Independently reconstructed P/maxH/minL labels differ: {path}")
            require(values["valid"].dtype == np.bool_ and values["valid"].shape == truth.shape and
                    values["valid"].all(), f"Availability mask differs from complete source labels: {path}")
            close(values["p0"], source.price[origins], f"Origin price differs from source: {path}")
            close(values["denominator"], denominator, f"Training-only fixed RMS denominator differs: {path}")
            require(np.array_equal(values["date"], source.dates[origins]), f"Origin dates differ from source: {path}")
            mean, samples = np.asarray(values["mean"], dtype=float), np.asarray(values["samples"], dtype=float)
            require(mean.shape == (len(origins), 3) and np.isfinite(mean).all(), f"Invalid mean forecast array: {path}")
            count = 128 if family in GENERATIVE else 0
            require(samples.shape == (len(origins), count, 3, 3), f"Incorrect forecast sample shape/count: {path}")
            if family == "rw":
                close(mean, np.repeat(source.price[origins, None], 3, axis=1), f"RW is not current-price persistence: {path}")
            elif family == "drift":
                drift = float(np.mean(np.diff(source.price[:start])))
                close(mean, source.price[origins, None] + HORIZONS[None] * drift,
                      f"Drift baseline differs from its declared historical estimator: {path}")
            if count:
                require(np.isfinite(samples).all() and (samples > 0).all(), f"Nonpositive/nonfinite generated summary: {path}")
                require((mean > 0).all(), f"Nonpositive analytical price mean: {path}")
                require(np.all(samples[:, :, :, 1] >= samples[:, :, :, 2]), f"Generated high is below low: {path}")
                require(np.all(np.diff(samples[:, :, :, 1], axis=2) >= -1e-10), f"Cumulative high decreases across h: {path}")
                require(np.all(np.diff(samples[:, :, :, 2], axis=2) <= 1e-10), f"Cumulative low increases across h: {path}")
                # No test of P containment: the reported price is a separate field.
                score = float(np.mean(energy_v(samples, truth, denominator)))
                seed_es.append(score)
                energy_rows.append({"scope": "fold", "source": source.name, "fold": name,
                                    "family": family, "estimate": "seed", "seed": seed,
                                    "origins": len(origins), "samples_per_seed": count,
                                    "dimensions": 9, "energy_v": score})
            for j, horizon in enumerate(HORIZONS):
                metric_rows.append(metric_row(source.name, name, horizon, family, "seed", seed,
                                              mean[:, j] - truth[:, j, 0], rw_mse[j]))
            seed_means.append(mean)
            receipts.append({"family": family, "seed": seed, "forecast_sha256": metadata["forecast_sha256"],
                             "model_sha256": metadata.get("model_sha256"),
                             "metadata_sha256": digest(path.with_suffix(".json")),
                             "nonpositive_point_predictions": int(np.count_nonzero(mean <= 0))})
        ensemble = seed_mean(seed_means)
        means.append(ensemble)
        for j, horizon in enumerate(HORIZONS):
            metric_rows.append(metric_row(source.name, name, horizon, family, "mean_ensemble", "ensemble",
                                          ensemble[:, j] - truth[:, j, 0], rw_mse[j]))
        if seed_es:
            energy_rows.append({"scope": "fold", "source": source.name, "fold": name,
                                "family": family, "estimate": "seed_average", "seed": "average",
                                "origins": len(origins), "samples_per_seed": 128,
                                "dimensions": 9, "energy_v": float(np.mean(seed_es))})
    errors = np.stack(means, axis=1) - truth[:, None, :, 0]
    losses = errors * errors
    require(np.isfinite(losses).all(), f"Overflow/nonfinite squared loss in {source.name}/{name}")
    audit = {"source": source.name, "fold": name, "test_start": start, "test_end_exclusive": end,
             "common_origins": len(origins), "forecast_files": len(receipts),
             "truth_index_and_fixed_denominator_reconstructed": True,
             "train_max_label_index": start - 1, "selection_train_max_label_index": start - 121,
             "validation_origin_start": start - 120, "validation_origin_end_exclusive": start - 20,
             "validation_selected_baseline": selected, "validation_baseline_scores": scores,
             "denominator": denominator.tolist(), "receipts": receipts}
    return Fold(source.name, name, start, end, selected, losses, losses.mean(axis=0)), metric_rows, energy_rows, audit


def relative_cells(mse: np.ndarray, fold: Fold) -> np.ndarray:
    """... x family x horizon -> ... x comparator x horizon."""
    candidate = mse[..., FAMILIES.index(CANDIDATE), :]
    controls = np.stack([mse[..., FAMILIES.index(fold.selected if c == SELECTED else c), :]
                         for c in COMPARATORS], axis=-2)
    require(np.all(controls > 0) and np.isfinite(controls).all(),
            f"Comparison has zero/nonfinite denominator loss: {fold.source}/{fold.name}")
    result = candidate[..., None, :] / controls
    require(np.isfinite(result).all(), "A relative MSE is nonfinite")
    return result


def moving_block_means(losses: np.ndarray, length: int, repetitions: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Non-circular overlapping moving blocks; common indices for every model/h.

    Each replicate samples ceil(N/L) starts uniformly from [0,N-L], concatenates
    complete contiguous blocks, and truncates to N observations. No bootstrap
    origin indices or per-origin losses are exported.
    """
    n = len(losses)
    require(1 <= length <= n, "Bootstrap block length is longer than a fold")
    starts = rng.integers(0, n - length + 1, size=(repetitions, math.ceil(n / length)))
    indices = (starts[:, :, None] + np.arange(length)[None, None, :]).reshape(repetitions, -1)[:, :n]
    counts = np.zeros((repetitions, n), dtype=np.float64)
    np.add.at(counts, (np.arange(repetitions)[:, None], indices), 1.0 / n)
    return (counts @ losses.reshape(n, -1)).reshape(repetitions, *losses.shape[1:])


def comparisons(folds: list[Fold]) -> list[dict]:
    """Percentile CIs of the predeclared equally weighted mean cell ratios."""
    point = np.stack([relative_cells(fold.mean_losses, fold).mean(axis=-1) for fold in folds])
    groups = [("overall", "", "", list(range(len(folds))))]
    groups += [("source", source, "", [i for i, fold in enumerate(folds) if fold.source == source])
               for source in ("quantum", "ding")]
    groups += [("fold", fold.source, fold.name, [i]) for i, fold in enumerate(folds)]
    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for length in BLOCK_LENGTHS:
        bootstrap = np.stack([relative_cells(moving_block_means(fold.losses, length,
                                       BOOTSTRAP_REPETITIONS, rng), fold).mean(axis=-1) for fold in folds], axis=1)
        for scope, source, name, indexes in groups:
            values = point[indexes].mean(axis=0)
            draws = bootstrap[:, indexes].mean(axis=1)
            lower, upper = np.quantile(draws, [0.025, 0.975], axis=0)
            for j, comparator in enumerate(COMPARATORS):
                rows.append({"scope": scope, "source": source, "fold": name,
                             "candidate": CANDIDATE, "comparator": comparator,
                             "metric": "equal_weight_mean_cell_mse_ratio",
                             "ratio": float(values[j]), "gain_percent": float(100 * (1 - values[j])),
                             "ci_lower": float(lower[j]), "ci_upper": float(upper[j]),
                             "block_length": length, "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
                             "bootstrap_seed": BOOTSTRAP_SEED})
    return rows


def gates(rows: list[dict]) -> dict:
    main = [row for row in rows if row["block_length"] == 20]
    overall = {row["comparator"]: row for row in main if row["scope"] == "overall"}
    source_rw = {row["source"]: row["ratio"] for row in main
                 if row["scope"] == "source" and row["comparator"] == "rw"}
    fold_rw = {f'{row["source"]}/{row["fold"]}': row["ratio"] for row in main
               if row["scope"] == "fold" and row["comparator"] == "rw"}
    require(len(source_rw) == 2 and len(fold_rw) == 6, "Gate aggregation is missing sources/folds")
    checks = []

    def add(name: str, value: float, operator: str, threshold: float) -> None:
        passed = value <= threshold if operator == "<=" else value < threshold if operator == "<" else value >= threshold
        checks.append({"gate": name, "value": value, "operator": operator,
                       "threshold": threshold, "passed": bool(passed)})

    add("overall_candidate_rw_ratio", overall["rw"]["ratio"], "<=", 0.98)
    add("validation_selected_same_information_point_ratio", overall[SELECTED]["ratio"], "<=", 0.99)
    add("candidate_state_price_ratio", overall["state_price"]["ratio"], "<=", 0.99)
    add("candidate_state_price_ci_upper", overall["state_price"]["ci_upper"], "<", 1.0)
    add("candidate_free_path_ratio", overall["free_path"]["ratio"], "<=", 0.99)
    add("candidate_free_path_ci_upper", overall["free_path"]["ci_upper"], "<", 1.0)
    for source, value in source_rw.items():
        add(f"{source}_candidate_rw_ratio", value, "<", 1.0)
    add("folds_better_than_rw", float(sum(value < 1 for value in fold_rw.values())), ">=", 4.0)
    add("overall_candidate_rw_ci_upper", overall["rw"]["ci_upper"], "<", 1.0)
    passed = all(check["passed"] for check in checks)
    return {"candidate": CANDIDATE, "all_passed": passed,
            "decision": "PASS_PREDEFINED_EMPIRICAL_GATES" if passed else "STOP_CANDIDATE_NO_RESCUE",
            "checks": checks, "source_ratios_to_rw": source_rw, "fold_ratios_to_rw": fold_rw,
            "ci": {"level": 0.95, "method": "noncircular_moving_block_percentile",
                   "primary_block_length": 20, "repetitions": BOOTSTRAP_REPETITIONS,
                   "seed": BOOTSTRAP_SEED},
            "note": "Passing these empirical gates does not certify methodological novelty or publication readiness."}


def energy_aggregates(rows: list[dict]) -> list[dict]:
    result = list(rows)
    for family in sorted(GENERATIVE):
        base = [row for row in rows if row["family"] == family and row["estimate"] == "seed_average"]
        require(len(base) == 6, "Incomplete fold energy scores")
        for scope, source in (("source", "quantum"), ("source", "ding"), ("overall", "")):
            subset = [row for row in base if not source or row["source"] == source]
            result.append({"scope": scope, "source": source, "fold": "", "family": family,
                           "estimate": "seed_average_then_equal_fold_average", "seed": "average",
                           "origins": sum(row["origins"] for row in subset), "samples_per_seed": 128,
                           "dimensions": 9, "energy_v": float(np.mean([row["energy_v"] for row in subset]))})
    return result


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), f"No aggregate rows for {path.name}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def self_test() -> None:
    """Small numerical checks use synthetic arrays and never inspect market data."""
    # Price lies deliberately outside the reported interval. Index/high/low
    # reconstruction must preserve it, and exclude the origin's extreme high.
    price = np.arange(100.0, 140.0)
    high, low = np.arange(10.0, 50.0), np.arange(1.0, 41.0)
    high[2] = 10000.0
    source = Source("synthetic", Path("/tmp/unused"), np.repeat("", 40), price, high, low, "")
    truth = source.labels(np.array([2]))
    close(truth[0, 0], np.array([103.0, 13.0, 4.0]), "Synthetic future-window check")
    require(truth[0, 0, 0] > truth[0, 0, 1], "Synthetic P containment must not be repaired")
    close(truth[0, 2], np.array([122.0, 32.0, 4.0]), "Synthetic h20 endpoint check")
    # Compare the optimized V-statistic with the literal full distance matrix.
    rng = np.random.default_rng(7)
    x, y = rng.normal(size=(4, 8, 3, 3)), rng.normal(size=(4, 3, 3))
    den = np.arange(1.0, 10.0).reshape(3, 3)
    flat, target = (x / den).reshape(4, 8, 9), (y / den).reshape(4, 9)
    distance = np.linalg.norm(flat[:, :, None, :] - flat[:, None, :, :], axis=-1)
    literal = (np.linalg.norm(flat - target[:, None, :], axis=-1).mean(axis=1)
               - 0.5 * distance.mean(axis=(1, 2))) / 3.0
    close(energy_v(x, y, den), literal, "Finite V-statistic implementation")
    # Every column must receive exactly the same bootstrap origins. Ratios of
    # proportional loss arrays consequently stay constant in every replicate.
    a = np.arange(1.0, 51.0)
    loss = np.stack([a, 2 * a, 4 * a], axis=1)[:, :, None]
    boot = moving_block_means(loss, 10, 100, np.random.default_rng(31))
    close(boot[:, 1, :] / boot[:, 0, :], np.full((100, 1), 2.0), "Bound model bootstrap pairing")
    close(boot[:, 2, :] / boot[:, 0, :], np.full((100, 1), 4.0), "Bound horizon bootstrap pairing")
    # Averaging seed means precedes squaring: opposing errors cancel.
    ensemble = seed_mean([np.array([1.0, 3.0]), np.array([3.0, 1.0])])
    close((ensemble - 2.0)**2, np.zeros(2), "Mean ensemble must precede MSE")
    identical = np.array([0.1, 3.3, 102.81])
    require(np.array_equal(seed_mean([identical] * 3), identical),
            "Identical seed forecasts must remain exactly identical")
    print(json.dumps({"self_test": "PASS", "checks": ["future_window_and_uncontained_price",
                      "finite_energy_v", "paired_moving_blocks", "ensemble_before_mse",
                      "identical_seed_forecast_preservation"]}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("/tmp/carbon-path-local"))
    parser.add_argument("--run-root", type=Path, default=Path("/tmp/carbon-path-run"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    data_dir, run_root = args.data_dir.resolve(), args.run_root.resolve()
    outside_git(data_dir)
    outside_git(run_root)
    require(args.output_dir.resolve() not in (data_dir, run_root), "Aggregate output cannot replace an input directory")
    fingerprint = code_fingerprint()
    plan = read_json(run_root / "plan.json")
    require(plan.get("code_fingerprint") == fingerprint, "Frozen runner fingerprint differs from current source code")
    require(plan.get("neural_families") == list(NEURAL) and plan.get("classical_families") == list(CLASSICAL)
            and plan.get("seeds") == list(NEURAL_SEEDS) and plan.get("evaluation_samples") == 128,
            "The run plan differs from the fixed model/seed/sample contract")
    fold_results, metrics, energies, fold_audits, source_audits = [], [], [], [], []
    for name in ("quantum", "ding"):
        source = load_source(data_dir, name)
        source_audits.append({"source": name, "rows": len(source.price), "data_sha256": source.sha256,
                              "calendar_available": name == "ding",
                              "price_outside_reported_daily_range": int(np.count_nonzero(
                                  (source.price < source.low) | (source.price > source.high))),
                              "zero_range_rows": int(np.count_nonzero(source.high == source.low)),
                              "price_containment_repairs_applied": False})
        for fold_name, start, end in fold_specs(source):
            fold, cell_rows, energy_rows, audit = read_fold(source, fold_name, start, end, run_root, fingerprint)
            fold_results.append(fold)
            metrics.extend(cell_rows)
            energies.extend(energy_rows)
            fold_audits.append(audit)
    require(len(fold_results) == 6 and sum(a["forecast_files"] for a in fold_audits) == 156,
            "Full experiment must contain six folds and 156 forecast files")
    comparison_rows = comparisons(fold_results)
    gate_result = gates(comparison_rows)
    seed_rows = []
    for family in FAMILIES:
        for seed in (NEURAL_SEEDS if family in NEURAL else (0,)):
            cells = [row for row in metrics if row["family"] == family and row["estimate"] == "seed" and row["seed"] == seed]
            require(len(cells) == 18, "Single-seed report is missing cells")
            seed_rows.append({"family": family, "seed": seed, "cells": 18,
                              "equal_weight_mean_mse_ratio_to_rw": float(np.mean([r["mse_ratio_to_rw"] for r in cells]))})
    now = datetime.now(timezone.utc).isoformat()
    audit_result = {"schema_version": SCHEMA_VERSION, "status": "PASS", "created_utc": now,
                    "scorer_sha256": digest(Path(__file__)), "runner_code_fingerprint": fingerprint,
                    "sources": source_audits, "folds": fold_audits,
                    "per_origin_records_exported": False,
                    "limitations": ["Receipt boundaries are checked independently; they are not a runtime data-access trace.",
                        "Only supplied summary samples can be checked; their unexported daily rollout is audited separately.",
                        "Two author sources concern EUA; they are not two independent carbon markets.",
                        "Quantum has no authenticated calendar, so cross-source calendar covariance cannot be recovered."]}
    summary = {"schema_version": SCHEMA_VERSION, "status": "COMPLETE", "created_utc": now,
               "candidate": CANDIDATE, "horizons_observed_records": HORIZONS.tolist(),
               "mse_aggregation": "Average neural analytical/deterministic means over seeds, compute MSE per source/fold/h, then equally average the 18 candidate/control MSE ratios.",
               "source_weight": {"quantum": 0.5, "ding": 0.5},
               "energy_aggregation": "Per-seed ES of its equal-weight 128-sample finite forecast (V statistic), nine train-scaled coordinates divided by sqrt(9); average seeds, then folds equally. Never substitute sample means for supplied analytical means.",
               "bootstrap": {"type": "noncircular_overlapping_moving_block_percentile", "blocks": list(BLOCK_LENGTHS),
                             "repetitions": BOOTSTRAP_REPETITIONS, "seed": BOOTSTRAP_SEED,
                             "paired_dimensions": "All horizons and models at the same origin; folds resampled independently."},
               "gates": gate_result,
               "overall_comparisons_primary_block": [row for row in comparison_rows if row["scope"] == "overall" and row["block_length"] == 20],
               "single_seed_overall": seed_rows,
               "validation_selected_baselines": {f"{f.source}/{f.name}": f.selected for f in fold_results},
               "no_test_driven_model_selection_or_rescue": True}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(args.output_dir / "mse_by_cell.csv", metrics)
    atomic_csv(args.output_dir / "single_seed_overall.csv", seed_rows)
    atomic_csv(args.output_dir / "comparisons.csv", comparison_rows)
    atomic_csv(args.output_dir / "energy_scores.csv", energy_aggregates(energies))
    atomic_json(args.output_dir / "audit.json", audit_result)
    atomic_json(args.output_dir / "gates.json", gate_result)
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"status": "COMPLETE", "audit": "PASS", "decision": gate_result["decision"],
                      "aggregate_output": str(args.output_dir.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AuditError, OSError, ValueError, KeyError, TypeError) as error:
        # No array values are included in our audit messages; no partial public
        # results are written before the complete audit/statistical pass.
        print(json.dumps({"status": "AUDIT_FAILED", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        sys.exit(2)
