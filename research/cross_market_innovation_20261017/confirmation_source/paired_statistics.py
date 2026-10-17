"""Frozen, NumPy-only confirmation inference; this module never opens data files.

Primary estimand: equal-year/equal-horizon difference in MSE / RW-MSE, where
each model's point loss is averaged over the declared, fixed training seeds.
This is NOT the squared error of a seed-ensemble prediction. The two fixed
comparators are the strongest development baseline and matched augmentation.

Paired circular moving-block bootstrap within each year preserves pairing
across models, seeds, and horizons. L=40 is primary; L=20 and L=60 are labelled
sensitivity analyses. All cells' RW denominators are recomputed in every draw.
Primary one-sided tests use the centered bootstrap null and Holm over exactly
two comparisons. Confidence intervals are basic-bootstrap intervals, rather
than percentile intervals; both 95% marginal and 97.5% Bonferroni intervals
are reported. The latter give an approximate 95% simultaneous family, subject
to the same bootstrap assumptions. No CI is labelled "Holm-adjusted".

Inference is conditional on frozen development selection and the supplied
seed set. Weak dependence / adequate within-year sampling are assumptions,
not established by choosing a block at least as long as H. The procedure does
not provide distribution-free coverage under arbitrary drift.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

PRIMARY_HORIZONS = (1, 5, 20)
PRIMARY_BLOCK = 40
SENSITIVITY_BLOCKS = (20, 60)
DEFAULT_REPLICATES = 10000
DEFAULT_BOOTSTRAP_SEED = 20261019
PRIMARY_ALPHA = 0.05
EXPECTED_YEARS = (2024, 2025)
MAXIMUM_CONFIRMATION_ORIGINS = 221


def seed_mean_squared_loss(prediction, raw_y):
    """Return [origin,horizon] losses from [seed,origin,horizon] predictions."""
    y = np.asarray(raw_y, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.ndim != 2 or p.ndim != 3 or p.shape[1:] != y.shape or not p.shape[0]:
        raise ValueError("Expected truth [N,H] and predictions [S,N,H], with S >= 1.")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Truth and predictions must be finite; dropping individual methods' rows is forbidden.")
    result = np.mean(np.square(p - y[None]), axis=0)
    if not np.isfinite(result).all():
        raise ValueError("Squared losses overflowed; supply finite, consistently scaled raw changes.")
    return result


def circular_block_indices(n, block_length, replicates, rng):
    """Draw fixed-length circular blocks, concatenate, and truncate to n rows.

    Every block start is uniform on 0,...,n-1. A block may wrap within the same
    year. No block crosses years. The caller uses this single index matrix for
    every method and horizon. L >= n is intentionally refused: it supplies
    too few independent block starts and can degenerate to a full cyclic mean.
    """
    if not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("Each stratum requires at least two ordered origins.")
    if not isinstance(block_length, (int, np.integer)) or not 1 <= block_length < n:
        raise ValueError("Block length must be a positive integer strictly smaller than its year's n.")
    if not isinstance(replicates, (int, np.integer)) or replicates < 1:
        raise ValueError("replicates must be positive.")
    starts = rng.integers(0, n, size=(replicates, (n + block_length - 1) // block_length))
    return ((starts[:, :, None] + np.arange(block_length)) % n).reshape(replicates, -1)[:, :n]


def holm_adjust(pvalues, alpha=PRIMARY_ALPHA):
    """Holm step-down adjusted p values, in the input comparison order."""
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 1 or not p.size or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("pvalues must be a nonempty vector of finite values in [0,1].")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one.")
    order = np.argsort(p, kind="stable")
    sorted_adjusted = np.minimum(1.0, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    adjusted = np.empty_like(sorted_adjusted)
    adjusted[order] = sorted_adjusted
    return adjusted, adjusted <= alpha


def _basic_interval(observed, draws, confidence):
    alpha = 1.0 - confidence
    low, high = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear")
    return [float(2.0 * observed - high), float(2.0 * observed - low)]


def _inference_summary(observed, draws, comparison_count):
    # Under H0: delta=0, draws-observed approximates the null sampling error.
    # H1 is delta<0, so count null errors at or below the observed statistic.
    p = (1 + int(np.count_nonzero(draws - observed <= observed))) / (len(draws) + 1)
    simultaneous_confidence = 1.0 - PRIMARY_ALPHA / comparison_count
    return {
        "observed_delta": float(observed),
        "basic_ci_95": _basic_interval(observed, draws, 0.95),
        "basic_bonferroni_ci_97_5": _basic_interval(observed, draws, simultaneous_confidence),
        "centered_bootstrap_one_sided_p": float(p),
        "bootstrap_standard_deviation": float(np.std(draws, ddof=1)),
        "bootstrap_mean": float(np.mean(draws)),
        "monte_carlo_p_resolution": 1.0 / (len(draws) + 1),
    }


def _coerce_years(values, n):
    year = np.asarray(values)
    if year.shape != (n,):
        raise ValueError("years must contain one year per aligned forecast origin.")
    labels = np.array([str(x) for x in year])
    if any(len(x) != 4 or not x.isdigit() for x in labels):
        raise ValueError("years must be exact four-digit years, not fractional years or dates.")
    return labels.astype(np.int64)


def _prepare(predictions, raw_y, years, origins, names, seed_ids):
    if not isinstance(predictions, Mapping) or set(predictions) != set(names):
        raise ValueError("Supply exactly the candidate and the two development-frozen comparator names.")
    if len(names) != 3 or len(set(names)) != 3 or not all(isinstance(x, str) and x for x in names):
        raise ValueError("The three model names must be distinct nonempty strings.")
    y = np.asarray(raw_y, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != 20 or not 1 <= len(y) <= MAXIMUM_CONFIRMATION_ORIGINS:
        raise ValueError("Confirmation truth must have H=20 and 1..221 jointly aligned origins.")
    if not np.isfinite(y).all():
        raise ValueError("Truth must be finite. Never perform model-specific missing-row deletion.")
    year = _coerce_years(years, len(y))
    if tuple(np.unique(year)) != EXPECTED_YEARS or np.any(np.diff(year) < 0):
        raise ValueError("This frozen confirmation requires ordered strata 2024 and 2025 only.")
    origin = np.asarray(origins)
    if origin.shape != (len(y),) or origin.dtype.kind not in "iu" or np.any(origin[1:] <= origin[:-1]):
        raise ValueError("origins must be strictly increasing, unique integer source-row indices.")
    groups = [np.flatnonzero(year == value) for value in EXPECTED_YEARS]
    for ids in groups:
        if np.any(np.diff(origin[ids]) != 1):
            raise ValueError("Origins must be consecutive within each year; an origin-block cannot silently bridge removed rows.")
    seeds = tuple(seed_ids)
    if not seeds or len(set(seeds)) != len(seeds) or any(not isinstance(x, (int, np.integer)) for x in seeds):
        raise ValueError("Declare a nonempty unique sequence of integer training seed IDs.")
    hidx = np.array(PRIMARY_HORIZONS) - 1
    losses = []
    for name in names:
        pred = np.asarray(predictions[name], dtype=np.float64)
        if pred.shape != (len(seeds), len(y), 20):
            raise ValueError(f"{name}: expected [{len(seeds)}, {len(y)}, 20]; seed order must match seed_ids.")
        losses.append(seed_mean_squared_loss(pred, y)[:, hidx])
    rw = np.square(y[:, hidx])
    if not np.isfinite(rw).all():
        raise ValueError("RW squared changes overflowed.")
    # [origin, method (three + RW), selected horizon]. No labels are revisited
    # after constructing this fixed point-loss panel.
    panel = np.stack(losses + [rw], axis=1)
    cells = np.stack([panel[ids].mean(axis=0) for ids in groups])
    if np.any(cells[:, -1] <= 0):
        raise ValueError("Observed year/horizon RW MSE is zero; the registered relative-MSE estimand is undefined.")
    normalized = cells[:, :3] / cells[:, -1:, :]
    scores = normalized.mean(axis=(0, 2))
    normalized_effect = np.mean(normalized[:, :1] - normalized[:, 1:3], axis=(0, 2))
    raw_effect = np.mean(cells[:, :1] - cells[:, 1:3], axis=(0, 2))
    return panel, groups, cells, scores, normalized_effect, raw_effect, seeds, origin


def confirmation_inference(
    predictions: Mapping[str, np.ndarray],
    raw_y,
    years,
    origins,
    *,
    candidate: str,
    strongest_development_baseline: str,
    matched_augmentation: str,
    seed_ids: Sequence[int],
    replicates: int = DEFAULT_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
):
    """Return JSON-serializable inference for the single frozen Shanghai panel.

    Inputs are raw cumulative-change forecasts/truth, NOT price levels, ordered
    identically by seed/origin/horizon. The upstream audited support builder
    must ensure all H20 targets belong to the forecast origin's calendar year;
    this function checks year/origin order and the 221-origin bound, but cannot
    infer target dates from raw_y. A deterministic baseline may be broadcast to
    every seed, which leaves its loss unchanged and creates no extra samples.

    No model selection, data loading, model training, seed resampling, or
    bootstrap block-length selection occurs here. Fixed defaults are part of
    the pre-confirmation protocol. Override replicates/seed only in synthetic
    checks, or explicitly register the override before opening confirmation.
    """
    if not isinstance(replicates, (int, np.integer)) or replicates < 99:
        raise ValueError("At least 99 bootstrap draws are required; the frozen formal default is 10,000.")
    if not isinstance(bootstrap_seed, (int, np.integer)) or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer.")
    names = (candidate, strongest_development_baseline, matched_augmentation)
    panel, groups, cells, scores, delta, raw_delta, seeds, origin = _prepare(
        predictions, raw_y, years, origins, names, seed_ids
    )
    result = {
        "protocol": {
            "estimand": "equal-year/equal-horizon mean of (seed-mean candidate squared loss minus seed-mean comparator squared loss) / RW MSE",
            "negative_delta_favors_candidate": True,
            "years": list(EXPECTED_YEARS), "horizons": list(PRIMARY_HORIZONS),
            "primary_block_length": PRIMARY_BLOCK,
            "sensitivity_block_lengths": list(SENSITIVITY_BLOCKS),
            "bootstrap": "paired, stratified-year, circular moving blocks; uniform starts, concatenate and truncate within year",
            "replicates": int(replicates), "bootstrap_seed": int(bootstrap_seed),
            "rng": "NumPy PCG64(SeedSequence([bootstrap_seed, block_length]))",
            "bootstrap_chunk_size": 256,
            "bootstrap_denominators_recomputed": True,
            "null": "candidate relative-MSE delta >= 0", "alternative": "delta < 0",
            "p_value": "(1 + count(delta_star - delta_hat <= delta_hat)) / (B+1)",
            "ci_method": "basic bootstrap; 2*delta_hat minus reversed empirical quantiles",
            "multiplicity": "Holm over two primary block-40 comparisons only, alpha=0.05; positive-evidence gate requires adjusted p < 0.05",
            "sensitivity_results_are_confirmatory": False,
            "seed_loss_aggregation": "average squared losses, never square the average prediction",
            "seed_ids": [int(x) for x in seeds], "seed_resampling": False,
            "model_order": list(names),
            "comparator_roles": {"strongest_development_baseline": names[1], "matched_augmentation": names[2]},
            "confirmation_maximum_origins": MAXIMUM_CONFIRMATION_ORIGINS,
        },
        "support": [{"year": y, "origins": len(ids), "first_origin": int(origin[ids[0]]), "last_origin": int(origin[ids[-1]])}
                    for y, ids in zip(EXPECTED_YEARS, groups)],
        "observed_scores": {name: float(value) for name, value in zip(names, scores)},
        "observed_cells": [],
        "blocks": {},
        "limitations": [
            "Coverage and p values are approximate under adequate within-year weak dependence; arbitrary regime changes are not covered by an exact validity claim.",
            "Inference is conditional on the frozen development selection and fixed training seeds; it does not estimate across-seed or across-market sampling uncertainty.",
            "Two year strata do not supply two independent replications of a general market population.",
            "If the fixed 2025 refresh uses matured 2024 labels, year strata are not literally independent: this fixed-loss bootstrap does not repeat fitting or propagate that training-estimation dependence.",
            "Circular wrap is a resampling device, not a claim that year-end and year-start observations are adjacent in real time.",
            "Large blocks in a short year yield few effective blocks; do not select a block length after examining significance.",
            "Same-year mature H20 target support must be verified upstream; raw outcome arrays alone cannot verify target dates.",
            "Any zero bootstrap RW denominator invalidates normalized inference at that block length; no epsilon, dropped draw, or block-length rescue is used.",
        ],
    }
    for yi, year in enumerate(EXPECTED_YEARS):
        for hi, h in enumerate(PRIMARY_HORIZONS):
            result["observed_cells"].append({
                "year": year, "horizon": h, "origins": len(groups[yi]),
                "rw_mse": float(cells[yi, -1, hi]),
                "mse_by_model": {name: float(cells[yi, mi, hi]) for mi, name in enumerate(names)},
                "relative_mse_by_model": {name: float(cells[yi, mi, hi] / cells[yi, -1, hi]) for mi, name in enumerate(names)},
            })
    for length in (PRIMARY_BLOCK,) + SENSITIVITY_BLOCKS:
        block = {
            "role": "primary" if length == PRIMARY_BLOCK else "sensitivity_only",
            "block_length": length,
            "effective_blocks_by_year": {str(year): len(ids) / length for year, ids in zip(EXPECTED_YEARS, groups)},
            "drawn_blocks_per_year": {str(year): (len(ids) + length - 1) // length for year, ids in zip(EXPECTED_YEARS, groups)},
        }
        result["blocks"][str(length)] = block
        if any(len(ids) <= length for ids in groups):
            block.update(status="NOT_ESTIMABLE", reason="At least one year has n <= registered block length. No automatic cap or alternate primary block.")
            continue
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(bootstrap_seed), length])))
        boot_delta = np.empty((replicates, 2))
        boot_raw = np.empty((replicates, 2))
        invalid = np.zeros(replicates, dtype=bool)
        for first in range(0, replicates, 256):
            stop = min(first + 256, replicates)
            n_draws = stop - first
            normalized_sum, raw_sum = np.zeros((n_draws, 2)), np.zeros((n_draws, 2))
            batch_invalid = np.zeros(n_draws, dtype=bool)
            for ids in groups:
                choices = circular_block_indices(len(ids), length, n_draws, rng)
                sampled = panel[ids][choices].mean(axis=1)
                numerator = sampled[:, :1] - sampled[:, 1:3]
                denominator = sampled[:, -1:, :]
                batch_invalid |= np.any(denominator[:, 0] <= 0, axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    normalized_sum += np.mean(numerator / denominator, axis=2) / len(groups)
                raw_sum += np.mean(numerator, axis=2) / len(groups)
            boot_delta[first:stop] = normalized_sum
            boot_raw[first:stop] = raw_sum
            invalid[first:stop] = batch_invalid
        block["zero_rw_denominator_draws"] = int(invalid.sum())
        block["secondary_raw_mse_difference"] = {
            name: {"observed_delta": float(raw_delta[j]), "basic_ci_95": _basic_interval(raw_delta[j], boot_raw[:, j], 0.95)}
            for j, name in enumerate(names[1:])
        }
        if invalid.any() or not np.isfinite(boot_delta).all():
            block.update(status="NOT_ESTIMABLE", reason="At least one normalized bootstrap draw is undefined; all draws retained in accounting and no normalized CI or p value issued.")
            continue
        block["status"] = "ESTIMATED"
        block["comparisons"] = {name: _inference_summary(delta[j], boot_delta[:, j], 2) for j, name in enumerate(names[1:])}
        if length == PRIMARY_BLOCK:
            adjusted, rejected = holm_adjust([block["comparisons"][name]["centered_bootstrap_one_sided_p"] for name in names[1:]])
            for j, name in enumerate(names[1:]):
                block["comparisons"][name].update(holm_adjusted_p=float(adjusted[j]),
                                                   holm_reject=bool(rejected[j]),
                                                   candidate_direction_and_holm_pass=bool(adjusted[j] < PRIMARY_ALPHA and delta[j] < 0))
            block["both_primary_comparisons_pass"] = bool(np.all(adjusted < PRIMARY_ALPHA) and np.all(delta < 0))
        else:
            block["multiplicity_adjustment"] = "none; descriptive sensitivity p values are not additional primary tests"
    primary = result["blocks"][str(PRIMARY_BLOCK)]
    result["primary_inference_status"] = primary["status"]
    result["both_primary_comparisons_pass"] = primary.get("both_primary_comparisons_pass", False)
    result["numpy_version"] = np.__version__
    return result
