"""Synthetic-only checks of frozen confirmation statistics. No market I/O."""
from __future__ import annotations

import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("carbon_confirmation_statistics", HERE / "statistics.py")
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)


def call(predictions, y, *, counts=(80, 101), replicates=99):
    n0, n1 = counts
    years = np.r_[np.full(n0, 2024), np.full(n1, 2025)]
    origins = np.r_[np.arange(n0), np.arange(n0 + 20, n0 + 20 + n1)]
    return stats.confirmation_inference(
        predictions, y, years, origins, candidate="candidate",
        strongest_development_baseline="baseline", matched_augmentation="augment",
        seed_ids=(17, 29, 43), replicates=replicates,
    )


def expect_error(function, label):
    try:
        function()
    except ValueError:
        return label
    raise AssertionError(label + ": expected ValueError")


def main():
    names = ("candidate", "baseline", "augment")
    n = 181
    y = np.ones((n, 20))
    checks = []
    maximum = 0.0

    # Formal bootstrap count and random seed are exercised on synthetic data.
    perfect = {"candidate": np.broadcast_to(y, (3, n, 20)),
               "baseline": np.zeros((3, n, 20)),
               "augment": np.zeros((3, n, 20))}
    result = call(perfect, y, replicates=stats.DEFAULT_REPLICATES)
    assert result["protocol"]["replicates"] == 10000
    assert result["protocol"]["bootstrap_seed"] == 20261019
    assert result["both_primary_comparisons_pass"]
    for length in (20, 40, 60):
        for item in result["blocks"][str(length)]["comparisons"].values():
            assert item["observed_delta"] == -1.0
            assert item["basic_ci_95"] == [-1.0, -1.0]
            assert item["centered_bootstrap_one_sided_p"] == 1 / 10001
    checks.append("formal 10000 replicates/seed20261019; known constant benefit and finite plus-one p")

    # Average loss across seeds retains disagreement that prediction ensembling
    # would cancel. Truth=1 and predictions=0,1,2 give MSE 2/3, not zero.
    disagreement = np.stack([np.zeros_like(y), np.ones_like(y), 2 * np.ones_like(y)])
    loss = stats.seed_mean_squared_loss(disagreement, y)
    assert np.all(loss == 2 / 3)
    assert np.mean((disagreement.mean(axis=0) - y) ** 2) == 0
    checks.append("seed-average squared loss is distinct from squared ensemble error")

    identical = call({name: disagreement for name in names}, y)
    assert not identical["both_primary_comparisons_pass"]
    for length in (20, 40, 60):
        for item in identical["blocks"][str(length)]["comparisons"].values():
            assert item["observed_delta"] == 0.0 and item["basic_ci_95"] == [0.0, 0.0]
            assert item["centered_bootstrap_one_sided_p"] == 1.0
    checks.append("identical paired forecasts have exactly zero effect/interval and conservative p=1")

    # Unequal stratum sizes must not change their declared equal weights.
    by_year = np.r_[np.ones((80, 20)), 3 * np.ones((101, 20))]
    variable = {"candidate": np.broadcast_to(y - by_year, (3, n, 20)),
                "baseline": np.broadcast_to(y, (3, n, 20)),
                "augment": np.broadcast_to(y, (3, n, 20))}
    weights = call(variable, y)
    assert weights["observed_scores"]["candidate"] == (1 + 9) / 2
    assert weights["observed_scores"]["candidate"] != (80 + 101 * 9) / 181
    checks.append("unequal years receive equal cell weights, not pooled-origin weights")

    adjusted, rejected = stats.holm_adjust([0.01, 0.04])
    assert np.array_equal(adjusted, [0.02, 0.04]) and np.array_equal(rejected, [True, True])
    adjusted, rejected = stats.holm_adjust([0.03, 0.04])
    assert np.array_equal(adjusted, [0.06, 0.06]) and not rejected.any()
    adjusted, rejected = stats.holm_adjust([0.9, 0.01])
    assert np.array_equal(adjusted, [0.9, 0.02]) and np.array_equal(rejected, [False, True])
    checks.append("Holm step-down thresholds and original comparison ordering")

    # Independent scalar-loop bootstrap replay. It does not call the module's
    # index generator, effect evaluator, p-value, or confidence interval helper.
    rng = np.random.default_rng(925)
    hscale = np.sqrt(np.arange(1, 21))[None]
    y2 = (1 + 0.2 * np.sin(np.arange(n)[:, None] / 7.0)) * hscale
    shared_error = np.cumsum(rng.normal(0, 0.025, n))[:, None] * hscale
    p2 = {
        "candidate": np.stack([y2 + 0.7 * shared_error + k * 0.01 * hscale for k in (-1, 0, 1)]),
        "baseline": np.stack([y2 + shared_error + k * 0.02 * hscale for k in (-1, 0, 1)]),
        "augment": np.stack([y2 + 0.9 * shared_error - k * 0.01 * hscale for k in (-1, 0, 1)]),
    }
    actual = call(p2, y2)
    assert actual == call(p2, y2), "fixed-seed results must be deterministic"
    loss2 = {name: ((p2[name] - y2[None]) ** 2).sum(axis=0) / 3 for name in names}
    hids = (0, 4, 19)
    groups = (np.arange(80), np.arange(80, 181))
    for length in (20, 40, 60):
        draw_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261019, length])))
        indices = []
        for group in groups:
            starts = draw_rng.integers(0, len(group), size=(99, (len(group) + length - 1) // length))
            # Literal loop concatenation verifies wrap, truncation, and pairing.
            sample = [[int(group[(start + offset) % len(group)])
                       for start in row for offset in range(length)][:len(group)] for row in starts]
            indices.append(sample)
        observed = []
        draws = []
        for comparator in names[1:]:
            obs_terms = [(loss2["candidate"][group, h].mean() - loss2[comparator][group, h].mean())
                         / (y2[group, h] ** 2).mean() for group in groups for h in hids]
            observed.append(float(np.mean(obs_terms)))
            samples = []
            for b in range(99):
                terms = []
                for group_index in range(2):
                    sampled = indices[group_index][b]
                    for h in hids:
                        terms.append((loss2["candidate"][sampled, h].mean() - loss2[comparator][sampled, h].mean())
                                     / (y2[sampled, h] ** 2).mean())
                samples.append(float(np.mean(terms)))
            draws.append(np.asarray(samples))
        for j, comparator in enumerate(names[1:]):
            item = actual["blocks"][str(length)]["comparisons"][comparator]
            ci = 2 * observed[j] - np.quantile(draws[j], [0.975, 0.025])
            p = (1 + sum(float(x - observed[j]) <= observed[j] for x in draws[j])) / 100
            errors = [abs(item["observed_delta"] - observed[j]),
                      max(abs(np.asarray(item["basic_ci_95"]) - ci)),
                      abs(item["bootstrap_mean"] - draws[j].mean()),
                      abs(item["centered_bootstrap_one_sided_p"] - p)]
            maximum = max(maximum, *errors)
            assert max(errors) < 2e-14, (length, comparator, errors)
    checks.append("independent scalar-loop replay of all three paired blocks, changing RW denominators, centered p and basic CI")

    # Strict bounds and undefined-ratio handling have no adaptive rescue.
    failures = []
    failures.append(expect_error(lambda: call({name: np.zeros((3, n, 20)) for name in names}, np.zeros_like(y)), "observed zero RW cell"))
    failures.append(expect_error(lambda: call(perfect, y, counts=(81, 100), replicates=1), "too few bootstrap draws"))
    failures.append(expect_error(lambda: call(perfect, y, counts=(81, 101)), "misaligned year length"))
    failures.append(expect_error(lambda: stats.circular_block_indices(40, 40, 99, np.random.default_rng(1)), "block equals stratum length"))
    missing = dict(perfect)
    missing["candidate"] = np.zeros((2, n, 20))
    failures.append(expect_error(lambda: call(missing, y), "mismatched declared seed count"))
    short_y = np.ones((121, 20))
    short = call({name: np.ones((3, 121, 20)) for name in names}, short_y, counts=(40, 81))
    assert short["primary_inference_status"] == "NOT_ESTIMABLE"
    assert short["blocks"]["20"]["status"] == "ESTIMATED"
    assert not short["both_primary_comparisons_pass"], "Do not promote sensitivity after primary is inestimable"
    checks.append("strict support/seed/ratio checks and no block-length rescue when primary support is short")

    sparse_y = np.zeros((n, 20))
    sparse_y[[0, 80]] = 1.0
    undefined = call({name: np.ones((3, n, 20)) for name in names}, sparse_y)
    assert undefined["primary_inference_status"] == "NOT_ESTIMABLE"
    assert undefined["blocks"]["40"]["zero_rw_denominator_draws"] > 0
    assert "comparisons" not in undefined["blocks"]["40"]
    checks.append("zero RW bootstrap denominator invalidates the registered ratio inference without epsilon or discarded draws")

    receipt = {
        "status": "PASS", "source_sha256": hashlib.sha256((HERE / "statistics.py").read_bytes()).hexdigest(),
        "check_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__, "synthetic_checks": checks, "expected_errors": failures,
        "independent_replay_maximum_absolute_difference": maximum,
        "formal_bootstrap_replicates_checked": 10000,
        "formal_bootstrap_seed_checked": 20261019,
        "market_file_reads": 0, "confirmation_file_reads": 0,
        "market_model_fits": 0, "optimizer_steps": 0,
    }
    (HERE / "STATISTICS_SYNTHETIC_RECEIPT.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
