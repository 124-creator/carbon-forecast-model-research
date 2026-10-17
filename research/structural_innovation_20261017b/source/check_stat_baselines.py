"""Pure synthetic checks for standard statistical baselines; no market I/O."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np

from stat_baselines import (
    HORIZON,
    RIDGE_ALPHAS,
    fit_markov_mle,
    fit_ridge,
    predict_markov_mle,
    predict_ridge,
    state_transition_table,
)


def run_checks() -> dict:
    rng = np.random.default_rng(20261017)
    checks: list[dict] = []
    comparisons = 0
    max_error = 0.0

    def close(name, actual, expected, atol=3e-11, rtol=3e-11):
        nonlocal comparisons, max_error
        actual = np.asarray(actual)
        expected = np.asarray(expected)
        if actual.shape != expected.shape:
            raise AssertionError((name, actual.shape, expected.shape))
        difference = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=name)
        comparisons += actual.size
        max_error = max(max_error, difference)
        checks.append({"name": name, "status": "PASS", "scalar_comparisons": actual.size,
                       "max_abs_error": difference})

    def truth(name, condition):
        nonlocal comparisons
        if not bool(condition):
            raise AssertionError(name)
        comparisons += 1
        checks.append({"name": name, "status": "PASS", "scalar_comparisons": 1})

    def rejects(name, function):
        try:
            function()
        except ValueError:
            truth(name, True)
        else:
            raise AssertionError(name + " did not reject invalid input")

    # Multivariate inputs and genuinely different horizons.
    n = 73
    x = rng.normal(size=(n, HORIZON, 2))
    x[:, 8, 1] = 7.0
    scales = np.exp(rng.uniform(-3, 3, size=n))
    latent = rng.normal(size=(40, HORIZON))
    y = x.reshape(n, -1) @ latent + np.linspace(-1, 2, HORIZON)
    train = {"x": x, "y": y, "scale": scales}
    model = fit_ridge(train, 0.1)

    # Independent objective gradient / normal-equation oracle.
    raw = x.reshape(n, -1)
    mu = np.sum(raw, axis=0) / n
    sd = np.sqrt(np.sum((raw - mu) ** 2, axis=0) / n)
    sd[sd < 1e-6] = 1e-6
    design = np.column_stack((np.ones(n), (raw - mu) / sd))
    weights = scales ** 2 / np.mean(scales ** 2)
    residual = design @ model.coefficients - y
    gradient = design.T @ (weights[:, None] * residual) / n
    gradient[1:] += 0.1 * model.coefficients[1:]
    close("weighted_ridge_normal_equation_all_20_targets", gradient,
          np.zeros_like(gradient), atol=2e-10, rtol=0)
    close("unweighted_training_population_means", model.feature_mean, mu)
    close("unweighted_training_population_std_floor", model.feature_std, sd)
    close("predictor_matches_independent_design", predict_ridge(model, train),
          design @ model.coefficients)

    # Intercept-only closed form establishes that the intercept is not penalized.
    constant_x = np.full((6, HORIZON, 1), 3.0)
    constant_y = np.arange(120, dtype=float).reshape(6, HORIZON) / 10
    constant_scale = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    constant_train = {"x": constant_x, "y": constant_y, "scale": constant_scale}
    constant_model = fit_ridge(constant_train, 100.0)
    expected_intercept = np.average(constant_y, axis=0, weights=constant_scale ** 2)
    close("unpenalized_intercept_closed_form", constant_model.coefficients[0],
          expected_intercept)
    close("constant_features_have_zero_coefficients", constant_model.coefficients[1:],
          np.zeros((HORIZON, HORIZON)))
    truth("weighted_intercept_differs_from_unweighted",
          np.max(np.abs(expected_intercept - constant_y.mean(axis=0))) > 1)

    # Extreme scale units must neither overflow nor alter the relative-weight fit.
    unit_scales = np.linspace(0.5, 2.0, n)
    small_model = fit_ridge({**train, "scale": unit_scales * 1e-200}, 0.1)
    large_model = fit_ridge({**train, "scale": unit_scales * 1e200}, 0.1)
    normal_model = fit_ridge({**train, "scale": unit_scales}, 0.1)
    close("extreme_global_scale_units_small", small_model.coefficients,
          normal_model.coefficients)
    close("extreme_global_scale_units_large", large_model.coefficients,
          normal_model.coefficients)
    imbalance = fit_ridge(
        {**constant_train, "scale": np.array([1e-200] * 5 + [1e200])}, 0.1
    )
    close("extreme_scale_imbalance_finite_dominant_intercept",
          imbalance.coefficients[0], constant_y[-1])
    query = {"x": rng.normal(size=(7, HORIZON, 2))}
    prediction = predict_ridge(model, query)
    close("prediction_batch_partition_does_not_refit_normalization", prediction,
          np.concatenate([predict_ridge(model, {"x": query["x"][i:i+1]})
                          for i in range(7)]))
    close("ridge_empty_prediction_batch", predict_ridge(model, {"x": query["x"][:0]}),
          np.empty((0, HORIZON)))
    truth("ridge_model_arrays_are_read_only", all(
        not a.flags.writeable for a in
        (model.feature_mean, model.feature_std, model.coefficients)))
    for alpha in RIDGE_ALPHAS:
        candidate = fit_ridge(train, alpha)
        truth("fixed_alpha_finite_" + str(alpha),
              np.isfinite(predict_ridge(candidate, query)).all())

    # Count and pseudo-magnitude oracle uses an explicit hand-constructed table.
    state = np.array([0, 0, 0, 1, 1, 6, 6, 8])
    event = np.array([0, 1, 1, 2, 0, 2, 2, 0])
    first = np.array([0, -1, -3, 4, 0, 2, 8, 0], dtype=float)
    mle_train = {"state": state, "event": event,
                 "y": first[:, None] * np.arange(1, HORIZON + 1)}
    q, b = fit_markov_mle(mle_train)
    counts = np.array([
        [1, 2, 0], [1, 0, 1], [0, 0, 0],
        [0, 0, 0], [0, 0, 0], [0, 0, 0],
        [0, 0, 2], [0, 0, 0], [1, 0, 0],
    ], dtype=float)
    global_q = np.array([4, 3, 4], dtype=float) / 11
    expected_q = np.vstack([(row + global_q) / (sum(row) + 1) for row in counts])
    expected_b = np.tile(np.array([2, 14/3]), (9, 1))
    expected_b[1, 1] = 13/3
    expected_b[6, 1] = 44/9
    close("laplace_global_and_one_global_pseudo_count", q, expected_q)
    close("one_directional_global_pseudo_magnitude", b, expected_b)
    truth("all_fitted_event_probabilities_strictly_positive", np.all(q > 0))
    truth("all_fitted_magnitudes_above_floor", np.all(b > 1e-4))
    changed_later = dict(mle_train)
    changed_later["y"] = mle_train["y"].copy()
    changed_later["y"][:, 1:] = rng.normal(size=(8, HORIZON - 1)) * 100
    q_later, b_later = fit_markov_mle(changed_later)
    close("mle_uses_only_next_event_not_later_targets_q", q_later, q, atol=0, rtol=0)
    close("mle_uses_only_next_event_not_later_targets_b", b_later, b, atol=0, rtol=0)

    one_direction = {"state": np.array([0, 0]), "event": np.array([1, 1]),
                     "y": np.array([-1, -3])[:, None] * np.ones((2, HORIZON))}
    one_q, one_b = fit_markov_mle(one_direction)
    close("missing_global_direction_uses_nonzero_absolute_mean", one_b,
          np.full((9, 2), 2.0))
    truth("missing_global_direction_probability_is_not_zero", np.all(one_q[:, 2] > 0))
    zeros = {"state": np.full(11, 2), "event": np.zeros(11),
             "y": np.zeros((11, HORIZON))}
    zero_model = fit_markov_mle(zeros)
    close("all_constant_training_global_magnitude_fallback", zero_model[1],
          np.ones((9, 2)))
    close("all_constant_training_has_zero_exact_mean",
          predict_markov_mle(zero_model, {"state": np.arange(9)}),
          np.zeros((9, HORIZON)), atol=0, rtol=0)
    tiny = {"state": np.array([6]), "event": np.array([2]),
            "y": np.full((1, HORIZON), 1e-12)}
    truth("tiny_positive_magnitude_floor_is_strict",
          np.all(fit_markov_mle(tiny)[1] > 1e-4))

    # Raw-price zero tolerance must be invariant to the query scale. Dyadic
    # scales preserve the exact threshold/nextafter values through division
    # and multiplication, making the inclusive boundary test unambiguous.
    tolerance = 1e-10
    just_above = np.nextafter(tolerance, np.inf)
    raw = np.array([0, tolerance/2, -tolerance/2, tolerance, -tolerance,
                    just_above, -just_above])
    boundary_scales = np.repeat(np.array([2.0**-30, 2.0**30]), 7)
    boundary_first = np.tile(raw, 2) / boundary_scales
    boundary_events = np.tile(np.array([0, 0, 0, 0, 0, 2, 1]), 2)
    boundary_train = {
        "state": np.full(14, 2), "event": boundary_events,
        "scale": boundary_scales,
        "y": boundary_first[:, None] * np.ones((14, HORIZON)),
    }
    boundary_q, boundary_b = fit_markov_mle(boundary_train)
    boundary_global_q = np.array([11, 3, 3], dtype=float) / 17
    boundary_expected_q = np.tile(boundary_global_q, (9, 1))
    boundary_expected_q[2] = (np.array([10, 2, 2]) + boundary_global_q) / 15
    close("raw_zero_inclusive_boundary_at_two_extreme_scales", boundary_q,
          boundary_expected_q)
    expected_boundary_magnitude = just_above * (2.0**30 + 2.0**-30) / 2
    close("tiny_raw_zero_events_are_excluded_from_nonzero_magnitudes", boundary_b,
          np.full((9, 2), expected_boundary_magnitude))
    within_wrong = dict(boundary_train)
    within_wrong["event"] = boundary_events.copy()
    within_wrong["event"][1] = 2
    rejects("reject_nonzero_label_for_within_raw_tolerance",
            lambda: fit_markov_mle(within_wrong))
    beyond_wrong = dict(boundary_train)
    beyond_wrong["event"] = boundary_events.copy()
    beyond_wrong["event"][12] = 0
    rejects("reject_zero_label_just_above_raw_tolerance",
            lambda: fit_markov_mle(beyond_wrong))
    exact_legacy = {key: value for key, value in boundary_train.items()
                    if key != "scale"}
    rejects("no_scale_callers_retain_exact_zero_semantics",
            lambda: fit_markov_mle(exact_legacy))

    expected_transition = np.array([
        [1, 0, 6], [2, 0, 6], [2, 0, 6],
        [4, 0, 6], [5, 0, 6], [5, 0, 6],
        [7, 0, 6], [8, 0, 6], [8, 0, 6],
    ])
    close("entire_prefix_last_sign_and_zero_age_state_transitions",
          state_transition_table(), expected_transition, atol=0, rtol=0)

    # Independent finite-tree enumeration: never uses the baseline transition
    # helper or its Bellman recurrence.
    random_q = rng.uniform(0.1, 1, size=(9, 3))
    random_q /= random_q.sum(axis=1, keepdims=True)
    random_b = rng.uniform(0.1, 3, size=(9, 2))
    enumerated = np.zeros((9, 5))
    for start in range(9):
        paths = [(start, 1.0, 0.0)]
        for horizon in range(5):
            following = []
            for current, probability, cumulative in paths:
                for ev in range(3):
                    if ev == 0:
                        next_state = (current // 3) * 3 + min(current % 3 + 1, 2)
                        reward = 0.0
                    elif ev == 1:
                        next_state = 0
                        reward = -random_b[current, 0]
                    else:
                        next_state = 6
                        reward = random_b[current, 1]
                    branch_probability = probability * random_q[current, ev]
                    branch_reward = cumulative + reward
                    enumerated[start, horizon] += branch_probability * branch_reward
                    following.append((next_state, branch_probability, branch_reward))
            paths = following
    close("exact_markov_mean_matches_all_paths_h1_to_h5",
          predict_markov_mle((random_q, random_b), {"state": np.arange(9)})[:, :5],
          enumerated)

    # Twenty-step deterministic paths provide a separate full-horizon oracle.
    action = np.array([2, 0, 1, 0, 0, 2, 1, 0, 2])
    deterministic_q = np.eye(3)[action]
    deterministic_b = np.arange(1, 19, dtype=float).reshape(9, 2) / 4
    deterministic = np.zeros((9, HORIZON))
    for start in range(9):
        current, total = start, 0.0
        for horizon in range(HORIZON):
            ev = action[current]
            if ev == 1:
                total -= deterministic_b[current, 0]
                current = 0
            elif ev == 2:
                total += deterministic_b[current, 1]
                current = 6
            else:
                current = (current // 3) * 3 + min(current % 3 + 1, 2)
            deterministic[start, horizon] = total
    close("exact_markov_mean_matches_deterministic_all_20_steps",
          predict_markov_mle((deterministic_q, deterministic_b),
                             {"state": np.arange(9)}),
          deterministic, atol=0, rtol=0)
    context_batch = {"state": np.array([2, 2]), "scale": np.array([0.01, 1e5]),
                     "x": rng.normal(size=(2, HORIZON, 9))}
    same_state = predict_markov_mle((random_q, random_b), context_batch)
    close("markov_normalized_mean_does_not_apply_future_scales",
          same_state[0], same_state[1], atol=0, rtol=0)
    close("markov_empty_prediction_batch",
          predict_markov_mle((q, b), {"state": np.array([], dtype=int)}),
          np.empty((0, HORIZON)))

    rejects("reject_nonpositive_alpha", lambda: fit_ridge(train, 0))
    rejects("reject_zero_training_scale",
            lambda: fit_ridge({**train, "scale": np.zeros(n)}, 0.1))
    rejects("reject_mismatched_prediction_channels",
            lambda: predict_ridge(model, {"x": np.zeros((1, HORIZON, 1))}))
    rejects("reject_wrong_event_sign",
            lambda: fit_markov_mle({**mle_train, "event": np.zeros(8)}))
    rejects("reject_invalid_state",
            lambda: predict_markov_mle((q, b), {"state": np.array([9])}))
    rejects("reject_fractional_state",
            lambda: predict_markov_mle((q, b), {"state": np.array([1.5])}))
    rejects("reject_non_probability_rows",
            lambda: predict_markov_mle((q * 2, b), {"state": np.array([0])}))
    truth("no_neural_model_import", "torch" not in sys.modules)

    return {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "PASS",
        "seed": 20261017,
        "check_count": len(checks),
        "scalar_comparisons": int(comparisons),
        "max_abs_error": max_error,
        "checks": checks,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "market_data_reads": 0,
        "confirmation_reads": 0,
        "real_market_fits": 0,
        "neural_model_imports": 0,
        "scope": "Synthetic array construction, baseline fits and independent algebra/path oracles only.",
        "files": [
            {"path": name, "sha256": hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()).hexdigest()}
            for name in ("stat_baselines.py", "check_stat_baselines.py")
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        help="Optional new receipt path; existing files are never overwritten.")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    receipt = run_checks()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    summary = {k: v for k, v in receipt.items() if k not in ("checks", "files")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

