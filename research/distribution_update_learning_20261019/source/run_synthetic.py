"""Fixed general-operator synthetic study. Does not read any market prices."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np

from scores import SQRT_2_PI, reflected_control, pair_terms, self_distance
from learning import bernstein_scalar


SCENARIOS = {
    "gaussian_h1_negative": {"h": 1, "kind": "gaussian", "sign_delta": 0.},
    "gaussian_h5": {"h": 5, "kind": "gaussian", "sign_delta": 0.},
    "gaussian_h20": {"h": 20, "kind": "gaussian", "sign_delta": 0.},
    "heavy_tail_h5": {"h": 5, "kind": "student5", "sign_delta": 0.},
    "leverage_h5": {"h": 5, "kind": "leverage", "sign_delta": 0.},
    "sign_bias_h5": {"h": 5, "kind": "sign_bias", "sign_delta": .3},
    "harmful_scale_h5": {"h": 5, "kind": "gaussian", "sign_delta": 0., "scale": .65},
    "deployment_shift_h5": {"h": 5, "kind": "gaussian", "sign_delta": 0., "test_scale": .45},
}
CALIBRATION_N = (128, 512, 2048)
REPETITIONS = 300
TEST_PATHS = 100000
SEED = 20261020
WEIGHTS = np.array([1.])
SIGMA0 = np.array([.8])
FACTOR = 1.5


def paths(rng, n, setting, test=False):
    h, kind = setting["h"], setting["kind"]
    scale = setting.get("test_scale", setting.get("scale", 1.)) if test else setting.get("scale", 1.)
    result, prefix = np.zeros((n, h)), np.zeros(n)
    for j in range(h):
        z = rng.standard_t(5, n) * np.sqrt(3 / 5) if kind == "student5" else rng.normal(size=n)
        if kind == "leverage":
            # Future amplitude depends on the preceding sign, not an independent
            # whole-path sign resampling. Every current innovation is symmetric.
            amplitude = np.sqrt(1 + .65 * np.tanh(prefix * np.sqrt(h / max(j, 1))))
            z *= amplitude
        elif kind == "sign_bias":
            imbalance = -.3 * np.tanh(prefix * np.sqrt(h / max(j, 1)))
            z = np.abs(z) * np.where(rng.random(n) < (1 + imbalance) / 2, 1., -1.)
        result[:, j] = z * scale / np.sqrt(h)
        prefix += result[:, j]
    return result


def score_paths(r):
    y = r.sum(axis=1)
    a, b, score0, score1 = pair_terms(y, WEIGHTS, SIGMA0, SIGMA0 * FACTOR)
    c, cap, spent = reflected_control(r, WEIGHTS, SIGMA0, FACTOR)
    cfull, capfull, spentfull = reflected_control(r, WEIGHTS, SIGMA0, FACTOR, cap_fraction=1.)
    cu, _, spentu = reflected_control(r, WEIGHTS, SIGMA0, FACTOR, capped=False)
    return a, b, score0, score1, c, cap, spent, cfull, capfull, spentfull, cu, spentu


def mean_se(values):
    x = np.asarray(values)
    return {"mean": float(x.mean()), "standard_error_over_independent_repetitions": float(x.std(ddof=1) / np.sqrt(len(x)))}


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    source = Path(__file__)
    freeze = {"timestamp_before_first_simulation_unix": started,
              "parameters": {"scenarios": SCENARIOS, "n": CALIBRATION_N, "repetitions": REPETITIONS, "test_paths": TEST_PATHS, "seed": SEED, "sigma0": SIGMA0.tolist(), "factor": FACTOR},
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, source.with_name("scores.py"), source.with_name("learning.py"))}}
    (output / "SYNTHETIC_PREFIT.json").write_text(json.dumps(freeze, indent=2))
    rows, scenario_receipts = [], []
    for scenario_number, (name, setting) in enumerate(SCENARIOS.items()):
        rng = np.random.default_rng(SEED + 100 * scenario_number)
        test = paths(rng, TEST_PATHS, setting, test=True)
        a, btest, s0test, _, ctest, captest, spenttest, _, _, _, _, _ = score_paths(test)
        aa, kk = float(a[0]), float(captest[0])
        b_population, b_mc_se = float(btest.mean()), float(btest.std(ddof=1) / np.sqrt(TEST_PATHS))
        base_risk, exact_population = float(s0test.mean()), False
        if setting["kind"] == "gaussian":
            true_sigma = setting.get("test_scale", setting.get("scale", 1.))
            expected0 = SQRT_2_PI * np.sqrt(.8 ** 2 + true_sigma ** 2) - .8 / np.sqrt(np.pi)
            expected1 = SQRT_2_PI * np.sqrt(1.2 ** 2 + true_sigma ** 2) - 1.2 / np.sqrt(np.pi)
            b_population = .5 * (aa + expected0 - expected1)
            base_risk, b_mc_se, exact_population = expected0, 0., True
        if setting["h"] == 1 and not np.all(np.abs(ctest) < 1e-13):
            raise AssertionError("One-step symmetric-scale negative control failed")
        bzero = float(pair_terms(np.array([0.]), WEIGHTS, SIGMA0, SIGMA0 * FACTOR)[1][0])
        binfinity = .5 * (aa + .5 * float(self_distance(WEIGHTS, SIGMA0 * FACTOR)) - .5 * float(self_distance(WEIGHTS, SIGMA0)))
        b_range = abs(binfinity - bzero)
        summaries = {key: [] for key in ("b", "c", "cfull", "cu")}
        for _ in range(6):
            # Fifty independent repetitions at once limits temporary memory.
            r = paths(rng, 50 * max(CALIBRATION_N), setting)
            values = score_paths(r)
            for key, index in (("b", 1), ("c", 4), ("cfull", 7), ("cu", 10)):
                summaries[key].append(values[index].reshape(50, max(CALIBRATION_N)))
        summaries = {k: np.concatenate(v, axis=0) for k, v in summaries.items()}
        receipt = {"scenario": name, "setting": setting, "A": aa, "B_test": b_population,
                   "B_test_MC_standard_error": b_mc_se, "population_analytic": exact_population,
                   "baseline_test_CRPS": base_risk, "b_proved_range": b_range, "cap": kk,
                   "test_mean_control": float(ctest.mean()), "test_control_mean_MC_SE": float(ctest.std(ddof=1) / np.sqrt(TEST_PATHS)),
                   "test_mean_spent": float(spenttest.mean()), "test_cap_exhausted_fraction": float(np.mean(spenttest >= kk - 1e-12)),
                   "test_variance_ratio": float(np.var(btest-ctest) / np.var(btest)),
                   "test_paths": TEST_PATHS, "calibration_paths": REPETITIONS * max(CALIBRATION_N)}
        scenario_receipts.append(receipt)
        for n in CALIBRATION_N:
            bb, cc, cf, cu = [summaries[k][:, :n] for k in ("b", "c", "cfull", "cu")]
            rules = {"baseline": (np.zeros(REPETITIONS), np.zeros(REPETITIONS)),
                     "proposal": (np.ones(REPETITIONS), np.zeros(REPETITIONS)),
                     "empirical_mixture": (np.clip(bb.mean(axis=-1) / aa, 0, 1), np.zeros(REPETITIONS)),
                     "oracle": (np.full(REPETITIONS, np.clip(b_population / aa, 0, 1)), np.zeros(REPETITIONS))}
            rules["raw_bernstein"] = bernstein_scalar(aa, bb, cc, kk, b_range, 0., lambdas=(0.,))
            rules["raw_bernstein_same_joint_budget"] = bernstein_scalar(aa, bb, cc, kk, b_range, 0., lambdas=(0.,), delta=.05/3)
            rules["path_bernstein_zero_budget"] = bernstein_scalar(aa, bb, cc, kk, b_range, 0.)
            rules["path_bernstein_valid_budget"] = bernstein_scalar(aa, bb, cc, kk, b_range, setting["sign_delta"])
            rules["path_bernstein_assumed005_budget"] = bernstein_scalar(aa, bb, cc, kk, b_range, .05)
            rules["path_fullcap_valid_budget"] = bernstein_scalar(aa, bb, cf, kk * 4, b_range, setting["sign_delta"])
            # Every untruncated reflection is bounded by W1/4, so sum <= h*K.
            rules["path_uncapped_valid_budget"] = bernstein_scalar(aa, bb, cu, kk * setting["h"], b_range, setting["sign_delta"])
            for method, (alpha, selected) in rules.items():
                risk_difference = aa * alpha ** 2 - 2 * b_population * alpha
                # For MC truth, also distinguish violations clear of truth MC noise.
                clear = risk_difference > np.maximum(1e-12, 3 * 2 * np.abs(alpha) * b_mc_se)
                rows.append({"scenario": name, "n": n, "method": method,
                             "repetitions": REPETITIONS, "mean_alpha": float(np.mean(alpha)),
                             "update_fraction": float(np.mean(alpha > 1e-10)),
                             "control_selected_fraction": float(np.mean(selected != 0)),
                             "risk_difference": mean_se(risk_difference),
                             "risk_ratio": float(np.mean(risk_difference) / base_risk + 1),
                             "observed_positive_risk_fraction": float(np.mean(risk_difference > 1e-12)),
                             "MC_clear_positive_risk_fraction": float(np.mean(clear)),
                             "population_analytic": exact_population,
                             "distribution_shift_outside_theorem": "test_scale" in setting})
        print(json.dumps({"scenario": name, "finished": True, "elapsed_s": round(time.time()-started, 1), "variance_ratio": receipt["test_variance_ratio"]}), flush=True)
    result = {"status": "completed", "elapsed_seconds": time.time()-started,
              "unique_calibration_repetitions": REPETITIONS * len(SCENARIOS),
              "nested_calibration_sizes_are_not_independent_new_datasets": True,
              "gradient_model_fits": 0,
              "scenarios": scenario_receipts, "results": rows, "prefit": freeze}
    (output / "GENERAL_SYNTHETIC.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
