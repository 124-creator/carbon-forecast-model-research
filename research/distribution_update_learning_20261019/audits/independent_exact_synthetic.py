#!/usr/bin/env python3
"""Independent, finite, atomic CRPS audit. Never reads market data.

The only stochastic operation is multinomial sampling of the four fully enumerated
two-step paths. Every rule sees the same calibration sample in each replicate.
Forecast CRPS and population risk are implemented here, independently of the parent
runner. The control coefficient grid is fixed, not estimated on calibration data.
"""
from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction as Q
from itertools import product
from pathlib import Path

import numpy as np


SEED = 20261019
REPETITIONS = 1000
SAMPLE_SIZES = (64, 256, 1024, 4096)
EPSILONS = (Q(0), Q(-1, 20), Q(-3, 20), Q(3, 20))
P0, P1 = Q(2, 5), Q(3, 5)
LAMBDAS = (Q(0), Q(1, 10))
TOTAL_DELTA = 0.05
SUPPORT = (Q(-2), Q(0), Q(2))
PATHS = tuple(product((-1, 1), repeat=2))
TOLERANCE = 1e-12
OUT = Path(__file__).resolve().parent


def probabilities(p: Q) -> tuple[Q, Q, Q]:
    return (p / 2, 1 - p, p / 2)


def pair_distance(weights_a, weights_b) -> Q:
    return sum(
        wa * wb * abs(xa - xb)
        for xa, wa in zip(SUPPORT, weights_a)
        for xb, wb in zip(SUPPORT, weights_b)
    )


def crps_atoms_fraction(p: Q, observation: Q) -> Q:
    weights = probabilities(p)
    return sum(w * abs(x - observation) for x, w in zip(SUPPORT, weights)) - (
        pair_distance(weights, weights) / 2
    )


def exact_quadratic_and_labels():
    w0, w1 = probabilities(P0), probabilities(P1)
    a = pair_distance(w0, w1) - pair_distance(w0, w0) / 2 - pair_distance(w1, w1) / 2
    b_values = tuple(
        (a + crps_atoms_fraction(P0, Q(r1 + r2)) - crps_atoms_fraction(P1, Q(r1 + r2))) / 2
        for r1, r2 in PATHS
    )
    c_values = tuple(Q(r1 * r2) for r1, r2 in PATHS)
    assert a == Q(1, 25)
    assert all(b == Q(1, 50) + Q(1, 10) * c for b, c in zip(b_values, c_values))
    z_values = {str(lam): tuple(b - lam * c for b, c in zip(b_values, c_values)) for lam in LAMBDAS}
    assert set(z_values[str(Q(1, 10))]) == {Q(1, 50)}
    cdf0, cdf1, w1_distance = Q(0), Q(0), Q(0)
    for j in range(len(SUPPORT) - 1):
        cdf0 += w0[j]
        cdf1 += w1[j]
        w1_distance += abs(cdf1 - cdf0) * (SUPPORT[j + 1] - SUPPORT[j])
    assert w1_distance == Q(2, 5)
    return a, b_values, c_values, z_values, w1_distance


def float_crps_atoms(p: np.ndarray, observation: float) -> np.ndarray:
    """Vectorized direct atom formula, not the derived quadratic risk formula."""
    p = np.asarray(p, dtype=np.float64)
    weights = (p / 2, 1 - p, p / 2)
    score = np.zeros_like(p)
    for x, w in zip(SUPPORT, weights):
        score = score + w * abs(float(x) - observation)
    for x, w in zip(SUPPORT, weights):
        for z, v in zip(SUPPORT, weights):
            score = score - 0.5 * w * v * abs(float(x - z))
    return score


def population_risk(alpha: np.ndarray, path_probs: np.ndarray) -> np.ndarray:
    p = float(P0) + np.asarray(alpha) * float(P1 - P0)
    risk = np.zeros_like(p)
    for (r1, r2), probability in zip(PATHS, path_probs):
        risk = risk + probability * float_crps_atoms(p, float(r1 + r2))
    return risk


def sample_statistics(counts: np.ndarray, values, n: int):
    """Exact sufficient statistics of all n observations in each calibration draw."""
    # Equality is established using rational values over the complete prior support.
    # This is never inferred from the observed calibration minimum/maximum.
    if len(set(values)) == 1:
        return np.full(counts.shape[0], float(values[0])), np.zeros(counts.shape[0])
    x = np.asarray([float(v) for v in values])
    mean = (counts @ x) / n
    variance = np.sum(counts * (x[None, :] - mean[:, None]) ** 2, axis=1) / (n - 1)
    return mean, variance


def empirical_bernstein(sample_variance: np.ndarray, n: int, prior_range: float, eta: float):
    """Maurer--Pontil 2009 Theorem 4, rescaled to an a-priori bounded interval."""
    assert n >= 2 and prior_range >= 0 and 0 < eta < 1
    if prior_range == 0:
        assert np.max(sample_variance) == 0
        return np.zeros_like(sample_variance)
    logterm = math.log(2 / eta)
    return np.sqrt(2 * sample_variance * logterm / n) + 7 * prior_range * logterm / (3 * (n - 1))


def interval95(successes: int, n: int):
    """Wilson interval for the repeated-calibration violation frequency, descriptive."""
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def summarize(alpha, risk, baseline_risk, selected_lambda=None, lower=None, true_b=None):
    change = risk - baseline_risk
    violations = int(np.count_nonzero(change > TOLERANCE))
    result = {
        "population_risk_mean": float(np.mean(risk)),
        "population_risk_sd_across_calibrations": float(np.std(risk, ddof=1)),
        "population_risk_mean_mcse": float(np.std(risk, ddof=1) / math.sqrt(REPETITIONS)),
        "risk_change_mean": float(np.mean(change)),
        "risk_change_q05": float(np.quantile(change, 0.05)),
        "risk_change_q95": float(np.quantile(change, 0.95)),
        "risk_change_max": float(np.max(change)),
        "violation_count": violations,
        "violation_rate": violations / REPETITIONS,
        "violation_rate_wilson95": interval95(violations, REPETITIONS),
        "update_count": int(np.count_nonzero(alpha > TOLERANCE)),
        "update_rate": float(np.mean(alpha > TOLERANCE)),
        "alpha_mean": float(np.mean(alpha)),
        "alpha_sd": float(np.std(alpha, ddof=1)),
        "alpha_q05": float(np.quantile(alpha, 0.05)),
        "alpha_median": float(np.median(alpha)),
        "alpha_q95": float(np.quantile(alpha, 0.95)),
    }
    if selected_lambda is not None:
        result["selected_lambda_0_1_rate"] = float(np.mean(selected_lambda == 0.1))
    if lower is not None:
        result["B_lower_mean"] = float(np.mean(lower))
        result["B_lower_exceeds_truth_count"] = int(np.count_nonzero(lower > true_b + TOLERANCE))
    return result


def normal_abs_expectation(mean: float, sd: float) -> float:
    if sd == 0:
        return abs(mean)
    z = mean / sd
    phi = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    two_cdf_minus_one = math.erf(z / math.sqrt(2))
    return 2 * sd * phi + mean * two_cdf_minus_one


def reflected_cap_numerical_audit():
    """Independent finite enumeration of the parent's capped odd control principle."""
    horizon = 5
    sd0, sd1 = 0.8 * math.sqrt(horizon), 1.1 * math.sqrt(horizon)
    w1_cap = math.sqrt(2 / math.pi) * abs(sd1 - sd0)  # Exact W1 for centered Gaussians.
    cap = 0.2 * w1_cap  # Deliberate stress case so that the predictable clipping branch executes.

    def continuation_up_to_additive_constant(s, remaining_variance):
        return 0.5 * (
            normal_abs_expectation(s, math.sqrt(sd0 * sd0 + remaining_variance))
            - normal_abs_expectation(s, math.sqrt(sd1 * sd1 + remaining_variance))
        )

    def increment(s, r, remaining_variance):
        return 0.5 * (
            continuation_up_to_additive_constant(s + r, remaining_variance)
            - continuation_up_to_additive_constant(s - r, remaining_variance)
        )

    maximum_oddness_error = 0.0
    maximum_budget_excess = 0.0
    summaries = []
    for imbalance in (0.0, 0.1):
        mean_control = 0.0
        mean_total_amplitude = 0.0
        probability_sum = 0.0
        clipped_paths = 0
        for signs in product((-1, 1), repeat=horizon):
            probability = math.prod((1 + imbalance * sign) / 2 for sign in signs)
            probability_sum += probability
            s, used, control = 0.0, 0.0, 0.0
            did_clip = False
            for j, sign in enumerate(signs):
                # The amplitude depends on the realized past: leverage is retained.
                amplitude = 1.0 + 0.7 * (s < 0)
                r = amplitude * sign
                remaining_variance = float(horizon - j - 1) * 1.3
                m = increment(s, r, remaining_variance)
                opposite = increment(s, -r, remaining_variance)
                maximum_oddness_error = max(maximum_oddness_error, abs(m + opposite))
                remaining_cap = max(0.0, cap - used)
                clipped = min(remaining_cap, max(-remaining_cap, m))
                did_clip |= abs(clipped - m) > TOLERANCE
                used += abs(clipped)
                control += clipped
                s += r
            maximum_budget_excess = max(maximum_budget_excess, used - cap, abs(control) - cap)
            mean_control += probability * control
            mean_total_amplitude += probability * used
            clipped_paths += did_clip
        assert abs(probability_sum - 1) < TOLERANCE
        assert abs(mean_control) <= imbalance * mean_total_amplitude + TOLERANCE
        assert abs(mean_control) <= imbalance * cap + TOLERANCE
        assert clipped_paths > 0
        summaries.append({
            "conditional_sign_imbalance": imbalance,
            "E_control": mean_control,
            "E_total_amplitude": mean_total_amplitude,
            "bias_upper_using_E_amplitude": imbalance * mean_total_amplitude,
            "bias_upper_using_cap": imbalance * cap,
            "cap_active_path_count": clipped_paths,
            "enumerated_path_count": 2 ** horizon,
        })
    first_step_values = [increment(0.0, r, 0.0) for r in (-4.0, -1.0, 0.0, 1.0, 4.0)]
    assert max(map(abs, first_step_values)) == 0
    assert maximum_oddness_error < TOLERANCE
    assert maximum_budget_excess < TOLERANCE
    tv_delta, tv_horizon, tv_cap = 0.05, 5, 1.0
    return {
        "horizon": horizon,
        "exact_W1": w1_cap,
        "cap_fraction_for_clipping_stress_check": 0.2,
        "cap": cap,
        "maximum_oddness_error": maximum_oddness_error,
        "maximum_amplitude_budget_excess": maximum_budget_excess,
        "h1_centered_scale_control_max_abs": max(map(abs, first_step_values)),
        "leverage_enumerations": summaries,
        "ordinary_reflection_tv_counterexample": {
            "definition": "At each step r=+1 with probability delta, otherwise r=0; spend remaining cap on first nonzero odd-control event.",
            "per_step_reflection_TV": tv_delta,
            "horizon": tv_horizon,
            "cap": tv_cap,
            "E_control": tv_cap * (1 - (1 - tv_delta) ** tv_horizon),
            "invalid_claim_delta_times_cap": tv_delta * tv_cap,
            "sign_imbalance_at_nonzero_magnitude": 1.0,
        },
    }


def main():
    a, b_values, c_values, z_values, w1_distance = exact_quadratic_and_labels()
    raw_min, raw_max = min(b_values), max(b_values)
    raw_range = raw_max - raw_min
    # Generic range deliberately does not use algebraic cancellation of B-lambda*C.
    # It uses the prior full-support B interval and |C|<=1, never the sample range.
    ranges = {}
    for lam in LAMBDAS:
        z = z_values[str(lam)]
        ranges[str(lam)] = {
            "exact_complete_support_min": str(min(z)),
            "exact_complete_support_max": str(max(z)),
            "exact_range": float(max(z) - min(z)),
            "generic_min": str(raw_min - abs(lam)),
            "generic_max": str(raw_max + abs(lam)),
            "generic_range": float(raw_range + 2 * abs(lam)),
            "looser_W1_range_not_used": 2 * float(w1_distance + abs(lam)),
        }
    rng = np.random.default_rng(SEED)
    result_rows = []
    sample_diagnostics = []
    numeric_errors = []
    for eps in EPSILONS:
        p_align = Q(1, 2) + eps
        path_probs_q = tuple((p_align if r1 == r2 else 1 - p_align) / 2 for r1, r2 in PATHS)
        path_probs = np.asarray([float(p) for p in path_probs_q])
        true_b = sum(p * b for p, b in zip(path_probs_q, b_values))
        true_c = sum(p * c for p, c in zip(path_probs_q, c_values))
        c_bias_budget = 2 * abs(eps)
        assert abs(true_c) <= c_bias_budget
        oracle_alpha = float(min(Q(1), max(Q(0), true_b / a)))
        baseline_risk = float(population_risk(np.asarray(0.0), path_probs))
        oracle_risk = float(population_risk(np.asarray(oracle_alpha), path_probs))
        for n in SAMPLE_SIZES:
            counts = rng.multinomial(n, path_probs, size=REPETITIONS)
            assert np.all(np.sum(counts, axis=1) == n)
            label_stats = {str(lam): sample_statistics(counts, z_values[str(lam)], n) for lam in LAMBDAS}
            b_mean, b_variance = label_stats[str(Q(0))]
            c_mean, c_variance = sample_statistics(counts, c_values, n)
            cv_mean, cv_variance = label_stats[str(Q(1, 10))]
            sample_diagnostics.append({
                "epsilon": float(eps),
                "n": n,
                "path_count_sha256": hashlib.sha256(np.ascontiguousarray(counts).tobytes()).hexdigest(),
                "true_E_C": float(true_c),
                "mean_C_across_calibrations": float(np.mean(c_mean)),
                "variance_of_mean_C_across_calibrations": float(np.var(c_mean, ddof=1)),
                "theory_variance_of_mean_C": float((1 - true_c * true_c) / n),
                "mean_within_calibration_C_sample_variance": float(np.mean(c_variance)),
                "true_E_B": float(true_b),
                "mean_B_across_calibrations": float(np.mean(b_mean)),
                "variance_of_mean_B_across_calibrations": float(np.var(b_mean, ddof=1)),
                "mean_within_calibration_B_sample_variance": float(np.mean(b_variance)),
                "corrected_B_mean": float(np.mean(cv_mean)),
                "corrected_B_within_sample_variance_max": float(np.max(cv_variance)),
                "corrected_B_mean_variance_across_calibrations": float(np.var(cv_mean, ddof=1)),
            })
            for confidence_scheme in ("equal_total_delta", "equal_per_bound_eta"):
                raw_eta = TOTAL_DELTA if confidence_scheme == "equal_total_delta" else TOTAL_DELTA / len(LAMBDAS)
                cv_eta = TOTAL_DELTA / len(LAMBDAS)
                raw_lower = b_mean - empirical_bernstein(b_variance, n, float(raw_range), raw_eta)
                policies = {
                    "raw_safe": {
                        "alpha": np.clip(raw_lower / float(a), 0, 1),
                        "lower": raw_lower,
                        "lambda": np.zeros(REPETITIONS),
                        "eta_per_bound": raw_eta,
                        "nominal_total_error": raw_eta,
                        "budget_valid": True,
                    },
                }
                for range_name in ("exact_complete_support", "generic_envelope"):
                    for budget_name, budget in (("correct_budget", c_bias_budget), ("zero_budget", Q(0))):
                        lower_columns = []
                        for lam in LAMBDAS:
                            mean, variance = label_stats[str(lam)]
                            width = ranges[str(lam)]["exact_range" if range_name == "exact_complete_support" else "generic_range"]
                            radius = empirical_bernstein(variance, n, width, cv_eta)
                            lower_columns.append(mean - radius - float(abs(lam) * budget))
                        all_lower = np.stack(lower_columns, axis=1)
                        chosen = np.argmax(all_lower, axis=1)  # Ordered ties prefer lambda=0.
                        best_lower = all_lower[np.arange(REPETITIONS), chosen]
                        name = f"CV_safe_{budget_name}" + ("_generic_range" if range_name == "generic_envelope" else "")
                        policies[name] = {
                            "alpha": np.clip(best_lower / float(a), 0, 1),
                            "lower": best_lower,
                            "lambda": np.asarray([float(v) for v in LAMBDAS])[chosen],
                            "eta_per_bound": cv_eta,
                            "nominal_total_error": TOTAL_DELTA,
                            "budget_valid": abs(true_c) <= budget,
                        }
                policies["ordinary_empirical_mixture"] = {
                    "alpha": np.clip(b_mean / float(a), 0, 1),
                    "budget_valid": None,
                }
                policies["oracle"] = {
                    "alpha": np.full(REPETITIONS, oracle_alpha),
                    "budget_valid": None,
                }
                raw_risks = population_risk(policies["raw_safe"]["alpha"], path_probs)
                for name, policy in policies.items():
                    alpha = policy["alpha"]
                    assert np.all((alpha >= 0) & (alpha <= 1))
                    risks = population_risk(alpha, path_probs)
                    quadratic_change = float(a) * alpha ** 2 - 2 * float(true_b) * alpha
                    numeric_errors.append(float(np.max(np.abs(risks - baseline_risk - quadratic_change))))
                    assert np.min(risks) >= oracle_risk - TOLERANCE
                    stats = summarize(
                        alpha, risks, baseline_risk,
                        selected_lambda=policy.get("lambda"),
                        lower=policy.get("lower"), true_b=float(true_b),
                    )
                    paired = risks - raw_risks
                    stats.update({
                        "paired_risk_difference_vs_raw_safe_mean": float(np.mean(paired)),
                        "paired_risk_difference_vs_raw_safe_mcse": float(np.std(paired, ddof=1) / math.sqrt(REPETITIONS)),
                    })
                    result_rows.append({
                        "epsilon": float(eps), "n": n,
                        "confidence_scheme": confidence_scheme, "method": name,
                        "true_B": float(true_b), "known_A": float(a),
                        "baseline_population_risk": baseline_risk,
                        "oracle_population_risk": oracle_risk,
                        "control_mean_budget": float(c_bias_budget) if "correct_budget" in name else 0.0,
                        "assumed_control_mean_budget_valid": policy["budget_valid"],
                        "eta_per_bound": policy.get("eta_per_bound"),
                        "nominal_total_error": policy.get("nominal_total_error"),
                        **stats,
                    })
    assert max(numeric_errors) < TOLERANCE
    reflected_audit = reflected_cap_numerical_audit()
    result = {
        "metadata": {
            "task": "independent finite atomic CRPS synthetic audit",
            "seed": SEED, "repetitions_per_case": REPETITIONS,
            "n_values": list(SAMPLE_SIZES), "epsilon_values": [float(v) for v in EPSILONS],
            "calibration_sample_sets": len(EPSILONS) * len(SAMPLE_SIZES) * REPETITIONS,
            "forecast_predictor_fits": 0, "market_rows_read": 0,
            "lambdas": [float(v) for v in LAMBDAS], "total_delta": TOTAL_DELTA,
            "sampling": "Independent multinomial counts over the four prior-enumerated paths; exact sufficient statistics; shared samples across methods and confidence schemes.",
            "risk": "Exact population CRPS via atom absolute distances, averaged over the four true paths; no sampled test set.",
            "known_A_reason": "A is the context-free exact integral of the two fixed prior forecast CDFs; no empirical estimate is substituted for a population expectation.",
            "equal_total_delta": "raw one-sided eta=.05; CV two simultaneous one-sided bounds each eta=.025; A known exactly.",
            "equal_per_bound_eta": "raw eta=.025 and each CV eta=.025; raw total error .025, CV union total error .05.",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "numpy_version": np.__version__,
            "outputs_contain_no_market_data": True,
        },
        "exact_support": {
            "paths": [list(p) for p in PATHS],
            "p0": str(P0), "p1": str(P1), "A": str(a),
            "B_values": [str(v) for v in b_values], "C_values": [str(v) for v in c_values],
            "identity": "B = 1/50 + C/10, proved exactly with Fraction atomic CRPS",
            "W1": str(w1_distance), "ranges": ranges,
        },
        "numerical_audit": {
            "population_risk_quadratic_max_abs_error": max(numeric_errors),
            "all_assertions_passed": True,
        },
        "calibration_diagnostics": sample_diagnostics,
        "results": result_rows,
        "reflected_cap_audit": reflected_audit,
        "scope": [
            "Valid iid synthetic calibration theorem only; no nonstationary future-market guarantee.",
            "epsilon-derived budget is declared from the known DGP, never estimated or certified by a symmetry test.",
            "Zero-budget methods intentionally violate the asserted weak-symmetry budget when epsilon is nonzero.",
            "The exact zero range is an algebraic identity on the full specified prior support, not a sample-observed range.",
            "These fixed finite cases validate a learning-update mechanism; they do not establish carbon-price accuracy or general method novelty.",
        ],
    }
    json_path = OUT / "INDEPENDENT_EXACT_SYNTHETIC.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = render_markdown(result)
    md_path = OUT / "INDEPENDENT_EXACT_SYNTHETIC_ZH.md"
    md_path.write_text(markdown, encoding="utf-8")
    print(json.dumps({
        "json": str(json_path), "report": str(md_path),
        "result_rows": len(result_rows), "calibration_sample_sets": result["metadata"]["calibration_sample_sets"],
        "quadratic_error": max(numeric_errors), "assertions_passed": True,
    }, ensure_ascii=False))


def render_markdown(result):
    rows = result["results"]
    lines = [
        "# 独立四路径 CRPS 有限合成审计", "",
        "日期：2026-10-19。仅新合成数据；未读取市场。独立实现原子 CRPS，与通用模型/runner 无代码复用。", "",
        "## 1. 固定任务和合法范围", "",
        f"固定 seed={SEED}，epsilon={list(map(float, EPSILONS))}，n={list(SAMPLE_SIZES)}，每格 {REPETITIONS} 个独立校准重复，共 16,000 份校准样本。每次重复的所有方法共享相同四路径计数；两个置信口径复用这同一组样本，不另造有利随机数。",
        "固定 F_p 在 ±2 各放 p/2、0 处放 1−p；p0=.4、p1=.6。a=.04 是两个无条件固定 CDF 的可解析常量，因此本合成可视为已知，不把估计 a 冒充真期望。", "",
        "用 Fraction 原子绝对距离公式严格证明 B=1/50+C/10，C=r1*r2；B 的先验范围为[−.08,.12]，range=.2。λ=.1 时 B−λC 恒=.02，精确完整支持范围为0。generic-range 对照只使用先验边际包络[−.08−.1,.12+.1]，range=.4，故不利用代数抵消；绝不使用样本观察到的最小最大值。更松的 W1 包络也写入 JSON，但未用来制造第三套结果。", "",
        "对 C 的正确预算是 |EC|≤2|epsilon|，其来源是已知合成 DGP。λ=.1 的实际风险偏差惩罚为.2|epsilon|。该预算未从校准样本估计。", "",
        "主口径：raw 单侧 eta=.05；CV 的 λ∈{0,.1} 两个同时界各 eta=.025，总δ=.05。另给同单界 eta=.025 的公平敏感性口径，此时 raw 总δ=.025、CV 总δ=.05。A 无估计误差。", "",
        "校准使用 Maurer–Pontil (2009) Theorem 4：u=sqrt(2 s² log(2/eta)/n)+7 range log(2/eta)/(3(n−1))；CV取 L=max_lambda[mean(B−lambda*C)−u−|lambda|budget]，alpha=clip(L/a,0,1)。ordinary empirical mixture 直接用原始 mean(B)/a，oracle 用真 B/a。", "",
        "## 2. 主口径结果", "",
        "下表 risk change 为新分布的真实总体 CRPS 减基准，负值更好；违规表示真实风险增加超过1e−12。风险用四路径分布精确求和；不是从同一个校准集或随机测试集估计。", "",
        "| epsilon | n | 方法 | 平均真实风险差 | 违规率 | 更新率 | 平均 alpha | 选CV比例 |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "raw_safe": "raw-safe",
        "CV_safe_correct_budget": "CV-correct/exact",
        "CV_safe_zero_budget": "CV-zero/exact",
        "CV_safe_correct_budget_generic_range": "CV-correct/generic",
        "CV_safe_zero_budget_generic_range": "CV-zero/generic",
        "ordinary_empirical_mixture": "empirical-mixture",
        "oracle": "oracle",
    }
    for row in rows:
        if row["confidence_scheme"] != "equal_total_delta":
            continue
        selected = row.get("selected_lambda_0_1_rate")
        selected_text = "—" if selected is None else f"{selected:.1%}"
        lines.append(f"| {row['epsilon']:+.2f} | {row['n']} | {labels[row['method']]} | {row['risk_change_mean']:+.6f} | {row['violation_rate']:.1%} | {row['update_rate']:.1%} | {row['alpha_mean']:.4f} | {selected_text} |")
    lines += ["", "## 3. 相同单界 eta 的公平比较", "",
        "CV在这一口径与主口径相同；raw使用eta=.025，以显示CV相对raw的差异是否来自置信预算分配。完整置信半宽、均值、误差与所有方法结果均在 JSON。", "",
        "| epsilon | n | raw风险差 | CV-correct/exact风险差 | CV-correct/generic风险差 | generic−raw配对风险差 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    indexed = {(r["epsilon"], r["n"], r["confidence_scheme"], r["method"]): r for r in rows}
    for eps in map(float, EPSILONS):
        for n in SAMPLE_SIZES:
            get = lambda method: indexed[(eps, n, "equal_per_bound_eta", method)]
            raw, exact, generic = get("raw_safe"), get("CV_safe_correct_budget"), get("CV_safe_correct_budget_generic_range")
            lines.append(f"| {eps:+.2f} | {n} | {raw['risk_change_mean']:+.6f} | {exact['risk_change_mean']:+.6f} | {generic['risk_change_mean']:+.6f} | {generic['paired_risk_difference_vs_raw_safe_mean']:+.6f} |")
    lines += ["", "## 4. 控制均值方差", "",
        "理论 Var(mean C)=(1−4 epsilon²)/n。控制后的 B−.1C 标签在整个先验支持上恒定，两个范围设计的样本方差都为0；generic版仍须支付其合法非零range项。因此方差为0不意味着任意取range=0。", "",
        "| epsilon | n | mean(C)重复均值 | Var(mean C)重复估计 | 理论Var(mean C) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for d in result["calibration_diagnostics"]:
        lines.append(f"| {d['epsilon']:+.2f} | {d['n']} | {d['mean_C_across_calibrations']:+.6f} | {d['variance_of_mean_C_across_calibrations']:.6g} | {d['theory_variance_of_mean_C']:.6g} |")
    valid_rows = [r for r in rows if r["confidence_scheme"] == "equal_total_delta" and r["method"] in ("raw_safe", "CV_safe_correct_budget", "CV_safe_correct_budget_generic_range")]
    zero_bad = indexed[(-0.15, 4096, "equal_total_delta", "CV_safe_zero_budget")]
    sym_exact = indexed[(0.0, 64, "equal_total_delta", "CV_safe_correct_budget")]
    sym_generic_small = indexed[(0.0, 64, "equal_total_delta", "CV_safe_correct_budget_generic_range")]
    sym_generic_large = indexed[(0.0, 4096, "equal_total_delta", "CV_safe_correct_budget_generic_range")]
    lines += ["", "## 5. 结果解释", "",
        f"正确预算的raw/CV规则在主口径所有格子的最大观测违规率为 {max(r['violation_rate'] for r in valid_rows):.1%}。这1000次重复的频率不是定理证明，也不是非平稳市场保证；JSON给出每格违规计数和Wilson区间。",
        f"精确对称、n=64时，利用严格恒等范围的CV真实风险差为 {sym_exact['risk_change_mean']:+.6f}，更新率 {sym_exact['update_rate']:.1%}。相同情形generic范围风险差为 {sym_generic_small['risk_change_mean']:+.6f}；n=4096时generic为 {sym_generic_large['risk_change_mean']:+.6f}。请分别理解精确先验恒等式的价值与一般有限range成本，不能把前者当作神经控制必然具有的性质。",
        f"epsilon=−.15、n=4096时，忽略偏差预算的精确范围CV违规率为 {zero_bad['violation_rate']:.1%}，平均真实风险差为 {zero_bad['risk_change_mean']:+.6f}；这是有意暴露的错误假设反例，不能删去。",
        "generic-CV并未在所有样本规模上优于无保护的经验混合。例如epsilon=0、n=1024时，generic-CV风险差约−.009602，经验混合约−.009765；前者付出了风险保护成本。不能只展示对raw-safe的优势就声称全面精度创新。",
        "本试验只学习两个固定分布之间的混合步，没有拟合基模型或控制网络。4条路径高度特殊、完整概率支持已知，不能将此机制正控制当作碳价格预测精度证据。", "",
        "## 6. 通用反射控制与可预测幅度cap审查", "",
        "令G_j(s)=E_{Z~N(0,v_rem)} B(s+Z)，M_j=.5[G_j(Sprev+r_j)−G_j(Sprev−r_j)]。只要参考v_rem、F0/F1在当前r_j出现前确定，M_j对当前增量是奇函数。G无需是真实未来条件期望；参考模型错误主要影响相关性和效率，不破坏奇性。Gaussian-mixture F下可解析卷积：CRPS的观测绝对正态项把方差加v_rem，预测样本间项不变；M差分时常数抵消。", "",
        "普通条件反射TV≤delta给|E M_j|≤delta sup|M_j|。若逐步相加，只能先得到H倍的最坏情形界。更强的条件符号假设 |E[sign(r_j)||r_j|,past]|≤delta 才给 |EC|≤delta E sum|M_j|。两种假设不能混用。", "",
        "追加的可预测cap成立：remaining=K−sum_{l<j}|clipped_M_l|是当前增量以前已知量，clip(M_j,−remaining,+remaining)保持奇性；归纳得sum|clipped_M_j|≤K、|C|≤K。强符号假设下|EC|≤delta E sum|clipped_M_j|≤delta E K≤delta Kmax，不付H。若K=K(X)随原点变化，要用E K的上界或Kmax，不能用未保护的样本均值。此处cap是控制幅度总额，与概率分布的total variation不是同一概念。", "",
        "若只假定普通reflection-TV≤delta，反例为每步r=+1概率delta，否则r=0，首次非零奇控制消耗剩余全部K。此时EC=K[1−(1−delta)^H]，可远高于delta K；给定非零幅度的符号偏差为1，故明确违反强符号假设。", "",
        "范围还能实质改进：B'(y)=−(F1(y)−F0(y))，故osc(B)≤∫|F1−F0|=W1，不必从|B|≤W1付出2W1的范围。F0/F1同中心对称时B也对称，所以更紧有osc(B)≤W1/2。卷积不增加osc，M又是G两点之差的一半，因此sup|M|≤.5 osc(G)≤.5 osc(B)，同中心情形至多W1/4。预算后generic范围可用R_B+2|lambda|Kmax，其中R_B=W1或在同中心对称情形用W1/2；也可在合成完整支持上严格证明更紧的联合范围。不能用只在样本中看见的更小范围。", "",
        "本四路径例子与通用反射控制直接对应：第一步M1=0；第二步G2=B、M2=.1*r1*r2。所以本脚本C=r1*r2、lambda=.1等价于反射控制C_ref=M2、lambda_ref=1；不是与拟议operator无关的另一个任意控制。", "",
        "若使用delta E sum|clipped_M|代替delta Kmax，前者必须有合法统计上界并支付联合误差预算；它是标签相关的未来幅度总量，不可在部署时当原点输入。Kmax版本最容易先保证正确。", "",
        "H=1、同零中心对称F0/F1、零均值Gaussian参考下B和G为偶函数，M1恒0。若F0/F1含不同均值，该负控制结论不再适用。", "",
        "独立附加数值枚举使用H=5、幅度依赖历史符号的32条路径，检查oddness、cap、条件对称及delta=.1强符号预算。为确实触发clip分支，检查使用K=.2×W1的较紧预算，这是数值压力检查而非真实模型超参选择；这些不是新的模型训练。", "",
        "```json", json.dumps(result["reflected_cap_audit"], ensure_ascii=False, indent=2), "```", "",
        "## 7. 验证与来源", "",
        f"原子总体CRPS与二次风险恒等式最大绝对误差：{result['numerical_audit']['population_risk_quadratic_max_abs_error']:.3g}。所有数值断言通过。代码SHA256：{result['metadata']['script_sha256']}。",
        "集中界原始来源：Maurer & Pontil (2009), Empirical Bernstein Bounds and Sample Variance Penalization, Theorem4 / Corollary5，https://arxiv.org/pdf/0907.3740 。",
        "风险控制+CV近邻：Almeida et al. (2025), High Probability Risk Control Under Covariate Shift, AppendixD，https://research.feedzai.com/oshuzuxa/2025/10/high_probability_risk_control_under_covariate_shift.pdf 。其分样与range成本亦可能抵消方差收益。这里的合成不建立通用新颖性。", "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
