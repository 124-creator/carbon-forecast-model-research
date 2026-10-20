"""Independent saved-trace audit, specified before the resumed formal run.

Does not import the production core or runner. Uses Fourier polynomial matrix
powers (not the production reward DP), direct likelihood filtering, erf-based
absolute moments, original-objective KKT conditions and a separate aggregator.
Validation selections are checked against every saved grid score; this script
does not independently solve all validation trajectories.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.special import erf

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (1, 5, 20)
V = np.array([.16, .64, .64, 2.56])
CENTER = np.eye(4) - np.ones((4, 4)) / 4
PI_COV = CENTER / 4
COMPETITORS = ["current_cdf", "future_variance", "probability_euclidean",
               "rotated_future_cdf", "chi_square_only", "crps_kl",
               "tempered_bayes", "contaminated_bayes"]
COUNTS = {}
MAXIMA = {}
TOLERANCES = {"score": 2e-9, "mass": 2e-10, "kkt": 2e-8,
              "posterior": 2e-12, "aggregate": 2e-9, "oracle_excess": 1e-10}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(name, value, limit):
    value = float(value)
    MAXIMA[name] = max(MAXIMA.get(name, 0.), value)
    if not np.isfinite(value) or value > limit:
        raise AssertionError(f"{name}: {value} > {limit}")


def compare(name, a, b, limit):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise AssertionError(f"{name} shape mismatch: {a.shape}, {b.shape}")
    COUNTS[name] = COUNTS.get(name, 0) + a.size
    record(name, np.max(np.abs(a - b), initial=0.), limit)


def transition(slow, fast):
    result = np.empty((4, 4))
    for i in range(4):
        for j in range(4):
            result[i, j] = ((1-slow)/2 + slow*(i//2 == j//2)) * (
                (1-fast)/2 + fast*(i % 2 == j % 2))
    return result


def fourier_law(p, h):
    length = 5*h+1
    roots = np.exp(-2j*np.pi*np.arange(length)/length)
    offsets = np.array([0, 1, 1, 5])
    values = np.stack([np.linalg.matrix_power(p*z**offsets[None, :], h).sum(axis=1)
                       for z in roots], axis=1)
    coefficients = np.fft.ifft(values, axis=1)
    record("fourier_imaginary", np.max(np.abs(coefficients.imag)), 2e-13)
    weights = coefficients.real
    record("fourier_negative_roundoff", max(0., -weights.min()), 2e-13)
    # The independent FFT introduces tiny signed roundoff at exact zero mass.
    # Preserve its magnitude in the receipt; this repair is not model output.
    weights = np.maximum(weights, 0.)
    record("fourier_mass", np.max(np.abs(weights.sum(axis=1)-1)), 2e-13)
    weights = weights / weights.sum(axis=1, keepdims=True)
    variances = .16 * (h+3*np.arange(length))/h
    return weights, variances


def binomial_masses(n, epsilon):
    return [(k, math.comb(n, k)*epsilon**k*(1-epsilon)**(n-k))
            for k in range(n+1)]


def abs_moment(y, variance):
    y = np.abs(np.asarray(y))
    sd = np.sqrt(variance)
    return sd*np.sqrt(2/np.pi)*np.exp(-.5*(y/sd)**2) + y*erf(y/(sd*np.sqrt(2)))


def distance(va, vb):
    return np.sqrt(2/np.pi)*np.sqrt(va[:, None]+vb[None, :])


def normalize_metric(g):
    g = (g+g.T)/2
    scale = np.trace(PI_COV@g)
    if scale <= 1e-14*max(1., np.linalg.norm(g, 2)):
        return np.zeros_like(g)
    return g/scale


def reference():
    p = transition(.97, .1)
    laws = {h: fourier_law(p, h) for h in HORIZONS}
    pairs = {h: w@distance(v, v)@w.T for h, (w, v) in laws.items()}
    current_pair = distance(V, V)
    future = -.25*CENTER@(pairs[5]+pairs[20])@CENTER
    moments = {h: CENTER@(laws[h][0]@laws[h][1]) for h in (5, 20)}
    variance = .5*sum(np.outer(a, a) for a in moments.values())
    metrics = {"future_cdf": normalize_metric(future),
               "current_cdf": normalize_metric(-.5*CENTER@current_pair@CENTER),
               "future_variance": normalize_metric(variance),
               "probability_euclidean": normalize_metric(CENTER),
               "chi_square_only": np.zeros((4, 4))}
    u = np.array([1., 2., 4., -7.])/np.sqrt(70)
    r = np.eye(4)-2*np.outer(u, u)
    metrics["rotated_future_cdf"] = r@metrics["future_cdf"]@r
    metrics["future_cdf_ratio_preserving"] = metrics["future_cdf"]
    return p, laws, pairs, current_pair, metrics


def likelihood(y, epsilon=0., shock_variance=16.):
    variance = V[None, :]
    y2 = np.asarray(y)[:, None]**2
    narrow = np.exp(-y2/(2*variance))/np.sqrt(2*np.pi*variance)
    wide = np.exp(-y2/(2*(variance+shock_variance)))/np.sqrt(2*np.pi*(variance+shock_variance))
    return (1-np.asarray(epsilon)[:, None])*narrow + np.asarray(epsilon)[:, None]*wide if np.ndim(epsilon) else (1-epsilon)*narrow+epsilon*wide


def true_filter(obs, p, epsilon, shock_variance):
    prior = np.full((len(obs), 4), .25)
    post = np.empty(obs.shape+(4,))
    for t in range(obs.shape[1]):
        q = prior*likelihood(obs[:, t], epsilon, shock_variance)
        q /= q.sum(axis=1, keepdims=True)
        post[:, t] = q
        prior = q@p
    return post


def kkt(q, gradient):
    active = q > 1e-11
    multiplier = (gradient*active).sum(axis=-1)/active.sum(axis=-1)
    residual = gradient-multiplier[:, None]
    error = np.maximum(np.abs(residual)*active, np.maximum(-residual, 0)*(~active))
    return np.max(error, initial=0.)


def audit_updates(z, protocol, p, current_pair, metrics):
    obs = z["observed_returns"]
    origins = z["test_origins"]
    for family in protocol["families"]+["nominal_bayes", "fixed_prior"]:
        q0 = z[f"q__{family}"]
        record("probability_mass", np.max(np.abs(q0.sum(axis=-1)-1)), TOLERANCES["mass"])
        record("probability_negative", max(0., -q0.min()), 1e-12)
        if family == "fixed_prior":
            compare("fixed_prior", q0, np.full_like(q0, .25), 0.)
            continue
        incoming = np.empty_like(q0)
        incoming[:, 0] = z[f"initial_test_prior__{family}"]
        incoming[:, 1:] = q0[:, :-1]@p
        incoming[:, 1:] /= incoming[:, 1:].sum(axis=-1, keepdims=True)
        q, prior = q0.reshape(-1, 4), incoming.reshape(-1, 4)
        y = obs[:, origins].reshape(-1)
        parameter = np.repeat(np.ones(len(obs)) if family == "nominal_bayes" else
                              z[f"selected_parameter__{family}"], len(origins))
        COUNTS["test_update_rows"] = COUNTS.get("test_update_rows", 0)+len(q)
        if family in ("tempered_bayes", "nominal_bayes"):
            loglike = -.5*(np.log(2*np.pi*V)+y[:, None]**2/V)
            logweights = np.log(prior)+parameter[:, None]*loglike
            logweights -= logweights.max(axis=1, keepdims=True)
            exact = np.exp(logweights)
            exact /= exact.sum(axis=1, keepdims=True)
            compare("bayes_update", q, exact, 2e-12)
        elif family == "contaminated_bayes":
            exact = prior*likelihood(y, parameter, 16.)
            exact /= exact.sum(axis=1, keepdims=True)
            compare("contaminated_bayes_update", q, exact, 2e-12)
        else:
            zero = parameter == 0
            compare("zero_eta_identity", q[zero], prior[zero], 2e-12)
            used = ~zero
            if not used.any():
                continue
            qq, pp, ee, yy = q[used], prior[used], parameter[used], y[used]
            gradient = abs_moment(yy[:, None], V)-qq@current_pair
            if family == "crps_kl":
                if np.min(qq) <= 0:
                    raise AssertionError("KL positive interior solution lost")
                gradient += (np.log(qq/pp)+1)/ee[:, None]
                record("kl_original_kkt", np.max(np.abs(gradient-gradient.mean(axis=1, keepdims=True))), TOLERANCES["kkt"])
                COUNTS["kl_kkt_rows"] = COUNTS.get("kl_kkt_rows", 0)+len(qq)
            else:
                kappa = 1. if family == "chi_square_only" else .01
                gradient += 2*((qq-pp)@metrics[family]+kappa*(qq-pp)/pp)/ee[:, None]
                if family == "future_cdf_ratio_preserving":
                    ratio = pp[:, 1]/(pp[:, 1]+pp[:, 2])
                    gradient = np.column_stack((gradient[:, 0], ratio*gradient[:, 1]+(1-ratio)*gradient[:, 2], gradient[:, 3]))
                    reduced = np.column_stack((qq[:, 0], qq[:, 1]+qq[:, 2], qq[:, 3]))
                    record("ratio_constraint", np.max(np.abs(qq[:, 1]*pp[:, 2]-qq[:, 2]*pp[:, 1])), 2e-13)
                else:
                    reduced = qq
                record("qp_original_kkt", kkt(reduced, gradient), TOLERANCES["kkt"])
                COUNTS["qp_kkt_rows"] = COUNTS.get("qp_kkt_rows", 0)+len(qq)


def verify_simulation(z, scenario, protocol):
    p = transition(scenario["slow_persistence"], scenario["fast_persistence"])
    n = protocol["simulation"]["total_observations_per_scenario_seed"]
    for j, seed in enumerate(protocol["simulation"]["seeds"]):
        rng = np.random.default_rng(seed)
        uniforms, innovations, flags, shocks = rng.random(n), rng.standard_normal(n), rng.random(n), rng.standard_normal(n)
        states = np.empty(n, dtype=np.int64)
        states[0] = int(uniforms[0]*4)
        for t in range(1, n):
            states[t] = int(np.sum(uniforms[t] >= np.cumsum(p[states[t-1]])))
        clean = np.sqrt(V[states])*innovations
        contaminated = flags < scenario["contamination_probability"]
        observed = clean+contaminated*scenario["additive_shock_sd"]*shocks
        compare("simulated_states", z["latent_states"][j], states, 0.)
        compare("simulated_flags", z["contamination_flags"][j].astype(int), contaminated.astype(int), 0.)
        compare("simulated_clean", z["latent_clean_returns"][j], clean, 2e-14)
        compare("simulated_observations", z["observed_returns"][j], observed, 2e-14)


def scenario_replay(scenario, trace_dir, public, protocol, ref):
    name = scenario["name"]
    path = trace_dir/f"{name}.npz"
    receipt = json.loads((public/name/"EXECUTION_RECEIPT.json").read_text())
    if digest(path) != receipt["trace_sha256"]:
        raise AssertionError("Trace hash differs from execution receipt")
    z = np.load(path, allow_pickle=False)
    verify_simulation(z, scenario, protocol)
    p, laws, pairs, current_pair, metrics = ref
    audit_updates(z, protocol, p, current_pair, metrics)
    obs, origins = z["observed_returns"], z["test_origins"]
    sim = protocol["simulation"]
    compare("test_indices", origins, np.arange(sim["test_start_index"], sim["test_last_index"]+1), 0.)
    compare("validation_indices", z["validation_origins"], np.arange(sim["validation_start_index"], sim["validation_last_index"]+1), 0.)
    if sim["validation_last_index"]+20 > origins[0]:
        raise AssertionError("Validation targets not mature at first test issue")
    selection_path = public/name/"SELECTION_BEFORE_TEST.json"
    if digest(selection_path) != receipt["selection_file_sha256"]:
        raise AssertionError("Selection record hash differs")
    selection = json.loads(selection_path.read_text())
    selected_names = []
    for si, entry in enumerate(selection["seeds"]):
        vals = {}
        for family in protocol["families"]:
            gridname = "beta" if family == "tempered_bayes" else "epsilon" if family == "contaminated_bayes" else "eta"
            scores = z[f"validation_scores__{family}"][si].mean(axis=1)
            index = min(range(len(scores)), key=lambda i: scores[i])
            chosen = entry["families"][family]
            if chosen["grid_index"] != index or chosen["parameter"] != protocol["grids"][gridname][index]:
                raise AssertionError("Validation selection mismatch")
            compare("selected_parameters", np.array(z[f"selected_parameter__{family}"][si]), np.array(chosen["parameter"]), 0.)
            compare("selected_validation_score", np.array(scores[index]), np.array(chosen["validation_crps"]), 2e-12)
            vals[family] = float(scores[index])
        chosen_name = min(COMPETITORS, key=lambda f: vals[f])
        if chosen_name != entry["validation_selected_competitor"]:
            raise AssertionError("Composite competitor mismatch")
        selected_names.append(chosen_name)
    true_p = transition(scenario["slow_persistence"], scenario["fast_persistence"])
    truth = true_filter(obs, true_p, scenario["contamination_probability"], scenario["additive_shock_sd"]**2)[:, origins]
    compare("oracle_posterior", truth, z["oracle_posterior"], TOLERANCES["posterior"])
    families = protocol["families"]+["nominal_bayes", "fixed_prior", "true_oracle"]
    rows = {(seed, f): {"scenario": name, "seed": seed, "family": f,
                        "validation_selected_competitor": selected_names[i]}
            for i, seed in enumerate(sim["seeds"]) for f in families}
    states = z["latent_states"]
    changes = np.zeros_like(states, dtype=bool)
    changes[:, 1:] = states[:, 1:]//2 != states[:, :-1]//2
    recent = np.stack([changes[:, max(1, t-4):t+1].any(axis=1) for t in origins], axis=1)
    compare("recent_change_mask", recent.astype(int), z["recent_true_slow_change"].astype(int), 0.)
    change_scores = {f: np.zeros(truth.shape[:2]) for f in families}
    for h in HORIZONS:
        wm, vm = laws[h]
        wt, vt = fourier_law(true_p, h)
        eps, shockvar = scenario["contamination_probability"], scenario["additive_shock_sd"]**2
        cross = sum(mass*distance(vm, vt+k*shockvar/h) for k, mass in binomial_masses(h, eps))
        true_pair = sum(mass*distance(vt, vt+k*shockvar/h) for k, mass in binomial_masses(2*h, eps))
        cross, true_pair = wm@cross@wt.T, wt@true_pair@wt.T
        target = np.stack([obs[:, t+1:t+h+1].sum(axis=1) for t in origins], axis=1)/np.sqrt(h)
        a = abs_moment(target[:, :, None], vm)@wm.T
        true_a = sum(mass*(abs_moment(target[:, :, None], vt+k*shockvar/h)@wt.T)
                     for k, mass in binomial_masses(h, eps) if mass)
        oracle_expected = .5*np.sum((truth@true_pair)*truth, axis=-1)
        oracle_empirical = np.sum(truth*true_a, axis=-1)-oracle_expected
        compare("oracle_expected_score", oracle_expected, z[f"oracle_expected__h{h}"], TOLERANCES["score"])
        compare("oracle_empirical_score", oracle_empirical, z[f"oracle_empirical__h{h}"], TOLERANCES["score"])
        for family in families:
            if family == "true_oracle":
                expected, empirical = oracle_expected, oracle_empirical
            else:
                q = z[f"q__{family}"]
                self_term = .5*np.sum((q@pairs[h])*q, axis=-1)
                expected = np.sum((q@cross)*truth, axis=-1)-self_term
                empirical = np.sum(q*a, axis=-1)-self_term
                compare("expected_score", expected, z[f"expected__{family}__h{h}"], TOLERANCES["score"])
                compare("empirical_score", empirical, z[f"empirical__{family}__h{h}"], TOLERANCES["score"])
            record("oracle_excess_negative", max(0., -np.min(expected-oracle_expected)), TOLERANCES["oracle_excess"])
            if name == "clean_correct" and family == "nominal_bayes":
                compare("clean_bayes_oracle_identity", expected, oracle_expected, 2e-10)
            if h in (5, 20):
                change_scores[family] += expected/2
            for si, seed in enumerate(sim["seeds"]):
                row = rows[seed, family]
                row[f"expected_crps_h{h}"] = float(expected[si].mean())
                row[f"empirical_crps_h{h}"] = float(empirical[si].mean())
                row[f"excess_crps_h{h}"] = float((expected[si]-oracle_expected[si]).mean())
    for (seed, family), row in rows.items():
        row["expected_primary"] = (row["expected_crps_h5"]+row["expected_crps_h20"])/2
        row["empirical_primary"] = (row["empirical_crps_h5"]+row["empirical_crps_h20"])/2
        si = sim["seeds"].index(seed)
        row["recent_true_slow_change_origins"] = int(recent[si].sum())
        if name == "clean_correct" and recent[si].any():
            row["recent_change_expected_primary"] = float(change_scores[family][si, recent[si]].mean())
    # Check every stored validation score against the public trial table.
    trial_rows = [r for r in csv.DictReader((public/"TRIAL_VALIDATION.csv").open()) if r["scenario"] == name]
    for row in trial_rows:
        si = sim["seeds"].index(int(row["seed"]))
        scores = z[f"validation_scores__{row['family']}"][si, int(row["grid_index"])]
        for j, h in enumerate(HORIZONS):
            compare("validation_saved_grid_score", np.array(scores[j]), np.array(float(row[f"validation_crps_h{h}"])), 0.)
    return list(rows.values())


def audit_aggregate(rows, protocol, public):
    original = list(csv.DictReader((public/"SELECTED_RESULTS.csv").open()))
    indexed = {(r["scenario"], r["seed"], r["family"]): r for r in rows}
    for original_row in original:
        key = original_row["scenario"], int(original_row["seed"]), original_row["family"]
        row = indexed[key]
        for metric, value in row.items():
            if isinstance(value, (float, int)) and metric != "seed":
                compare("aggregate_results", np.array(value), np.array(float(original_row[metric])), TOLERANCES["aggregate"])
    summary = json.loads((public/"SUMMARY.json").read_text())
    draws = np.random.default_rng(20261019).integers(0, 20, (9999, 20))
    recomputed = {}
    for saved in summary["comparisons"]:
        scenarios = [saved["stratum"]] if saved["stratum"] != "positive_scenarios_equal_weight" else [
            s["name"] for s in protocol["scenarios"] if s["role"].startswith("Prespecified positive")]
        values = []
        for seed in protocol["simulation"]["seeds"]:
            diffs = []
            for scenario in scenarios:
                candidate = indexed[scenario, seed, "future_cdf"]
                control = candidate["validation_selected_competitor"] if saved["control"] == "validation_selected_competitor" else saved["control"]
                diffs.append(candidate[saved["metric"]]/indexed[scenario, seed, control][saved["metric"]]-1)
            values.append(np.mean(diffs))
        values = np.asarray(values)
        interval = np.quantile(np.mean(values[draws], axis=1), [.025, .975])
        compare("paired_seed_values", values, saved["seed_relative_values"], 2e-9)
        compare("paired_bootstrap", np.r_[values.mean(), interval],
                np.array([saved["mean_relative_minus_one"], saved["bootstrap95_lower"], saved["bootstrap95_upper"]]), 2e-9)
        recomputed[saved["stratum"], saved["control"], saved["metric"]] = (float(values.mean()), float(interval[1]))
    pos = "positive_scenarios_equal_weight"
    key = recomputed[pos, "validation_selected_competitor", "expected_primary"]
    gates = {
        "positive_expected_gain_at_least_one_percent": key[0] <= -.01,
        "positive_expected_seed_interval_upper_negative": key[1] < 0,
        "positive_each_beats_current_cdf": all(recomputed[s["name"], "current_cdf", "expected_primary"][0] < 0 for s in protocol["scenarios"] if s["role"].startswith("Prespecified positive")),
        "positive_beats_rotated_metric": recomputed[pos, "rotated_future_cdf", "expected_primary"][0] < 0,
        "positive_empirical_gain": recomputed[pos, "validation_selected_competitor", "empirical_primary"][0] < 0,
        "clean_noninferiority_half_percent": recomputed["clean_correct", "validation_selected_competitor", "expected_primary"][0] <= .005,
        "numerical_and_causal_checks": True,
    }
    for name, value in gates.items():
        if value != summary["decision"]["checks"][name]["passed"]:
            raise AssertionError(f"Gate disagreement: {name}")
    if all(gates.values()) != summary["decision"]["all_passed"]:
        raise AssertionError("Final decision mismatch")
    return gates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--public", type=Path, default=ROOT/"results"/"synthetic")
    parser.add_argument("--out", type=Path, default=ROOT/"audits"/"INDEPENDENT_TRACE_REPLAY.json")
    args = parser.parse_args()
    started = time.monotonic()
    protocol = json.loads((ROOT/"provenance"/"SYNTHETIC_PROTOCOL.json").read_text())
    rows = []
    ref = reference()
    for scenario in protocol["scenarios"]:
        rows += scenario_replay(scenario, args.traces, args.public, protocol, ref)
        print(json.dumps({"independent_scenario_passed": scenario["name"]}), flush=True)
    gates = audit_aggregate(rows, protocol, args.public)
    result = {"status": "PASS", "completed_utc": datetime.now(timezone.utc).isoformat(),
              "audit_script_sha256": digest(__file__),
              "protocol_sha256": digest(ROOT/"provenance"/"SYNTHETIC_PROTOCOL.json"),
              "production_modules_imported": False, "market_data_read": False,
              "independence": "Fourier matrix polynomial reward propagation; direct likelihood oracle; erf absolute moments; original objective KKT; separate selections/aggregation/bootstrap/gates",
              "tolerances": TOLERANCES, "coordinate_counts": COUNTS,
              "maximum_errors": MAXIMA, "independent_gate_decisions": gates,
              "aggregate_rows": len(rows), "elapsed_seconds": time.monotonic()-started,
              "limits": ["Validation choices verified against saved complete grid scores, not independently solved validation trajectories.",
                         "Test optimality conditions prove per-update convex solutions conditional on the stored incoming first-test prior; prefix replay is a separate check.",
                         "Tiny signed FFT coefficient roundoff is recorded and zeroed only in the independent reconstruction.",
                         "This is a numerical audit, not a new scientific confirmation dataset."]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": "PASS", "coordinates": sum(COUNTS.values()), "seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
