"""Post-fit numerical qualification of saved lognormal-mixture price CRPS.

No model, production score, saved prediction, market result, or original replay
cache is modified. The author implemented the candidate model/aggregator but
did not author mixture_metrics.py. This is an internal numerical investigation,
not an independent re-training audit or an external review.

The stable expression symbolically cancels E[X] before floating evaluation:
 CRPS = y sum_i w_i erf(z_i/sqrt(2))
        -2 sum_i w_i M_i Phi(z_i-s_i)
        +2 sum_ij w_i w_j M_i Phi((ell_j-ell_i-s_i^2)/hypot(s_i,s_j)).
Here M_i=exp(ell_i+s_i^2/2), z_i=(log(y)-ell_i)/s_i. The products
M_i Phi(.) are evaluated with exp(log(M_i)+log_ndtr(.)); no score clipping
or enlarged acceptance tolerance is used. Only a separate audit is written.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import mpmath as mp
import numpy as np
from scipy.integrate import quad
from scipy.special import erf, log_ndtr, logsumexp, ndtr

STUDY = Path(__file__).resolve().parents[1]
RTOL, ATOL = 2e-10, 2e-9
SQRT2 = math.sqrt(2.)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def normalize(w):
    w = np.asarray(w, dtype=float)
    assert np.all(w >= 0) and np.all(np.isfinite(w))
    assert np.all(np.abs(w.sum(axis=1) - 1) < 5e-5)
    return w / w.sum(axis=1, keepdims=True)


def stable_crps(y, w, ell, sd, chunk=48):
    """Return stable price CRPS and the three reduced-term magnitudes."""
    y, ell, sd = np.asarray(y), np.asarray(ell), np.asarray(sd)
    w = normalize(w)
    assert np.all(y > 0) and np.all(sd > 0)
    result, terms = np.empty(len(y)), np.empty((len(y), 3))
    for a in range(0, len(y), chunk):
        b = min(a + chunk, len(y))
        ww, mm, ss, yy = w[a:b], ell[a:b], sd[a:b], y[a:b]
        lw = np.full_like(ww, -np.inf)
        np.log(ww, out=lw, where=ww > 0)
        log_mean = mm + .5 * ss * ss
        z = (np.log(yy)[:, None] - mm) / ss
        first = yy * np.sum(ww * erf(z / SQRT2), axis=1)
        left = np.exp(lw + log_mean + log_ndtr(z - ss)).sum(axis=1)
        pair_sd = np.hypot(ss[:, :, None], ss[:, None, :])
        argument = (mm[:, None, :] - mm[:, :, None] - ss[:, :, None] ** 2) / pair_sd
        ordered_minimum = np.exp(lw[:, :, None] + lw[:, None, :] + log_mean[:, :, None]
                                 + log_ndtr(argument)).sum(axis=(1, 2))
        result[a:b] = first - 2 * left + 2 * ordered_minimum
        terms[a:b] = np.column_stack((first, -2 * left, 2 * ordered_minimum))
    assert np.all(np.isfinite(result)) and np.all(np.isfinite(terms))
    return result, terms


def old_replay_crps(y, w, ell, sd, chunk=48):
    """Reconstruct the original replay's uncancelled expectation structure."""
    w = normalize(w)
    result, magnitudes = np.empty(len(y)), np.empty((len(y), 2))
    for a in range(0, len(y), chunk):
        b = min(a + chunk, len(y))
        ww, mm, ss, yy = w[a:b], ell[a:b], sd[a:b], y[a:b]
        means = np.exp(mm + .5 * ss * ss)
        z = (np.log(yy)[:, None] - mm) / ss
        first_abs = yy[:, None] * (2 * ndtr(z) - 1) + means * (1 - 2 * ndtr(z - ss))
        pair_sd = np.sqrt(ss[:, :, None] ** 2 + ss[:, None, :] ** 2)
        truncated_i = means[:, :, None] * ndtr(
            (mm[:, None, :] - mm[:, :, None] - ss[:, :, None] ** 2) / pair_sd)
        minimum = truncated_i + truncated_i.transpose(0, 2, 1)
        absolute_pair = means[:, :, None] + means[:, None, :] - 2 * minimum
        first = np.sum(ww * first_abs, axis=1)
        half_pair = .5 * np.einsum("ni,nij,nj->n", ww, absolute_pair, ww)
        result[a:b] = first - half_pair
        magnitudes[a:b] = np.column_stack((first, half_pair))
    return result, magnitudes


def production_structure_crps(y, w, ell, sd):
    """Read-only re-expression of the production erf structure, not an import.

    Two float normalizations reflect score_mixture then lognormal_mixture_crps.
    This diagnostic does not make the saved production values a gold standard.
    """
    w = normalize(normalize(w))
    means = np.exp(ell + .5 * sd * sd)
    ly = np.log(y)[:, None]
    first_abs = means * erf((ell + sd * sd - ly) / (SQRT2 * sd))
    first_abs += y[:, None] * erf((ly - ell) / (SQRT2 * sd))
    delta = ell[:, :, None] - ell[:, None, :]
    pair_sd = np.hypot(sd[:, :, None], sd[:, None, :])
    pair_abs = means[:, :, None] * erf((delta + sd[:, :, None] ** 2) / (SQRT2 * pair_sd))
    pair_abs += means[:, None, :] * erf((-delta + sd[:, None, :] ** 2) / (SQRT2 * pair_sd))
    return (w * first_abs).sum(axis=1) - .5 * np.einsum("ni,nij,nj->n", w, pair_abs, w)


def mp_crps(y, weights, ell, sd, dps=80):
    """High precision on exact binary64 input values; weights normalized in mp."""
    with mp.workdps(dps):
        keep = np.asarray(weights) > 0
        y = mp.mpf(float(y))
        ww = [mp.mpf(float(x)) for x in np.asarray(weights)[keep]]
        total = mp.fsum(ww)
        ww = [x / total for x in ww]
        mm = [mp.mpf(float(x)) for x in np.asarray(ell)[keep]]
        ss = [mp.mpf(float(x)) for x in np.asarray(sd)[keep]]
        means = [mp.exp(m + s * s / 2) for m, s in zip(mm, ss)]
        phi = lambda x: mp.erfc(-x / mp.sqrt(2)) / 2
        z = [(mp.log(y) - m) / s for m, s in zip(mm, ss)]
        first = y * mp.fsum(w * mp.erf(v / mp.sqrt(2)) for w, v in zip(ww, z))
        left = mp.fsum(w * mean * phi(v - s) for w, mean, v, s in zip(ww, means, z, ss))
        contributions = []
        for i in range(len(ww)):
            for j in range(i + 1):
                scale = mp.sqrt(ss[i] ** 2 + ss[j] ** 2)
                minimum = means[i] * phi((mm[j] - mm[i] - ss[i] ** 2) / scale)
                minimum += means[j] * phi((mm[i] - mm[j] - ss[j] ** 2) / scale)
                contributions.append(ww[i] * ww[j] * minimum * (1 if i == j else 2))
        result = first - 2 * left + mp.fsum(contributions)
        return +result


def cdf_integral(y, w, ell, sd):
    """Independent defining CDF integrals in log-price coordinates."""
    w = normalize(np.asarray(w)[None, :])[0]
    keep = w > 0
    w, ell, sd = w[keep], np.asarray(ell)[keep], np.asarray(sd)[keep]
    lw, boundary = np.log(w), np.log(y)
    def integrand(t, right):
        argument = (ell - t) / sd if right else (t - ell) / sd
        return float(np.exp(t + 2 * logsumexp(lw + log_ndtr(argument))))
    left, le = quad(lambda t: integrand(t, False), -np.inf, boundary,
                    epsabs=2e-10, epsrel=2e-12, limit=400)
    right, re = quad(lambda t: integrand(t, True), boundary, np.inf,
                     epsabs=2e-10, epsrel=2e-12, limit=400)
    return left + right, le + re


def inventory(candidate, gas):
    tasks = []
    public = STUDY / "results/market/persistence"
    for receipt in sorted((public / "fits").glob("*.json")) + sorted((public / "references").glob("*.json")):
        row = json.loads(receipt.read_text())
        for name, expected in row["private_artifacts"].items():
            if name.endswith(".npz"):
                tasks.append((candidate / name, "persistence", expected))
    for receipt in sorted((gas / "predictions").glob("*/*/*.json")):
        row = json.loads(receipt.read_text())
        tasks.append((receipt.with_suffix(".npz"), "gas", row["prediction_sha256"]))
    assert len(tasks) == 690 and sum(kind == "persistence" for _, kind, _ in tasks) == 338
    assert len({str(p) for p, _, _ in tasks}) == 690
    return tasks


def distributions(path, kind):
    with np.load(path, allow_pickle=False) as z:
        roles = ("validation", "test") if kind == "persistence" else ("test",)
        cutoff = "2022-01-01" if "SHENZHEN_PLOS" in path.name else "2024-01-01"
        for role in roles:
            dates = z[role + "_dates"] if kind == "persistence" else z["date"]
            targets = z[role + "_target_dates"] if kind == "persistence" else z["target_date"]
            assert max(dates.tolist()) < cutoff and max(np.asarray(targets).reshape(-1).tolist()) < cutoff
        if kind == "gas" and "atoms" in z:
            return []  # Atomic scores contain no lognormal cancellation formula.
        result = []
        for role in roles:
            hs = ((0, 1), (1, 5), (2, 20)) if kind == "persistence" else ((None, None),)
            for col, horizon in hs:
                prefix = role + "_" if kind == "persistence" else ""
                if kind == "persistence":
                    weights, mu, sigma = [z[prefix + name + f"_h{horizon}"].copy()
                                           for name in ("weights", "means", "sigmas")]
                    unit, target = z[prefix + "unit"][:, col].copy(), z[prefix + "target_price"][:, col].copy()
                    p0 = z[prefix + "origin_price"].copy()
                    saved = z[prefix + f"score_h{horizon}_price_CRPS"].copy()
                else:
                    weights, mu, sigma = [z[name].copy() for name in ("weights", "means", "sigmas")]
                    unit, target, p0 = [z[name].copy() for name in ("unit", "target_price", "origin_price")]
                    saved = z["score_price_CRPS"].copy()
                    horizon = int(path.parent.parent.name.rsplit("_h", 1)[1])
                ell = np.log(p0)[:, None] + unit[:, None] * mu
                sd = unit[:, None] * sigma
                result.append({"role": role, "horizon": horizon, "weights": weights, "ell": ell,
                    "sd": sd, "target": target, "saved": saved, "p0": p0, "unit": unit,
                    "mu": mu, "sigma": sigma})
        return result


def mismatch(a, b):
    return np.abs(a - b) > ATOL + RTOL * np.abs(b)


def evaluate_point(locator, data, index):
    y, w, ell, sd = data["target"][index], data["weights"][index], data["ell"][index], data["sd"][index]
    started = time.perf_counter()
    ref80 = mp_crps(y, w, ell, sd, 80)
    ref120 = mp_crps(y, w, ell, sd, 120)
    stable, terms = stable_crps(np.array([y]), w[None, :], ell[None, :], sd[None, :])
    replay, large = old_replay_crps(np.array([y]), w[None, :], ell[None, :], sd[None, :])
    production_copy = production_structure_crps(np.array([y]), w[None, :], ell[None, :], sd[None, :])[0]
    integral, integral_bound = cdf_integral(y, w, ell, sd)
    reference = float(ref120)
    def error(value):
        return {"absolute": abs(float(value) - reference), "relative": abs(float(value) - reference) / max(abs(reference), 1e-300),
                "within_original_tolerance": bool(not mismatch(np.array([value]), np.array([reference]))[0])}
    result = {"locator": locator, "active_components": int(np.sum(w > 0)),
        "mp_digits": [80, 120], "mp_precision_relative_disagreement": float(abs(ref80 - ref120) / abs(ref120)),
        "saved_error": error(data["saved"][index]), "original_replay_error": error(replay[0]),
        "production_structure_error": error(production_copy), "stable_error": error(stable[0]),
        "cdf_definition_integral_error": error(integral), "quadrature_estimated_absolute_error": float(integral_bound),
        "uncancelled_term_to_result_ratio": float(np.max(np.abs(large)) / abs(reference)),
        "reduced_term_to_result_ratio": float(np.max(np.abs(terms)) / abs(reference)),
        "seconds": time.perf_counter() - started}
    # Full values remain private; the public numerical qualification omits scores/prices.
    private = {**result, "price_CRPS_reference_120_digits": mp.nstr(ref120, 120),
               "saved_price_CRPS": float(data["saved"][index]), "old_replay_price_CRPS": float(replay[0]),
               "stable_price_CRPS": float(stable[0]), "cdf_integral_price_CRPS": float(integral)}
    return result, private


def cached_evaluate_point(locator, data, index, directory):
    name = hashlib.sha256(json.dumps(locator, sort_keys=True).encode()).hexdigest() + ".json"
    path = directory / "point_checks" / name
    identity = sha(__file__)
    if path.exists():
        previous = json.loads(path.read_text())
        assert previous["audit_sha256"] == identity and previous["public"]["locator"] == locator
        return previous["public"], previous["private"]
    public, private = evaluate_point(locator, data, index)
    save_new(path, {"audit_sha256": identity, "public": public, "private": private})
    return public, private


def scan(tasks, private_output):
    counts = {"lognormal_artifacts": 0, "atomic_artifacts_skipped": 0, "distributions": 0, "origin_scores": 0,
              "saved_vs_stable_violations": 0, "old_replay_vs_stable_violations": 0,
              "old_replay_vs_saved_violations": 0}
    affected, candidates = [], []
    worst = {"saved_vs_stable_absolute": 0., "saved_vs_stable_relative": 0.,
             "old_replay_vs_stable_absolute": 0., "old_replay_vs_stable_relative": 0.}
    for ordinal, (path, kind, expected_hash) in enumerate(tasks, 1):
        assert sha(path) == expected_hash
        ds = distributions(path, kind)
        if not ds:
            counts["atomic_artifacts_skipped"] += 1
            continue
        counts["lognormal_artifacts"] += 1
        for data in ds:
            stable, _ = stable_crps(data["target"], data["weights"], data["ell"], data["sd"])
            old, _ = old_replay_crps(data["target"], data["weights"], data["ell"], data["sd"])
            saved = data["saved"]
            assert np.all(stable >= -ATOL)
            bad_saved, bad_old, bad_pair = mismatch(saved, stable), mismatch(old, stable), mismatch(old, saved)
            counts["distributions"] += 1
            counts["origin_scores"] += len(saved)
            for name, bad in (("saved_vs_stable_violations", bad_saved),
                              ("old_replay_vs_stable_violations", bad_old),
                              ("old_replay_vs_saved_violations", bad_pair)):
                counts[name] += int(np.sum(bad))
            for name, values in (("saved", saved), ("old_replay", old)):
                err = np.abs(values - stable)
                worst[name + "_vs_stable_absolute"] = max(worst[name + "_vs_stable_absolute"], float(err.max()))
                worst[name + "_vs_stable_relative"] = max(worst[name + "_vs_stable_relative"], float(np.max(err / np.maximum(np.abs(stable), 1e-300))))
            bad_any = bad_saved | bad_old | bad_pair
            if np.any(bad_any):
                locator = {"artifact": path.name, "artifact_sha256": expected_hash, "kind": kind,
                           "role": data["role"], "horizon": data["horizon"]}
                affected.append({**locator, "origins": len(saved), "union_affected_origins": int(np.sum(bad_any)),
                    "saved_vs_stable": int(np.sum(bad_saved)), "old_replay_vs_stable": int(np.sum(bad_old)),
                    "old_replay_vs_saved": int(np.sum(bad_pair))})
                ranked = np.argsort(np.maximum(np.abs(saved - stable), np.abs(old - stable)) /
                                    (ATOL + RTOL * np.abs(stable)))[::-1]
                for index in ranked[:min(2, int(np.sum(bad_any)))]:
                    candidates.append({"path": str(path), "locator": {**locator, "offset": int(index)},
                        "priority": float(max(abs(saved[index] - stable[index]), abs(old[index] - stable[index])) /
                                          (ATOL + RTOL * abs(stable[index])))})
        if ordinal % 50 == 0:
            print(json.dumps({"stage": "numerical_scan", "artifacts_considered": ordinal,
                              "saved_vs_stable_violations": counts["saved_vs_stable_violations"]}), flush=True)
    result = {"counts": counts, "maximum_errors": worst, "affected_distributions": affected,
              "candidate_high_precision_points": sorted(candidates, key=lambda x: x["priority"], reverse=True)}
    save_new(private_output / "NUMERICAL_SCREEN.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persistence-private", type=Path, default=Path("/tmp/carbon-persistence-market-runs"))
    parser.add_argument("--gas-private", type=Path, default=Path("/tmp/carbon-gas-normalization-run"))
    parser.add_argument("--private-output", type=Path, default=Path("/tmp/carbon-price-crps-cancellation-20261019b"))
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--precision-only", action="store_true")
    args = parser.parse_args()
    tasks = inventory(args.persistence_private, args.gas_private)
    if args.precision_only:
        screen = json.loads((args.private_output / "NUMERICAL_SCREEN.json").read_text())
    else:
        screen = scan(tasks, args.private_output)
    if args.scan_only:
        print(json.dumps({"stage": "numerical_scan_complete", "counts": screen["counts"]}), flush=True)
        return
    initial = args.persistence_private / "forecasts/SHENZHEN_PLOS_2020__GRU16__conditional_endpoint__s17.npz"
    requested = [(initial, "validation", 20, 8), (initial, "test", 20, 171), (initial, "test", 20, 177)]
    for row in screen["candidate_high_precision_points"][:5]:
        loc = row["locator"]
        requested.append((Path(row["path"]), loc["role"], loc["horizon"], loc["offset"]))
    requested = list(dict.fromkeys(requested))
    public_points, private_points = [], []
    for path, role, h, index in requested:
        kind = "persistence" if path.parent.name == "forecasts" else "gas"
        data = next(d for d in distributions(path, kind) if d["role"] == role and d["horizon"] == h)
        locator = {"artifact": path.name, "artifact_sha256": sha(path), "kind": kind,
                   "role": role, "horizon": h, "offset": index}
        public, private = cached_evaluate_point(locator, data, index, args.private_output)
        public_points.append(public)
        private_points.append(private)
        print(json.dumps({"stage": "precision_point_complete", "completed": len(public_points),
            "total": len(requested), "saved_passes_original_tolerance": public["saved_error"]["within_original_tolerance"],
            "stable_passes_original_tolerance": public["stable_error"]["within_original_tolerance"]}), flush=True)
    save_new(args.private_output / "HIGH_PRECISION_POINTS_PRIVATE.json", private_points)
    result = {"status": "POST_FIT_PRICE_CRPS_NUMERICAL_QUALIFICATION", "created_utc": datetime.now(timezone.utc).isoformat(),
        "audit_sha256": sha(__file__), "production_score_sha256": sha(STUDY / "source/mixture_metrics.py"),
        "original_tolerance": {"rtol": RTOL, "atol": ATOL, "unchanged": True},
        "counts": screen["counts"], "maximum_errors_against_stable_screen": screen["maximum_errors"],
        "affected_distributions": screen["affected_distributions"], "high_precision_points": public_points,
        "model_fits_or_gradient_steps": 0, "frozen_artifacts_modified": False,
        "score_module_imported": False, "all_old_price_scores_gold_standard": False,
        "role": "Internal numerical investigation by candidate-model implementer, who did not author the production scoring module",
        "primary_scope": "Only price CRPS arithmetic is investigated. No normalized CRPS, primary comparison, gate, checkpoint, or fitted parameter is changed",
        "screen_limit": "All lognormal scores are compared with an algebraically reduced double expression; selected original failures and the largest screened discrepancies additionally receive 80/120-digit and independent defining-CDF integral checks",
        "recommended_action": "Keep frozen scores unchanged. Repair the independent replay algebra, and explicitly qualify saved price-CRPS values that fail the unchanged tolerance against high precision; do not certify all saved values merely by increasing a tolerance"}
    save_new(STUDY / "audits/LOGNORMAL_CRPS_CANCELLATION_AUDIT.json", result)
    print(json.dumps({"stage": "numerical_qualification_complete", "counts": result["counts"],
                      "high_precision_points": len(public_points)}), flush=True)


if __name__ == "__main__":
    main()
