#!/usr/bin/env python3
"""Finite, exact synthetic counterexamples: no data input and no model fitting.

Uses the standard library only. These examples test mathematical claims, not
whether a particular market has predictive signal.
"""
from __future__ import annotations

import argparse
import cmath
import hashlib
import json
import platform
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path


def atom_world(atoms):
    # Each atom is (probability, observed-past scale, normalized future change).
    ez = sum(p*z for p, s, z in atoms)
    es2 = sum(p*s*s for p, s, z in atoms)
    es2z = sum(p*s*s*z for p, s, z in atoms)
    rw = sum(p*s*s*z*z for p, s, z in atoms)
    f_risk = es2z / es2

    def raw_risk(f):
        return sum(p*s*s*(f-z)**2 for p, s, z in atoms)

    return {
        "atoms": [{"probability": float(p), "scale": float(s), "z": float(z)} for p, s, z in atoms],
        "unweighted_normalized_optimum": float(ez),
        "unweighted_normalized_optimum_with_shrink1": float(ez/2),
        "raw_risk_optimum": float(f_risk), "raw_risk_optimum_exact": str(f_risk),
        "rw_raw_risk": float(rw), "best_raw_risk": float(raw_risk(f_risk)),
        "best_relative_raw_risk": float(raw_risk(f_risk)/rw),
        "risk_if_using_opposite_world_optimum": float(raw_risk(-f_risk)),
        "exact_weighting_identity_checked": all(
            sum(p*(s*s)*(f-z)**2 for p, s, z in atoms) == raw_risk(f)
            for f in [F(-2), F(-15,17), F(0), F(15,17), F(2)]),
    }


def step_distribution(dist, transitions):
    result = defaultdict(F)
    for (state, reward), weight in dist.items():
        for new_state, increment, probability in transitions[state]:
            result[(new_state, reward+increment)] += weight*probability
    return dict(result)


def cf_by_matrix(transitions, omega, horizon, initial_state=0):
    n = len(transitions)
    kernel = [[0j]*n for _ in range(n)]
    for state, rows in enumerate(transitions):
        for nxt, reward, probability in rows:
            kernel[state][nxt] += float(probability)*cmath.exp(1j*omega*reward)
    vector = [complex(i == initial_state) for i in range(n)]
    for _ in range(horizon):
        vector = [sum(vector[i]*kernel[i][j] for i in range(n)) for j in range(n)]
    return sum(vector)


def mean_by_reward_recursion(transitions, horizon, initial_state=0):
    n = len(transitions)
    state_p = [F(i == initial_state) for i in range(n)]
    total = F(0)
    for _ in range(horizon):
        total += sum(state_p[a]*probability*reward
                     for a, rows in enumerate(transitions) for _, reward, probability in rows)
        state_p = [sum(state_p[a]*probability for a, rows in enumerate(transitions)
                       for nxt, _, probability in rows if nxt == b) for b in range(n)]
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("RISK_CLOCK_SYNTHETIC.json"))
    args = parser.parse_args()
    checks = []

    def check(name, passed):
        if not passed:
            raise AssertionError(name)
        checks.append(name)

    # Identical (X,z) laws and scale marginals; only the scale-label coupling changes.
    worlds = {
        "A": atom_world([(F(1,2), F(1), F(1)), (F(1,2), F(4), F(-1))]),
        "B": atom_world([(F(1,2), F(4), F(1)), (F(1,2), F(1), F(-1))]),
    }
    check("same normalized target mean in two worlds", worlds["A"]["unweighted_normalized_optimum"] == worlds["B"]["unweighted_normalized_optimum"] == 0)
    check("opposite raw risk optima", worlds["A"]["raw_risk_optimum_exact"] == "-15/17" and worlds["B"]["raw_risk_optimum_exact"] == "15/17")
    check("plain WLS represents exact raw price risk", all(w["exact_weighting_identity_checked"] for w in worlds.values()))
    check("zero is not raw risk optimum in either synthetic world", all(w["best_raw_risk"] < w["rw_raw_risk"] for w in worlds.values()))

    # General finite Markov additive model, including signed and zero increments.
    transitions = [
        [(0,-1,F(1,10)), (0,2,F(3,10)), (1,0,F(6,10))],
        [(0,-1,F(2,10)), (0,2,F(5,10)), (1,0,F(3,10))],
    ]
    check("Markov reward kernel rows sum to one", all(sum(p for _,_,p in r) == 1 for r in transitions))
    dist = {(0,0): F(1)}
    maximum_cf_error = 0.
    maximum_cf_mean_error = 0.
    horizons = []
    for h in range(1,21):
        dist = step_distribution(dist, transitions)
        check(f"H{h} reward distribution normalized", sum(dist.values()) == 1)
        mean_exact = sum(p*r for (_,r),p in dist.items())
        check(f"H{h} reward recursion equals exact distribution mean", mean_exact == mean_by_reward_recursion(transitions,h))
        for omega in [0., .01, .13, .5, 1., 2.7]:
            direct = sum(float(p)*cmath.exp(1j*omega*r) for (_,r),p in dist.items())
            matrix = cf_by_matrix(transitions,omega,h)
            error = abs(direct-matrix)
            maximum_cf_error = max(maximum_cf_error,error)
            check(f"H{h} omega{omega} matrix transform equals independent convolution", error < 3e-13)
        fd_mean = cf_by_matrix(transitions,1e-7,h).imag/1e-7
        maximum_cf_mean_error = max(maximum_cf_mean_error,abs(fd_mean-float(mean_exact)))
        check(f"H{h} CF derivative matches reward mean", abs(fd_mean-float(mean_exact)) < 3e-9)
        if h in [1,5,20]:
            horizons.append({"horizon":h,"mean_exact":str(mean_exact),"mean":float(mean_exact),"state_reward_support":len(dist)})

    # Age0 has arrival probability .2; age>=1 has .8. Marks are symmetric.
    symmetric = [
        [(0,-1,F(1,10)), (0,1,F(1,10)), (1,0,F(8,10))],
        [(0,-1,F(4,10)), (0,1,F(4,10)), (1,0,F(2,10))],
    ]
    age_examples=[]
    for state in [0,1]:
        dist = {(state,0):F(1)}
        for h in range(1,21):
            dist=step_distribution(dist,symmetric)
            mean=sum(p*r for (_,r),p in dist.items())
            check(f"symmetric marks initial age{state} H{h} imply RW mean",mean==0)
            if h in [1,5,20]:
                base_risk=sum(p*r*r for (_,r),p in dist.items())
                wrong_correction=F(1,4)
                wrong_risk=sum(p*(wrong_correction-r)**2 for (_,r),p in dist.items())
                check(f"age{state} H{h} nonzero correction adds squared bias",wrong_risk-base_risk==wrong_correction**2)
                age_examples.append({"initial_age_state":state,"horizon":h,"conditional_mean":0.,"rw_mse":float(base_risk),"extra_mse_for_correction_0p25":float(wrong_risk-base_risk)})
    check("predictable activity remains distinct from signed conditional mean", F(1,5) != F(4,5))

    # Same hazards, age-specific mark means +.5 and -.5: ordinary state models can exploit it.
    signed = [
        [(0,-1,F(1,20)), (0,1,F(3,20)), (1,0,F(16,20))],
        [(0,-1,F(12,20)), (0,1,F(4,20)), (1,0,F(4,20))],
    ]
    signed_one_step=[float(mean_by_reward_recursion(signed,1,a)) for a in [0,1]]
    check("same hazard supports both zero-mean and nonzero-mean worlds",signed_one_step==[.1,-.4])

    # Observations alone cannot identify sub-observation trade/event activity.
    latent_no_change=[100.,100.,100.]
    latent_cancelled_change=[100.,101.,100.]
    observed_no_change=[latent_no_change[i] for i in [0,2]]
    observed_cancelled_change=[latent_cancelled_change[i] for i in [0,2]]
    check("same observed net zero does not identify latent event count",observed_no_change==observed_cancelled_change and latent_no_change!=latent_cancelled_change)

    # A conservative as-of rule can only use releases strictly before a query date
    # when within-date order is unverified; no price or model is involved here.
    synthetic_dates=[1,3,5,8]
    query_date=5
    admitted=[d for d in synthetic_dates if d<query_date]
    check("unknown same-day release is excluded",admitted==[1,3])
    check("future suffix cannot change synthetic as-of support",admitted==[d for d in synthetic_dates+[9,11] if d<query_date])

    output={
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "scope":"Standard-library synthetic mathematics only, not a market experiment",
        "python_version":platform.python_version(),
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "status":"PASS","checks":len(checks),"check_names":checks,
        "two_worlds":worlds,
        "two_world_identical_observable_law":"X is constant; P(z=+1)=P(z=-1)=1/2; P(scale=1)=P(scale=4)=1/2. Scale is omitted from the model input.",
        "scale_input_boundary":"Adding scale distinguishes the atoms; an ordinary conditional regressor can then fit these noiseless worlds. No renewal model is needed.",
        "markov_additive_transform":{"horizons":horizons,"maximum_cf_vs_exact_distribution_error":maximum_cf_error,"maximum_cf_derivative_vs_exact_mean_error":maximum_cf_mean_error,"novelty":"Standard finite Markov additive / reward algebra, not a new transform method"},
        "predictable_activity_zero_signed_mean":age_examples,
        "same_hazard_nonzero_signed_mean_one_step":signed_one_step,
        "observed_zero_nonidentifiability":{"observation":[100.,100.],"latent_no_change":latent_no_change,"latent_cancelled_change":latent_cancelled_change},
        "activity":{"market_reads":0,"confirmation_reads":0,"model_fits":0,"optimizer_steps":0,"random_searches":0,"old_file_writes":0},
        "limitations":["The worlds are constructed counterexamples, not evidence about any approved market.","The transform is exact only for the specified finite state model with its current context fixed or otherwise explicitly propagated.","Age-dependent nonzero conditional means can be fit by ordinary state/hurdle regressions; this example gives no neural novelty claim."]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"output":str(args.output),"status":output["status"],"checks":len(checks),"two_world_optima":{k:v["raw_risk_optimum_exact"] for k,v in worlds.items()},"maximum_cf_error":maximum_cf_error,"activity":output["activity"]},ensure_ascii=False))


if __name__=="__main__":
    main()
