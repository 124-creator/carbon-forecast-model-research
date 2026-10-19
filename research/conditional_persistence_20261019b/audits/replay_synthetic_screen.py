"""Reconcile every synthetic result with a separate polynomial/FFT algorithm.

This is an independent numerical algorithm, not a claim that a different person
authored this file. It does not import either production propagation or scoring.
No model fitting or market reads are performed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = Path("/tmp/carbon-persistence-screen-runs")
PI = np.array([.45,.4,.15])
Q = np.array([.25,1.,4.])
HS = (1,5,20)


def spectral_weights(rho, h):
    """Coefficient extraction from pi D(z) [P D(z)]^(h-1) 1."""
    count = 1 << int(np.ceil(np.log2(5*h+1)))
    roots = np.exp(2j*np.pi*np.arange(count)/count)
    diagonal = roots[:,None]**np.array([0,1,5])[None,:]
    transition = rho*np.eye(3)+(1-rho)*np.ones((3,1))*PI[None,:]
    matrix = transition[None,:,:]*diagonal[:,None,:]
    powers = np.linalg.matrix_power(matrix, h-1)
    polynomial = np.einsum("ni,nij,j->n", diagonal*PI, powers, np.ones(3))
    coefficients = np.fft.fft(polynomial)/count
    if np.max(np.abs(coefficients.imag)) > 1e-12 or coefficients.real.min() < -1e-12:
        raise AssertionError("Invalid numerical coefficient extraction")
    mass = np.maximum(coefficients.real[:5*h+1],0)
    if abs(mass.sum()-1) > 1e-11:
        raise AssertionError("Lost polynomial mass")
    mass /= mass.sum()
    sigma = np.sqrt((h*.25+.75*np.arange(5*h+1))/(h*(PI@Q)))
    return mass, sigma


def expected_score(rho, true_rho, h):
    w,s = spectral_weights(rho,h)
    v,_ = spectral_weights(true_rho,h)
    b = np.sqrt(2/np.pi)*np.hypot(s[:,None],s[None,:])
    return float(w@b@v-.5*w@b@w)


def main():
    receipt_path=ROOT/"results/synthetic/ALL_FIT_RECEIPTS.json"
    summary_path=ROOT/"results/synthetic/SUMMARY.json"
    receipts=json.loads(receipt_path.read_text())
    summary=json.loads(summary_path.read_text())
    rows=[r for receipt in receipts for r in receipt["records"]]
    mc=np.load(PRIVATE/"misspecified_test_statistics.npz")
    max_h_error=max_mean_error=0.
    exact=approx=0
    for row in rows:
        rho=[row["rho_minus"],row["rho_plus"]]
        expected=[]
        for h in HS:
            if row["scenario"] != "misspecified_leverage":
                truth=[.1,.9] if row["scenario"] == "conditional_persistence" else [.5,.5]
                score=np.mean([expected_score(rho[g],truth[g],h) for g in range(2)])
            else:
                parts=[]
                for g in range(2):
                    w,s=spectral_weights(rho[g],h)
                    b=np.sqrt(2/np.pi)*np.hypot(s[:,None],s[None,:])
                    parts.append(mc[f"g{g}_h{h}"]@w-.5*w@b@w)
                score=float(np.mean(parts))
            expected.append(score)
            max_h_error=max(max_h_error,abs(score-row["population_CRPS_h"][str(h)]))
        max_mean_error=max(max_mean_error,abs(np.mean(expected)-row["population_CRPS"]))
        if row["scenario"] == "misspecified_leverage": approx+=1
        else: exact+=1
    if max(max_h_error,max_mean_error)>1e-10:
        raise AssertionError("Stored expected score does not match spectral replay")
    group_error=0.
    for group in summary["groups"]:
        selection=[r for r in rows if r["scenario"]==group["scenario"] and r["n"]==group["n"]]
        mapping={seed:{r["arm"]:r for r in selection if r["seed"]==seed} for seed in range(1000,1020)}
        relative=[v["conditional_joint005"]["population_CRPS"]/v["constant_joint005"]["population_CRPS"]-1 for v in mapping.values()]
        group_error=max(group_error,abs(np.mean(relative)-group["primary_relative_mean"]))
        assert int(np.sum(np.array(relative)<0))==group["primary_wins"]
    gate=next(g for g in summary["groups"] if g["scenario"]=="conditional_persistence" and g["n"]==1024)
    replay_gate={"at_least_2bps":gate["primary_relative_mean"]<=-.0002,
                 "at_least_14_of_20_wins":gate["primary_wins"]>=14,
                 "no_worse_than_endpoint_constant":gate["primary_vs_endpoint_constant_relative_mean"]<=0}
    assert replay_gate==summary["gate_checks"]
    assert len(rows)==720 and len(receipts)==60
    assert sum(len(r["scalar_optimizers"]) for r in rows)==1080
    out={"created_utc":datetime.now(timezone.utc).isoformat(),"passed":True,
         "algorithm":"FFT coefficient extraction of matrix probability-generating polynomial; no production propagation/scoring imports",
         "authorship_scope":"separate algorithm replay by root, not a different-person claim",
         "exact_population_records":exact,"independent_MC_records":approx,
         "all_horizon_scores_replayed":len(rows)*len(HS),
         "maximum_horizon_score_error":max_h_error,"maximum_mean_score_error":max_mean_error,
         "maximum_group_relative_error":group_error,"gate_replayed":replay_gate,
         "models_refitted":0,"market_data_read":False,
         "summary_sha256":hashlib.sha256(summary_path.read_bytes()).hexdigest(),
         "all_fit_receipts_sha256":hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
         "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path=ROOT/"audits/SYNTHETIC_SPECTRAL_REPLAY.json"
    path.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out))


if __name__=="__main__": main()
