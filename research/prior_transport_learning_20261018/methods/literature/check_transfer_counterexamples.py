#!/usr/bin/env python3
"""Exact-population synthetic checks, with no data/model/optimizer imports.

This is our independent algebra, not execution or reproduction of Auto-Lambda
author software. It uses two centered uncorrelated unit-variance coordinates and
closed-form moments; there is no market read and no fitted estimator.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np


def risk(c, beta):
    """0.5 E[(beta * (x1+c*x2)-x2)^2], E[x x^T]=I."""
    return 0.5 * (beta**2 + (beta*c - 1.0)**2)


def profile_beta(c, ridge=0.0):
    return c / (1.0 + c*c + ridge)


def profile_risk(c, ridge=0.0):
    return risk(c, profile_beta(c, ridge))


def profile_risk_derivative(c, ridge=0.0):
    b = profile_beta(c, ridge)
    db = (1.0 + ridge - c*c) / (1.0 + ridge + c*c)**2
    direct = b * (b*c - 1.0)
    head_term = (b*(1.0 + c*c) - c) * db
    return direct + head_term


def finite_difference(function, value, eps=1e-6):
    return (function(value+eps) - function(value-eps)) / (2*eps)


def run_checks():
    checks = []

    def check(name, condition, **details):
        assert condition, name
        checks.append({"name": name, "passed": True, **details})

    # Primary y=x2, phi=x1+c*x2; auxiliary y_A=x1+2*x2,
    # with its current head fixed at 1 for evaluating the current gradient.
    c, beta, alpha = 0.5, -1.0, 0.05
    grad_primary_c = beta*(beta*c - 1.0)
    grad_primary_beta = beta*(1+c*c) - c
    grad_auxiliary_c = c - 2.0
    beta_virtual = beta - alpha*grad_primary_beta
    c_base = c - alpha*grad_primary_c
    dc_dlambda = -alpha*grad_auxiliary_c

    def c_virtual(weight):
        return c_base + dc_dlambda*weight

    def one_step_meta(weight):
        return risk(c_virtual(weight), beta_virtual)

    weight = 0.1
    c1 = c_virtual(weight)
    one_grad = beta_virtual*(beta_virtual*c1 - 1.0)*dc_dlambda
    check("one_step_joint_virtual_update_rejects_auxiliary", one_grad > 0,
          current_encoder_c=c, current_primary_head=beta,
          virtual_primary_head=beta_virtual, virtual_c=c1,
          weight=weight, hypergradient=one_grad)
    fd = finite_difference(one_step_meta, weight)
    check("one_step_hypergradient_matches_fd", abs(one_grad-fd) < 1e-8,
          analytic=one_grad, finite_difference=fd)

    for ridge in (0.0, 0.1):
        grad = profile_risk_derivative(c1, ridge)*dc_dlambda
        fd = finite_difference(lambda w: profile_risk(c_virtual(w), ridge), weight)
        check("profiled_head_accepts_auxiliary_ridge_"+str(ridge), grad < 0,
              hypergradient=grad, profile_beta=profile_beta(c1, ridge))
        check("profiled_hypergradient_matches_fd_ridge_"+str(ridge),
              abs(grad-fd) < 1e-8, analytic=grad, finite_difference=fd)
        check("finite_auxiliary_increase_improves_profiled_risk_ridge_"+str(ridge),
              profile_risk(c_virtual(weight+0.1), ridge) < profile_risk(c1, ridge),
              initial_risk=profile_risk(c1, ridge),
              increased_auxiliary_weight_risk=profile_risk(c_virtual(weight+0.1), ridge))
    check("finite_auxiliary_increase_worsens_one_step_meta",
          one_step_meta(weight+0.1) > one_step_meta(weight),
          initial_risk=one_step_meta(weight),
          increased_auxiliary_weight_risk=one_step_meta(weight+0.1))

    # For a cumulative RW label, each vector marks the future increments it uses.
    # At theta=0, theta'=alpha*lambda*y_train and the exact expected outer
    # hypergradient is alpha^2*lambda*Var(y_train)-alpha*Cov(y_train,y_meta).
    length = 20
    train_incidence = np.r_[np.ones(length), np.zeros(length)] / np.sqrt(length)
    overlap_incidence = np.r_[0.0, np.ones(length), np.zeros(length-1)] / np.sqrt(length)
    nonoverlap_incidence = np.r_[np.zeros(length), np.ones(length)] / np.sqrt(length)
    variance = float(train_incidence @ train_incidence)
    overlap_cov = float(train_incidence @ overlap_incidence)
    independent_cov = float(train_incidence @ nonoverlap_incidence)
    a, w = 0.1, 0.1
    overlap_hypergrad = a*a*w*variance - a*overlap_cov
    disjoint_hypergrad = a*a*w*variance - a*independent_cov
    check("rw_overlap_covariance_is_19_over_20", abs(overlap_cov-0.95) < 1e-14,
          covariance=overlap_cov, label_variance=variance)
    check("rw_overlapping_meta_rewards_pure_noise", overlap_hypergrad < 0,
          expected_hypergradient=overlap_hypergrad)
    check("rw_disjoint_meta_does_not_reward_pure_noise", disjoint_hypergrad > 0,
          covariance=independent_cov, expected_hypergradient=disjoint_hypergrad)
    check("rw_disjoint_gradient_at_zero_weight_is_zero", -a*independent_cov == 0.0)

    # Ridge's ordinary coefficient prior is not invariant to an arbitrary feature
    # rescaling. Profiling therefore does not by itself identify a unique learned
    # representation or remove all optimization/regularization choices.
    def ridge_risk_under_rescaling(scale, rho=0.1):
        variance = scale*scale*(1+c*c)
        covariance = scale*c
        b = covariance/(variance+rho)
        return 0.5*(1 - 2*b*covariance + b*b*variance)
    check("ridge_profile_not_feature_scale_invariant",
          abs(ridge_risk_under_rescaling(1.0)-ridge_risk_under_rescaling(2.0)) > 1e-7,
          scale_1_risk=ridge_risk_under_rescaling(1.0),
          scale_2_risk=ridge_risk_under_rescaling(2.0))

    return {"status": "PASS", "checks": checks, "check_count": len(checks),
            "scope": "Exact moments and arithmetic finite differences only; no sampled market or benchmark observations.",
            "author_software_executed": False, "market_data_reads": 0,
            "confirmation_file_reads": 0, "optimizer_fits": 0,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "utc": dt.datetime.now(dt.timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    receipt = run_checks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({key: receipt[key] for key in ("status", "check_count", "optimizer_fits", "source_sha256")}))


if __name__ == "__main__":
    main()
