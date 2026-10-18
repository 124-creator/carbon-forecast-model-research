#!/usr/bin/env python3
"""Independent, synthetic-only validation of the factorized auction operator.

No market files, evaluation predictions, fitting, or optimizers are used.
The tested implementation is imported, while numerical reference quantities
use SciPy integration, SciPy distributions, and independently sampled laws.
Each execution preserves an immutable receipt and exact source snapshots in
an attempt directory before replacing the convenient latest receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import softmax
from scipy.stats import gamma, norm
import torch


def digest(data):
    return hashlib.sha256(data).hexdigest()


def as_numpy(x):
    return x.detach().cpu().double().numpy()


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def crps_by_cdf_integration(mu, sigma, weights, y):
    """Independent definition: integral of squared CDF error in log x.

    Integration breakpoints resolve narrow and separated components. Omitted
    tails are bounded by F(lower)^2*lower and E[X 1{X>upper}], respectively.
    The upper tail uses survival probabilities, avoiding 1-CDF cancellation.
    """
    mu, sigma, weights = map(lambda a: np.asarray(a, dtype=float),
                             (mu, sigma, weights))
    ly = math.log(y)
    lo = min(float(np.min(mu-12*sigma)), ly)
    hi = max(float(np.max(mu+12*sigma+sigma*sigma)), ly)
    points = [lo, hi, ly]
    for m, s in zip(mu, sigma):
        points.extend(m+z*s for z in (-8, -4, -1, 0, 1, 4, 8))
    points = sorted(set(float(p) for p in points if lo <= p <= hi))
    answer = 0.0
    integration_error = 0.0
    for left, right in zip(points[:-1], points[1:]):
        if right <= ly:
            fn = lambda t: float(np.dot(weights, norm.cdf((t-mu)/sigma)))**2*math.exp(t)
        else:
            fn = lambda t: float(np.dot(weights, norm.sf((t-mu)/sigma)))**2*math.exp(t)
        value, error = quad(fn, left, right, epsabs=2e-11, epsrel=2e-10,
                            limit=150)
        answer += value
        integration_error += error
    means = np.exp(mu+.5*sigma*sigma)
    upper_bound = float(np.dot(weights, means*norm.sf((hi-mu-sigma*sigma)/sigma)))
    lower_bound = math.exp(lo)*float(np.dot(weights, norm.cdf((lo-mu)/sigma)))**2
    return answer, integration_error, upper_bound+lower_bound


def single_lognormal_crps(mu, sigma, y):
    z = (math.log(y)-mu)/sigma
    mean = math.exp(mu+.5*sigma*sigma)
    return (y*(2*norm.cdf(z)-1)
            -2*mean*(norm.cdf(z-sigma)+norm.cdf(sigma/math.sqrt(2))-1))


def scipy_clearing_weights(fractions, concentration, means, supply, mixture):
    fractions = np.asarray(fractions, float)
    a = fractions*np.asarray(concentration, float)[:, None]
    tails = np.flip(np.cumsum(np.flip(a, axis=-1), axis=-1), axis=-1)
    sf = gamma.sf(supply, a=tails,
                  scale=(np.asarray(means)/np.asarray(concentration))[:, None])
    # Strict upper tail has no mass at the upper endpoint. Positive Gamma
    # quantities make the equality event probability zero in this model.
    exclusive = np.concatenate((sf[:, 1:], np.zeros((len(sf), 1))), axis=1)
    atom_mass = sf-exclusive
    joint = np.asarray(mixture)[:, None]*atom_mass
    z = float(joint.sum())
    return (joint/z).ravel(), 1-z


def construct_model(module, mode="stochastic", dtype=torch.float64,
                    components=2, knots=5):
    model = module.FactorizedDemandNetwork(3, 6, hidden=6,
                                           components=components, knots=knots,
                                           mode=mode).to(dtype=dtype)
    model.eval()
    return model


def synthetic_inputs(n=4, dtype=torch.float64):
    generator = torch.Generator().manual_seed(1701)
    x = torch.randn(n, 7, 3, generator=generator, dtype=dtype)*.2
    context = torch.randn(n, 6, generator=generator, dtype=dtype)*.2
    values = [[.45, .60, .040, .002, 1.005],
              [.70, .82, .095, .008, .970],
              [.35, .48, .170, .025, 1.035],
              [.85, .65, .060, .004, 1.012]]
    # Base quantities are double even for production-style float32 networks.
    base = torch.tensor(values[:n], dtype=torch.float64)
    return x, context, base


CFG = {"crps_weight": .2, "median_auxiliary_weight": .1,
       "basis_auxiliary_weight": .02,
       "future_bid_marginal_nll_weight": .02}


def make_labels(base):
    n = len(base)
    price_z = torch.tensor([-.30, .70, -1.10, .25], dtype=torch.float64)[:n]
    median_z = torch.tensor([-.25, .55, -.90, .30], dtype=torch.float64)[:n]
    median_ratio = 1+base[:, 2]*median_z/base[:, 4]
    observed_q = (1+base[:, 2]*price_z)/(base[:, 4]+base[:, 2]*median_z)
    basis_z = (observed_q-1/base[:, 4])/base[:, 3]
    future_bid = torch.tensor([1.10, .95, 1.45, 1.05], dtype=torch.float64)[:n]
    mask = torch.ones(n, dtype=torch.bool)
    return dict(target=price_z, completed=mask.clone(),
                future_bid_ratio=future_bid, has_bid=mask.clone(),
                median_z=median_z, median_ratio=median_ratio,
                has_median=mask.clone(), basis_z=basis_z,
                has_basis=mask.clone())


def numeric_check(analytic, numeric, atol, rtol):
    error = abs(analytic-numeric)
    tolerance = atol+rtol*max(abs(analytic), abs(numeric))
    require(math.isfinite(analytic) and math.isfinite(numeric),
            f"nonfinite gradient {analytic}, {numeric}")
    require(error <= tolerance,
            f"analytic={analytic:.12g}, numeric={numeric:.12g}, "
            f"error={error:.6g} > tolerance={tolerance:.6g}")
    return dict(analytic=analytic, central_difference=numeric,
                absolute_error=error, tolerance=tolerance)


def run_checks(module, rng, mc_samples):
    checks = {}

    def check(name, function):
        try:
            result = function()
            checks[name] = {"pass": True, **(result or {})}
        except Exception as exc:
            checks[name] = {"pass": False, "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(limit=5)}
        print(f"{name}: {'PASS' if checks[name]['pass'] else 'FAIL'}", flush=True)

    def definition_integrals():
        cases = [
            ([0.0], [.20], [1.0], 1.0),
            ([.30], [.75], [1.0], .12),
            ([-.70], [.06], [1.0], 1.2),
            ([-.30, .10, .80], [.12, .40, .85], [.12, .63, .25], 1.15),
            ([-.90, .20, 1.10], [.70, .11, .35], [.68, .30, .02], .18),
            ([0.0, .0003], [.00010, .00015], [.40, .60], 1.0001),
            ([0.1, .1, .7], [.25, .25, .45], [.15, .55, .30], 5.0),
        ]
        results = []
        for mu, sig, w, y in cases:
            actual = float(module.lognormal_crps(torch.tensor([mu]), torch.tensor([sig]),
                                                  torch.tensor([w]), torch.tensor([y])))
            # Explicit float64 rerun avoids using Python-list default float32
            # rounding as the reference definition of the input parameters.
            actual = float(module.lognormal_crps(torch.tensor([mu], dtype=torch.float64),
                                torch.tensor([sig], dtype=torch.float64),
                                torch.tensor([w], dtype=torch.float64),
                                torch.tensor([y], dtype=torch.float64)))
            numerical, integration_error, tail_bound = crps_by_cdf_integration(mu, sig, w, y)
            tolerance = 2e-8+2e-8*abs(numerical)
            require(actual >= -1e-10, f"negative CRPS {actual}")
            require(abs(actual-numerical) <= tolerance,
                    f"CRPS {actual} vs CDF integral {numerical}")
            results.append(dict(mu=mu, sigma=sig, weights=w, y=y, crps=actual,
                                cdf_integral=numerical, absolute_error=abs(actual-numerical),
                                quadrature_error=integration_error,
                                omitted_tail_bound=tail_bound))
        return {"cases": results}

    check("lognormal_mixture_crps_vs_independent_cdf_integral", definition_integrals)

    def known_single():
        results = []
        for mu, sig, y in [(0., .3, 1.), (.7, .8, .2), (-.4, .08, 1.5),
                           (0., .0001, 1.), (.1, 1.5, 4.)]:
            model_value = float(module.lognormal_crps(
                torch.tensor([[mu]], dtype=torch.float64),
                torch.tensor([[sig]], dtype=torch.float64),
                torch.ones(1, 1, dtype=torch.float64),
                torch.tensor([y], dtype=torch.float64)))
            reference = single_lognormal_crps(mu, sig, y)
            require(abs(model_value-reference) < 1e-10*(1+abs(reference)),
                    f"single-component mismatch {model_value} vs {reference}")
            results.append({"mu": mu, "sigma": sig, "y": y,
                            "actual": model_value, "known_formula": reference})
        return {"cases": results,
                "diagonal_note": "Independent draws from the same component have nonzero pair distance."}

    check("known_single_lognormal_formula_including_diagonal_pair", known_single)

    def mc_crps():
        mu = np.array([-.2, .15, .8])
        sigma = np.array([.1, .35, .65])
        w = np.array([.2, .55, .25])
        y = 1.15
        i = rng.choice(3, mc_samples, p=w)
        j = rng.choice(3, mc_samples, p=w)
        x = np.exp(mu[i]+sigma[i]*rng.normal(size=mc_samples))
        xx = np.exp(mu[j]+sigma[j]*rng.normal(size=mc_samples))
        scores = abs(x-y)-.5*abs(x-xx)
        estimate = float(scores.mean())
        se = float(scores.std(ddof=1)/math.sqrt(mc_samples))
        actual = float(module.lognormal_crps(torch.from_numpy(mu[None]),
                          torch.from_numpy(sigma[None]), torch.from_numpy(w[None]),
                          torch.tensor([y], dtype=torch.float64)))
        tolerance = 6*se+1e-8
        require(abs(actual-estimate) <= tolerance, "MC CRPS outside 6SE tolerance")
        return dict(samples=mc_samples, analytic=actual, monte_carlo=estimate,
                    standard_error=se, tolerance=tolerance)

    check("lognormal_mixture_crps_vs_independent_monte_carlo", mc_crps)

    def crps_gradients():
        p = torch.tensor([-.25, .18, .65, -.4, -.7, -.2, -.8, .3, -.1],
                         dtype=torch.float64, requires_grad=True)
        def evaluate(v):
            mu = v[:3][None]
            sig = (.03+torch.nn.functional.softplus(v[3:6]))[None]
            w = torch.softmax(v[6:9], -1)[None]
            return module.lognormal_crps(mu, sig, w,
                      torch.tensor([1.13], dtype=torch.float64)).sum()
        g = torch.autograd.grad(evaluate(p), p)[0]
        results = []
        for index in range(len(p)):
            plus, minus = p.detach().clone(), p.detach().clone()
            eps = 1e-5
            plus[index] += eps
            minus[index] -= eps
            fd = float((evaluate(plus)-evaluate(minus))/(2*eps))
            results.append({"index": index, **numeric_check(float(g[index]), fd, 2e-7, 2e-5)})
        return {"variables": "log means, positive sigma raw variables, softmax logits",
                "gradients": results}

    check("standalone_crps_gradients_vs_central_difference", crps_gradients)

    def gamma_reference_and_mean():
        model = construct_model(module)
        raw = np.array([[-.3, .13, -.3, .03, -.2, -.4, .1, .2, -.1, .5],
                        [.4, -.1, .25, -.05, .3, .3, -.1, .4, -.2, .1]])
        with torch.no_grad():
            model.micro_head.weight.zero_()
            model.micro_head.bias.copy_(torch.from_numpy(raw.ravel()))
            model.macro_head.weight.zero_()
            model.macro_head.bias.copy_(torch.tensor([.25, -.1], dtype=torch.float64))
            model.supply_to_bid_volume.weight.zero_()
        x, context, base = synthetic_inputs()
        with torch.no_grad():
            output = model(x, context, base)
        fractions = softmax(raw[:, 5:], axis=-1)
        results = []
        for row in range(len(base)):
            weights, failure = scipy_clearing_weights(fractions,
                    as_numpy(output["concentration"])[row],
                    as_numpy(output["mean_bid_ratio"])[row],
                    float(base[row, 1]), as_numpy(output["mix"])[row])
            actual_w = as_numpy(output["weights"])[row]
            require(np.max(abs(actual_w-weights)) < 2e-10,
                    "Gamma clearing weights disagree with independent SciPy survival differences")
            require(abs(float(output["mechanical_failure"][row])-failure) < 2e-10,
                    "mechanical failure mismatch")
            mu = as_numpy(output["lognormal_mu_ratio"])[row]
            sigma = float(output["lognormal_sigma"][row, 0])
            reference_mean = float(np.dot(weights, np.exp(mu+.5*sigma*sigma)))
            returned_mean = 1+float(output["rprice"][row])*float(output["mean"][row])
            require(abs(reference_mean-returned_mean) < 2e-12,
                    "conditional arithmetic mean mismatch")
            require(np.isfinite(mu).all() and sigma > 0 and reference_mean > 0,
                    "invalid positive price law")
            selected = rng.choice(len(weights), size=mc_samples, p=weights)
            draws = np.exp(mu[selected]+sigma*rng.normal(size=mc_samples))
            se = float(draws.std(ddof=1)/math.sqrt(mc_samples))
            require(abs(float(draws.mean())-reference_mean) < 6*se+1e-9,
                    "positive price mixture mean outside 6SE MC tolerance")
            require(bool(np.all(draws > 0)), "nonpositive sampled price ratio")
            numerical_crps, error, bound = crps_by_cdf_integration(
                mu, np.full_like(mu, sigma), weights, 1.05)
            code_crps = float(module.lognormal_crps(
                output["lognormal_mu_ratio"][row:row+1],
                output["lognormal_sigma"][row:row+1], output["weights"][row:row+1],
                torch.tensor([1.05], dtype=torch.float64)))
            require(abs(code_crps-numerical_crps) < 2e-8,
                    "model-induced CRPS fails independent integral")
            results.append(dict(row=row, mean_ratio=reference_mean,
                                mc_mean_ratio=float(draws.mean()), mc_standard_error=se,
                                min_draw_ratio=float(draws.min()),
                                gamma_mass_max_error=float(np.max(abs(actual_w-weights))),
                                crps=code_crps, cdf_integral=numerical_crps,
                                integral_error=error, omitted_tail_bound=bound))
        return {"cases": results, "samples_per_case": mc_samples}

    check("full_operator_gamma_masses_positive_law_and_integrated_mean", gamma_reference_and_mean)

    def rw_reference():
        results = []
        for mode in ("stochastic", "direct", "no_supply", "unseparated"):
            model = construct_model(module, mode=mode)
            with torch.no_grad():
                model.macro_head.weight.zero_()
                model.macro_head.bias.zero_()
                model.micro_head.weight.zero_()
                model.micro_head.bias.zero_()
                model.supply_to_bid_volume.weight.zero_()
            x, context, base = synthetic_inputs()
            base[:, 1] = base[:, 0]
            with torch.no_grad():
                out = model(x, context, base)
            residual = as_numpy(out["mean"])
            require(np.max(abs(residual)) < 2e-10,
                    f"RW mean fails in {mode}: {residual}")
            require(np.max(abs(as_numpy(out["median_mean_z"]))) < 1e-14,
                    "macro median mean is not anchored")
            require(np.all(as_numpy(out["lognormal_sigma"]) > 0),
                    "RW test must retain positive macro uncertainty")
            results.append(dict(mode=mode, standardized_price_mean=residual.tolist(),
                                minimum_macro_sigma=float(out["lognormal_sigma"].min())))
        return {"cases": results,
                "scope": "No clipping/floors active. Plugin's mean-book point need not share the Gamma-reference RW identity."}

    check("rw_conditional_mean_embedding_with_positive_macro_variance", rw_reference)

    def production_dtypes_and_modes():
        results = []
        for mode in ("stochastic", "direct", "no_supply", "plugin", "unseparated"):
            model = construct_model(module, mode=mode, dtype=torch.float32)
            x, context, base = synthetic_inputs(dtype=torch.float32)
            out = model(x, context, base)
            labels = make_labels(base)
            loss, pieces = module.factorized_loss(out, **labels, cfg=CFG)
            require(torch.isfinite(loss), f"nonfinite {mode} production loss")
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            require(all(bool(torch.isfinite(g).all()) for g in grads),
                    f"nonfinite gradient in {mode} production dtype")
            w = out["weights"]
            require(bool((w >= 0).all()) and bool(torch.isfinite(w).all()),
                    f"invalid mixture weights for {mode}")
            require(float((w.sum(-1)-1).abs().max()) < 2e-7,
                    f"mixture weights not normalized for {mode}")
            require(bool(torch.isfinite(out["lognormal_mu_ratio"]).all()),
                    f"invalid price component for {mode}")
            require(bool((1+out["rprice"]*out["mean"] > 0).all()),
                    f"nonpositive arithmetic mean for {mode}")
            if mode == "direct":
                require(bool(torch.isnan(out["mechanical_failure"]).all()),
                        "pure direct is expected to have no mechanical failure law")
            else:
                require(bool(torch.isfinite(out["mechanical_failure"]).all()),
                        f"invalid mechanical diagnostic for {mode}")
            results.append(dict(mode=mode, parameter_dtype=str(next(model.parameters()).dtype),
                                base_dtype=str(base.dtype), loss=float(loss),
                                price_components=int(w.shape[1]),
                                max_weight_sum_error=float((w.sum(-1)-1).abs().max())))
        return {"cases": results,
                "note": "Direct's intentionally absent mechanical diagnostic is NaN and never enters its loss."}

    check("all_modes_float32_network_double_base_finite_and_normalized", production_dtypes_and_modes)

    def bounded_micro_parameters():
        results = []
        for sign in (-1., 1.):
            model = construct_model(module)
            raw = np.zeros((2, 10))
            raw[:, 3] = sign*100
            raw[:, 4] = sign*100
            with torch.no_grad():
                model.micro_head.weight.zero_()
                model.micro_head.bias.copy_(torch.from_numpy(raw.ravel()))
                model.supply_to_bid_volume.weight.zero_()
            x, context, base = synthetic_inputs(n=1)
            base[0, 1] = base[0, 0]
            base[0, 3] = .001
            base[0, 4] = 1.
            with torch.no_grad():
                out = model(x, context, base)
            q_atoms = as_numpy(out["basis_atoms_z"])[0]*.001+1.
            q_atoms = q_atoms.reshape(2, 5)
            logr = np.log(q_atoms)
            spans = np.diff(logr, axis=-1)/(.001*1.5)
            # Independent Gamma reference mean recovers the location term.
            ref_w, _ = scipy_clearing_weights(np.full((2, 5), .2),
                as_numpy(out["concentration"])[0], np.ones(2),
                float(base[0, 0]), np.array([.5, .5]))
            ref_w = ref_w.reshape(2, 5)
            ref_w /= ref_w.sum(-1, keepdims=True)
            locations = np.log((ref_w*q_atoms).sum(-1))/.001
            require(np.min(spans) >= .25-1e-8 and np.max(spans) <= 4+1e-8,
                    f"effective micro span outside [.25,4]: {spans.tolist()}")
            require(np.max(abs(locations)) <= 4+1e-7,
                    f"effective micro location outside [-4,4]: {locations.tolist()}")
            results.append(dict(raw_sign=sign, inferred_spans=spans.tolist(),
                                inferred_locations=locations.tolist()))
        return {"cases": results, "uses_no_implementation_internal_location_or_span": True}

    check("micro_span_and_location_are_bounded_from_observable_atoms", bounded_micro_parameters)

    def full_gradients():
        model = construct_model(module)
        x, context, base = synthetic_inputs(n=3)
        labels = make_labels(base)
        with torch.no_grad():
            model.macro_head.bias.copy_(torch.tensor([.13, -.2], dtype=torch.float64))
            model.micro_head.bias[1] = .1
            model.micro_head.bias[3] = .02
            model.micro_head.bias[4] = -.1
        def objective():
            return module.factorized_loss(model(x, context, base), **labels, cfg=CFG)[0]
        loss = objective()
        loss.backward()
        selections = [("macro_head.bias", (0,)), ("macro_head.bias", (1,)),
                      ("micro_head.bias", (0,)), ("micro_head.bias", (1,)),
                      ("micro_head.bias", (2,)), ("micro_head.bias", (3,)),
                      ("micro_head.bias", (4,)), ("micro_head.bias", (5,)),
                      ("micro_head.bias", (8,)),
                      ("supply_to_bid_volume.weight", (0, 0)),
                      ("encoder.combine.0.weight", (0, 0))]
        parameters = dict(model.named_parameters())
        results = []
        for name, index in selections:
            p = parameters[name]
            analytic = float(p.grad[index])
            original = float(p[index])
            eps = 2e-5
            with torch.no_grad():
                p[index] = original+eps
                plus = float(objective())
                p[index] = original-eps
                minus = float(objective())
                p[index] = original
            fd = (plus-minus)/(2*eps)
            results.append(dict(parameter=name, index=list(index),
                                **numeric_check(analytic, fd, 3e-5, 3e-3)))
        return {"synthetic_loss": float(loss), "gradients": results,
                "tolerance_note": "Includes upstream Gamma's approximate shape derivative; first-order gradients only."}

    check("full_composite_loss_selected_gradients_vs_central_difference", full_gradients)

    def missing_labels_safe():
        model = construct_model(module)
        x, context, base = synthetic_inputs()
        patterns = [
            dict(completed=[1, 0, 1, 0], has_median=[1, 1, 0, 0],
                 has_basis=[1, 0, 0, 0], has_bid=[1, 0, 1, 1]),
            dict(completed=[0]*4, has_median=[0]*4,
                 has_basis=[0]*4, has_bid=[0]*4),
        ]
        results = []
        for pattern in patterns:
            labels = make_labels(base)
            for name, values in pattern.items():
                labels[name] = torch.tensor(values, dtype=torch.bool)
            dirty = {key: value.clone() for key, value in labels.items()}
            for label_name, mask_name in [("target", "completed"),
                      ("median_z", "has_median"), ("median_ratio", "has_median"),
                      ("basis_z", "has_basis"), ("future_bid_ratio", "has_bid")]:
                dirty[label_name][~labels[mask_name]] = float("nan")
            model.zero_grad(set_to_none=True)
            dirty_loss, pieces = module.factorized_loss(model(x, context, base), **dirty, cfg=CFG)
            dirty_loss.backward()
            dirty_grads = {name: p.grad.detach().clone() for name, p in model.named_parameters()
                           if p.grad is not None}
            require(bool(torch.isfinite(dirty_loss)), "NaN labels contaminate loss")
            require(all(bool(torch.isfinite(g).all()) for g in dirty_grads.values()),
                    "NaN labels contaminate gradients")
            require(all(math.isfinite(v) for v in pieces.values()),
                    "NaN labels contaminate logged pieces")
            model.zero_grad(set_to_none=True)
            clean_loss, _ = module.factorized_loss(model(x, context, base), **labels, cfg=CFG)
            clean_loss.backward()
            require(abs(float(clean_loss-dirty_loss)) < 1e-12,
                    "masked label contents alter scalar loss")
            gradient_diff = max(float((p.grad-dirty_grads[name]).abs().max())
                                for name, p in model.named_parameters() if p.grad is not None)
            require(gradient_diff < 1e-12, "masked label contents alter gradients")
            if not any(pattern["completed"]+pattern["has_median"]+
                       pattern["has_basis"]+pattern["has_bid"]):
                require(float(dirty_loss) == 0., "all-missing batch has nonzero loss")
            results.append(dict(pattern=pattern, loss=float(dirty_loss),
                                maximum_gradient_difference=gradient_diff,
                                pieces=pieces))
        return {"cases": results}

    check("nan_missing_labels_and_all_missing_batch_are_safe", missing_labels_safe)

    def forward_boundary_and_pure_direct():
        parameters = list(inspect.signature(module.FactorizedDemandNetwork.forward).parameters)
        require(parameters == ["self", "x", "context", "base"],
                f"unexpected forward inputs: {parameters}")
        model = construct_model(module)
        x, context, base = synthetic_inputs()
        before = model(x, context, base)
        labels = make_labels(base)
        changed = {key: value.clone() for key, value in labels.items()}
        changed["median_z"] += .8
        changed["median_ratio"] += .15
        changed["basis_z"] += .5
        # Evaluate alternate labels without parameter updates; a label argument
        # must not be cached into a later prediction.
        module.factorized_loss(before, **changed, cfg=CFG)
        after = model(x, context, base)
        difference = float((before["mean"]-after["mean"]).abs().max())
        require(difference == 0., "auxiliary labels affect later forward state")
        # Pure direct must bypass the Gamma operator, including reference centering.
        direct = construct_model(module, mode="direct")
        saved = module.clearing_masses
        def forbidden_gamma(*args, **kwargs):
            raise AssertionError("direct invoked Gamma clearing")
        try:
            module.clearing_masses = forbidden_gamma
            output = direct(x, context, base)
            require(bool(torch.isfinite(output["mean"]).all()), "invalid pure-direct forecast")
        finally:
            module.clearing_masses = saved
        return dict(forward_parameters=parameters, max_prediction_change_after_label_changes=difference,
                    pure_direct_bypasses_gamma_in_probability_and_centering=True,
                    scope="Forward API and absence of label-state mutation only. No claim that upstream feature construction is leakage-free; no real dataset read.")

    check("future_median_not_in_forward_api_and_direct_really_bypasses_gamma", forward_boundary_and_pure_direct)

    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path,
                        default=Path(__file__).resolve().with_name("factorized_models.py"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--mc-samples", type=int, default=200000)
    args = parser.parse_args()
    require(args.mc_samples >= 10000, "use at least 10000 Monte Carlo draws")
    model_path = args.model_path.resolve()
    output_dir = args.output_dir.resolve()
    source_files = {"factorized_models.py": model_path,
                    "clearing_models.py": model_path.parents[1]/"clearing_models.py",
                    "verify_factorized_operator.py": Path(__file__).resolve()}
    snapshots = {name: path.read_bytes() for name, path in source_files.items()}
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = output_dir/"operator_check_runs"/timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    for name, data in snapshots.items():
        (run_dir/name).write_bytes(data)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    spec = importlib.util.spec_from_file_location("factorized_operator_under_test", model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    checks = run_checks(module, rng, args.mc_samples)
    changed = [name for name, path in source_files.items() if path.read_bytes() != snapshots[name]]
    checks["source_bytes_stable_during_execution"] = {"pass": not changed,
                                                       "changed_files": changed}
    status = "PASS" if all(item["pass"] for item in checks.values()) else "FAIL"
    record = {
        "status": status, "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Synthetic operator and first-order gradient validation only; no real data, evaluation reads, training, optimizer steps or Git writes.",
        "not_validated": ["feature-builder time availability", "real evaluation or predictive superiority",
                          "market or private-book identification", "conditional independence truth",
                          "rare-event accuracy outside tested numerical cases", "higher-order gradients"],
        "seed": args.seed, "mc_samples_per_case": args.mc_samples,
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "torch": torch.__version__,
        "device": "cpu", "source_model_path": str(model_path),
        "source_sha256": {name: digest(data) for name, data in snapshots.items()},
        "source_snapshot_directory": str(run_dir),
        "checks": checks,
    }
    serialized = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    (run_dir/"FACTOR_OPERATOR_CHECK.json").write_text(serialized)
    latest = output_dir/"FACTOR_OPERATOR_CHECK.json"
    if latest.exists():
        previous = latest.read_bytes()
        (run_dir/"previous_latest_receipt.json").write_bytes(previous)
    temporary = output_dir/(".factor_operator_check_"+timestamp+".tmp")
    temporary.write_text(serialized)
    temporary.replace(latest)
    print(json.dumps({"status": status, "checks": len(checks),
                      "passed": sum(c["pass"] for c in checks.values()),
                      "receipt": str(latest), "immutable_receipt": str(run_dir/"FACTOR_OPERATOR_CHECK.json"),
                      "tested_model_sha256": record["source_sha256"]["factorized_models.py"]}, indent=2), flush=True)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
