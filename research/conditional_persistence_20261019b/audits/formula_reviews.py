"""Synthetic-only numerical checks for the independent filter and score formulas.

No market values are loaded.  Numerical integration provides a definition-based
CRPS check, rather than only comparing two algebraically similar programs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import t as student

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from gas_filter import (GASParameters, GASFit, gas_update, filter_gas, forecast_gas,
                        penalized_objective, fit_gas_normalizer)
from mixture_metrics import (gaussian_mixture_crps, gaussian_mixture_quantiles,
                             lognormal_mixture_crps, score_mixture, equal_atom_crps, score_atoms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "audits" / "formula_reviews.json")
    args = parser.parse_args()
    evidence = {}

    p = GASParameters(.4, .7, .8, .9, .03, .02, .1, .8)
    x = np.array([.4, -.8, 1.2, .2, -3., .7])
    fit = GASFit(p, 1.5, 2.3, {"synthetic_only": True})
    hist = filter_gas(x, fit)
    changed = x.copy(); changed[3:] = [40., -90., 110.]
    later = filter_gas(changed, fit)
    for name in hist.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(hist, name)[:3], getattr(later, name)[:3])
    evidence["fixed_parameter_prefix_causality"] = True
    np.testing.assert_allclose(hist.normalized, (x - hist.prior_mean) / hist.prior_std, atol=2e-15)
    np.testing.assert_allclose(hist.prior_std[0], 2.3 * np.sqrt(.8 * 20 / 18), atol=1e-15)
    evidence["student_scale_to_actual_variance"] = True

    m1, sd1 = forecast_gas(fit, hist, horizon=1)
    np.testing.assert_allclose(m1, fit.input_mean + fit.input_std * hist.next_mean_internal)
    np.testing.assert_allclose(sd1, fit.input_std * np.sqrt(hist.next_scale2_internal * 20 / 18))
    mh, sdh = forecast_gas(fit, hist, horizon=5)
    em, ev = hist.next_mean_internal.copy(), hist.next_scale2_internal.copy()
    for _ in range(4):
        em, ev = p.omega_mean + p.beta_mean * em, p.omega_scale + p.beta_scale * ev
    np.testing.assert_allclose(mh, fit.input_mean + fit.input_std * em, atol=2e-15)
    np.testing.assert_allclose(sdh, fit.input_std * np.sqrt(ev * 20 / 18), atol=2e-15)
    evidence["horizon_index_and_pure_AR_forecast"] = True

    mean_score = quad(lambda z: z / (1 + z*z/20) * student.pdf(z, 20), -np.inf, np.inf)[0]
    scale_score = quad(lambda z: ((21*z*z/(20+z*z))-1) * student.pdf(z, 20), -np.inf, np.inf)[0]
    assert abs(mean_score) < 1e-10 and abs(scale_score) < 1e-10
    evidence["working_Student_expected_scores"] = [float(mean_score), float(scale_score)]
    _, no_data_update = gas_update(p.initial_mean, p.initial_mean, p.initial_scale2, p)
    pure_AR = p.omega_scale + p.beta_scale * p.initial_scale2
    assert abs(no_data_update - pure_AR) > 1e-3
    evidence["mean_plugin_is_not_pure_AR"] = True

    # Evaluate the T-1-target objective using scipy's density implementation.
    u = np.array([-.2, .5, 1.1])
    mean, scale2 = p.initial_mean, p.initial_scale2
    terms = []
    for i in range(2):
        nm, ns = gas_update(u[i], mean, scale2, p)
        term = -(p.gamma_mean+p.gamma_scale)*student.logpdf(u[i+1], p.nu, loc=nm, scale=np.sqrt(ns))
        term += .5*(1-p.gamma_mean)*(nm-mean)**2 + .5*(1-p.gamma_scale)*(ns-scale2)**2
        terms.append(term); mean, scale2 = nm, ns
    obj = penalized_objective(p.vector(), u)
    np.testing.assert_allclose(obj, np.mean(terms), atol=2e-15)
    evidence["genuine_next_target_objective"] = True

    # Numerically integrate the defining CDF expression for nonzero means.
    w = np.array([[.4, .6], [.7, .3]])
    m = np.array([[-.3, .8], [-1.2, .15]])
    s = np.array([[.2, .7], [.9, .3]])
    y = np.array([.45, -1.1])
    value = gaussian_mixture_crps(y, w, m, s)
    errors = []
    for i in range(2):
        cdf = lambda a: np.sum(w[i] * ndtr((a-m[i])/s[i]))
        sf = lambda a: np.sum(w[i] * ndtr((m[i]-a)/s[i]))
        numerical = quad(lambda a: cdf(a)**2, -np.inf, y[i], epsabs=1e-10)[0]
        numerical += quad(lambda a: sf(a)**2, y[i], np.inf, epsabs=1e-10)[0]
        errors.append(abs(numerical-value[i]))
    assert max(errors) < 1e-9
    evidence["Gaussian_CRPS_CDF_integration_max_error"] = float(max(errors))

    lm, ls = np.log(60.) + m*.12, s*.12
    prices = np.array([67., 52.])
    value_price = lognormal_mixture_crps(prices, w, lm, ls)
    price_errors = []
    for i in range(2):
        cdf = lambda a: np.sum(w[i]*ndtr((np.log(a)-lm[i])/ls[i])) if a > 0 else 0.
        sf = lambda a: np.sum(w[i]*ndtr((lm[i]-np.log(a))/ls[i]))
        numerical = quad(lambda a: cdf(a)**2, 0., prices[i], epsabs=1e-9)[0]
        numerical += quad(lambda a: sf(a)**2, prices[i], np.inf, epsabs=1e-9)[0]
        price_errors.append(abs(numerical-value_price[i]))
    assert max(price_errors) < 1e-8
    evidence["lognormal_CRPS_CDF_integration_max_error"] = float(max(price_errors))

    old_path = ROOT.parent / "distribution_update_learning_20261019" / "source" / "scores.py"
    spec = importlib.util.spec_from_file_location("old_score_formula_check", old_path)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    np.testing.assert_allclose(gaussian_mixture_crps(y, w, np.zeros_like(m), s), old.crps(y, w, s), atol=2e-15)
    np.testing.assert_allclose(value_price, old.lognormal_crps(prices, lm, ls, w), atol=2e-13)
    evidence["old_symmetric_CRPS_and_price_formula_crosscheck"] = True

    q = gaussian_mixture_quantiles(w, m, s)
    for j, level in enumerate((.05, .95)):
        np.testing.assert_allclose(np.sum(w * ndtr((q[:, j, None]-m)/s), axis=1), level, atol=2e-15)
    result = score_mixture(y, w, m, s, origin_price=np.array([60., 70.]), unit=np.array([.05, .08]))
    manual_mean = (w * np.exp(np.log([60., 70.])[:,None]+np.array([.05,.08])[:,None]*m+
                             .5*(np.array([.05,.08])[:,None]*s)**2)).sum(axis=1)
    np.testing.assert_allclose(result["mean_price"], manual_mean, atol=2e-14)
    evidence["mixture_quantiles_and_exact_price_mean"] = True

    atoms = np.array([[-2.,-.1,.4,.4,1.],[-.8,-.3,0.,.2,2.1]])
    direct = np.mean(np.abs(atoms-y[:,None]),axis=1)-.5*np.mean(np.abs(atoms[:,:,None]-atoms[:,None,:]),axis=(1,2))
    np.testing.assert_allclose(equal_atom_crps(y,atoms),direct,atol=3e-16)
    atomic = score_atoms(y,atoms,origin_price=np.array([60.,70.]),unit=np.array([.05,.08]))
    assert "normalized_NLL" not in atomic and "price_NLL" not in atomic
    evidence["atomic_pair_sum_and_undefined_density_NLL"] = True

    # Deterministic synthetic optimizer smoke check, not market fitting.
    rng = np.random.default_rng(20261019)
    artificial = rng.standard_t(8, 120) * np.linspace(.5, 1.2, 120)
    synthetic_fit = fit_gas_normalizer(artificial, maxiter=8, maxfev=180)
    assert synthetic_fit.metadata["genuine_next_targets"] == 119
    assert np.isfinite(filter_gas(artificial, synthetic_fit).normalized).all()
    evidence["synthetic_fit_smoke"] = {"n": 120, "nfev": synthetic_fit.metadata["nfev"],
        "optimizer_success": synthetic_fit.metadata["optimizer_success"],
        "budget_is_deliberately_smaller_than_market": True}

    if importlib.util.find_spec("torch") is not None:
        import torch
        from mixture_metrics import torch_gaussian_mixture_crps
        yt, wt, mt, st = [torch.tensor(a, dtype=torch.float64, requires_grad=i>1)
                          for i,a in enumerate((y,w,m,s))]
        np.testing.assert_allclose(torch_gaussian_mixture_crps(yt,wt,mt,st).detach().numpy(), value, atol=2e-15)
        assert torch.autograd.gradcheck(lambda a,b: torch_gaussian_mixture_crps(yt,wt,a,b).sum(), (mt,st))
        evidence["torch_numpy_and_autograd_gradcheck"] = True
    else:
        evidence["torch_numpy_and_autograd_gradcheck"] = "not_run_torch_unavailable_in_this_interpreter"

    evidence["source_sha256"] = {name: hashlib.sha256((ROOT/"source"/name).read_bytes()).hexdigest()
                                 for name in ("gas_filter.py", "mixture_metrics.py")}
    evidence["market_values_read"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(evidence, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
