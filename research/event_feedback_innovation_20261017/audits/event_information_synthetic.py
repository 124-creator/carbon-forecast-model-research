"""Algebra and counterexamples for same-target delayed feedback.

Only synthetic arrays are constructed. No market file or original weight is
read, no training framework is imported, and no iterative model fit is run.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def normal_cdf(x):
    return .5 * (1 + math.erf(x / math.sqrt(2)))


def cov(x):
    center = x - x.mean(0)
    return center.T @ center / len(center)


def numerical_rank(x):
    eig = np.linalg.eigvalsh(x)
    tol = max(1e-12, float(np.max(np.abs(eig))) * 1e-10)
    return int(np.sum(eig > tol)), eig


def check_affine_rank():
    rng = np.random.default_rng(202610171)
    h, n = 20, 401
    pa, pb = rng.normal(size=(2, h))
    scale = np.sqrt(np.arange(1, h+1))
    price = rng.normal(.2, .7, size=n)
    loss_difference = ((price[:, None]-pa)**2 - (price[:, None]-pb)**2) / scale**2
    q = -2 * (pa-pb) / scale**2
    constant = (pa**2-pb**2) / scale**2
    affine = constant + price[:, None] * q
    assert np.allclose(loss_difference, affine, atol=2e-14, rtol=1e-14)
    theoretical_sample_cov = np.var(price) * np.outer(q, q)
    assert np.allclose(cov(loss_difference), theoretical_sample_cov, atol=2e-14, rtol=1e-13)
    rank, eigen = numerical_rank(cov(loss_difference))
    assert rank == 1
    raw_loss = (price[:, None] - pa)**2
    raw_rank, _ = numerical_rank(cov(raw_loss))
    assert raw_rank == 2
    absolute_difference = np.abs(price[:, None]-pa) - np.abs(price[:, None]-pb)
    absolute_rank, _ = numerical_rank(cov(absolute_difference))
    assert absolute_rank > 1
    residuals = (price[:, None]-pa) / scale
    residual_rank, _ = numerical_rank(cov(residuals))
    assert residual_rank == 1
    return dict(
        horizons=h, synthetic_possible_prices=n,
        paired_squared_loss_covariance_rank=rank,
        residual_covariance_rank=residual_rank,
        raw_squared_loss_covariance_rank=raw_rank,
        paired_absolute_loss_covariance_rank=absolute_rank,
        maximum_affine_identity_error=float(np.max(np.abs(loss_difference-affine))),
        maximum_covariance_identity_error=float(np.max(np.abs(cov(loss_difference)-theoretical_sample_cov))),
        covariance_eigenvalues=eigen.tolist(),
        restriction="Rank-one covariance is for previsible affine functions of one quote; arbitrary nonlinear losses need not have rank-one covariance.",
    )


def check_false_precision():
    rng = np.random.default_rng(202610172)
    z = rng.standard_normal(100000)
    records = []
    for h in (1, 2, 5, 20):
        threshold = 1.959963984540054
        naive_exact = 2 * (1-normal_cdf(threshold/math.sqrt(h)))
        naive_mc = float(np.mean(np.abs(z)*math.sqrt(h) > threshold))
        correct_mc = float(np.mean(np.abs(z) > threshold))
        assert abs(naive_mc-naive_exact) < .008 and abs(correct_mc-.05) < .008
        records.append(dict(
            duplicate_horizons=h, true_independent_events=100,
            incorrectly_counted_rows=100*h,
            naive_variance_over_true_variance=1/h,
            naive_two_sided_null_rejection_probability=naive_exact,
            synthetic_monte_carlo_rejection=naive_mc,
            event_aware_monte_carlo_rejection=correct_mc,
            event_posterior_variance_for_unit_prior_and_noise=.5,
            naive_diagonal_posterior_variance=1/(1+h),
        ))
    coefficients = {
        "same_sign": np.ones(20),
        "varying_same_sign": np.arange(1, 21, dtype=float),
        "alternating_cancel": np.tile([1., -1.], 10),
    }
    variance_cases = {}
    for name, q in coefficients.items():
        # Equally weighted sum of paired loss contrasts driven by one innovation.
        variance_cases[name] = dict(
            true_variance=float(np.mean(q)**2),
            false_independence_variance=float(np.sum(q*q)/len(q)**2),
            true_over_false=float(np.sum(q)**2/np.sum(q*q)),
        )
    assert variance_cases["same_sign"]["true_over_false"] == 20
    assert variance_cases["alternating_cancel"]["true_variance"] == 0
    return dict(records=records, coefficient_sign_counterexamples=variance_cases,
                restriction="Ignoring covariance can inflate or deflate uncertainty; H-fold inflation is an aligned-duplicate example, not a universal equality.")


def check_gls_equivalence_and_bias():
    h, sigma2 = 20, 2.
    loading = np.linspace(.2, 2, h)
    matrix = sigma2 * np.outer(loading, loading)
    inverse = np.linalg.pinv(matrix)
    differences = []
    for innovation in (-2., -.1, 0., .4, 3.):
        r = loading * innovation
        q = float(r @ inverse @ r)
        differences.append(abs(q - innovation**2/sigma2))
    assert max(differences) < 1e-12
    # Different old origins legitimately have different conditional means.
    current_conditional_mean = 10.
    issued_prices = np.array([10., 8., 6., 4.])
    conditional_old_error_mean = current_conditional_mean-issued_prices
    assert np.array_equal(conditional_old_error_mean, [0., 2., 4., 6.])
    noise = np.array([-1., 0., 1.])
    errors = current_conditional_mean+noise[:, None]-issued_prices[None, :]
    assert np.allclose(errors-errors.mean(0), noise[:, None])
    return dict(
        exact_rank_one_gls_equals_scalar_innovation_max_error=max(differences),
        issued_predictions=issued_prices.tolist(),
        pre_event_current_conditional_mean=current_conditional_mean,
        conditional_old_error_means=conditional_old_error_mean.tolist(),
        expected_naive_average_old_error=float(conditional_old_error_mean.mean()),
        expected_new_innovation=0.,
        error_mean_contrast_norm=float(np.linalg.norm(conditional_old_error_mean-conditional_old_error_mean.mean())),
        restriction="A singular zero-mean duplicate-measurement model is invalid for old forecasts with different information sets unless their predictable revision offsets are included.",
    )


def check_sse_and_gradient_decomposition():
    rng = np.random.default_rng(202610173)
    x = rng.normal(size=(20, 4))
    theta = rng.normal(size=4)
    offset = rng.normal(size=20)
    weight = np.linspace(.5, 1.5, 20)
    price = .7
    a = offset + x @ theta
    wsum = weight.sum()
    abar = float(weight @ a / wsum)
    xbar = weight @ x / wsum
    original = float(np.sum(weight * (price-a)**2))
    mean_term = float(wsum * (price-abar)**2)
    contrast = float(np.sum(weight * (a-abar)**2))
    grad = 2*x.T @ (weight*(a-price))
    grad_mean = 2*wsum*(abar-price)*xbar
    grad_contrast = 2*(x-xbar).T @ (weight*(a-abar))
    assert abs(original-mean_term-contrast) < 1e-12
    assert np.allclose(grad,grad_mean+grad_contrast,atol=1e-12)
    # This is an algebraic update comparison, not an iterative fit.
    eta = .01
    average_loss_update = theta-eta*grad/wsum
    scaled_sum_update = theta-(eta/wsum)*grad
    assert np.allclose(average_loss_update,scaled_sum_update,atol=1e-15)
    # One average prediction discards a real contrast component of task risk.
    xs = np.eye(2)
    ts = np.array([1., -1.])
    full_loss = float(np.mean((xs@ts)**2))
    average_prediction_loss = float(np.mean(xs@ts)**2)
    full_gradient = xs.T@(xs@ts)
    average_gradient = 2*np.mean(xs@ts)*xs.mean(0)
    assert full_loss == 1 and average_prediction_loss == 0
    assert np.linalg.norm(full_gradient) > 1 and np.linalg.norm(average_gradient) == 0
    hessian_rank = int(np.linalg.matrix_rank(2*x.T@(weight[:,None]*x)))
    noise_gradient_cov_rank = int(np.linalg.matrix_rank(np.outer(weight@x,weight@x)))
    assert hessian_rank == 4 and noise_gradient_cov_rank == 1
    return dict(
        weighted_sse_identity_error=abs(original-mean_term-contrast),
        gradient_identity_max_error=float(np.max(np.abs(grad-grad_mean-grad_contrast))),
        mean_sgd_equals_rescaled_sum_sgd_error=float(np.max(np.abs(average_loss_update-scaled_sum_update))),
        deterministic_task_contrast_loss=contrast,
        empirical_risk_hessian_rank=hessian_rank,
        innovation_gradient_noise_rank=noise_gradient_cov_rank,
        discard_contrast_counterexample=dict(task_loss=full_loss,mean_prediction_loss=average_prediction_loss,
                                             task_gradient=full_gradient.tolist(),mean_prediction_gradient=average_gradient.tolist()),
        restriction="One stochastic innovation does not imply one-dimensional empirical-risk curvature or authorize dropping distinct task losses.",
    )


def synthetic_stream(prices, horizon=5):
    issued = {}
    mature_events = []
    state = 0.
    for now in range(len(prices)):
        frozen_old = dict(issued)
        matured = [(key,value) for key,value in issued.items() if key[0]+key[1] == now]
        assert all(origin+lead == now and origin < now for (origin,lead),_ in matured)
        if now:
            # The most recent one-step prediction was irrevocably issued before this quote.
            previous = issued[(now-1,1)]
            innovation = float(prices[now]-previous)
            state = .8*state + .2*innovation
        mature_events.append(dict(event=now, observed_quote_index=now, matured_keys=[list(k) for k,_ in matured]))
        for lead in range(1,horizon+1):
            key = (now,lead)
            assert key not in issued
            issued[key] = float(prices[now]+state*lead/(lead+1))
        assert all(issued[key] == value for key,value in frozen_old.items())
    return issued,mature_events


def check_time_boundary():
    rng = np.random.default_rng(202610174)
    price = np.cumsum(rng.normal(size=40))
    altered = price.copy()
    cut = 18
    altered[cut+1:] += np.linspace(50.,100.,len(altered)-cut-1)
    first,events = synthetic_stream(price)
    second,_ = synthetic_stream(altered)
    frozen_keys = [key for key in first if key[0] <= cut]
    assert all(first[key] == second[key] for key in frozen_keys)
    assert all(len(event["matured_keys"]) <= 5 for event in events)
    return dict(
        synthetic_quotes=len(price), horizon=5,
        future_perturbation_cut=cut, unchanged_issued_predictions=len(frozen_keys),
        event_group_keys_checked=sum(len(e["matured_keys"]) for e in events),
        events=events,
        historical_issued_forecast_rewrites=0,
        future_prices_used_for_update=0,
        description="Fixed illustrative state arithmetic, not a proposed learned model; update follows quote arrival and affects only newly issued forecasts.",
    )


def check_bcm_population_counterexample():
    a = np.array([-1., -1., 1., 1.])
    b = np.array([-1., 1., -1., 1.])
    records = []
    for shrink in (0., .1, 1.):
        y = .6*a+.2*b
        fine = y/(1+shrink)
        coarse = .6*a/(1+shrink)
        conditional_means = [float(np.mean((fine-coarse)[a == value])) for value in (-1,1)]
        assert np.allclose(conditional_means,0,atol=1e-15)
        changed = -.6*a+.2*b
        rw = float(np.mean(changed**2))
        risk = float(np.mean((fine-changed)**2))
        assert risk > rw
        # An arbitrary common coarse error is invisible to this difference penalty.
        common_wrong = 3*a
        assert np.allclose((fine+common_wrong)-(coarse+common_wrong),fine-coarse)
        records.append(dict(shrink=shrink,exact_conditional_difference=conditional_means,
                            training_mse=float(np.mean((fine-y)**2)),
                            next_state_mse=risk,next_state_rw_mse=rw,
                            next_state_relative_mse=risk/rw))
    return dict(records=records,
                conclusion="The shrink-regularized full/coarse population optimum has zero tower moment, even when a common predictive relation reverses in the next state.")


def check_shared_local_identifiability():
    markets = 3
    design = np.c_[np.ones(markets),np.eye(markets)]
    null_direction = np.r_[1.,-np.ones(markets)]
    assert np.array_equal(design@null_direction,np.zeros(markets))
    shared_world = np.r_[1.,np.zeros(markets)]
    local_world = np.r_[0.,np.ones(markets)]
    assert np.array_equal(design@shared_world,design@local_world)
    one_market = design[[0]]
    all_market_rank = int(np.linalg.matrix_rank(design))
    one_market_rank = int(np.linalg.matrix_rank(one_market))
    assert all_market_rank == markets and one_market_rank == 1
    # Sum-local-zero defines an allocation; it does not establish causal origin.
    truly_local_effect = np.array([1.,0.,0.])
    constrained_shared = float(truly_local_effect.mean())
    constrained_local = truly_local_effect-constrained_shared
    assert np.allclose(constrained_shared+constrained_local,truly_local_effect)
    assert np.isclose(constrained_local.sum(),0)
    return dict(
        markets=markets,shared_plus_free_local_parameter_count=markets+1,
        one_market_event_design_rank=one_market_rank,
        all_market_event_design_rank=all_market_rank,
        null_direction=null_direction.tolist(),
        indistinguishable_shared_world=shared_world.tolist(),
        indistinguishable_local_world=local_world.tolist(),
        local_only_actual_effect=truly_local_effect.tolist(),
        sum_local_zero_allocated_shared=constrained_shared,
        sum_local_zero_allocated_locals=constrained_local.tolist(),
        restriction="An identifying centering convention allocates coefficients but does not distinguish common causal drift from simultaneous local drift or common shocks.",
    )


def main():
    result = dict(
        status="PASS",created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        numpy_version=np.__version__,
        paired_rank=check_affine_rank(),
        false_precision=check_false_precision(),
        gls_and_predictable_revision=check_gls_equivalence_and_bias(),
        sse_and_gradients=check_sse_and_gradient_decomposition(),
        immutable_issuance_boundary=check_time_boundary(),
        bcm_population_counterexample=check_bcm_population_counterexample(),
        shared_local_identifiability=check_shared_local_identifiability(),
        market_price_file_reads=0,market_weight_reads=0,market_fits=0,
        market_optimizer_steps=0,iterative_synthetic_model_fits=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (HERE/"EVENT_INFORMATION_SYNTHETIC.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ("status","paired_rank","false_precision","gls_and_predictable_revision",
                                          "sse_and_gradients","bcm_population_counterexample")},indent=2))


if __name__ == "__main__":
    main()
