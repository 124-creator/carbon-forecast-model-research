"""Only algebraic/synthetic SOLID boundary checks; no market or model fitting."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent


def check_kernel_equivalence():
    rng=np.random.default_rng(202610179)
    n,d,h=9,6,20
    z=rng.normal(0,.5,(n,d));z[:,-1]=1.
    q=rng.normal(0,.5,d);q[-1]=1.
    y=rng.normal(0,.6,(n,h))
    initial=rng.normal(0,.1,(d,h))
    weight=rng.uniform(.2,1.3,n)
    error=y-z@initial
    lr=.07
    # Author non-AMP shape: each sample MSE averages H, all gradients are
    # evaluated at the same initial head then summed before one manual step.
    gradient=sum(weight[i]*2/h*np.outer(z[i],z[i]@initial-y[i]) for i in range(n))
    one_step=q@(initial-lr*gradient)
    kernel=q@initial+2*lr/h*((z@q)*weight)@error
    one_error=float(np.max(np.abs(one_step-kernel)))
    assert one_error<1e-13
    epochs=4
    repeated_fixed=q@(initial-lr*epochs*gradient)
    repeated_kernel=q@initial+epochs*2*lr/h*((z@q)*weight)@error
    assert np.allclose(repeated_fixed,repeated_kernel,atol=1e-13,rtol=0.)
    # A genuinely sequential multi-step variant is a different spectral filter.
    a=z.T@(weight[:,None]*z)
    b=z.T@(weight[:,None]*y)
    eta=2*lr/h
    current=initial.copy()
    for _ in range(epochs):
        current-=eta*(a@current-b)
    eigen,vec=np.linalg.eigh(a)
    filt=(1-(1-eta*eigen)**epochs)/eigen
    p=(vec*filt)@vec.T
    spectral=q@initial+q@p@z.T@(weight[:,None]*error)
    multi_error=float(np.max(np.abs(q@current-spectral)))
    assert multi_error<1e-13
    assert np.max(np.abs(q@current-repeated_fixed))>1e-7
    penalty=.4
    proximal=np.linalg.solve(a+penalty*np.eye(d),b+penalty*initial)
    ridge_kernel=q@initial+q@np.linalg.solve(a+penalty*np.eye(d),z.T@(weight[:,None]*error))
    ridge_error=float(np.max(np.abs(q@proximal-ridge_kernel)))
    assert ridge_error<1e-13
    # Input-dependent inverse normalization is an affine Jacobian, so the same
    # identity becomes matrix-valued rather than disappearing.
    pcount=8
    jac=rng.normal(0,.4,(n,h,pcount))
    jq=rng.normal(0,.4,(h,pcount))
    theta=rng.normal(0,.1,pcount)
    offset=rng.normal(0,.2,(n,h))
    offsetq=rng.normal(0,.2,h)
    residual=y-(np.einsum("nhp,p->nh",jac,theta)+offset)
    update=sum(weight[i]*(jac[i].T@residual[i]) for i in range(n))
    direct=jq@(theta+eta*update)+offsetq
    block_kernel=jq@theta+offsetq+eta*sum(weight[i]*(jq@jac[i].T@residual[i]) for i in range(n))
    block_error=float(np.max(np.abs(direct-block_kernel)))
    assert block_error<1e-13
    return dict(
        samples=n,head_features_including_bias=d,horizon=h,
        author_style_unnormalized_one_step_max_error=one_error,
        repeated_fixed_gradient_equals_step_multiplier=True,
        actual_sequential_steps=epochs,
        actual_multistep_spectral_kernel_max_error=multi_error,
        repeated_fixed_vs_actual_multistep_difference=float(np.max(np.abs(q@current-repeated_fixed))),
        proximal_head_ridge_kernel_max_error=ridge_error,
        affine_normalization_matrix_kernel_max_error=block_error,
        restriction="Exact for a head affine in its adapted parameters with fixed features, normalization and selected weights. Joint nonlinear feature updates are outside the equivalence.",
    )


def check_pure_variance():
    records=[]
    for sigma in (np.array([1.,3.]),np.array([1.,100.])):
        variance=sigma*sigma
        marginal=variance.mean()
        # Same Gaussian conditional and moment-matched marginal KL score as
        # Algorithm 1; conditional means are identically zero.
        component=.5*(np.log(marginal/variance)+variance/marginal-1.)
        direct=float(component.mean())
        identity=float(.5*(np.log(marginal)-np.log(variance).mean()))
        assert direct>0 and abs(direct-identity)<1e-13
        records.append(dict(context_standard_deviations=sigma.tolist(),
                            conditional_means=[0.,0.],
                            gaussian_KL_score=direct,
                            exact_true_MI_upper_bound=float(np.log(2.)),
                            gaussian_score_exceeds_true_MI_upper_bound=bool(direct>np.log(2.)),
                            Bayes_conditional_mean_correction=0.))
    # Discrete exact information example, separate from the Gaussian calculation.
    # C0 has residual +/-1, C1 has +/-3, so |R| identifies C but means are zero.
    c=np.array([0,0,1,1]);r=np.array([-1.,1.,-3.,3.])
    prob=np.full(4,.25)
    true_mi=sum(prob[i]*np.log(prob[i]/(.5*.25)) for i in range(4))
    assert abs(true_mi-np.log(2.))<1e-13
    assert all(r[c==k].mean()==0 for k in (0,1))
    # In a conditionally mean-correct model, an independent finite-sample local
    # intercept estimate adds variance. This is a population calculation only.
    n,penalty=10,1.
    bayes_risk=5.
    extra=bayes_risk/(n*(1+penalty)**2)
    assert extra>0
    return dict(
        gaussian_cases=records,
        exact_discrete_pure_variance_MI=float(true_mi),
        exact_discrete_conditional_mean_energy=0.,
        finite_sample_local_ridge_intercept=dict(
            training_samples_per_context=n,ridge_penalty=penalty,
            base_Bayes_MSE=bayes_risk,expected_added_MSE=extra,
            adapted_expected_MSE=bayes_risk+extra),
        restriction="A variance-sensitive distribution diagnostic is not evidence of reducible squared-error conditional-mean bias. Gaussian-to-Gaussian KL is not generally the exact mutual information of a Gaussian mixture.",
    )


def check_same_input_different_map():
    x=np.array([-2.,-1.,0.,1.,2.])
    y_a=x.copy()
    y_b=-x
    assert np.array_equal(x,x.copy())
    q=1.
    old_nearest=x[np.argmin((x-q)**2)]
    old_prediction=y_a[np.argmin((x-q)**2)]
    next_truth=-q
    wrong_error=float((old_prediction-next_truth)**2)
    assert old_nearest==q and wrong_error==4.
    return dict(
        historical_context_values=x.tolist(),
        same_contexts_next_regime=x.tolist(),
        historical_conditional_means=y_a.tolist(),
        next_regime_conditional_means=y_b.tolist(),
        query=q,nearest_context_distance=0.,
        old_map_nearest_prediction=float(old_prediction),new_map_conditional_mean=next_truth,
        squared_error_despite_zero_input_distance=wrong_error,
        restriction="Identical input distributions and zero nearest-neighbor distance do not identify an unobserved change in the conditional future map.",
    )


def check_delayed_observational_equivalence():
    rng=np.random.default_rng(202610180)
    horizon,current,delta=20,60,.1
    past=100+np.r_[0.,np.cumsum(rng.normal(0,.1,current))]
    future_noise=rng.normal(0,.1,horizon)
    world_a=np.r_[past,past[-1]+np.cumsum(delta+future_noise)]
    world_b=np.r_[past,past[-1]+np.cumsum(-delta+future_noise)]
    candidates=np.arange(20,current-horizon+1)
    labels_a=world_a[candidates[:,None]+np.arange(1,horizon+1)]-world_a[candidates,None]
    labels_b=world_b[candidates[:,None]+np.arange(1,horizon+1)]-world_b[candidates,None]
    assert np.array_equal(world_a[:current+1],world_b[:current+1])
    assert np.array_equal(labels_a,labels_b)
    query_a=np.diff(world_a[current-20:current+1])
    query_b=np.diff(world_b[current-20:current+1])
    assert np.array_equal(query_a,query_b)
    means=np.array([-horizon*delta,horizon*delta])
    possible=np.linspace(-4.,4.,801)
    average_risk=((possible[:,None]-means[None,:])**2).mean(1)
    lower=(horizon*delta)**2
    assert np.isclose(average_risk.min(),lower)
    # Label timing nuance: one new quote completes an old full vector
    # immediately; it does not create a 20-step total information blackout.
    first_new_quote=current+1
    newly_mature_origin=first_new_quote-horizon
    assert newly_mature_origin+horizon==first_new_quote
    difference=float(world_a[first_new_quote]-world_b[first_new_quote])
    assert difference!=0.
    return dict(
        H=horizon,current_origin=current,latest_fully_mature_origin=current-horizon,
        mature_origin_count=len(candidates),
        visible_history_identical=True,all_fully_mature_labels_identical=True,
        current_input_identical=True,
        future_H_conditional_means=means.tolist(),
        minimax_average_excess_squared_error_lower_bound=lower,
        newly_mature_origin_after_one_new_quote=newly_mature_origin,
        newly_mature_vector_contains_latest_quote=True,
        next_quote_can_begin_to_distinguish_worlds=True,
        restriction="At the current issue the two futures are observationally indistinguishable. This is a no-assumption limit on every forecaster, not a theorem that SOLID alone fails, and not a claim that H20 causes 20 steps with no new information.",
    )


def check_context_mean_can_miss_conditional_map():
    # Removing variance sensitivity by looking only at E[R|C] also misses
    # zero-intercept slope mistakes: E[R|C]=0 does not imply E[R|X,C]=0.
    x=np.array([-1.,1.,-1.,1.])
    context=np.array([0,0,1,1])
    coefficient=np.array([1.,1.,-1.,-1.])
    residual=coefficient*x
    mean_energy=np.mean([residual[context==k].mean()**2 for k in (0,1)])
    full_signal=float(np.mean(residual**2))
    assert mean_energy==0 and full_signal==1
    return dict(
        context_only_residual_mean_energy=float(mean_energy),
        conditional_mean_signal_given_input_and_context=full_signal,
        restriction="A context-only mean gate may miss slope changes. Estimating E[R|X,C] is ordinary residual regression and requires a new learning rule/evidence before any novelty claim.",
    )


def main():
    result=dict(
        status="PASS",created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        numpy_version=np.__version__,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        kernel_equivalence=check_kernel_equivalence(),
        pure_variance=check_pure_variance(),
        context_distance_counterexample=check_same_input_different_map(),
        H20_delayed_identification=check_delayed_observational_equivalence(),
        context_mean_blindspot=check_context_mean_can_miss_conditional_map(),
        market_file_reads=0,market_model_reads=0,market_fits=0,author_model_execution=False,
        synthetic_neural_fits=0,
        serialization_note="Initial execution passed algebra checks but JSON rejected a NumPy Boolean; cast to Python bool, with no mathematical or model change.",
        decision="No new algorithm supported by these observations. Keep SOLID as an established-method applicability benchmark; do not restart residual-utility memory, horizon competition or renamed standard gates."
    )
    (HERE/"SOLID_THEORETICAL_SYNTHETIC.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
