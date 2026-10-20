"""Minimal DSOF target counterexamples and strictly prequential calibration.

All prices here are generated synthetic Gaussian paths. No market/source-data
file, trained market weight, or training framework is read or imported.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
H, AGE, NOISE_VAR, STUDENT_PROXY_VAR = 20, 5, 1., .02
REMAINING = H-AGE


def analytic_population(drift_var):
    a, r, sigma2, tau2 = AGE, REMAINING, NOISE_VAR, drift_var
    var_prefix = a*a*tau2+a*sigma2
    cov_current = a*r*tau2
    var_current = r*r*tau2+r*sigma2
    right = cov_current/var_prefix
    wrong = 1+right
    def risk(slope):
        return var_current-2*slope*cov_current+slope*slope*var_prefix
    student_contrast_var = r*r*(tau2+STUDENT_PROXY_VAR)
    student_cov = r*r*tau2
    mix = student_cov/student_contrast_var
    result = dict(
        horizon=H, age=AGE, remaining=REMAINING, drift_variance=tau2,
        increment_noise_variance=sigma2,
        prefix_variance=var_prefix,
        prefix_old_tail_covariance=var_prefix+cov_current,
        prefix_current_teacher_revision_covariance=cov_current,
        correct_current_revision_linear_slope=right,
        incorrect_old_tail_linear_slope=wrong,
        raw_teacher_target_mse=var_current,
        correct_gaussian_conditional_target_mse=risk(right),
        wrong_old_tail_correction_target_mse=risk(wrong),
        double_counting_extra_risk_over_correct=risk(wrong)-risk(right),
        teacher_current_student_optimal_convex_weight=mix,
        teacher_current_student_population_target_mse=var_current-student_cov**2/student_contrast_var,
        student_history_proxy_noise_variance=STUDENT_PROXY_VAR,
    )
    assert np.isclose(result["double_counting_extra_risk_over_correct"],var_prefix)
    if tau2 == 0:
        assert right == 0 and risk(wrong) == H*sigma2 and var_current == r*sigma2
    return result


class RevealedPath:
    def __init__(self, prices):
        self._prices = np.asarray(prices)
        self.clock = 0
        self.reads = []

    def reveal(self, now):
        assert self.clock <= now < len(self._prices)
        self.clock = now

    def read(self, index):
        if not 0 <= index <= self.clock:
            raise AssertionError("Unmatured synthetic price requested")
        self.reads.append((self.clock,index))
        return float(self._prices[index])


def make_targets(path, student_history_proxy, state, gaussian_slope):
    """Called at AGE only. Future revision targets are deliberately inaccessible."""
    assert path.clock == AGE
    prefix = path.read(AGE)-path.read(0)
    current_teacher = path.read(AGE)
    u = np.array([1.,prefix])
    ridge = np.diag([1.,1.])
    right_coef = np.linalg.solve(state["xx"]+ridge,state["xy_right"])
    wrong_coef = np.linalg.solve(state["xx"]+ridge,state["xy_wrong"])
    intercept = state["sum_right"]/max(state["n"],1)
    student_difference = REMAINING*student_history_proxy
    mix_weight = np.clip(state["zr"]/(state["zz"]+1.),0.,1.)
    return dict(
        raw_teacher=current_teacher,
        intercept=current_teacher+intercept,
        current_revision_linear=current_teacher+float(u@right_coef),
        gaussian_known_dgp=current_teacher+gaussian_slope*prefix,
        wrong_old_tail_linear=current_teacher+float(u@wrong_coef),
        teacher_student_half=current_teacher+.5*student_difference,
        teacher_student_all=current_teacher+student_difference,
        teacher_student_prequential_mix=current_teacher+mix_weight*student_difference,
    ), dict(prefix=prefix,u=u,current_teacher=current_teacher,student_difference=student_difference,
            current_revision_coefficients=right_coef.tolist(),wrong_old_tail_coefficients=wrong_coef.tolist(),
            mix_weight=float(mix_weight))


def run_prequential(drift_var, seed, episodes=2000, warmup=400):
    rng = np.random.default_rng(seed)
    state = dict(xx=np.zeros((2,2)),xy_right=np.zeros(2),xy_wrong=np.zeros(2),
                 sum_right=0.,n=0,zr=0.,zz=0.)
    population = analytic_population(drift_var)
    losses = {}
    latest_label_time = -1
    checks, forbidden_checks = 0, 0
    example = None
    for episode in range(episodes):
        drift = rng.normal(0,np.sqrt(drift_var))
        increments = drift+rng.normal(0,np.sqrt(NOISE_VAR),size=H)
        prices = np.r_[0.,np.cumsum(increments)]
        # An observable proxy from pre-origin history: never a future label.
        student_proxy = drift+rng.normal(0,np.sqrt(STUDENT_PROXY_VAR))
        path = RevealedPath(prices)
        origin_time = episode*(H+1)
        issue_time = origin_time+AGE
        assert latest_label_time < issue_time
        path.reveal(AGE)
        targets, features = make_targets(path,student_proxy,state,
                                         population["correct_current_revision_linear_slope"])
        # Attempting to use the end label at issue time must be rejected.
        try:
            path.read(H)
            raise AssertionError("Guard did not reject future target")
        except AssertionError as error:
            assert "Unmatured" in str(error)
            forbidden_checks += 1
        altered = prices.copy()
        altered[AGE+1:] += np.linspace(10.,20.,H-AGE)
        future_changed = RevealedPath(altered)
        future_changed.reveal(AGE)
        altered_targets,_ = make_targets(future_changed,student_proxy,state,
                                          population["correct_current_revision_linear_slope"])
        assert targets == altered_targets
        checks += len(targets)
        # Store issued values immutably; maturity supplies a label, not a new prediction.
        issued = dict(targets)
        path.reveal(H)
        truth = path.read(H)
        revision = truth-features["current_teacher"]
        old_tail_error = truth-path.read(0)
        assert np.isclose(old_tail_error,features["prefix"]+revision)
        if episode >= warmup:
            for name,prediction in issued.items():
                losses.setdefault(name,[]).append((prediction-truth)**2)
        # Only now update calibrators for subsequent episodes.
        u,z = features["u"],features["student_difference"]
        state["xx"] += np.outer(u,u)
        state["xy_right"] += u*revision
        state["xy_wrong"] += u*old_tail_error
        state["sum_right"] += revision
        state["n"] += 1
        state["zr"] += z*revision
        state["zz"] += z*z
        latest_label_time = origin_time+H
        assert targets == issued
        if episode == warmup:
            example = dict(episode=episode,issue_time=issue_time,
                           last_used_calibration_target_time=(episode-1)*(H+1)+H,
                           target_maturity_time=origin_time+H,
                           input_prefix=features["prefix"],
                           current_teacher=features["current_teacher"],
                           coefficient_fit_uses_episodes_before=episode,
                           coefficients_before_issue=features["current_revision_coefficients"],
                           issued_pseudo_targets=issued)
        assert all(index <= clock for clock,index in path.reads)
    return dict(
        drift_variance=drift_var,seed=seed,episodes=episodes,warmup_episodes=warmup,
        scored_episodes=episodes-warmup,analytical_population=population,
        prequential_target_mse={name:float(np.mean(value)) for name,value in losses.items()},
        final_current_revision_coefficients=np.linalg.solve(state["xx"]+np.eye(2),state["xy_right"]).tolist(),
        final_wrong_old_tail_coefficients=np.linalg.solve(state["xx"]+np.eye(2),state["xy_wrong"]).tolist(),
        final_teacher_student_mix=float(np.clip(state["zr"]/(state["zz"]+1),0,1)),
        future_perturbation_prediction_checks=checks,
        immature_target_guard_rejections=forbidden_checks,
        issued_pseudo_target_rewrites=0,example=example,
        restriction="These are pseudo-target prediction risks on synthetic paths, not trained-forecaster or market forecast results.",
    )


def check_fast_td_bias():
    # Exact author-style alignment: true prefix then discounted current teacher.
    horizon=np.arange(1,H+1,dtype=float)
    current_drift=.05
    true=horizon*current_drift
    now_price=AGE*current_drift
    gamma=.9 # Illustrative only; no claim about the paper's chosen hyperparameter.
    weights=np.r_[np.ones(AGE),gamma**np.arange(1,REMAINING+1)]
    teacher=np.r_[true[:AGE],np.full(REMAINING,now_price)]
    corrected=np.r_[true[:AGE],now_price+np.arange(1,REMAINING+1)*current_drift]
    student_now_tail=now_price+np.arange(1,REMAINING+1)*current_drift
    student_bootstrap=np.r_[true[:AGE],student_now_tail]
    assert np.array_equal(corrected,student_bootstrap)
    gradient=lambda target:float(np.mean(2*weights*horizon*(true-target)))
    curvature=float(np.mean(2*weights*horizon**2))
    step=.05/curvature
    raw_gradient=gradient(teacher)
    corrected_gradient=gradient(corrected)
    updated_drift=current_drift-step*raw_gradient
    new_issued_squared_bias=float(np.mean((horizon*(updated_drift-current_drift))**2))
    assert raw_gradient > 0 and abs(corrected_gradient) < 1e-14
    assert updated_drift < current_drift and new_issued_squared_bias > 0
    return dict(
        H=H,age=AGE,true_constant_drift=current_drift,illustrative_gamma=gamma,
        raw_teacher_tail=teacher[AGE:].tolist(),correct_revision_tail=corrected[AGE:].tolist(),
        raw_teacher_td_gradient_at_correct_student=raw_gradient,
        corrected_teacher_td_gradient_at_correct_student=corrected_gradient,
        full_current_student_bootstrap_gradient_at_correct_student=gradient(student_bootstrap),
        illustrative_step=step,student_drift_before=current_drift,student_drift_after_raw_teacher_step=updated_drift,
        newly_issued_forecast_mean_squared_bias_after_raw_teacher_step=new_issued_squared_bias,
        previously_issued_forecasts_modified=False,
        limitation="Exact one-parameter deterministic counterexample; does not imply every actual DSOF teacher or student has this bias.",
    )


def check_unpredictable_reversal():
    tau2=.04
    source=analytic_population(tau2)
    xvar=source["prefix_variance"]
    targetvar=source["raw_teacher_target_mse"]
    reversed_cov=-source["prefix_current_teacher_revision_covariance"]
    source_slope=source["correct_current_revision_linear_slope"]
    risk=targetvar-2*source_slope*reversed_cov+source_slope**2*xvar
    optimal=reversed_cov/xvar
    assert risk > targetvar and optimal < 0 < source_slope
    return dict(
        prefix_drift_persists_in_training=True,
        tail_drift_reverses_after_issue_in_evaluation=True,
        observable_prefix_distribution_unchanged=True,
        source_calibration_slope=source_slope,target_oracle_slope=optimal,
        raw_teacher_target_mse=targetvar,
        source_calibrated_target_mse=risk,
        restriction="No pre-issue feature reveals the sign reversal; history-stable conditional-error calibration has no guarantee under this new conditional shift.",
    )


def check_observable_state_interaction():
    """Known state V=+/-1 selects persistent/reversed post-issue drift.

    V is observed before issue; it is not estimated from the future tail.
    Exact second moments suffice for these linear feature-space risks.
    """
    base=analytic_population(.04)
    xvar=base["prefix_variance"]
    covariance=base["prefix_current_teacher_revision_covariance"]
    rvar=base["raw_teacher_target_mse"]
    # Features [1,X,V,VX]. V is independent, centered, unit squared.
    gram=np.diag([1.,xvar,1.,xvar])
    cross=np.array([0.,0.,0.,covariance])
    coefficients=np.linalg.solve(gram,cross)
    linear_coef=np.linalg.solve(gram[:3,:3],cross[:3])
    interaction_risk=float(rvar-cross@coefficients)
    linear_risk=float(rvar-cross[:3]@linear_coef)
    assert np.allclose(coefficients,[0.,0.,0.,.5])
    assert linear_risk == rvar and interaction_risk < linear_risk
    return dict(
        process="prefix increments D+noise; future increments V*D+noise",
        state_values=[-1,1],state_visible_before_issue=True,
        feature_order=["1","prefix X","visible state V","V*X"],
        population_coefficients=coefficients.tolist(),
        no_interaction_population_coefficients=linear_coef.tolist(),
        raw_or_no_interaction_target_mse=linear_risk,
        interaction_target_mse=interaction_risk,
        state_hidden_at_issue_bayes_target_mse=rvar,
        novelty_claim="None: ordinary varying-coefficient/bilinear regression; known-state Gaussian conditioning has the same result.",
        testable_condition="Observable state must alter the prefix-to-current-revision relation in a learnable recurring way.",
    )


def main():
    result=dict(
        status="PASS",created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),numpy_version=np.__version__,
        dsof_target_alignment="old o=t-a: h<=a uses true P_(o+h); h>a uses current teacher T_t(h-a), weighted by gamma^(h-a).",
        author_reference_commit="11cebd980a1f77ef0e1db9ac313dbfe6d37984e9",
        author_implementation_executed=False,
        random_walk=run_prequential(0.,202610175),
        persistent_drift=run_prequential(.04,202610176),
        fast_td_bias_conflict=check_fast_td_bias(),
        unpredictable_conditional_shift=check_unpredictable_reversal(),
        observable_state_interaction=check_observable_state_interaction(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        market_price_reads=0,market_weight_reads=0,market_fits=0,market_optimizer_steps=0,
        synthetic_paths=4000,synthetic_closed_form_calibrator_updates_per_path=1,
        neural_fits=0,
        novelty_claim="None. Correct scalar linear calibration is ordinary conditional-error regression; its Gaussian population form is a Kalman/Bayesian drift update.",
    )
    (HERE/"DSOF_TARGET_SYNTHETIC.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps(dict(
        status=result["status"],
        random_walk_population=result["random_walk"]["analytical_population"],
        random_walk_prequential=result["random_walk"]["prequential_target_mse"],
        persistent_population=result["persistent_drift"]["analytical_population"],
        persistent_prequential=result["persistent_drift"]["prequential_target_mse"],
        bias_conflict=result["fast_td_bias_conflict"],
        reversal=result["unpredictable_conditional_shift"]),indent=2))


if __name__=="__main__":
    main()
