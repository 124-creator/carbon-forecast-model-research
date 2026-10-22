"""Fixed-protocol online combination of four saved variance experts.
No base-model refits, parameter search, or claim of a new aggregation family.
Full feedback uses s+h<=t; partial feedback only uses observed native returns.
"""
from pathlib import Path
import hashlib
import json
import math
import sys
import time

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize, check_grad
import corrected_risk_experiment as base

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'normalized_comparison_predictions_common.csv'
EXPERTS = ['EWMA97', 'HAR_return', 'QHAR_joint', 'QHAR_liquidity_plus_scale']
COMBINATIONS = ['Online_dynamic', 'PartialFeedback_dynamic', 'LeadBalanced_partial', 'Online_expanding', 'Equal_geometric',
                'Equal_arithmetic', 'Frozen2024']
HALF_LIFE = 22
WINDOW = 252
MIN_FEEDBACK = 44
RIDGE = .01
UNIFORM = np.full(4, .25)
DIAGNOSTICS = []
FLAGS = []


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def objective(w, log_forecasts, y, importance):
    eta = log_forecasts @ w
    ratio = y * np.exp(-eta)
    value = float(importance @ (eta + ratio) + RIDGE / 2 * np.sum((w - UNIFORM) ** 2))
    grad = log_forecasts.T @ (importance * (1 - ratio)) + RIDGE * (w - UNIFORM)
    return value, grad


def fit_weights(x, y, importance, context):
    assert len(y) >= MIN_FEEDBACK
    assert np.isfinite(x).all() and np.isfinite(y).all() and (y >= 0).all()
    importance = np.asarray(importance, float)
    importance = importance / importance.sum()
    opt = minimize(objective, UNIFORM.copy(), args=(x, y, importance), jac=True,
                   method='SLSQP', bounds=[(0., 1.)] * 4,
                   constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1.,
                                 'jac': lambda w: np.ones(4)}],
                   options={'maxiter': 1000, 'ftol': 1e-11, 'disp': False})
    w = opt.x
    feasible = bool(np.isfinite(w).all() and w.min() >= 0 and w.max() <= 1
                    and abs(w.sum() - 1) <= 1e-8)
    # First-order optimality gap on the simplex is zero at a constrained optimum.
    value, gradient = objective(w, x, y, importance)
    kkt_gap = float(w @ gradient - gradient.min())
    record = dict(context, n_feedback=len(y), success=bool(opt.success),
                  status=int(opt.status), message=str(opt.message), iterations=int(opt.nit),
                  objective=value, simplex_sum=float(w.sum()), min_weight=float(w.min()),
                  simplex_first_order_gap=kkt_gap, effective_feedback=float(1 / (importance @ importance)))
    DIAGNOSTICS.append(record)
    if not opt.success or not feasible or not np.isfinite(value) or kkt_gap > 1e-5:
        FLAGS.append(record)
        with (ROOT / 'online_optimization_flags.jsonl').open('a', encoding='utf-8') as out:
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
        raise RuntimeError(f'Online weight fit failed (no silent fallback): {record}')
    return w.copy(), record


def nw_t(d, lag):
    d = np.asarray(d, float)
    z = d - d.mean()
    n = len(d)
    var = z @ z / n
    for k in range(1, min(lag, n - 1) + 1):
        var += 2 * (1 - k / (lag + 1)) * (z[k:] @ z[:-k]) / n
    return float(d.mean() / np.sqrt(var / n)) if var > 0 else (0. if np.all(d == 0) else np.nan)


def solve_group(asset, h, frame, native):
    info_cols = ['origin_index', 'date', 'year', 'target', 'target_end', 'scored_common']
    base = frame[frame.model == EXPERTS[0]][info_cols].sort_values('origin_index').reset_index(drop=True)
    forecasts = frame.pivot(index='origin_index', columns='model', values='forecast').loc[base.origin_index, EXPERTS].to_numpy()
    assert np.isfinite(forecasts).all() and (forecasts > 0).all()
    assert base.scored_common.all()
    origins = base.origin_index.to_numpy(int)
    dates = pd.to_datetime(base.date).to_numpy()
    ends = pd.to_datetime(base.target_end).to_numpy()
    years = base.year.to_numpy(int)
    target = base.target.to_numpy(float)
    log_forecasts = np.log(forecasts)
    native_dates, native_r2 = native
    assert np.array_equal(native_dates[origins], dates)
    # Native full labels only verify the saved measurement and feed mature rows.
    # Their tiny CSV representation differences never alter evaluation targets.
    feedback_y = np.array([np.mean(native_r2[t+1:t+h+1]) for t in origins])
    np.testing.assert_allclose(feedback_y, target, rtol=1e-10, atol=1e-15)
    assert np.isfinite(target).all() and (target >= 0).all()
    assert (np.diff(origins) > 0).all()
    assert (ends > dates).all()
    frozen_w = None
    frozen_train = np.array([], dtype=int)
    frozen_fit_t = None
    rows, weights = [], []
    for i, t in enumerate(origins):
        # Labels are indexed by forecast origin. Y_i itself ends after t.
        mature = np.flatnonzero(origins + h <= t)
        assert np.all(mature < i)
        assert i not in mature
        assert np.all(origins[mature] + h <= t)
        assert np.all(ends[mature] <= dates[i])
        shared = np.flatnonzero(origins < t)[-WINDOW:]
        full_window = shared[origins[shared] + h <= t]
        solutions = {}
        for name, selected in [('Online_dynamic', full_window), ('Online_expanding', mature)]:
            age = t - origins[selected]
            importance = np.exp2(-age / HALF_LIFE) if name == 'Online_dynamic' else np.ones(len(selected))
            context = {'asset': asset, 'horizon': int(h), 'date': str(pd.Timestamp(dates[i]).date()),
                       'origin_index': int(t), 'model': name,
                       'train_origin_max': int(origins[selected[-1]]) if len(selected) else -1,
                       'train_label_end_max': int(origins[selected[-1]] + h) if len(selected) else -1}
            if len(selected) >= MIN_FEEDBACK:
                w, diag = fit_weights(log_forecasts[selected], feedback_y[selected], importance, context)
                mode = 'optimized_mature_feedback'
                effective = diag['effective_feedback']
            else:
                w, mode = UNIFORM.copy(), 'uniform_less_than_44'
                effective = float(importance.sum() ** 2 / (importance @ importance)) if len(selected) else 0.
            solutions[name] = (w, selected, mode, effective, t)
        # Partial QLIKE is exactly ordinary weighted QLIKE with response
        # knownsum/m and normalized weight decay*m/h; no future return is used.
        m = np.minimum(h, t - origins[shared])
        assert np.all(m >= 1) and np.all(origins[shared] + m <= t)
        knownsum = np.array([np.sum(native_r2[s+1:s+int(mm)+1])
                             for s, mm in zip(origins[shared], m)])
        partial_y = knownsum / m
        partial_importance = np.exp2(-(t-origins[shared]) / HALF_LIFE) * (m / h)
        partial_context = {'asset': asset, 'horizon': int(h), 'date': str(pd.Timestamp(dates[i]).date()),
                           'origin_index': int(t), 'model': 'PartialFeedback_dynamic',
                           'train_origin_max': int(origins[shared[-1]]) if len(shared) else -1,
                           'train_label_end_max': int(np.max(origins[shared]+m)) if len(shared) else -1,
                           'n_full_window': len(full_window)}
        if len(full_window) >= MIN_FEEDBACK:
            pw, diag = fit_weights(log_forecasts[shared], partial_y, partial_importance, partial_context)
            pmode, peffective = 'optimized_mature_feedback', diag['effective_feedback']
        else:
            pw, pmode = UNIFORM.copy(), 'uniform_less_than_44'
            peffective = float(partial_importance.sum()**2 / (partial_importance @ partial_importance)) if len(shared) else 0.
        solutions['PartialFeedback_dynamic'] = (pw, shared, pmode, peffective, t)
        if h == 1:
            assert np.array_equal(shared, full_window)
            assert np.array_equal(partial_y, feedback_y[full_window])
            assert np.array_equal(pw, solutions['Online_dynamic'][0])
            assert pmode == solutions['Online_dynamic'][2]
        # Each lead contributes exactly 1/h total mass among observed origins.
        if len(full_window) >= MIN_FEEDBACK:
            leads = np.arange(1, h+1)
            observed = leads[None, :] <= m[:, None]
            decay = np.exp2(-(t-origins[shared]) / HALF_LIFE)
            zlead = (decay[:, None] * observed).sum(axis=0)
            assert np.all(zlead > 0)
            mass = (decay[:, None] * observed) / zlead[None, :] / h
            np.testing.assert_allclose(mass.sum(axis=0), np.full(h, 1/h), rtol=1e-12, atol=1e-14)
            a = mass.sum(axis=1)
            np.testing.assert_allclose(a.sum(), 1., rtol=1e-12, atol=1e-14)
            known = np.zeros_like(mass)
            rr, cc = np.nonzero(observed)
            observed_indices = origins[shared][rr] + leads[cc]
            assert np.all(observed_indices <= t)
            known[rr, cc] = native_r2[observed_indices]
            q = (mass * known).sum(axis=1)
            balanced_y = q/a
            if np.all(m == h):
                np.testing.assert_allclose(a, decay/decay.sum(), rtol=1e-12, atol=1e-14)
                np.testing.assert_allclose(balanced_y, feedback_y[shared], rtol=1e-12, atol=1e-15)
                for test_w in [UNIFORM, solutions['Online_dynamic'][0]]:
                    bv, bg = objective(test_w, log_forecasts[shared], balanced_y, a/a.sum())
                    fv, fg = objective(test_w, log_forecasts[shared], feedback_y[shared], decay/decay.sum())
                    np.testing.assert_allclose(bv, fv, rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(bg, fg, rtol=1e-10, atol=1e-11)
            if h == 1:
                np.testing.assert_allclose(balanced_y, partial_y, rtol=1e-12, atol=1e-15)
                np.testing.assert_allclose(a, partial_importance/partial_importance.sum(), rtol=1e-12, atol=1e-14)
            balanced_context = dict(partial_context, model='LeadBalanced_partial')
            bw, diag = fit_weights(log_forecasts[shared], balanced_y, a, balanced_context)
            beffective = diag['effective_feedback']
            bmode = 'optimized_mature_feedback'
        else:
            bw, bmode, beffective = UNIFORM.copy(), 'uniform_less_than_44', 0.
        solutions['LeadBalanced_partial'] = (bw, shared, bmode, beffective, t)
        if h == 1:
            np.testing.assert_allclose(bw, pw, rtol=1e-9, atol=1e-9)
            assert bmode == pmode
        if years[i] >= 2025 and frozen_w is None:
            assert years[i] == 2025, 'Frozen2024 must be fit at first 2025 origin'
            frozen_train = mature[years[mature] == 2024]
            frozen_fit_t = int(t)
            context = {'asset': asset, 'horizon': int(h), 'date': str(pd.Timestamp(dates[i]).date()),
                       'origin_index': int(t), 'model': 'Frozen2024',
                       'train_origin_max': int(origins[frozen_train[-1]]) if len(frozen_train) else -1,
                       'train_label_end_max': int(origins[frozen_train[-1]] + h) if len(frozen_train) else -1}
            if len(frozen_train) >= MIN_FEEDBACK:
                frozen_w, _ = fit_weights(log_forecasts[frozen_train], feedback_y[frozen_train],
                                          np.ones(len(frozen_train)), context)
            else:
                frozen_w = UNIFORM.copy()
        solutions['Equal_geometric'] = (UNIFORM.copy(), np.array([], dtype=int), 'fixed_equal', 0., -1)
        solutions['Equal_arithmetic'] = (UNIFORM.copy(), np.array([], dtype=int), 'fixed_equal', 0., -1)
        if frozen_w is None:
            assert years[i] == 2024
            solutions['Frozen2024'] = (UNIFORM.copy(), np.array([], dtype=int), '2024_uniform_warmup', 0., -1)
        else:
            assert np.all(years[frozen_train] == 2024)
            assert np.all(origins[frozen_train] + h <= frozen_fit_t) and frozen_fit_t <= t
            solutions['Frozen2024'] = (frozen_w.copy(), frozen_train, 'frozen_at_first_2025_origin',
                                      float(len(frozen_train)), frozen_fit_t)
        for name in COMBINATIONS:
            w, selected, mode, effective, fit_t = solutions[name]
            assert w.min() >= 0 and abs(w.sum() - 1) < 1e-8
            known_counts = np.minimum(h, t-origins[selected]) if name in ['PartialFeedback_dynamic', 'LeadBalanced_partial'] else np.full(len(selected), h)
            assert np.all(origins[selected] + known_counts <= t)
            assert np.all(native_dates[origins[selected] + known_counts] <= dates[i])
            pred = float(forecasts[i] @ w) if name == 'Equal_arithmetic' else float(np.exp(log_forecasts[i] @ w))
            assert np.isfinite(pred) and pred > 0
            if h == 1 and name == 'PartialFeedback_dynamic':
                assert pred == float(np.exp(log_forecasts[i] @ solutions['Online_dynamic'][0]))
            if h == 1 and name == 'LeadBalanced_partial':
                np.testing.assert_allclose(pred, float(np.exp(log_forecasts[i] @ solutions['Online_dynamic'][0])), rtol=1e-10, atol=1e-15)
            common = {'asset': asset, 'horizon': int(h), 'date': str(pd.Timestamp(dates[i]).date()),
                      'origin_index': int(t), 'year': int(years[i]),
                      'stage': 'warmup_exploration' if years[i] == 2024 else 'seen_mechanism_diagnostic',
                      'target_end': str(pd.Timestamp(ends[i]).date()), 'scored_common': True, 'model': name}
            rows.append(dict(common, target=float(target[i]), forecast=pred,
                             qlike=float(np.log(pred) + target[i] / pred),
                             squared_error=float((target[i] - pred) ** 2)))
            weights.append(dict(common, **{f'w_{ex}': float(w[k]) for k, ex in enumerate(EXPERTS)},
                                mode=mode, n_mature_available=len(mature), n_feedback=len(selected),
                                effective_feedback=effective, weight_fit_origin_index=int(fit_t),
                                train_origin_max=int(origins[selected[-1]]) if len(selected) else -1,
                                train_label_end_max=int(np.max(origins[selected] + known_counts)) if len(selected) else -1,
                                shared_window_count=len(shared), full_feedback_in_shared_window=len(full_window),
                                total_observed_components=int(np.sum(known_counts)),
                                partial_origins=int(np.sum(known_counts < h)),
                                oldest_feedback_origin=int(origins[selected[0]]) if len(selected) else -1,
                                no_future_assertion=True))
    print(f'{asset} h={h}: {len(origins)} common origins, frozen2024 n={len(frozen_train)}', flush=True)
    return rows, weights


def summarize(all_predictions):
    summaries, comparisons = [], []
    benchmarks = ['Online_expanding', 'Equal_geometric', 'Equal_arithmetic', 'Frozen2024'] + EXPERTS
    for (asset, h), frame in all_predictions.groupby(['asset', 'horizon'], sort=False):
        for period in ['2024', '2025', '2026', '2025_2026']:
            g = frame[frame.year.isin([2025, 2026])] if period == '2025_2026' else frame[frame.year == int(period)]
            if g.empty:
                continue
            losses = g.pivot(index='origin_index', columns='model', values='qlike')
            errors = g.pivot(index='origin_index', columns='model', values='squared_error')
            assert losses.notna().all().all() and errors.notna().all().all()
            assert len(losses.columns) == len(EXPERTS) + len(COMBINATIONS)
            for name in losses.columns:
                summaries.append({'asset': asset, 'horizon': h, 'period': period, 'model': name,
                                  'n': len(losses), 'qlike': float(losses[name].mean()),
                                  'MSE': float(errors[name].mean()),
                                  'date_start': g.date.min(), 'date_end': g.date.max(),
                                  'stage': 'warmup_exploration' if period == '2024' else 'seen_mechanism_diagnostic'})
            for candidate, benchmark in [(candidate, benchmark) for candidate in ['Online_dynamic', 'PartialFeedback_dynamic', 'LeadBalanced_partial']
                                          for benchmark in benchmarks + ([] if candidate == 'Online_dynamic' else ['Online_dynamic']) + (['PartialFeedback_dynamic'] if candidate == 'LeadBalanced_partial' else [])]:
                d = losses[candidate] - losses[benchmark]
                stat = nw_t(d, max(5, h - 1))
                comparisons.append({'asset': asset, 'horizon': h, 'period': period,
                                    'model': candidate, 'benchmark': benchmark, 'n': len(d),
                                    'delta_QLIKE_negative_better': float(d.mean()),
                                    'NW_t_negative_better': stat,
                                    'NW_normal_p_two_sided': math.erfc(abs(stat) / math.sqrt(2)) if np.isfinite(stat) else np.nan,
                                    'NW_lag': max(5, h - 1),
                                    'MSE_ratio': float(errors[candidate].mean() / errors[benchmark].mean())})
    return pd.DataFrame(summaries), pd.DataFrame(comparisons)


def main():
    started = time.time()
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    (ROOT / 'online_optimization_flags.jsonl').write_text('', encoding='utf-8')
    source_hash = digest(SOURCE)
    source = pd.read_csv(SOURCE, float_precision='round_trip')
    assert source.scored_common.eq(True).all()
    keys = ['asset', 'horizon', 'origin_index']
    assert not source.duplicated(keys + ['model']).any()
    assert source.groupby(keys).model.nunique().eq(18).all()
    for col in ['date', 'target_end', 'year']:
        assert source.groupby(keys)[col].nunique().eq(1).all(), col
    target_spread = source.groupby(keys).target.agg(['max', 'min'])
    max_target_spread = float((target_spread['max'] - target_spread['min']).max())
    assert max_target_spread <= 1e-18, max_target_spread
    canonical = source[source.model == 'MA22'].set_index(keys).target
    source['target'] = canonical.reindex(pd.MultiIndex.from_frame(source[keys])).to_numpy()
    data = source[source.model.isin(EXPERTS)].copy()
    assert data.groupby(keys).model.nunique().eq(4).all()
    # Deterministic derivative check does not touch actual forecasting outcomes.
    rng = np.random.default_rng(20261022)
    xcheck = rng.normal(-8., .7, size=(60, 4))
    ycheck = np.exp(rng.normal(-8., .7, size=60))
    acheck = np.full(60, 1/60)
    gradient_error = float(check_grad(lambda w: objective(w, xcheck, ycheck, acheck)[0],
                                     lambda w: objective(w, xcheck, ycheck, acheck)[1], UNIFORM))
    assert gradient_error < 1e-5, gradient_error
    # Independent random all-mature windows: lead-balanced and full objectives
    # and analytic gradients coincide at arbitrary simplex weights.
    for hh in [1, 5, 22]:
        raw = np.exp(rng.normal(-8., .7, size=(60, hh)))
        decay_test = np.exp2(-np.arange(60) / HALF_LIFE)
        mass_test = np.broadcast_to((decay_test/decay_test.sum()/hh)[:, None], raw.shape)
        a_test = mass_test.sum(axis=1)
        y_test = (mass_test*raw).sum(axis=1)/a_test
        for _ in range(5):
            w_test = rng.dirichlet(np.ones(4))
            fv, fg = objective(w_test, xcheck, raw.mean(axis=1), decay_test/decay_test.sum())
            bv, bg = objective(w_test, xcheck, y_test, a_test/a_test.sum())
            np.testing.assert_allclose(fv, bv, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(fg, bg, rtol=1e-12, atol=1e-12)
    native = {}
    for asset in data.asset.unique():
        nd, _, nr2, _, _, _, _ = base.feature_data(asset)
        native[asset] = (nd.date.to_numpy(), nr2)
    rows, weights = [], []
    for (asset, h), group in data.groupby(['asset', 'horizon'], sort=False):
        rr, ww = solve_group(asset, h, group, native[asset])
        rows.extend(rr); weights.extend(ww)
    predictions = pd.DataFrame(rows)
    weight = pd.DataFrame(weights)
    assert predictions.groupby(keys).model.nunique().eq(len(COMBINATIONS)).all()
    assert len(predictions) == len(source) // 18 * len(COMBINATIONS)
    # Rescore every saved expert from its unchanged saved target and forecast.
    data['qlike'] = np.log(data.forecast) + data.target / data.forecast
    data['squared_error'] = (data.target - data.forecast) ** 2
    data['stage'] = np.where(data.year == 2024, 'warmup_exploration', 'seen_mechanism_diagnostic')
    all_predictions = pd.concat([predictions, data[predictions.columns]], ignore_index=True)
    assert all_predictions.groupby(keys).model.nunique().eq(len(EXPERTS) + len(COMBINATIONS)).all()
    assert not all_predictions.duplicated(keys + ['model']).any()
    assert (weight.train_label_end_max <= weight.origin_index).all()
    frozen = weight[(weight.model == 'Frozen2024') & (weight.year >= 2025)]
    for _, group in frozen.groupby(['asset', 'horizon']):
        assert group[[f'w_{x}' for x in EXPERTS]].nunique().eq(1).all()
        assert group.weight_fit_origin_index.nunique() == 1
    summary, pairwise = summarize(all_predictions)
    aggregate = pairwise.groupby(['period', 'horizon', 'model', 'benchmark'], as_index=False).agg(
        delta_QLIKE_equal_asset=('delta_QLIKE_negative_better', 'mean'),
        MSE_ratio_equal_asset=('MSE_ratio', 'mean'),
        assets=('asset', 'nunique'),
        assets_QLIKE_better=('delta_QLIKE_negative_better', lambda s: int((s < 0).sum())))
    weight_summary = weight.groupby(['asset', 'year', 'horizon', 'model'], as_index=False).agg(
        n=('date', 'size'), mean_feedback=('n_feedback', 'mean'),
        mean_effective_feedback=('effective_feedback', 'mean'),
        **{f'mean_w_{ex}': (f'w_{ex}', 'mean') for ex in EXPERTS})
    outputs = {'online_predictions.csv': predictions, 'online_comparison_predictions_common.csv': all_predictions,
               'online_weights.csv': weight, 'online_weight_summary.csv': weight_summary,
               'online_summary.csv': summary, 'online_pairwise.csv': pairwise,
               'online_pairwise_equal_asset.csv': aggregate,
               'online_fit_diagnostics.csv': pd.DataFrame(DIAGNOSTICS)}
    for filename, table in outputs.items():
        table.to_csv(ROOT / filename, index=False, float_format='%.17g', encoding='utf-8-sig')
    assert source_hash == digest(SOURCE), 'Input source changed during experiment'
    manifest = {'protocol': __doc__, 'input': str(SOURCE), 'input_sha256': source_hash,
                'script_sha256': digest(__file__), 'experts_fixed': EXPERTS, 'max_source_target_serialization_spread': max_target_spread,
                'models': COMBINATIONS, 'half_life_origin_trading_days': HALF_LIFE,
                'dynamic_max_shared_published_origins': WINDOW, 'minimum_feedback': MIN_FEEDBACK,
                'ridge': RIDGE, 'objective': 'weighted mean(log f + Y/f) + .01/2*||w-uniform||^2',
                'forecast': 'exp(sum w_k log saved_forecast_k), except fixed arithmetic mean',
                'feedback': 'full/partial share last252 scored_common published s<t origins; full uses s+h<=t, partial uses m=min(h,t-s) native daily squared returns; both start only with >=44 full labels in shared window',
                'partial_objective': 'sum decay*((m/h)*eta+knownsum/h*exp(-eta))/sum decay*(m/h) + .01/2||w-uniform||^2',
                'lead_balanced': 'For each lead j normalize decay over observed origins to mass1/h; use a_s=sum_j mass_sj and response sum_j mass_sj*r2_sj/a_s. Same shared window/startup/ridge; h1 independently optimizes then verifies weights rtol/atol1e-9 and forecasts rtol1e-10/atol1e-15 against full/partial',
                'native_return_sources': base.INPUTS, 'base_script_sha256': digest(base.__file__),
                'label_precision': 'Mature training labels recomputed from identical native r2, checked against saved targets to atol1e-15/rtol1e-10; evaluation targets unchanged',
                'Frozen2024': 'equal in 2024; at first 2025 common origin use all mature 2024 feedback, equal training weights; then freeze through 2026',
                'decision_periods': '2025 and partial 2026; 2024 warmup/exploration; all years already seen',
                'gradient_check_error': gradient_error,
                'checks': {'no_future_labels': True, 'historical_origin_excludes_current': True,
                           'target_end_date_mature': True, 'identical_common_samples': True,
                           'frozen_weights_constant_2025_2026': True, 'input_unchanged': True, 'h1_partial_full_exact_forecast_weights_startup': True, 'partial_observed_days_only': True, 'lead_balanced_mass_each_lead_one_over_h': True, 'h1_balanced_independent_optimization_tight_tolerance': True, 'all_components_mature_balanced_reduces_full_objective_gradient_random_and_actual': True},
                'counts': {'source_models': 18, 'common_origins': len(source)//18,
                           'new_predictions': len(predictions), 'optimizations': len(DIAGNOSTICS), 'failures': len(FLAGS)},
                'max_simplex_first_order_gap': float(max(x['simplex_first_order_gap'] for x in DIAGNOSTICS)),
                'outputs': {name: digest(ROOT/name) for name in list(outputs) + ['online_optimization_flags.jsonl']},
                'environment': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__},
                'runtime_seconds': time.time() - started,
                'limitations': ['Exploratory mechanism diagnostic; experts and years already seen',
                                'No hyperparameter changes or novelty claim', '2026 coverage partial and differs across assets',
                                'Newey-West lag max(5,h-1); approximate unadjusted tests, overlapping labels',
                                'Frozen2024 equal in 2024 is warmup only; its fitted 2024 training performance is never backfilled']}
    (ROOT / 'online_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    focus = aggregate[aggregate.period.isin(['2025', '2026'])]
    print(focus.to_string(index=False), flush=True)
    print('ONLINE COMPLETE', json.dumps(manifest['counts']), 'seconds', round(manifest['runtime_seconds'], 2), flush=True)


if __name__ == '__main__':
    main()
