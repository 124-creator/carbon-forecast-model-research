"""GRN strong baselines: reviewed-code handoff; NEVER fit by default.

--check-only: synthetic checks, no market files or optimizer/estimator fitting.
--review-validation: pure checks + frozen pre2026 NPZ/audit schema, no fitting.
--run-development: explicit later review/run action; train labels pre2025 only.
Raw zero targets and the exact 228 validation origins are retained throughout.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import os
for thread_key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[thread_key] = '1'
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import pickle
import re
import types
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / 'grn_range_development.npz'
AUDIT = HERE / 'grn_range_development_audit.json'
ALLOW_FROZEN_READ = False
IO_LOG = []


def io_guard(event, args):
    """No CSV/GRN2026 reads; only two approved market-data inputs exist."""
    if event != 'open' or not isinstance(args[0], (str, bytes, Path)):
        return
    raw = args[0].decode() if isinstance(args[0], bytes) else str(args[0])
    p = Path(raw).resolve()
    if p.suffix.lower() == '.csv':
        raise PermissionError('This baseline runner never opens any CSV')
    mode = args[1]
    reading = mode is None or ('r' in mode) or not any(c in mode for c in 'wax')
    if not reading:
        return
    if p.suffix.lower() not in ('.py', '.pyc', '.pyd', '.dll') and 'grn' in p.name.lower() and '2026' in p.name.lower():
        raise PermissionError('No GRN2026 data/results/checkpoints route')
    if p.is_relative_to(HERE) and p.suffix.lower() in ('.npz', '.json', '.pt', '.pkl'):
        if p in (DATA, AUDIT) and ALLOW_FROZEN_READ:
            IO_LOG.append(str(p))
        elif p.name.startswith('grn_strong_baselines'):
            pass  # This runner's own immutable validation/output artifacts.
        else:
            raise PermissionError('Unapproved project data read: ' + p.name)


sys.addaudithook(io_guard)
import scipy
from scipy.optimize import minimize
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
import grn_range_development_builder as G
import mem_baselines_round16 as MEM

B = G.B
ASSETS = ('GRN',)
HORIZONS = (1, 5, 22)
DATA_SHA = 'acd4a33c31a49c62563a7eae2dcaa236e27a1a0432c3dab42f015d8969d47a3f'
AUDIT_SHA = '4d12814c03a0202b63ba1045005806c3d231f952e2427a15acc5e831cf603e10'
G_SHA = '78e55425a0e6a9a04476dbb4dbaa2cb0a1f210e268c3d8a6cfb6ee18f085eb12'
B_SHA = '9ac2704711779ace771c64a6807a074c006850d294c96d2fdd48f3fbe95a273a'
MEM_SHA = '1fa3db4ef5ecf1106abab47db71c64b3152250a46d3726b62d6f8a873787324a'
HGB_HP = dict(loss='poisson', max_iter=150, max_leaf_nodes=7,
              learning_rate=.04, l2_regularization=1, min_samples_leaf=20,
              early_stopping=False, random_state=11)
LBFGS_HP = dict(maxiter=300, ftol=1e-12, gtol=1e-7, maxls=50)
GLM_HP = dict(ridge=.01, input_clip=5., beta_bounds=[-3., 3.],
              intercept_bounds=[-8., 8.], std_epsilon=1e-5,
              initialization='log(training mean y/scale), beta zero')
MODEL_NAMES = ('EWMA22', 'MEM_stationary', 'MEM_integrated',
               'HAR_geometry', 'HGB_geometry', 'HGB_Poisson', 'HAR_QLIKE')
GLOBALS_STAMP = G.module_globals_signature()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    with Path(path).open('x', encoding='utf-8') as f:
        f.write(text)


def array_record(x):
    x = np.asarray(x)
    return dict(shape=list(x.shape), dtype=str(x.dtype),
                sha256=hashlib.sha256(x.tobytes()).hexdigest())


def source_contract():
    for module, expected in ((G, G_SHA), (B, B_SHA), (MEM, MEM_SHA)):
        if sha(module.__file__) != expected:
            raise PermissionError('Reviewed original source changed: ' + module.__name__)
    assert G.module_globals_signature() == GLOBALS_STAMP
    tree = ast.parse(inspect.getsource(B.direct_baselines))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'HistGradientBoostingRegressor']
    assert len(calls) == 1
    assert {k.arg: ast.literal_eval(k.value) for k in calls[0].keywords} == HGB_HP
    assert MEM.B is B
    return dict(real_asset='GRN', original_ASSETS_SOURCES_unchanged=True,
                original_HGB_geometry_already_poisson=True,
                direct_baseline_function_sha256=hashlib.sha256(inspect.getsource(B.direct_baselines).encode()).hexdigest(),
                MEM_decode_forecasts_reused=True)


def source_manifest():
    source_contract()
    paths = dict(runner=Path(__file__), GRN_builder=Path(G.__file__),
                 original_MEM=Path(MEM.__file__), original_direct_baselines=Path(B.__file__),
                 HB=Path(G.HB.__file__), S17=Path(G.S17.__file__),
                 proxy=Path(G.S17.PROXY.__file__), R14=Path(B.R14.__file__),
                 sklearn_HGB=Path(inspect.getsourcefile(HistGradientBoostingRegressor)),
                 scipy_minimize=Path(inspect.getsourcefile(minimize)))
    assert all(p.suffix == '.py' for p in paths.values())
    return dict(files={k: dict(path=str(p.resolve()), sha256=sha(p)) for k, p in paths.items()},
                versions=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                              sklearn=sklearn.__version__, torch=G.torch.__version__))


def require_fit(allow_fit):
    if allow_fit is not True:
        raise PermissionError('Fitting requires the later explicit --run-development action')


def load_frozen_grn():
    """Never call G.build_pair/load_frame/source_paths or re-fit preprocessing."""
    global ALLOW_FROZEN_READ
    source_contract()
    ALLOW_FROZEN_READ = True
    try:
        if sha(DATA) != DATA_SHA or sha(AUDIT) != AUDIT_SHA:
            raise PermissionError('Frozen GRN NPZ/audit changed')
        audit = json.loads(AUDIT.read_text(encoding='utf-8'))
        with np.load(DATA, allow_pickle=False) as z:
            assert z['assets'].tolist() == ['GRN']
            np.testing.assert_array_equal(z['horizons'], HORIZONS)
            D = {k[4:]: (v.copy() if v.ndim else v.item())
                 for k in z.files if k.startswith('GRN_') for v in [z[k]]}
    finally:
        ALLOW_FROZEN_READ = False
    validate_schema(D, audit)
    return D, audit


def validate_schema(D, audit):
    assert D['instrument'] == audit['asset'] == 'GRN'
    assert (D['n'], D['cut'], D['test_cut']) == (1581, 1331, 1581)
    assert len(D['tr']) == 1158 and len(D['va']) == 228 and len(D['te']) == 0
    dates, origin, tr, va, y = (D[k] for k in ('date', 'origin', 'tr', 'va', 'y'))
    assert len(dates) == D['n'] and np.all(np.diff(dates) > np.timedelta64(0, 'D'))
    assert np.all(dates < np.datetime64('2026-01-01'))
    assert np.array_equal(np.sort(np.r_[tr, va]), np.arange(len(origin)))
    assert np.all(origin[tr] + 22 < D['cut'])
    assert np.all(dates[origin[va]] >= np.datetime64('2025-01-01'))
    assert np.all(origin + 22 < D['n'])
    expected_labels = np.column_stack([dates[origin + h] for h in HORIZONS])
    np.testing.assert_array_equal(D['label_dates'], expected_labels)
    assert np.all(D['label_dates'] < np.datetime64('2026-01-01'))
    p = D['proxy_raw']
    assert p.shape == (D['n'],) and np.isfinite(p).all() and (p >= 0).all()
    assert int((p == 0).sum()) == 342
    assert y.shape == (len(origin), 3) and np.isfinite(y).all() and (y >= 0).all()
    for j, h in enumerate(HORIZONS):
        expected = np.array([p[t + 1:t + h + 1].mean() for t in origin])
        np.testing.assert_array_equal(y[:, j], expected)
    assert ((y[tr] == 0).sum(0) == [167, 65, 17]).all()
    assert ((y[va] == 0).sum(0) == [33, 0, 0]).all()
    X = D['baseline_x_F16']
    assert X.shape == (len(origin), 15) and np.isfinite(X).all()
    np.testing.assert_array_equal(X, np.column_stack([D['context12'], D['F16'][origin]]))
    np.testing.assert_array_equal(D['ewma'], D['ewma22_all'][origin])
    assert np.isfinite(D['ewma']).all() and (D['ewma'] > 0).all()
    scale = float(p[:D['cut']].mean())
    fixed = audit['fixed_preprocessing']['GRN']
    assert fixed['scale_n'] == D['cut'] and fixed['scale_includes_zero']
    np.testing.assert_allclose(scale, fixed['scale'], rtol=1e-15, atol=0)
    assert G.module_globals_signature() == GLOBALS_STAMP
    return dict(asset='GRN', n=D['n'], pre2025_rows=D['cut'], train_rows=len(tr), validation_rows=len(va),
                train_label_last=str(dates[origin[tr[-1]] + 22].astype('datetime64[D]')),
                validation_first=str(dates[origin[va[0]]].astype('datetime64[D]')),
                validation_last=str(dates[origin[va[-1]]].astype('datetime64[D]')),
                validation_label_last=str(D['label_dates'][va[-1], -1].astype('datetime64[D]')),
                train_zero_targets=[int(x) for x in (y[tr] == 0).sum(0)],
                validation_zero_targets=[int(x) for x in (y[va] == 0).sum(0)],
                pre25_scale=scale, fixed_scale_exact=scale == fixed['scale'],
                F16_information='same15 original context12+F16; no geometry/volume substitutions',
                exact_arrays={k: array_record(D[k]) for k in
                              ('tr', 'va', 'te', 'origin', 'date', 'label_dates', 'y', 'ewma', 'proxy_raw', 'baseline_x_F16')},
                no_rows_or_zero_targets_dropped=True, target_reconstructed_exact_rawP=True,
                model_fit_performed=False, GRN2026_opened=False)


def normalizer(X, tr):
    X = np.asarray(X, float)
    if not np.isfinite(X).all():
        raise ValueError('Frozen baseline features must already be finite')
    means = X[tr].mean(0)
    stds = X[tr].std(0) + GLM_HP['std_epsilon']
    z = np.clip((X - means) / stds, -GLM_HP['input_clip'], GLM_HP['input_clip'])
    return np.column_stack([np.ones(len(X)), z]), dict(
        mean=means.tolist(), std=stds.tolist(), scope='mature pre2025 training X[tr] only',
        input_clip=GLM_HP['input_clip'], n_train=len(tr), n_features=X.shape[1])


def glm_loss_grad(theta, design, y_norm):
    """Convex log-link mean QLIKE + .01 sum(beta^2), no log/floor of y."""
    design, y_norm, theta = np.asarray(design, float), np.asarray(y_norm, float), np.asarray(theta, float)
    if y_norm.ndim != 1 or len(y_norm) != len(design) or not np.isfinite(y_norm).all() or (y_norm < 0).any():
        raise ValueError('All finite nonnegative targets required')
    f = design @ theta
    ratio = y_norm * np.exp(-f)
    loss = float(np.mean(f + ratio) + GLM_HP['ridge'] * np.dot(theta[1:], theta[1:]))
    grad = design.T @ (1 - ratio) / len(y_norm)
    grad[1:] += 2 * GLM_HP['ridge'] * theta[1:]
    if not np.isfinite(loss) or not np.isfinite(grad).all():
        raise FloatingPointError('Nonfinite bounded QLIKE GLM')
    return loss, grad


def fit_mem_grn(D, scale, integrated, *, allow_fit=False):
    require_fit(allow_fit)
    # Original decode/recursion, two starts, bounds, optimizer and training-only
    # selection. Sole data adaptation: use exact G rawP, including genuine zeros.
    proxy = np.asarray(D['proxy_raw'], float)
    tr, va, origin = D['tr'], D['va'], D['origin']
    np.testing.assert_allclose(proxy[:D['cut']].mean(), scale, rtol=1e-15, atol=0)
    target = D['y'][tr] / scale
    def objective(theta):
        pred = MEM.forecasts(proxy[:D['cut']] / scale, origin[tr], MEM.decode(theta, integrated))
        return float(np.mean(np.log(pred) + target / pred))
    starts = ([np.array([np.log(1e-5), np.log(a / (1 - a))]) for a in [.06, .2]] if integrated else
              [np.array([np.log(1-r), np.log((r/.999999)/(1-r/.999999)), np.log(.1/.9)]) for r in [.9, .98]])
    bounds = [(-16, 2), (-8, 16)] + ([] if integrated else [(-8, 8)])
    records = []
    for start in starts:
        out = minimize(objective, start, method='L-BFGS-B', bounds=bounds, options=LBFGS_HP)
        if not np.isfinite(out.fun) or not np.isfinite(out.x).all():
            raise FloatingPointError('Nonfinite MEM optimization')
        records.append(dict(initial_theta=start.tolist(), theta=out.x.tolist(), training_objective=float(out.fun),
                            converged=bool(out.success), status=int(out.status), message=str(out.message),
                            iterations=int(out.nit), function_calls=int(out.nfev)))
    selected = min(records, key=lambda x: x['training_objective'])
    parameters = MEM.decode(np.asarray(selected['theta']), integrated)
    values = MEM.forecasts(proxy / scale, origin[va], parameters) * scale
    return values, dict(asset='GRN', source='original MEM.decode/forecasts; private generic fit adapter',
                        exact_rawP_instead_of_log_proxy_roundtrip=True, starts=records, selected=selected,
                        selected_by='minimum training objective only; no validation selection',
                        bounds=bounds, optimizer=dict(method='L-BFGS-B', options=LBFGS_HP),
                        omega_actual=parameters[0]*scale, alpha=parameters[1], beta=parameters[2],
                        rho=parameters[1]+parameters[2], integrated=integrated,
                        scale_from_all_pre2025_rawP=scale)


def original_direct_function(captured):
    def constructor(*args, **kwargs):
        assert not args and kwargs == HGB_HP
        model = B.HistGradientBoostingRegressor(**kwargs)
        captured.append(model)
        return model
    fn = G.HB.clone_function(B.direct_baselines,
            {**vars(B), 'HistGradientBoostingRegressor': constructor})
    assert fn.__code__ is B.direct_baselines.__code__ and fn.__globals__ is not vars(B)
    return fn


def fit_original_direct(D, *, allow_fit=False):
    require_fit(allow_fit)
    local = dict(D)
    local['baseline_x'] = D['baseline_x_F16']
    captured = []
    predictions = original_direct_function(captured)(local)
    assert len(captured) == 3
    X, tr, va = local['baseline_x'], D['tr'], D['va']
    means, stds = X[tr].mean(0), X[tr].std(0) + 1e-5
    design = np.column_stack([np.ones(len(X)), (X - means) / stds])
    har, hgb = {}, {}
    for j, h in enumerate(HORIZONS):
        log_target = np.log(np.maximum(D['y'][tr, j], B.EPS))
        coef = np.linalg.lstsq(design[tr], log_target, rcond=None)[0]
        smear = float(np.exp(log_target - design[tr] @ coef).mean())
        np.testing.assert_allclose(np.exp(design[va] @ coef)*smear,
                                   predictions['HAR_geometry'][:, j], rtol=1e-12, atol=1e-15)
        har[str(h)] = dict(coef=coef.tolist(), smear=smear,
                          EPS_for_internal_log_regression_only=B.EPS,
                          training_zero_targets=int((D['y'][tr, j] == 0).sum()))
        hgb[str(h)] = dict(parameters=captured[j].get_params(), iterations=int(captured[j].n_iter_),
                          normalization=float(D['y'][tr, j].mean()),
                          normalization_scope='mature pre2025 target mean of this horizon',
                          zero_targets_retained=True)
    return predictions, dict(HAR_geometry=dict(mean=means.tolist(), std=stds.tolist(), per_h=har),
                             HGB_geometry=dict(per_h=hgb), original_direct_bytecode_unchanged=True), captured


def fit_hgb_poisson(D, scale, *, allow_fit=False):
    require_fit(allow_fit)
    X, tr, va = D['baseline_x_F16'], D['tr'], D['va']
    values = np.empty((len(va), 3)); models, metadata = [], {}
    for j, h in enumerate(HORIZONS):
        model = HistGradientBoostingRegressor(**HGB_HP)
        model.fit(X[tr], D['y'][tr, j] / scale)
        values[:, j] = model.predict(X[va]) * scale
        models.append(model)
        metadata[str(h)] = dict(parameters=model.get_params(), iterations=int(model.n_iter_),
                               training_zeros=int((D['y'][tr, j] == 0).sum()))
    return values, dict(per_h=metadata, scale=scale,
                        difference_from_original='one shared G fixed pre25 rawP scale, instead of horizon-specific training target means',
                        meaning='positive-mean Poisson quasi loss; prices/ranges are not counts'), models


def fit_har_qlike(D, scale, *, allow_fit=False):
    require_fit(allow_fit)
    design, norm = normalizer(D['baseline_x_F16'], D['tr'])
    tr, va = D['tr'], D['va']; p = design.shape[1]-1
    bounds = [tuple(GLM_HP['intercept_bounds'])] + [tuple(GLM_HP['beta_bounds'])]*p
    assert 8 + p*5*3 < 700  # Given 15 features, |f| <= 233, no exp clipping.
    predictions, metadata = np.empty((len(va), 3)), {}
    for j, h in enumerate(HORIZONS):
        y_norm = D['y'][tr, j] / scale
        initial = np.zeros(p+1)
        initial[0] = np.clip(np.log(y_norm.mean()), -8, 8) if y_norm.mean() > 0 else -8
        out = minimize(glm_loss_grad, initial, args=(design[tr], y_norm), jac=True,
                       method='L-BFGS-B', bounds=bounds, options=LBFGS_HP)
        if not np.isfinite(out.fun) or not np.isfinite(out.x).all():
            raise FloatingPointError('Nonfinite HAR_QLIKE optimization')
        predictions[:, j] = scale * np.exp(design[va] @ out.x)
        metadata[str(h)] = dict(theta=out.x.tolist(), initial=initial.tolist(), training_objective=float(out.fun),
                                converged=bool(out.success), status=int(out.status), message=str(out.message),
                                iterations=int(out.nit), function_calls=int(out.nfev),
                                training_zero_targets=int((y_norm == 0).sum()),
                                active_bounds=[int(i) for i, (v, (lo, hi)) in enumerate(zip(out.x, bounds))
                                               if min(abs(v-lo), abs(v-hi)) < 1e-7])
    return predictions, dict(normalizer=norm, scale=scale, bounds=bounds, hyperparameters=GLM_HP,
                             optimizer=dict(method='L-BFGS-B', options=LBFGS_HP), per_h=metadata,
                             selected_by='training objective only; no validation hyperparameter choice')


def normalized(values, target):
    zero = int((target == 0).sum())
    return dict(value=None if zero else float(np.mean(np.log(values)-np.log(target)+target/values-1)),
                status='undefined_contains_zero_targets' if zero else 'defined_all_targets_positive',
                n=int(target.size), zero_targets=zero, no_positive_subset_rescue=True)


def safe_metrics(predictions, D):
    y, ewma = D['y'][D['va']], D['ewma'][D['va']]
    if not np.isfinite(y).all() or (y < 0).any() or not (ewma > 0).all():
        raise ValueError('Invalid immutable raw target/reference')
    output = {}
    for name, v in predictions.items():
        if v.shape != y.shape or not np.isfinite(v).all() or not (v > 0).all():
            raise FloatingPointError('Invalid positive point forecast: ' + name)
        raw = np.log(v) + y/v; known = raw - np.log(ewma)[:, None]; squared = (v-y)**2
        if not np.isfinite(raw).all() or not np.isfinite(squared).all():
            raise FloatingPointError('Nonfinite forecast scores')
        output[name] = dict(primary_all_targets=dict(n_rows=len(y), n_cells=y.size,
            raw_QLIKE=float(raw.mean()), knownscale_QLIKE=float(known.mean()), MSE=float(squared.mean()),
            zero_targets=int((y == 0).sum())), normalized_QLIKE_aggregate=normalized(v, y),
            per_h={str(h): dict(n=len(y), zero_targets=int((y[:, j] == 0).sum()),
                raw_QLIKE=float(raw[:, j].mean()), knownscale_QLIKE=float(known[:, j].mean()),
                MSE=float(squared[:, j].mean()), normalized_QLIKE=normalized(v[:, j], y[:, j]))
                for j, h in enumerate(HORIZONS)})
    return output


def pure_checks():
    contract = source_contract(); start_IO = len(IO_LOG)
    rng = np.random.default_rng(921)
    X = rng.normal(size=(20, 4)); tr = np.arange(12); va = np.arange(12, 20)
    design, norm = normalizer(X, tr)
    mutated = X.copy(); mutated[va] = 1e12
    changed, second = normalizer(mutated, tr)
    assert norm == second
    np.testing.assert_array_equal(design[tr], changed[tr])
    assert design.shape == (20, 5) and (np.abs(design[:, 1:]) <= 5).all()
    y = np.array([0., .2, 0., 1., 2., 0., .5, 0., 3., .1, .3, 0.])
    theta = rng.normal(0, .15, 5)
    value, grad = glm_loss_grad(theta, design[tr], y)
    numeric = np.empty_like(grad); step = 1e-6
    for k in range(len(theta)):
        delta = np.zeros_like(theta); delta[k] = step
        numeric[k] = (glm_loss_grad(theta+delta, design[tr], y)[0]
                      - glm_loss_grad(theta-delta, design[tr], y)[0])/(2*step)
    error = float(np.max(np.abs(grad-numeric)))
    assert error < 1e-7
    zero_loss, zero_grad = glm_loss_grad(np.zeros(5), design[tr], np.zeros(12))
    assert zero_loss == 0 and zero_grad[0] == 1 and np.isfinite(zero_grad).all()
    for direction in (-1, 1):
        edge = np.r_[direction*8., np.full(15, direction*3.)]
        glm_loss_grad(edge, np.column_stack([np.ones(3), np.full((3, 15), 5.)]), np.array([0., 1., 2.]))
    proxy = rng.uniform(0, 3, 120); proxy[::4] = 0
    origin = np.array([20, 49]); altered = proxy.copy(); altered[50:] = 1e5
    for integrated, pars in ((False, (.04, .15, .8)), (True, (1e-5, .1, .9))):
        a = MEM.forecasts(proxy, origin, pars)
        np.testing.assert_array_equal(a, MEM.forecasts(altered, origin, pars))
        assert (a > 0).all()
    target = np.array([[0., 1., 2.], [3., 0., 1.]])
    D = dict(y=target, va=np.arange(2), ewma=np.ones(2))
    scored = safe_metrics({'probe': np.ones_like(target)}, D)['probe']
    assert scored['primary_all_targets']['n_cells'] == 6 and scored['primary_all_targets']['zero_targets'] == 2
    assert scored['primary_all_targets']['raw_QLIKE'] == target.mean()
    assert scored['normalized_QLIKE_aggregate']['value'] is None
    assert scored['per_h']['1']['normalized_QLIKE']['value'] is None
    assert scored['per_h']['22']['normalized_QLIKE']['value'] is not None
    for fn, args in ((fit_mem_grn, (None, None, False)), (fit_original_direct, (None,)),
                     (fit_hgb_poisson, (None, None)), (fit_har_qlike, (None, None))):
        try:
            fn(*args)
        except PermissionError:
            pass
        else:
            raise AssertionError('Implicit fit admitted')
    clone = original_direct_function([])
    assert clone.__code__ is B.direct_baselines.__code__ and clone.__globals__ is not vars(B)
    assert G.module_globals_signature() == GLOBALS_STAMP and len(IO_LOG) == start_IO
    return dict(status='PURE CHECKS PASSED; no optimizer/estimator fitting', contract=contract,
                analytic_GLM_gradient_max_error=error, zero_target_no_log_floor_drop=True,
                zero_target_at_f0_loss0_intercept_gradient1=True, bounded_exp_finite_at_f233=True,
                validation_mutation_leaves_training_normalizer_exact=True,
                all_raw_Qlike_rows_preserved=True, normalized_zero_group_None=True,
                MEM_recursion_future_mutation_invariant=True, private_direct_function_bytecode_exact=True,
                implicit_fit_rejected=True, market_files_opened=False, formal_training_performed=False)


def check_tag(tag):
    if not re.fullmatch(r'grn_strong_baselines(?:_[A-Za-z0-9-]+)*', tag) or '2026' in tag:
        raise ValueError('Local GRN strong-baseline development basename only')
    return tag


def review_paths(tag):
    return {name: HERE/(tag+suffix) for name, suffix in
            [('purechecks', '_purechecks.json'), ('schema', '_schema_validation.json'),
             ('manifest', '_delivery_manifest.json')]}


def reject_existing(paths):
    for p in paths:
        if p.exists():
            raise FileExistsError('Refuse overwrite: ' + p.name)


def review_validation(tag):
    paths = review_paths(tag); reject_existing(paths.values())
    sources = source_manifest(); checks = pure_checks()
    D, audit = load_frozen_grn(); schema = validate_schema(D, audit)
    design, norm = normalizer(D['baseline_x_F16'], D['tr'])
    schema.update(GLM_training_normalizer=norm, GLM_design_shape=list(design.shape),
                  GLM_input_clip_count=int((np.abs((D['baseline_x_F16']-np.array(norm['mean']))/np.array(norm['std'])) > 5).sum()),
                  source_market_reads=sorted(set(IO_LOG)), formal_training_performed=False)
    assert set(IO_LOG) == {str(DATA), str(AUDIT)}
    assert sources == source_manifest()
    save_json(paths['purechecks'], checks); save_json(paths['schema'], schema)
    manifest = dict(status='CODE/PURE/REAL-SCHEMA VALIDATED; FORMAL FIT NOT EXECUTED', created_at_utc=utc(),
        asset='GRN', script_sha256=sha(__file__), sources=sources,
        frozen_inputs=dict(NPZ=dict(path=str(DATA), sha256=DATA_SHA), audit=dict(path=str(AUDIT), sha256=AUDIT_SHA)),
        evidence={name: dict(path=str(p), sha256=sha(p)) for name, p in paths.items() if name != 'manifest'},
        hyperparameters=dict(HGB=HGB_HP, MEM_LBFGS=LBFGS_HP, HAR_QLIKE=GLM_HP),
        no_formal_fit=True, no_GRN2026_IO=True, zero_targets_retained=True,
        later_explicit_command=f'python grn_strong_baselines.py --run-development --tag {tag}',
        later_run_requires_same_code_and_validation_manifest=True,
        research_status='Already-used development; not a fresh independent confirmation')
    save_json(paths['manifest'], manifest)
    return manifest


def run_development(tag):
    # This branch is never called by check/review modes. An explicit later review
    # and --run-development operator action is required.
    review = review_paths(tag)
    prior = json.loads(review['manifest'].read_text(encoding='utf-8'))
    if prior['script_sha256'] != sha(__file__) or not prior['no_formal_fit']:
        raise PermissionError('Current source does not match the validated delivery')
    for info in prior['evidence'].values():
        if sha(info['path']) != info['sha256']:
            raise PermissionError('Validation evidence changed')
    sources = source_manifest()
    if sources != prior['sources']:
        raise PermissionError('Validated dependency source or library version changed')
    paths = {key: HERE/(tag+suffix) for key, suffix in
             [('reservation', '_development_reservation.json'), ('results', '_development_results.json'),
              ('predictions', '_development_validation_predictions.npz'),
              ('models', '_development_fitted_models.pkl'), ('manifest', '_development_manifest.json')]}
    reject_existing(paths.values())
    D, audit = load_frozen_grn(); schema = validate_schema(D, audit)
    input_fingerprint = {k: array_record(D[k]) for k in
                         ('tr', 'va', 'origin', 'date', 'label_dates', 'y', 'ewma', 'proxy_raw', 'baseline_x_F16')}
    save_json(paths['reservation'], dict(started_at_utc=utc(), asset='GRN', source_sha256=sha(__file__),
        validation_manifest_sha256=sha(review['manifest']), immutable_one_run_reservation=True,
        completed_manifest_required=True))
    scale = audit['fixed_preprocessing']['GRN']['scale']
    pred, params, trees = fit_original_direct(D, allow_fit=True)
    params['source_original_direct'] = params.pop('original_direct_bytecode_unchanged')
    params['EWMA22'] = dict(source='exact frozen G ewma[va]', span=22,
                           fitting_performed=False, span_selected_using_validation=False)
    fitted = {'HGB_geometry': dict(zip(HORIZONS, trees))}
    for name, integrated in [('MEM_stationary', False), ('MEM_integrated', True)]:
        pred[name], params[name] = fit_mem_grn(D, scale, integrated, allow_fit=True)
    pred['HGB_Poisson'], params['HGB_Poisson'], trees = fit_hgb_poisson(D, scale, allow_fit=True)
    fitted['HGB_Poisson'] = dict(zip(HORIZONS, trees))
    pred['HAR_QLIKE'], params['HAR_QLIKE'] = fit_har_qlike(D, scale, allow_fit=True)
    assert set(pred) == set(MODEL_NAMES)
    assert input_fingerprint == {k: array_record(D[k]) for k in input_fingerprint}
    assert G.module_globals_signature() == GLOBALS_STAMP and sources == source_manifest()
    scores = safe_metrics(pred, D)
    for info in scores.values():
        assert info['primary_all_targets']['n_rows'] == 228
        assert info['per_h']['1']['zero_targets'] == 33
        assert info['normalized_QLIKE_aggregate']['value'] is None
    va = D['va']; origin = D['origin'][va]
    arrays = dict(assets=np.array(ASSETS), horizons=np.array(HORIZONS), GRN_date=D['date'][origin],
                  GRN_origin=origin, GRN_label_end=D['date'][origin + 22], GRN_target=D['y'][va],
                  GRN_ewma=D['ewma'][va], GRN_tr=D['tr'], GRN_va=va,
                  GRN_all_origin=D['origin'], GRN_all_date=D['date'], GRN_all_y=D['y'],
                  GRN_all_label_dates=D['label_dates'], GRN_all_ewma=D['ewma'],
                  GRN_baseline_x_F16=D['baseline_x_F16'])
    arrays.update({'GRN_'+k: v for k, v in pred.items()})
    with paths['predictions'].open('xb') as f:
        np.savez_compressed(f, **arrays)
    with paths['models'].open('xb') as f:
        pickle.dump(dict(asset='GRN', fitted_trees=fitted, versions=sources['versions'],
                         feature_contract='exact frozen baseline_x_F16; no asset aliases'), f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    results = dict(status='REUSED2025 GRN DEVELOPMENT; NOT FRESH CONFIRMATION', asset='GRN',
        schema=schema, source_manifest=sources, fitted_parameters=params, scores=scores,
        hyperparameters=dict(HGB=HGB_HP, LBFGS=LBFGS_HP, HAR_QLIKE=GLM_HP),
        training_scope='pre2025 mature labels only; no validation hyperparameter/seed/start selection',
        tree_models_saved=str(paths['models']), all_zero_targets_retained=True,
        HGB_Poisson_note='original HGB already poisson; additional variant fixes one pre25 rawP scale',
        poisson_quasi_mean_not_count_claim=True, GRN2026_opened=False,
        normalised_any_zero_is_None=True, module_globals_unchanged=True)
    save_json(paths['results'], results)
    assert sources == source_manifest()
    manifest = dict(status='COMPLETED GRN STRONG BASELINE DEVELOPMENT', completed_at_utc=utc(),
        source_sha256=sha(__file__), sources=sources, data_sha256=DATA_SHA, audit_sha256=AUDIT_SHA,
        artifact={k: dict(path=str(p), sha256=sha(p)) for k, p in paths.items() if k != 'manifest'},
        validation_manifest_sha256=sha(review['manifest']), no_GRN2026_IO=True,
        validation_rows=228, zero_targets_by_h=[33, 0, 0], primary_uses_all_targets=True)
    save_json(paths['manifest'], manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check-only', action='store_true')
    modes.add_argument('--review-validation', action='store_true')
    modes.add_argument('--run-development', action='store_true')
    parser.add_argument('--tag', default='grn_strong_baselines')
    args = parser.parse_args(); tag = check_tag(args.tag)
    if args.run_development:
        output = run_development(tag)
    elif args.review_validation:
        output = review_validation(tag)
    else:
        output = pure_checks()
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
