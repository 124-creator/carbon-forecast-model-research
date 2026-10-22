"""Frozen R20 holdout runner. DEFAULT: development-only, never authorizes test.

No selection file is created. No neural/MEM/KF fitting or epoch selection occurs.
HAR/HGB alone use the explicitly prescribed original B.direct_baselines with
pre2025 tr and F16 inputs; merged validation/test is predicted once, then its
2025 prefix MUST match its old R16 archive before any test tail is retained.
Test requires a separately frozen selection binding this runner and all inputs.
Predictions are written before scoring. Test primary never changes after opening.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
import torch
import holdout_2026_range_builder as HB
import round20_observation_operator_gru as R20
import mem_baselines_round16 as MEM
import round18_classical_measurements as CLASSICAL

C18, S18, S17, B = R20.C18, R20.S18, R20.S17, R20.B
ASSETS, DTYPE, SEEDS, HORIZONS = R20.ASSETS, R20.DTYPE, R20.SEEDS, B.HORIZONS
HERE = Path(__file__).resolve().parent
PREPROCESSING_SHA256 = '85bd63bb036677e8cd6d2c2941bd1ddbc34d86df15f9a8441f5b09fabd15a513'
RTOL, ATOL = 1e-11, 1e-14
RESULTS = HERE / 'round20_observation_operator_gru_development_results.json'
PREDICTIONS = HERE / 'round20_observation_operator_gru_development_validation_predictions.npz'
DEV_REPORT = HERE / 'round20_holdout_runner_development_replay.json'
DEV_PREDICTIONS = HERE / 'round20_holdout_runner_development_predictions.npz'
TEST_REPORT = HERE / 'round20_holdout_runner_test_report.json'
TEST_PREDICTIONS = HERE / 'round20_holdout_runner_test_predictions.npz'
BASE_NAMES = ('C18_GRU_ensemble', 'R16_frozen', 'R16_raw', 'MEM_stationary', 'MEM_integrated',
              'HAR_geometry', 'HGB_geometry', 'GK_global_KF', 'RS_global_KF', 'EWMA22')
ENSEMBLE_NAMES = tuple(f'R20_{v}_ensemble' for v in R20.VARIANTS) + BASE_NAMES


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def required_source_paths(*, include_development_report=False):
    """Static paths: can validate selection before opening any model/data input."""
    paths = dict(HB.source_paths())
    paths.update(runner=Path(__file__).resolve(), R20=Path(R20.__file__), C18=Path(C18.__file__),
        warmstart_exporter=Path(R20.REPLAY.__file__), probability_adapter=Path(R20.SDP.__file__),
        density_moments=Path(R20.SDP.FROZEN.__file__), density_JSON=R20.SDP.CALIBRATION,
        density_NPZ=R20.SDP.CALIBRATION.with_suffix('.npz'), R20_results=RESULTS, R20_predictions=PREDICTIONS,
        R20_zero_gate=R20.ZERO_GATE_PATH, builder_development_report=HERE/'holdout_2026_range_builder_development_validation.json',
        C18_results=R20.REPLAY.ARCHIVE_RESULTS, C18_predictions=R20.REPLAY.ARCHIVE_PREDICTIONS,
        MEM_source=Path(MEM.__file__), MEM_results=HERE/'mem_baselines_round16_results.json',
        MEM_predictions=HERE/'mem_baselines_round16_predictions.npz',
        classical_source=Path(CLASSICAL.__file__), classical_results=HERE/'round18_classical_measurements_results.json',
        classical_predictions=HERE/'round18_classical_measurements_validation_predictions.npz')
    warm_paths, manifest = R20.REPLAY.output_paths(); paths['warmstart_manifest'] = manifest
    for seed, path in warm_paths.items():
        paths[f'C18_checkpoint_seed{seed}'] = path
    for variant in R20.VARIANTS:
        for seed in SEEDS:
            paths[f'R20_checkpoint_{variant}_seed{seed}'] = HERE/f'round20_observation_operator_gru_development_{variant}_seed{seed}.pt'
    for asset in ASSETS:
        paths[f'R16_results_{asset}'] = S18.STAGE16[asset]['results']
        paths[f'R16_predictions_{asset}'] = S18.STAGE16[asset]['predictions']
    if include_development_report:
        paths['runner_development_report'] = DEV_REPORT
        paths['runner_development_predictions'] = DEV_PREDICTIONS
    return {k: Path(v).resolve() for k, v in paths.items()}


def source_hashes():
    return {k: dict(path=str(v), sha256=digest(v)) for k, v in required_source_paths().items()}


def validate_selection_contract(payload):
    """Structural checks have no input I/O and never grant missing permission."""
    if payload.get('allow_test') is not True:
        raise PermissionError('Explicit allow_test=true required; this runner never creates it')
    if payload.get('preprocessing_sha256', '').lower() != PREPROCESSING_SHA256:
        raise PermissionError('Frozen preprocessing digest must match the approved development digest')
    primary = payload.get('selected_candidate')
    if primary not in tuple(f'R20_{v}_ensemble' for v in R20.VARIANTS):
        raise PermissionError('selected_candidate must name ONE frozen R20_<variant>_ensemble')
    comparisons = payload.get('comparisons')
    if not isinstance(comparisons, list) or not comparisons or len(set(comparisons)) != len(comparisons):
        raise PermissionError('Freeze a nonempty unique comparisons list')
    if any(c not in ENSEMBLE_NAMES or c == primary for c in comparisons):
        raise PermissionError('Unknown comparison or primary compared against itself')
    if primary != 'R20_continued_ensemble' and 'R20_continued_ensemble' not in comparisons:
        raise PermissionError('The strong matched continued comparator must be frozen')
    if not isinstance(payload.get('criteria'), dict) or not payload['criteria']:
        raise PermissionError('Freeze the confirmatory criteria before test')
    epochs = payload.get('checkpoint_selected_epochs')
    keys = {f'R20_{v}_seed{s}' for v in R20.VARIANTS for s in SEEDS} | {f'C18_GRU_seed{s}' for s in SEEDS}
    if not isinstance(epochs, dict) or set(epochs) != keys:
        raise PermissionError('Bind every R20/C18 checkpoint selected epoch')
    if any(type(v) is not int or v < 0 for v in epochs.values()):
        raise PermissionError('Checkpoint epochs must be nonnegative integers')
    return payload


def authorize_test(selection_path):
    # Original builder verifies its full source protocol first, before test I/O.
    original = HB.authorize_test(selection_path)
    validate_selection_contract(original)
    return HB.validate_selection_payload(original, required_source_paths(include_development_report=True))


def reject_existing(paths):
    for path in paths:
        if Path(path).exists():
            raise FileExistsError(f'Protected existing output: {path}')


def assert_matching(actual, expected):
    assert actual.shape == expected.shape and np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)
    difference = np.abs(actual - expected)
    return dict(n=len(actual), max_absolute_error=float(difference.max()),
        max_relative_error=float((difference / np.maximum(np.abs(expected), np.finfo(float).tiny)).max()),
        bitwise_equal=bool(np.array_equal(actual, expected)), rtol=RTOL, atol=ATOL)


def archive_dates(D, saved, prefix=''):
    ids = D['va']; q = D['origin'][ids]
    assert (saved[prefix+'date'] < HB.TEST_START).all()
    np.testing.assert_array_equal(saved[prefix+'date'], D['date'][q])
    np.testing.assert_array_equal(saved[prefix+'target'], D['y'][ids])
    if prefix+'origin' in saved:
        np.testing.assert_array_equal(saved[prefix+'origin'], q)
    if prefix+'label_end' in saved:
        assert (saved[prefix+'label_end'] < HB.TEST_START).all()
        np.testing.assert_array_equal(saved[prefix+'label_end'], D['date'][q+22])
    if 'horizons' in saved:
        np.testing.assert_array_equal(saved['horizons'], np.array(HORIZONS))


def development_view(datasets):
    """Recreate checkpoint-size buffers before loading; never refit anything."""
    out = {}
    for a in ASSETS:
        D = datasets[a]; n = int(D.get('test_cut', D['n']))
        m = len(D['tr']) + len(D['va']); E = {}
        for key, value in D.items():
            if isinstance(value, np.ndarray) and value.ndim:
                E[key] = value[:n].copy() if len(value) == D['n'] else value[:m].copy() if len(value) == len(D['origin']) else value.copy()
            else:
                E[key] = copy.deepcopy(value)
        E.update(n=n, te=np.array([], dtype=int))
        np.testing.assert_array_equal(E['origin'], D['origin'][:m])
        assert (E['date'] < HB.TEST_START).all()
        out[a] = E
    return out


def extend_data_buffers(model, datasets, full_baseline, *, r20):
    """Only listed data buffers may grow. All learned/frozen state stays exact."""
    allowed = {f'{k}_{a}' for a in ASSETS for k in ('m0','P0','K0')}
    if r20:
        allowed |= {f'{k}_{a}' for a in ASSETS for k in ('observed','cells')}
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for a in ASSETS:
        oldn = len(getattr(model, f'm0_{a}'))
        for key in ('m0', 'P0', 'K0'):
            value = full_baseline[a][key].detach().clone()
            assert torch.equal(value[:oldn], getattr(model, f'{key}_{a}')), f'Frozen cache prefix changed: {a}/{key}'
            setattr(model, f'{key}_{a}', value)
        if r20:
            observed = torch.tensor(datasets[a]['observed'], dtype=torch.bool)
            cells = R20.geometry31(torch.tensor(datasets[a]['F17'], dtype=DTYPE), observed)
            for key, value in (('observed', observed), ('cells', cells)):
                assert torch.equal(value[:oldn], getattr(model, f'{key}_{a}'))
                setattr(model, f'{key}_{a}', value)
    for key, value in model.state_dict().items():
        if key not in allowed:
            assert torch.equal(value, before[key]), f'Unauthorized state change: {key}'
    return dict(expanded_buffers=sorted(allowed), all_old_prefixes_exact=True,
                parameters_theta_bank_decoder_AR_unchanged=True)


@torch.no_grad()
def predict_r20(model, datasets, T, ids):
    out = {}
    for a in ASSETS:
        D = datasets[a]; fields = model.filtered_fields(a, D, T[a], D['n'], 0)
        out[a] = model.forecast(a, T[a], D['origin'][ids[a]], fields)[0].numpy()
    return out


@torch.no_grad()
def predict_old_gru(model, datasets, T, ids):
    out = {}
    for a in ASSETS:
        D = datasets[a]; delta, _, _ = model.fields(a, D, T[a], D['n'], 0)
        m, P = model.corrected_state(a, delta); q = D['origin'][ids[a]]
        out[a] = model.processes[a].forecast(m[q], P[q]).numpy()
    return out


def fixed_kf(D, theta, ids, *, raw=False, log_proxy=None, observed=None, bias=None, variance=None):
    process = S18.FrozenProcess(theta); mu, phi, Q, sR = process.parameters_bounded()
    x = torch.tensor(D['log_proxy'] if log_proxy is None else log_proxy, dtype=DTYPE)
    b = (D['global_b'] if raw else D['cell_b']) if bias is None else bias
    R = (D['global_R'] if raw else D['cell_R']) if variance is None else variance
    R = torch.as_tensor(R, dtype=DTYPE) * torch.exp(sR)
    R = torch.broadcast_to(R, x.shape)
    flag = D['observed'] if observed is None else observed
    with torch.no_grad():
        m, P, _ = R20.KP.kalman_prefix(x - torch.as_tensor(b, dtype=DTYPE), R, flag, phi, Q, mu,
                                     *process.initial_state(), validate=False)
        q = D['origin'][ids]
        return process.forecast(m[q], P[q]).numpy()


def classical_observations(D):
    # Exact OHLC geometry identities; no extra CSV/OHLC reader or re-calibration.
    P = D['proxy_raw']; a = (D['F17'][:,0]+1)/2; z = (D['F17'][:,1]+1)/2
    xi_squared = (z * (1-a))**2
    return dict(GK=P*(4*CLASSICAL.LOG2)*(.5-CLASSICAL.CLOSE_COEF*a*a),
                RS=P*(2*CLASSICAL.LOG2)*(1+xi_squared-a*a))


def selected_epochs(archive):
    out = {f'R20_{v}_seed{f["seed"]}': f['best_joint_epoch'] for v in R20.VARIANTS for f in archive['fits'][v]}
    out.update({f'C18_GRU_seed{s}': e for s, e in R20.REPLAY.SELECTED.items()})
    return out


def collect_predictions(datasets, T, archive, authorization=None):
    """No optimizer. Fail on any development mismatch, retain no weaker substitute."""
    dev = development_view(datasets); devT = S17.make_tensors(dev)
    parameters = archive['frozen_parameters']
    base_dev = C18.precompute_baseline(dev, parameters, devT)
    base_full = C18.precompute_baseline(datasets, parameters, T)
    decoder = R20.SDP.load_probability_decoder()
    warm_paths, _ = R20.REPLAY.output_paths()
    predictions = {a: {} for a in ASSETS}; gates, details = {}, {}
    ids = {a: np.concatenate([D['va'], D['te']]) for a, D in datasets.items()}
    epoch_binding = selected_epochs(archive)
    if authorization is not None and authorization['checkpoint_selected_epochs'] != epoch_binding:
        raise PermissionError('Selected epochs differ from the frozen checkpoint/result bindings')
    with np.load(PREDICTIONS, allow_pickle=False) as saved:
        for a in ASSETS:
            archive_dates(datasets[a], saved, a+'_')
        for variant in R20.VARIANTS:
            for seed in SEEDS:
                key = f'{variant}_seed{seed}'; canonical = 'R20_'+key
                row = archive['checkpoints'][key]
                expected_path = required_source_paths()[f'R20_checkpoint_{key}']
                assert Path(row['path']).resolve() == expected_path and digest(expected_path) == row['sha256']
                ckpt = torch.load(expected_path, weights_only=True, map_location='cpu')
                meta = ckpt['metadata']
                assert meta['seed'] == seed and meta['variant'] == variant
                assert meta['selected_epoch'] == epoch_binding[canonical]
                warm = torch.load(warm_paths[seed], weights_only=True, map_location='cpu')
                model = R20.OperatorGRU(dev, parameters, base_dev, decoder, meta['volume_AR'], variant, seed, warm['state_dict'])
                model.load_state_dict(ckpt['state_dict'], strict=True)
                details[canonical] = dict(selected_epoch=meta['selected_epoch'], checkpoint_sha256=row['sha256'],
                    extension=extend_data_buffers(model, datasets, base_full, r20=True))
                value = predict_r20(model, datasets, T, ids)
                gates[canonical] = {}
                for a in ASSETS:
                    gates[canonical][a] = assert_matching(value[a][:len(datasets[a]['va'])], saved[f'{a}_{key}'])
                    predictions[a][canonical] = value[a]
            for a in ASSETS:
                name = f'R20_{variant}_ensemble'
                predictions[a][name] = np.mean([predictions[a][f'R20_{variant}_seed{s}'] for s in SEEDS], axis=0)
                gates.setdefault(name, {})[a] = assert_matching(predictions[a][name][:len(datasets[a]['va'])], saved[f'{a}_{variant}_ensemble'])
    with np.load(R20.REPLAY.ARCHIVE_PREDICTIONS, allow_pickle=False) as saved:
        for seed in SEEDS:
            warm = torch.load(warm_paths[seed], weights_only=True, map_location='cpu')
            assert warm['metadata']['seed'] == seed and warm['metadata']['epoch'] == epoch_binding[f'C18_GRU_seed{seed}']
            model = C18.DirectControls(dev, parameters, base_dev, 'shared_GRU', seed)
            model.load_state_dict(warm['state_dict'], strict=True)
            extend_data_buffers(model, datasets, base_full, r20=False)
            value = predict_old_gru(model, datasets, T, ids); name = f'C18_GRU_seed{seed}'; gates[name] = {}
            for a in ASSETS:
                archive_dates(datasets[a], saved, a+'_')
                gates[name][a] = assert_matching(value[a][:len(datasets[a]['va'])], saved[f'{a}_shared_GRU_seed{seed}'])
                predictions[a][name] = value[a]
        for a in ASSETS:
            predictions[a]['C18_GRU_ensemble'] = np.mean([predictions[a][f'C18_GRU_seed{s}'] for s in SEEDS], axis=0)
            gates.setdefault('C18_GRU_ensemble', {})[a] = assert_matching(predictions[a]['C18_GRU_ensemble'][:len(datasets[a]['va'])], saved[f'{a}_shared_GRU_ensemble'])
    mem = read_json(HERE/'mem_baselines_round16_results.json')
    classical = read_json(HERE/'round18_classical_measurements_results.json')
    with np.load(HERE/'mem_baselines_round16_predictions.npz', allow_pickle=False) as mem_saved, \
         np.load(HERE/'round18_classical_measurements_validation_predictions.npz', allow_pickle=False) as classical_saved:
        for a in ASSETS:
            D = datasets[a]; nva = len(D['va']); theta16 = read_json(S18.STAGE16[a]['results'])
            with np.load(S18.STAGE16[a]['predictions'], allow_pickle=False) as saved:
                archive_dates(D, saved)
                for name, mode in (('R16_frozen','fixed_precision'), ('R16_raw','raw_constant')):
                    fits = theta16['fits'][mode]; fit = fits[0]
                    assert all(f['dynamics'] == fit['dynamics'] for f in fits), 'Non-neural R16 seed reuse changed'
                    value = fixed_kf(D, fit['dynamics'], ids[a], raw=mode=='raw_constant')
                    gates.setdefault(name, {})[a] = assert_matching(value[:nva], saved[mode])
                    predictions[a][name] = value
                # Literal original implementation; ONLY pre2025 common tr is fit.
                # A single predict call per horizon covers merged va+te.
                directD = dict(D, baseline_x=D['baseline_x_F16'], va=ids[a])
                direct = B.direct_baselines(directD)
                for name, value in direct.items():
                    gates.setdefault(name, {})[a] = assert_matching(value[:nva], saved[name])
                    predictions[a][name] = value
            archive_dates(D, mem_saved, a+'_')
            proxy = np.zeros(D['n']); proxy[D['observed']] = np.exp(D['log_proxy'][D['observed']])
            for name, integrated in (('MEM_stationary',False), ('MEM_integrated',True)):
                fit = mem['markets'][a]['fits'][name]; scale = fit['scale_from_pre2025_observations']
                assert proxy[:D['cut']].mean() == scale, 'MEM pre2025 scale changed'
                pars = MEM.decode(np.array(fit['selected']['theta']), integrated)
                value = MEM.forecasts(proxy/scale, D['origin'][ids[a]], pars) * scale
                gates.setdefault(name, {})[a] = assert_matching(value[:nva], mem_saved[f'{a}_{name}'])
                predictions[a][name] = value
            archive_dates(D, classical_saved, a+'_')
            measures = classical_observations(D)
            for method in ('GK','RS'):
                logvalue, observed = CLASSICAL.observations(measures[method])
                cal = classical['calibration']['estimators'][method]
                fit = classical['fits'][a+'_'+method]
                value = fixed_kf(D, fit['dynamics'], ids[a], raw=True, log_proxy=logvalue,
                    observed=observed, bias=cal['observation_bias_for_log_cP_sigma2'], variance=cal['global_R'])
                name = method+'_global_KF'; gates.setdefault(name, {})[a] = assert_matching(value[:nva], classical_saved[a+'_'+method])
                predictions[a][name] = value
    return predictions, gates, details, epoch_binding


def model_scores(value, y, ewma):
    assert value.shape == y.shape and (value > 0).all() and np.isfinite(value).all()
    assert (y >= 0).all() and np.isfinite(y).all() and (ewma > 0).all()
    raw = np.log(value) + y/value
    known = raw - np.log(ewma)[:,None]
    squared = (value-y)**2
    out = dict(raw_QLIKE=float(raw.mean()), known_scale_QLIKE=float(known.mean()), MSE=float(squared.mean()), per_h={})
    normalized_means = []
    for j, h in enumerate(HORIZONS):
        positive = y[:,j] > 0
        norm = raw[positive,j] - np.log(y[positive,j]) - 1
        normalized = float(norm.mean()) if len(norm) else None
        normalized_means.append(normalized)
        out['per_h'][str(h)] = dict(n=len(y), positive_y=int(positive.sum()), raw_QLIKE=float(raw[:,j].mean()),
            known_scale_QLIKE=float(known[:,j].mean()), normalized_QLIKE=normalized, MSE=float(squared[:,j].mean()))
    out['normalized_QLIKE'] = float(np.mean(normalized_means)) if all(v is not None for v in normalized_means) else None
    return out


def synchronized_block_ci(differences, dates, block, *, repetitions=2000):
    """Resample the same origin-date blocks jointly across both assets/horizons."""
    common = np.intersect1d(dates[ASSETS[0]], dates[ASSETS[1]])
    if len(common) < block:
        return dict(available=False, reason='fewer synchronized origin dates than block length', n_dates=len(common), block=block)
    rows = [differences[a][np.searchsorted(dates[a], common)].mean(axis=1) for a in ASSETS]
    series = np.mean(rows, axis=0); n = len(series)
    rng = np.random.default_rng(20261022+block); count = (n+block-1)//block
    starts = rng.integers(0, n-block+1, size=(repetitions,count))
    sample = (starts[...,None]+np.arange(block)).reshape(repetitions,-1)[:,:n]
    means = series[sample].mean(axis=1)
    lo, hi = np.quantile(means, [.025,.975])
    return dict(available=True, block=block, n_dates=n, repetitions=repetitions, synchronized_mean=float(series.mean()),
        lower95=float(lo), upper95=float(hi), date_first=str(common[0]), date_last=str(common[-1]),
        method='noncircular moving origin-date block percentile CI; same draw across assets and horizons',
        limitation='overlapping h22 targets; limited effective sample; seeds are not independent datasets')


def score_predictions(pred, datasets, split, selection=None):
    # Called ONLY after predictions have been written successfully.
    ids = {a: datasets[a][split] for a in ASSETS}
    scores, joint = {}, {}
    names = sorted(pred[ASSETS[0]])
    for a in ASSETS:
        D = datasets[a]; y = D['y'][ids[a]]; ewma = D['ewma'][ids[a]]
        scores[a] = {name: model_scores(pred[a][name], y, ewma) for name in names}
    for name in names:
        joint[name] = {metric: sum(scores[a][name][metric] for a in ASSETS)/2 for metric in
                       ('raw_QLIKE','known_scale_QLIKE','MSE')}
        positive = [scores[a][name]['normalized_QLIKE'] for a in ASSETS]
        joint[name]['normalized_QLIKE'] = float(np.mean(positive)) if all(v is not None for v in positive) else None
    report = dict(asset_scores=scores, equal_asset_horizon=joint,
        sign='QLIKE/MSE difference primary-minus-comparator <0 favors primary; gain=comparator-minus-primary >0 favors primary',
        no_raw_QLIKE_percentages=True,
        limitations=['Parkinson range proxy, not true IV or return variance',
            'Repeated 2025 development selection; one later frozen 2026 confirmation',
            'No identification of true sampling M, activity A, or causal measurement bias',
            'No economic trading-return or P&L claim; seeds are paired initializations, not independent datasets'])
    if selection is None:
        return report
    primary = selection['selected_candidate']; comparisons = selection['comparisons']; paired = {}; comparisons_report = {}
    dates = {a: datasets[a]['date'][datasets[a]['origin'][ids[a]]] for a in ASSETS}
    for other in comparisons:
        delta = {}
        for a in ASSETS:
            y = datasets[a]['y'][ids[a]]; p, b = pred[a][primary], pred[a][other]
            delta[a] = np.log(p)+y/p-np.log(b)-y/b
        comparisons_report[other] = dict(
            raw_QLIKE_difference=joint[primary]['raw_QLIKE']-joint[other]['raw_QLIKE'],
            known_scale_QLIKE_difference=joint[primary]['known_scale_QLIKE']-joint[other]['known_scale_QLIKE'],
            MSE_difference=joint[primary]['MSE']-joint[other]['MSE'],
            loss_gain=joint[other]['raw_QLIKE']-joint[primary]['raw_QLIKE'],
            asset_per_h={a:{str(h):float(delta[a][:,j].mean()) for j,h in enumerate(HORIZONS)} for a in ASSETS},
            synchronized_CI={str(b):synchronized_block_ci(delta,dates,b) for b in (22,44)})
        pbase, obase = primary.removesuffix('_ensemble'), other.removesuffix('_ensemble')
        if all(f'{pbase}_seed{s}' in pred[ASSETS[0]] and f'{obase}_seed{s}' in pred[ASSETS[0]] for s in SEEDS):
            paired[other] = {str(s):dict(raw_QLIKE_difference=joint[f'{pbase}_seed{s}']['raw_QLIKE']-joint[f'{obase}_seed{s}']['raw_QLIKE'],
                known_scale_QLIKE_difference=joint[f'{pbase}_seed{s}']['known_scale_QLIKE']-joint[f'{obase}_seed{s}']['known_scale_QLIKE']) for s in SEEDS}
    report.update(primary=primary, frozen_criteria=selection['criteria'], comparisons=comparisons_report,
        seed_paired=paired, primary_unchanged_after_open=True, criteria_note='Frozen criteria reported verbatim; no post-test model or rule selection')
    return report


def run(*, development_only=True, selection_path=None):
    authorization = None
    if not development_only:
        authorization = authorize_test(selection_path)  # FIRST, before inputs or test frame I/O
    elif selection_path is not None:
        raise ValueError('Selection may only be consumed with explicit --test')
    report_path = DEV_REPORT if development_only else TEST_REPORT
    pred_path = DEV_PREDICTIONS if development_only else TEST_PREDICTIONS
    reject_existing([report_path,pred_path])
    before = source_hashes()
    archive = read_json(RESULTS)
    # Bind the source implementation that wrote R20 weights.
    for name, entry in archive['source_hashes'].items():
        assert digest(entry['path']) == entry['sha256'], f'Archived R20 dependency changed: {name}'
    if authorization is not None and selected_epochs(archive) != authorization['checkpoint_selected_epochs']:
        raise PermissionError('Checkpoint selected epochs differ from frozen selection')
    started = time.time()
    datasets,T,audit = HB.build_pair(development_only=development_only, selection_path=selection_path, check_npz=True)
    assert audit['preprocessing_sha256'] == PREPROCESSING_SHA256
    if development_only:
        assert all(len(D['te']) == 0 and (D['date'] < HB.TEST_START).all() for D in datasets.values())
    combined,gates,details,epochs = collect_predictions(datasets,T,archive,authorization)
    split = 'va' if development_only else 'te'; predictions = {a:{} for a in ASSETS}; arrays = dict(horizons=np.array(HORIZONS))
    for a in ASSETS:
        D = datasets[a]; chosen = D[split]; q = D['origin'][chosen]
        arrays.update({a+'_date':D['date'][q],a+'_origin':q,a+'_label_end':D['date'][q+22],a+'_target':D['y'][chosen],a+'_ewma':D['ewma'][chosen]})
        for name,value in combined[a].items():
            tail = value[:len(D['va'])] if development_only else value[len(D['va']):]
            assert len(tail) == len(chosen)
            predictions[a][name] = tail; arrays[f'{a}_{name}'] = tail
    assert source_hashes() == before, 'Inputs changed during prediction replay'
    with pred_path.open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    # Predictions are now durable; only now compute scores/comparisons/CIs.
    scored = score_predictions(predictions,datasets,split,authorization)
    assert source_hashes() == before, 'Inputs changed during scoring'
    report = dict(status='DEVELOPMENT REPLAY ONLY; TEST CLOSED' if development_only else 'FROZEN AUTHORIZED TEST',
        source_hashes=before, selected_checkpoint_epochs=epochs, builder=audit, development_replay=gates,
        checkpoint_reconstruction=details, predictions=dict(path=str(pred_path),sha256=digest(pred_path)),
        scores=scored, primary=None if authorization is None else authorization['selected_candidate'],
        selection_path=None if selection_path is None else str(Path(selection_path).resolve()),
        selection_sha256=None if selection_path is None else digest(selection_path),
        selection_file_created=False, test_targets_or_predictions_constructed=not development_only,
        neural_MEM_KF_training_performed=False, epoch_selection_performed=False,
        HAR_HGB='original B.direct_baselines; pre2025 tr only, F16; one merged va+te predict per horizon; old2025 gate mandatory',
        normalization_bank_AR='original pre2025/checkpoint values; no2025 refit',
        scoring_after_prediction_write=True, elapsed_seconds=time.time()-started)
    with report_path.open('x',encoding='utf-8') as handle:
        json.dump(report,handle,ensure_ascii=False,indent=2,allow_nan=False)
    print('REPLAY_COMPLETE',str(report_path),'source_sha256',digest(__file__),flush=True)
    print('MAX_DEVELOPMENT_ERROR',max(x['max_absolute_error'] for g in gates.values() for x in g.values()),flush=True)


def _synthetic_checks():
    # Missing/falsely enabled selection must fail before any CSV/NPZ/checkpoint I/O.
    touched = []
    with patch('pandas.read_csv',side_effect=lambda *a,**k:touched.append('csv')), \
         patch('numpy.load',side_effect=lambda *a,**k:touched.append('npz')), \
         patch('torch.load',side_effect=lambda *a,**k:touched.append('pt')):
        try:
            run(development_only=False,selection_path=None)
        except PermissionError:
            pass
        else:
            raise AssertionError('Test opened without an explicit selection')
    assert not touched
    for p in ({},dict(allow_test=False),dict(allow_test=True,preprocessing_sha256='0'*64)):
        try:validate_selection_contract(p)
        except PermissionError:pass
        else:raise AssertionError('Invalid test contract accepted')
    D,_ = S17.synthetic_pair(); parameters=S18._synthetic_parameters(); T=S17.make_tensors(D)
    base=C18.precompute_baseline(D,parameters,T); decoder=R20.synthetic_decoder(); ar=R20.fit_volume_ar(D)
    warm=C18._nonzero(C18.DirectControls(D,parameters,base,'shared_GRU',11))
    model=R20.OperatorGRU(D,parameters,base,decoder,ar,'joint',11,warm.state_dict())
    before={k:v.clone() for k,v in model.state_dict().items()}
    extended=copy.deepcopy(D); extendedT={a:{k:v.clone() for k,v in T[a].items()} for a in ASSETS}
    for a in ASSETS:
        extra=7; oldn=D[a]['n']; extended[a]['n']+=extra
        for key in ('F17','log_proxy','cell_b','cell_R','even'):
            row=np.asarray(D[a][key])[-1:]
            extended[a][key]=np.concatenate([D[a][key],np.repeat(row,extra,axis=0)])
        extended[a]['observed']=np.concatenate([D[a]['observed'],np.ones(extra,dtype=bool)])
        for key in ('F17','log_proxy','cell_b','cell_R','even'):
            extendedT[a][key]=torch.tensor(extended[a][key],dtype=DTYPE)
    fullbase=C18.precompute_baseline(extended,parameters,extendedT)
    extension=extend_data_buffers(model,extended,fullbase,r20=True)
    assert extension['all_old_prefixes_exact']
    for key,value in before.items():
        if key.startswith(('m0_','P0_','K0_','observed_','cells_')):
            torch.testing.assert_close(model.state_dict()[key][:len(value)],value,rtol=0,atol=0)
        else:
            assert torch.equal(model.state_dict()[key],value)
    # Gate rejects differences at the fixed tolerance; no fallback is substituted.
    expected=np.full((4,3),.001)
    assert_matching(expected,expected)
    try:assert_matching(expected*1.01,expected)
    except AssertionError:pass
    else:raise AssertionError('Prediction mismatch accepted')
    y=np.array([[0.,.001,.002],[.001,.002,.003]])
    scores=model_scores(np.full_like(y,.002),y,np.full(2,.002))
    assert scores['per_h']['1']['positive_y']==1 and scores['normalized_QLIKE'] is not None
    dates={a:np.arange(np.datetime64('2025-01-01'),np.datetime64('2025-03-02')) for a in ASSETS}
    diff={a:np.full((60,3),-.02) for a in ASSETS}
    for block in (22,44):
        ci=synchronized_block_ci(diff,dates,block,repetitions=100)
        np.testing.assert_allclose([ci['lower95'],ci['upper95']],[-.02,-.02])
    return dict(test_without_selection_rejected_before_data_IO=True, invalid_selection_contract_rejected=True,
        frozen_cache_extension_prefix_exact=True, learned_theta_bank_decoder_AR_unchanged=True,
        mismatch_gate_rejects_without_tolerance_relaxation=True, positive_y_only_normalized_QLIKE=True,
        synchronized_block22_44_correct=True, no_raw_QLIKE_percentage=True,
        real2026_targets_or_predictions_constructed=False)


def synthetic_checks():
    with ExitStack() as stack:
        for name in ('pathlib.Path.read_text','pandas.read_csv','numpy.load','torch.load',
                     'round17_shared_precision.prepare_pair','sampling_density_prior_probability.load_probability_decoder'):
            stack.enter_context(patch(name,side_effect=AssertionError('real input forbidden in purechecks')))
        return _synthetic_checks()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    group=ap.add_mutually_exclusive_group()
    group.add_argument('--development-only',action='store_true');group.add_argument('--test',action='store_true')
    ap.add_argument('--selection',type=Path);ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if args.check_only:
        print(json.dumps(dict(checks=synthetic_checks(),source_sha256=digest(__file__)),indent=2),flush=True)
    else:
        run(development_only=not args.test,selection_path=args.selection)


if __name__=='__main__':main()
