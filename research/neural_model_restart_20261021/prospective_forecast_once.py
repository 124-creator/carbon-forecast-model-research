"""One-off prospective prediction logger. Default is synthetic purechecks.

This delivery runs ONLY --check-only and --development-replay. The separate
--record-prospective entry is for later human-reviewed execution before the fixed
2026-10-22T13:30:00Z deadline. No training, evaluate/scoring calls, fake query
labels, source-global monkeypatching, network requests, or scheduled jobs.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import copy
from datetime import datetime, timezone, timedelta
import hashlib
import io
import json
from pathlib import Path
import types
import numpy as np
import pandas as pd
import torch
import round20_holdout_runner as RR

HB, R20, C18, S18, S17, B, MEM = RR.HB, RR.R20, RR.C18, RR.S18, RR.S17, RR.B, RR.MEM
HERE = Path(__file__).resolve().parent
ACQ = HERE / 'data_acquisition_round13'
ASSETS, HORIZONS, SEEDS = tuple(RR.ASSETS), tuple(RR.HORIZONS), tuple(RR.SEEDS)
SELECTION = HERE / 'round20_frozen_selection.json'
SELECTION_SHA = 'd39a08d0db8f99eefb37e0fbac75c0821eee9578719d8d6264c785a387785617'
INPUT_MANIFEST = ACQ / 'KRBN_KCCA_latest_complete_inputs_20261021_round17_manifest.json'
INPUT_MANIFEST_SHA = 'fc34fb4b1e0fa1d22ce87a4b36849ad7f89494b6a277a5136374ceb44c1a1ef3'
LIVE_ORIGIN = '2026-10-21'
DEADLINE = datetime(2026, 10, 22, 13, 30, tzinfo=timezone.utc)
REQUIRED_INPUT_CLOSE = datetime(2026, 10, 21, 20, tzinfo=timezone.utc)
COLUMNS = ('date', 'timestamp_utc', 'open', 'high', 'low', 'close', 'volume', 'adj_close')
DATA_BUFFERS = ('m0_', 'P0_', 'K0_', 'observed_', 'cells_')


def utc_now():
    """No CLI override, date spoofing option, or supplied forecast creation time."""
    return datetime.now(timezone.utc)


def iso(t):
    return t.astimezone(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def checked_time(now, fetch_times=()):
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError('Timezone-aware host UTC required')
    now = now.astimezone(timezone.utc)
    if now >= DEADLINE:
        raise PermissionError('Prospective deadline passed; refuse, never backdate')
    if now < REQUIRED_INPUT_CLOSE:
        raise PermissionError('Origin regular session not complete')
    if any(t.tzinfo is None or t > now or t < REQUIRED_INPUT_CLOSE for t in fetch_times):
        raise PermissionError('Input fetch time is future-dated or before completed close')
    return now


def reject_existing(paths):
    for path in paths:
        if Path(path).exists():
            raise FileExistsError(f'Exclusive output already exists: {path}')


def globals_stamp():
    return (B.R14.SOURCE, B.RESERVED, tuple(sorted(B.SOURCES.items())))


def validate_sources():
    if digest(SELECTION) != SELECTION_SHA:
        raise PermissionError('Reviewed frozen-selection bytes changed')
    selected = read_json(SELECTION)
    # This reuses the original integrity API, NOT its test builder or test runner.
    HB.validate_selection_payload(selected, RR.required_source_paths(include_development_report=True))
    RR.validate_selection_contract(selected)
    before = RR.source_hashes()
    before.update(selection=dict(path=str(SELECTION), sha256=SELECTION_SHA))
    archive = read_json(RR.RESULTS)
    for key, entry in archive['source_hashes'].items():
        if digest(entry['path']) != entry['sha256']:
            raise PermissionError(f'Archived dependency changed: {key}')
    if RR.selected_epochs(archive) != selected['checkpoint_selected_epochs']:
        raise PermissionError('Selected epochs changed')
    return archive, before


def assert_sources_unchanged(before):
    for name, entry in before.items():
        if digest(entry['path']) != entry['sha256']:
            raise PermissionError(f'Frozen source/checkpoint changed: {name}')


def inference_tensors(datasets):
    # Neither future nor historical labels are passed to any prediction API.
    return {a: {k: torch.tensor(D[k], dtype=RR.DTYPE)
                for k in ('log_proxy', 'even', 'cell_b', 'cell_R', 'F17')}
            for a, D in datasets.items()}


def state_digest(model, omit_data=False):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if omit_data and name.startswith(DATA_BUFFERS):
            continue
        h.update(name.encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def positive(value):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (len(HORIZONS),) or not np.isfinite(value).all() or not (value > 0).all():
        raise ValueError('Expected positive finite h1/h5/h22 point forecasts')
    return value


def development_gate(archive):
    """Reconstruct all requested frozen models and replay 2025, with no fitting."""
    stamp = globals_stamp()
    dev, _, audit = HB.build_pair(development_only=True, check_npz=True)
    if audit['preprocessing_sha256'] != RR.PREPROCESSING_SHA256:
        raise PermissionError('Pre2025 normalizer/bank digest changed')
    if any(len(D['te']) or not (D['date'] < HB.TEST_START).all() for D in dev.values()):
        raise PermissionError('Development gate must contain no real2026 observations')
    T = inference_tensors(dev)
    parameters = archive['frozen_parameters']
    base = C18.precompute_baseline(dev, parameters, T)
    decoder = R20.SDP.load_probability_decoder()
    warm_paths, _ = R20.REPLAY.output_paths()
    models, gates, expected, predictions = {}, {}, {}, {a: {} for a in ASSETS}
    ids = {a: D['va'] for a, D in dev.items()}
    with np.load(RR.PREDICTIONS, allow_pickle=False) as saved:
        for a in ASSETS:
            RR.archive_dates(dev[a], saved, a+'_')
            expected[a] = {'dates': saved[a+'_date'].copy(), 'values': {}}
        for variant in R20.VARIANTS:
            for seed in SEEDS:
                short = f'{variant}_seed{seed}'; name = 'R20_'+short
                path = RR.required_source_paths()[f'R20_checkpoint_{short}']
                row = archive['checkpoints'][short]
                if Path(row['path']).resolve() != path or digest(path) != row['sha256']:
                    raise PermissionError(f'Checkpoint binding changed: {short}')
                ckpt = torch.load(path, weights_only=True, map_location='cpu')
                meta = ckpt['metadata']
                if meta['seed'] != seed or meta['variant'] != variant:
                    raise PermissionError('Checkpoint seed/variant mismatch')
                if meta['selected_epoch'] != RR.selected_epochs(archive)[name]:
                    raise PermissionError('Checkpoint epoch changed')
                warm = torch.load(warm_paths[seed], weights_only=True, map_location='cpu')
                model = R20.OperatorGRU(dev, parameters, base, decoder, meta['volume_AR'], variant, seed, warm['state_dict'])
                model.load_state_dict(ckpt['state_dict'], strict=True)
                model.eval(); model.requires_grad_(False)
                before = state_digest(model)
                value = RR.predict_r20(model, dev, T, ids)
                assert state_digest(model) == before
                gates[name] = {}
                for a in ASSETS:
                    arr = saved[f'{a}_{short}'].copy()
                    gates[name][a] = RR.assert_matching(value[a], arr)
                    expected[a]['values'][name] = arr
                    predictions[a][name] = value[a]
                models[name] = dict(model=model, r20=True, checkpoint=str(path), sha256=digest(path), epoch=meta['selected_epoch'])
            name = f'R20_{variant}_ensemble'; gates[name] = {}
            for a in ASSETS:
                value = np.mean([predictions[a][f'R20_{variant}_seed{s}'] for s in SEEDS], axis=0)
                arr = saved[f'{a}_{variant}_ensemble'].copy()
                gates[name][a] = RR.assert_matching(value, arr)
                expected[a]['values'][name] = arr
    with np.load(R20.REPLAY.ARCHIVE_PREDICTIONS, allow_pickle=False) as saved:
        for seed in SEEDS:
            name = f'C18_GRU_seed{seed}'
            ckpt = torch.load(warm_paths[seed], weights_only=True, map_location='cpu')
            if ckpt['metadata']['seed'] != seed or ckpt['metadata']['epoch'] != R20.REPLAY.SELECTED[seed]:
                raise PermissionError('OldGRU checkpoint metadata changed')
            model = C18.DirectControls(dev, parameters, base, 'shared_GRU', seed)
            model.load_state_dict(ckpt['state_dict'], strict=True)
            model.eval(); model.requires_grad_(False)
            before = state_digest(model)
            value = RR.predict_old_gru(model, dev, T, ids)
            assert state_digest(model) == before
            gates[name] = {}
            for a in ASSETS:
                RR.archive_dates(dev[a], saved, a+'_')
                arr = saved[f'{a}_shared_GRU_seed{seed}'].copy()
                gates[name][a] = RR.assert_matching(value[a], arr)
                expected[a]['values'][name] = arr
                predictions[a][name] = value[a]
            models[name] = dict(model=model, r20=False, checkpoint=str(warm_paths[seed]), sha256=digest(warm_paths[seed]), epoch=ckpt['metadata']['epoch'])
        gates['C18_GRU_ensemble'] = {}
        for a in ASSETS:
            value = np.mean([predictions[a][f'C18_GRU_seed{s}'] for s in SEEDS], axis=0)
            arr = saved[f'{a}_shared_GRU_ensemble'].copy()
            gates['C18_GRU_ensemble'][a] = RR.assert_matching(value, arr)
            expected[a]['values']['C18_GRU_ensemble'] = arr
    mem = read_json(HERE/'mem_baselines_round16_results.json')
    theta = {}; scales = {}
    with np.load(HERE/'mem_baselines_round16_predictions.npz', allow_pickle=False) as saved_mem:
        for a in ASSETS:
            D = dev[a]; theta16 = read_json(S18.STAGE16[a]['results']); theta[a] = {}; scales[a] = {}
            with np.load(S18.STAGE16[a]['predictions'], allow_pickle=False) as saved:
                RR.archive_dates(D, saved)
                for name, mode in (('R16_frozen', 'fixed_precision'), ('R16_raw', 'raw_constant')):
                    fits = theta16['fits'][mode]
                    assert all(f['dynamics'] == fits[0]['dynamics'] for f in fits)
                    theta[a][name] = fits[0]['dynamics']
                    value = RR.fixed_kf(D, theta[a][name], D['va'], raw=name=='R16_raw')
                    gates.setdefault(name, {})[a] = RR.assert_matching(value, saved[mode])
                    expected[a]['values'][name] = saved[mode].copy()
                value = np.repeat(D['ewma'][D['va'], None], len(HORIZONS), axis=1)
                gates.setdefault('EWMA22', {})[a] = RR.assert_matching(value, saved['EWMA22'])
                expected[a]['values']['EWMA22'] = saved['EWMA22'].copy()
            RR.archive_dates(D, saved_mem, a+'_')
            proxy = np.zeros(D['n']); proxy[D['observed']] = np.exp(D['log_proxy'][D['observed']])
            for name, integrated in (('MEM_stationary', False), ('MEM_integrated', True)):
                fit = mem['markets'][a]['fits'][name]
                scale = fit['scale_from_pre2025_observations']
                assert proxy[:D['cut']].mean() == scale
                pars = MEM.decode(np.array(fit['selected']['theta']), integrated)
                scales[a][name] = dict(scale=scale, parameters=pars)
                value = MEM.forecasts(proxy/scale, D['origin'][D['va']], pars)*scale
                gates.setdefault(name, {})[a] = RR.assert_matching(value, saved_mem[f'{a}_{name}'])
                expected[a]['values'][name] = saved_mem[f'{a}_{name}'].copy()
    assert globals_stamp() == stamp
    return dict(dev=dev, tensors=T, audit=audit, models=models, gates=gates, expected=expected,
                parameters=parameters, theta=theta, mem=scales)


def label_free_prefix(D, stop):
    """An observation-only view. No origin/target/split compatibility keys."""
    keys = ('date', 'F17', 'even', 'cell_b', 'cell_R', 'observed', 'log_proxy',
            'proxy_raw', 'ewma22_all')
    E = {k: D[k][:stop].copy() for k in keys if k in D}
    E.update(n=stop, cut=D['cut'], global_b=D['global_b'], global_R=D['global_R'],
             bank_index=D['bank_index'].copy(), initial_mu=D['initial_mu'])
    assert stop >= E['cut'] and E['bank_index'].max() < E['cut']
    return E


@torch.no_grad()
def query_predictions(state, datasets, queries):
    """Direct observation-index queries; no invented target row or evaluate call."""
    Q = {a: label_free_prefix(datasets[a], queries[a]+1) for a in ASSETS}
    T = inference_tensors(Q)
    out = {a: {} for a in ASSETS}
    for name, entry in state['models'].items():
        model = entry['model']; before = state_digest(model)
        for a in ASSETS:
            D = Q[a]; q = np.array([queries[a]], dtype=int)
            assert q[0] == D['n']-1
            if entry['r20']:
                fields = model.filtered_fields(a, D, T[a], D['n'], 0)
                value = model.forecast(a, T[a], q, fields)[0].numpy()[0]
            else:
                delta, _, _ = model.fields(a, D, T[a], D['n'], 0)
                m, P = model.corrected_state(a, delta)
                value = model.processes[a].forecast(m[q], P[q]).numpy()[0]
            out[a][name] = positive(value)
        assert state_digest(model) == before, 'Forward changed frozen state'
    for a in ASSETS:
        for variant in R20.VARIANTS:
            out[a][f'R20_{variant}_ensemble'] = positive(np.mean([out[a][f'R20_{variant}_seed{s}'] for s in SEEDS], axis=0))
        out[a]['C18_GRU_ensemble'] = positive(np.mean([out[a][f'C18_GRU_seed{s}'] for s in SEEDS], axis=0))
        # RR.fixed_kf accepts a query origin array, never accesses y; this view has no y.
        D = dict(Q[a], origin=np.array([queries[a]], dtype=int))
        for name in ('R16_raw', 'R16_frozen'):
            out[a][name] = positive(RR.fixed_kf(D, state['theta'][a][name], np.array([0]), raw=name=='R16_raw')[0])
        proxy = np.zeros(D['n']); proxy[D['observed']] = np.exp(D['log_proxy'][D['observed']])
        for name in ('MEM_stationary', 'MEM_integrated'):
            fit = state['mem'][a][name]
            out[a][name] = positive((MEM.forecasts(proxy/fit['scale'], D['origin'], fit['parameters'])*fit['scale'])[0])
        out[a]['EWMA22'] = positive(np.repeat(Q[a]['ewma22_all'][-1], len(HORIZONS)))
    return out


def append_bar_exact(old, bar):
    if tuple(old.columns) != COLUMNS or tuple(bar.columns) != COLUMNS:
        raise ValueError('Unexpected OHLC schema or column order')
    if len(bar) != 1 or str(bar.date.iloc[0].date()) != LIVE_ORIGIN:
        raise ValueError('Exactly one completed2026-10-21 input bar required')
    if str(old.date.iloc[-1].date()) != '2026-10-20':
        raise ValueError('Unexpected frozen snapshot endpoint')
    if not old.date.is_unique or not old.date.is_monotonic_increasing:
        raise ValueError('Invalid frozen date order')
    out = pd.concat([old, bar], ignore_index=True)
    if not out.date.is_unique or not out.date.is_monotonic_increasing:
        raise ValueError('Append duplicated or reordered a date')
    assert_prefix_exact(old, out)
    return out


def assert_prefix_exact(old, extended):
    if tuple(old.columns) != tuple(extended.columns) or len(extended) < len(old):
        raise ValueError('Snapshot schema/prefix length changed')
    for column in old:
        np.testing.assert_array_equal(old[column].to_numpy(), extended[column].iloc[:len(old)].to_numpy(), err_msg='Old numeric/date prefix changed: '+column)


def load_live_frames(now):
    if digest(INPUT_MANIFEST) != INPUT_MANIFEST_SHA:
        raise PermissionError('Reviewed input manifest changed')
    manifest = read_json(INPUT_MANIFEST); frames = {}; audit = {}; fetch_times = []
    for rec in manifest['requests']:
        a = rec['ticker']
        if a not in ASSETS or rec['http_status'] != 200 or rec['new_completed_rows'] != 1:
            raise PermissionError('Unexpected input request identity')
        for filekey, hashkey in (('new_bar_file', 'new_bar_sha256'), ('raw_file', 'raw_sha256')):
            if Path(rec[filekey]).name != rec[filekey] or digest(ACQ/rec[filekey]) != rec[hashkey]:
                raise PermissionError('New input/source hash or basename changed')
        fetched = datetime.fromisoformat(rec['retrieved_utc']); fetch_times.append(fetched)
        checked_time(now, fetch_times)
        old = pd.read_csv(B.SOURCES[a], parse_dates=['date'])
        bar = pd.read_csv(ACQ/rec['new_bar_file'], parse_dates=['date'], float_precision='round_trip')
        frame = append_bar_exact(old, bar)
        HB.observations(frame)  # validates known OHLC and volume only
        frames[a] = frame
        audit[a] = dict(old_snapshot=str(B.SOURCES[a]), old_sha256=digest(B.SOURCES[a]),
            new_bar=str(ACQ/rec['new_bar_file']), new_bar_sha256=rec['new_bar_sha256'],
            raw_response=str(ACQ/rec['raw_file']), raw_sha256=rec['raw_sha256'], source_url=rec['url'],
            fetch_time_utc=iso(fetched), old_rows=len(old), extended_rows=len(frame), old_prefix_exact=True)
    if set(frames) != set(ASSETS):
        raise PermissionError('Both reviewed asset inputs required')
    return frames, audit, fetch_times


def assemble_live(frames, state):
    model = read_json(B.REFERENCE)['grid512']
    reference = read_json(S17.PROXY.REFERENCE_RESULTS)
    bundle = dict(model=model, c=float(reference['resolution_sensitivity']['mean_P_by_train_resolution']['512']))
    datasets, audit = HB.assemble_pair(frames, bundle, check_npz=True, expected_preprocessing=RR.PREPROCESSING_SHA256)
    for a, D in datasets.items():
        assert str(pd.Timestamp(D['date'][-1]).date()) == LIVE_ORIGIN
        # Helper compatibility labels are historical and fully mature only.
        assert (D['origin']+max(HORIZONS) < D['n']).all()
        assert (D['label_dates'] <= D['date'][-1]).all()
        assert D['n']-1 not in D['origin'], 'Future query must not have a target row'
        for field in ('F17', 'even', 'cell_b', 'cell_R', 'observed', 'log_proxy'):
            np.testing.assert_array_equal(D[field][:state['dev'][a]['n']], state['dev'][a][field])
    T = inference_tensors(datasets)
    baseline = C18.precompute_baseline(datasets, state['parameters'], T)
    extensions = {}
    for name, entry in state['models'].items():
        model = entry['model']; before = state_digest(model, omit_data=True)
        extensions[name] = RR.extend_data_buffers(model, datasets, baseline, r20=entry['r20'])
        assert state_digest(model, omit_data=True) == before
    return datasets, dict(builder=audit, extensions=extensions, helper_targets='historical mature only; future query has no target row')


def build_entry(predictions, origins, mode, created, source_hashes, gate, extra):
    return dict(schema='prospective_range_point_forecast_v1', mode=mode,
        is_forward=mode=='PROSPECTIVE', created_at_utc=iso(created), host_clock='datetime.now(timezone.utc); never CLI supplied',
        origin=origins, horizons=list(HORIZONS), horizon_clock='next observed sessions after each origin; not fixed calendar days',
        target_definition='mean of next1/5/22 observed Parkinson proxies, exact HB.observations proxy frame',
        target=None, future_label_dates=None, future_actuals_policy='write separate immutable actuals file later; never overwrite this prediction',
        forecasts={a: {name: positive(value).tolist() for name, value in models.items()} for a, models in predictions.items()},
        source_hashes=source_hashes, script_sha256=digest(__file__), development_gate=gate,
        training_performed=False, normalization_bank_AR_theta_checkpoint_unchanged=True,
        evaluate_or_scoring_called=False, unknown_future_targets_constructed=False,
        HAR_HGB_GK_RS='omitted; no fitting substitutes', old2026_reconfirmation=False,
        scheduled_daily_task=False, **extra)


def write_outputs(prefix, entry, predictions, prospective=False, fetch_times=()):
    """Exclusive predictions plus a separate commit receipt. Missing receipt is invalid.

    Receipt is the acceptance marker and binds both payload hashes. A late receipt
    is rejected rather than backdating or rewriting either immutable prediction.
    """
    paths = {suffix: HERE/(prefix+suffix) for suffix in ('.json', '.npz', '.commit.json')}
    reject_existing(paths.values())
    arrays = dict(horizons=np.array(HORIZONS), assets=np.array(ASSETS), created_at_utc=np.array(entry['created_at_utc']),
                  mode=np.array(entry['mode']), is_forward=np.array(entry['is_forward']))
    for a in ASSETS:
        arrays[a+'_origin_date'] = np.array(entry['origin'][a])
        for name, value in predictions[a].items(): arrays[a+'_'+name] = positive(value)
    bio = io.BytesIO(); np.savez_compressed(bio, **arrays); npz = bio.getvalue()
    entry['npz_sha256'] = hashlib.sha256(npz).hexdigest()
    entry['acceptance_rule'] = 'valid only with matching accepted exclusive commit receipt; receipt host time strictly before deadline for prospective'
    body = (json.dumps(entry, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')
    if prospective: checked_time(utc_now(), fetch_times)
    with paths['.npz'].open('xb') as f: f.write(npz)
    with paths['.json'].open('xb') as f: f.write(body)
    completed = utc_now()
    accepted = not prospective or (completed < DEADLINE and completed >= datetime.fromisoformat(entry['created_at_utc']))
    receipt = dict(accepted=accepted, completed_at_utc=iso(completed),
        json_sha256=hashlib.sha256(body).hexdigest(), npz_sha256=hashlib.sha256(npz).hexdigest(),
        mode=entry['mode'], is_forward=entry['is_forward'], deadline_utc=iso(DEADLINE) if prospective else None)
    with paths['.commit.json'].open('x', encoding='utf-8') as f: json.dump(receipt, f, indent=2, allow_nan=False)
    if not accepted: raise PermissionError('Commit crossed deadline/clock rollback; entry rejected, immutable payloads NOT overwritten')
    return {key: dict(path=str(path), sha256=digest(path)) for key, path in paths.items()}


def development_replay():
    archive, before = validate_sources(); stamp = globals_stamp()
    state = development_gate(archive)
    queries, origins, row_ids = {}, {}, {}
    for a in ASSETS:
        D = state['dev'][a]; row_ids[a] = len(D['va'])-1
        queries[a] = int(D['origin'][D['va'][row_ids[a]]])
        origins[a] = str(pd.Timestamp(D['date'][queries[a]]).date())
        assert origins[a].startswith('2025-')
    predictions = query_predictions(state, state['dev'], queries)
    terminal_gate = {}
    for a in ASSETS:
        for name, value in predictions[a].items():
            terminal_gate.setdefault(name, {})[a] = RR.assert_matching(value[None], state['expected'][a]['values'][name][row_ids[a]:row_ids[a]+1])
    assert globals_stamp() == stamp; assert_sources_unchanged(before)
    created = utc_now()
    entry = build_entry(predictions, origins, 'DEVELOPMENT_REPLAY_NOT_FORWARD', created, before, state['gates'],
        dict(historical_query_gate=terminal_gate, preprocessing_sha256=state['audit']['preprocessing_sha256'],
             real2026_inputs_processed=False, historical_targets_for_helper_only=True,
             replay_note='reused2025 archived origin; generated now; not a prospective record or a new confirmation'))
    return write_outputs('prospective_forecast_once_development_replay', entry, predictions)


def record_prospective():
    started = checked_time(utc_now())  # first operation; expired entry rejected before model/data I/O
    output_prefix = 'prospective_forecast_once_origin20261021'
    claim = HERE/(output_prefix+'.claim.json')
    reject_existing([claim]+[HERE/(output_prefix+s) for s in ('.json', '.npz', '.commit.json')])
    archive, before = validate_sources(); stamp = globals_stamp()
    state = development_gate(archive)  # no prospective forward until every requested dev gate passes
    checked_time(utc_now())
    frames, input_audit, fetched = load_live_frames(utc_now())
    datasets, assembly = assemble_live(frames, state)
    queries = {a: D['n']-1 for a, D in datasets.items()}
    predictions = query_predictions(state, datasets, queries)
    assert globals_stamp() == stamp; assert_sources_unchanged(before)
    created = checked_time(utc_now(), fetched)
    if created < started: raise PermissionError('Host clock moved backwards')
    with claim.open('x', encoding='utf-8') as f:
        json.dump(dict(started_at_utc=iso(started), claimed_at_utc=iso(utc_now()), script_sha256=digest(__file__)), f, indent=2)
    snapshots = {}
    for a, frame in frames.items():
        path = HERE/(output_prefix+'_'+a+'_input_snapshot.csv')
        reject_existing([path])
        # 17 significant digits plus round_trip reload preserve every old numeric value.
        with path.open('x', encoding='utf-8', newline='') as f: frame.to_csv(f, index=False, float_format='%.17g', date_format='%Y-%m-%d')
        restored = pd.read_csv(path, parse_dates=['date'], float_precision='round_trip')
        for col in COLUMNS: np.testing.assert_array_equal(restored[col].to_numpy(), frame[col].to_numpy())
        snapshots[a] = dict(path=str(path), sha256=digest(path), rows=len(frame), old_prefix_exact=True)
    created = checked_time(utc_now(), fetched)
    entry = build_entry(predictions, {a:LIVE_ORIGIN for a in ASSETS}, 'PROSPECTIVE', created, before, state['gates'],
        dict(started_at_utc=iso(started), deadline_utc=iso(DEADLINE), known_inputs=input_audit,
             known_input_asof_utc=iso(max(fetched)), input_manifest_sha256=INPUT_MANIFEST_SHA,
             input_snapshots=snapshots, observation_extension=assembly,
             query_observation_index=queries, historical_targets_for_helper_only=True,
             future_target=None, next_observation_dates=None,
             model_checkpoints={k: {field:v for field,v in e.items() if field!='model'} for k,e in state['models'].items()}))
    assert_sources_unchanged(before)
    return write_outputs(output_prefix, entry, predictions, prospective=True, fetch_times=fetched)


def purechecks():
    """Synthetic frames/models only; no checkpoint, archive, CSV or real label I/O."""
    checks = {}
    t = DEADLINE-timedelta(seconds=1)
    checked_time(t, [REQUIRED_INPUT_CLOSE+timedelta(seconds=1)])
    for bad in (DEADLINE, DEADLINE+timedelta(microseconds=1)):
        try: checked_time(bad)
        except PermissionError: pass
        else: raise AssertionError('Deadline equality/late time accepted')
    try: checked_time(t, [t+timedelta(seconds=1)])
    except PermissionError: pass
    else: raise AssertionError('Future-dated input accepted')
    checks['deadline_equal_and_late_rejected_no_backdating'] = True
    checks['future_input_asof_rejected'] = True
    old = pd.DataFrame(dict(date=pd.to_datetime(['2026-10-19','2026-10-20']), timestamp_utc=[1,2],
        open=[10.,11.], high=[12.,13.], low=[9.,10.], close=[11.,12.], volume=[100,200], adj_close=[11.,12.]))
    bar = old.iloc[-1:].copy(); bar['date'] = pd.Timestamp(LIVE_ORIGIN); bar['timestamp_utc']=3
    extended = append_bar_exact(old,bar)
    changed = extended.copy(); changed.loc[0,'close'] += .01
    try: assert_prefix_exact(old, changed)
    except AssertionError: pass
    else: raise AssertionError('Changed old prefix accepted')
    try: append_bar_exact(old,pd.concat([bar,bar]))
    except ValueError: pass
    else: raise AssertionError('Duplicate newbar accepted')
    checks['old_prefix_tamper_and_duplicate_bar_rejected'] = True
    D, _ = S17.synthetic_pair(); parameters=S18._synthetic_parameters()
    for a in ASSETS:
        D[a]['ewma22_all'] = np.full(D[a]['n'], .001)
        D[a].pop('y',None)
    T=inference_tensors(D)
    assert all('y' not in tt for tt in T.values())
    base=C18.precompute_baseline(D,parameters,T)
    warm=C18.DirectControls(D,parameters,base,'shared_GRU',11)
    model=R20.OperatorGRU(D,parameters,base,R20.synthetic_decoder(),
        {a:dict(mu=0.,rho=.3,residual_std=.2) for a in ASSETS},'joint',11,warm.state_dict())
    model.eval(); model.requires_grad_(False)
    with torch.no_grad():
        for a in ASSETS:
            q=np.array([D[a]['n']-1]); fields=model.filtered_fields(a,D[a],T[a],D[a]['n'],0)
            positive(model.forecast(a,T[a],q,fields)[0].numpy()[0])
    before=state_digest(model,omit_data=True); full=copy.deepcopy(D)
    for a in ASSETS:
        full[a]['n']+=1
        for key in ('F17','log_proxy','cell_b','cell_R','even'):
            full[a][key]=np.concatenate([full[a][key],full[a][key][-1:]])
        full[a]['observed']=np.concatenate([full[a]['observed'],[True]])
    fullT=inference_tensors(full); fullbase=C18.precompute_baseline(full,parameters,fullT)
    extension=RR.extend_data_buffers(model,full,fullbase,r20=True)
    assert extension['all_old_prefixes_exact'] and state_digest(model,omit_data=True)==before
    checks.update(last_observation_query_without_any_y=True, positive_future_points=True,
        only_data_buffers_extended=True, learned_theta_bank_decoder_AR_unchanged=True,
        real_market_data_or_weights_opened=False, real_future_labels_constructed=False,
        training_or_evaluation_or_scoring_called=False)
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--check-only',action='store_true')
    modes.add_argument('--development-replay',action='store_true')
    modes.add_argument('--record-prospective',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if args.record_prospective:
        result=record_prospective()
    elif args.development_replay:
        result=development_replay()
    else:
        result=dict(mode='PURECHECKS_ONLY', created_at_utc=iso(utc_now()), checks=purechecks(), script_sha256=digest(__file__))
        path=HERE/'prospective_forecast_once_purechecks.json'; reject_existing([path])
        with path.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
