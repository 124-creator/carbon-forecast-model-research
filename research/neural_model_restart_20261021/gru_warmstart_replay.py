"""Export selected C18 GRU weights by exact replay, without new epoch selection.

Default is pure synthetic checks. --replay-selected reproduces only archived
seed11/22/33 epochs26/26/28 on the original pre2026 development preparation.
Nothing is saved until every seed passes its archived prediction identity gate.
All outputs have new protected names. Checkpoints contain own tensors and plain
metadata only; load with torch.load(..., weights_only=True).
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import platform
import time
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
import torch
import round18_direct_controls as C18

S18, S17, B = C18.S18, C18.S17, C18.B
ASSETS = C18.ASSETS
HERE = Path(__file__).resolve().parent
ARCHIVE_RESULTS = HERE / 'round18_direct_controls_development_results.json'
ARCHIVE_PREDICTIONS = HERE / 'round18_direct_controls_development_validation_predictions.npz'
SELECTED = {11: 26, 22: 26, 33: 28}
TAG = 'gru_warmstart_replay_c18'
RTOL, ATOL = 1e-11, 1e-14


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output_paths():
    checkpoints = {seed: HERE / f'{TAG}_seed{seed}_epoch{epoch}.pt' for seed, epoch in SELECTED.items()}
    return checkpoints, HERE / f'{TAG}_manifest.json'


def assert_new_outputs(paths):
    for path in paths:
        if path.exists():
            raise FileExistsError(f'Protected existing output: {path}')


def payload(model, metadata):
    return dict(state_dict={name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                metadata=copy.deepcopy(metadata))


def replay(datasets, parameters, baseline, seed, epoch):
    """Exactly epoch original Adam updates; no early stop, new selection or extension."""
    T = S17.make_tensors(datasets)
    model = C18.DirectControls(datasets, parameters, baseline, 'shared_GRU', seed)
    immutable = {k: v.clone() for k, v in model.named_buffers()}
    _, joint, scores, _, centers = C18.evaluate(model, datasets, T)
    history = [dict(epoch=0, validation_joint_QLIKE=joint, validation_asset_QLIKE=scores, bankcenter=centers)]
    optimizer = torch.optim.Adam(model.parameters(), lr=C18.LR)
    for step in range(1, epoch + 1):
        info = C18.train_epoch(model, datasets, T, optimizer)
        _, joint, scores, _, centers = C18.evaluate(model, datasets, T)
        history.append(dict(epoch=step, validation_joint_QLIKE=joint,
                            validation_asset_QLIKE=scores, bankcenter=centers, **info))
    pred, joint, scores, aux, centers = C18.evaluate(model, datasets, T)
    for name, value in model.named_buffers():
        assert torch.equal(value, immutable[name]), f'Frozen buffer changed: {name}'
    return model, pred, dict(seed=seed, epoch=epoch, joint_QLIKE=joint, asset_QLIKE=scores,
                            bankcenter=centers, history=history)


def prediction_gate(D, saved, pred, asset, seed):
    q = D['origin'][D['va']]
    # Verify dates before touching any archive target or prediction arrays.
    dates, end = saved[f'{asset}_date'], saved[f'{asset}_label_end']
    assert (dates < B.RESERVED).all() and (end < B.RESERVED).all()
    np.testing.assert_array_equal(dates, D['date'][q])
    np.testing.assert_array_equal(end, D['date'][q + 22])
    np.testing.assert_array_equal(saved[f'{asset}_origin'], q)
    np.testing.assert_array_equal(saved['horizons'], np.array(B.HORIZONS))
    np.testing.assert_array_equal(saved[f'{asset}_target'], D['y'][D['va']])
    expected = saved[f'{asset}_shared_GRU_seed{seed}']
    np.testing.assert_allclose(pred, expected, rtol=RTOL, atol=ATOL)
    diff = np.abs(pred - expected)
    return dict(n=len(q), max_absolute_error=float(diff.max()),
        max_relative_error=float((diff / np.maximum(np.abs(expected), np.finfo(float).tiny)).max()),
        bitwise_equal=bool(np.array_equal(pred, expected)), rtol=RTOL, atol=ATOL,
        date_origin_target_horizons_exact=True)


def _synthetic_checks():
    D, _ = S17.synthetic_pair(); parameters = S18._synthetic_parameters(); T = S17.make_tensors(D)
    baseline = C18.precompute_baseline(D, parameters, T)
    model, actual, info = replay(D, parameters, baseline, 11, 2)
    reference = C18.DirectControls(D, parameters, baseline, 'shared_GRU', 11)
    optimizer = torch.optim.Adam(reference.parameters(), lr=C18.LR)
    for _ in range(2):
        C18.train_epoch(reference, D, T, optimizer)
    expected, _, _, _, _ = C18.evaluate(reference, D, T)
    saved = dict(horizons=np.array(B.HORIZONS))
    for a in ASSETS:
        np.testing.assert_array_equal(actual[a], expected[a])
        q = D[a]['origin'][D[a]['va']]
        saved.update({f'{a}_date': D[a]['date'][q], f'{a}_label_end': D[a]['date'][q + 22],
            f'{a}_origin': q, f'{a}_target': D[a]['y'][D[a]['va']], f'{a}_shared_GRU_seed11': expected[a]})
        prediction_gate(D[a], saved, actual[a], a, 11)
        try:
            prediction_gate(D[a], saved, actual[a] * 1.001, a, 11)
        except AssertionError:
            pass
        else:
            raise AssertionError('Prediction gate accepted a changed model')
    checkpoint = payload(model, dict(seed=11, epoch=2, source_hashes={}, synthetic=True))
    buffer = io.BytesIO(); torch.save(checkpoint, buffer); buffer.seek(0)
    restored_payload = torch.load(buffer, weights_only=True, map_location='cpu')
    restored = C18.DirectControls(D, parameters, baseline, 'shared_GRU', 11)
    restored.load_state_dict(restored_payload['state_dict'], strict=True)
    recovered, _, _, _, _ = C18.evaluate(restored, D, T)
    for a in ASSETS:
        np.testing.assert_array_equal(actual[a], recovered[a])
    for k, v in model.state_dict().items():
        assert torch.equal(v, restored_payload['state_dict'][k])
    assert len(info['history']) == 3 and info['epoch'] == 2
    # The exporter rejects an existing source path as an output, without writing it.
    try:
        assert_new_outputs([Path(__file__)])
    except FileExistsError:
        pass
    else:
        raise AssertionError('Overwrite protection did not reject an existing path')
    return dict(exact_fixed_epoch_replay=True, archive_gate_rejects_changed_prediction=True,
        weights_only_in_memory_roundtrip_exact=True, checkpoint_includes_all_model_buffers=True,
        immutable_frozen_buffers=True, protected_existing_outputs=True, new_epoch_selection=False,
        no_optimizer_pickle=True, synthetic_training_steps=2)


def synthetic_checks():
    with ExitStack() as stack:
        for name in ('pandas.read_csv', 'numpy.load', 'pathlib.Path.read_text',
                     'round17_shared_precision.prepare_pair', 'round18_frozen_moments.load_stage16_parameters'):
            stack.enter_context(patch(name, side_effect=AssertionError('real reader forbidden in purecheck')))
        checks = _synthetic_checks()
    checks.update(real_input_readers_guarded=True, market_files_opened=False)
    return checks


def source_hashes():
    hashes = C18.source_hashes(); hashes['direct_controls'] = hashes.pop('runner')
    sources = dict(runner=Path(__file__), archive_results=ARCHIVE_RESULTS, archive_predictions=ARCHIVE_PREDICTIONS,
        reference_models=B.REFERENCE, reference_results=S17.PROXY.REFERENCE_RESULTS)
    for a in ASSETS:
        sources[f'csv_{a}'] = B.SOURCES[a]
        for key, path in S18.STAGE16[a].items():
            sources[f'stage16_{a}_{key}'] = path
    hashes.update({name: dict(path=str(Path(p).resolve()), sha256=sha256(p)) for name, p in sources.items()})
    return hashes


def replay_selected():
    paths, manifest_path = output_paths(); assert_new_outputs([*paths.values(), manifest_path])
    started = time.time(); hashes = source_hashes()
    archive = json.loads(ARCHIVE_RESULTS.read_text(encoding='utf-8'))
    current_hashes = C18.source_hashes()
    for name, value in archive['hashes'].items():
        assert current_hashes[name]['sha256'] == value['sha256'], f'Archived source changed: {name}'
    assert archive['config'] == dict(modes=list(C18.MODES), seeds=list(C18.SEEDS), lr=C18.LR,
        regularization=C18.LAMBDA, hidden=8, chunk=C18.CHUNK, max_epochs=C18.MAX_EPOCHS,
        patience=C18.PATIENCE, min_epochs=C18.MIN_EPOCHS)
    fits = {fit['seed']: fit for fit in archive['fits']['shared_GRU']}
    assert {seed: fit['best_joint_epoch'] for seed, fit in fits.items()} == SELECTED
    parameters, provenance = S18.load_stage16_parameters(); datasets, audit = S17.prepare_pair()
    assert parameters == archive['frozen_parameters']
    for a in ASSETS:
        assert abs(provenance[a]['reference_c512'] - audit['proxy_c512']) < 1e-14
        assert (datasets[a]['date'] < B.RESERVED).all()
    frozen_pred, baseline_gate = S18.verify_real_baseline(datasets, parameters)
    T = S17.make_tensors(datasets); baseline = C18.precompute_baseline(datasets, parameters, T)
    ready, results = {}, {}
    with np.load(ARCHIVE_PREDICTIONS, allow_pickle=False) as saved:
        for a in ASSETS:
            D = datasets[a]; bank = D['bank_index']
            # Archive date gate precedes any saved predictions/targets access.
            assert (saved[f'{a}_date'] < B.RESERVED).all()
            assert (saved[f'{a}_label_end'] < B.RESERVED).all()
            np.testing.assert_array_equal(saved[f'{a}_bank_date'], D['date'][bank])
            np.testing.assert_array_equal(saved[f'{a}_bank_F17'], D['F17'][bank])
            np.testing.assert_allclose(saved[f'{a}_frozen'], frozen_pred[a], rtol=RTOL, atol=ATOL)
            for key, value in baseline[a].items():
                np.testing.assert_array_equal(saved[f'{a}_{key}'], value.numpy())
        for seed, epoch in SELECTED.items():
            print('REPLAY_BEGIN', seed, 'fixed_epoch', epoch, flush=True)
            model, pred, info = replay(datasets, parameters, baseline, seed, epoch)
            errors = {a: prediction_gate(datasets[a], saved, pred[a], a, seed) for a in ASSETS}
            # No epoch selection: every replayed diagnostic must match its archived row.
            reference_history = fits[seed]['history'][:epoch + 1]
            assert len(reference_history) == len(info['history'])
            history_errors = []
            for actual, reference in zip(info['history'], reference_history):
                assert actual['epoch'] == reference['epoch']
                error = abs(actual['validation_joint_QLIKE'] - reference['validation_joint_QLIKE'])
                assert error <= 1e-11
                history_errors.append(error)
            metadata = dict(format_version=1, model_class='round18_direct_controls.DirectControls',
                mode='shared_GRU', seed=seed, epoch=epoch, source_hashes=hashes,
                selected_by='existing C18 archive; not selected again',
                training='exact C18.train_epoch; original Adam .01, clip2, detach128; fixed archived epoch',
                data_scope='original pre2026 development preparation; no holdout features or targets',
                state_dict_content='GRU/head, frozen theta, bank indices and m0/P0/K0 buffers',
                parameter_count=sum(p.numel() for p in model.parameters()), prediction_gate=errors,
                max_joint_history_error=max(history_errors),
                normalization=audit['normalization'], frozen_parameters=parameters)
            ready[seed] = payload(model, metadata)
            results[seed] = dict(epoch=epoch, checkpoint=str(paths[seed]), errors=errors,
                max_joint_history_error=max(history_errors), replay=info)
            print('REPLAY_MATCH', seed, json.dumps(errors), flush=True)
    # All three seeds must pass before the first artifact is written.
    assert source_hashes() == hashes, 'Replay inputs changed during execution'
    assert_new_outputs([*paths.values(), manifest_path])
    for seed, path in paths.items():
        with path.open('xb') as handle:
            torch.save(ready[seed], handle)
        # Verify own saved tensors using the intended downstream safe loader.
        loaded = torch.load(path, weights_only=True, map_location='cpu')
        for name, value in ready[seed]['state_dict'].items():
            assert torch.equal(value, loaded['state_dict'][name])
        results[seed]['checkpoint_sha256'] = sha256(path)
    manifest = dict(status='ALL ARCHIVED SEEDS MATCH; exact selected-epoch replay only',
        source_hashes=hashes, selected_epochs=SELECTED, baseline_gate=baseline_gate,
        source_archive=str(ARCHIVE_RESULTS), normalizer=audit['normalization'], stage16_checkpoints=provenance,
        environment=dict(python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__)),
        results=results, weights_only_load_verified=True, all_seeds_passed_before_save=True,
        no_new_epoch_selection=True, no_continuation=True, holdout_features_targets_used=False,
        elapsed_seconds=time.time() - started)
    with manifest_path.open('x', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print('REPLAY_EXPORTED', str(manifest_path), 'sha256', sha256(manifest_path), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    gate = ap.add_mutually_exclusive_group()
    gate.add_argument('--check-only', action='store_true'); gate.add_argument('--replay-selected', action='store_true')
    args = ap.parse_args(); torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if args.replay_selected:
        replay_selected()
    else:
        print(json.dumps(dict(checks=synthetic_checks(), source_sha256=sha256(__file__)), indent=2), flush=True)


if __name__ == '__main__':
    main()
