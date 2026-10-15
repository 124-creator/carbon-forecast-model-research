"""Eight prespecified full real-price refits; separate from independent scoring."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from fit_recurrent import fit, predict
from fair_data import load_prefix
from protocol import plan, configurations

def audit_pair(price, end, family, config, seed, max_epochs, deadline=None, archive=None, trace_path=None):
    started = time.monotonic()
    first, scale, original = fit(price, end, family, config, seed, max_epochs=max_epochs, deadline=deadline)
    if trace_path is not None:
        with (Path(trace_path) / (family + '_original.json')).open('w', encoding='utf-8', newline='\n') as stream:
            json.dump(original, stream, indent=2, allow_nan=False)
            stream.write('\n')
    changed = np.array(price, copy=True)
    changed[end:] += 100 + 7 * np.arange(len(changed) - end)
    second, scale2, altered = fit(changed, end, family, config, seed, max_epochs=max_epochs, deadline=deadline)
    if trace_path is not None:
        with (Path(trace_path) / (family + '_altered.json')).open('w', encoding='utf-8', newline='\n') as stream:
            json.dump(altered, stream, indent=2, allow_nan=False)
            stream.write('\n')
    a = {name: value.detach().numpy() for name, value in first.state_dict().items()}
    b = {name: value.detach().numpy() for name, value in second.state_dict().items()}
    parameter_difference = max(float(np.max(np.abs(a[name] - b[name]))) for name in a)
    prefix_origins = np.array([end - 2, end - 1])
    prefix_difference = float(np.max(np.abs(predict(first, scale, price, prefix_origins)
                                           - predict(second, scale2, changed, prefix_origins))))
    reference_parameter_difference = None
    reference_prediction_difference = None
    differences = [parameter_difference, prefix_difference, abs(scale - scale2)]
    if archive is not None:
        with np.load(archive, allow_pickle=False) as saved:
            reference_parameter_difference = max(float(np.max(np.abs(a[name] - saved['param__' + name]))) for name in a)
            reference_prediction_difference = float(np.max(np.abs(
                predict(first, scale, price, saved['origins']) - saved['prediction'])))
            differences.extend((reference_parameter_difference, reference_prediction_difference))
    if not np.isfinite(differences).all() or max(differences) > 1e-11:
        raise ValueError('Full-fit future isolation or saved-fit reproduction failed')
    digests = [hashlib.sha256(b''.join(name.encode() + b'\0' + state[name].tobytes()
                                      for name in sorted(state))).hexdigest() for state in (a, b)]
    return dict(family=family, config_id=config['id'], end=end, seed=seed, actual_fits=2,
                original_epochs=original['epochs'], altered_epochs=altered['epochs'],
                original_selected_epoch=original['selected_epoch'], altered_selected_epoch=altered['selected_epoch'],
                parameter_max_difference=parameter_difference, unchanged_prefix_prediction_max_difference=prefix_difference,
                source_parameter_max_difference=reference_parameter_difference,
                source_prediction_max_difference=reference_prediction_difference,
                parameter_sha256=digests, seconds=time.monotonic() - started,
                independent_implementation=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    here = args.directory
    target = here / 'BOUNDARY_AUDIT.json'
    if target.exists() or (here / 'BOUNDARY_INCOMPLETE.json').exists():
        raise FileExistsError('Preserve first audit artifacts; do not retry failed audits')
    selection = json.loads((here / 'SELECTION.json').read_text(encoding='utf-8'))
    before = json.loads((here / 'PRE_RUN.json').read_text(encoding='utf-8'))
    for name, digest in before['source_sha256'].items():
        if hashlib.sha256((here / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen training source changed: ' + name)
    grid = {row['id']: row for row in configurations()}
    archives = {family: here / 'local_only/market_fits' / f'{family}_{selection[family]["config_id"]}_s17_b00.npz'
                for family in plan()['families']}
    if not all(path.exists() for path in archives.values()):
        raise ValueError('All selected first-cut source fits must exist before auditing')
    traces = here / 'local_only/boundary_fits'
    traces.mkdir()
    price, _, _ = load_prefix()
    started = time.monotonic()
    deadline = started + plan()['audit_budget_seconds']
    receipts = []
    try:
        for family in plan()['families']:
            cid = selection[family]['config_id']
            receipt = audit_pair(price, 459, family, grid[cid], 17, plan()['max_epochs'],
                                 deadline=deadline, archive=archives[family], trace_path=traces)
            receipts.append(receipt)
            print(f'BOUNDARY {family} passed', flush=True)
    except Exception as error:
        with (here / 'BOUNDARY_INCOMPLETE.json').open('w', encoding='utf-8', newline='\n') as stream:
            json.dump(dict(complete=False, reason=str(error), completed_pairs=receipts,
                           actual_saved_fits=len(list(traces.glob('*.json')))), stream, indent=2)
            stream.write('\n')
        raise
    with target.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(dict(complete=True, actual_fits=len(list(traces.glob('*.json'))), pairs=receipts,
                       seconds=time.monotonic() - started,
                       scope='Full same-source original/future-perturbed refits, not independently implemented training'),
                  stream, indent=2, allow_nan=False)
        stream.write('\n')
