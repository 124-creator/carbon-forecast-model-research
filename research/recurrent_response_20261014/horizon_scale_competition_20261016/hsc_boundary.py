"""Six representative real fresh suffix refits; no saved checkpoint loading."""
import json
import time
from pathlib import Path
import numpy as np
from hsc_core import MODES
from hsc_fit import fit, predict
from fair_data import load_prefix


def compare_parameters(original, replay):
    if set(original) != set(replay):
        raise ValueError('Parameter keys differ')
    for name, value in original.items():
        if not np.array_equal(value, replay[name]):
            raise ValueError('Parameter changed after suffix perturbation: ' + name)


def first_refresh_paths(directory, selection):
    root = Path(directory) / 'local_only' / 'market_fits'
    paths = {m: root / f'{m}_{selection[m]["config_id"]}_s17_t459.npz' for m in MODES}
    if any(not p.exists() or not p.with_suffix('.json').exists() for p in paths.values()):
        raise ValueError('All six first-refresh fit pairs are required')
    return paths


def audit(directory):
    directory = Path(directory)
    if (directory / 'BOUNDARY.json').exists():
        raise ValueError('Preserve existing boundary audit')
    price, _, _ = load_prefix()
    altered = price.copy()
    altered[459:] += 1000 + np.arange(len(altered) - 459) * 3
    before = json.loads((directory / 'PRE_RUN.json').read_text(encoding='utf-8'))
    selection = json.loads((directory / 'SELECTION.json').read_text(encoding='utf-8'))
    paths = first_refresh_paths(directory, selection)
    grid = {c['id']: c for c in before['configurations']}
    deadline = time.monotonic() + 300
    records = []
    for mode in MODES:
        cid = selection[mode]['config_id']
        path = paths[mode]
        with np.load(path, allow_pickle=False) as z:
            old = {key[7:]: z[key].copy() for key in z.files if key.startswith('param__')}
            saved_forecast = z['prediction'].copy()
            ids = z['origins'].copy()
            old_scale = float(z['scale'])
        model, scale, log = fit(altered, 459, mode, grid[cid], 17, deadline=deadline)
        fresh = {key: value.detach().numpy() for key, value in model.state_dict().items()}
        compare_parameters(old, fresh)
        if old_scale != scale:
            raise ValueError('Training scale changed')
        a, wa = predict(model, scale, price, np.array([458]))
        b, wb = predict(model, scale, altered, np.array([458]))
        if not np.array_equal(a, b) or not np.array_equal(wa, wb):
            raise ValueError('Prediction before changed suffix differs')
        original_forecast, _ = predict(model, scale, price, ids)
        if not np.array_equal(original_forecast, saved_forecast):
            raise ValueError('Fresh refit does not reproduce original saved forecasts')
        records.append(dict(family=mode, parameters_bitwise_equal=True, prefix_forecast_bitwise_equal=True,
                            original_saved_forecast_reproduced=True, epochs=log['epochs'],
                            fit_seconds=log['seconds'], selected_epoch=log['selected_epoch']))
    return dict(passed=True, fresh_refits=len(records), cut=459, seed=17, records=records,
                full_run_complete=(directory / 'RUN_COMPLETE.json').exists(), candidate_qualified=False,
                scope='Six representative first-refresh refits; not all192 fits or independent implementation')


if __name__ == '__main__':
    result = audit(Path(__file__).resolve().parent)
    with Path(__file__).with_name('BOUNDARY.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2))
