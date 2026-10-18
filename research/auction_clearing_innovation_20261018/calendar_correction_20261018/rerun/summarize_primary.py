"""Aggregate frozen forecasts, without refitting or selecting a model on test."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def crps_and_intervals(atoms, weights, target):
    order = np.argsort(atoms, axis=1)
    a = np.take_along_axis(atoms, order, axis=1)
    w = np.take_along_axis(weights, order, axis=1)
    cw = np.cumsum(w, axis=1)
    past_w = cw-w
    past_aw = np.cumsum(a*w, axis=1)-a*w
    crps = np.sum(weights*np.abs(atoms-target[:, None]), axis=1)-np.sum(w*(a*past_w-past_aw), axis=1)
    lo_index = (cw < .1).sum(1).clip(max=a.shape[1]-1)
    hi_index = (cw < .9).sum(1).clip(max=a.shape[1]-1)
    row = np.arange(len(a))
    return crps, a[row, lo_index], a[row, hi_index]


def main():
    root = Path(__file__).resolve().parent
    cfg = json.loads((root/'experiment_config.json').read_text())
    run = json.loads((root/'results/neural/RUN_SUMMARY.json').read_text())
    assert run['status'] == 'COMPLETE' and run['refits'] == 72
    meta = pd.read_csv(root/'local_only/example_metadata.csv')
    data = np.load(root/'local_only/examples_development.npz')
    outputs, cells = {}, []
    out_dir = root/'local_only/ensembles'
    out_dir.mkdir(exist_ok=True)
    for method in cfg['classical_models']+cfg['neural_models']:
        for year in cfg['evaluation_issue_years']:
            if method in cfg['classical_models']:
                files = [root/f'local_only/classical/{method}_{year}.npz']
            else:
                files = [root/f'local_only/neural/{method}_{year}_s{s}.npz' for s in cfg['refit_seeds']]
            parts = [np.load(f, allow_pickle=False) for f in files]
            idx = parts[0]['example_indices'].astype(int)
            assert all(np.array_equal(idx, p['example_indices']) for p in parts)
            mean = np.mean([p['mean_z'] for p in parts], axis=0)
            atoms = np.concatenate([p['atoms_z'] for p in parts], axis=1)
            weights = np.concatenate([p['weights']/len(parts) for p in parts], axis=1)
            assert np.allclose(weights.sum(1), 1, rtol=0, atol=2e-6)
            scale, anchor, price = data['scale'][idx], data['anchor'][idx], data['price'][idx]
            predicted_price = anchor+scale*mean
            error = predicted_price-price
            good = data['completed'][idx].astype(bool)
            # Price-scale transformation of exact CRPS of standardized atoms.
            crps_z, lo, hi = crps_and_intervals(atoms, weights, data['y'][idx])
            crps = scale*crps_z
            np.savez_compressed(out_dir/f'{method}_{year}.npz', example_indices=idx,
                mean_z=mean, atoms_z=atoms, weights=weights, predicted_price=predicted_price,
                squared_error=error**2, absolute_error=np.abs(error), price_crps=crps)
            for horizon in cfg['horizons']:
                take = good & (meta.horizon.to_numpy()[idx] == horizon)
                rw_mse = float(np.mean((anchor[take]-price[take])**2))
                target_z = data['y'][idx][take]
                row = {'method': method, 'year': year, 'horizon': horizon,
                       'n': int(take.sum()), 'mse': float(np.mean(error[take]**2)),
                       'rw_mse': rw_mse, 'mse_ratio': float(np.mean(error[take]**2)/rw_mse),
                       'mae': float(np.mean(np.abs(error[take]))), 'crps': float(np.mean(crps[take])),
                       'coverage80': float(np.mean((target_z >= lo[take]) & (target_z <= hi[take]))),
                       'width80': float(np.mean(scale[take]*(hi[take]-lo[take]))),
                       'outside_atom_support': float(np.mean((target_z < atoms[take].min(1)) | (target_z > atoms[take].max(1))))}
                cells.append(row)
    frame = pd.DataFrame(cells)
    rw_crps = frame[frame.method == 'RW'].set_index(['year', 'horizon']).crps
    frame['crps_ratio'] = [r.crps/rw_crps.loc[(r.year,r.horizon)] for r in frame.itertuples()]
    aggregate = []
    for method, group in frame.groupby('method', sort=False):
        aggregate.append({'method': method, 'MSE_RW_ratio': float(group.mse_ratio.mean()),
                          'CRPS_RW_ratio': float(group.crps_ratio.mean()),
                          'mean_coverage80': float(group.coverage80.mean()),
                          'RW_winning_cells': int((group.mse_ratio < 1).sum()),
                          'year_MSE_RW_ratios': group.groupby('year').mse_ratio.mean().to_dict()})
    aggregate.sort(key=lambda r:r['MSE_RW_ratio'])
    frame.to_csv(root/'results/PRIMARY_CELLS.csv', index=False)
    result = {'aggregation':cfg['primary_metric'], 'point_estimator':cfg['primary_estimator'],
              'forecast_pairs_per_model':int(frame[frame.method == 'RW'].n.sum()),
              'aggregate':aggregate, 'neural_training':{k:run[k] for k in ['fits','refits','updates','fit_seconds','failures']},
              'status':'PRIMARY_SCORES_COMPUTED; independent bootstrap gate pending'}
    (root/'results/PRIMARY_SUMMARY.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
