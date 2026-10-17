"""Post-result fixed-parameter diagnostics; no fitting, new model scoring or selection.
This script is deliberately separate from the predeclared training/score pipeline.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
import core


def run(source: str, destination: str) -> None:
    r=Path(source); out=Path(destination)
    if out.exists():raise FileExistsError('Do not overwrite a prior diagnostic')
    core.configure()
    with np.load(r/'EVALUATION.npz',allow_pickle=False) as z:
        truth=z['truth'].copy(); years=z['years'].copy(); eligible=z['eligible'].copy()
        origins=z['origins'].copy()
    eligible_by_origin={int(o):bool(m) for o,m in zip(origins,eligible)}
    base=np.array([1.,2.,3.,6.,11.,20.]); geometry={}
    for mode in ('MESH','GLOBAL','TILT'):
        rows=[];differences=[];all_knots=[];trained_knots=[]
        for p in sorted((r/'fits').glob(f'{mode}_*.npz')):
            log=json.loads(p.with_suffix('.json').read_text())
            with np.load(p,allow_pickle=False) as z:
                take=np.array([eligible_by_origin[int(o)] for o in z['origins']])
                knots=z['detail__knots'][take].copy();a=z['detail__coefficients'][take].copy()
                sd=float(z['scale'])
            if not len(knots):continue
            with torch.no_grad():
                k=torch.as_tensor(knots);v=torch.as_tensor(a)
                current=core.interpolate(k,v).numpy()
                fixed=(v@core.initial_basis(6).T).numpy()
            delta=sd*(current-fixed)
            rows.append(dict(fit=p.stem,selected_epoch=log['selected_epoch'],origins=len(knots),
                mean_absolute_knot_displacement=float(np.abs(knots-base).mean()),
                maximum_knot_displacement=float(np.abs(knots-base).max()),
                within_fit_mean_knot_std=float(knots.std(axis=0).mean()),
                minimum_gap=float(np.diff(knots,axis=1).min()),
                fixed_parameter_output_change_rms=float(np.sqrt(np.mean(delta**2)))))
            differences.append(delta);all_knots.append(knots)
            if log['selected_epoch']>0:trained_knots.append(knots)
        d=np.concatenate(differences);ks=np.concatenate(all_knots)
        geometry[mode]=dict(fits=len(rows),epoch_zero=sum(v['selected_epoch']==0 for v in rows),
            origins_across_seeds=len(ks),mean_absolute_displacement=float(np.abs(ks-base).mean()),
            maximum_displacement=float(np.abs(ks-base).max()),
            mean_within_fit_knot_std=float(np.mean([v['within_fit_mean_knot_std'] for v in rows])),
            minimum_gap=float(np.diff(ks,axis=1).min()),
            output_change_rms_all20=float(np.sqrt(np.mean(d**2))),
            output_change_rms_by_evaluated_horizon={str(h):float(np.sqrt(np.mean(d[:,h-1]**2))) for h in (1,5,20)},
            output_change_max=float(np.abs(d).max()),
            trained_only_mean_absolute_displacement=float(np.abs(np.concatenate(trained_knots)-base).mean()) if trained_knots else None,
            fits_detail=rows)
    decomposition={};max_error=0.
    for mode in core.MODES:
        with np.load(r/'predictions'/f'{mode}.npz',allow_pickle=False) as z:p=z['prediction'].copy()
        average=p.mean(axis=0);cells=[]
        for year in ('2020','2021'):
            ix=eligible & (years==year)
            for h in (1,5,20):
                den=np.mean(truth[ix,h-1]**2)
                individual=np.mean((p[:,ix,h-1]-truth[None,ix,h-1])**2)/den
                ensemble=np.mean((average[ix,h-1]-truth[ix,h-1])**2)/den
                spread=np.mean((p[:,ix,h-1]-average[None,ix,h-1])**2)/den
                error=abs(individual-ensemble-spread);max_error=max(max_error,error)
                cells.append(dict(year=year,horizon=h,mean_seed_relative_MSE=float(individual),
                    ensemble_relative_MSE=float(ensemble),realized_prediction_dispersion=float(spread),identity_error=float(error)))
        decomposition[mode]=dict(mean_seed_relative_MSE=float(np.mean([c['mean_seed_relative_MSE'] for c in cells])),
            ensemble_relative_MSE=float(np.mean([c['ensemble_relative_MSE'] for c in cells])),
            realized_prediction_dispersion=float(np.mean([c['realized_prediction_dispersion'] for c in cells])),cells=cells)
    if max_error>1e-12:raise ValueError('Squared-loss ensemble identity failed')
    result=dict(scope='POSTHOC: saved parameters only; no training, alternate score, hyperparameter choice or new time period',
        geometry=geometry,ensemble_decomposition=decomposition,maximum_identity_error=max_error,
        limitations=['Moving nodes are not identified market event times.',
        'Zero selected epochs are not proof of absence of signal or of optimizer failure.',
        'Replacing knots only measures local action of a fixed trained model, not a retrained ablation.',
        'Realized seed dispersion is an exact algebraic decomposition, not an identified bias/variance decomposition of the unknown population risk.',
        'Diagnostics cannot authorize increasing knot flexibility or retuning on these same scores.'])
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'geometry':{k:{a:v for a,v in value.items() if a!='fits_detail'} for k,value in geometry.items()},
        'dispersion':{k:v['realized_prediction_dispersion'] for k,v in decomposition.items()},'max_error':max_error},indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--source',required=True);a.add_argument('--destination',required=True)
    args=a.parse_args();run(args.source,args.destination)
