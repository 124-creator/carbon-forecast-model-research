"""Secondary, fully paired comparison of frozen candidate and benchmark runs.

This is a contextual comparison: different input representations, mean freedom,
and horizon parameterizations prevent an architectural causal attribution.
No training or future data access occurs here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
SEEDS=(17,41,89)
HORIZONS=(5,20)
PANELS=tuple(sorted(f'{m}_{y}' for m in ('EUA_LONG','GUANGZHOU','HUBEI','SHANGHAI')
                    for y in (2021,2022,2023) if (m,y)!=('SHANGHAI',2021)))
CP_ARMS=('conditional_joint005','constant_joint005','scale_capacity_joint005',
         'conditional_endpoint','independent_horizon_endpoint')
GAS_FAMILIES=('gas_gru_gmm','ewma_gru_gmm','global_gru_gmm',
              'gas_no_neural','ewma_no_neural','global_no_neural',
              'gas_empirical101','ewma_empirical101','global_empirical101','daily_ewma_normal')
FAMILIES=tuple('CP_'+x for x in CP_ARMS)+GAS_FAMILIES
METRICS=('normalized_CRPS','price_CRPS','normalized_NLL','price_NLL',
         'mean_price_squared_error','rw_mean_price_squared_error','coverage90','normalized_width90')
REFERENCE=('gas_gru_gmm','ewma_gru_gmm','global_gru_gmm',
           'gas_empirical101','ewma_empirical101','global_empirical101','daily_ewma_normal')
PAIRS=tuple(('CP_conditional_joint005',x) for x in REFERENCE)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def seed_mean(values):
    a=np.asarray(values)
    return a[0]+np.mean(a-a[0],axis=0)


def load_complete(cp_private,gas_private):
    cp_public=STUDY/'results/market/persistence'
    assert len(list((cp_public/'fits').glob('*.json')))==312
    assert len(list((cp_public/'references').glob('*.json')))==13
    assert (cp_public/'SUMMARY.json').exists()
    assert (STUDY/'results/gas_benchmark/SUMMARY.json').exists()
    losses,dates,rows={}, {}, []
    checked=0
    for panel in PANELS:
        scores_by_h=[]
        common_ids=common_dates=None
        for h in HORIZONS:
            col={5:1,20:2}[h]
            family_scores=[]
            for family in FAMILIES:
                metrics={key:[] for key in METRICS}
                seeds=SEEDS if family.startswith('CP_') or family.endswith('_gru_gmm') else (0,)
                for seed in seeds:
                    if family.startswith('CP_'):
                        name=f'{panel}__GRU16__{family[3:]}__s{seed}'
                        path=cp_private/'forecasts'/f'{name}.npz'
                        receipt=json.loads((cp_public/'fits'/f'{name}.json').read_text())
                        assert sha(path)==receipt['private_artifacts'][f'forecasts/{name}.npz']
                        prefix=f'test_score_h{h}_'
                    else:
                        path=gas_private/'predictions'/f'{panel}_h{h}'/family/f'seed_{seed}.npz'
                        receipt=json.loads(path.with_suffix('.json').read_text())
                        assert sha(path)==receipt['prediction_sha256']
                        prefix='score_'
                    with np.load(path,allow_pickle=False) as z:
                        origin=z['test_indices'] if family.startswith('CP_') else z['origin']
                        origin_dates=z['test_dates'] if family.startswith('CP_') else z['date']
                        if common_ids is None:
                            common_ids,common_dates=origin.copy(),origin_dates.copy()
                        np.testing.assert_array_equal(origin,common_ids)
                        np.testing.assert_array_equal(origin_dates,common_dates)
                        for metric in METRICS:
                            key=prefix+metric
                            if key in z:
                                metrics[metric].append(z[key].copy())
                    checked+=1
                means={metric:seed_mean(values) for metric,values in metrics.items() if values}
                assert len(means['normalized_CRPS'])==len(common_ids)
                family_scores.append(means['normalized_CRPS'])
                row={'panel':panel,'horizon':h,'family':family,'seed_count':len(seeds),'n':len(common_ids),
                     **{key:float(a.mean()) for key,a in means.items()}}
                rows.append(row)
            scores_by_h.append(np.stack(family_scores,axis=-1))
        losses[panel]=np.stack(scores_by_h,axis=1)
        dates[panel]=common_dates
    return losses,dates,pd.DataFrame(rows),checked


def ratios(mean):
    return np.stack([mean[...,FAMILIES.index(a)]/mean[...,FAMILIES.index(b)]-1 for a,b in PAIRS],axis=-1)


def panel_resamples(losses):
    rng=np.random.default_rng(20261019)
    output=np.zeros((1999,len(PAIRS)))
    for panel in PANELS:
        loss=losses[panel];n=len(loss)
        starts=rng.integers(0,n,size=(1999,math.ceil(n/40)))
        for a in range(0,1999,100):
            ix=((starts[a:a+100,:,None]+np.arange(40))%n).reshape(len(starts[a:a+100]),-1)[:,:n]
            output[a:a+100]+=ratios(loss[ix].mean(1)).sum(1)/22
    return output


def synchronized_resamples(losses,dates):
    rng=np.random.default_rng(20261019)
    output=np.zeros((1999,len(PAIRS)))
    for year in (2021,2022,2023):
        panels=[p for p in PANELS if p.endswith(str(year))]
        union=np.unique(np.concatenate([dates[p] for p in panels]))
        n=len(union)
        starts=rng.integers(0,n,size=(1999,math.ceil(n/40)))
        positions={p:np.searchsorted(union,dates[p]) for p in panels}
        for a in range(0,1999,100):
            selected=starts[a:a+100]
            ix=((selected[:,:,None]+np.arange(40))%n).reshape(len(selected),-1)[:,:n]
            counts=np.zeros((len(selected),n),dtype=float)
            for row,idx in enumerate(ix):
                counts[row]=np.bincount(idx,minlength=n)
            for panel in panels:
                weights=counts[:,positions[panel]]
                denom=weights.sum(1)
                if np.any(denom==0):
                    raise ValueError('A synchronous draw has no observations for a panel; no hidden redraw')
                mean=np.einsum('bn,nhf->bhf',weights,losses[panel])/denom[:,None,None]
                output[a:a+len(selected)]+=ratios(mean).sum(1)/22
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--persistence-private',type=Path,default=Path('/tmp/carbon-persistence-market-runs'))
    p.add_argument('--gas-private',type=Path,default=Path('/tmp/carbon-gas-normalization-run'))
    p.add_argument('--public-dir',type=Path,default=STUDY/'results/cross_benchmark')
    args=p.parse_args()
    losses,dates,frame,checked=load_complete(args.persistence_private,args.gas_private)
    args.public_dir.mkdir(parents=True,exist_ok=True)
    frame.to_csv(args.public_dir/'CELL_METRICS.csv',index=False)
    point=np.concatenate([ratios(losses[p].mean(0)) for p in PANELS],axis=0)
    boot=panel_resamples(losses)
    sync=synchronized_resamples(losses,dates)
    comparisons=[]
    for j,(candidate,reference) in enumerate(PAIRS):
        ci=np.quantile(boot[:,j],[.025,.975]);sci=np.quantile(sync[:,j],[.025,.975])
        comparisons.append({'candidate':candidate,'reference':reference,'cells':22,
             'mean_cell_relative_CRPS':float(point[:,j].mean()),'strictly_better_cells':int(np.sum(point[:,j]<0)),
             'panel_ci95_lower':float(ci[0]),'panel_ci95_upper':float(ci[1]),
             'synchronized_ci95_lower':float(sci[0]),'synchronized_ci95_upper':float(sci[1])})
    pd.DataFrame(comparisons).to_csv(args.public_dir/'PAIRED_COMPARISONS.csv',index=False)
    common=frame[frame.family=='daily_ewma_normal'].set_index(['panel','horizon'])
    families=[]
    for family in FAMILIES:
        f=frame[frame.family==family].set_index(['panel','horizon'])
        row={'family':family,'cells':22,
             'normalized_CRPS_relative_daily_normal':float((f.normalized_CRPS/common.normalized_CRPS-1).mean()),
             'price_CRPS_relative_daily_normal':float((f.price_CRPS/common.price_CRPS-1).mean()),
             'mean_price_MSE_relative_RW':float((f.mean_price_squared_error/f.rw_mean_price_squared_error).mean()),
             'coverage90_equal_cell':float(f.coverage90.mean()),
             'normalized_width90_equal_cell':float(f.normalized_width90.mean())}
        families.append(row)
    pd.DataFrame(families).to_csv(args.public_dir/'FAMILY_SUMMARY.csv',index=False)
    spec=STUDY/'provenance/SECONDARY_COMPARISON_SPEC.json'
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'secondary_analysis':True,
            'model_fit_or_update_count':0,'source_sha256':sha(__file__),'spec_sha256':sha(spec),
            'paired_forecast_files_read_with_horizon_reuse':checked,'paired_cells':22,'families':families,
            'comparisons':comparisons,'bootstrap_replicates':1999,'block_length':40,'seed':20261019,
            'primary_gate_unchanged':True,'caveats':json.loads(spec.read_text())['comparison_caveats'],
            'synchronized_scope':'Same-year union-of-observed-origin-date circular block weights shared by markets. Conditional on fixed fitted models and three test years; not a population-of-markets confidence statement.'}
    save(args.public_dir/'SUMMARY.json',result)
    print(json.dumps({'complete':True,'families':len(families),'cells':22,'new_fits':0}))


if __name__=='__main__':
    main()
