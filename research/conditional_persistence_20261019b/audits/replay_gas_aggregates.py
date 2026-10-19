"""Independent GAS benchmark aggregation replay; no production imports/fits."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd

STUDY=Path(__file__).resolve().parents[1]
PRIVATE=Path('/tmp/carbon-gas-normalization-run')
PUBLIC=STUDY/'results/gas_benchmark'
PANELS=[f'{m}_{y}' for m in ('EUA_LONG','GUANGZHOU','HUBEI','SHANGHAI')
        for y in (2021,2022,2023) if (m,y)!=('SHANGHAI',2021)]
FAMILIES=[f'{n}_gru_gmm' for n in ('global','ewma','gas')]
FAMILIES += [f'{n}_{kind}' for n in ('global','ewma','gas') for kind in ('no_neural','empirical101')]
FAMILIES += ['daily_ewma_normal']


def main():
    summary=json.loads((PUBLIC/'SUMMARY.json').read_text())
    pairs=[(x['candidate'],x['baseline']) for x in summary['comparisons']]
    frame=pd.read_csv(PUBLIC/'CELL_METRICS.csv').set_index(['panel','horizon','family'])
    rng=np.random.default_rng(20261019)
    draws=np.zeros((1999,len(pairs)))
    points=[[] for _ in pairs]
    artifacts=origins=metric_cells=0
    max_metric_error=0.
    for panel in PANELS:
        horizon_scores=[]
        common=None
        for h in (5,20):
            family_scores=[]
            for family in FAMILIES:
                seed_arrays=[];seed_means=[]
                seeds=(17,41,89) if family.endswith('gru_gmm') else (0,)
                for seed in seeds:
                    path=PRIVATE/'predictions'/f'{panel}_h{h}'/family/f'seed_{seed}.npz'
                    receipt=json.loads(path.with_suffix('.json').read_text())
                    assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['prediction_sha256']
                    with np.load(path,allow_pickle=False) as z:
                        if common is None:common=z['origin'].copy()
                        np.testing.assert_array_equal(common,z['origin'])
                        seed_arrays.append(z['score_normalized_CRPS'].copy())
                        means={key[6:]:float(z[key].mean()) for key in z.files if key.startswith('score_')}
                    for key,value in means.items():
                        np.testing.assert_allclose(value,receipt['metrics'][key],rtol=1e-13,atol=1e-12)
                    seed_means.append(means)
                    artifacts+=1;origins+=len(common)
                family_scores.append(np.mean(seed_arrays,axis=0))
                public=frame.loc[(panel,h,family)]
                for key in seed_means[0]:
                    mean=float(np.mean([x[key] for x in seed_means]))
                    error=abs(mean-float(public[key]))
                    max_metric_error=max(max_metric_error,error)
                    np.testing.assert_allclose(mean,public[key],rtol=2e-12,atol=2e-10)
                    metric_cells+=1
            horizon_scores.append(np.column_stack(family_scores))
        loss=np.stack(horizon_scores,axis=1)
        actual=loss.mean(0)
        for j,(a,b) in enumerate(pairs):
            points[j].extend((actual[:,FAMILIES.index(a)]/actual[:,FAMILIES.index(b)]-1).tolist())
        n=len(loss)
        # Weighted-count reconstruction of a paired circular-block sample.
        starts=rng.integers(0,n,size=(1999,(n+39)//40))
        for b in range(1999):
            ix=((starts[b,:,None]+np.arange(40))%n).reshape(-1)[:n]
            weights=np.bincount(ix,minlength=n)/n
            sampled=np.einsum('n,nhf->hf',weights,loss)
            for j,(a,c) in enumerate(pairs):
                draws[b,j]+=(sampled[:,FAMILIES.index(a)]/sampled[:,FAMILIES.index(c)]-1).sum()/22
    max_point=max_ci=0.
    for j,row in enumerate(summary['comparisons']):
        point=float(np.mean(points[j]));ci=np.quantile(draws[:,j],[.025,.975])
        max_point=max(max_point,abs(point-row['mean_cell_relative_CRPS']))
        max_ci=max(max_ci,abs(ci[0]-row['ci95_lower']),abs(ci[1]-row['ci95_upper']))
        np.testing.assert_allclose(point,row['mean_cell_relative_CRPS'],rtol=0,atol=2e-13)
        np.testing.assert_allclose(ci,[row['ci95_lower'],row['ci95_upper']],rtol=0,atol=2e-13)
        assert int(np.sum(np.asarray(points[j])<0))==row['strictly_better_cells']
    result={'status':'PASS_COMPLETE_GAS_AGGREGATION_REPLAY',
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'audit_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'prediction_artifacts':artifacts,'origin_forecasts':origins,'cell_metric_values':metric_cells,
            'paired_comparisons':len(pairs),'resamples_per_comparison':1999,
            'maximum_cell_metric_absolute_error':max_metric_error,
            'maximum_relative_CRPS_absolute_error':max_point,'maximum_CI_absolute_error':max_ci,
            'source_summary_sha256':hashlib.sha256((PUBLIC/'SUMMARY.json').read_bytes()).hexdigest(),
            'new_fits_or_steps':0,'production_aggregator_imported':False,
            'scope':'All352 saved forecast receipts and220 seed-averaged model/cell rows, plus all paired comparison means/wins/1999-block intervals. Scores themselves are checked by the separate independent score replay.'}
    (STUDY/'audits/GAS_AGGREGATION_REPLAY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
