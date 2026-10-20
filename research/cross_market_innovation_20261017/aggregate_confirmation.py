"""Read completed, frozen confirmation forecasts; compute the declared statistics."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'confirmation_source'))
import csv
import hashlib
import json
from collections import defaultdict
import numpy as np
from paired_statistics import confirmation_inference


def main():
    complete=json.loads((ROOT/'CONFIRMATION_RUN_COMPLETE.json').read_text())
    if not complete['all_completed']:raise ValueError('Cannot aggregate an incomplete confirmation')
    plan=json.loads((ROOT/'CONFIRMATION_PRE_RUN.json').read_text())
    frozen=plan['source_sha256']['paired_statistics.py']
    if hashlib.sha256((ROOT/'confirmation_source/paired_statistics.py').read_bytes()).hexdigest()!=frozen:
        raise ValueError('Registered statistics code changed')
    results=json.loads((ROOT/'CONFIRMATION_RESULTS.json').read_text())['results']
    groups=defaultdict(list)
    for r in results:groups[(r['task']['scope'],r['task']['config']['id'])].append(r)
    model_predictions={};allviews={};summary=[];reference=None;full_rows=[]
    for (scope,cid),rows in groups.items():
        key=scope+':'+cid
        seeds=sorted({r['task']['seed'] for r in rows})
        views=set(rows[0]['metrics'])
        per_view={v:[] for v in views}
        for seed in seeds:
            selected=sorted([r for r in rows if r['task']['seed']==seed],key=lambda r:r['task']['cutoff'])
            arrays=[np.load(ROOT/'local_only/confirmation'/(r['task']['id']+'_predictions.npz'),allow_pickle=False) for r in selected]
            metadata={k:np.concatenate([a[k] for a in arrays]) for k in ('origin','dates','raw_y','stat_raw')}
            if reference is None:reference=metadata
            for k in reference:
                if not np.array_equal(reference[k],metadata[k]):raise ValueError('Unpaired confirmation metadata: '+key+'/'+k)
            for v in views:per_view[v].append(np.concatenate([a['prediction_'+v] for a in arrays]))
            for a in arrays:a.close()
        views_np={v:np.stack(p) for v,p in per_view.items()}
        full=views_np['full']
        # A deterministic baseline may be repeated for a fixed-seed comparison;
        # this leaves its loss unchanged and is not an extra model fit.
        model_predictions[key]=np.repeat(full,3,axis=0) if len(seeds)==1 else full
        allviews[key]=views_np
        y=reference['raw_y'];years=np.array([int(str(d)[:4]) for d in reference['dates']]);horizons=[0,4,19]
        cells=[]
        for year in (2024,2025):
            use=years==year
            for h in horizons:
                mse=float(np.mean((full[:,use,h]-y[None,use,h])**2))
                rw=float(np.mean(y[use,h]**2))
                cells.append(dict(year=year,horizon=h+1,origins=int(use.sum()),mse=mse,rw_mse=rw,
                                  relative_mse=mse/rw,mae=float(np.mean(abs(full[:,use,h]-y[None,use,h])))))
        scores={v:float(np.mean([np.mean((arr[:,years==yr,h]-y[None,years==yr,h])**2)/np.mean(y[years==yr,h]**2)
                                for yr in (2024,2025) for h in horizons])) for v,arr in views_np.items()}
        ss={str(seed):float(np.mean([np.mean((full[j,years==yr,h]-y[years==yr,h])**2)/np.mean(y[years==yr,h]**2)
                                    for yr in (2024,2025) for h in horizons])) for j,seed in enumerate(seeds)}
        summary.append(dict(name=key,scope=scope,config_id=cid,config=rows[0]['task']['config'],
                            score=float(np.mean([c['relative_mse'] for c in cells])),seeds=seeds,fits=len(rows),
                            per_seed=ss,per_year={str(yr):float(np.mean([c['relative_mse'] for c in cells if c['year']==yr])) for yr in (2024,2025)},
                            per_horizon={str(h):float(np.mean([c['relative_mse'] for c in cells if c['horizon']==h])) for h in (1,5,20)},
                            views=scores,cells=cells))
    y=reference['raw_y'];years=np.array([int(str(d)[:4]) for d in reference['dates']]);horizons=[0,4,19]
    for key,pred in [('rw',np.zeros_like(y)),('stat',reference['stat_raw'])]:
        name='deterministic:'+key;model_predictions[name]=np.repeat(pred[None,:,:],3,axis=0)
        cells=[]
        for yr in (2024,2025):
            use=years==yr
            for h in horizons:
                mse=float(np.mean((pred[use,h]-y[use,h])**2));rw=float(np.mean(y[use,h]**2))
                cells.append(dict(year=yr,horizon=h+1,origins=int(use.sum()),mse=mse,rw_mse=rw,relative_mse=mse/rw,mae=float(np.mean(abs(pred[use,h]-y[use,h])))))
        summary.append(dict(name=name,scope='deterministic',config_id=key,score=float(np.mean([c['relative_mse'] for c in cells])),
                            seeds=[],fits=0,cells=cells,
                            per_year={str(yr):float(np.mean([c['relative_mse'] for c in cells if c['year']==yr])) for yr in (2024,2025)},
                            per_horizon={str(h):float(np.mean([c['relative_mse'] for c in cells if c['horizon']==h])) for h in (1,5,20)}))
    candidate='pooled:'+plan['selected_candidate']['id']
    baseline=plan['selected_baseline']['scope']+':'+plan['selected_baseline']['config_id']
    augmentation='pooled:augment_h32_s1_l0'
    primary_predictions={name:model_predictions[name] for name in (candidate,baseline,augmentation)}
    inference=confirmation_inference(primary_predictions,y,years,reference['origin'],candidate=candidate,
                                    strongest_development_baseline=baseline,matched_augmentation=augmentation,
                                    seed_ids=(17,29,43))
    (ROOT/'CONFIRMATION_INFERENCE.json').write_text(json.dumps(inference,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    summary.sort(key=lambda r:r['score'])
    out=dict(models=summary,primary_candidate=candidate,primary_baseline=baseline,matched_augmentation=augmentation,
             origins=int(len(y)),origins_by_year={str(yr):int((years==yr).sum()) for yr in (2024,2025)},
             seed_loss_mean_not_ensemble=True,all_frozen_fits_completed=True,selection_changed_after_confirmation=False,
             both_primary_comparisons_pass=inference['both_primary_comparisons_pass'])
    (ROOT/'CONFIRMATION_SUMMARY.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    destination=ROOT/'results';destination.mkdir(exist_ok=True)
    fieldnames=['model','seed','date','origin','horizon','raw_future_change','predicted_change']
    with (destination/'CONFIRMATION_FORECASTS_H1_H5_H20.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fieldnames,lineterminator='\n');writer.writeheader()
        for name in sorted(model_predictions):
            p=model_predictions[name]
            seeds_to_write=[17] if name.startswith('deterministic:') or ':ridge_' in name else [17,29,43]
            for si,seed in enumerate(seeds_to_write):
                for j in range(len(y)):
                    for h in horizons:
                        writer.writerow(dict(model=name,seed=seed,date=str(reference['dates'][j]),origin=int(reference['origin'][j]),horizon=h+1,
                                             raw_future_change=format(float(y[j,h]),'.17g'),predicted_change=format(float(p[si,j,h]),'.17g')))
    np.savez_compressed(destination/'CONFIRMATION_PRIMARY_INPUTS.npz',raw_y=y,years=years,origins=reference['origin'],dates=reference['dates'],
                        candidate=model_predictions[candidate],baseline=model_predictions[baseline],augmentation=model_predictions[augmentation],seed_ids=np.array([17,29,43]))
    print(json.dumps(dict(models=[{k:r[k] for k in ('name','score','per_year')} for r in summary],
                          primary=inference['blocks']['40'],origins_by_year=out['origins_by_year']),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
