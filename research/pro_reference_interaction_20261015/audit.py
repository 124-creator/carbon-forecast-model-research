"""Independent arithmetic plus saved-weight and representative fresh causal checks."""
from __future__ import annotations
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
import torch
from model import (load_prefix,reference,InteractionGRU,PlainGRU,predict,HORIZONS,SEEDS)
from experiment import fit_one,dump,setup

def run(data_path,out,causal=True):
    setup();p,dates,_=load_prefix(data_path);ref=reference(p,240)
    origins=[o for o in range(459,930) if dates[o][:4]==dates[o+20][:4]]
    scores={r['name']:r for r in csv.DictReader((out/'all_scores.csv').open())}
    largest=0.;checked=0
    for path in sorted((out/'predictions').glob('*.npz')):
        with np.load(path,allow_pickle=False) as z:pr=z['pred'].copy()
        cells=[];ens_cells=[]
        for year in ('2020','2021'):
            idx=[o for o in origins if dates[o].startswith(year)]
            for h in HORIZONS:
                truth=np.array([p[o+h]-p[o] for o in idx]);den=float(np.dot(truth,truth)/len(idx))
                errs=[float(np.dot(pr[s,idx,h-1]-truth,pr[s,idx,h-1]-truth)/len(idx)) for s in range(pr.shape[0])]
                cells.append(sum(errs)/len(errs)/den)
                err=np.mean(pr[:,idx,h-1],axis=0)-truth
                ens_cells.append(float(np.dot(err,err)/len(idx))/den)
        for key,value in [('mean_seed_relative_MSE',sum(cells)/6),('prediction_ensemble_relative_MSE',sum(ens_cells)/6)]:
            delta=abs(value-float(scores[path.stem][key]));largest=max(largest,delta);checked+=1
            if delta>1e-10:raise ValueError('Independent score mismatch')
    weight_max=0.;fit_count=0;history_epochs=0
    for path in sorted((out/'fits').glob('*.json')):
        r=json.loads(path.read_text());fit_count+=1;history_epochs+=r['epochs']
        if r['max_training_target']>=r['first_validation_origin'] or r['max_validation_target']>=r['fit_end']:
            raise ValueError('Future/purge failure')
        if len(r['history'])!=r['epochs'] or not np.isfinite(r['history']).all():raise ValueError('Invalid training log')
        model=PlainGRU() if r['mode']=='GRU' else InteractionGRU(r['mode'])
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
            sd=float(z['scale']);idx=z['origins'].copy();stored=z['prediction'].copy()
            model.load_state_dict({k:torch.from_numpy(z[k].copy()) for k in model.state_dict()})
        recomputed=predict(model,sd,p,ref,idx)
        delta=float(np.max(abs(recomputed-stored)));weight_max=max(weight_max,delta)
        if delta>1e-10:raise ValueError('Saved-weight prediction mismatch')
    if fit_count!=624:raise ValueError('Incomplete fit registry')
    online=[]
    for seed in SEEDS:
        rows=json.loads((out/'online'/f's{seed}.json').read_text());events=set()
        for r in rows:
            for o in range(r['origin_min'],r['origin_max']+1):
                event=(o,r['time']-o)
                if event in events or not 1<=event[1]<=20:raise ValueError('Repeated/invalid feedback')
                events.add(event)
        if len(rows)!=470 or len(events)!=9210:raise ValueError('Missing online events')
        online.append(dict(seed=seed,updates=470,unique_feedback_events=9210))
    causal_rows=[]
    if causal:
        pert=p.copy();pert[459:]+=10000;other=reference(pert,240)
        for mode,lam in [('RESIDUAL',1.),('ADDITIVE',1.),('FULL',1.),('MIXED',1.),('GRU',0.)]:
            model,sd,receipt=fit_one(pert,other,459,mode,lam,17)
            path=out/'fits'/f'{mode}_l{lam:g}_s17_b00.npz'
            with np.load(path,allow_pickle=False) as z:
                original=PlainGRU() if mode=='GRU' else InteractionGRU(mode)
                original.load_state_dict({k:torch.from_numpy(z[k].copy()) for k in original.state_dict()})
                scale=float(z['scale'])
            param_max=max(float(torch.max(torch.abs(v-original.state_dict()[k]))) for k,v in model.state_dict().items())
            delta=float(np.max(abs(predict(model,sd,pert,other,np.array([458]))-predict(original,scale,p,ref,np.array([458])))))
            if param_max!=0 or delta!=0 or sd!=scale:raise ValueError('Fresh causal refit mismatch')
            causal_rows.append(dict(mode=mode,seed=17,fit_end=459,max_parameter_difference=param_max,
                                    max_prediction_difference=delta,scope='one representative fresh refit'))
    current={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}
    initial=json.loads((out/'source_hashes_before_fits.json').read_text())
    unchanged={k:current[k]==v for k,v in initial.items()}
    if not all(unchanged.values()):raise ValueError('Initially executed source changed')
    result=dict(passed=True,metrics_recomputed=checked,max_score_difference=largest,
                saved_weight_files=fit_count,max_saved_prediction_difference=weight_max,
                training_epochs=history_epochs,online=online,representative_causal_refits=causal_rows,
                originally_executed_source_unchanged=unchanged,
                scope='Independent scoring implementation, all saved weights, representative fresh refits; not third-party replication')
    dump(out/'AUDIT.json',result)
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--no-causal',action='store_true')
    a=ap.parse_args();run(a.data,a.out,not a.no_causal)
