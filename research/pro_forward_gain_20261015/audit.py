"""Independent metric arithmetic, all saved-weight reconstruction, representative refits."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import core
from experiment import jwrite,fit,predict

def run(out,fresh=True):
    out=Path(out);core.configure();p,dates,receipt=core.load_prefix(out/'approved_prefix.csv');ref=core.reference(p)
    with np.load(out/'EVALUATION.npz',allow_pickle=False) as z:
        ori=z['origins'].copy();truth=z['truth'].copy();mask=z['eligible'].copy();years=z['years'].copy()
    independent_truth=np.array([[p[int(t)+h]-p[int(t)] for h in range(1,21)] for t in ori])
    if not np.array_equal(truth,independent_truth):raise ValueError('Target mismatch')
    metrics=json.loads((out/'ALL_MODEL_METRICS.json').read_text());errors=[];count=0
    for name,m in metrics.items():
        with np.load(out/'predictions'/f'{name}.npz',allow_pickle=False) as z:pr=z['prediction'].copy()
        cells=[];ecells=[]
        for year in ('2020','2021'):
            ix=np.flatnonzero(mask & (years==year))
            for h in (1,5,20):
                den=sum(truth[i,h-1]**2 for i in ix)/len(ix)
                per=[sum((pr[s,i,h-1]-truth[i,h-1])**2 for i in ix)/len(ix) for s in range(len(pr))]
                ratio=float(np.mean(per)/den);cells.append(ratio)
                er=sum((np.mean(pr[:,i,h-1])-truth[i,h-1])**2 for i in ix)/len(ix)/den;ecells.append(er)
                c=next(c for c in m['cells'] if c['year']==year and c['horizon']==h)
                errors.append(abs(ratio-c['ratio']));count+=1
        errors.extend([abs(float(np.mean(cells))-m['score']),abs(float(np.mean(ecells))-m['ensemble'])]);count+=2
    max_rebuild=0.;fit_count=0;epochs=0;late=0;details_errors=[]
    for path in sorted((out/'fits').glob('*.npz')):
        log=json.loads(path.with_suffix('.json').read_text());mode=log['mode'];seed=log['seed']
        m=core.make_model(mode,seed)
        with np.load(path,allow_pickle=False) as z:
            state={k[7:]:torch.from_numpy(z[k].copy()) for k in z.files if k.startswith('param__')};m.load_state_dict(state)
            origins=z['origins'].copy();pred=z['prediction'].copy();sd=float(z['scale']);saved_details={k[8:]:z[k].copy() for k in z.files if k.startswith('detail__')}
        rebuilt,detail=predict(m,sd,p,ref,origins);max_rebuild=max(max_rebuild,float(abs(rebuilt-pred).max()))
        for k,v in saved_details.items():details_errors.append(float(abs(detail[k]-v).max()))
        if not (log['max_training_target']<log['first_validation_origin'] and log['max_validation_target']<log['end']):late+=1
        for fold in log['folds']:
            if fold['support_max_target']>=fold['query_min_origin']:late+=1
        if log['scale_max_observation']>=log['folds'][0]['query_min_origin']:late+=1
        fit_count+=1;epochs+=log['epochs']
    if fit_count != 384 or max(errors)>1e-11 or max_rebuild!=0 or late:raise ValueError('Audit failure')
    combos=[]
    with np.load(out/'COMBINATION_WEIGHTS.npz',allow_pickle=False) as z:weights=z['mature'].copy()
    with np.load(out/'predictions/GRU.npz',allow_pickle=False) as z:gp=z['prediction'].copy()
    b=ref['b'][ori];recalc=np.full_like(weights,.5)
    for s in range(len(gp)):
        for k,t in enumerate(ori):
            for h in range(1,21):
                available=[i for i,o in enumerate(ori) if o+h<=t][-120:]
                if len(available)<30:continue
                dx=gp[s,available,h-1]-b[available,h-1];dy=truth[available,h-1]-b[available,h-1]
                den=sum(v*v for v in dx)
                if den>1e-12:recalc[s,k,h-1]=min(1,max(0,sum(x*y for x,y in zip(dx,dy))/den))
    combo_error=float(abs(weights-recalc).max())
    causal=[];start=time.perf_counter()
    if fresh:
        for mode in ('GAIN','FWD_FIXED','IN_SAMPLE','FIXED6'):
            end=459;seed=17;q=p.copy();q[end:]+=10000
            r2=core.reference(q);a,sa,la=fit(p,ref,end,mode,seed);bmodel,sb,lb=fit(q,r2,end,mode,seed)
            equal=all(torch.equal(v,bmodel.state_dict()[k]) for k,v in a.state_dict().items())
            pa,_=predict(a,sa,p,ref,[end-1]);pb,_=predict(bmodel,sb,q,r2,[end-1])
            if not equal or sa!=sb or not np.array_equal(pa,pb):raise ValueError('Fresh prefix perturbation failed')
            with np.load(out/'fits'/f'{mode}_s17_b00.npz',allow_pickle=False) as z:
                first_equal=all(np.array_equal(v.detach().numpy(),z['param__'+k]) for k,v in a.state_dict().items())
            if not first_equal:raise ValueError('Representative fit did not reproduce first run')
            causal.append(dict(mode=mode,seed=17,fit_end=end,perturbed_from=end,unchanged_forecast_origin=end-1,parameters_exact=True,first_fit_exact=True))
            print('Fresh causal audit',mode,flush=True)
    report=dict(passed=True,metrics_recomputed=count,metric_max_error=max(errors),weight_files_rebuilt=fit_count,weight_prediction_max_error=max_rebuild,detail_max_error=max(details_errors),epochs_checked=epochs,maturity_violations=late,combination_weight_max_error=combo_error,representative_fresh_refits=causal,extra_fits=len(causal)*2,extra_fit_wall_seconds=time.perf_counter()-start,data_receipt=receipt,scope='Independent arithmetic implementation and representative fresh local refits; NOT third-party replication or all-fit second training')
    jwrite(out/'AUDIT.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',required=True);a.add_argument('--no-fresh',action='store_true');o=a.parse_args();run(o.out,not o.no_fresh)
