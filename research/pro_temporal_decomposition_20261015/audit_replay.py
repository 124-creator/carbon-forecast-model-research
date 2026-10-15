"""Independent arithmetic, stored-weight checks, and representative fresh refits."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from model import read_prefix,training_arrays,MODES,SEEDS,make_model,forecast,fit_model,numeric_state
from run_market import dump
ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--data',default=str(ROOT/'data/guangzhou_prefix950.csv'));ap.add_argument('--refits',action='store_true')
    a=ap.parse_args();r=Path(a.out);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    p,dates,receipt=read_prefix(a.data)
    original_hashes=json.loads((r/'SOURCE_HASHES.json').read_text())
    for f,h in original_hashes.items():
        if hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=h:raise ValueError('source changed since training')
    ids=np.array([i for i in range(459,930) if dates[i][:4]==dates[i+20][:4]],int)
    truth=np.array([[p[i+h]-p[i] for h in range(1,21)] for i in ids]);years=np.array([d[:4] for d in dates[ids]])
    with np.load(r/'TARGETS.npz',allow_pickle=False) as z:
        for k,v in [('origins',ids),('truth',truth),('years',years)]:np.testing.assert_array_equal(z[k],v)
    expected={(m,s,b) for m in MODES for s in SEEDS for b in range(12)};actual=set();weight_max=0.;theta_init={};first_models={}
    for file in sorted((r/'fits').glob('*.json')):
        row=json.loads(file.read_text());mode=row['mode'];seed=row['seed'];end=row['end_exclusive'];block=(end-459)//40
        actual.add((mode,seed,block));items=training_arrays(p,end);rec=items[-1]
        for k,v in rec.items():
            if row[k]!=v:raise ValueError('log time/scale mismatch')
        if mode in MODES[:4]:
            key=(seed,block);theta_init.setdefault(key,row['theta_initial_digest'])
            if theta_init[key]!=row['theta_initial_digest']:raise ValueError('factorial initial parameters mismatch')
        exp_router=40 if mode.endswith('TEMP') else 120 if mode.endswith('ERM') else 0
        if len(row['router_history'])!=exp_router:raise ValueError('routing budget mismatch')
        if row['C_last_target']>=end or row['A_last_target']>=row['B_first_origin'] or row['F_last_target']>=row['C_first_origin']:raise ValueError('maturity failure')
        with np.load(file.with_suffix('.npz'),allow_pickle=False) as z:
            m=make_model(mode,seed)
            state={k.removeprefix('parameter__'):torch.as_tensor(z[k].copy()) for k in z.files if k.startswith('parameter__')}
            m.load_state_dict(state)
            if 'router.phi' in state and float(state['router.phi'].abs().max())>2.000001:raise ValueError('router bound')
            pr,_=forecast(m,float(z['scale']),p,z['origins']);delta=float(np.max(abs(pr-z['prediction'])))
            weight_max=max(weight_max,delta)
            if delta!=0:raise ValueError('saved weights do not reconstruct predictions')
    if actual!=expected:raise ValueError('fit identities incomplete')
    preds={}
    with np.load(r/'NEURAL_PREDICTIONS.npz',allow_pickle=False) as z:
        for m in MODES:preds[m]=np.stack([z[f'{m}_s{s}'][:,[0,4,19]] for s in SEEDS])
    with np.load(r/'BASELINE_PREDICTIONS.npz',allow_pickle=False) as z:preds.update({k:z[k][None] for k in z.files})
    cells={(row['model'],row['statistic'],row['year'],int(row['h'])):row for row in csv.DictReader((r/'by_year_horizon.csv').open())}
    allscores={row['model']:row for row in csv.DictReader((r/'all_scores.csv').open())};maxerr=0.;nmetrics=0;variance_identity=0.
    def near(x,y):
        nonlocal maxerr,nmetrics
        maxerr=max(maxerr,abs(float(x)-float(y)));nmetrics+=1
        if not np.isclose(x,float(y),rtol=2e-12,atol=2e-12):raise ValueError('arithmetic discrepancy')
    for name,pr in preds.items():
        ratios=[];ensratios=[]
        errors=pr-truth[None,:,[0,4,19]]
        lhs=np.mean(errors**2,0)-(np.mean(pr,0)-truth[:,[0,4,19]])**2
        rhs=np.mean((pr-pr.mean(0))**2,0);variance_identity=max(variance_identity,float(np.max(abs(lhs-rhs))))
        for stat,er in [('seed_mean',errors),('prediction_ensemble',errors.mean(0)[None])]:
            for yr in ('2020','2021'):
                ix=np.where(years==yr)[0]
                for j,h in enumerate([1,5,20]):
                    # Explicit scalar accumulation, separate from analysis implementation.
                    values=[float(er[s,i,j]) for s in range(er.shape[0]) for i in ix]
                    mse=sum(e*e for e in values)/len(values);mae=sum(abs(e) for e in values)/len(values)
                    rw=sum(float(truth[i,h-1])**2 for i in ix)/len(ix);ratio=mse/rw
                    row=cells[(name,stat,yr,h)]
                    for k,val in [('MSE',mse),('RMSE',np.sqrt(mse)),('MAE',mae),('relative_MSE',ratio)]:near(val,row[k])
                    (ratios if stat=='seed_mean' else ensratios).append(ratio)
        near(sum(ratios)/6,allscores[name]['primary_seed_mean_MSE_ratio']);near(sum(ensratios)/6,allscores[name]['seed_ensemble_MSE_ratio'])
    replay=[]
    if a.refits:
        checks=[(m,17,0) for m in MODES]+[('D_TEMP',53,11)]
        for mode,seed,b in checks:
            key=f'{mode}_s{seed}_b{b:02d}';end=459+40*b;start=time.perf_counter()
            # Perturb only prices beyond training cutoff; compare issuance end-1.
            q=p.copy();q[end:]+=10000
            fresh,sd,log=fit_model(q,end,mode,seed)
            with np.load(r/'fits'/f'{key}.npz',allow_pickle=False) as z:
                state={k.removeprefix('parameter__'):z[k].copy() for k in z.files if k.startswith('parameter__')}
                equal=all(np.array_equal(v,numeric_state(fresh)[k]) for k,v in state.items())
                stored=make_model(mode,seed);stored.load_state_dict({k:torch.from_numpy(v) for k,v in state.items()})
                x,_=forecast(stored,float(z['scale']),p,[end-1]);y,_=forecast(fresh,sd,q,[end-1])
                if not equal or not np.array_equal(x,y) or sd!=float(z['scale']):raise ValueError('representative full refit mismatch')
            replay.append(dict(mode=mode,seed=seed,block=b,perturbed_from=end,unchanged_issuance=end-1,all_parameters_exact=True,prediction_exact=True,seconds=time.perf_counter()-start))
            print('representative replay',key,'passed',flush=True)
    dump(r/'AUDIT.json',dict(passed=True,fit_identities=len(actual),metrics_recomputed=nmetrics,arithmetic_max_abs_error=maxerr,saved_parameter_prediction_max_abs_difference=weight_max,variance_identity_max_abs_error=variance_identity,source_unchanged=True,scale_and_maturity_all_fits_checked=True,factorial_initial_theta_matched=True,representative_fresh_replays=replay,full_pipeline_retrained=False,independent_external_replication=False,data=receipt))
    print('AUDIT PASSED',nmetrics,maxerr)
if __name__=='__main__':main()
