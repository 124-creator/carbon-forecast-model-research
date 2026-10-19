#!/usr/bin/env python3
"""Independent data/split/scale reconstruction and worker checks; zero fits."""
from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

STUDY=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(STUDY/'source'))
import run_new_market_replication as replication
import run_model_search as common
from innovation_models import ShockAllocationForecaster

EXPECTED={
    'BEIJING_MEAN':(673,403,537,319,114,453,116),
    'TIANJIN_MEAN':(457,274,365,190,71,281,72),
    'CHONGQING_MEAN':(240,144,192,60,28,108,28),
}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def worker_view(market):
    data=common.prepare(market,'fixed602020')
    return {'prepare':common.prepare.__module__+'.'+common.prepare.__name__,
            'make_model':common.make_model.__module__+'.'+common.make_model.__name__,
            'data_root':str(replication.DATA_ROOT.resolve()),
            'counts':data['manifest']['counts'],'shape':list(data['x'].shape)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError('Choose a new audit receipt')
    replication.DATA_ROOT=args.data_root.resolve()
    os.environ['CARBON_SCIENCEDB_DATA_DIR']=str(replication.DATA_ROOT)
    torch.set_num_threads(1)
    files=[STUDY/'source'/name for name in ['run_model_search.py','innovation_models.py',
        'run_liquidity_extension.py','liquidity_models.py','run_new_market_replication.py']]
    source_hashes={p.name:sha(p) for p in files}
    assert source_hashes['run_model_search.py']=='26347433fe6416298e4969da618cbba291999f3a66131172bc8f2e510f3633bf'
    assert source_hashes['innovation_models.py']=='cca83f45474c5110f180ae8b7763131f33596a664939c4ebd34c32db24b4dc1e'
    assert source_hashes['run_liquidity_extension.py']=='95665858a1945b00bba6d6233022c34c0a58993899ccadda8c895f64ffe69807'
    assert source_hashes['liquidity_models.py']=='91dd66c0b33b8d75100e37cc382f01be44021ae538f6cbbedddda7f9b9386e0b'
    assert common.prepare is replication.prepare_replication
    assert common.make_model is replication.make_replication_model
    energy={}
    for name in ['brent','wti','henryhub']:
        p=common.REPO/'data/parsed'/(name+'_eia_daily.csv')
        observations=[]
        for r in read_csv(p):
            try:v=float(r['spot_price'])
            except (ValueError,TypeError):continue
            if math.isfinite(v):observations.append((r['date'],v))
        observations.sort(key=lambda x:x[0])
        energy[name]=([r[0] for r in observations],[r[1] for r in observations],sha(p))
    comparisons=[]
    for market,(n,a,b,nt,nv,nf,nz) in EXPECTED.items():
        rows=read_csv(args.data_root/(market+'.csv'))
        dates=[r['date'] for r in rows]
        prices=[float(r['price']) for r in rows]
        volumes=[float(r['volume_tons']) for r in rows]
        assert len(rows)==n and dates==sorted(set(dates))
        assert all(v>0 and math.isfinite(v) for v in volumes)
        assert all(p>0 and math.isfinite(p) for p in prices)
        assert all(p==float(r['turnover_yuan'])/v for r,p,v in zip(rows,prices,volumes))
        token_columns=[[0.0]+[prices[i]-prices[i-1] for i in range(1,n)]]
        energy_ages={}
        for name,(ed,ep,source_sha) in energy.items():
            aligned=[];ages=[]
            for date in dates:
                day=dt.date.fromisoformat(date);cutoff=(day-dt.timedelta(days=14)).isoformat()
                j=bisect.bisect_right(ed,cutoff)-1
                assert j>=0 and ed[j]<=cutoff
                aligned.append(ep[j]);ages.append((day-dt.date.fromisoformat(ed[j])).days)
            token_columns.append([0.0]+[aligned[i]-aligned[i-1] for i in range(1,n)])
            energy_ages[name]={'minimum_days':min(ages),'maximum_days':max(ages),'source_sha256':source_sha}
        tokens=np.array(token_columns,dtype=np.float64).T
        logs=[math.log1p(v) for v in volumes]
        context=np.zeros((n,2),dtype=np.float32)
        for t in range(10,n):
            prior=sorted(logs[max(0,t-60):t]);k=len(prior);mid=k//2
            median=prior[mid] if k%2 else (prior[mid-1]+prior[mid])/2
            mu=math.fsum(prior)/k
            sd=math.sqrt(math.fsum((v-mu)**2 for v in prior)/k)
            context[t]=[min(5,max(-5,(logs[t]-median)/max(sd,.25))),1]
        origins=np.array(list(range(64,n-20)),dtype=int)
        roles={'train':[t+20<a for t in origins],
               'val':[t>=a and t+20<b for t in origins],
               'final':[t+20<b for t in origins],
               'test':[t>=b for t in origins]}
        roles={k:np.array(v,dtype=bool) for k,v in roles.items()}
        expected_counts=dict(train=nt,val=nv,final=nf,test=nz)
        assert {k:int(v.sum()) for k,v in roles.items()}==expected_counts
        scale=np.array([max(1e-7,math.sqrt(math.fsum(v*v for v in col[1:a])/(a-1))) for col in token_columns])
        xx=np.stack([tokens[t-63:t+1]/scale for t in origins]).astype(np.float32)
        xx=np.concatenate([xx,np.stack([context[t-63:t+1] for t in origins])],axis=2)
        yy=np.array([[prices[t+h]-prices[t] for h in [1,5,20]] for t in origins])
        variance=np.array([max(math.fsum(float(v)*float(v) for v in yy[roles['train'],j])/nt,scale[0]**2*1e-6) for j in range(3)])
        actual=common.prepare(market,'fixed602020')
        assert np.array_equal(origins,actual['origins'])
        for name,mask in roles.items():assert np.array_equal(mask,actual['masks'][name])
        assert np.allclose(xx,actual['x'],atol=1e-6,rtol=1e-6)
        assert np.allclose(yy,actual['y'],atol=1e-11,rtol=1e-12)
        assert np.allclose(scale,actual['scale'],atol=1e-12,rtol=1e-12)
        assert np.allclose(variance,actual['variance'],atol=1e-10,rtol=1e-12)
        assert np.array_equal(context[:,1],actual['volume_context'][:,1])
        comparisons.append({'market':market,'source_sha256':sha(args.data_root/(market+'.csv')),
            'row_boundaries':[0,a,b,n],'counts':expected_counts,
            'origin_ranges':{k:[int(origins[v][0]),int(origins[v][-1])] for k,v in roles.items()},
            'maximum_input_difference':float(np.max(np.abs(xx-actual['x']))),
            'maximum_target_difference':float(np.max(np.abs(yy-actual['y']))),
            'maximum_rms_difference':float(np.max(np.abs(scale-actual['scale']))),
            'maximum_target_variance_difference':float(np.max(np.abs(variance-actual['variance']))),
            'energy_ages':energy_ages,
            'last_training_scale_row':a-1,'last_training_label_row':int(origins[roles['train']][-1])+20,
            'last_validation_label_row':int(origins[roles['val']][-1])+20,
            'first_test_origin_row':b,'context_valid_rows':n-10})
    expected_worker=worker_view('BEIJING_MEAN');workers={}
    for method in ['fork','spawn']:
        if method in mp.get_all_start_methods():
            with ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context(method)) as pool:
                workers[method]=pool.submit(worker_view,'BEIJING_MEAN').result(timeout=45)
            assert workers[method]==expected_worker
    torch.manual_seed(199)
    synthetic=torch.randn(2,64,6);synthetic[:,:,5]=1
    changed=synthetic.clone();changed[:,:,4]=20;changed[:,:,5]=0
    params={};wrapper_checks={}
    with torch.no_grad():
        for family in replication.FAMILIES:
            model=common.make_model(family,17)
            output=model(synthetic)
            assert output.shape==(2,20) and torch.isfinite(output).all()
            params[family]=sum(p.numel() for p in model.parameters())
            if family in ['allocation_constant','allocation_readout']:
                assert torch.equal(output,model(changed))
                torch.manual_seed(17)
                old=ShockAllocationForecaster(n_exog=3,horizon=20,hidden=16,modes=2,
                    variant='constant_allocation' if family=='allocation_constant' else 'readout_gate')
                assert torch.equal(output,old(synthetic[:,:,:4]))
                wrapper_checks[family]={'ignores_added_volume_exactly':True,'equals_original_fixed_model_exactly':True}
    assert source_hashes=={p.name:sha(p) for p in files}
    receipt={'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'status':'PASS_NO_MARKET_FIT_PREFLIGHT','market_fits':0,'market_forecasts_scored':0,
        'source_sha256':source_hashes,'audit_script_sha256':sha(__file__),
        'independent_parts':['Python csv/float date and quantity identities','bisect backward energy alignment','scalar past-only volume context','fixed split and label maturity masks','scalar RMS and target variance','all input windows and target arrays'],
        'reused_parts':['Production implementation called only for comparisons','Existing model forwards on synthetic tensors; no optimizer'],
        'data_checks':comparisons,'worker_hooks':workers,'parameters':params,'wrapper_checks':wrapper_checks,
        'execution_plan':{'executor_jobs':27,'summary_rows':39,'neural_train_fit_calls':120,
                          'selection_calls':48,'refit_calls':72,'classical_fit_calls':99},
        'limitations':['Price is the author-cleaned daily transaction mean, not close.',
                       'Author month-based cleaning and current historical EIA vintages do not establish live-time availability.',
                       'Current reported price and volume are assumed observed at issue time.',
                       'CQ 28 common origins are a separate small-sample stress cohort.',
                       'Earlier source QC and parsing are disclosed separately from model selection or scoring.',
                       'No claim about forecast gains follows from this preflight.']}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'status':receipt['status'],'markets':len(comparisons),'worker_methods':list(workers),
                     'max_input_difference':max(x['maximum_input_difference'] for x in comparisons),
                     'market_fits':0,'out':str(args.out)}))


if __name__=='__main__':main()
