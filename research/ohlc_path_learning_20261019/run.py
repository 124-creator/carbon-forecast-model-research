"""Frozen chronological experiment runner; raw data/forecasts stay outside git.

This runner prints completion and validation results, never interim test scores.
Resume requires the identical code fingerprint and complete file receipts.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
import argparse, concurrent.futures, hashlib, json, time, copy
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from pipeline import OriginData,Prepared,load_source,folds_for,sha256,FIRST_ORIGIN,HORIZONS
from model import make_model,sobol_normal,objective

torch.set_num_threads(1)
NEURAL=('summary_state','state_price','state_marginal','state_step1','free_path','gru_price','gru_full')
CLASSICAL=('rw','drift','ridge_price','ridge_full','hgb_full')
SEEDS=(17,29,43)
LRS=(.001,.003)
MAX_EPOCHS=30
PATIENCE=5
BATCH=64
TRAIN_SAMPLES=32
EVAL_SAMPLES=128
ROOT=Path(__file__).resolve().parent

def write_json(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n'); tmp.replace(path)

def fingerprint():
    return hashlib.sha256(b''.join((ROOT/n).read_bytes() for n in ('model.py','pipeline.py','run.py'))).hexdigest()

def seed_all(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)

def forward(model,a,eps):
    return model(a['x'],a['context'],a['p0'],a['scale'],a['basis0'],a['radius0'],eps)

def ensure_finite(out):
    if 'clip_count' in out and int(out['clip_count'])!=0:
        raise RuntimeError('Numerical exponential guard activated; analytical mean no longer matches samples')
    if not torch.isfinite(out['mean']).all(): raise RuntimeError('Nonfinite mean')
    if out['samples'] is not None and not torch.isfinite(out['samples']).all(): raise RuntimeError('Nonfinite samples')

def validation(model,a,den):
    model.eval()
    with torch.no_grad():
        out=forward(model,a,sobol_normal(32,99173)); ensure_finite(out)
        return float((((out['mean']-a['y'][:,:,0])/den[:,0])**2).mean())

def fit_neural(family,prepared,train,val,seed,lr,epochs,select):
    seed_all(seed); t0=time.perf_counter(); model=make_model(family)
    tr=prepared.tensors(train); va=prepared.tensors(val) if len(val) else None
    den=torch.as_tensor(prepared.denominator,dtype=torch.float64)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    best_epoch=0; best_state=copy.deepcopy(model.state_dict())
    best_val=validation(model,va,den) if select else None
    history=[]; stale=0; steps=0
    generator=np.random.default_rng(seed+881)
    for epoch in range(1,epochs+1):
        model.train(); eps=sobol_normal(TRAIN_SAMPLES,70000+epoch)
        order=generator.permutation(len(train)); total=0.; totalm=0.; totale=0.
        for st in range(0,len(order),BATCH):
            ix=order[st:st+BATCH]; a={k:v[ix] for k,v in tr.items()}
            opt.zero_grad(set_to_none=True); out=forward(model,a,eps); ensure_finite(out)
            loss,detail=objective(out,a['y'],a['valid'],den,family)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            opt.step(); steps+=1
            total+=float(loss.detach())*len(ix);totalm+=detail['mse']*len(ix);totale+=detail['energy']*len(ix)
        entry={'epoch':epoch,'train_loss':total/len(train),'train_mse':totalm/len(train),'train_energy':totale/len(train)}
        if select:
            score=validation(model,va,den);entry['validation_mse']=score
            if score<best_val-1e-8:
                best_val=score;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict());stale=0
            else:stale+=1
        history.append(entry)
        if select and stale>=PATIENCE:break
    if select:model.load_state_dict(best_state)
    else:best_epoch=epochs
    return model,{'seed':seed,'learning_rate':lr,'selected_epoch':best_epoch,'epochs_executed':len(history),
                  'optimizer_steps':steps,'validation_mse':best_val,'history':history,
                  'parameters':sum(p.numel() for p in model.parameters()),'elapsed_seconds':time.perf_counter()-t0}

def predict(model,prepared,ix):
    model.eval(); a=prepared.tensors(ix); eps=sobol_normal(EVAL_SAMPLES,908113)
    means=[]; samples=[]
    with torch.no_grad():
        for st in range(0,len(ix),64):
            batch={k:v[st:st+64] for k,v in a.items()}; out=forward(model,batch,eps); ensure_finite(out)
            means.append(out['mean'].numpy())
            if out['samples'] is not None:samples.append(out['samples'].numpy())
    return np.concatenate(means),np.concatenate(samples) if samples else np.empty((len(ix),0,3,3))

def save_forecast(path,prepared,ix,mean,samples,metadata):
    d=prepared.data; path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,origin=d.origin[ix],target_index=d.origin[ix,None]+np.array(HORIZONS)[None],
        truth=d.y[ix],valid=d.valid[ix],p0=d.p0[ix],mean=mean,samples=samples,
        denominator=prepared.denominator,date=d.dates[ix])
    metadata['forecast_sha256']=sha256(path);metadata['forecast_rows']=len(ix)
    metadata['negative_point_predictions']=int((mean<=0).sum())
    write_json(path.with_suffix('.json'),metadata)

def fold_setup(data,a,b):
    val_start=a-120;val_end=a-20;train_end=val_start-20
    tr=data.positions(FIRST_ORIGIN,train_end);va=data.positions(val_start,val_end)
    final=data.positions(FIRST_ORIGIN,a-20);te=data.positions(a,b)
    return tr,va,final,te,Prepared.fit(data,tr),Prepared.fit(data,final)

def base_metadata(source,fold,family,seed,a,b,data_sha,code_sha):
    return {'source':source,'fold':fold,'family':family,'seed':seed,'test_start':a,'test_end':b,
       'train_origin_start':FIRST_ORIGIN,'train_origin_end_exclusive':a-20,'train_max_label_index':a-1,
       'validation_origin_start':a-120,'validation_origin_end_exclusive':a-20,
       'selection_train_origin_end_exclusive':a-140,'selection_train_max_label_index':a-121,
       'data_sha256':data_sha,'code_fingerprint':code_sha,'horizons':list(HORIZONS),
       'labels_count_observed_records':True,'numerical_clip_count':0}

def neural_task(args):
    source,fold,a,b,family,data_dir,out_dir,code_sha=args
    path=Path(data_dir)/f'{source}.csv'; data=OriginData(load_source(path,source))
    target=Path(out_dir)/source/fold/family
    done=target/'complete.json'
    if done.exists():
        old=json.loads(done.read_text());assert old['code_fingerprint']==code_sha
        assert old['data_sha256']==sha256(path), 'Refuse stale data resume'
        for seed in SEEDS:
            f=target/f'seed_{seed}.npz';m=json.loads(f.with_suffix('.json').read_text());assert sha256(f)==m['forecast_sha256']
            assert sha256(target/f'seed_{seed}.pt')==m['model_sha256'], 'Checkpoint receipt mismatch'
        return old
    tr,va,final,te,sel,prep=fold_setup(data,a,b);data_sha=sha256(path)
    trials=[]
    for lr in LRS:
        _,receipt=fit_neural(family,sel,tr,va,11,lr,MAX_EPOCHS,True);trials.append(receipt)
    best=min(trials,key=lambda q:(q['validation_mse'],q['selected_epoch'],q['learning_rate']))
    target.mkdir(parents=True,exist_ok=True)
    final_receipts=[]
    for seed in SEEDS:
        model,receipt=fit_neural(family,prep,final,np.array([],int),seed,best['learning_rate'],best['selected_epoch'],False)
        model_path=target/f'seed_{seed}.pt'
        torch.save({'model_state':model.state_dict(),'family':family,'preprocessing':prep.state(),
                    'data_sha256':data_sha,'code_fingerprint':code_sha},model_path)
        mean,samples=predict(model,prep,te)
        metadata=base_metadata(source,fold,family,seed,a,b,data_sha,code_sha)
        metadata.update({'selected_learning_rate':best['learning_rate'],'selected_epoch':best['selected_epoch'],
            'validation_mse':best['validation_mse'],'model_sha256':sha256(model_path),'receipt':receipt})
        save_forecast(target/f'seed_{seed}.npz',prep,te,mean,samples,metadata)
        final_receipts.append(receipt)
    result={'source':source,'fold':fold,'family':family,'code_fingerprint':code_sha,'data_sha256':data_sha,
        'validation_mse':best['validation_mse'],'selected_learning_rate':best['learning_rate'],
        'selected_epoch':best['selected_epoch'],'selection_fits':trials,'final_fits':final_receipts}
    write_json(done,result);return result

def classical_task(args):
    source,fold,a,b,family,data_dir,out_dir,code_sha=args
    path=Path(data_dir)/f'{source}.csv';data=OriginData(load_source(path,source))
    tr,va,final,te,sel,prep=fold_setup(data,a,b);data_sha=sha256(path)
    target=Path(out_dir)/source/fold/family;done=target/'complete.json'
    if done.exists():
        old=json.loads(done.read_text());assert old['code_fingerprint']==code_sha
        assert old['data_sha256']==data_sha, 'Refuse stale data resume'
        m=json.loads((target/'seed_0.json').read_text());assert sha256(target/'seed_0.npz')==m['forecast_sha256']
        return old
    t0=time.perf_counter();trials=[]
    def fit_predict(prep,train,predict_ix,param):
        x=prep.classical(train,family=='ridge_price');xp=prep.classical(predict_ix,family=='ridge_price')
        yt=(data.y[train,:,0]-data.p0[train,None])/prep.denominator[:,0]
        if family.startswith('ridge'):
            model=Ridge(alpha=param,solver='svd');model.fit(x,yt);yp=model.predict(xp)
        else:
            yp=[]
            for h in range(3):
                model=HistGradientBoostingRegressor(max_iter=100,learning_rate=.05,max_leaf_nodes=param,
                    l2_regularization=10.,min_samples_leaf=20,early_stopping=False,random_state=11)
                model.fit(x,yt[:,h]);yp.append(model.predict(xp))
            yp=np.stack(yp,1)
        return data.p0[predict_ix,None]+yp*prep.denominator[:,0]
    if family in ('rw','drift'):
        def simple(ix,train_end):
            drift=0. if family=='rw' else float(np.diff(data.frame.price.to_numpy(float)[:train_end]).mean())
            return data.p0[ix,None]+np.array(HORIZONS)[None]*drift
        vp=simple(va,a-120);mean=simple(te,a)
        best_param=None
        val_score=float((((vp-data.y[va,:,0])/sel.denominator[:,0])**2).mean())
    else:
        for param in ((.1,1.,10.,100.) if family.startswith('ridge') else (7,15)):
            vp=fit_predict(sel,tr,va,param)
            score=float((((vp-data.y[va,:,0])/sel.denominator[:,0])**2).mean())
            trials.append({'parameter':param,'validation_mse':score})
        best=min(trials,key=lambda q:q['validation_mse']);best_param=best['parameter'];val_score=best['validation_mse']
        mean=fit_predict(prep,final,te,best_param)
    metadata=base_metadata(source,fold,family,0,a,b,data_sha,code_sha)
    metadata.update({'validation_mse':val_score,'selected_parameter':best_param,'model_sha256':None})
    save_forecast(target/'seed_0.npz',prep,te,mean,np.empty((len(te),0,3,3)),metadata)
    result={'source':source,'fold':fold,'family':family,'code_fingerprint':code_sha,'data_sha256':data_sha,
        'validation_mse':val_score,'selected_parameter':best_param,'selection_fits':trials,
        'final_fit_count':3 if family=='hgb_full' else (1 if family.startswith('ridge') else 0),
        'elapsed_seconds':time.perf_counter()-t0}
    write_json(done,result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',default='/tmp/carbon-path-local');p.add_argument('--out',default='/tmp/carbon-path-run')
    p.add_argument('--workers',type=int,default=3);p.add_argument('--manifest-only',action='store_true');args=p.parse_args()
    code_sha=fingerprint();plan=[];jobs=[]
    for source in ('quantum','ding'):
        d=load_source(Path(args.data_dir)/f'{source}.csv',source)
        for fold,a,b in folds_for(d,source):
            item={'source':source,'fold':fold,'test_start':a,'test_end':b,'test_origins':b-a,
                'selection_train_origins':a-140-FIRST_ORIGIN,'validation_origins':100,'final_train_origins':a-20-FIRST_ORIGIN,
                'data_sha256':sha256(Path(args.data_dir)/f'{source}.csv')}
            plan.append(item)
            for family in NEURAL+CLASSICAL:jobs.append((source,fold,a,b,family,args.data_dir,args.out,code_sha))
    write_json(Path(args.out)/'plan.json',{'code_fingerprint':code_sha,'folds':plan,'neural_families':NEURAL,'classical_families':CLASSICAL,
        'seeds':SEEDS,'learning_rates':LRS,'maximum_epochs':MAX_EPOCHS,'patience':PATIENCE,'train_samples':TRAIN_SAMPLES,
        'evaluation_samples':EVAL_SAMPLES,'expected_neural_selection_fits':84,'expected_neural_final_fits':126})
    if args.manifest_only:
        print(json.dumps({'code_fingerprint':code_sha,'folds':plan},indent=2));return
    results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(neural_task if job[4] in NEURAL else classical_task,job):job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            result=future.result();results.append(result)
            print(json.dumps({'completed_tasks':len(results),'total_tasks':len(jobs),'source':result['source'],'fold':result['fold'],
                              'family':result['family'],'selected_epoch':result.get('selected_epoch')},ensure_ascii=False),flush=True)
    for item in plan:
        selected={r['family']:r['validation_mse'] for r in results if r['source']==item['source'] and r['fold']==item['fold'] and r['family'] in ('gru_full','ridge_full','hgb_full')}
        item.update({'validation_selected_baseline':min(selected,key=lambda k:(selected[k],k)),'validation_baseline_scores':selected,
                     'code_fingerprint':code_sha})
        write_json(Path(args.out)/item['source']/item['fold']/'manifest.json',item)
    write_json(Path(args.out)/'all_fit_receipts.json',results)
    print(json.dumps({'status':'COMPLETE','tasks':len(results),'code_fingerprint':code_sha}),flush=True)

if __name__=='__main__':main()
