"""Bounded fresh fits. All configurations, checkpoints and traces are retained."""
from __future__ import annotations
import os
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_v]='1'
import argparse, copy, hashlib, json, platform, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
import core

ROOT=Path(__file__).resolve().parent
MAX_EPOCHS=160; PATIENCE=20

def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def fit(p,ref,end,mode,seed):
    ids,x,y,tr,va,sd=core.support(p,end)
    args=core.features(p,ref,ids,sd)
    const=float(args[2][tr].mean())
    m=core.make_model(mode,seed,const)
    opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.001)
    tx=tuple(a[tr] for a in args); vx=tuple(a[va] for a in args)
    ty=torch.tensor(y[tr]/sd);vy=torch.tensor(y[va]/sd)
    with torch.no_grad():best=float((m(*vx)[0]-vy).square().mean())
    initial_val=best;state=copy.deepcopy(m.state_dict());selected=0;wait=0;trace=[]
    for epoch in range(1,MAX_EPOCHS+1):
        opt.zero_grad();pred,corr=m(*tx)
        mse=(pred-ty).square().mean();reg=corr.square().mean() if mode!='GRU' else mse*0
        loss=mse+reg;loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.))
        opt.step();m.project()
        with torch.no_grad():v=float((m(*vx)[0]-vy).square().mean())
        trace.append([epoch,float(mse.detach()),float(reg.detach()),float(loss.detach()),v,gn])
        if not np.isfinite(trace[-1]).all():raise FloatingPointError(f'nonfinite {mode} {seed} {end}')
        if v<best-1e-6:
            best=v;selected=epoch;state=copy.deepcopy(m.state_dict());wait=0
        else:wait+=1
        if wait>=PATIENCE:break
    m.load_state_dict(state)
    rec={'mode':mode,'seed':seed,'fit_end':end,'train_origins':len(tr),'validation_origins':len(va),
         'first_train_origin':int(ids[tr[0]]),'last_train_origin':int(ids[tr[-1]]),
         'max_train_target':int(ids[tr[-1]]+20),'first_validation_origin':int(ids[va[0]]),
         'max_validation_target':int(ids[-1]+20),'scale':sd,'constant_variance':const,
         'parameters':sum(p.numel() for p in m.parameters()),'selected_epoch':selected,
         'epochs':len(trace),'initial_validation':initial_val,'best_validation':best,
         'trace_columns':['epoch','mse','correction_penalty','objective','validation','preclip_gradient_norm'],
         'trace':trace,'dtype':'float64'}
    if isinstance(m,core.MomentGRU):
        with torch.no_grad():
            aa=m.a.detach().numpy();vv=args[2][tr].numpy()
            attenuation=np.exp(-.5*vv[:,:,None]*aa[None,None,:]**2)
        rec['mechanism']={'a_norm':float(np.linalg.norm(aa)),
                          'expected_attenuation_min':float(attenuation.min()),
                          'expected_attenuation_mean':float(attenuation.mean())}
        if mode=='FREE_DAMP':rec['mechanism']['untied_distance']=float(torch.linalg.norm(m.a-m.damp))
    return m,sd,rec

_GLOBAL={}
def init_worker(p,ref,out):
    core.configure();_GLOBAL.update(p=p,ref=ref,out=Path(out))

def fit_job(task):
    mode,seed,block=task;p=_GLOBAL['p'];ref=_GLOBAL['ref'];out=_GLOBAL['out'];end=459+40*block
    stamp=time.perf_counter();m,sd,rec=fit(p,ref,end,mode,seed)
    ids=np.arange(end,min(end+40,930));pr=core.forecast(m,sd,p,ref,ids)
    rec.update(block=block,seconds=time.perf_counter()-stamp,issue_first=int(ids[0]),issue_last=int(ids[-1]))
    key=f'{mode}_s{seed}_b{block:02}'
    state={f'parameter__{k}':v.detach().numpy() for k,v in m.state_dict().items()}
    np.savez_compressed(out/'fits'/f'{key}.npz',**state,predictions=pr,origins=ids,scale=np.array(sd))
    write_json(out/'logs'/f'{key}.json',rec)
    return rec

def fit_baselines(p,dates,ref,out):
    ids=np.arange(459,930);preds={'RW':np.zeros((1,len(ids),20))};receipts=[]
    for w in (120,240,480):
        rr=core.reference(p,w);preds[f'ML_w{w}']=rr['b'][ids][None]
        drift=np.array([(p[t]-p[max(0,t-w)])/min(t,w)*np.arange(1,21) for t in ids])
        preds[f'DRIFT_w{w}']=drift[None]
    specs=[('RIDGE',l,a) for l in (5,20) for a in (.01,.1,1.,10.)]
    specs += [('TREE',l,r) for l in (7,15) for r in (.05,.1)]
    for family,l,a in specs:
        key=f'{family}_{l}_{a}';arr=np.zeros((len(ids),20))
        for block in range(12):
            end=459+40*block;oi,x,y,tr,va,sd=core.support(p,end)
            pi=ids[(ids>=end)&(ids<end+40)];sel=pi-459
            train=x[tr,-(l if family=='RIDGE' else 20):]/sd
            xx=np.array([np.diff(p[t-60:t+1]) for t in pi])[:,-train.shape[1]:]/sd
            st=time.perf_counter()
            for h in core.HORIZONS:
                target=y[tr,h-1]/sd
                if family=='RIDGE':model=Ridge(alpha=a*len(tr),fit_intercept=True,solver='cholesky')
                else:model=HistGradientBoostingRegressor(max_leaf_nodes=l,learning_rate=a,max_iter=80,
                                   early_stopping=False,l2_regularization=1.,random_state=0)
                model.fit(train,target);arr[sel,h-1]=model.predict(xx)*sd
            receipts.append({'kind':key,'block':block,'fit_end':end,'n_train':len(tr),
                         'max_train_target':int(oi[tr[-1]]+20),'seconds':time.perf_counter()-st})
        preds[key]=arr[None]
    write_json(out/'BASELINE_FITS.json',receipts)
    return preds

def combine(p,ref,gru):
    """Mature issued errors only; coefficients are existing OLS forecast weights."""
    ids=np.arange(459,930);b=ref['b'][ids];fixed=.5*(gru+b[None]);dynamic=gru.copy();ws=np.full_like(gru,.5)
    for seed in range(len(gru)):
        for ix,t in enumerate(ids):
            for h in range(1,21):
                old=ids[(ids+h<=t)][-120:]
                if len(old)<30:w=.5
                else:
                    y=p[old+h]-p[old];eg=gru[seed,old-459,h-1]-y
                    d=b[old-459,h-1]-gru[seed,old-459,h-1]
                    den=float(d@d);w=float(np.clip(-(eg@d)/den,0,1)) if den>1e-18 else .5
                ws[seed,ix,h-1]=w;dynamic[seed,ix,h-1]=w*b[ix,h-1]+(1-w)*gru[seed,ix,h-1]
    return fixed,dynamic,ws

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--data',required=True);pa.add_argument('--out',required=True)
    pa.add_argument('--workers',type=int,default=3);args=pa.parse_args()
    core.configure();out=Path(args.out)
    if out.exists():raise FileExistsError('Use a new output directory; first results are immutable')
    (out/'fits').mkdir(parents=True);(out/'logs').mkdir()
    p,dates,audit=core.load_prefix(args.data);ref=core.reference(p)
    write_json(out/'INPUT_RECEIPT.json',audit)
    source={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob('*.py')}
    write_json(out/'SOURCE_HASHES.json',source)
    import scipy,sklearn
    write_json(out/'ENVIRONMENT.json',{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,
            'torch':torch.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,
            'dtype':'float64','workers':args.workers,'mkldnn':False,'threads':1})
    np.savez_compressed(out/'reference.npz',**ref)
    issued=np.arange(459,930);truth=np.array([p[t+1:t+21]-p[t] for t in issued])
    scored=np.array([dates[t][:4]==dates[t+20][:4] for t in issued])
    np.savez_compressed(out/'evaluation.npz',origins=issued,truth=truth,score_mask=scored,years=np.array([d[:4] for d in dates[issued]]))
    tasks=[(m,s,b) for m in core.MODES for s in core.SEEDS for b in range(12)]
    st=time.perf_counter();logs=[]
    write_json(out/'STATUS.json',{'stage':'training','completed':0,'planned':len(tasks)})
    try:
        with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,initargs=(p,ref,str(out))) as ex:
            fs=[ex.submit(fit_job,t) for t in tasks]
            for f in as_completed(fs):
                rec=f.result();logs.append(rec)
                write_json(out/'STATUS.json',{'stage':'training','completed':len(logs),'planned':len(tasks),
                                            'elapsed':time.perf_counter()-st})
                print(rec['mode'],rec['seed'],rec['block'],rec['epochs'],rec['selected_epoch'],flush=True)
    except BaseException as err:
        write_json(out/'FAILURE.json',{'type':type(err).__name__,'message':str(err),'completed':len(logs)})
        raise
    neural_seconds=time.perf_counter()-st;pred={}
    for mode in core.MODES:
        mat=np.full((len(core.SEEDS),len(issued),20),np.nan)
        for si,seed in enumerate(core.SEEDS):
            for block in range(12):
                with np.load(out/'fits'/f'{mode}_s{seed}_b{block:02}.npz',allow_pickle=False) as z:
                    mat[si,z['origins']-459]=z['predictions']
        if not np.isfinite(mat).all():raise ValueError('incomplete forecasts')
        pred[mode]=mat
    st=time.perf_counter();pred.update(fit_baselines(p,dates,ref,out));baseline_seconds=time.perf_counter()-st
    fixed,dynamic,weights=combine(p,ref,pred['GRU']);pred['CORE_GRU_FIXED']=fixed;pred['CORE_GRU_MATURE']=dynamic
    np.savez_compressed(out/'combination_weights.npz',weights=weights)
    np.savez_compressed(out/'predictions.npz',**pred)
    short=[{k:v for k,v in rec.items() if k!='trace'} for rec in logs]
    short=sorted(short,key=lambda r:(r['mode'],r['seed'],r['block']))
    write_json(out/'FIT_REGISTER.json',short)
    write_json(out/'TIMING.json',{'neural_wall_seconds':neural_seconds,'baseline_wall_seconds':baseline_seconds,
               'summed_neural_fit_seconds':sum(r['seconds'] for r in logs),'planned_neural_fits':288,'completed_neural_fits':len(logs)})
    write_json(out/'STATUS.json',{'stage':'fits_complete','completed':len(logs),'planned':len(tasks)})
    print('ALL_FITS_COMPLETE',flush=True)

if __name__=='__main__':main()
