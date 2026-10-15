"""Frozen fit loop, safe numeric outputs, complete epoch records, no hidden rescue."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import copy,json,time,hashlib,platform,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
import core

def fit(p,ref,end,mode,seed):
    core.configure();ids,x,y,tr,va,sd=core.support(p,end);m=core.make_model(mode,seed)
    tb=core.batch(p,ref,ids[tr],sd,mode);vb=core.batch(p,ref,ids[va],sd,mode)
    yt=torch.tensor(y[tr]/sd);yv=torch.tensor(y[va]/sd)
    prof=not mode.endswith('WINDOW');operator=torch.tensor(core.operator_array(len(tr),mode)) if prof else None
    target=yt-tb['base']
    params=list(m.feature.parameters()) if prof else list(m.parameters())
    opt=torch.optim.AdamW(params,lr=.003,weight_decay=.001)
    def refresh():
        if prof:
            with torch.no_grad():m.B.copy_(core.solve_head(m.encode(tb),target,operator))
    refresh()
    with torch.no_grad():best=float(((m(vb)-yv)**2).mean())
    initial={k:v.detach().clone() for k,v in m.state_dict().items()};state=copy.deepcopy(m.state_dict())
    selected=0;wait=0;history=[]
    if mode!='FROZEN':
        for epoch in range(1,81):
            opt.zero_grad(set_to_none=True)
            if prof:
                g=m.encode(tb)
                with torch.no_grad():b=core.solve_head(g.detach(),target,operator);m.B.copy_(b)
                pred=g@b.T;score=core.score_direction(pred,target,operator)
                # stop-gradient score: not the gradient of e'Qe when U is asymmetric
                surrogate=2*(pred*score.detach()).mean()
                surrogate.backward();train_mse=float(((pred.detach()-target)**2).mean());score_norm=float(score.detach().norm())
            else:
                pred=m(tb);err=pred-yt
                loss=(err**2).mean()
                if mode=='FULL_WINDOW':loss=loss+core.GAMMA*((pred-tb['base'])**2).mean()+core.ALPHA*(m.B**2).sum()/core.H
                loss.backward();train_mse=float((err.detach()**2).mean());score_norm=float(err.detach().norm())
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.));opt.step();refresh()
            with torch.no_grad():validation=float(((m(vb)-yv)**2).mean())
            row=[epoch,train_mse,validation,norm,score_norm]
            if not np.isfinite(row).all():raise FloatingPointError(f'nonfinite {mode}/{seed}/{end}')
            history.append(row)
            if validation<best-1e-6:
                best=validation;selected=epoch;state=copy.deepcopy(m.state_dict());wait=0
            else:wait+=1
            if wait>=12:break
    m.load_state_dict(state)
    row={'mode':mode,'seed':seed,'end':end,'n_train':len(tr),'n_validation':len(va),'train_max_origin':int(ids[tr[-1]]),
         'train_max_target':int(ids[tr[-1]]+20),'validation_first_origin':int(ids[va[0]]),'validation_max_target':int(ids[va[-1]]+20),
         'sd':sd,'parameters':sum(t.numel() for t in m.parameters()),'selected_epoch':selected,'epochs':len(history),
         'history_columns':['epoch','training_unweighted_MSE','validation_unweighted_MSE','preclip_direction_norm','score_norm'],
         'history':history,'initial_validation':0.,
         'minimum_validation':best,'note':'surrogate scalar is not logged as a predictive loss; head re-solved after every feature update'}
    # Retain the actual initial score without recomputing with a later state.
    old=copy.deepcopy(m.state_dict());m.load_state_dict(initial)
    with torch.no_grad():row['initial_validation']=float(((m(vb)-yv)**2).mean())
    m.load_state_dict(old)
    if prof:
        with torch.no_grad():
            gg=m.encode(tb);bb=core.solve_head(gg,target,operator)
            row['head_equation_prediction_difference']=float((gg@(bb-m.B).T).abs().max())
            ug=operator@gg;a=gg.T[None]@ug/len(tr)+core.GAMMA*(gg.T@gg)[None]/len(tr)+core.ALPHA*torch.eye(8)
            row['min_head_symmetric_eigenvalue']=float(torch.linalg.eigvalsh((a+a.transpose(-1,-2))/2).min())
    return m,sd,row

def worker(args):
    p,ref,out,mode,seed,block=args;out=Path(out);end=459+40*block
    orig=np.arange(end,min(end+40,930));key=f'{mode}_s{seed}_b{block:02}'
    start=time.perf_counter();m,sd,row=fit(p,ref,end,mode,seed)
    pr=core.predict(m,p,ref,orig,sd);row['seconds']=time.perf_counter()-start
    data={'origins':orig,'prediction':pr,'sd':np.array(sd),'selected_epoch':np.array(row['selected_epoch'])}
    data.update({'state__'+k:v.detach().numpy() for k,v in m.state_dict().items()})
    np.savez_compressed(out/'fits'/(key+'.npz'),**data)
    (out/'fits'/(key+'.json')).write_text(json.dumps(row,indent=2),encoding='utf8')
    print(key,row['epochs'],row['selected_epoch'],round(row['seconds'],3),flush=True)
    return {k:v for k,v in row.items() if k!='history'}

def run(path,out,workers=3):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a fresh output directory')
    out.mkdir(parents=True,exist_ok=True);(out/'fits').mkdir();(out/'predictions').mkdir()
    p,dates,raw=core.load_prefix(path);ref=core.reference(p);(out/'approved_prefix.csv').write_bytes(raw)
    np.savez_compressed(out/'reference.npz',**ref)
    origins=np.array([i for i in range(459,930) if dates[i][:4]==dates[i+20][:4]])
    y=np.stack([p[i+1:i+21]-p[i] for i in origins]);years=np.array([str(d)[:4] for d in dates[origins]])
    np.savez_compressed(out/'evaluation.npz',origins=origins,truth=y,years=years)
    env={'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'platform':platform.platform(),'workers':workers,'source':core.BASE_SHA,
         'protocol':core.PROTOCOL_SHA,'prefix_hash':hashlib.sha256(raw).hexdigest(),'max_date':str(dates[-1]),'dtype':'float64','new_prices':False,'later_period_read':False}
    (out/'ENVIRONMENT.json').write_text(json.dumps(env,indent=2));start=time.perf_counter();register=[]
    tasks=[(p,ref,str(out),mode,seed,block) for mode in core.MODES for seed in core.SEEDS for block in range(12)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(worker,x) for x in tasks]):
            register.append(future.result());(out/'progress.json').write_text(json.dumps({'completed':len(register),'total':len(tasks),'elapsed':time.perf_counter()-start}))
    for mode in core.MODES:
        prediction=np.empty((4,451,20))
        for s,seed in enumerate(core.SEEDS):
            issued=np.full((471,20),np.nan)
            for block in range(12):
                with np.load(out/'fits'/f'{mode}_s{seed}_b{block:02}.npz',allow_pickle=False) as z:issued[z['origins']-459]=z['prediction']
            if not np.isfinite(issued).all():raise ValueError('incomplete forecasts')
            prediction[s]=issued[origins-459]
        np.savez_compressed(out/'predictions'/(mode+'.npz'),prediction=prediction)
    register.sort(key=lambda z:(z['mode'],z['seed'],z['end']))
    (out/'FIT_REGISTER.json').write_text(json.dumps(register,indent=2))
    env.update({'formal_wall_seconds':time.perf_counter()-start,'fit_wall_seconds_sum':sum(z['seconds'] for z in register),'fit_records':len(register),
                'optimization_fit_records':sum(z['mode']!='FROZEN' for z in register)})
    (out/'ENVIRONMENT.json').write_text(json.dumps(env,indent=2))
    return out
