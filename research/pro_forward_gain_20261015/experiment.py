"""Fresh finite protocol; all issued forecasts, decoder states and fit traces saved."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import copy,json,sys,time,hashlib,platform,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from torch import nn
import core


def jwrite(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def fit(p,ref,end,mode,seed):
    core.configure();ids,x,y,tr,va,sd=core.support(p,end)
    m=core.make_model(mode,seed);args=core.batch(p,ref,ids,sd)
    tx,tq,tb=args;ty=torch.as_tensor(y/sd);res=ty[tr]-tb[tr]
    is_profile=mode in core.PROFILE_MODES
    init_info=m.install(tx[tr],tq[tr],res) if is_profile else {}
    with torch.no_grad():initial=float((m(tx[va],tq[va],tb[va])-ty[va]).square().mean())
    state=copy.deepcopy(m.state_dict());best=initial;selected=0;wait=0;trace=[]
    opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.001)
    for epoch in range(0 if mode=='FROZEN_GAIN' else 160):
        opt.zero_grad()
        if is_profile:
            z=m.features(tx[tr],tq[tr]);loss,_,extra=core.system(z,res,mode,need_projector=False)
            rank=extra['rank'];gain_top=float(extra['eigenvalues'][-1])
        else:
            pred,info=m(tx[tr],tq[tr],tb[tr],details=True)
            loss=(pred-ty[tr]).square().mean()
            if mode!='GRU':loss=loss+info['correction'].square().mean()
            rank=-1;gain_top=0.
        loss.backward();grad=float(nn.utils.clip_grad_norm_(m.parameters(),1.));opt.step()
        post=m.install(tx[tr],tq[tr],res) if is_profile else {}
        with torch.no_grad():val=float((m(tx[va],tq[va],tb[va])-ty[va]).square().mean())
        row=[epoch+1,float(loss.detach()),val,grad,rank,gain_top,float(post.get('condition',0.))]
        if not np.isfinite(row).all():raise FloatingPointError('nonfinite; no rescue tuning')
        trace.append(row)
        if val<best-1e-6:best=val;state=copy.deepcopy(m.state_dict());selected=epoch+1;wait=0
        else:wait+=1
        if wait>=20:break
    m.load_state_dict(state)
    if is_profile:
        with torch.no_grad():
            z=m.features(tx[tr],tq[tr]);loss,pm,info=core.system(z,res,mode)
            if not torch.allclose(pm,m.P,atol=1e-12,rtol=1e-12):raise ValueError('stored projection mismatch')
            final=dict(rank=int(torch.linalg.matrix_rank(m.P,atol=1e-8)),eigenvalues=info['eigenvalues'].tolist(),condition=float(torch.linalg.cond(2*z.T@z/len(z)+core.RIDGE*torch.eye(9))))
    else:final={}
    pairs=core.folds(len(tr))
    receipt=dict(mode=mode,seed=seed,end=end,train_n=len(tr),validation_n=len(va),scale=sd,
      first_origin=int(ids[0]),max_training_target=int(ids[tr[-1]]+20),first_validation_origin=int(ids[va[0]]),max_validation_target=int(ids[-1]+20),
      scale_max_observation=int(ids[tr[pairs[0][0][-1]]]),
      folds=[dict(support_n=len(a),query_n=len(b),support_max_target=int(ids[tr[a[-1]]]+20),query_min_origin=int(ids[tr[b[0]]])) for a,b in pairs],
      initial_validation=initial,best_validation=best,selected_epoch=selected,epochs=len(trace),
      parameter_count=sum(v.numel() for v in m.parameters()),decoder_numeric_count=580 if is_profile else 0,
      trace_columns=['epoch','objective','validation_mse','gradient_norm_preclip','rank_preupdate','top_gain_eigenvalue_preupdate','ridge_condition_postupdate'],trace=trace,final=final)
    return m,sd,receipt


def predict(m,sd,p,ref,orig):
    with torch.no_grad():pred,info=m(*core.batch(p,ref,orig,sd),details=True)
    return pred.numpy()*sd,{k:v.detach().numpy() for k,v in info.items()}


def trajectory(job):
    mode,seed,out=job;out=Path(out);p,dates,_=core.load_prefix(out/'approved_prefix.csv');ref=core.reference(p)
    ori=np.arange(459,930);pr=np.full((len(ori),20),np.nan);start=time.perf_counter()
    for block in range(12):
        end=459+40*block;idx=np.flatnonzero((ori-459)//40==block);t=time.perf_counter()
        m,sd,log=fit(p,ref,end,mode,seed);vals,detail=predict(m,sd,p,ref,ori[idx]);pr[idx]=vals
        log.update(block=block,seconds=time.perf_counter()-t)
        name=f'{mode}_s{seed}_b{block:02d}'
        saved={f'param__{k}':v.detach().numpy() for k,v in m.state_dict().items()}
        saved.update(prediction=vals,origins=ori[idx],scale=np.array(sd));saved.update({f'detail__{k}':v for k,v in detail.items()})
        np.savez_compressed(out/'fits'/f'{name}.npz',**saved);jwrite(out/'fits'/f'{name}.json',log)
        print(f'{mode} seed={seed} block={block} epochs={log["epochs"]} chosen={log["selected_epoch"]}',flush=True)
    if not np.isfinite(pr).all():raise ValueError('incomplete grid')
    np.savez_compressed(out/'trajectories'/f'{mode}_{seed}.npz',prediction=pr)
    return dict(mode=mode,seed=seed,fits=12,optimized_fits=0 if mode=='FROZEN_GAIN' else 12,seconds=time.perf_counter()-start)


def run(out,prefix,workers=3):
    core.configure();out=Path(out)
    if out.exists():raise FileExistsError('New output directory required')
    out.mkdir(parents=True);[(out/k).mkdir() for k in ('fits','predictions','trajectories')]
    p,dates,receipt=core.load_prefix(prefix)
    with Path(prefix).open('rb',buffering=0) as f:(out/'approved_prefix.csv').write_bytes(f.read(29204))
    jwrite(out/'DATA_RECEIPT.json',receipt)
    ori=np.arange(459,930);years=dates[ori].astype('U4');eligible=years==dates[ori+20].astype('U4')
    truth=np.stack([p[o+1:o+21]-p[o] for o in ori])
    np.savez_compressed(out/'EVALUATION.npz',origins=ori,truth=truth,years=years,eligible=eligible)
    import scipy,sklearn
    jwrite(out/'ENVIRONMENT.json',dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,workers=workers))
    jwrite(out/'SOURCE_HASHES.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')})
    records=[];t=time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs=[pool.submit(trajectory,(mode,s,str(out))) for mode in core.MODES for s in core.SEEDS]
            for job in as_completed(jobs):records.append(job.result());jwrite(out/'FIT_RUN_REGISTER.json',records)
    except Exception:
        (out/'FAILURE.txt').write_text(traceback.format_exc());raise
    for mode in core.MODES:
        pp=[]
        for seed in core.SEEDS:
            with np.load(out/'trajectories'/f'{mode}_{seed}.npz',allow_pickle=False) as z:pp.append(z['prediction'].copy())
        np.savez_compressed(out/'predictions'/f'{mode}.npz',prediction=np.stack(pp))
    jwrite(out/'TRAINING_RUN.json',dict(complete=True,optimized_neural_fits=sum(r['optimized_fits'] for r in records),decoder_only_fits=48,total_fit_tasks=sum(r['fits'] for r in records),seconds=time.perf_counter()-t))
