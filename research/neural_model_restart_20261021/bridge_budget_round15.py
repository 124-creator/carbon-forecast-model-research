"""Round 15: OHLC bridge-conditioned conservative risk-budget transport.

Development only: discard 2026 before all feature/target construction. Positive
22-step total risk is frozen; geometry can only move its allocation between
1, 2--5, and 6--22. A detailed-balance generator has the base allocation as its
stationary state at zero neural edge tilt, so the candidate nests the baseline.
No novelty or publication claim is implied by this implementation.
"""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_k]='1'
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import torch
from torch import nn
import range_clock_round14 as R

HERE=Path(__file__).resolve().parent
EPS=1e-12
MODES=['bridge_transport','flat_shape','context_transport']


def prepare():
    old,audit=R.build()
    raw=pd.read_csv(R.SOURCE,parse_dates=['date'])
    d=raw[raw.date<'2026-01-01'].copy().reset_index(drop=True)
    assert d.date.is_unique and d.date.is_monotonic_increasing
    r=np.log(d.close/d.open).to_numpy()
    u=np.log(d.high/d.open).to_numpy();l=np.log(d.low/d.open).to_numpy();q=u-l
    eta=np.divide(r,q,out=np.zeros(len(r)),where=q>1e-12)
    xi=np.divide(u+l-r,q,out=np.zeros(len(r)),where=q>1e-12)
    assert (np.abs(eta)+np.abs(xi)<=1+1e-6).all()
    shape=np.column_stack([eta,xi,eta*xi,eta**2,xi**2,np.abs(eta),np.abs(xi)])
    ids=np.flatnonzero(old['h']==22);t=old['t'][ids]
    rv=(np.log(d.high/d.low).to_numpy()**2)/(4*np.log(2))
    Y=np.column_stack([rv[t+1],np.array([rv[i+2:i+6].sum() for i in t]),
                       np.array([rv[i+6:i+23].sum() for i in t])])
    assert np.isfinite(Y).all() and (Y>=0).all()
    np.testing.assert_allclose(Y.sum(1)/22,old['y'][ids],rtol=2e-6,atol=1e-12)
    dates=d.date.to_numpy()[t];ends=d.date.to_numpy()[t+22]
    D=dict(t=t,date=dates,end=ends,x=old['x'][ids],shape=shape[t].astype(np.float32),Y=Y,
           train=(ends<np.datetime64('2025-01-01')),val=(dates>=np.datetime64('2025-01-01')))
    assert len(D['shape'])==len(D['x'])==len(D['Y'])==len(D['t'])
    assert pd.Timestamp(ends.max())<pd.Timestamp('2026-01-01')
    audit.update(target='3 future sums of Parkinson intraday range variance; bucket lengths 1/4/17',
                 all_horizons_common_origins=True,geometry='eta=r/(u-l), xi=(u+l-r)/(u-l)',
                 geometry_domain='abs(eta)+abs(xi)<=1',
                 background_information='the prior 12 inputs already include range and body returns; not strictly shape-free; extra wick geometry is conditional on that information',
                 total_risk='frozen sum of 3 common-input HGB forecasts')
    return D,audit


def fit_bucket_model(X,Y,train,pred):
    out=np.empty((int(pred.sum()),3))
    for k in range(3):
        scale=Y[train,k].mean()
        model=HistGradientBoostingRegressor(loss='poisson',max_iter=120,max_leaf_nodes=7,
            learning_rate=.04,min_samples_leaf=20,l2_regularization=1,early_stopping=False,random_state=11)
        model.fit(X[train],Y[train,k]/scale)
        out[:,k]=model.predict(X[pred])*scale
    assert (out>0).all() and np.isfinite(out).all()
    return out


def background(D):
    n=len(D['t']);B=np.full((n,3),np.nan);trace=[]
    for year in (2022,2023,2024,2025):
        start=np.datetime64(f'{year}-01-01');end=np.datetime64(f'{year+1}-01-01')
        train=D['end']<start
        pred=(D['date']>=start)&(D['date']<end)
        if year<2025:pred &= D['train']
        assert train.sum()>100 and pred.sum()>50
        B[pred]=fit_bucket_model(D['x'],D['Y'],train,pred)
        trace.append(dict(prediction_year=year,training_targets=int(train.sum()),
                          last_training_label=str(pd.Timestamp(D['end'][train].max()).date()),
                          predicted_origins=int(pred.sum())))
    tr=D['train']&np.isfinite(B).all(1);va=D['val']&np.isfinite(B).all(1)
    assert not (tr&va).any() and va.sum()>100
    return B,np.flatnonzero(tr),np.flatnonzero(va),trace


def generator_transport(w,delta):
    # Column convention: Q[i,j] is the flow j -> i. At delta=0,
    # w1*(w2)=w2*(w1) and w2*(w3)=w3*(w2), hence Q@w=0.
    d1,d2=delta.unbind(1)
    rate12=w[:,1]*torch.exp(d1);rate21=w[:,0]*torch.exp(-d1)
    rate23=w[:,2]*torch.exp(d2);rate32=w[:,1]*torch.exp(-d2)
    z=torch.zeros_like(d1)
    row0=torch.stack([-rate12,rate21,z],1)
    row1=torch.stack([rate12,-rate21-rate23,rate32],1)
    row2=torch.stack([z,rate23,-rate32],1)
    Q=torch.stack([row0,row1,row2],1)
    A=torch.matrix_exp(Q)
    return (A@w[:,:,None]).squeeze(2)


class ShapeTransport(nn.Module):
    def __init__(self,nf,mode,neutral):
        super().__init__();self.mode=mode
        self.body=nn.Sequential(nn.Linear(nf,12),nn.Tanh(),nn.Linear(12,8),nn.Tanh())
        self.head=nn.Linear(8,2)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        self.register_buffer('neutral',torch.tensor(neutral,dtype=torch.float32))
    def forward(self,x,w):
        zero=torch.cat([x[:,:-7],self.neutral[None,:].expand(len(x),-1)],1)
        if self.mode=='context_transport':
            delta=1.25*torch.tanh(self.head(self.body(zero)))
        else:
            actual=self.head(self.body(x));neutral=self.head(self.body(zero))
            delta=1.25*torch.tanh(actual-neutral)
        if self.mode=='flat_shape':
            adjustment=torch.cat([delta,torch.zeros((len(x),1),dtype=x.dtype)],1)
            out=torch.softmax(torch.log(w)+adjustment,dim=1)
        else:out=generator_transport(w,delta)
        return out,delta


def mean_risks(total,w):
    return torch.stack([total*w[:,0],total*(w[:,0]+w[:,1])/5,total/22],1)


def loss_rows(v,y):return torch.log(v.clamp_min(EPS))+y/v.clamp_min(EPS)


def fit_head(D,B,tr,va,mode,seed,epochs):
    X0=np.column_stack([D['x'],D['shape']])
    mean=X0[tr].mean(0);sd=X0[tr].std(0)+1e-5
    X=torch.tensor((X0-mean)/sd,dtype=torch.float32)
    total=torch.tensor(np.nansum(B,axis=1),dtype=torch.float32)
    w=torch.tensor(B/np.nansum(B,axis=1,keepdims=True),dtype=torch.float32)
    Y=torch.tensor(np.column_stack([D['Y'][:,0],D['Y'][:,:2].sum(1)/5,D['Y'].sum(1)/22]),dtype=torch.float32)
    basev=mean_risks(total,w)
    torch.manual_seed(seed)
    net=ShapeTransport(X.shape[1],mode,(-mean[-7:]/sd[-7:]).astype(np.float32))
    opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.002)
    with torch.no_grad():
        initial_w,_=net(X[va],w[va]);initial_v=mean_risks(total[va],initial_w)
        initial_score=float((loss_rows(initial_v[:,:2],Y[va,:2])-loss_rows(basev[va,:2],Y[va,:2])).mean())
    best=initial_score;state=copy.deepcopy(net.state_dict());bep=0;bad=0;hist=[[0,initial_score]]
    for ep in range(1,epochs+1):
        net.train();order=np.random.default_rng(seed+ep).permutation(tr)
        for b in np.array_split(order,max(1,int(np.ceil(len(order)/128)))):
            opt.zero_grad();ww,delta=net(X[b],w[b]);v=mean_risks(total[b],ww)
            loss=(loss_rows(v[:,:2],Y[b,:2])-loss_rows(basev[b,:2],Y[b,:2])).mean()
            loss=loss+.001*delta.pow(2).mean()
            assert torch.isfinite(loss)
            loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
        net.eval()
        with torch.no_grad():
            ww,_=net(X[va],w[va]);v=mean_risks(total[va],ww)
            val=float((loss_rows(v[:,:2],Y[va,:2])-loss_rows(basev[va,:2],Y[va,:2])).mean())
        hist.append([ep,val])
        if val<best-1e-5:best=val;state=copy.deepcopy(net.state_dict());bep=ep;bad=0
        else:bad+=1
        if bad>=15 and ep>=20:break
    net.load_state_dict(state);net.eval()
    with torch.no_grad():
        ww,delta=net(X[va],w[va]);v=mean_risks(total[va],ww)
    assert torch.all(ww>0) and torch.allclose(ww.sum(1),torch.ones(len(va)),atol=2e-6)
    np.testing.assert_allclose(v[:,2].numpy(),basev[va,2].numpy(),rtol=1e-6,atol=1e-12)
    return v.numpy(),dict(seed=seed,best_epoch=bep,epochs_run=len(hist),
        validation_short_loss_diff=best,parameters=sum(p.numel() for p in net.parameters()),
        mean_abs_edge_tilt=float(delta.abs().mean()),history=hist)


def sanity():
    w=torch.tensor([[.05,.15,.8],[.3,.3,.4]],dtype=torch.float64)
    z=torch.zeros(2,2,dtype=torch.float64,requires_grad=True)
    out=generator_transport(w,z)
    assert torch.allclose(out,w,atol=1e-12)
    tilted=generator_transport(w,torch.tensor([[.7,-.8],[-.5,.3]],dtype=torch.float64))
    assert (tilted>0).all() and torch.allclose(tilted.sum(1),torch.ones(2,dtype=torch.float64),atol=1e-12)
    assert torch.autograd.gradcheck(lambda dd:generator_transport(w,dd),(z,),eps=1e-6,atol=1e-5)
    return dict(zero_tilt_recovers_baseline=True,positive_conservative_operator=True,gradient_check=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--seeds',nargs='+',type=int,default=[11,22,33]);ap.add_argument('--tag',default='bridge_budget_round15_development')
    a=ap.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    D,audit=prepare();B,tr,va,trace=background(D)
    config=dict(config=vars(a),status='DEVELOPMENT ONLY; no 2026 targets or forecasts',audit=audit,
        source_sha256=hashlib.sha256(R.SOURCE.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checks=sanity(),
        background_trace=trace,head_training_origins=len(tr),validation_origins=len(va),
        selection='2025 mean standard-direction QLIKE difference, h=1/5 only; h22 fixed',
        models=MODES,fits={},scores={})
    (HERE/(a.tag+'_manifest.json')).write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    total=B[va].sum(1);w0=B[va]/total[:,None]
    pred={'level_allocator':mean_risks(torch.tensor(total),torch.tensor(w0)).numpy(),
          'uniform':np.repeat((total/22)[:,None],3,axis=1)}
    # Same full geometry information, flexible strong allocator, identical total.
    Xfull=np.column_stack([D['x'],D['shape']]);full=fit_bucket_model(Xfull,D['Y'],D['train'],D['val'])
    # va equals the complete common-maturity validation origins.
    assert np.array_equal(va,np.flatnonzero(D['val']))
    wf=full/full.sum(1,keepdims=True)
    pred['HGB_full_allocator']=mean_risks(torch.tensor(total),torch.tensor(wf)).numpy()
    start=time.time()
    for mode in MODES:
        pp=[];fits=[]
        for seed in a.seeds:
            v,fit=fit_head(D,B,tr,va,mode,seed,a.epochs);pp.append(v);fits.append(fit)
            print(mode,seed,'best',fit['best_epoch'],'val_delta',round(fit['validation_short_loss_diff'],7),flush=True)
        pred[mode]=np.mean(pp,axis=0).astype(np.float64)
        pred[mode][:,2]=total/22
        config['fits'][mode]=fits
    targets=np.column_stack([D['Y'][va,0],D['Y'][va,:2].sum(1)/5,D['Y'][va].sum(1)/22])
    for j,h in enumerate([1,5,22]):
        base=np.log(pred['level_allocator'][:,j])+targets[:,j]/pred['level_allocator'][:,j]
        config['scores'][str(h)]={}
        for name,v in pred.items():
            score=np.log(v[:,j])+targets[:,j]/v[:,j]
            config['scores'][str(h)][name]=dict(n=len(va),loss_difference_vs_level=float((score-base).mean()),
                MSE=float(np.mean((v[:,j]-targets[:,j])**2)))
        print('h',h,{name:round(v['loss_difference_vs_level'],7) for name,v in config['scores'][str(h)].items()},flush=True)
    config['elapsed_seconds']=time.time()-start
    (HERE/(a.tag+'.json')).write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(HERE/(a.tag+'_validation_predictions.npz'),date=D['date'][va],target=targets,**pred)
    print('DEVELOPMENT FINISHED; HOLDOUT UNUSED',round(time.time()-start,1),flush=True)


if __name__=='__main__':main()
