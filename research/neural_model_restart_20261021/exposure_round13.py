"""Round 13: latent calendar exposure versus observed event-count exposure.

Exploratory candidate, not a novelty claim. Reused historical tests are explicitly
labelled. All training/validation/refit targets must mature before the boundary.
The closed-form CRPS and quantile inversion use exactly the same distributions.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_k]='1'
import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.ensemble import RandomForestRegressor
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
PANEL=HERE.parent/'fullcycle_h20_20261020'/'PANEL_5MARKET.csv'
CUT={'CEA':'2024-01-01','GUANGZHOU':'2024-01-01','SHANGHAI':'2024-01-01',
     'HUBEI':'2023-01-01','EU':'2023-01-01'}
HORIZONS=np.array([1,5,20])
TAUS=np.array([.025,.05,.10,.15,.20,.30,.40,.50,.60,.70,.80,.85,.90,.95,.975])
KMAX=20
MODES=['observation_gate','event_count','stale_clock','no_carry','hurdle']


def daily_features(d):
    lp=np.log(d.price.to_numpy(float)); r=np.diff(lp,prepend=lp[0]); n=len(r)
    activity=np.abs(r)>1e-12
    ages=np.zeros(n); ev8=np.zeros(n);ev32=np.zeros(n)
    # A causal event-time filter. Before the first event, no row is used for fitting.
    last=0;v8=0.;v32=0.
    for t in range(1,n):
        gap=t-last
        if activity[t]:
            z=r[t]**2/max(gap,1)
            v8=z if v8==0 else (1-2/9)*v8+(2/9)*z
            v32=z if v32==0 else (1-2/33)*v32+(2/33)*z
            last=t
        ages[t]=t-last;ev8[t]=v8;ev32[t]=v32
    s=pd.Series(r)
    rv250=s.pow(2).rolling(250).mean().to_numpy()
    rv5=s.pow(2).rolling(5).mean().to_numpy()
    rv22=s.pow(2).rolling(22).mean().to_numpy()
    act=pd.Series(activity.astype(float))
    p22=((act.rolling(22).sum()+1)/24).to_numpy()
    p250=((act.rolling(250).sum()+1)/252).to_numpy()
    scale=np.sqrt(np.maximum(rv250,1e-12))
    base=np.maximum((ev8+ev32)/2,1e-12)/np.maximum(rv250,1e-12)
    prior=np.clip(.8*p22+.2*p250,1e-4,1-1e-4)
    day=d.date.dt.dayofyear.to_numpy();ang=day/365.25*2*np.pi
    x=np.column_stack([np.log(np.maximum(rv5,1e-12)/np.maximum(rv250,1e-12)),
        np.log(np.maximum(rv22,1e-12)/np.maximum(rv250,1e-12)),
        np.log(np.maximum(ev8,1e-12)/np.maximum(rv250,1e-12)),
        np.log(np.maximum(ev32,1e-12)/np.maximum(rv250,1e-12)),
        np.log1p(ages),p22,p250,r/scale,np.log(scale),
        np.sin(ang),np.cos(ang)])
    return dict(lp=lp,x=x,age=ages,base=base,prior=prior,scale=scale)


def build(d,step=2):
    F=daily_features(d); n=len(d)
    cut=int(np.searchsorted(d.date.to_numpy(),np.datetime64(CUT[str(d.market.iloc[0])])) )
    val=max(350,int(cut*.75))
    rows=[]
    for hi,h in enumerate(HORIZONS):
        for t in range(250,n-int(h),step):
            split=0 if t+h<val else (1 if t>=val and t+h<cut else (2 if t>=cut else -1))
            if split>=0 and np.isfinite(F['x'][t]).all():rows.append((t,hi,split))
    rr=np.array(rows,dtype=int);t,hi,split=rr.T
    h=HORIZONS[hi];scale=F['scale'][t]*np.sqrt(h)
    D=dict(t=t,hi=hi,h=h,split=split,x=F['x'][t].astype(np.float32),
           age=F['age'][t].astype(np.float32),base=F['base'][t].astype(np.float32),
           prior=F['prior'][t].astype(np.float32),scale=scale,
           y=F['lp'][t+h]-F['lp'][t],cut=cut,val=val)
    D['z']=(D['y']/scale).astype(np.float32)
    local=[]
    for ti,hh in zip(t,h):
        hist=F['lp'][max(0,ti-250):ti+1]
        observed=hist[int(hh):]-hist[:-int(hh)]
        local.append((np.sum(np.abs(observed)>1e-12)+1)/(len(observed)+2))
    D['local_nonzero']=np.asarray(local)
    D['tr']=np.flatnonzero(split==0);D['va']=np.flatnonzero(split==1)
    D['rf']=np.flatnonzero(t+h<cut);D['te']=np.flatnonzero(split==2)
    assert (t[D['tr']]+h[D['tr']]<val).all()
    assert (t[D['va']]+h[D['va']]<cut).all()
    assert (t[D['rf']]+h[D['rf']]<cut).all()
    assert (t[D['te']]>=cut).all()
    assert len(D['tr'])>200 and len(D['va'])>30
    return D


class ExposureNet(nn.Module):
    def __init__(self,nf,mode):
        super().__init__();self.mode=mode
        self.body=nn.Sequential(nn.Linear(nf,12),nn.Tanh(),nn.Linear(12,8),nn.Tanh())
        self.head=nn.Linear(8,4)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def forward(self,x,base,prior,age,h):
        a=self.head(self.body(x))
        v=base*torch.exp(1.5*torch.tanh(a[:,0]))
        p=torch.sigmoid(torch.logit(prior)+1.5*torch.tanh(a[:,1])).clamp(1e-4,1-1e-4)
        eta=torch.sigmoid(a[:,2])
        mu=.15*torch.sqrt(v)*torch.tanh(a[:,3])
        if self.mode=='event_count':eta=eta*0
        if self.mode=='stale_clock':eta=eta*0+1
        if self.mode=='no_carry':age=age*0
        k=torch.arange(1,KMAX+1,dtype=x.dtype,device=x.device)[None,:]
        hh=h[:,None];valid=k<=hh
        q=1-p;atom=q**h
        stale=p[:,None]*q[:,None]**(hh-k).clamp_min(0)*valid
        logbinom=(torch.lgamma(hh+1)-torch.lgamma(k+1)-torch.lgamma((hh-k).clamp_min(0)+1)
                  +k*torch.log(p[:,None])+(hh-k).clamp_min(0)*torch.log(q[:,None]))
        event=torch.exp(logbinom)*valid
        es=(age[:,None]+k)/hh
        ee=k/(hh*p[:,None])
        ss=torch.sqrt(v[:,None]*es)
        se=torch.sqrt(v[:,None]*ee)
        ms=mu[:,None]*(age[:,None]+k)/torch.sqrt(hh)
        me=mu[:,None]*k/(p[:,None]*torch.sqrt(hh))
        if self.mode=='hurdle':
            ww=1-atom
            w=torch.stack([atom,ww],1)
            m=torch.stack([torch.zeros_like(mu),mu*torch.sqrt(h)/ww],1)
            sig=torch.stack([torch.zeros_like(v),torch.sqrt(v/ww)],1)
        else:
            zero=torch.zeros_like(atom)[:,None]
            w=torch.cat([atom[:,None],eta[:,None]*stale,(1-eta[:,None])*event],1)
            m=torch.cat([zero,ms,me],1)
            sig=torch.cat([zero,ss,se],1)
        return (w,m,sig),torch.stack([eta,p,v],1)


def absnorm(delta,sigma):
    safe=sigma.clamp_min(1e-8)
    zz=delta/safe
    vv=2*safe*torch.exp(-.5*zz**2)/math.sqrt(2*math.pi)+delta*torch.erf(zz/math.sqrt(2))
    return torch.where(sigma>0,vv,delta.abs())


def crps(y,comp):
    w,m,s=comp
    first=(w*absnorm(y[:,None]-m,s)).sum(1)
    pair=absnorm(m[:,:,None]-m[:,None,:],torch.sqrt(s[:,:,None]**2+s[:,None,:]**2+1e-20))
    return first-.5*(w[:,:,None]*w[:,None,:]*pair).sum((1,2))


def tensors(D,fit):
    mean=D['x'][fit].mean(0);sd=D['x'][fit].std(0)+1e-5
    return {k:torch.tensor(v,dtype=torch.float32) for k,v in dict(
        x=(D['x']-mean)/sd,base=D['base'],prior=D['prior'],age=D['age'],h=D['h'],z=D['z']).items()}


def predict(net,X,ids,batch=256):
    comps=[[],[],[]];aa=[]
    net.eval()
    with torch.no_grad():
        for ix in np.array_split(ids,max(1,int(np.ceil(len(ids)/batch)))):
            par,aux=net(*(X[k][ix] for k in ['x','base','prior','age','h']))
            for k in range(3):comps[k].append(par[k].numpy())
            aa.append(aux.numpy())
    return tuple(np.concatenate(v) for v in comps),np.concatenate(aa)


def train(D,mode,seed,epochs):
    def setup(ids):
        torch.manual_seed(seed)
        X=tensors(D,ids);net=ExposureNet(X['x'].shape[1],mode)
        opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.002)
        return X,net,opt
    def epoch(X,net,opt,ids):
        net.train();order=np.random.default_rng(seed+epoch.count).permutation(ids);epoch.count+=1
        for b in np.array_split(order,max(1,int(np.ceil(len(order)/256)))):
            opt.zero_grad()
            cp,aux=net(*(X[k][b] for k in ['x','base','prior','age','h']))
            loss=crps(X['z'][b],cp).mean()
            one=(D['h'][b]==1)
            if one.any():
                observed=torch.tensor((np.abs(D['y'][b[one]])>1e-12).astype(np.float32))
                loss=loss+.1*nn.functional.binary_cross_entropy(aux[one,1],observed)
            assert torch.isfinite(loss)
            loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
    epoch.count=0
    X,net,opt=setup(D['tr']);best=np.inf;bep=1;bad=0;history=[]
    for ep in range(1,epochs+1):
        epoch(X,net,opt,D['tr'])
        cp,_=predict(net,X,D['va'])
        v=float(crps(torch.tensor(D['z'][D['va']]),tuple(torch.tensor(v) for v in cp)).mean())
        history.append([ep,v])
        if v<best-1e-5:best=v;bep=ep;bad=0
        else:bad+=1
        if bad>=10 and ep>=15:break
    X,net,opt=setup(D['rf']);epoch.count=0
    for ep in range(bep):epoch(X,net,opt,D['rf'])
    cp,aux=predict(net,X,D['te'])
    assert np.allclose(cp[0].sum(1),1,atol=3e-5)
    return cp,dict(seed=seed,best_epoch=bep,inner_epochs=len(history),val_CRPS_normalized=best,
                  parameters=sum(p.numel() for p in net.parameters()),mean_gate=float(aux[:,0].mean()),
                  mean_refresh=float(aux[:,1].mean()),history=history)


def ensemble(comps):
    S=len(comps)
    return tuple(np.concatenate([cp[k]/S if k==0 else cp[k] for cp in comps],1) for k in range(3))


def quantiles(cp):
    w,m,s=cp
    lo=np.min(m-12*s,axis=1)[:,None]*np.ones((1,len(TAUS)))
    hi=np.max(m+12*s,axis=1)[:,None]*np.ones((1,len(TAUS)))
    atom=s==0
    for _ in range(40):
        z=(lo+hi)/2
        # Keep one component at a time to avoid N*Q*K temporary explosions.
        cdf=np.zeros_like(z)
        for k in range(w.shape[1]):
            v=ndtr((z-m[:,k,None])/np.maximum(s[:,k,None],1e-12))
            v=np.where(atom[:,k,None],z>=m[:,k,None],v)
            cdf+=w[:,k,None]*v
        below=cdf<TAUS
        lo=np.where(below,z,lo);hi=np.where(below,hi,z)
    q=(lo+hi)/2
    left=np.zeros(len(w));mass=np.zeros(len(w))
    for k in range(w.shape[1]):
        left+=np.where(atom[:,k],w[:,k]*(m[:,k]<0),w[:,k]*ndtr(-m[:,k]/np.maximum(s[:,k],1e-12)))
        mass+=np.where(atom[:,k] & (m[:,k]==0),w[:,k],0)
    q[(TAUS[None,:]>=left[:,None])&(TAUS[None,:]<=left[:,None]+mass[:,None])]=0
    assert np.all(np.diff(q,axis=1)>=-1e-7)
    return q


def mqs(y,q):
    e=y[:,None]-q
    return (2*np.maximum(TAUS*e,(TAUS-1)*e)).mean(1)


def empirical_predict(z,weights,y):
    ix=np.argsort(z,kind='stable');z=z[ix];weights=weights[:,ix]
    c=np.cumsum(weights,axis=1)
    q=np.column_stack([z[np.minimum((c<t).sum(1),len(z)-1)] for t in TAUS])
    first=(weights*np.abs(y[:,None]-z[None,:])).sum(1)
    second=(weights*z[None,:]*(2*c-weights-1)).sum(1)
    return q,first-second


def qrf_fit_predict(D,fit,pred,leaf):
    q=np.empty((len(pred),len(TAUS)));sc=np.empty(len(pred))
    qr=np.empty_like(q);sr=np.empty_like(sc)
    for hi in range(len(HORIZONS)):
        tr=fit[D['hi'][fit]==hi];sel=np.flatnonzero(D['hi'][pred]==hi);te=pred[sel]
        rf=RandomForestRegressor(n_estimators=120,min_samples_leaf=leaf,max_depth=7,
                                 random_state=11,n_jobs=1,max_features=1.)
        rf.fit(D['x'][tr],D['z'][tr])
        L=rf.apply(D['x'][tr]);T=rf.apply(D['x'][te]);W=np.zeros((len(te),len(tr)))
        for j in range(L.shape[1]):
            for node in np.unique(T[:,j]):
                members=np.flatnonzero(L[:,j]==node);queries=np.flatnonzero(T[:,j]==node)
                W[np.ix_(queries,members)]+=1/(len(members)*L.shape[1])
        qh,ch=empirical_predict(D['z'][tr],W,D['z'][te]);q[sel]=qh;sc[sel]=ch
        # Fixed strong reference from previous exploration: 5% uniform mixing,
        # zero mass from 80% observed local h-return frequency + 20% fit frequency.
        W=.95*W+.05/len(tr)
        nz=np.abs(D['y'][tr])>1e-12
        pn=.8*D['local_nonzero'][te]+.2*float(nz.mean())
        wn=W[:,nz];wn=wn/wn.sum(1,keepdims=True)*pn[:,None]
        if (~nz).any():
            wz=W[:,~nz];wz=wz/wz.sum(1,keepdims=True)*(1-pn[:,None])
            zr=np.r_[D['z'][tr[nz]],D['z'][tr[~nz]]]
            wr=np.column_stack([wn,wz])
        else:
            zr=np.r_[D['z'][tr[nz]],0.];wr=np.column_stack([wn,1-pn])
        qr[sel],sr[sel]=empirical_predict(zr,wr,D['z'][te])
    return {'QRF':(q,sc),'QRF_reference':(qr,sr)}


def controls(D):
    trials={'QRF':[],'QRF_reference':[]}
    for leaf in (15,30):
        predictions=qrf_fit_predict(D,D['tr'],D['va'],leaf)
        for name,(q,s) in predictions.items():
            trials[name].append(dict(leaf=leaf,val_CRPS_normalized=float(s.mean())))
    best={name:min(rows,key=lambda r:r['val_CRPS_normalized']) for name,rows in trials.items()}
    out={}
    for name in best:
        out[name]=qrf_fit_predict(D,D['rf'],D['te'],best[name]['leaf'])[name]
    q=out['QRF'][0];qq=np.empty_like(q);ss=np.empty(len(q))
    for hi in range(len(HORIZONS)):
        tr=D['rf'][D['hi'][D['rf']]==hi];sel=np.flatnonzero(D['hi'][D['te']]==hi);te=D['te'][sel]
        weights=np.ones((len(te),len(tr)))/len(tr)
        qq[sel],ss[sel]=empirical_predict(D['z'][tr],weights,D['z'][te])
    out['empirical']=(qq,ss)
    return out,dict(selection=best,trials=trials)


def sanity():
    # Independent Normal and point-atom CRPS identities.
    vv=crps(torch.tensor([0.,2.]),(torch.ones(2,1),torch.zeros(2,1),torch.tensor([[1.],[0.]])))
    assert abs(float(vv[0])-(math.sqrt(2)-1)/math.sqrt(math.pi))<1e-6
    assert abs(float(vv[1])-2)<1e-6
    # Simulate refresh paths and compare the implied final-return variance.
    rng=np.random.default_rng(100);n=100000;h=5;p=.35;age=3
    refresh=rng.random((n,h))<p
    latest=np.max(np.where(refresh,np.arange(1,h+1),0),axis=1)
    implied=np.where(latest>0,latest+age,0).mean()
    analytic=sum(p*(1-p)**(h-k)*(age+k) for k in range(1,h+1))
    assert abs(implied-analytic)<.04
    return dict(CRPS_known_cases=True,last_refresh_variance_MC=implied,analytic=analytic)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--markets',nargs='+',default=list(CUT))
    ap.add_argument('--seeds',nargs='+',type=int,default=[11]);ap.add_argument('--epochs',type=int,default=50)
    ap.add_argument('--models',nargs='+',default=MODES);ap.add_argument('--tag',default='exposure_round13')
    a=ap.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    config=dict(config=vars(a),status='EXPLORATORY; all historical tests reused',horizons=HORIZONS.tolist(),
        mechanism='last observed price-change time mixture versus event-count mixture; hypotheses, not proven nontrading states',
        horizon_unit='subsequent observed panel rows, not guaranteed consecutive business days',
        no_trade_assumption='zero price changes are observed indicators, not proven absence of trading; zero endpoint can arise from cancellation',
        selection='validation CRPS on standardized returns; epoch selection followed by pre-cut refit',
        training_objective='mean standardized CRPS + 0.1 BCE of one-step observed price-change probability',
        ablation_no_carry='removes explicit age from stale mixture component exposures; age context features remain available',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        panel_sha256=hashlib.sha256(PANEL.read_bytes()).hexdigest(),checks=sanity(),markets={})
    (HERE/(a.tag+'_manifest.json')).write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    raw=pd.read_csv(PANEL,parse_dates=['date']);arrays={};start=time.time()
    for market in a.markets:
        d=raw[raw.market==market].sort_values('date').reset_index(drop=True);D=build(d);te=D['te']
        controls_,info=controls(D)
        res=dict(audit=dict(cut=D['cut'],validation_start=D['val'],
            max_training_label_end=int(np.max(D['t'][D['tr']]+D['h'][D['tr']])),
            max_refit_label_end=int(np.max(D['t'][D['rf']]+D['h'][D['rf']])),
            first_test_origin=int(D['t'][te].min()),
            counts={k:int(len(D[k])) for k in ['tr','va','rf','te']}),controls=info,fits={},scores={})
        predq={k:q*D['scale'][te,None] for k,(q,s) in controls_.items()}
        preds={k:s*D['scale'][te] for k,(q,s) in controls_.items()}
        for mode in a.models:
            cps=[];fits=[]
            for seed in a.seeds:
                cp,fit=train(D,mode,seed,a.epochs);cps.append(cp);fits.append(fit)
                print(market,mode,seed,'best_epoch',fit['best_epoch'],'val',round(fit['val_CRPS_normalized'],6),
                      'seconds',round(time.time()-start,1),flush=True)
            cp=ensemble(cps);q=quantiles(cp)
            # Chunk exact CRPS of the same coherent seed-mixture distribution.
            sc=[]
            for ix in np.array_split(np.arange(len(te)),max(1,int(np.ceil(len(te)/128)))):
                sc.append(crps(torch.tensor(D['z'][te[ix]]),tuple(torch.tensor(v[ix]) for v in cp)).numpy())
            predq[mode]=q*D['scale'][te,None];preds[mode]=np.concatenate(sc)*D['scale'][te]
            res['fits'][mode]=fits
        for hi,h in enumerate(HORIZONS):
            mask=D['hi'][te]==hi;y=D['y'][te][mask]
            res['scores'][str(h)]={name:dict(n=int(mask.sum()),MQS=float(mqs(y,q[mask]).mean()),
                CRPS=float(preds[name][mask].mean()),MAE_median=float(np.abs(y-q[mask,7]).mean()))
                for name,q in predq.items()}
            print(market,'h',h,{k:round(v['CRPS'],7) for k,v in res['scores'][str(h)].items()},flush=True)
        config['markets'][market]=res
        arrays[market+'_target']=D['y'][te];arrays[market+'_origin']=D['t'][te]
        arrays[market+'_horizon']=D['h'][te]
        for name in predq:
            arrays[market+'_'+name]=predq[name];arrays[market+'_'+name+'_CRPS']=preds[name]
        config['elapsed_seconds']=time.time()-start
        (HERE/(a.tag+'.json')).write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
        np.savez_compressed(HERE/(a.tag+'_predictions.npz'),**arrays)
    print('FINISHED',round(time.time()-start,1),'seconds',flush=True)


if __name__=='__main__':main()
