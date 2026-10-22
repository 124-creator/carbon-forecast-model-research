# -*- coding: utf-8 -*-
"""Causal empirical-tail transport network, exploratory candidate.

Historical standardized h-return scenarios supply a flexible distribution.
The network predicts bounded log-weight adjustments to nonzero scenarios;
zero mass uses static, rolling, or learned frequency priors. Training retrieval
uses only labels matured by the query origin. Validation/refit memories contain
only labels available before their boundary. Weighted-ECDF CRPS is exact.
This is a neural empirical-distribution adaptation, not a novel distribution
family or a claim of causal economic identification.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits
import marked_event_round2 as E
import neural_restart as B

P=Path(__file__).resolve().parent
HOR=E.HOR
TT=torch.tensor(B.TAUS,dtype=torch.float32)


def add_context(d,D):
    lp=np.log(d.price.to_numpy(float));dates=d.date.to_numpy()
    extra=[]
    for t,hi in zip(D['t'],D['hi']):
        hist=lp[max(0,t-249):t+1]
        r=np.diff(lp[t-E.W:t+1]);sd=max(r.std(ddof=1),1e-8)
        frac=2*(lp[t]-hist.min())/(hist.max()-hist.min()+1e-8)-1
        day=pd.Timestamp(dates[t]).dayofyear;ang=2*np.pi*day/365.25
        extra.append([lp[t],(lp[t]-lp[t-119:t+1].mean())/(sd*np.sqrt(120)),
                      (lp[t]-hist.mean())/(sd*np.sqrt(len(hist))),frac,
                      np.sin(ang),np.cos(ang),np.sin(2*ang),np.cos(2*ang)])
    D['x_state']=D['x'].copy()
    D['x_aug']=np.column_stack([D['x'],np.array(extra,np.float32)]).astype(np.float32)
    return D


def basis(z):
    q90=np.quantile(np.abs(z),.9)
    a=np.stack([np.tanh(z/2),np.abs(z),np.minimum(z*z,64),
                np.sign(z)*np.maximum(np.abs(z)-q90,0),
                np.maximum(np.abs(z)-q90,0)],1)
    a=(a-a.mean(0))/(a.std(0)+1e-5)
    return np.clip(a,-3,3).astype(np.float32)


def make_bank(D,idx):
    banks=[]
    for hi,h in enumerate(HOR):
        ids=idx[D['hi'][idx]==hi]
        nz=ids[np.abs(D['y'][ids])>1e-12]
        order=np.argsort(D['z'][nz],kind='stable');nz=nz[order]
        z=D['z'][nz].astype(np.float32)
        assert len(z)>32
        zero_pos=int(np.searchsorted(z,0.))
        support=np.insert(z,zero_pos,0.).astype(np.float32)
        banks.append({'z':torch.tensor(z),'phi':torch.tensor(basis(z)),
                      'mature':torch.tensor(D['t'][nz]+int(h)),
                      'support':torch.tensor(support),'zero_pos':zero_pos,
                      'p':float(np.mean(np.abs(D['y'][ids])>1e-12)),'n':len(ids)})
    return banks


class TiltNet(nn.Module):
    def __init__(self,nf,mode):
        super().__init__();self.mode=mode
        self.body=nn.Sequential(nn.Linear(nf,24),nn.Tanh(),nn.Linear(24,16),nn.Tanh())
        self.head=nn.Linear(16,7)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def weights(self,x,local,t,bank,causal=False):
        a=self.head(self.body(x))
        prior=torch.full_like(local,bank['p'])
        if self.mode in ['uniform_rule','tilt_rule','tilt_neural','tilt_neural_no_context','probability_only']:
            prior=.8*local+.2*prior
        if self.mode in ['tilt_neural','tilt_neural_no_context','probability_only']:
            gate=torch.sigmoid(np.log(4.)+a[:,5])
            pp=(1-gate)*bank['p']+gate*local
            p=torch.sigmoid(torch.logit(pp.clamp(1e-5,1-1e-5))+.75*torch.tanh(a[:,6]))
        else:p=prior
        allowed=bank['mature'][None,:]<=t[:,None] if causal else torch.ones((len(t),len(bank['z'])),dtype=torch.bool)
        beta=.35*torch.tanh(a[:,:5])
        if self.mode in ['uniform_static','uniform_rule','probability_only']:beta=beta*0
        logits=beta@bank['phi'].T
        logits=logits.masked_fill(~allowed,-1e9)
        mark=logits.softmax(1)*allowed
        mark=mark/mark.sum(1,keepdim=True).clamp_min(1e-20)
        # All training queries have >=32 matured nonzero scenarios.
        assert torch.all(mark.sum(1)>.99)
        j=bank['zero_pos']
        weight=torch.cat([p[:,None]*mark[:,:j],(1-p)[:,None],p[:,None]*mark[:,j:]],1)
        return weight,p,beta


def crps(y,weights,support):
    # 1/2 E|X-X'| over sorted support is sum_i w_i (z_i W_<i - A_<i).
    cw=weights.cumsum(1)-weights
    zw=weights*support[None,:]
    ca=zw.cumsum(1)-zw
    term=(weights*(support[None,:]*cw-ca)).sum(1)
    return (weights*(support[None,:]-y[:,None]).abs()).sum(1)-term


def quantiles(weights,support):
    cw=weights.cumsum(1).contiguous()
    cw[:,-1]=1.
    indices=torch.searchsorted(cw,TT[None,:].expand(len(cw),-1).contiguous()).clamp_max(len(support)-1)
    return support[indices]


def prepare(D,idx,aug):
    raw=D['x_aug']
    mean=raw[idx].mean(0);std=raw[idx].std(0)+1e-5
    xx=np.clip((raw-mean)/std,-8,8)
    if not aug:xx[:,D['x_state'].shape[1]:]=0.
    x=torch.tensor(xx,dtype=torch.float32)
    return {'x':x,'z':torch.tensor(D['z'],dtype=torch.float32),
            'hi':torch.tensor(D['hi']),'local':torch.tensor(D['local']),
            't':torch.tensor(D['t'])},make_bank(D,idx)


def forward(net,X,idx,banks,causal=False):
    result={}
    for hi,bank in enumerate(banks):
        keep=idx[X['hi'][idx]==hi]
        if len(keep):
            w,p,b=net.weights(X['x'][keep],X['local'][keep],X['t'][keep],bank,causal)
            result[hi]=(keep,w,p,b)
    return result


def optimize(net,X,banks,train,epochs,seed,validation=None):
    torch.manual_seed(seed+731)
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.003)
    # Exclude cold-start training origins rather than looking ahead in memory.
    eligible=[]
    for hi,bank in enumerate(banks):
        ii=train[np.asarray(X['hi'][train])==hi]
        c=(bank['mature'][None,:]<=X['t'][ii,None]).sum(1)
        eligible.extend(ii[c.numpy()>=32].tolist())
    ids=torch.tensor(eligible)
    assert len(ids)>32
    def vl():
        with torch.no_grad():
            scores=[]
            for hi,(ii,w,_,_) in forward(net,X,validation,banks).items():
                q=quantiles(w,banks[hi]['support'])
                e=X['z'][ii,None]-q
                scores.append(torch.maximum(TT*e,(TT-1)*e).mean())
            return float(torch.stack(scores).mean())
    net.eval();best=vl() if validation is not None else None
    epbest,bad,state=0,0,copy.deepcopy(net.state_dict())
    count=np.bincount(X['hi'][ids].numpy(),minlength=len(HOR))
    rowweight=torch.tensor(len(ids)/(len(HOR)*count),dtype=torch.float32)
    for ep in range(epochs):
        net.train();order=ids[torch.randperm(len(ids))]
        for ix in order.split(512):
            loss=torch.tensor(0.)
            for hi,(ii,w,_,beta) in forward(net,X,ix,banks,True).items():
                loss=loss+(crps(X['z'][ii],w,banks[hi]['support'])*rowweight[hi]).sum()/len(ix)
                loss=loss+.003*beta.square().mean()*len(ii)/len(ix)
            assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        if validation is not None:
            net.eval();value=vl()
            if value<best-1e-6:
                best,epbest,bad=value,ep+1,0;state=copy.deepcopy(net.state_dict())
            else:bad+=1
            if ep>=29 and bad>=16:break
    if validation is not None:net.load_state_dict(state)
    net.eval()
    return {'best_epoch':epbest,'validation_normalized_macro_pinball':best,
            'epochs_run':ep+1 if epochs else 0,'eligible_training_queries':len(ids),
            'training_memory':'only nonzero outcomes with t_key+h<=query_origin; minimum32'}


def fit(D,mode,seeds,epochs):
    aug=mode!='tilt_neural_no_context'
    X,banks=prepare(D,D['tr'],aug)
    fits=[];selected=[]
    for seed in seeds:
        torch.manual_seed(seed);net=TiltNet(X['x'].shape[1],mode)
        f=({'best_epoch':0,'validation_normalized_macro_pinball':None,
            'epochs_run':0,'eligible_training_queries':None,'fit_kind':'fixed statistical control'}
           if mode in ['uniform_static','uniform_rule'] else
           optimize(net,X,banks,D['tr'],epochs,seed,torch.tensor(D['va'])))
        selected.append(f['best_epoch']);fits.append(dict(seed=seed,**f))
    Xf,bf=prepare(D,D['rf'],aug);te=torch.tensor(D['te'])
    sumweights={hi:None for hi in range(len(HOR))}
    for seed,ep in zip(seeds,selected):
        torch.manual_seed(seed);net=TiltNet(Xf['x'].shape[1],mode)
        if ep:optimize(net,Xf,bf,D['rf'],ep,seed)
        with torch.no_grad():
            for hi,(ii,w,_,_) in forward(net,Xf,te,bf).items():
                sumweights[hi]=w.clone()/len(seeds) if sumweights[hi] is None else sumweights[hi]+w/len(seeds)
    Q=np.empty((len(te),len(TT)));S=np.empty(len(te));Pro=np.empty(len(te))
    for hi,bank in enumerate(bf):
        keep=np.flatnonzero(D['hi'][D['te']]==hi);idx=te[keep];w=sumweights[hi]
        with torch.no_grad():
            q=quantiles(w,bank['support']).numpy();sc=D['scale'][D['te'][keep]]
            Q[keep]=q*sc[:,None]
            S[keep]=crps(Xf['z'][idx],w,bank['support']).numpy()*sc
            Pro[keep]=1-w[:,bank['zero_pos']].numpy()
    assert np.isfinite(Q).all() and np.min(np.diff(Q,axis=1))>=-1e-8
    return Q,S,Pro,fits


def check_crps():
    y=torch.tensor([-.5,.4,1.2],dtype=torch.float64)
    z=torch.tensor([-1.,0.,2.],dtype=torch.float64)
    w=torch.tensor([[.2,.3,.5],[.3,.5,.2],[.1,.4,.5]],dtype=torch.float64,requires_grad=True)
    fast=crps(y,w,z)
    brute=(w*(z[None,:]-y[:,None]).abs()).sum(1)-.5*(w[:,:,None]*w[:,None,:]*(z[None,:,None]-z[None,None,:]).abs()).sum((1,2))
    error=float((fast-brute).abs().max().detach());assert error<1e-12
    fast.sum().backward();assert torch.isfinite(w.grad).all()
    return {'weighted_ECDF_CRPS_matches_pairwise':error,'gradients_finite':True}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--markets',nargs='+',default=list(B.CUT))
    ap.add_argument('--modes',nargs='+',default=['uniform_static','uniform_rule','tilt_static','tilt_rule','tilt_neural','probability_only','tilt_neural_no_context'])
    ap.add_argument('--seeds',nargs='+',type=int,default=[11,22]);ap.add_argument('--epochs',type=int,default=64)
    ap.add_argument('--tag',default='empirical_tilt_round3');a=ap.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    panel=pd.read_csv(B.PANEL,parse_dates=['date'])
    report={'status':'exploratory; reused historical tests; no publication or novelty claim',
            'config':vars(a),'sanity':check_crps(),'horizons':HOR.tolist(),
            'geometry':'weighted empirical nonzero returns plus explicit zero atom; exact finite-distribution CRPS',
            'training':'causally matured support labels; bounded 5-basis log-weight transport; small L2 penalty',
            'selection':'inner equal-horizon normalized pinball; epoch0 eligible; refit all precut labels',
            'ensemble':'average seed probabilities on the same refit support, coherent mixture CRPS and quantiles',
            'panel_sha256':hashlib.sha256(B.PANEL.read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'markets':{}}
    arrays={};start=time.perf_counter()
    oldmeta=json.loads((P/'marked_event_round2.json').read_text(encoding='utf-8'))
    old=np.load(P/'marked_event_round2_predictions.npz')
    for market in a.markets:
        d=panel[panel.market==market].sort_values('date').reset_index(drop=True)
        D=add_context(d,E.build(d));te=D['te'];y=D['y'][te];hi=D['hi'][te]
        # Keep prior-stage empirical/Gaussian quantile baselines, with identical targets.
        assert np.array_equal(D['t'][te],old[market+'_origin']) and np.allclose(y,old[market+'_target'],atol=0,rtol=0)
        result={'audit':D['audit'],'fits':{},'by_horizon':{}}
        preds={};scores={};probs={}
        # Baseline input and predictive representation remain explicitly labeled.
        pred0,crps0,p0=E.baseline_scores(D)
        preds.update(pred0);scores.update(crps0);probs.update(p0)
        for mode in a.modes:
            q,s,p,f=fit(D,mode,a.seeds,a.epochs)
            preds[mode]=q;scores[mode]=s;probs[mode]=p;result['fits'][mode]=f
            print(market,mode,'epochs',[ff['best_epoch'] for ff in f],
                  'elapsed_s',round(time.perf_counter()-start,1),flush=True)
        if 'uniform_static' in scores:
            assert np.allclose(scores['uniform_static'],crps0['empirical_constant'],rtol=1e-4,atol=1e-6), 'uniform ECDF CRPS mismatch'
            result['uniform_ecdf_CRPS_max_error']=float(np.max(np.abs(scores['uniform_static']-crps0['empirical_constant'])))
        for hidx,h in enumerate(HOR):
            mask=hi==hidx;rr={}
            for name,q in preds.items():
                yh,qh=y[mask],q[mask]
                rr[name]={'MQS':float(B.score(yh,qh).mean()),'CRPS':float(scores[name][mask].mean()),
                          'WIS':float(B.wis(yh,qh).mean()),'MAE':float(np.abs(yh-qh[:,7]).mean()),
                          'coverage80':float(np.mean((yh>=qh[:,2])&(yh<=qh[:,12]))),
                          'coverage95':float(np.mean((yh>=qh[:,0])&(yh<=qh[:,14]))),
                          'Brier':float(np.mean((probs[name][mask]-(np.abs(yh)>1e-12))**2))}
            result['by_horizon'][str(h)]={'n':int(mask.sum()),'scores':rr}
        report['markets'][market]=result
        arrays[market+'_origin']=D['t'][te];arrays[market+'_horizon']=HOR[hi];arrays[market+'_target']=y
        for name in preds:
            arrays[market+'_'+name]=preds[name];arrays[market+'_'+name+'_CRPS']=scores[name];arrays[market+'_'+name+'_p']=probs[name]
        (P/(a.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        np.savez_compressed(P/(a.tag+'_predictions.npz'),**arrays)
    report['summary']={};ref='tilt_neural' if 'tilt_neural' in a.modes else a.modes[-1]
    for base in preds:
        if base==ref:continue
        out={}
        for metric in ['MQS','CRPS']:
            r=[c['scores'][ref][metric]/c['scores'][base][metric] for m in report['markets'].values() for c in m['by_horizon'].values()]
            out[metric]={'geometric_gain_percent':float(100*(1-np.exp(np.log(r).mean()))),
                         'wins':int((np.array(r)<1).sum()),'cells':len(r)}
        report['summary'][ref+'_vs_'+base]=out
    report['elapsed_seconds']=time.perf_counter()-start
    (P/(a.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with threadpool_limits(limits=2):main()
