# -*- coding: utf-8 -*-
"""Round 2: local nonzero-return prior + proper closed-form mixture CRPS.
Only marked_event_round2* outputs are written. Exploratory reused holdouts.
Horizon means observed daily-panel steps. Endpoint-zero is not no-path-update.
Normal component centers have signs; overlapping components are not sign bins.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.special import ndtr
from marked_event_round1 import build as build_first, initial, nll, quantiles, constant_predictions, HOR, W
from neural_restart import PANEL, CUT, TAUS, score, wis

HERE = Path(__file__).resolve().parent
PRIORS = ['static_prior', 'rolling_rule_prior', 'rolling_neural_prior']
RULE_LOCAL_WEIGHT = .8


def build(d):
    D = build_first(d)
    lp = np.log(d.price.to_numpy(float))
    local = []
    counts = []
    for t, hi in zip(D['t'], D['hi']):
        h = int(HOR[hi])
        history = lp[t-W:t+1]  # Last available endpoint is exactly t.
        past_h_returns = history[h:] - history[:-h]
        n = len(past_h_returns)
        assert n == W+1-h
        local.append((np.sum(np.abs(past_h_returns)>1e-12)+1)/(n+2))
        counts.append(n)
    D['local'] = np.asarray(local, np.float32)
    D['x'] = np.column_stack([D['x'], np.log(D['local']/(1-D['local']))]).astype(np.float32)
    D['audit']['local_prior'] = dict(lookback=W, beta_alpha=1, beta_beta=1,
        history='lp[t-W:t+1]; all h-return target endpoints <=t',
        observations_per_h={str(h):int(W+1-h) for h in HOR},
        rule_local_weight=RULE_LOCAL_WEIGHT)
    return D


class PriorNet(nn.Module):
    def __init__(self, nf, base, prior):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(nf,24),nn.Tanh(),nn.Linear(24,16),nn.Tanh())
        self.prior_head = nn.Linear(16,2)
        self.mark_head = nn.Linear(16,5)
        for layer in [self.prior_head,self.mark_head]:
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
        self.register_buffer('base',torch.tensor(base))
        self.prior = prior

    def forward(self,x,hi,local):
        a = self.trunk(x)
        b = self.base[hi]
        static = torch.sigmoid(b[:,0])
        gate = torch.zeros_like(static)
        residual = torch.zeros_like(static)
        if self.prior == 'static_prior':
            p = static
        elif self.prior == 'rolling_rule_prior':
            gate = torch.full_like(static,RULE_LOCAL_WEIGHT)
            p = (1-gate)*static+gate*local
        else:
            pr = self.prior_head(a)
            gate = torch.sigmoid(math.log(RULE_LOCAL_WEIGHT/(1-RULE_LOCAL_WEIGHT))+pr[:,0])
            residual = .75*torch.tanh(pr[:,1])
            prior = (1-gate)*static+gate*local
            p = torch.sigmoid(torch.logit(prior.clamp(1e-6,1-1e-6))+residual)
        m = self.mark_head(a)
        w = torch.sigmoid(b[:,1]+m[:,0])
        mp = torch.exp(b[:,2]+1.5*torch.tanh(m[:,1]))
        mn = -torch.exp(b[:,3]+1.5*torch.tanh(m[:,2]))
        sp = torch.exp(b[:,4]+2*torch.tanh(m[:,3]))+.025
        sn = torch.exp(b[:,5]+2*torch.tanh(m[:,4]))+.025
        return torch.stack([p,w,mp,mn,sp,sn],1),torch.stack([gate,residual],1)


def abs_normal(delta,sigma):
    """E|N(delta,sigma^2)| for strictly positive sigma."""
    z = delta/sigma
    return 2*sigma*torch.exp(-.5*z*z)/math.sqrt(2*math.pi)+delta*torch.erf(z/math.sqrt(2))


def crps_rows(y,par):
    """Exact CRPS for atom-at-zero plus two normal marks, not a quantile grid."""
    p,w,mp,mn,sp,sn = par.unbind(1)
    a,b,c = 1-p,p*w,p*(1-w)
    first = a*y.abs()+b*abs_normal(y-mp,sp)+c*abs_normal(y-mn,sn)
    half_pair = (a*b*abs_normal(mp,sp)+a*c*abs_normal(mn,sn)
        +b*b*sp/math.sqrt(math.pi)+c*c*sn/math.sqrt(math.pi)
        +b*c*abs_normal(mp-mn,torch.sqrt(sp*sp+sn*sn)))
    return first-half_pair


def components(seed_pars):
    """Equal mixture of fitted seed distributions; merge their zero atoms."""
    pars = np.asarray(seed_pars,float)
    S,N,_ = pars.shape
    ps,ws = pars[:,:,0],pars[:,:,1]
    weights = np.concatenate([(1-ps.mean(0))[:,None],(ps*ws).T/S,(ps*(1-ws)).T/S],1)
    mus = np.concatenate([np.zeros((N,1)),pars[:,:,2].T,pars[:,:,3].T],1)
    sigmas = np.concatenate([np.zeros((N,1)),pars[:,:,4].T,pars[:,:,5].T],1)
    assert np.allclose(weights.sum(1),1) and np.all(weights>=0)
    return weights,mus,sigmas


def mixture_quantiles(comp):
    w,m,s = comp
    lo = np.min(m-10*s,axis=1)[:,None]+np.zeros((len(w),len(TAUS)))
    hi = np.max(m+10*s,axis=1)[:,None]+np.zeros_like(lo)
    for _ in range(32):
        mid = (lo+hi)/2
        cdf = w[:,0,None]*(mid>=0)
        for j in range(1,w.shape[1]):
            cdf += w[:,j,None]*ndtr((mid-m[:,j,None])/s[:,j,None])
        left = cdf < TAUS[None,:]
        lo=np.where(left,mid,lo);hi=np.where(left,hi,mid)
    q=(lo+hi)/2
    left0=np.sum(w[:,1:]*ndtr(-m[:,1:]/s[:,1:]),axis=1)[:,None]
    q[(TAUS[None,:]>=left0)&(TAUS[None,:]<=left0+w[:,0,None])]=0.
    assert np.isfinite(q).all() and np.all(np.diff(q,axis=1)>=-1e-8)
    return q


def abs_normal_np(delta,sigma):
    # Handles zero sigma explicitly, avoiding undefined atom/atom operations.
    ss=np.maximum(sigma,1e-200)
    z=np.divide(delta,ss)
    z=np.clip(z,-40,40)
    val=2*ss*np.exp(-.5*z*z)/math.sqrt(2*math.pi)+delta*(2*ndtr(z)-1)
    return np.where(sigma>0,val,np.abs(delta))


def mixture_crps(y,comp):
    w,m,s=comp
    first=np.sum(w*abs_normal_np(y[:,None]-m,s),axis=1)
    second=np.zeros(len(y))
    for j in range(w.shape[1]):
        for k in range(w.shape[1]):
            second+=w[:,j]*w[:,k]*abs_normal_np(m[:,j]-m[:,k],np.sqrt(s[:,j]**2+s[:,k]**2))
    out=first-.5*second
    assert np.isfinite(out).all() and out.min()>-1e-7
    return np.maximum(out,0)


def loss_sanity():
    par=torch.tensor([[1.,.5,0.,0.,1.,1.],[0.,.5,0.,0.,1.,1.]],dtype=torch.float64)
    y=torch.tensor([0.,2.],dtype=torch.float64)
    v=crps_rows(y,par).numpy()
    expected=np.array([(math.sqrt(2)-1)/math.sqrt(math.pi),2.])
    assert np.allclose(v,expected,atol=1e-10)
    p=torch.tensor([[.7,.6,.8,-.5,1.1,.7]],dtype=torch.float64,requires_grad=True)
    grad_ok=torch.autograd.gradcheck(lambda x:crps_rows(torch.tensor([.35],dtype=torch.float64),x).sum(),(p,),eps=1e-6,atol=1e-5)
    par_np=p.detach().numpy()
    assert np.allclose(crps_rows(torch.tensor([.35],dtype=torch.float64),p).detach().numpy(),mixture_crps(np.array([.35]),components([par_np])),atol=1e-10)
    # Independent numerical integration of (F(z)-I(z>=y))^2, split at atom and y.
    from scipy.integrate import quad
    pv,wv,mp,mn,sp,sn=par_np[0]
    def integrand(x):
        F=(1-pv)*(x>=0)+pv*(wv*ndtr((x-mp)/sp)+(1-wv)*ndtr((x-mn)/sn))
        return (F-(x>=.35))**2
    integ=sum(quad(integrand,l,h,epsabs=1e-10)[0] for l,h in [(-np.inf,0),(0,.35),(.35,np.inf)])
    exact=float(crps_rows(torch.tensor([.35],dtype=torch.float64),p).detach()[0])
    assert abs(integ-exact)<1e-8
    return dict(standard_normal_at_zero=float(v[0]),point_mass_error=float(v[1]),
                gradcheck=bool(grad_ok),CDF_integral=integ,closed_form=exact)


def train_predict(D,prior,objective,seed,epochs):
    z=torch.tensor(D['z'],dtype=torch.float32)
    hid=torch.tensor(D['hi'],dtype=torch.long)
    local=torch.tensor(D['local'],dtype=torch.float32)
    def setup(idx):
        torch.manual_seed(seed)
        mean=D['x'][idx].mean(0);sd=D['x'][idx].std(0)+1e-5
        x=torch.tensor((D['x']-mean)/sd,dtype=torch.float32)
        net=PriorNet(x.shape[1],initial(D,idx),prior)
        opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.002)
        return x,net,opt
    def epoch(x,net,opt,idx):
        net.train()
        order=torch.tensor(idx)[torch.randperm(len(idx))]
        for b in order.split(512):
            opt.zero_grad()
            par,_=net(x[b],hid[b],local[b])
            loss=crps_rows(z[b],par).mean() if objective=='crps' else nll(z[b],par,'dual_head')
            assert torch.isfinite(loss)
            loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
    x,net,opt=setup(D['tr'])
    best,bep,bad=np.inf,0,0;hist=[]
    for ep in range(1,epochs+1):
        epoch(x,net,opt,D['tr'])
        if ep%2 and ep!=1 and ep!=epochs:
            continue
        net.eval()
        with torch.no_grad():
            pa,_=net(x[D['va']],hid[D['va']],local[D['va']])
        vl=float(score(D['z'][D['va']],quantiles(pa.numpy())).mean())
        hist.append([ep,vl])
        if vl<best-1e-6:
            best,bep,bad=vl,ep,0
        else:
            bad+=1
        if bad>=8 and ep>=20:
            break
    x,net,opt=setup(D['rf'])
    for _ in range(bep):
        epoch(x,net,opt,D['rf'])
    net.eval()
    with torch.no_grad():
        par,aux=net(x[D['te']],hid[D['te']],local[D['te']])
    return par.numpy(),dict(seed=seed,best_epoch=bep,inner_epochs=ep,refit_epochs=bep,
        validation_MQS_standardized=best,validation_history=hist,
        allocated_parameters=sum(p.numel() for p in net.parameters()),
        mean_local_gate=float(aux[:,0].mean()),mean_prior_logit_residual=float(aux[:,1].mean()),
        mean_nonzero_probability=float(par[:,0].mean()))


def baseline_scores(D):
    """Empirical baseline representations deliberately retain round1 MQS:
    MQS uses linearly interpolated sample quantiles, exact CRPS uses the complete
    fitting-sample ECDF, and Brier uses Beta(1,1)-smoothed nonzero frequency.
    These are separately labelled baseline metrics, not one coherent ECDF.
    """
    te,rf=D['te'],D['rf'];y=D['z'][te];scale=D['scale'][te]
    q=constant_predictions(D)
    cs={k:np.empty(len(te)) for k in q};ps={k:np.empty(len(te)) for k in q}
    for hi in range(4):
        z=D['z'][rf[D['hi'][rf]==hi]];mask=D['hi'][te]==hi
        mu=z.mean();sd=max(z.std(ddof=1),1e-5)
        zz=(y[mask]-mu)/sd
        cs['gaussian_constant'][mask]=sd*(zz*(2*ndtr(zz)-1)+2*np.exp(-.5*zz*zz)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi))
        zs=np.sort(z);n=len(zs)
        halfpair=np.sum((2*np.arange(1,n+1)-n-1)*zs)/(n*n)
        cs['empirical_constant'][mask]=np.abs(y[mask,None]-z[None,:]).mean(1)-halfpair
        ps['empirical_constant'][mask]=(np.sum(z!=0)+1)/(n+2)
        ps['gaussian_constant'][mask]=1.
    return q,{k:v*scale for k,v in cs.items()},ps


def summarize(report):
    pairs=[('rolling_rule_prior_crps','static_prior_crps'),
           ('rolling_neural_prior_crps','static_prior_crps'),
           ('rolling_neural_prior_crps','rolling_rule_prior_crps'),
           ('rolling_neural_prior_crps','rolling_neural_prior_nll'),
           ('rolling_rule_prior_nll','static_prior_nll'),
           ('rolling_neural_prior_nll','static_prior_nll'),
           ('static_prior_crps','static_prior_nll'),
           ('rolling_rule_prior_crps','rolling_rule_prior_nll')]
    pairs += [(p+'_'+o,b) for p in PRIORS for o in ['nll','crps'] for b in ['empirical_constant','gaussian_constant']]
    out={}
    for model,base in pairs:
        cells=[c['scores'] for m in report['markets'].values() for c in m['by_horizon'].values()]
        vals={}
        for metric in ['MQS','CRPS','Brier']:
            ratios=np.array([c[model][metric]/max(c[base][metric],1e-15) for c in cells])
            vals[metric]=dict(cells=len(ratios),wins=int(np.sum(ratios<1)),
                median_ratio=float(np.median(ratios)),geometric_gain_percent=float(100*(1-np.exp(np.log(np.maximum(ratios,1e-15)).mean()))))
        out[model+'_vs_'+base]=vals
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--markets',nargs='+',default=list(CUT));ap.add_argument('--seeds',nargs='+',type=int,default=[11,22])
    ap.add_argument('--epochs',type=int,default=48);ap.add_argument('--tag',default='marked_event_round2')
    args=ap.parse_args()
    assert args.tag.startswith('marked_event_round2')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    sanity=loss_sanity()
    panel=pd.read_csv(PANEL,parse_dates=['date']);start=time.perf_counter()
    report=dict(status='exploratory; reused historical test periods; not publication success',config=vars(args),
        horizons=HOR.tolist(),quantiles=TAUS.tolist(),horizon_unit='observed daily-panel steps',
        target='direct marginal log return; endpoint zero is not no-intervening-update',
        architecture='matched shared 24/16 MLP + 5 signed-center normal-mixture mark parameters; gate/residual prior head',
        local_prior='Beta(1,1) smoothed h-return nonzero proportion in lp[t-128:t+1]',
        feature_policy='all variants receive identical features including logit(local prior)',
        training_objectives=['mixed NLL','exact analytic atom+normal-mixture CRPS on standardized returns'],
        selection='same normalized validation MQS; full precut fixed selected epoch refit',
        ensemble='equal probability mixture of seed distributions; coherent CDF-MQS/closed-form CRPS/Brier',
        round1_ensemble_difference='round1 averaged seed quantiles; compare architecture increments primarily within this round',
        loss_sanity=sanity,panel_sha256=hashlib.sha256(PANEL.read_bytes()).hexdigest(),
        python=platform.python_version(),torch=torch.__version__,markets={})
    arrays={};rows=[]
    for market in args.markets:
        d=panel[panel.market==market].sort_values('date').reset_index(drop=True);D=build(d)
        te=D['te'];y=D['y'][te];hi=D['hi'][te];sc=D['scale'][te]
        preds,crps,probs=baseline_scores(D)
        arrays[market+'_origin']=D['t'][te];arrays[market+'_horizon']=HOR[hi]
        arrays[market+'_target']=y;arrays[market+'_scale']=sc
        arrays[market+'_origin_date']=d.date.to_numpy()[D['t'][te]].astype('datetime64[D]')
        arrays[market+'_local_prior']=D['local'][te]
        mres=dict(audit=D['audit'],fits={},by_horizon={})
        for prior in PRIORS:
            for objective in ['nll','crps']:
                model=prior+'_'+objective;pars=[];fits=[]
                for seed in args.seeds:
                    pa,fit=train_predict(D,prior,objective,seed,args.epochs)
                    pars.append(pa);fits.append(fit)
                    arrays[f'{market}_{model}_seed{seed}_parameters']=pa
                    print(market,model,seed,'epoch',fit['best_epoch'],'elapsed',round(time.perf_counter()-start,1),flush=True)
                comp=components(pars)
                preds[model]=mixture_quantiles(comp)*sc[:,None]
                crps[model]=mixture_crps(D['z'][te],comp)*sc
                probs[model]=np.mean(np.stack([p[:,0] for p in pars]),axis=0)
                mres['fits'][model]=fits
        for k,h in enumerate(HOR):
            mask=hi==k;scores={}
            for model,q in preds.items():
                ym,qm,pm=y[mask],q[mask],probs[model][mask]
                binary=(ym!=0).astype(float);clipped=np.clip(pm,1e-12,1-1e-12)
                scores[model]=dict(MQS=float(score(ym,qm).mean()),WIS=float(wis(ym,qm).mean()),
                    CRPS=float(crps[model][mask].mean()),Brier=float(np.mean((pm-binary)**2)),
                    log_loss_nonzero=float(np.mean(-binary*np.log(clipped)-(1-binary)*np.log1p(-clipped))),
                    coverage80=float(np.mean((ym>=qm[:,2])&(ym<=qm[:,12]))),
                    coverage95=float(np.mean((ym>=qm[:,0])&(ym<=qm[:,14]))),
                    MAE_median=float(np.abs(ym-qm[:,7]).mean()),
                    mean_nonzero_probability=float(pm.mean()),empirical_nonzero_rate=float(binary.mean()))
            mres['by_horizon'][str(h)]=dict(n=int(mask.sum()),scores=scores)
        for model,q in preds.items():
            arrays[market+'_'+model]=q;arrays[market+'_'+model+'_prob_nonzero']=probs[model]
            arrays[market+'_'+model+'_CRPS']=crps[model]
            for i,t in enumerate(D['t'][te]):
                row=dict(market=market,model=model,origin=int(t),origin_date=str(d.date.iloc[t].date()),
                    target_date=str(d.date.iloc[t+HOR[hi[i]]].date()),horizon=int(HOR[hi[i]]),target=float(y[i]),
                    scale=float(sc[i]),prob_nonzero=float(probs[model][i]),local_prior=float(D['local'][te[i]]),
                    CRPS=float(crps[model][i]))
                row.update({f'q{float(tau):.3f}':float(q[i,j]) for j,tau in enumerate(TAUS)})
                rows.append(row)
        report['markets'][market]=mres;report['summary']=summarize(report)
        report['elapsed_seconds']=time.perf_counter()-start
        (HERE/(args.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        np.savez_compressed(HERE/(args.tag+'_predictions.npz'),**arrays)
        pd.DataFrame(rows).to_csv(HERE/(args.tag+'_predictions.csv'),index=False)
        print(market,'completed',round(time.perf_counter()-start,1),flush=True)
    report['elapsed_seconds']=time.perf_counter()-start
    (HERE/(args.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report['summary'].items() if k.startswith('rolling_neural_prior_crps')},indent=2),flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
