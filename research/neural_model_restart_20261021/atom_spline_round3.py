# -*- coding: utf-8 -*-
"""Round3 direct mark-quantile spline + zero atom. All CRPS integrals exact.
Only atom_spline_round3* is written. Reused historical test periods: exploratory.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import argparse,json,math,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.integrate import quad
import marked_event_round2 as old
from neural_restart import PANEL,CUT,TAUS,score,wis
HERE=Path(__file__).resolve().parent
U=np.array([0,.005,.01,.025,.05,.075,.1,.125,.15,.175,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.825,.85,.875,.9,.925,.95,.975,.99,.995,1],float)
MODELS=['continuous_spline','static_atom_spline','rule_atom_spline','neural_atom_spline','neural_atom_no_price_calendar']


def mark_cdf(x,k,u):
    j=torch.searchsorted(k.contiguous(),x.contiguous(),right=True).clamp(1,k.shape[1]-1)-1
    lo=k.gather(1,j);hi=k.gather(1,j+1)
    return (u[j]+(u[j+1]-u[j])*(x-lo)/(hi-lo).clamp_min(1e-12)).clamp(0,1)


def exact_crps(y,knots,ps,u):
    """CRPS of equal mixture of spline+atom distributions, exact on merged knots."""
    xx=torch.sort(torch.cat(knots+[torch.zeros_like(y[:,None])],1),dim=1).values
    mark=sum(p[:,None]*mark_cdf(xx,k,u) for k,p in zip(knots,ps))/len(knots)
    atom=1-torch.stack(ps).mean(0)
    right=mark+atom[:,None]*(xx>=0);left=mark+atom[:,None]*(xx>0)
    dx=xx[:,1:]-xx[:,:-1];A=right[:,:-1];B=left[:,1:]-A
    v=((y[:,None]-xx[:,:-1])/dx.clamp_min(1e-12)).clamp(0,1)
    val=dx*(A*A*v+A*B*v*v+B*B*v**3/3+(A-1)**2*(1-v)+(A-1)*B*(1-v*v)+B*B*(1-v**3)/3)
    return val.sum(1)+(xx[:,0]-y).clamp_min(0)+(y-xx[:,-1]).clamp_min(0)


def analytic_quantiles(k,p,u,taus=None):
    tau=torch.tensor(TAUS,dtype=k.dtype,device=k.device) if taus is None else taus
    f0=mark_cdf(torch.zeros_like(p[:,None]),k,u).squeeze(1)
    a=p*f0;b=a+1-p
    tt=tau[None,:].expand(len(k),-1)
    uu=torch.where(tt<a[:,None],tt/p[:,None].clamp_min(1e-12),(tt-(1-p[:,None]))/p[:,None].clamp_min(1e-12)).clamp(0,1)
    j=(torch.searchsorted(u,uu.contiguous(),right=True)-1).clamp(0,len(u)-2)
    q=k.gather(1,j)+(k.gather(1,j+1)-k.gather(1,j))*(uu-u[j])/(u[j+1]-u[j])
    return torch.where((tt>=a[:,None])&(tt<=b[:,None]),torch.zeros_like(q),q)


def ensemble_quantiles(ks,ps,u):
    lo=torch.min(torch.cat(ks+[torch.zeros_like(ps[0][:,None])],1),dim=1).values[:,None].expand(-1,len(TAUS)).clone()
    hi=torch.max(torch.cat(ks+[torch.zeros_like(ps[0][:,None])],1),dim=1).values[:,None].expand_as(lo).clone()
    tau=torch.tensor(TAUS,dtype=ks[0].dtype)[None,:]
    for _ in range(34):
        mid=(lo+hi)/2
        f=sum(p[:,None]*mark_cdf(mid,k,u)+(1-p[:,None])*(mid>=0) for k,p in zip(ks,ps))/len(ks)
        test=f<tau;lo=torch.where(test,mid,lo);hi=torch.where(test,hi,mid)
    q=(lo+hi)/2
    a=sum(p[:,None]*mark_cdf(torch.zeros_like(p[:,None]),k,u) for k,p in zip(ks,ps))/len(ks)
    b=a+1-torch.stack(ps).mean(0)[:,None]
    return torch.where((tau>=a)&(tau<=b),torch.zeros_like(q),q)


def checks():
    u=torch.tensor(U,dtype=torch.float64);k=(-1+3*u)[None,:];y=torch.tensor([.3],dtype=torch.float64)
    uniform=exact_crps(y,[k],[torch.ones(1,dtype=torch.float64)],u)
    expected=(.3+1)**2/3-(.3+1)+1
    assert abs(float(uniform[0])-expected)<1e-10
    atom=exact_crps(y,[k],[torch.zeros(1,dtype=torch.float64)],u)
    assert abs(float(atom[0])-.3)<1e-10
    p=torch.tensor([.6],dtype=torch.float64)
    def cdf(x):return .6*np.clip((x+1)/3,0,1)+.4*(x>=0)
    numerical=sum(quad(lambda x:(cdf(x)-(x>=.3))**2,a,b,epsabs=1e-10)[0] for a,b in [(-np.inf,-1),(-1,0),(0,.3),(.3,2),(2,np.inf)])
    exact=float(exact_crps(y,[k],[p],u)[0]);assert abs(numerical-exact)<1e-9
    q=analytic_quantiles(k,p,u,torch.tensor([.1,.4,.9],dtype=torch.float64))
    assert np.allclose(q.numpy(),[[-.5,0,1.5]],atol=1e-10)
    qb=ensemble_quantiles([k],[p],u);qa=analytic_quantiles(k,p,u);assert float((qb-qa).abs().max())<1e-8
    ug=torch.tensor([0,.2,.7,1.],dtype=torch.float64)
    kg=torch.tensor([[-1.7,-.6,.45,1.9]],dtype=torch.float64,requires_grad=True)
    pg=torch.tensor([.67],dtype=torch.float64,requires_grad=True)
    ok=torch.autograd.gradcheck(lambda kk,pp:exact_crps(torch.tensor([.23],dtype=torch.float64),[kk],[pp],ug).sum(),(kg,pg),eps=1e-6,atol=1e-5,rtol=1e-4)
    k2=(-.3+2*u)[None,:];p2=torch.tensor([.8],dtype=torch.float64)
    def cdf2(x):return .5*cdf(x)+.5*(.8*np.clip((x+.3)/2,0,1)+.2*(x>=0))
    en=sum(quad(lambda x:(cdf2(x)-(x>=.3))**2,a,b,epsabs=1e-10)[0] for a,b in [(-np.inf,-1),(-1,-.3),(-.3,0),(0,.3),(.3,1.7),(1.7,2),(2,np.inf)])
    ee=float(exact_crps(y,[k,k2],[p,p2],u)[0]);assert abs(en-ee)<1e-9
    return dict(uniform_CRPS=float(uniform[0]),atom_CRPS=float(atom[0]),atom_uniform_exact=exact,independent_CDF_integral=numerical,gradient_check=bool(ok),ensemble_exact_CRPS=ee,ensemble_CDF_integral=en,analytic_quantile_check=True)


def build(d):
    D=old.build(d);lp=np.log(d.price.to_numpy(float));prices=d.price.to_numpy(float);extra=[]
    for t in D['t']:
        a=prices[max(0,t-119):t+1];b=prices[max(0,t-249):t+1];lb=np.log(b)
        day=d.date.iloc[t];phase=2*np.pi*(day.dayofyear-1)/(366 if day.is_leap_year else 365)
        extra.append([lp[t],np.log(prices[t]/a.mean()),np.log(prices[t]/b.mean()),(lp[t]-lb.min())/(lb.max()-lb.min()+1e-8),np.sin(phase),np.cos(phase),np.sin(2*phase),np.cos(2*phase)])
    D['x']=np.column_stack([D['x'],np.array(extra,np.float32)]).astype(np.float32)
    D['audit']['augmented_information']=['origin log price','log price/mean120','log price/mean250','position within log-price range250','origin day-of-year sin/cos and second harmonics']
    D['audit']['early_250_window']='available prefix up to250 at early training origins; no future observations'
    return D


def base_fit(D,idx,continuous):
    knots=[];priors=[];scales=[];adjust=[]
    for h in range(4):
        z=D['z'][idx[D['hi'][idx]==h]];nz=z[z!=0]
        p=(len(nz)+1)/(len(z)+2);sample=z if continuous else nz
        if not len(sample):sample=np.array([-.01,.01])
        raw=np.quantile(sample,U);sd=max(float(np.std(sample)),.05);eps=1e-5*sd+1e-6
        q=raw.copy()
        for j in range(1,len(q)):q[j]=max(q[j],q[j-1]+eps)
        knots.append(q);priors.append(p);scales.append(sd);adjust.append(float(np.max(np.abs(q-raw))))
    return np.array(knots,np.float32),np.array(priors,np.float32),np.array(scales,np.float32),adjust


class SplineNet(nn.Module):
    def __init__(self,nf,base,p0,scales,model):
        super().__init__();self.trunk=nn.Sequential(nn.Linear(nf,24),nn.Tanh(),nn.Linear(24,16),nn.Tanh());self.prior_head=nn.Linear(16,2);self.shape_head=nn.Linear(16,5)
        for layer in [self.prior_head,self.shape_head]:nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        self.register_buffer('base',torch.tensor(base));self.register_buffer('p0',torch.tensor(p0));self.register_buffer('scales',torch.tensor(scales));self.model=model
        uu=(U[:-1]+U[1:])-1
        self.register_buffer('basis',torch.tensor(np.stack([np.ones_like(uu),uu,uu**2-1/3,uu**3]),dtype=torch.float32))
    def forward(self,x,hi,local):
        if self.model=='neural_atom_no_price_calendar':
            x=x.clone();x[:,-8:]=0
        hidden=self.trunk(x);out=self.shape_head(hidden);base=self.base[hi]
        change=1.5*torch.tanh(out[:,1:]@self.basis)
        gaps=(base[:,1:]-base[:,:-1])*torch.exp(change)
        cum=torch.cat([torch.zeros_like(gaps[:,:1]),torch.cumsum(gaps,1)],1)
        q=base[:,16:17]+1.5*self.scales[hi,None]*torch.tanh(out[:,:1])+cum-cum[:,16:17]
        static=self.p0[hi]
        if self.model=='continuous_spline':p=torch.ones_like(static)
        elif self.model=='static_atom_spline':p=static
        elif self.model=='rule_atom_spline':p=.2*static+.8*local
        else:
            pr=self.prior_head(hidden);gate=torch.sigmoid(math.log(4)+pr[:,0]);prior=(1-gate)*static+gate*local
            p=torch.sigmoid(torch.logit(prior.clamp(1e-6,1-1e-6))+.75*torch.tanh(pr[:,1]))
        return q,p


def fit_predict(D,model,seed,epochs):
    y=torch.tensor(D['z'],dtype=torch.float32);hi=torch.tensor(D['hi'],dtype=torch.long);local=torch.tensor(D['local']);u=torch.tensor(U,dtype=torch.float32)
    def setup(idx):
        torch.manual_seed(seed);mean=D['x'][idx].mean(0);sd=D['x'][idx].std(0)+1e-5;x=torch.tensor((D['x']-mean)/sd)
        base,p,sc,adj=base_fit(D,idx,model=='continuous_spline');net=SplineNet(x.shape[1],base,p,sc,model);opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.002)
        return x,net,opt,adj
    def epoch(x,net,opt,idx):
        order=torch.tensor(idx)[torch.randperm(len(idx))]
        for b in order.split(512):
            opt.zero_grad();k,p=net(x[b],hi[b],local[b]);loss=exact_crps(y[b],[k],[p],u).mean();assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
    def val(x,net):
        with torch.no_grad():
            k,p=net(x[D['va']],hi[D['va']],local[D['va']]);q=analytic_quantiles(k,p,u).numpy()
        return float(score(D['z'][D['va']],q).mean())
    x,net,opt,adj=setup(D['tr']);best=val(x,net);bep=0;bad=0;hist=[[0,best]]
    for ep in range(1,epochs+1):
        epoch(x,net,opt,D['tr'])
        if ep%2 and ep!=1 and ep!=epochs:continue
        vl=val(x,net);hist.append([ep,vl])
        if vl<best-1e-6:best,bep,bad=vl,ep,0
        else:bad+=1
        if bad>=7 and ep>=18:break
    x,net,opt,adjrf=setup(D['rf'])
    for _ in range(bep):epoch(x,net,opt,D['rf'])
    with torch.no_grad():k,p=net(x[D['te']],hi[D['te']],local[D['te']])
    return k,p,dict(seed=seed,best_epoch=bep,inner_epochs=ep,refit_epochs=bep,validation_MQS_normalized=best,history=hist,initial_spacing_max_adjustment_train=adj,initial_spacing_max_adjustment_refit=adjrf,allocated_parameters=sum(t.numel() for t in net.parameters()))


def summarize(r):
    pairs=[('neural_atom_spline','continuous_spline'),('neural_atom_spline','rule_atom_spline'),('neural_atom_spline','static_atom_spline'),('neural_atom_spline','neural_atom_no_price_calendar'),('neural_atom_spline','double_normal'),('neural_atom_spline','empirical_constant'),('neural_atom_spline','gaussian_constant')]
    pairs += [(m,'empirical_constant') for m in MODELS if m!='neural_atom_spline']
    cells=[c['scores'] for v in r['markets'].values() for c in v['by_horizon'].values()];out={}
    for m,b in pairs:
        out[m+'_vs_'+b]={}
        for metric in ['MQS','CRPS','Brier']:
            rr=[c[m][metric]/max(c[b][metric],1e-15) for c in cells];out[m+'_vs_'+b][metric]=dict(geometric_gain_percent=float(100*(1-np.exp(np.log(np.maximum(rr,1e-15)).mean()))),wins=int(np.sum(np.array(rr)<1)),cells=len(rr))
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--markets',nargs='+',default=list(CUT));ap.add_argument('--seeds',nargs='+',type=int,default=[11,22]);ap.add_argument('--epochs',type=int,default=48);ap.add_argument('--tag',default='atom_spline_round3');args=ap.parse_args();assert args.tag.startswith('atom_spline_round3')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);san=checks();print('checks',json.dumps(san),flush=True)
    panel=pd.read_csv(PANEL,parse_dates=['date']);previous=np.load(HERE/'marked_event_round2_predictions.npz');start=time.perf_counter();u=torch.tensor(U,dtype=torch.float32)
    r=dict(status='exploratory reused test periods; no novelty or publication success established',config=vars(args),u_knots=U.tolist(),sanity=san,distribution='direct conditional continuous mark piecewise-linear quantile plus zero atom; no moment/budget coupling',CRPS='exact CDF linear-segment integral on all merged knots and zero, including cross-seed effects',selection='same normalized validation MQS, epoch0 allowed; selected epoch full-precut refit',information='same augmented features for four main variants; last8 standardized features zeroed in no-price-calendar control',initialization='atom marks use fitting nonzero empirical quantiles; continuous uses all fitting returns; positive-gap floor recorded',panel_sha256=hashlib.sha256(PANEL.read_bytes()).hexdigest(),markets={})
    arrays={};rows=[]
    for market in args.markets:
        data=panel[panel.market==market].sort_values('date').reset_index(drop=True);D=build(data);te=D['te'];hi=D['hi'][te];y=D['y'][te];scale=D['scale'][te];assert np.array_equal(D['t'][te],previous[market+'_origin']) and np.array_equal(y,previous[market+'_target'])
        mapping={'double_normal':'rolling_neural_prior_crps','empirical_constant':'empirical_constant','gaussian_constant':'gaussian_constant'}
        pred={m:previous[market+'_'+b] for m,b in mapping.items()};cr={m:previous[market+'_'+b+'_CRPS'] for m,b in mapping.items()};prob={m:previous[market+'_'+b+'_prob_nonzero'] for m,b in mapping.items()};fits={}
        for model in MODELS:
            ks=[];ps=[];fs=[]
            for seed in args.seeds:
                k,p,f=fit_predict(D,model,seed,args.epochs);ks.append(k);ps.append(p);fs.append(f);arrays[f'{market}_{model}_seed{seed}_knots']=k.numpy();arrays[f'{market}_{model}_seed{seed}_p']=p.numpy();print(market,model,seed,'epoch',f['best_epoch'],'elapsed',round(time.perf_counter()-start,1),flush=True)
            with torch.no_grad():
                qq=ensemble_quantiles(ks,ps,u).numpy();cc=exact_crps(torch.tensor(D['z'][te],dtype=torch.float32),ks,ps,u).numpy()
            assert np.isfinite(qq).all() and np.all(np.diff(qq,axis=1)>=-1e-5) and np.isfinite(cc).all() and cc.min()>-1e-4
            pred[model]=qq*scale[:,None];cr[model]=cc*scale;prob[model]=torch.stack(ps).mean(0).numpy();fits[model]=fs
        res=dict(audit=D['audit'],fits=fits,by_horizon={})
        for j,h in enumerate(old.HOR):
            mask=hi==j;ss={}
            for model,q in pred.items():
                yy,qq,pp=y[mask],q[mask],prob[model][mask];bb=(yy!=0);ss[model]=dict(MQS=float(score(yy,qq).mean()),CRPS=float(cr[model][mask].mean()),WIS=float(wis(yy,qq).mean()),coverage80=float(((yy>=qq[:,2])&(yy<=qq[:,12])).mean()),coverage95=float(((yy>=qq[:,0])&(yy<=qq[:,14])).mean()),Brier=float(((pp-bb)**2).mean()),mean_nonzero_probability=float(pp.mean()))
            res['by_horizon'][str(h)]=dict(n=int(mask.sum()),scores=ss)
        for suffix,value in [('origin',D['t'][te]),('horizon',old.HOR[hi]),('target',y),('scale',scale)]:arrays[market+'_'+suffix]=value
        for model,q in pred.items():
            arrays[market+'_'+model]=q;arrays[market+'_'+model+'_CRPS']=cr[model];arrays[market+'_'+model+'_prob_nonzero']=prob[model]
            for i,t in enumerate(D['t'][te]):
                row=dict(market=market,model=model,origin=int(t),origin_date=str(data.date.iloc[t].date()),target_date=str(data.date.iloc[t+old.HOR[hi[i]]].date()),horizon=int(old.HOR[hi[i]]),target=float(y[i]),scale=float(scale[i]),prob_nonzero=float(prob[model][i]),CRPS=float(cr[model][i]));row.update({f'q{float(tau):.3f}':float(q[i,z]) for z,tau in enumerate(TAUS)});rows.append(row)
        r['markets'][market]=res;r['summary']=summarize(r);r['elapsed_seconds']=time.perf_counter()-start
        (HERE/(args.tag+'.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');np.savez_compressed(HERE/(args.tag+'_predictions.npz'),**arrays);pd.DataFrame(rows).to_csv(HERE/(args.tag+'_predictions.csv'),index=False)
        print(market,'complete',round(time.perf_counter()-start,1),flush=True)
    r['elapsed_seconds']=time.perf_counter()-start;(HERE/(args.tag+'.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(r['summary'],indent=2),flush=True)

if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');main()
