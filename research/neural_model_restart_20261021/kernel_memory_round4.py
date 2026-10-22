# -*- coding: utf-8 -*-
"""Exploratory frequency-prior neural conditional memory, not frozen round3.

Class-balanced empirical prior preserves rolling zero/nonzero frequency. The
network scores query-key state similarities, maturity age, and response factors.
All retrieved labels are matured. New supports include observed zero records
and one explicit zero-prior item when no zero record is available. This is a
weighted conditional empirical distribution, not a novel distribution family.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits
import empirical_tilt_round3 as T
import neural_restart as B

P=Path(__file__).resolve().parent
GROUPS=[list(range(8)),list(range(8,24)),[24,25,27],list(range(28,32)),list(range(32,36))]


def prepare(D,ids):
    X,_=T.prepare(D,ids,True)
    banks=[]
    for hi,h in enumerate(T.HOR):
        key=ids[D['hi'][ids]==hi]
        key=key[np.argsort(D['z'][key],kind='stable')]
        realzero=np.abs(D['y'][key])<=1e-12
        assert np.sum(~realzero)>32
        z=D['z'][key].astype(np.float32);j=int(np.searchsorted(z,0.))
        support=np.insert(z,j,0.)
        raw=np.column_stack([np.tanh(z/2),np.abs(z)])
        mu=raw.mean(0);sd=raw.std(0)+1e-5
        phi=np.clip((raw-mu)/sd,-3,3)
        dummy=np.clip(-mu/sd,-3,3)
        phi=np.insert(phi,j,dummy,axis=0).astype(np.float32)
        qi=np.flatnonzero(D['hi']==hi);qq=X['x'][qi];kk=X['x'][key]
        vals=[]
        for group in GROUPS:
            dist=torch.cdist(qq[:,group],kk[:,group]).square()/len(group)
            vals.append(torch.exp(-.5*dist))
        age=(X['t'][qi,None]-(X['t'][key]+int(h))[None,:]).clamp_min(0).float()
        vals.append(torch.exp(-age/252.))
        lookup=torch.full((int(D['t'].max())+1,),-1,dtype=torch.long)
        lookup[X['t'][qi]]=torch.arange(len(qi))
        banks.append({'z':torch.tensor(z),'support':torch.tensor(support),
                      'mature':X['t'][key[~realzero]]+int(h),
                      'all_mature':X['t'][key]+int(h),
                      'real_zero':torch.tensor(realzero),'dummy_index':j,
                      'zero_support':torch.tensor(support==0),
                      'phi':torch.tensor(phi),'similarity':torch.stack(vals,2),
                      'lookup':lookup,'p':float(np.mean(~realzero)),
                      'keys_D_indices':key})
    return X,banks


class MemoryNet(nn.Module):
    def __init__(self,nf,variant):
        super().__init__();self.variant=variant
        self.body=nn.Sequential(nn.Linear(nf,24),nn.Tanh(),nn.Linear(24,16),nn.Tanh())
        self.head=nn.Linear(16,8)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def weights(self,x,local,t,bank,causal=False):
        inp=x
        kind=self.variant.replace('_fixed_p','')
        if kind=='horizon_coefficients':
            inp=torch.zeros_like(x);inp[:,26]=x[:,26]
        raw=self.head(self.body(inp))
        coef=torch.cat([1.5*torch.tanh(raw[:,:6]),.35*torch.tanh(raw[:,6:])],1)
        if kind=='response_only':coef=torch.cat([coef[:,:6]*0,coef[:,6:]],1)
        elif kind=='uniform_prior':coef=coef*0
        elif kind=='fixed_kernel':
            coef=coef*0;coef=coef+torch.tensor([1.,1.,1.,1.,1.,0.,0.,0.])
        allowed=(bank['all_mature'][None,:]<=t[:,None]) if causal else torch.ones((len(t),len(bank['z'])),dtype=torch.bool)
        nz=allowed&~bank['real_zero'][None,:];ze=allowed&bank['real_zero'][None,:]
        nnz=nz.sum(1,keepdim=True);nze=ze.sum(1,keepdim=True)
        assert torch.all(nnz>0)
        p=.8*local+.2*bank['p']
        base=p[:,None]*nz/nnz+(1-p)[:,None]*ze/nze.clamp_min(1)
        dbase=torch.where(nze[:,0]==0,1-p,torch.zeros_like(p))[:,None]
        j=bank['dummy_index'];prior=torch.cat([base[:,:j],dbase,base[:,j:]],1)
        qi=bank['lookup'][t];assert torch.all(qi>=0)
        sim=bank['similarity'][qi]
        e=(coef[:,:6,None].transpose(1,2)*sim).sum(2)
        e=torch.cat([e[:,:j],torch.zeros((len(t),1)),e[:,j:]],1)
        e=e+coef[:,6:]@bank['phi'].T
        logits=torch.log(prior.clamp_min(1e-30))+e
        valid=prior>0
        logits=logits.masked_fill(~valid,-1e9)
        w=logits.softmax(1)*valid
        w=w/w.sum(1,keepdim=True)
        if self.variant.endswith('_fixed_p'):
            ze_support=bank['zero_support']
            nz_total=w[:,~ze_support].sum(1,keepdim=True)
            ze_total=w[:,ze_support].sum(1,keepdim=True)
            factor=torch.where(ze_support[None,:],(1-p)[:,None]/ze_total.clamp_min(1e-20),p[:,None]/nz_total.clamp_min(1e-20))
            w=w*factor
        assert torch.isfinite(w).all()
        prob=1-w[:,bank['zero_support']].sum(1)
        assert torch.all(w[~valid]==0)
        if self.variant.endswith('_fixed_p'):assert torch.allclose(prob,p,atol=2e-6,rtol=0)
        return w,prob,coef


def distribution(D,X,banks,weights):
    te=D['te'];q=np.empty((len(te),15));s=np.empty(len(te));p=np.empty(len(te))
    for hi,bank in enumerate(banks):
        ix=np.flatnonzero(D['hi'][te]==hi);w=weights[hi]
        with torch.no_grad():
            q[ix]=T.quantiles(w,bank['support']).numpy()*D['scale'][te[ix],None]
            s[ix]=T.crps(X['z'][te[ix]],w,bank['support']).numpy()*D['scale'][te[ix]]
            p[ix]=(1-w[:,bank['zero_support']].sum(1)).numpy()
    assert np.isfinite(q).all() and np.isfinite(s).all() and np.min(np.diff(q,axis=1))>=-1e-8
    return q,s,p


def fit(D,variant,seeds,epochs,Xi,bi,Xf,bf):
    sums={hi:None for hi in range(4)};fits=[];seedpred={}
    for seed in seeds:
        torch.manual_seed(seed);net=MemoryNet(36,variant)
        if variant in ['uniform_prior','fixed_kernel']:
            info={'best_epoch':0,'epochs_run':0,'fixed_control':True}
        else:info=T.optimize(net,Xi,bi,D['tr'],epochs,seed,torch.tensor(D['va']))
        torch.manual_seed(seed);final=MemoryNet(36,variant)
        if info['best_epoch']:T.optimize(final,Xf,bf,D['rf'],info['best_epoch'],seed)
        ws={}
        with torch.no_grad():
            for hi,(ii,w,p,coef) in T.forward(final,Xf,torch.tensor(D['te']),bf).items():
                ws[hi]=w.clone()
                sums[hi]=w.clone()/len(seeds) if sums[hi] is None else sums[hi]+w/len(seeds)
        seedpred[seed]=distribution(D,Xf,bf,ws)
        info.update(seed=seed,allocated_parameters=sum(p.numel() for p in net.parameters()))
        fits.append(info)
    return distribution(D,Xf,bf,sums),fits,seedpred,sums


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--markets',nargs='+',default=list(B.CUT))
    ap.add_argument('--variants',nargs='+',default=['uniform_prior','fixed_kernel','response_only_fixed_p','state_kernel_fixed_p','state_kernel','horizon_coefficients_fixed_p'])
    ap.add_argument('--seeds',nargs='+',type=int,default=[11,22,33,44,55]);ap.add_argument('--epochs',type=int,default=64)
    ap.add_argument('--tag',default='kernel_memory_round4');a=ap.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    panel=pd.read_csv(B.PANEL,parse_dates=['date'])
    old=np.load(P/'empirical_tilt_round3_followup_predictions.npz');qrf=np.load(P/'qrf_round4_predictions.npz')
    report={'status':'new exploratory architecture; same historical tests reused','config':vars(a),
            'architecture':'class-frequency prior times learned query-key group-similarity, age, and response energies',
            'zero_mass':'actual zero memories retained; explicit zero prior when no matured zero case exists',
            'causality':'support label maturity mask; fit-partition model statistics are fitted normally',
            'groups':GROUPS,'similarity_bandwidth':'exp(-mean standardized squared group distance/2); age exp(-steps/252)',
            'same_training_queries':'minimum32 matured NONZERO scenarios, matching frozen tilt',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'panel_sha256':hashlib.sha256(B.PANEL.read_bytes()).hexdigest(),'markets':{}}
    arrays={};start=time.perf_counter()
    for market in a.markets:
        d=panel[panel.market==market].sort_values('date').reset_index(drop=True)
        D=T.add_context(d,T.E.build(d));te=D['te'];key=market+'_'
        for source in [old,qrf]:
            assert np.array_equal(D['t'][te],source[key+'origin']) and np.array_equal(D['y'][te],source[key+'target'])
        Xi,bi=prepare(D,D['tr']);Xf,bf=prepare(D,D['rf'])
        pred={};crp={};pr={};fits={};weightout={}
        for variant in a.variants:
            seeds=[11] if variant in ['uniform_prior','fixed_kernel'] else a.seeds
            (q,c,p),f,each,ws=fit(D,variant,seeds,a.epochs,Xi,bi,Xf,bf)
            pred[variant]=q;crp[variant]=c;pr[variant]=p;fits[variant]=f
            for seed,(qq,cc,pp) in each.items():
                arrays[key+variant+f'_seed{seed}_Q']=qq;arrays[key+variant+f'_seed{seed}_CRPS']=cc
            for hi,h in enumerate(T.HOR):
                weightout[variant+f'_weights_h{h}']=ws[hi].numpy()
                weightout[f'support_h{h}']=bf[hi]['support'].numpy()
                weightout[f'test_positions_h{h}']=np.flatnonzero(D['hi'][te]==hi)
            print(market,variant,'epochs',[ff['best_epoch'] for ff in f],
                  'elapsed_s',round(time.perf_counter()-start,1),flush=True)
        for label,source,src in [('frozen_tilt',old,'full_ensemble'),('uniform_rule',old,'uniform_rule'),
                                 ('uniform_static',old,'uniform_static'),('qrf',qrf,'qrf_ensemble')]:
            pred[label]=source[key+src+'_Q'];crp[label]=source[key+src+'_CRPS'];pr[label]=source[key+src+'_p']
        if 'uniform_prior' in crp:
            assert np.allclose(crp['uniform_prior'],crp['uniform_rule'],rtol=1e-4,atol=1e-6)
        cells={};y=D['y'][te];hi=D['hi'][te]
        for hidx,h in enumerate(T.HOR):
            ix=hi==hidx
            cells[str(h)]={'n':int(ix.sum()),'scores':{name:{'MQS':float(B.score(y[ix],q[ix]).mean()),
                     'CRPS':float(crp[name][ix].mean()),'Brier':float(np.mean((pr[name][ix]-(np.abs(y[ix])>1e-12))**2))}
                     for name,q in pred.items()}}
        report['markets'][market]={'audit':D['audit'],'fits':fits,'by_horizon':cells}
        arrays[key+'origin']=D['t'][te];arrays[key+'horizon']=T.HOR[hi];arrays[key+'target']=y
        for name in pred:
            arrays[key+name+'_Q']=pred[name];arrays[key+name+'_CRPS']=crp[name];arrays[key+name+'_p']=pr[name]
        (P/(a.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        np.savez_compressed(P/(a.tag+'_predictions.npz'),**arrays)
        np.savez_compressed(P/(a.tag+'_'+market+'_weights.npz'),**weightout)
    report['summary']={};ref='state_kernel_fixed_p' if 'state_kernel_fixed_p' in pred else ('state_kernel' if 'state_kernel' in pred else a.variants[-1])
    for base in pred:
        if base==ref:continue
        out={}
        for metric in ['MQS','CRPS']:
            rr=np.array([c['scores'][ref][metric]/c['scores'][base][metric]
                         for m in report['markets'].values() for c in m['by_horizon'].values()])
            out[metric]={'geometric_gain_percent':float(100*(1-np.exp(np.log(rr).mean()))),
                         'wins':int((rr<1).sum()),'cells':len(rr)}
        report['summary'][ref+'_vs_'+base]=out
    report['elapsed_seconds']=time.perf_counter()-start
    (P/(a.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with threadpool_limits(limits=2):main()
