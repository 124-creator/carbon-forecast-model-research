# -*- coding: utf-8 -*-
"""Round 2: low-rank uncertainty experts and auxiliary expert training.

The candidate keeps separate location/spacing routes. A three-dimensional gap
basis reduces the task of learning fourteen unrelated spacing corrections.
The expert auxiliary loss is proper pinball on the same target and is used only
for optimization. Hyperparameters and epochs are selected using inner validation;
all selected models are then refitted on all available pre-cut labels.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.stats import norm
from threadpoolctl import threadpool_limits
import multiscale_round1 as M
import neural_restart as B

P=Path(__file__).resolve().parent


class AuxiliaryRouter(M.ScaleRouter):
    def __init__(self,base,variant,rank=3):
        super().__init__(base,variant)
        self.rank=rank
        if rank==3:
            mid=(M.TAUS[1:]+M.TAUS[:-1])/2
            basis=np.stack([np.ones(len(mid)),2*(mid-.5),
                            np.abs(norm.ppf(mid))/2],1).astype(np.float32)
            self.register_buffer('basis',torch.tensor(basis))
            self.expert=nn.ModuleList([nn.Sequential(nn.Linear(32,24),nn.Tanh(),nn.Linear(24,4))
                                      for _ in M.SCALES])
            for head in self.expert:
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
        # Begin with uniform assignment; experts initially share the same anchor.
        for route in [self.route_location,self.route_gaps]:
            nn.init.zeros_(route[-1].weight)
            nn.init.zeros_(route[-1].bias)

    def forward_all(self,seq,ctx,hi):
        if self.variant=='single_scale':
            seq=seq[:,0:1].expand(-1,len(M.SCALES),-1,-1)
        full=torch.cat([ctx,self.hlog[hi,None]],1)
        cc=self.context(full)
        e=torch.stack([self.expert[j](torch.cat([self.enc[j](seq[:,j].flatten(1)),cc],1))
                       for j in range(len(M.SCALES))],1)
        if self.variant=='uniform':
            wl=wg=torch.full((len(hi),len(M.SCALES)),1/len(M.SCALES))
        else:
            wl=self.route_location(full).softmax(1)
            wg=wl if self.variant=='shared_route' else self.route_gaps(full).softmax(1)
        gaplog=e[:,:,1:]@self.basis.T if self.rank==3 else e[:,:,1:]
        base=self.base[hi]
        expert_gaps=(base[:,None,1:]-base[:,None,:-1])*torch.exp(.8*torch.tanh(gaplog))
        expert_medians=base[:,None,7]+e[:,:,0]
        def decode(m,g):
            cum=torch.cat([torch.zeros((*g.shape[:-1],1)),g.cumsum(-1)],-1)
            return m[...,None]+cum-cum[...,7:8]
        q=decode((wl*expert_medians).sum(1),(wg[:,:,None]*expert_gaps).sum(1))
        eq=decode(expert_medians,expert_gaps)
        return q,eq,wl,wg

    def forward(self,seq,ctx,hi):
        q,_,wl,wg=self.forward_all(seq,ctx,hi)
        return q,wl,wg


def optimize(net,X,indices,epochs,seed,lr,aux,validation=None):
    opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=.002)
    torch.manual_seed(seed+180)
    ids=torch.tensor(indices)
    def forward(idx):
        return net.forward_all(X['seq'][idx],X['ctx'][idx],X['hi'][idx])
    def vl():
        with torch.no_grad():
            q=forward(validation)[0]
            ls=M.losses(X['z'][validation],q)
            return float(torch.stack([ls[X['hi'][validation]==h].mean() for h in range(len(M.HOR))]).mean())
    net.eval()
    best=vl() if validation is not None else None
    initial=best
    state,epbest,bad=copy.deepcopy(net.state_dict()),0,0
    for ep in range(epochs):
        net.train()
        order=ids[torch.randperm(len(ids))]
        for idx in order.split(512):
            q,eq,_,_=forward(idx)
            ls=M.losses(X['z'][idx],q)
            if aux:
                el=torch.stack([M.losses(X['z'][idx],eq[:,j]) for j in range(len(M.SCALES))],1).mean(1)
                ls=(ls+aux*el)/(1+aux)
            loss=(ls*X['weight'][idx]).mean()
            assert torch.isfinite(loss)
            opt.zero_grad();loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step()
        if validation is not None:
            net.eval();value=vl()
            if value<best-1e-5:
                best,epbest,bad=value,ep+1,0
                state=copy.deepcopy(net.state_dict())
            else:
                bad+=1
            if ep>=49 and bad>=30:
                break
    if validation is not None:
        net.load_state_dict(state)
    net.eval()
    return {'best_epoch':epbest,'best_validation':best,'epoch0_validation':initial,
            'epochs_run':ep+1 if epochs else 0}


def selected_fit(D,variant,lrs,auxs,epochs,seeds):
    X,base=M.prepare(D,False)
    trials=[]
    selected=None
    # Same tuning budget for every architecture. Pilot seed is fixed at 11.
    for lr in lrs:
        for aux in auxs:
            torch.manual_seed(11)
            net=AuxiliaryRouter(base,variant)
            fit=optimize(net,X,D['indices']['train'],epochs,11,lr,aux,
                         torch.tensor(D['indices']['validation']))
            fit.update(lr=lr,aux_weight=aux,selection_seed=11)
            trials.append(fit)
            if selected is None or fit['best_validation']<selected['best_validation']-1e-7:
                selected=fit
    Xf,basef=M.prepare(D,True)
    te=torch.tensor(D['indices']['test'])
    Q,logs=[],[]
    for seed in seeds:
        torch.manual_seed(seed)
        final=AuxiliaryRouter(basef,variant)
        optimize(final,Xf,D['indices']['development'],selected['best_epoch'],seed,
                 selected['lr'],selected['aux_weight'])
        with torch.no_grad():
            q,eq,wl,wg=final.forward_all(Xf['seq'][te],Xf['ctx'][te],Xf['hi'][te])
        pred=q.numpy()*D['scale'][te.numpy(),None]
        assert np.isfinite(pred).all() and (np.diff(pred,axis=1)>=-1e-6).all()
        Q.append(pred)
        logs.append({'seed':seed,'refit_epochs':selected['best_epoch'],
                     'route_difference_L1':float((wl-wg).abs().sum(1).mean()),
                     'mean_location_weights':wl.mean(0).tolist(),'mean_gap_weights':wg.mean(0).tolist(),
                     'expert_quantile_disagreement':float(eq.std(1).mean()),
                     'parameter_count':sum(p.numel() for p in final.parameters())})
    return np.mean(Q,axis=0),{'selected':selected,'validation_trials':trials,'final_fits':logs},Q


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--markets',nargs='+',default=list(B.CUT))
    ap.add_argument('--variants',nargs='+',default=['uniform','shared_route','dual_route','single_scale'])
    ap.add_argument('--epochs',type=int,default=160)
    ap.add_argument('--seeds',nargs='+',type=int,default=[11,22])
    ap.add_argument('--lrs',nargs='+',type=float,default=[.0003,.001])
    ap.add_argument('--auxs',nargs='+',type=float,default=[0.,.5])
    ap.add_argument('--tag',default='multiscale_round2')
    a=ap.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    panel=pd.read_csv(B.PANEL,parse_dates=['date'])
    oldmeta=json.loads((P/'multiscale_round1.json').read_text(encoding='utf-8'))
    phash=hashlib.sha256(B.PANEL.read_bytes()).hexdigest()
    assert oldmeta['panel_sha256']==phash
    old=np.load(P/'multiscale_round1_predictions.npz')
    report={'status':'exploratory; historically reused tests; no novelty/significance claim',
            'config':vars(a),'horizons':M.HOR,'scales':M.SCALES,'gap_rank':3,
            'selection_seed':11,'select_hyperparameters_from':'inner validation only',
            'refit_all_pretest_labels':True,'panel_sha256':phash,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'reused_baselines':'multiscale_round1_predictions.npz; arrays and panel hash checked',
            'markets':{}}
    arrays={};start=time.perf_counter()
    for market in a.markets:
        D=M.extract_data(panel[panel.market==market].sort_values('date').reset_index(drop=True))
        te=D['indices']['test'];y=D['y'][te];hi=D['hi'][te]
        assert np.array_equal(D['t'][te],old[market+'_origin'])
        assert np.array_equal(y,old[market+'_target'])
        assert np.array_equal(np.array(M.HOR)[hi],old[market+'_horizon'])
        pred={name:old[market+'_'+name] for name in ['empirical_anchor','gaussian_rw','ridge_quantile','hgb_quantile']}
        pred['previous_dual_route']=old[market+'_dual_route']
        result={'audit':D['audit'],'fits':{},'by_horizon':{}}
        for variant in a.variants:
            q,fit,byseed=selected_fit(D,variant,a.lrs,a.auxs,a.epochs,a.seeds)
            pred[variant]=q;result['fits'][variant]=fit
            for seed,sq in zip(a.seeds,byseed):
                arrays[market+'_'+variant+'_seed'+str(seed)]=sq
            print(market,variant,fit['selected'],'elapsed_s',round(time.perf_counter()-start,1),flush=True)
        for hidx,h in enumerate(M.HOR):
            keep=hi==hidx;scores={}
            for name,q in pred.items():
                yh,qh=y[keep],q[keep]
                scores[name]={'MQS':float(B.score(yh,qh).mean()),'WIS':float(B.wis(yh,qh).mean()),
                              'MAE':float(np.abs(yh-qh[:,7]).mean()),
                              'coverage80':float(np.mean((yh>=qh[:,2])&(yh<=qh[:,12]))),
                              'coverage95':float(np.mean((yh>=qh[:,0])&(yh<=qh[:,14])))}
            result['by_horizon'][str(h)]={'n':int(keep.sum()),'scores':scores}
        report['markets'][market]=result
        arrays[market+'_origin']=D['t'][te];arrays[market+'_horizon']=np.array(M.HOR)[hi]
        arrays[market+'_target']=y
        for name,q in pred.items():
            arrays[market+'_'+name]=q
        (P/(a.tag+'.json')).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
        np.savez_compressed(P/(a.tag+'_predictions.npz'),**arrays)
    ref='dual_route' if 'dual_route' in a.variants else a.variants[-1]
    report['summary']={}
    for base in pred:
        if base==ref:continue
        r=[cell['scores'][ref]['MQS']/cell['scores'][base]['MQS']
           for m in report['markets'].values() for cell in m['by_horizon'].values()]
        report['summary'][ref+'_vs_'+base]={'mean_log_MQS_ratio':float(np.log(r).mean()),
          'geometric_MQS_gain_percent':float(100*(1-np.exp(np.log(r).mean()))),
          'median_MQS_ratio':float(np.median(r)),'wins':int(np.sum(np.array(r)<1)),'cells':len(r)}
    report['elapsed_seconds']=time.perf_counter()-start
    (P/(a.tag+'.json')).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with threadpool_limits(limits=2):main()
