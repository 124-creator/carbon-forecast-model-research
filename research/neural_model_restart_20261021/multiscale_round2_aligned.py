# -*- coding: utf-8 -*-
"""A bounded follow-up: scale-specific information and raw-score-aligned risk.

All inner choices remain chronological. Compared to round2, experts cannot use
shared momentum/volatility context to bypass their scale inputs; routes still
observe that state. Selection minimizes log relative raw quantile risk, matching
the score aggregation used in reports instead of varying per-origin weights.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits
import neural_restart as B
import multiscale_round1 as M
import multiscale_round2 as R

P=Path(__file__).resolve().parent


class HorizonContext(nn.Module):
    def forward(self,x):
        y=torch.zeros_like(x)
        y[:,-1]=x[:,-1]
        return y


class RestrictedRouter(R.AuxiliaryRouter):
    def __init__(self,base,variant):
        super().__init__(base,variant)
        self.context=nn.Sequential(HorizonContext(),self.context)


def phase(net,X,base,D,indices,epochs,seed,lr,validation=None):
    ids=torch.tensor(indices)
    opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=.002)
    torch.manual_seed(seed+180)
    scale=torch.tensor(D['scale'])
    def forward(idx):return net.forward_all(X['seq'][idx],X['ctx'][idx],X['hi'][idx])
    with torch.no_grad():
        bq=torch.tensor(base)[X['hi'][ids]]
        rawbase=M.losses(X['z'][ids],bq)*scale[ids]
        denom=torch.stack([rawbase[X['hi'][ids]==h].mean() for h in range(len(M.HOR))]).clamp_min(1e-8)
        weights=X['weight']*scale/denom[X['hi']]
    if validation is not None:
        with torch.no_grad():
            vbase=M.losses(X['z'][validation],torch.tensor(base)[X['hi'][validation]])*scale[validation]
            vdenom=torch.stack([vbase[X['hi'][validation]==h].mean() for h in range(len(M.HOR))]).clamp_min(1e-8)
    def val():
        with torch.no_grad():
            q=forward(validation)[0]
            loss=M.losses(X['z'][validation],q)*scale[validation]
            risks=torch.stack([loss[X['hi'][validation]==h].mean() for h in range(len(M.HOR))])
            return float(torch.log((risks/vdenom).clamp_min(1e-8)).mean())
    net.eval();best=val() if validation is not None else None
    saved,epbest,bad=copy.deepcopy(net.state_dict()),0,0
    for ep in range(epochs):
        net.train();order=ids[torch.randperm(len(ids))]
        for idx in order.split(512):
            q,eq,_,_=forward(idx)
            main=M.losses(X['z'][idx],q)
            aux=torch.stack([M.losses(X['z'][idx],eq[:,j]) for j in range(len(M.SCALES))],1).mean(1)
            loss=((main+.5*aux)/1.5*weights[idx]).mean()
            assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        if validation is not None:
            net.eval();vl=val()
            if vl<best-1e-5:
                best,epbest,bad=vl,ep+1,0;saved=copy.deepcopy(net.state_dict())
            else:bad+=1
            if ep>=49 and bad>=30:break
    if validation is not None:net.load_state_dict(saved)
    net.eval()
    return {'best_epoch':epbest,'validation_log_relative_raw_MQS':best,
            'epochs_run':ep+1 if epochs else 0,'lr':lr,'aux_weight':.5}


def fit(D,variant,lrs,epochs,seeds):
    X,base=M.prepare(D,False);trials=[];chosen=None
    for lr in lrs:
        torch.manual_seed(11);net=RestrictedRouter(base,variant)
        t=phase(net,X,base,D,D['indices']['train'],epochs,11,lr,torch.tensor(D['indices']['validation']))
        trials.append(t)
        if chosen is None or t['validation_log_relative_raw_MQS']<chosen['validation_log_relative_raw_MQS']-1e-7:chosen=t
    Xf,bf=M.prepare(D,True);te=torch.tensor(D['indices']['test']);qs=[]
    for seed in seeds:
        torch.manual_seed(seed);n=RestrictedRouter(bf,variant)
        phase(n,Xf,bf,D,D['indices']['development'],chosen['best_epoch'],seed,chosen['lr'])
        with torch.no_grad():q=n(Xf['seq'][te],Xf['ctx'][te],Xf['hi'][te])[0].numpy()*D['scale'][te.numpy(),None]
        assert np.isfinite(q).all() and (np.diff(q,axis=1)>=-1e-6).all()
        qs.append(q)
    return np.mean(qs,axis=0),{'selected':chosen,'trials':trials,'refit_seeds':seeds}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=160)
    ap.add_argument('--markets',nargs='+',default=list(B.CUT));ap.add_argument('--seeds',nargs='+',type=int,default=[11,22])
    ap.add_argument('--variants',nargs='+',default=['uniform','shared_route','dual_route','single_scale'])
    ap.add_argument('--lrs',nargs='+',type=float,default=[.0003,.001]);ap.add_argument('--tag',default='multiscale_round2_aligned')
    a=ap.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    panel=pd.read_csv(B.PANEL,parse_dates=['date']);old=np.load(P/'multiscale_round1_predictions.npz')
    report={'status':'exploratory; reused tests; not a publication claim','config':vars(a),
            'expert_context':'horizon only; shared market state is exclusive to routers',
            'training':'raw pinball with fixed training-horizon risk normalizers and auxiliary expert loss',
            'selection':'mean log relative RAW quantile risk over horizons',
            'refit':True,'panel_sha256':hashlib.sha256(B.PANEL.read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'markets':{}}
    arrays={};start=time.perf_counter()
    for market in a.markets:
        D=M.extract_data(panel[panel.market==market].sort_values('date').reset_index(drop=True));te=D['indices']['test'];hi=D['hi'][te];y=D['y'][te]
        assert np.array_equal(D['t'][te],old[market+'_origin']) and np.array_equal(y,old[market+'_target'])
        pred={name:old[market+'_'+name] for name in ['empirical_anchor','gaussian_rw','ridge_quantile','hgb_quantile']}
        result={'audit':D['audit'],'fits':{},'by_horizon':{}}
        for variant in a.variants:
            q,f=fit(D,variant,a.lrs,a.epochs,a.seeds);pred[variant]=q;result['fits'][variant]=f
            print(market,variant,f['selected'],'elapsed_s',round(time.perf_counter()-start,1),flush=True)
        for hidx,h in enumerate(M.HOR):
            keep=hi==hidx
            result['by_horizon'][str(h)]={'n':int(keep.sum()),'scores':{
              name:{'MQS':float(B.score(y[keep],q[keep]).mean()),'WIS':float(B.wis(y[keep],q[keep]).mean()),
                    'MAE':float(np.abs(y[keep]-q[keep,7]).mean())} for name,q in pred.items()}}
        report['markets'][market]=result
        arrays[market+'_origin']=D['t'][te];arrays[market+'_horizon']=np.array(M.HOR)[hi];arrays[market+'_target']=y
        for name,q in pred.items():arrays[market+'_'+name]=q
        (P/(a.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        np.savez_compressed(P/(a.tag+'_predictions.npz'),**arrays)
    reference='dual_route' if 'dual_route' in a.variants else a.variants[-1];report['summary']={}
    for name in pred:
        if name==reference:continue
        ratios=[r['scores'][reference]['MQS']/r['scores'][name]['MQS'] for m in report['markets'].values() for r in m['by_horizon'].values()]
        report['summary'][reference+'_vs_'+name]={'mean_log_MQS_ratio':float(np.log(ratios).mean()),
          'geometric_MQS_gain_percent':float(100*(1-np.exp(np.log(ratios).mean()))),
          'wins':int((np.array(ratios)<1).sum()),'cells':len(ratios)}
    report['elapsed_seconds']=time.perf_counter()-start
    (P/(a.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with threadpool_limits(limits=2):main()
