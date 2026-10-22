# -*- coding: utf-8 -*-
"""State/horizon routing of different temporal scales for location and spacings.

Exploratory architecture experiment, not a novelty or publication claim.
Model selection uses chronological inner validation. Selected epoch count is
then refitted from scratch using every label available before the outer cut.
All variants use identical input arrays, origin sets, normalization and anchor.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

import neural_restart as B

HERE = Path(__file__).resolve().parent
HOR = [1, 5, 10, 20]
SCALES, K, W, STEP = [1, 5, 20], 12, 250, 2
TAUS = B.TAUS
TT = torch.tensor(TAUS)


def losses(y, q):
    e = y[:, None] - q
    return torch.maximum(TT*e, (TT-1)*e).mean(1)


def extract_data(d, fraction=.75):
    lp = np.log(d.price.to_numpy(float))
    market = str(d.market.iloc[0])
    cut = int(np.searchsorted(d.date.to_numpy(), np.datetime64(B.CUT[market])))
    v0 = max(W+45, int(cut*fraction))
    rows = []
    feature_by_t = {}
    for t in range(W, len(lp)-min(HOR), STEP):
        r = np.diff(lp[t-W:t+1])
        sd = r.std(ddof=1)
        if not np.isfinite(sd) or sd < 1e-8:
            continue
        seqs = []
        for scale in SCALES:
            block = r[-K*scale:].reshape(K, scale)
            z = block.sum(1)/(sd*np.sqrt(scale))
            nz = (np.abs(block)>1e-12).mean(1)
            seqs.append(np.stack([z, np.abs(z), nz], axis=1))
        nz = np.flatnonzero(np.abs(r)>1e-12)
        age = len(r)-1-nz[-1] if len(nz) else len(r)
        ac = np.corrcoef(r[1:], r[:-1])[0, 1] if r[:-1].std()>1e-9 else 0.
        ctx = np.array([np.log(sd), (np.abs(r)<=1e-12).mean(), np.log1p(age),
                        np.log((r[-20:].std(ddof=1)+1e-8)/sd),
                        r[-10:].mean()/sd, r[-60:].mean()/sd, ac,
                        r[-1]/sd], np.float32)
        feature_by_t[t] = (np.array(seqs, np.float32), ctx, float(sd))
    for hi, h in enumerate(HOR):
        for t, (seq, ctx, sd) in feature_by_t.items():
            if t+h >= len(lp):
                continue
            # split 3 holds purged inner-boundary rows, used ONLY in final refit.
            split = (0 if t+h<v0 else (1 if t>=v0 and t+h<cut
                                       else (3 if t+h<cut else (2 if t>=cut else -1))))
            if split < 0:
                continue
            rows.append((t, hi, split, seq, ctx, sd*np.sqrt(h), lp[t+h]-lp[t]))
    c = list(zip(*rows))
    D = {'t':np.array(c[0]), 'hi':np.array(c[1]), 'split':np.array(c[2]),
         'seq':np.stack(c[3]), 'ctx_raw':np.stack(c[4]),
         'scale':np.array(c[5],np.float32), 'y':np.array(c[6],np.float32)}
    D['z'] = D['y']/D['scale']
    tr, va, te, dev = [np.flatnonzero(D['split']==s) for s in [0,1,2,3]]
    dev = np.flatnonzero(D['split'] != 2)
    assert max(D['t'][tr]+np.array(HOR)[D['hi'][tr]]) < v0
    assert max(D['t'][va]+np.array(HOR)[D['hi'][va]]) < cut
    assert max(D['t'][dev]+np.array(HOR)[D['hi'][dev]]) < cut
    assert D['t'][te].min() >= cut
    D['indices'] = {'train':tr, 'validation':va, 'test':te, 'development':dev}
    D['audit'] = {'cut_index':cut, 'validation_start':v0,
                   'last_refit_target':int(max(D['t'][dev]+np.array(HOR)[D['hi'][dev]])),
                   'counts':{name:{str(h):int(np.sum(D['hi'][ind]==hi)) for hi,h in enumerate(HOR)}
                             for name,ind in D['indices'].items()}}
    return D


def prepare(D, refit):
    ids = D['indices']['development' if refit else 'train']
    mean = D['ctx_raw'][ids].mean(0)
    std = D['ctx_raw'][ids].std(0)+1e-5
    ctx = (D['ctx_raw']-mean)/std
    base = []
    for hi in range(len(HOR)):
        q = np.quantile(D['z'][ids[D['hi'][ids]==hi]], TAUS)
        q = np.maximum.accumulate(q-np.arange(len(TAUS))*.001)+np.arange(len(TAUS))*.001
        base.append(q)
    tensors = {k:torch.tensor(D[k], dtype=torch.long if k=='hi' else torch.float32)
               for k in ['seq','hi','z']}
    tensors['ctx'] = torch.tensor(ctx, dtype=torch.float32)
    counts = np.bincount(D['hi'][ids], minlength=len(HOR))
    tensors['weight'] = torch.tensor(len(ids)/(len(HOR)*counts[D['hi']]), dtype=torch.float32)
    return tensors, np.array(base, np.float32)


class ScaleRouter(nn.Module):
    """Each expert addresses the SAME target; location and positive gaps mix separately."""
    def __init__(self, base, variant, width=16):
        super().__init__()
        self.variant = variant
        self.enc = nn.ModuleList([nn.Sequential(nn.Linear(K*3,width), nn.Tanh(),
                                               nn.Linear(width,width), nn.Tanh()) for _ in SCALES])
        self.context = nn.Sequential(nn.Linear(9,width),nn.Tanh())
        self.expert = nn.ModuleList([nn.Sequential(nn.Linear(width*2,24),nn.Tanh(),
                                                   nn.Linear(24,len(TAUS))) for _ in SCALES])
        self.route_location = nn.Sequential(nn.Linear(9,12),nn.Tanh(),nn.Linear(12,len(SCALES)))
        self.route_gaps = nn.Sequential(nn.Linear(9,12),nn.Tanh(),nn.Linear(12,len(SCALES)))
        for expert in self.expert:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)
        self.register_buffer('base',torch.tensor(base))
        self.register_buffer('hlog',torch.tensor(np.log(HOR)/np.log(max(HOR)),dtype=torch.float32))

    def forward(self, seq, ctx, hi):
        if self.variant == 'single_scale':
            seq = seq[:,0:1].expand(-1,len(SCALES),-1,-1)
        full = torch.cat([ctx,self.hlog[hi,None]],1)
        latent_ctx = self.context(full)
        experts = torch.stack([self.expert[e](torch.cat([
            self.enc[e](seq[:,e].flatten(1)), latent_ctx],1)) for e in range(len(SCALES))],1)
        if self.variant == 'uniform':
            wl = wg = torch.full((len(hi),len(SCALES)),1/len(SCALES))
        else:
            route = full.clone()
            if self.variant == 'horizon_only':
                route[:,:-1] = 0.
            if self.variant == 'state_only':
                route[:,-1] = 0.
            wl = self.route_location(route).softmax(1)
            wg = wl if self.variant == 'shared_route' else self.route_gaps(route).softmax(1)
        base = self.base[hi]
        median = base[:,7] + (wl*experts[:,:,0]).sum(1)
        factors = torch.exp(.8*torch.tanh(experts[:,:,1:]))
        gaps = (base[:,1:]-base[:,:-1])*(wg[:,:,None]*factors).sum(1)
        cum = torch.cat([torch.zeros((len(hi),1)),gaps.cumsum(1)],1)
        q = median[:,None]+cum-cum[:,7:8]
        return q,wl,wg


def run_epochs(net, X, ids, epochs, seed, va=None):
    torch.manual_seed(seed+180)
    opt = torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.001)
    ids = torch.tensor(ids)
    def pred(idx):
        return net(X['seq'][idx],X['ctx'][idx],X['hi'][idx])
    def validation():
        with torch.no_grad():
            q,_,_ = pred(va)
            l = losses(X['z'][va],q)
            return float(torch.stack([l[X['hi'][va]==hi].mean() for hi in range(len(HOR))]).mean())
    net.eval()
    best = validation() if va is not None else None
    best_state,ep_best,bad = copy.deepcopy(net.state_dict()),0,0
    for ep in range(epochs):
        net.train()
        order = ids[torch.randperm(len(ids))]
        for idx in order.split(512):
            q,_,_ = pred(idx)
            loss = (losses(X['z'][idx],q)*X['weight'][idx]).mean()
            assert torch.isfinite(loss)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step()
        if va is not None:
            net.eval()
            value = validation()
            if value<best-1e-5:
                best,ep_best,bad = value,ep+1,0
                best_state = copy.deepcopy(net.state_dict())
            else:
                bad+=1
            if ep>=35 and bad>=22:
                break
    if va is not None:
        net.load_state_dict(best_state)
    net.eval()
    return {'best_epoch':ep_best,'best_validation':best,'epochs_run':ep+1 if epochs else 0}


def fit_neural(D, variant, seed, epochs):
    X,base = prepare(D,False)
    torch.manual_seed(seed)
    net = ScaleRouter(base,variant)
    log = run_epochs(net,X,D['indices']['train'],epochs,seed,
                     torch.tensor(D['indices']['validation']))
    # Same selected epoch count, same initial seed, fit all pre-cut labels.
    Xf,basef = prepare(D,True)
    torch.manual_seed(seed)
    final = ScaleRouter(basef,variant)
    if log['best_epoch']:
        run_epochs(final,Xf,D['indices']['development'],log['best_epoch'],seed)
    te = torch.tensor(D['indices']['test'])
    with torch.no_grad():
        q,wl,wg = final(Xf['seq'][te],Xf['ctx'][te],Xf['hi'][te])
    Q = q.numpy()*D['scale'][te.numpy(),None]
    assert np.isfinite(Q).all() and (np.diff(Q,axis=1)>=-1e-6).all()
    log.update(seed=seed,parameter_count=sum(p.numel() for p in net.parameters()),
               refit_epochs=log['best_epoch'],
               location_weights={str(h):wl.numpy()[D['hi'][te.numpy()]==hi].mean(0).tolist()
                                 for hi,h in enumerate(HOR)},
               gap_weights={str(h):wg.numpy()[D['hi'][te.numpy()]==hi].mean(0).tolist()
                            for hi,h in enumerate(HOR)})
    return Q,log


def feature_matrix(D,refit):
    X,_ = prepare(D,refit)
    seq = D['seq'][:,:,:,0]
    stats = np.concatenate([seq[:,:,-1],seq.mean(2),seq.std(2)],1)
    return np.concatenate([X['ctx'].numpy(),stats],1)


def classical(D, hgb=False):
    out, fits = {},{}
    te = D['indices']['test']
    Xi,Xf = feature_matrix(D,False),feature_matrix(D,True)
    tr,va,dev = [D['indices'][n] for n in ['train','validation','development']]
    ridgeQ = np.empty((len(te),len(TAUS)))
    hgbQ = np.empty_like(ridgeQ)
    for hi,h in enumerate(HOR):
        a,b,c = [idx[D['hi'][idx]==hi] for idx in [tr,va,dev]]
        tk = np.flatnonzero(D['hi'][te]==hi)
        best = (np.inf,None)
        for alpha in [.1,1.,10.,100.]:
            model = Ridge(alpha=alpha).fit(Xi[a],D['z'][a])
            rq = np.quantile(D['z'][a]-model.predict(Xi[a]),TAUS)
            pv = model.predict(Xi[b])[:,None]+rq
            score = B.score(D['z'][b],pv).mean()
            if score<best[0]:
                best=(score,alpha)
        final = Ridge(alpha=best[1]).fit(Xf[c],D['z'][c])
        rq = np.quantile(D['z'][c]-final.predict(Xf[c]),TAUS)
        ridgeQ[tk] = (final.predict(Xf[te[tk]])[:,None]+rq)*D['scale'][te[tk],None]
        fits[str(h)] = {'ridge_alpha':best[1]}
        if hgb:
            for j,tau in enumerate(TAUS):
                mod = HistGradientBoostingRegressor(loss='quantile',quantile=float(tau),
                      learning_rate=.05,max_iter=100,max_leaf_nodes=7,min_samples_leaf=25,
                      l2_regularization=1.,early_stopping=False,random_state=11)
                mod.fit(Xf[c],D['z'][c])
                hgbQ[tk,j] = mod.predict(Xf[te[tk]])*D['scale'][te[tk]]
    out['ridge_quantile']=np.sort(ridgeQ,axis=1)
    if hgb:
        out['hgb_quantile']=np.sort(hgbQ,axis=1)
    return out,fits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--markets',nargs='+',default=list(B.CUT))
    parser.add_argument('--seeds',nargs='+',type=int,default=[11,22])
    parser.add_argument('--variants',nargs='+',default=['uniform','horizon_only','state_only',
                                                       'shared_route','dual_route','single_scale'])
    parser.add_argument('--epochs',type=int,default=140)
    parser.add_argument('--hgb',action='store_true')
    parser.add_argument('--tag',default='multiscale_round1')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    panel=pd.read_csv(B.PANEL,parse_dates=['date'])
    report={'status':'exploratory; reused outer test history; no publication or novelty claim',
            'config':vars(args),'horizons':HOR,'scales':SCALES,'sequence_length':K,
            'refit_after_epoch_selection':True,'epoch0_eligible':True,
            'training_loss':'horizon-weighted normalized pinball',
            'selection_loss':'equal-horizon normalized pinball',
            'panel_sha256':hashlib.sha256(B.PANEL.read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'python':platform.python_version(),'torch':torch.__version__,'markets':{}}
    pred_arrays={}
    start=time.perf_counter()
    for market in args.markets:
        D=extract_data(panel[panel.market==market].sort_values('date').reset_index(drop=True))
        _,base=prepare(D,True)
        te=D['indices']['test']
        y,hi=D['y'][te],D['hi'][te]
        preds={'empirical_anchor':base[hi]*D['scale'][te,None]}
        # Classical Gaussian random-walk scale baseline uses the same observations.
        from scipy.stats import norm
        preds['gaussian_rw']=D['scale'][te,None]*norm.ppf(TAUS)[None]
        cr,cf=classical(D,args.hgb)
        preds.update(cr)
        market_result={'audit':D['audit'],'classical_fits':cf,'fits':{},'by_horizon':{}}
        for variant in args.variants:
            qp,fl=[],[]
            for seed in args.seeds:
                q,f=fit_neural(D,variant,seed,args.epochs)
                qp.append(q);fl.append(f)
                print(market,variant,seed,'best',f['best_epoch'],
                      'elapsed_s',round(time.perf_counter()-start,1),flush=True)
            preds[variant]=np.mean(qp,axis=0)
            market_result['fits'][variant]=fl
        for hidx,h in enumerate(HOR):
            keep=hi==hidx
            scores={}
            for variant,q in preds.items():
                yh,qh=y[keep],q[keep]
                scores[variant]={'MQS':float(B.score(yh,qh).mean()),
                                 'WIS':float(B.wis(yh,qh).mean()),
                                 'MAE':float(np.abs(yh-qh[:,7]).mean()),
                                 'coverage80':float(np.mean((yh>=qh[:,2])&(yh<=qh[:,12]))),
                                 'coverage95':float(np.mean((yh>=qh[:,0])&(yh<=qh[:,14])))}
            market_result['by_horizon'][str(h)]={'n':int(keep.sum()),'scores':scores}
        report['markets'][market]=market_result
        pred_arrays[market+'_origin']=D['t'][te]
        pred_arrays[market+'_horizon']=np.array(HOR)[hi]
        pred_arrays[market+'_target']=y
        for name,q in preds.items():
            pred_arrays[market+'_'+name]=q
        (HERE/(args.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        np.savez_compressed(HERE/(args.tag+'_predictions.npz'),**pred_arrays)
    ref='dual_route' if 'dual_route' in args.variants else args.variants[-1]
    report['summary']={}
    for base in [k for k in preds if k!=ref]:
        ratios=[r['scores'][ref]['MQS']/r['scores'][base]['MQS']
                for m in report['markets'].values() for r in m['by_horizon'].values()]
        report['summary'][ref+'_vs_'+base]={'median_MQS_ratio':float(np.median(ratios)),
                                          'mean_log_MQS_ratio':float(np.log(ratios).mean()),
                                          'wins':int(np.sum(np.array(ratios)<1)),
                                          'cells':len(ratios)}
    report['elapsed_seconds']=time.perf_counter()-start
    (HERE/(args.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with threadpool_limits(limits=2):
        main()
