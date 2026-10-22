# -*- coding: utf-8 -*-
"""Small exploratory marked-return experiment. Only this prefix is written.

The horizon is h observed daily-panel rows, not h price-change events or elapsed
civil days. The hurdle models P(r_h != 0), not P(any within-horizon price update).
Its mark density is a two-normal mixture with positive/negative component means;
normal components overlap: their mixture weight is not the sign probability.
Mixed NLL trains the hurdle; continuous NLL trains the no-atom ablation. Epochs
are selected using the SAME proper quantile score on inner validation, then a
new model is fitted on every available pre-cut target for exactly those epochs.
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
from scipy.special import ndtr, ndtri
from neural_restart import PANEL, CUT, TAUS, score, wis

HERE = Path(__file__).resolve().parent
HOR = np.array([1, 5, 10, 20])
W = 128


def build(d):
    lp = np.log(d.price.to_numpy(float))
    dates = d.date.to_numpy()
    market = str(d.market.iloc[0])
    cut = int(np.searchsorted(dates, np.datetime64(CUT[market])))
    vs = int(cut * .75)
    rows = []
    for t in range(W, len(d)-1, 2):
        r = np.diff(lp[t-W:t+1])
        sd = float(r.std(ddof=1))
        if not np.isfinite(sd) or sd <= 1e-8:
            continue
        z = r / sd
        nz = np.flatnonzero(np.abs(r) > 1e-12)
        age = W-1-nz[-1] if len(nz) else W
        f = list(np.clip(z[-8:], -20, 20))
        for k in [5, 20, 64, 128]:
            a = z[-k:]
            f += [float(a.mean()), float(a.std()), float(np.mean(a == 0)),
                  float(np.mean(a > 0))]
        f += [np.log(sd), np.log1p(age)]
        for hi, h in enumerate(HOR):
            if t+h >= len(d):
                continue
            sp = 0 if t+h < vs else (1 if t >= vs and t+h < cut
                                    else (2 if t >= cut else -1))
            refit = t+h < cut
            if sp < 0 and not refit:
                continue
            y = float(lp[t+h]-lp[t])
            rows.append((t, hi, sp, refit, f+[np.log(h)/np.log(20)], sd*np.sqrt(h), y))
    c = list(zip(*rows))
    D = dict(t=np.array(c[0]), hi=np.array(c[1]), split=np.array(c[2]),
             refit=np.array(c[3]), x=np.array(c[4], np.float32),
             scale=np.array(c[5], float), y=np.array(c[6], float))
    D['z'] = D['y']/D['scale']
    tr, va, te = [np.flatnonzero(D['split'] == i) for i in [0, 1, 2]]
    rf = np.flatnonzero(D['refit'])
    assert np.all(D['t'][tr]+HOR[D['hi'][tr]] < vs)
    assert np.all(D['t'][va] >= vs) and np.all(D['t'][va]+HOR[D['hi'][va]] < cut)
    assert np.all(D['t'][rf]+HOR[D['hi'][rf]] < cut)
    assert np.all(D['t'][te] >= cut)
    D['tr'], D['va'], D['te'], D['rf'] = tr, va, te, rf
    D['audit'] = dict(cut_index=cut, validation_start_index=vs, lookback=W,
                     step=2, train=len(tr), validation=len(va), refit=len(rf),
                     test=len(te), boundary_rows_recovered_for_refit=int(np.sum(D['split'][rf]<0)),
                     last_train_target=int(np.max(D['t'][tr]+HOR[D['hi'][tr]])),
                     last_refit_target=int(np.max(D['t'][rf]+HOR[D['hi'][rf]])),
                     test_zero_share={str(h): float(np.mean(D['y'][te[D['hi'][te]==i]] == 0))
                                      for i,h in enumerate(HOR)},
                     observed_daily_panel_horizon=True, target_clipping=False)
    return D


def initial(D, idx):
    # Every constant is estimated on the fitting partition only.
    base = []
    for hi in range(4):
        z = D['z'][idx[D['hi'][idx] == hi]]
        nz = z[np.abs(z)>1e-10]
        p = (len(nz)+1)/(len(z)+2)
        w = (np.sum(nz > 0)+1)/(len(nz)+2)
        pos, neg = nz[nz>0], -nz[nz<0]
        pm = max(float(pos.mean()) if len(pos) else .5, .02)
        nm = max(float(neg.mean()) if len(neg) else .5, .02)
        ps = max(float(pos.std()) if len(pos)>1 else .8, .15)
        ns = max(float(neg.std()) if len(neg)>1 else .8, .15)
        base.append([math.log(p/(1-p)), math.log(w/(1-w)),
                     math.log(pm), math.log(nm), math.log(ps), math.log(ns)])
    return np.array(base, np.float32)


class MarkedNet(nn.Module):
    def __init__(self, nf, base, model):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(nf, 24), nn.Tanh(), nn.Linear(24, 16), nn.Tanh())
        self.update_head = nn.Linear(16, 1)
        self.mark_head = nn.Linear(16, 5)
        for layer in [self.update_head, self.mark_head]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.register_buffer('base', torch.tensor(base))
        self.model = model

    def forward(self, x, hi):
        a = self.trunk(x)
        b = self.base[hi]
        upd = self.update_head(a).squeeze(1)
        p = torch.sigmoid(b[:,0]+upd) if self.model == 'dual_head' else torch.sigmoid(b[:,0])
        if self.model == 'no_zero_mass':
            p = torch.ones_like(p)
        m = self.mark_head(a)
        w = torch.sigmoid(b[:,1]+m[:,0])
        # Signed component centers; bounded residuals, strictly positive scales.
        mu_p = torch.exp(b[:,2]+1.5*torch.tanh(m[:,1]))
        mu_n = -torch.exp(b[:,3]+1.5*torch.tanh(m[:,2]))
        s_p = torch.exp(b[:,4]+2*torch.tanh(m[:,3])) + .025
        s_n = torch.exp(b[:,5]+2*torch.tanh(m[:,4])) + .025
        return torch.stack([p, w, mu_p, mu_n, s_p, s_n], 1)


def nll(z, par, model):
    p,w,mp,mn,sp,sn = par.unbind(1)
    logp = torch.log(p.clamp_min(1e-7))
    lw = torch.log(w.clamp_min(1e-7))
    lnw = torch.log((1-w).clamp_min(1e-7))
    a = lw - torch.log(sp) - .5*((z-mp)/sp)**2 - .5*np.log(2*np.pi)
    b = lnw - torch.log(sn) - .5*((z-mn)/sn)**2 - .5*np.log(2*np.pi)
    continuous = torch.logsumexp(torch.stack([a,b],1), 1)
    if model == 'no_zero_mass':
        return -continuous.mean()
    return -torch.where(z.abs() <= 1e-10, torch.log((1-p).clamp_min(1e-7)),
                        logp+continuous).mean()


def quantiles(par):
    p,w,mp,mn,sp,sn = [par[:,i,None].astype(float) for i in range(6)]
    lo = np.minimum(mp-10*sp, mn-10*sn) + np.zeros((len(par),len(TAUS)))
    hi = np.maximum(mp+10*sp, mn+10*sn) + np.zeros_like(lo)
    for _ in range(32):
        mid = (lo+hi)/2
        cdf = p*(w*ndtr((mid-mp)/sp)+(1-w)*ndtr((mid-mn)/sn))+(1-p)*(mid>=0)
        left = cdf < TAUS[None,:]
        lo = np.where(left,mid,lo)
        hi = np.where(left,hi,mid)
    q = (lo+hi)/2
    left0 = p*(w*ndtr(-mp/sp)+(1-w)*ndtr(-mn/sn))
    atom = (TAUS[None,:] >= left0) & (TAUS[None,:] <= left0+1-p)
    q[atom] = 0.
    assert np.isfinite(q).all() and np.all(np.diff(q,axis=1)>=-1e-8)
    return q


def train_predict(D, model, seed, epochs):
    z = torch.tensor(D['z'], dtype=torch.float32)
    hid = torch.tensor(D['hi'], dtype=torch.long)
    def setup(idx):
        torch.manual_seed(seed)
        mean = D['x'][idx].mean(0)
        sd = D['x'][idx].std(0)+1e-5
        x = torch.tensor((D['x']-mean)/sd)
        net = MarkedNet(x.shape[1], initial(D,idx), model)
        opt = torch.optim.AdamW(net.parameters(), lr=.003, weight_decay=.002)
        return x,net,opt
    def fit_epoch(x,net,opt,idx):
        net.train()
        perm = torch.tensor(idx)[torch.randperm(len(idx))]
        for batch in perm.split(512):
            opt.zero_grad()
            loss = nll(z[batch],net(x[batch],hid[batch]),model)
            assert torch.isfinite(loss), (model,seed)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),5.)
            opt.step()
    x,net,opt = setup(D['tr'])
    best_ep,best,bad = 0,np.inf,0
    history=[]
    for ep in range(1,epochs+1):
        fit_epoch(x,net,opt,D['tr'])
        if ep%2 and ep != 1 and ep != epochs:
            continue
        net.eval()
        with torch.no_grad():
            par = net(x[D['va']],hid[D['va']]).numpy()
        val = float(score(D['z'][D['va']],quantiles(par)).mean())
        history.append([ep,val])
        if val < best-1e-6:
            best_ep,best,bad=ep,val,0
        else:
            bad+=1
        if bad>=6 and ep>=16:
            break
    # Restart seed, recompute feature normalization and all initial statistics on
    # ALL available pre-cut labels, including rows excluded by inner boundaries.
    x,net,opt = setup(D['rf'])
    for _ in range(best_ep):
        fit_epoch(x,net,opt,D['rf'])
    net.eval()
    with torch.no_grad():
        par = net(x[D['te']],hid[D['te']]).numpy()
    q = quantiles(par)*D['scale'][D['te'],None]  # Only inverse transformation.
    return q,par,dict(seed=seed,best_epoch=best_ep,inner_epochs=ep,
                      validation_MQS_standardized=best,refit_epochs=best_ep,
                      allocated_parameters=sum(p.numel() for p in net.parameters()),
                      active_parameters=sum(p.numel() for n,p in net.named_parameters()
                                            if model=='dual_head' or 'update_head' not in n),
                      mean_predicted_nonzero_probability=float(par[:,0].mean()),
                      validation_history=history)


def constant_predictions(D):
    te,rf = D['te'],D['rf']
    qemp=np.empty((len(te),len(TAUS)))
    qgauss=np.empty_like(qemp)
    for hi in range(4):
        z=D['z'][rf[D['hi'][rf]==hi]]
        mask=D['hi'][te]==hi
        qemp[mask]=np.quantile(z,TAUS)
        qgauss[mask]=z.mean()+max(z.std(ddof=1),1e-5)*ndtri(TAUS)
    return dict(empirical_constant=qemp*D['scale'][te,None],
                gaussian_constant=qgauss*D['scale'][te,None])


def summarize(report):
    result={}
    for model,base in [('dual_head','fixed_probability'),('dual_head','no_zero_mass'),
                       ('dual_head','empirical_constant'),('dual_head','gaussian_constant'),
                       ('fixed_probability','empirical_constant')]:
        r=[cell['scores'][model]['MQS']/cell['scores'][base]['MQS']
           for m in report['markets'].values() for cell in m['by_horizon'].values()]
        if r:
            result[model+'_vs_'+base]=dict(cells=len(r),wins=int(np.sum(np.array(r)<1)),
                    median_MQS_ratio=float(np.median(r)),mean_log_MQS_ratio=float(np.log(r).mean()),
                    geometric_MQS_gain_percent=float(100*(1-np.exp(np.log(r).mean()))))
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--markets',nargs='+',default=list(CUT))
    ap.add_argument('--seeds',type=int,nargs='+',default=[11,22])
    ap.add_argument('--epochs',type=int,default=32)
    ap.add_argument('--tag',default='marked_event_round1')
    args=ap.parse_args()
    if not args.tag.startswith('marked_event_round1'):
        raise ValueError('All outputs must start with marked_event_round1')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    panel=pd.read_csv(PANEL,parse_dates=['date'])
    start=time.perf_counter()
    report=dict(status='exploratory candidate; no novelty or publication success established',
                config=vars(args),horizons=HOR.tolist(),quantiles=TAUS.tolist(),
                horizon_unit='observed daily-panel steps, not event steps or elapsed civil days',
                target='direct marginal log return; endpoint zero does not imply no intervening updates',
                loss='mixed-measure NLL for atom models, continuous NLL for no-zero; same validation MQS',
                architecture='shared 24/16 MLP, update head + 5-parameter signed-center two-normal mark head',
                seeds_combination='arithmetic quantile ensemble, not distribution mixture',
                panel_sha256=hashlib.sha256(PANEL.read_bytes()).hexdigest(),
                torch=torch.__version__,python=platform.python_version(),markets={})
    arrays={}
    csvrows=[]
    for market in args.markets:
        d=panel[panel.market==market].sort_values('date').reset_index(drop=True)
        D=build(d)
        te=D['te']; y=D['y'][te]; hid=D['hi'][te]
        preds=constant_predictions(D)
        mres=dict(audit=D['audit'],fits={},by_horizon={})
        arrays[market+'_origin']=D['t'][te]
        arrays[market+'_horizon']=HOR[hid]
        arrays[market+'_target']=y
        arrays[market+'_origin_date']=d.date.to_numpy()[D['t'][te]].astype('datetime64[D]')
        for model in ['dual_head','fixed_probability','no_zero_mass']:
            qs=[];fits=[]
            for seed in args.seeds:
                q,par,fit=train_predict(D,model,seed,args.epochs)
                qs.append(q);fits.append(fit)
                arrays[f'{market}_{model}_seed{seed}']=q
                arrays[f'{market}_{model}_seed{seed}_parameters']=par
                print(market,model,seed,'epoch',fit['best_epoch'],
                      'elapsed',round(time.perf_counter()-start,1),flush=True)
            preds[model]=np.mean(qs,axis=0)
            mres['fits'][model]=fits
        for hi,h in enumerate(HOR):
            mask=hid==hi
            scores={}
            for model,q in preds.items():
                yh,qh=y[mask],q[mask]
                scores[model]=dict(MQS=float(score(yh,qh).mean()),WIS=float(wis(yh,qh).mean()),
                        coverage80=float(np.mean((yh>=qh[:,2])&(yh<=qh[:,12]))),
                        coverage95=float(np.mean((yh>=qh[:,0])&(yh<=qh[:,14]))),
                        MAE_median=float(np.abs(yh-qh[:,7]).mean()))
            mres['by_horizon'][str(h)]=dict(n=int(mask.sum()),scores=scores)
        for model,q in preds.items():
            arrays[market+'_'+model]=q
            for i,t in enumerate(D['t'][te]):
                row=dict(market=market,model=model,origin=int(t),
                         origin_date=str(d.date.iloc[t].date()),
                         target_date=str(d.date.iloc[t+HOR[hid[i]]].date()),
                         horizon=int(HOR[hid[i]]),target=float(y[i]),scale=float(D['scale'][te[i]]))
                row.update({f'q{float(tau):.3f}':float(q[i,j]) for j,tau in enumerate(TAUS)})
                csvrows.append(row)
        report['markets'][market]=mres
        report['summary']=summarize(report)
        report['elapsed_seconds']=time.perf_counter()-start
        (HERE/(args.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        np.savez_compressed(HERE/(args.tag+'_predictions.npz'),**arrays)
        pd.DataFrame(csvrows).to_csv(HERE/(args.tag+'_predictions.csv'),index=False)
        print(market,json.dumps(report['summary'],ensure_ascii=False),flush=True)
    report['elapsed_seconds']=time.perf_counter()-start
    (HERE/(args.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['summary'],indent=2),flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
