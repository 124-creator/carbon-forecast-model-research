"""KRBN development-only experiment. NO 2026 targets or forecasts are built.

Candidate: positive short/long variance states with a learned relaxation time;
all h-risk forecasts are sums of the same positive daily risk curve. This is a
candidate inductive bias, not an established novelty claim. The validation year
is 2025; target is future mean Parkinson intraday-range variance, not return RV.
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

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'data_acquisition_round13'/'KRBN_ETF_USD_daily_completed_20200731_20261020.csv'
HOR=np.array([1,5,22])
EPS=1e-12
MODES=['direct','fixed_decay','adaptive_decay']


def build():
    raw=pd.read_csv(SOURCE,parse_dates=['date'])
    # Discard the reserved period BEFORE constructing returns, targets, features,
    # normalization or validation scores. No 2026 model outcome is computed.
    d=raw[raw.date<'2026-01-01'].copy().reset_index(drop=True)
    r=np.log(d.adj_close).diff()
    rng=np.log(d.high/d.low).pow(2)/(4*np.log(2))
    intra=np.log(d.close/d.open)
    # Adj factors are used only as ratios; corporate-action-adjusted open-to-close
    # decomposition. No adjusted close is combined directly with raw high/low.
    factor=d.adj_close/d.close
    overnight=np.log(d.open/d.close.shift())+np.log(factor/factor.shift())
    e5=rng.ewm(span=5,adjust=False).mean().clip(lower=EPS)
    e22=rng.ewm(span=22,adjust=False).mean().clip(lower=EPS)
    e63=rng.ewm(span=63,adjust=False).mean().clip(lower=EPS)
    f=pd.DataFrame({'logrange1':np.log(rng.clip(lower=EPS)),
        'logrange5':np.log(rng.rolling(5).mean().clip(lower=EPS)),
        'logrange22':np.log(rng.rolling(22).mean().clip(lower=EPS)),
        'r1':r,'r5':r.rolling(5).sum(),'r22':r.rolling(22).sum(),
        'intraday':intra,'logovernightvar':np.log(overnight.pow(2).rolling(5).mean().clip(lower=EPS)),
        'logreturnvar':np.log(r.pow(2).rolling(22).mean().clip(lower=EPS)),
        'logvolume':np.log(d.volume),
        'volume_ratio':np.log(d.volume/d.volume.rolling(22).mean()),
        'range_ratio':np.log(e5/e63)})
    cut=int(np.searchsorted(d.date.to_numpy(),np.datetime64('2025-01-01')))
    rows=[]
    for hi,h in enumerate(HOR):
        target=rng.rolling(int(h)).mean().shift(-int(h)).to_numpy()
        for t in range(100,len(d)-h):
            if not np.isfinite(f.iloc[t]).all():continue
            split=0 if t+h<cut else (1 if t>=cut else -1)
            if split>=0:rows.append((t,hi,split,target[t]))
    t,hi,split,y=np.array(rows).T;t=t.astype(int);hi=hi.astype(int);split=split.astype(int)
    assert (y>=0).all()
    D=dict(t=t,hi=hi,h=HOR[hi],split=split,y=y.astype(np.float32),
           x=f.to_numpy()[t].astype(np.float32),short=e5.to_numpy()[t].astype(np.float32),
           long=e63.to_numpy()[t].astype(np.float32),ewma=e22.to_numpy()[t].astype(np.float32),
           date=d.date.to_numpy()[t],cut=cut,
           tr=np.flatnonzero(split==0),va=np.flatnonzero(split==1))
    assert (t[D['tr']]+D['h'][D['tr']]<cut).all()
    assert pd.Timestamp(D['date'].max())<pd.Timestamp('2026-01-01')
    return D,dict(development_rows=len(d),first=str(d.date.min().date()),last=str(d.date.max().date()),
                  validation_start='2025-01-01',training_rows=len(D['tr']),validation_rows=len(D['va']),
                  withheld_period_used=False,target='future mean Parkinson high-low intraday variance',
                  feature_names=list(f.columns))


def loss_np(v,y):return np.log(np.maximum(v,EPS))+y/np.maximum(v,EPS)


class CurveNet(nn.Module):
    def __init__(self,nf,mode):
        super().__init__();self.mode=mode
        self.body=nn.Sequential(nn.Linear(nf,12),nn.Tanh(),nn.Linear(12,8),nn.Tanh())
        self.head=nn.Linear(8,3)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x,short,long,ewma,h,hi):
        a=self.head(self.body(x))
        if self.mode=='direct':
            correction=a.gather(1,hi[:,None]).squeeze(1)
            return ewma*torch.exp(1.5*torch.tanh(correction))
        ss=short*torch.exp(1.5*torch.tanh(a[:,0]))
        ll=long*torch.exp(1.5*torch.tanh(a[:,1]))
        tau=torch.exp(np.log(22.)+1.5*torch.tanh(a[:,2]))
        if self.mode=='fixed_decay':tau=tau*0+22.
        k=torch.arange(1,23,dtype=x.dtype)[None,:]
        weights=torch.exp(-k/tau[:,None])*(k<=h[:,None])
        factor=weights.sum(1)/h
        return ll+(ss-ll)*factor


def train(D,mode,seed,epochs):
    torch.manual_seed(seed)
    tr,va=D['tr'],D['va']
    mean=D['x'][tr].mean(0);sd=D['x'][tr].std(0)+1e-5
    X={k:torch.tensor(D[k],dtype=torch.long if k=='hi' else torch.float32)
       for k in ['short','long','ewma','h','hi','y']}
    X['x']=torch.tensor((D['x']-mean)/sd)
    net=CurveNet(X['x'].shape[1],mode)
    opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.002)
    best=np.inf;state=None;bestep=0;bad=0;history=[]
    for ep in range(1,epochs+1):
        net.train();order=np.random.default_rng(seed+ep).permutation(tr)
        for b in np.array_split(order,max(1,int(np.ceil(len(order)/256)))):
            opt.zero_grad();v=net(*(X[k][b] for k in ['x','short','long','ewma','h','hi'])).clamp_min(EPS)
            # log(v/ewma)+y/v differs from QLIKE only by row constants.
            loss=(torch.log(v/X['ewma'][b])+X['y'][b]/v).mean()
            assert torch.isfinite(loss)
            loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
        net.eval()
        with torch.no_grad():
            pred=net(*(X[k][va] for k in ['x','short','long','ewma','h','hi'])).numpy()
        val=float(loss_np(pred,D['y'][va]).mean());history.append([ep,val])
        if val<best-1e-5:best=val;bestep=ep;state=copy.deepcopy(net.state_dict());bad=0
        else:bad+=1
        if bad>=15 and ep>=20:break
    net.load_state_dict(state);net.eval()
    with torch.no_grad():
        v=net(*(X[k][va] for k in ['x','short','long','ewma','h','hi'])).numpy()
    assert np.isfinite(v).all() and (v>0).all()
    # Positive curve implies monotone cumulative risk; verify each validation x.
    if mode!='direct':
        idx=va[:50]
        with torch.no_grad():
            cumulative=[]
            for h in range(1,23):
                vv=net(X['x'][idx],X['short'][idx],X['long'][idx],X['ewma'][idx],
                       torch.full((len(idx),),float(h)),X['hi'][idx])
                cumulative.append((h*vv).numpy())
        assert np.all(np.diff(np.stack(cumulative),axis=0)>=-1e-10)
    return v,dict(seed=seed,best_epoch=bestep,epochs_run=len(history),
                  validation_loss=best,parameters=sum(p.numel() for p in net.parameters()),history=history)


def baselines(D):
    tr,va=D['tr'],D['va'];out={'EWMA22':D['ewma'][va]};har=np.empty(len(va));hgb=np.empty(len(va))
    for hi,h in enumerate(HOR):
        ids=tr[D['hi'][tr]==hi];sel=np.flatnonzero(D['hi'][va]==hi);te=va[sel]
        X=np.column_stack([np.ones(len(ids)),D['x'][ids,:3]])
        target=np.log(np.maximum(D['y'][ids],EPS));b=np.linalg.lstsq(X,target,rcond=None)[0]
        f=float(np.exp(target-X@b).mean())
        har[sel]=np.exp(np.column_stack([np.ones(len(te)),D['x'][te,:3]])@b)*f
        scale=D['y'][ids].mean()
        net=HistGradientBoostingRegressor(loss='poisson',max_iter=150,max_leaf_nodes=7,
              learning_rate=.04,l2_regularization=1,min_samples_leaf=20,early_stopping=False,random_state=11)
        net.fit(D['x'][ids],D['y'][ids]/scale);hgb[sel]=net.predict(D['x'][te])*scale
    out['HAR_range']=har;out['HGB_poisson']=hgb
    factor=np.mean(np.exp(-np.arange(1,23)[None,:]/22)*(np.arange(1,23)[None,:]<=D['h'][va,None]),axis=1)*22/D['h'][va]
    out['fixed_curve_untrained']=D['long'][va]+(D['short'][va]-D['long'][va])*factor
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--seeds',nargs='+',type=int,default=[11,22,33]);ap.add_argument('--tag',default='range_clock_round14_development')
    a=ap.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    D,audit=build();va=D['va'];start=time.time()
    meta=dict(config=vars(a),status='DEVELOPMENT ONLY; no 2026 test access',audit=audit,
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        hypotheses='state-dependent risk relaxation versus direct heads and fixed relaxation',
        horizons=HOR.tolist(),fits={},scores={})
    (HERE/(a.tag+'_manifest.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    pred=baselines(D)
    for mode in MODES:
        vv=[];fits=[]
        for seed in a.seeds:
            v,fit=train(D,mode,seed,a.epochs);vv.append(v);fits.append(fit)
            print(mode,seed,'best',fit['best_epoch'],'loss',round(fit['validation_loss'],6),flush=True)
        pred[mode]=np.mean(vv,axis=0);meta['fits'][mode]=fits
    for hi,h in enumerate(HOR):
        mask=D['hi'][va]==hi;y=D['y'][va][mask];base=loss_np(pred['EWMA22'][mask],y)
        meta['scores'][str(h)]={}
        for name,v in pred.items():
            p=v[mask];loss=loss_np(p,y)
            meta['scores'][str(h)][name]=dict(n=int(mask.sum()),
                mean_logscore=float(loss.mean()),loss_difference_vs_EWMA22=float((loss-base).mean()),
                MSE=float(np.mean((p-y)**2)))
        print('h',h,{name:round(s['loss_difference_vs_EWMA22'],6) for name,s in meta['scores'][str(h)].items()},flush=True)
    meta['elapsed_seconds']=time.time()-start
    (HERE/(a.tag+'.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(HERE/(a.tag+'_validation_predictions.npz'),date=D['date'][va],horizon=D['h'][va],target=D['y'][va],**pred)
    print('DEVELOPMENT FINISHED; 2026 REMAINS UNUSED',round(time.time()-start,1),flush=True)


if __name__=='__main__':main()
