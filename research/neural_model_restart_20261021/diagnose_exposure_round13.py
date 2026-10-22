"""Fixed follow-up diagnostics for the predeclared round-13 candidate.
All gains are exploratory. No configuration is chosen using these test scores.
"""
import json
import hashlib
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import exposure_round13 as E

HERE=Path(__file__).resolve().parent
TAG='exposure_round13_main'
BASES=['QRF','QRF_reference','empirical','event_count','no_carry','hurdle','fixed_untrained']


def ci(a,b,group=None,block=32,reps=1999):
    rng=np.random.default_rng(20261021)
    n=len(a);grp=np.ones(n,dtype=bool) if group is None else group
    if grp.sum()<10:return None
    starts=rng.integers(n,size=(reps,int(np.ceil(n/block))))
    ix=((starts[:,:,None]+np.arange(block))%n).reshape(reps,-1)[:,:n]
    g=grp[ix];counts=g.sum(1)
    am=np.sum(a[ix]*g,axis=1)/np.maximum(counts,1)
    bm=np.sum(b[ix]*g,axis=1)/np.maximum(counts,1)
    valid=(counts>0)&(bm>0)
    return np.quantile(100*(1-am[valid]/bm[valid]),[.025,.975]).tolist()


def dm(diff,lag=22):
    n=len(diff);mu=diff.mean();e=diff-mu
    s=float(e@e)/n
    for k in range(1,min(lag+1,n)):
        s+=2*(1-k/(lag+1))*float(e[k:]@e[:-k])/n
    z=float(mu/np.sqrt(s/n)) if s>0 else 0
    return math.erfc(abs(z)/np.sqrt(2))


def holm(p):
    order=np.argsort(p);out=np.empty(len(p));running=0.
    for rank,j in enumerate(order):
        running=max(running,(len(p)-rank)*p[j]);out[j]=min(running,1)
    return out


def main():
    torch.set_num_threads(1)
    meta=json.loads((HERE/(TAG+'.json')).read_text(encoding='utf-8'))
    assert set(meta['markets'])==set(E.CUT)
    assert meta['source_sha256']==hashlib.sha256(Path(E.__file__).read_bytes()).hexdigest()
    assert meta['panel_sha256']==hashlib.sha256(E.PANEL.read_bytes()).hexdigest()
    a=np.load(HERE/(TAG+'_predictions.npz'))
    panel=pd.read_csv(E.PANEL,parse_dates=['date'])
    rows=[];grouped=[];fixed_arrays={}
    for market,res in meta['markets'].items():
        d=panel[panel.market==market].sort_values('date').reset_index(drop=True)
        D=E.build(d);te=D['te'];F=E.daily_features(d)
        assert np.array_equal(D['t'][te],a[market+'_origin'])
        assert np.array_equal(D['h'][te],a[market+'_horizon'])
        np.testing.assert_allclose(D['y'][te],a[market+'_target'],rtol=1e-12,atol=1e-12)
        X=E.tensors(D,D['rf']);net=E.ExposureNet(X['x'].shape[1],'observation_gate')
        cp,_=E.predict(net,X,te);fixed=[]
        for ii in np.array_split(np.arange(len(te)),max(1,int(np.ceil(len(te)/128)))):
            fixed.append(E.crps(torch.tensor(D['z'][te[ii]]),tuple(torch.tensor(v[ii]) for v in cp)).numpy())
        fs=np.concatenate(fixed)*D['scale'][te];fixed_arrays[market+'_fixed_CRPS']=fs
        for h in E.HORIZONS:
            mask=a[market+'_horizon']==h
            loss=a[market+'_observation_gate_CRPS'][mask]
            age=F['age'][a[market+'_origin'][mask]]
            for base in BASES:
                comp=fs[mask] if base=='fixed_untrained' else a[market+'_'+base+'_CRPS'][mask]
                gain=float(100*(1-loss.mean()/comp.mean()))
                rows.append(dict(market=market,h=int(h),baseline=base,n=len(loss),
                    CRPS_gate=float(loss.mean()),CRPS_base=float(comp.mean()),gain_percent=gain,
                    ci64steps=ci(loss,comp),ci128steps=ci(loss,comp,block=64),p=dm(loss-comp)))
                if base in ['no_carry','event_count']:
                    for name,grp in [('age0',age==0),('age1plus',age>=1),('age2plus',age>=2)]:
                        if not grp.any():continue
                        spell_ids=a[market+'_origin'][mask][grp]-age[grp]
                        grouped.append(dict(market=market,h=int(h),baseline=base,group=name,
                            n=int(grp.sum()),distinct_spell_anchors=int(len(np.unique(spell_ids))),
                            note='descriptive conditional comparison, not an age-group interaction test',
                            gain_percent=float(100*(1-loss[grp].mean()/comp[grp].mean())),
                            ci64steps=ci(loss,comp,grp)))
    for base in BASES:
        sub=[r for r in rows if r['baseline']==base]
        for r,padj in zip(sub,holm([r['p'] for r in sub])):r['holm_within_baseline']=float(padj)
    aggregate={}
    for base in BASES:
        rr=[r for r in rows if r['baseline']==base]
        ratios=np.array([r['CRPS_gate']/r['CRPS_base'] for r in rr])
        aggregate[base]=dict(geomean_CRPS_gain_percent=float(100*(1-np.exp(np.log(ratios).mean()))),
                            wins=int((ratios<1).sum()),cells=len(rr),
                            note='descriptive equal market/horizon aggregate; not independent confirmatory evidence')
    out=dict(status='EXPLORATORY paired diagnostics; all historical periods reused',
             score='exact CRPS of the same seed-mixture distribution used for quantiles',
             bootstrap='circular blocks of 32/64 origins, step=2 observed records',
             holm_scope='15 market-horizon cells within each listed baseline only, not whole historical search',
             aggregate=aggregate,comparisons=rows,carry_groups=grouped)
    (HERE/(TAG+'_diagnostics.json')).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame(rows).to_csv(HERE/(TAG+'_comparisons.csv'),index=False)
    pd.DataFrame(grouped).to_csv(HERE/(TAG+'_carry_groups.csv'),index=False)
    np.savez_compressed(HERE/(TAG+'_fixed_predictions.npz'),**fixed_arrays)
    print(json.dumps(aggregate,indent=2),flush=True)
    print(pd.DataFrame(grouped).to_string(index=False),flush=True)


if __name__=='__main__':main()
