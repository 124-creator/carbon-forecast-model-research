# -*- coding: utf-8 -*-
"""Moment-budget/frequency coupled hurdle decoder; exploratory candidate only.
Read-only reuse of round2 data, trainer and analytic CRPS/CDF. Writes only this
prefix. Nonzero marks are two unrestricted-center normals, not sign classes.
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
import marked_event_round2 as direct
from marked_event_round1 import initial,quantiles
from neural_restart import PANEL,CUT,TAUS,score,wis

HERE=Path(__file__).resolve().parent
MODELS=['moment_static_prior','moment_rule_prior','moment_neural_prior','moment_neural_fixed_budget']


def decode(p,mu,s,w,d,a):
    """U has zero mean/unit variance; M=mu/p+s/sqrt(p)*U."""
    rp,rn=torch.exp(a),torch.exp(-a)
    m0=(2*w-1)*d
    v0=w*rp*rp+(1-w)*rn*rn+4*w*(1-w)*d*d
    root=torch.sqrt(v0)
    center=mu/p
    scale=s/torch.sqrt(p)
    m1=center+scale*(d-m0)/root
    m2=center+scale*(-d-m0)/root
    s1=scale*rp/root;s2=scale*rn/root
    return torch.stack([p,w,m1,m2,s1,s2],1)


def moments(par):
    p,w,m1,m2,s1,s2=par.unbind(1)
    cm=w*m1+(1-w)*m2
    cv=w*s1*s1+(1-w)*s2*s2+w*(1-w)*(m1-m2)**2
    mu=p*cm
    risk=p*cv
    var=risk+p*(1-p)*cm*cm
    return mu,torch.sqrt(risk),var


class MomentNet(nn.Module):
    def __init__(self,nf,base,prior):
        super().__init__()
        # Same allocation order/width as the old direct decoder.
        self.trunk=nn.Sequential(nn.Linear(nf,24),nn.Tanh(),nn.Linear(24,16),nn.Tanh())
        self.prior_head=nn.Linear(16,2)
        self.mark_head=nn.Linear(16,5)
        for layer in [self.prior_head,self.mark_head]:
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        b=torch.tensor(base,dtype=torch.float32)
        self.register_buffer('base',b)
        # Use the ACTUAL old float32 initialized parameters, including +.025,
        # then derive moment buffers in float64 to avoid cancellation error.
        p0=torch.sigmoid(b[:,0]).double();w0=torch.sigmoid(b[:,1]).double()
        mp=torch.exp(b[:,2]).double();mn=-torch.exp(b[:,3]).double()
        sp=(torch.exp(b[:,4])+.025).double();sn=(torch.exp(b[:,5])+.025).double()
        vm=w0*sp*sp+(1-w0)*sn*sn+w0*(1-w0)*(mp-mn)**2
        mu0=p0*(w0*mp+(1-w0)*mn)
        s0=torch.sqrt(p0*vm)
        sref=torch.sqrt(sp*sn)
        d0=(mp-mn)/(2*sref);a0=.5*torch.log(sp/sn)
        self.register_buffer('moment_base',torch.stack([mu0,s0,w0,d0,a0],1))
        self.prior=prior

    def forward(self,x,hi,local):
        hidden=self.trunk(x);b=self.base[hi];g=self.moment_base[hi]
        static=torch.sigmoid(b[:,0])
        gate=torch.zeros_like(static);residual=torch.zeros_like(static)
        if self.prior=='moment_static_prior':
            p=static
        elif self.prior=='moment_rule_prior':
            gate=torch.full_like(static,direct.RULE_LOCAL_WEIGHT)
            p=(1-gate)*static+gate*local
        else:
            pr=self.prior_head(hidden)
            gate=torch.sigmoid(math.log(4)+pr[:,0])
            residual=.75*torch.tanh(pr[:,1])
            prior=(1-gate)*static+gate*local
            p=torch.sigmoid(torch.logit(prior.clamp(1e-6,1-1e-6))+residual)
        out=self.mark_head(hidden)
        if self.prior=='moment_neural_fixed_budget':
            mu,s=g[:,0],g[:,1]
        else:
            mu=g[:,0]+1.5*g[:,1]*torch.tanh(out[:,0])
            s=g[:,1]*torch.exp(1.5*torch.tanh(out[:,1]))
        w=torch.sigmoid(torch.logit(g[:,2])+out[:,2])
        d=g[:,3]*torch.exp(1.5*torch.tanh(out[:,3]))
        a=g[:,4]+.75*torch.tanh(out[:,4])
        return decode(p.double(),mu,s,w,d,a),torch.stack([gate,residual],1)


def initialization_check(base,nf):
    old=direct.PriorNet(nf,base,'static_prior')
    new=MomentNet(nf,base,'moment_static_prior')
    x=torch.zeros((len(base),nf));hi=torch.arange(len(base));local=torch.full((len(base),),.7)
    with torch.no_grad():
        po,_=old(x,hi,local);pn,_=new(x,hi,local)
    err=float(torch.max(torch.abs(po.double()-pn)))
    assert err<1e-5,('initial distribution mismatch',err)
    return err


def sanity():
    p=torch.tensor([.03,.2,.7,.97],dtype=torch.float64)
    mu=torch.tensor([.2,-.1,.3,.05],dtype=torch.float64)
    s=torch.tensor([.8,1.1,.5,.7],dtype=torch.float64)
    w=torch.full_like(p,.37);d=torch.full_like(p,1.2);a=torch.full_like(p,.4)
    par=decode(p,mu,s,w,d,a)
    ey,budget,var=moments(par)
    target=s*s+(1-p)*mu*mu/p
    e1=float((ey-mu).abs().max());e2=float((var-target).abs().max());e3=float((budget-s).abs().max())
    assert e1<1e-10 and e2<1e-10 and e3<1e-10
    base=np.array([[.3,-.4,.2,-.1,-.2,.3]]*4,np.float32)
    eq=initialization_check(base,28)
    net=MomentNet(28,base,'moment_neural_prior')
    xx=torch.randn(8,28);hh=torch.arange(8)%4;ll=torch.linspace(.2,.9,8)
    pa,_=net(xx,hh,ll);loss=direct.crps_rows(torch.linspace(-1,1,8),pa).mean();loss.backward()
    assert all(t.grad is None or torch.isfinite(t.grad).all() for t in net.parameters())
    return dict(mean_identity_max_error=e1,variance_identity_max_error=e2,
                risk_budget_max_error=e3,zerohead_distribution_max_error=eq,gradients_finite=True,
                CRPS_sanity=direct.loss_sanity())


def train_predict(D,model,seed,epochs):
    # Factory substitution is confined to this process; no old source is edited.
    original=direct.PriorNet
    direct.PriorNet=MomentNet
    try:
        return direct.train_predict(D,model,'crps',seed,epochs)
    finally:
        direct.PriorNet=original


def summarize(report):
    pairs=[('moment_static_prior','direct_static_prior'),('moment_rule_prior','direct_rule_prior'),
      ('moment_neural_prior','direct_neural_prior'),('moment_neural_prior','moment_rule_prior'),
      ('moment_neural_prior','moment_static_prior'),('moment_neural_fixed_budget','moment_neural_prior'),
      ('moment_neural_fixed_budget','direct_neural_prior')]
    pairs += [(m,b) for m in MODELS for b in ['empirical_constant','gaussian_constant']]
    cells=[c['scores'] for m in report['markets'].values() for c in m['by_horizon'].values()]
    out={}
    for model,base in pairs:
        out[model+'_vs_'+base]={}
        for metric in ['MQS','CRPS','Brier']:
            rr=np.array([c[model][metric]/max(c[base][metric],1e-15) for c in cells])
            out[model+'_vs_'+base][metric]=dict(cells=len(rr),wins=int(np.sum(rr<1)),
                geometric_gain_percent=float(100*(1-np.exp(np.log(np.maximum(rr,1e-15)).mean()))))
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--markets',nargs='+',default=list(CUT))
    ap.add_argument('--seeds',nargs='+',type=int,default=[11,22]);ap.add_argument('--epochs',type=int,default=48)
    ap.add_argument('--tag',default='marked_moment_round2');args=ap.parse_args()
    assert args.tag.startswith('marked_moment_round2')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    checks=sanity();print('sanity',json.dumps(checks),flush=True)
    old=np.load(HERE/'marked_event_round2_predictions.npz')
    old_json=json.loads((HERE/'marked_event_round2.json').read_text(encoding='utf-8'))
    panel=pd.read_csv(PANEL,parse_dates=['date']);start=time.perf_counter()
    report=dict(status='exploratory reused test periods; no success or novelty claim',config=vars(args),
      horizons=direct.HOR.tolist(),quantiles=TAUS.tolist(),sanity=checks,
      mechanism='M=mu/p+s/sqrt(p)*U, E(U)=0 Var(U)=1; E(Y)=mu Var(Y)=s^2+(1-p)*mu^2/p',
      architecture='same 24/16 MLP, 2 prior outputs and 5 mark outputs; unrestricted component centers',
      objective='exact analytic CRPS on standardized returns; common validation MQS selection and full-precut refit',
      initialize='strict equality to actual old static-p mixture, including +.025 component std floors',
      ensemble='equal seed distribution mixture; CDF quantiles and exact mixture CRPS',
      horizon_unit='observed daily-panel steps; direct endpoint marginal, not a path update model',
      empirical_baseline='MQS linear sample quantiles, CRPS complete full-precut ECDF, Brier Beta-smoothed frequency; distinct representations',
      reused_direct_predictions='marked_event_round2 rolling/static/rule CRPS; no retraining',
      panel_sha256=hashlib.sha256(PANEL.read_bytes()).hexdigest(),markets={})
    arrays={};csv=[]
    mapping={'direct_static_prior':'static_prior_crps','direct_rule_prior':'rolling_rule_prior_crps',
             'direct_neural_prior':'rolling_neural_prior_crps','empirical_constant':'empirical_constant','gaussian_constant':'gaussian_constant'}
    for market in args.markets:
        d=panel[panel.market==market].sort_values('date').reset_index(drop=True);D=direct.build(d)
        te=D['te'];y=D['y'][te];hi=D['hi'][te];sc=D['scale'][te]
        assert np.array_equal(D['t'][te],old[market+'_origin'])
        assert np.array_equal(direct.HOR[hi],old[market+'_horizon']) and np.array_equal(y,old[market+'_target'])
        init_errors={part:initialization_check(initial(D,D[part]),D['x'].shape[1]) for part in ['tr','rf']}
        mres=dict(audit=D['audit'],initialization_errors=init_errors,aligned_with_saved_direct=True,fits={},by_horizon={})
        preds={name:old[market+'_'+source] for name,source in mapping.items()}
        crps={name:old[market+'_'+source+'_CRPS'] for name,source in mapping.items()}
        probs={name:old[market+'_'+source+'_prob_nonzero'] for name,source in mapping.items()}
        for k,val in [('origin',D['t'][te]),('horizon',direct.HOR[hi]),('target',y),('scale',sc),('local_prior',D['local'][te])]:
            arrays[market+'_'+k]=val
        arrays[market+'_origin_date']=d.date.to_numpy()[D['t'][te]].astype('datetime64[D]')
        for name,source in mapping.items():
            if name.startswith('direct_'):
                for seed in args.seeds:
                    arrays[f'{market}_{name}_seed{seed}_parameters']=old[f'{market}_{source}_seed{seed}_parameters']
        for model in MODELS:
            pars=[];fits=[]
            for seed in args.seeds:
                par,fit=train_predict(D,model,seed,args.epochs);pars.append(par);fits.append(fit)
                arrays[f'{market}_{model}_seed{seed}_parameters']=par
                mu,ss,vv=moments(torch.tensor(par))
                arrays[f'{market}_{model}_seed{seed}_moments']=np.column_stack([mu.numpy(),ss.numpy(),vv.numpy()])
                print(market,model,seed,'epoch',fit['best_epoch'],'elapsed',round(time.perf_counter()-start,1),flush=True)
            comp=direct.components(pars)
            preds[model]=direct.mixture_quantiles(comp)*sc[:,None]
            crps[model]=direct.mixture_crps(D['z'][te],comp)*sc
            probs[model]=np.mean(np.stack([p[:,0] for p in pars]),axis=0)
            mres['fits'][model]=fits
        for k,h in enumerate(direct.HOR):
            mask=hi==k;scores={}
            for model,q in preds.items():
                ym,qm,pm=y[mask],q[mask],probs[model][mask];binary=(ym!=0).astype(float)
                pc=np.clip(pm,1e-12,1-1e-12)
                scores[model]=dict(MQS=float(score(ym,qm).mean()),WIS=float(wis(ym,qm).mean()),
                  CRPS=float(crps[model][mask].mean()),Brier=float(((pm-binary)**2).mean()),
                  log_loss_nonzero=float((-binary*np.log(pc)-(1-binary)*np.log1p(-pc)).mean()),
                  coverage80=float(((ym>=qm[:,2])&(ym<=qm[:,12])).mean()),
                  coverage95=float(((ym>=qm[:,0])&(ym<=qm[:,14])).mean()),
                  MAE_median=float(np.abs(ym-qm[:,7]).mean()),mean_nonzero_probability=float(pm.mean()),
                  empirical_nonzero_rate=float(binary.mean()))
            mres['by_horizon'][str(h)]=dict(n=int(mask.sum()),scores=scores)
        for model,q in preds.items():
            arrays[market+'_'+model]=q;arrays[market+'_'+model+'_CRPS']=crps[model]
            arrays[market+'_'+model+'_prob_nonzero']=probs[model]
            for i,t in enumerate(D['t'][te]):
                row=dict(market=market,model=model,origin=int(t),origin_date=str(d.date.iloc[t].date()),
                  target_date=str(d.date.iloc[t+direct.HOR[hi[i]]].date()),horizon=int(direct.HOR[hi[i]]),
                  target=float(y[i]),scale=float(sc[i]),prob_nonzero=float(probs[model][i]),CRPS=float(crps[model][i]))
                row.update({f'q{float(tau):.3f}':float(q[i,j]) for j,tau in enumerate(TAUS)});csv.append(row)
        report['markets'][market]=mres;report['summary']=summarize(report);report['elapsed_seconds']=time.perf_counter()-start
        (HERE/(args.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        np.savez_compressed(HERE/(args.tag+'_predictions.npz'),**arrays)
        pd.DataFrame(csv).to_csv(HERE/(args.tag+'_predictions.csv'),index=False)
        print(market,'complete',round(time.perf_counter()-start,1),flush=True)
    report['elapsed_seconds']=time.perf_counter()-start
    (HERE/(args.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report['summary'].items() if k.startswith('moment_neural_prior')},indent=2),flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
