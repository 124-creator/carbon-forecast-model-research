# -*- coding: utf-8 -*-
"""Bounded state-only controls and validation-selected context. Own prefixes only."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import json,time,hashlib,argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import atom_spline_round3 as core
from neural_restart import PANEL,CUT,TAUS,score,wis
HERE=Path(__file__).resolve().parent
NEW={'state_continuous':'continuous_spline','state_static_atom':'static_atom_spline','state_rule_atom':'rule_atom_spline','state_neural_bce005':'neural_atom_spline'}
STATE='state_neural_original'


class StateNet(core.SplineNet):
    def forward(self,x,hi,local):
        x=x.clone();x[:,-8:]=0
        return super().forward(x,hi,local)


def fit(D,name,seed,epochs):
    cls,loss=core.SplineNet,core.exact_crps
    core.SplineNet=StateNet
    if name=='state_neural_bce005':
        def auxiliary(y,ks,ps,u):
            base=loss(y,ks,ps,u)
            p=ps[0].clamp(1e-7,1-1e-7);b=(y!=0).to(p.dtype)
            return base+.05*(-b*torch.log(p)-(1-b)*torch.log1p(-p))
        core.exact_crps=auxiliary
    try:
        return core.fit_predict(D,NEW[name],seed,epochs)
    finally:
        core.SplineNet=cls;core.exact_crps=loss


def scores(y,hi,preds,crps,probs):
    out={}
    for j,h in enumerate(core.old.HOR):
        mask=hi==j;ss={}
        for m,q in preds.items():
            yy,qq,pp=y[mask],q[mask],probs[m][mask];b=(yy!=0)
            ss[m]=dict(MQS=float(score(yy,qq).mean()),CRPS=float(crps[m][mask].mean()),WIS=float(wis(yy,qq).mean()),coverage80=float(((yy>=qq[:,2])&(yy<=qq[:,12])).mean()),coverage95=float(((yy>=qq[:,0])&(yy<=qq[:,14])).mean()),Brier=float(((pp-b)**2).mean()),mean_nonzero_probability=float(pp.mean()))
        out[str(h)]=dict(n=int(mask.sum()),scores=ss)
    return out


def summary(markets,pairs):
    cells=[c['scores'] for v in markets.values() for c in v['by_horizon'].values()];out={}
    for model,base in pairs:
        out[model+'_vs_'+base]={}
        for metric in ['MQS','CRPS','Brier']:
            rr=[c[model][metric]/max(c[base][metric],1e-15) for c in cells]
            out[model+'_vs_'+base][metric]=dict(geometric_gain_percent=float(100*(1-np.exp(np.log(np.maximum(rr,1e-15)).mean()))),wins=int(np.sum(np.array(rr)<1)),cells=len(rr))
    return out


def add_rows(rows,market,data,D,q,cr,p,model):
    te=D['te'];hi=D['hi'][te]
    for i,t in enumerate(D['t'][te]):
        row=dict(market=market,model=model,origin=int(t),origin_date=str(data.date.iloc[t].date()),target_date=str(data.date.iloc[t+core.old.HOR[hi[i]]].date()),horizon=int(core.old.HOR[hi[i]]),target=float(D['y'][te[i]]),scale=float(D['scale'][te[i]]),prob_nonzero=float(p[i]),CRPS=float(cr[i]));row.update({f'q{float(tau):.3f}':float(q[i,j]) for j,tau in enumerate(TAUS)});rows.append(row)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=48);args=ap.parse_args()
    tag='atom_spline_round3_statecontrols';ctag='atom_spline_round3_contextselected'
    frozen={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('atom_spline_round3*') if p.is_file() and not p.name.startswith(tag) and not p.name.startswith(ctag)}
    torch.set_num_threads(1);torch.set_num_interop_threads(1);u=torch.tensor(core.U,dtype=torch.float32)
    panel=pd.read_csv(PANEL,parse_dates=['date']);a=np.load(HERE/'atom_spline_round3_predictions.npz');old=json.loads((HERE/'atom_spline_round3.json').read_text(encoding='utf-8'));start=time.perf_counter()
    r=dict(status='bounded exploratory state-only controls, no old neural refitting',epochs=args.epochs,seeds=[11,22],objective='exact CRPS; BCE exploratory variant adds fixed .05 nonzero-event BCE during training only; evaluation remains exact CRPS',selection='same normalized validation MQS including epoch0, full-precut refit',frozen_sha256=frozen,markets={})
    ctx=dict(status='context chosen only by existing inner validation, no test tuning',rule='each market/seed choose lower stored best validation MQS; difference<=1e-12 chooses state-only',choices={},markets={})
    arrays={};rows=[];ca={};crows=[];seed_ratios={str(seed):{} for seed in [11,22]}
    pairs=[(STATE,'state_continuous'),(STATE,'state_static_atom'),(STATE,'state_rule_atom'),('state_neural_bce005',STATE),('state_neural_bce005','state_rule_atom')]+[(m,'empirical_constant') for m in [STATE]+list(NEW)]
    for market in CUT:
        data=panel[panel.market==market].sort_values('date').reset_index(drop=True);D=core.build(data);te=D['te'];hi=D['hi'][te];y=D['y'][te];sc=D['scale'][te]
        assert np.array_equal(D['t'][te],a[market+'_origin']) and np.array_equal(y,a[market+'_target']) and np.array_equal(core.old.HOR[hi],a[market+'_horizon'])
        # No retraining: choose stored fitted seed distributions from inner scores.
        cks=[];cps=[];decisions=[]
        for i,seed in enumerate([11,22]):
            av=old['markets'][market]['fits']['neural_atom_spline'][i]['validation_MQS_normalized'];sv=old['markets'][market]['fits']['neural_atom_no_price_calendar'][i]['validation_MQS_normalized']
            choice='neural_atom_spline' if av<sv-1e-12 else 'neural_atom_no_price_calendar'
            decisions.append(dict(seed=seed,aug_validation=av,state_validation=sv,chosen=choice))
            ck=torch.tensor(a[f'{market}_{choice}_seed{seed}_knots']);cp=torch.tensor(a[f'{market}_{choice}_seed{seed}_p']);cks.append(ck);cps.append(cp);ca[f'{market}_seed{seed}_knots']=ck.numpy();ca[f'{market}_seed{seed}_p']=cp.numpy()
        with torch.no_grad():cq=core.ensemble_quantiles(cks,cps,u).numpy()*sc[:,None];cc=core.exact_crps(torch.tensor(y/sc,dtype=torch.float32),cks,cps,u).numpy()*sc
        cp=torch.stack(cps).mean(0).numpy();ctx['choices'][market]=decisions
        cpreds={'contextselected':cq,'all_aug':a[market+'_neural_atom_spline'],'all_state':a[market+'_neural_atom_no_price_calendar'],'empirical_constant':a[market+'_empirical_constant'],'double_normal':a[market+'_double_normal']}
        ccr={'contextselected':cc};cprob={'contextselected':cp}
        for name,key in [('all_aug','neural_atom_spline'),('all_state','neural_atom_no_price_calendar'),('empirical_constant','empirical_constant'),('double_normal','double_normal')]:ccr[name]=a[market+'_'+key+'_CRPS'];cprob[name]=a[market+'_'+key+'_prob_nonzero']
        ctx['markets'][market]=dict(by_horizon=scores(y,hi,cpreds,ccr,cprob));add_rows(crows,market,data,D,cq,cc,cp,'contextselected')
        ca[market+'_contextselected']=cq;ca[market+'_CRPS']=cc;ca[market+'_prob_nonzero']=cp
        for suffix,value in [('origin',D['t'][te]),('horizon',core.old.HOR[hi]),('target',y),('scale',sc)]:ca[market+'_'+suffix]=value;arrays[market+'_'+suffix]=value
        pred={STATE:a[market+'_neural_atom_no_price_calendar'],'empirical_constant':a[market+'_empirical_constant'],'double_normal':a[market+'_double_normal'],'gaussian_constant':a[market+'_gaussian_constant']}
        cr={};pr={};fits={};seedpred={};seedcr={}
        for name,key in [(STATE,'neural_atom_no_price_calendar'),('empirical_constant','empirical_constant'),('double_normal','double_normal'),('gaussian_constant','gaussian_constant')]:cr[name]=a[market+'_'+key+'_CRPS'];pr[name]=a[market+'_'+key+'_prob_nonzero']
        for seed in [11,22]:
            k=torch.tensor(a[f'{market}_neural_atom_no_price_calendar_seed{seed}_knots']);p=torch.tensor(a[f'{market}_neural_atom_no_price_calendar_seed{seed}_p']);seedpred[(seed,STATE)]=core.analytic_quantiles(k,p,u).numpy()*sc[:,None];seedcr[(seed,STATE)]=core.exact_crps(torch.tensor(y/sc,dtype=torch.float32),[k],[p],u).numpy()*sc;arrays[f'{market}_{STATE}_seed{seed}_knots']=k.numpy();arrays[f'{market}_{STATE}_seed{seed}_p']=p.numpy()
        for model in NEW:
            ks=[];ps=[];fs=[]
            for seed in [11,22]:
                k,p,f=fit(D,model,seed,args.epochs);ks.append(k);ps.append(p);fs.append(f);arrays[f'{market}_{model}_seed{seed}_knots']=k.numpy();arrays[f'{market}_{model}_seed{seed}_p']=p.numpy();seedpred[(seed,model)]=core.analytic_quantiles(k,p,u).detach().numpy()*sc[:,None];seedcr[(seed,model)]=core.exact_crps(torch.tensor(y/sc,dtype=torch.float32),[k],[p],u).detach().numpy()*sc;print(market,model,seed,'epoch',f['best_epoch'],'elapsed',round(time.perf_counter()-start,1),flush=True)
            with torch.no_grad():pred[model]=core.ensemble_quantiles(ks,ps,u).numpy()*sc[:,None];cr[model]=core.exact_crps(torch.tensor(y/sc,dtype=torch.float32),ks,ps,u).numpy()*sc
            pr[model]=torch.stack(ps).mean(0).numpy();fits[model]=fs
        for seed in [11,22]:
            for model,base in pairs:
                if base not in ['state_continuous','state_static_atom','state_rule_atom',STATE,'empirical_constant']:continue
                key=model+'_vs_'+base;store=seed_ratios[str(seed)].setdefault(key,{'MQS':[],'CRPS':[]});qb=pred[base] if base=='empirical_constant' else seedpred[(seed,base)];cb=cr[base] if base=='empirical_constant' else seedcr[(seed,base)]
                for h in range(4):
                    mask=hi==h;store['MQS'].append(float(score(y[mask],seedpred[(seed,model)][mask]).mean()/score(y[mask],qb[mask]).mean()));store['CRPS'].append(float(seedcr[(seed,model)][mask].mean()/cb[mask].mean()))
        r['markets'][market]=dict(audit=D['audit'],fits=fits,by_horizon=scores(y,hi,pred,cr,pr),aligned_with_original=True)
        for model,q in pred.items():arrays[market+'_'+model]=q;arrays[market+'_'+model+'_CRPS']=cr[model];arrays[market+'_'+model+'_prob_nonzero']=pr[model];add_rows(rows,market,data,D,q,cr[model],pr[model],model)
        r['summary']=summary(r['markets'],pairs);r['elapsed_seconds']=time.perf_counter()-start
        (HERE/(tag+'.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');np.savez_compressed(HERE/(tag+'_predictions.npz'),**arrays);pd.DataFrame(rows).to_csv(HERE/(tag+'_predictions.csv'),index=False)
    r['seed_summary']={s:{key:{metric:dict(geometric_gain_percent=float(100*(1-np.exp(np.log(v).mean()))),wins=int(np.sum(np.array(v)<1)),cells=len(v)) for metric,v in ms.items()} for key,ms in ss.items()} for s,ss in seed_ratios.items()}
    ctx['summary']=summary(ctx['markets'],[('contextselected',b) for b in ['all_aug','all_state','empirical_constant','double_normal']]);ctx['choice_counts']={name:sum(z['chosen']==name for choices in ctx['choices'].values() for z in choices) for name in ['neural_atom_spline','neural_atom_no_price_calendar']}
    assert all(hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha for name,sha in frozen.items());r['frozen_sha_unchanged']=True;ctx['frozen_sha_unchanged']=True;r['elapsed_seconds']=time.perf_counter()-start
    (HERE/(tag+'.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');(HERE/(ctag+'.json')).write_text(json.dumps(ctx,ensure_ascii=False,indent=2),encoding='utf-8');np.savez_compressed(HERE/(ctag+'_predictions.npz'),**ca);pd.DataFrame(crows).to_csv(HERE/(ctag+'_predictions.csv'),index=False)
    lines=['# State-only同信息结构控制与固定BCE探索','',f'完成40fit，运行{r["elapsed_seconds"]:.2f}秒。原神经state模型直接复用，没有重训。所有模型相同36维布局，价格日历8维置0、同正间距rank4样条、同CRPS/epoch0选择与full-precut refit。BCE组仅训练目标增加预先固定0.05*BCE(nonzero)，不调lambda；评估CRPS始终为原完整精确CDF积分。','', '| 对照 | MQS改善 | 完整CRPS改善 |','|---|---:|---:|']
    for key,v in r['summary'].items():lines.append(f"| {key} | {v['MQS']['geometric_gain_percent']:+.4f}% ({v['MQS']['wins']}/20) | {v['CRPS']['geometric_gain_percent']:+.4f}% ({v['CRPS']['wins']}/20) |")
    lines+=['','## 两seed自身方向（均保留）','', '| seed | 对照 | MQS改善 | 完整CRPS改善 |','|---|---|---:|---:|']
    for seed,ss in r['seed_summary'].items():
        for key,v in ss.items():lines.append(f"| {seed} | {key} | {v['MQS']['geometric_gain_percent']:+.4f}% ({v['MQS']['wins']}/20) | {v['CRPS']['geometric_gain_percent']:+.4f}% ({v['CRPS']['wins']}/20) |")
    lines+=['','完整逐market/h的MQS、完整CRPS、WIS、coverage80/95、Brier及验证历史均保存在JSON；逐原点另存CSV/NPZ。原输出SHA均未变。历史测试复用，仅探索，不按本轮test重选最终模型或宣告创新成功。']
    (HERE/(tag+'_report_zh.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    clines=['# 仅验证集选择context的独立组合','', '每market/seed比较既有best validation normalized MQS，更低者入选；差<=1e-12选state。零次重训，没有用test选context。选出seed分布后重新合并所有knots+0，精确积分同一F的CRPS，CDF二分同一F求quantiles。','', '| market | seed | aug best val | state best val | chosen |','|---|---:|---:|---:|---|']
    for market,choices in ctx['choices'].items():
        for z in choices:clines.append(f"| {market} | {z['seed']} | {z['aug_validation']:.9f} | {z['state_validation']:.9f} | {z['chosen']} |")
    clines+=['', '| 对照 | MQS改善 | 完整CRPS改善 |','|---|---:|---:|']
    for key,v in ctx['summary'].items():clines.append(f"| {key} | {v['MQS']['geometric_gain_percent']:+.4f}% ({v['MQS']['wins']}/20) | {v['CRPS']['geometric_gain_percent']:+.4f}% ({v['CRPS']['wins']}/20) |")
    clines+=['','验证支持aug/state各5个设置；这与test更喜欢全state并不相同，不能以test反向取代这个验证规则。全部原点/目标/h与原实验一致，旧SHA全部保留。']
    (HERE/(ctag+'_report_zh.md')).write_text('\n'.join(clines)+'\n',encoding='utf-8')
    print('state_summary',json.dumps(r['summary']),flush=True);print('context_summary',json.dumps(ctx['summary']),flush=True);print('seed_summary',json.dumps(r['seed_summary']),flush=True);print('elapsed',r['elapsed_seconds'],'all frozen SHA unchanged',True,flush=True)

if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');main()
