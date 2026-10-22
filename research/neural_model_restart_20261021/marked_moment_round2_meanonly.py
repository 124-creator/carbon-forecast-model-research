# -*- coding: utf-8 -*-
"""Mean-only control for the frozen frequency/moment decoder. Writes own prefix.
No claim of a new two-part distribution family or fully p-independent variance.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import marked_moment_round2 as moment
import marked_event_round2 as direct
from marked_event_round1 import quantiles
from neural_restart import PANEL,CUT,TAUS,score,wis
HERE=Path(__file__).resolve().parent
NAME='meanonly_neural_prior'


def meanonly_from_budget(par,p0):
    p,w,m1,m2,s1,s2=par.unbind(1)
    center=w*m1+(1-w)*m2
    r=torch.sqrt(p/p0)  # Necessary differentiable dependence; never detached.
    return torch.stack([p,w,center+r*(m1-center),center+r*(m2-center),r*s1,r*s2],1)


class MeanOnlyNet(moment.MomentNet):
    def forward(self,x,hi,local):
        par,aux=super().forward(x,hi,local)
        p0=torch.sigmoid(self.base[hi,0]).double()
        return meanonly_from_budget(par,p0),aux


def sanity():
    p=torch.tensor([.05,.25,.7,.97],dtype=torch.float64,requires_grad=True)
    mu=torch.tensor([.2,-.1,.3,.05],dtype=torch.float64);s=torch.tensor([.8,1.1,.5,.7],dtype=torch.float64)
    p0=torch.full_like(p,.6);w=torch.full_like(p,.37);dd=torch.full_like(p,1.2);aa=torch.full_like(p,.4)
    new=meanonly_from_budget(moment.decode(p,mu,s,w,dd,aa),p0)
    ey,_,var=moment.moments(new)
    target=(p/p0)*s*s+(1-p)*mu*mu/p
    me=float((ey-mu).abs().max());ve=float((var-target).abs().max())
    grad=torch.autograd.grad(var.sum(),p)[0];ge=float((grad-(s*s/p0-mu*mu/(p*p))).abs().max())
    assert me<1e-10 and ve<1e-10 and ge<1e-9
    base=np.array([[.3,-.4,.2,-.1,-.2,.3]]*4,np.float32)
    old=moment.MomentNet(28,base,'moment_static_prior');net=MeanOnlyNet(28,base,'moment_static_prior')
    x=torch.zeros(4,28);hi=torch.arange(4);local=torch.full((4,),.7)
    with torch.no_grad():
        orig,_=old(x,hi,local);same,_=net(x,hi,local)
    eq=float((orig-same).abs().max());assert eq<1e-5
    nn=MeanOnlyNet(28,base,'moment_neural_prior');out,_=nn(torch.randn(8,28),torch.arange(8)%4,torch.linspace(.2,.9,8))
    loss=direct.crps_rows(torch.linspace(-1,1,8),out).mean();loss.backward()
    assert all(t.grad is None or torch.isfinite(t.grad).all() for t in nn.parameters())
    return dict(mean_error=me,variance_formula_error=ve,variance_probability_gradient_error=ge,zerohead_p_equals_p0_error=eq,finite_gradients=True)


def train(D,seed,epochs):
    orig=direct.PriorNet;direct.PriorNet=MeanOnlyNet
    try:
        return direct.train_predict(D,'moment_neural_prior','crps',seed,epochs)
    finally:
        direct.PriorNet=orig


def summarize(report):
    cells=[c['scores'] for m in report['markets'].values() for c in m['by_horizon'].values()]
    result={}
    for label,model,base in [('variance_budget_vs_meanonly','moment_neural_prior',NAME),('meanonly_vs_direct',NAME,'direct_neural_prior'),('meanonly_vs_empirical',NAME,'empirical_constant')]:
        result[label]={}
        for metric in ['MQS','CRPS','Brier']:
            rr=[c[model][metric]/max(c[base][metric],1e-15) for c in cells]
            result[label][metric]=dict(geometric_gain_percent=float(100*(1-np.exp(np.log(np.maximum(rr,1e-15)).mean()))),wins=int(np.sum(np.array(rr)<1)),cells=len(rr))
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--markets',nargs='+',default=list(CUT));ap.add_argument('--seeds',nargs='+',type=int,default=[11,22]);ap.add_argument('--epochs',type=int,default=48);ap.add_argument('--tag',default='marked_moment_round2_meanonly');args=ap.parse_args()
    assert args.tag.startswith('marked_moment_round2_meanonly')
    frozen={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('marked_moment_round2*') if p.is_file() and not p.name.startswith('marked_moment_round2_meanonly')}
    torch.set_num_threads(1);torch.set_num_interop_threads(1);checks=sanity();print('sanity',json.dumps(checks),flush=True)
    old=np.load(HERE/'marked_moment_round2_predictions.npz');panel=pd.read_csv(PANEL,parse_dates=['date']);start=time.perf_counter()
    report=dict(status='exploratory mean-only control; no retraining of frozen baselines',config=vars(args),sanity=checks,
      mechanism='mean-only M=mu/p+s/sqrt(p0_static)*U; E(Y)=mu; Var(Y)=(p/p0)*s^2+(1-p)*mu^2/p',
      comparator='variance-budget M=mu/p+s/sqrt(p)*U; Var(Y)=s^2+(1-p)*mu^2/p',
      gradient='through sqrt(p/p0), never detached',frozen_sha256=frozen,markets={})
    arrays={};rows=[];seed_rr={str(seed):{'MQS':[],'CRPS':[]} for seed in args.seeds}
    for market in args.markets:
        data=panel[panel.market==market].sort_values('date').reset_index(drop=True);D=direct.build(data);te=D['te'];y=D['y'][te];hi=D['hi'][te];sc=D['scale'][te]
        assert np.array_equal(D['t'][te],old[market+'_origin']) and np.array_equal(direct.HOR[hi],old[market+'_horizon']) and np.array_equal(y,old[market+'_target'])
        preds={m:old[market+'_'+m] for m in ['moment_neural_prior','direct_neural_prior','empirical_constant']}
        cs={m:old[market+'_'+m+'_CRPS'] for m in preds};ps={m:old[market+'_'+m+'_prob_nonzero'] for m in preds}
        pars=[];fits=[]
        for seed in args.seeds:
            par,fit=train(D,seed,args.epochs);pars.append(par);fits.append(fit);arrays[f'{market}_{NAME}_seed{seed}_parameters']=par
            bp=old[f'{market}_moment_neural_prior_seed{seed}_parameters'];qnew=quantiles(par)*sc[:,None];qold=quantiles(bp)*sc[:,None]
            cnew=direct.mixture_crps(D['z'][te],direct.components([par]))*sc;cold=direct.mixture_crps(D['z'][te],direct.components([bp]))*sc
            for h in direct.HOR:
                mask=direct.HOR[hi]==h;seed_rr[str(seed)]['MQS'].append(float(score(y[mask],qold[mask]).mean()/score(y[mask],qnew[mask]).mean()));seed_rr[str(seed)]['CRPS'].append(float(cold[mask].mean()/cnew[mask].mean()))
            print(market,'meanonly',seed,'epoch',fit['best_epoch'],'elapsed',round(time.perf_counter()-start,1),flush=True)
        comp=direct.components(pars);preds[NAME]=direct.mixture_quantiles(comp)*sc[:,None];cs[NAME]=direct.mixture_crps(D['z'][te],comp)*sc;ps[NAME]=np.mean(np.stack([p[:,0] for p in pars]),axis=0)
        mres=dict(audit=D['audit'],fits=fits,aligned_with_frozen=True,by_horizon={})
        for k,h in enumerate(direct.HOR):
            mask=hi==k;scores={}
            for model,q in preds.items():
                yy,qq,pp=y[mask],q[mask],ps[model][mask];b=(yy!=0).astype(float)
                scores[model]=dict(MQS=float(score(yy,qq).mean()),CRPS=float(cs[model][mask].mean()),WIS=float(wis(yy,qq).mean()),Brier=float(((pp-b)**2).mean()),coverage80=float(((yy>=qq[:,2])&(yy<=qq[:,12])).mean()),coverage95=float(((yy>=qq[:,0])&(yy<=qq[:,14])).mean()),mean_nonzero_probability=float(pp.mean()))
            mres['by_horizon'][str(h)]=dict(n=int(mask.sum()),scores=scores)
        for suffix,val in [('origin',D['t'][te]),('horizon',direct.HOR[hi]),('target',y),('scale',sc)]:arrays[market+'_'+suffix]=val
        for model,q in preds.items():
            arrays[market+'_'+model]=q;arrays[market+'_'+model+'_CRPS']=cs[model];arrays[market+'_'+model+'_prob_nonzero']=ps[model]
            for i,t in enumerate(D['t'][te]):
                row=dict(market=market,model=model,origin=int(t),origin_date=str(data.date.iloc[t].date()),target_date=str(data.date.iloc[t+direct.HOR[hi[i]]].date()),horizon=int(direct.HOR[hi[i]]),target=float(y[i]),scale=float(sc[i]),prob_nonzero=float(ps[model][i]),CRPS=float(cs[model][i]));row.update({f'q{float(tau):.3f}':float(q[i,j]) for j,tau in enumerate(TAUS)});rows.append(row)
        report['markets'][market]=mres;report['summary']=summarize(report);report['elapsed_seconds']=time.perf_counter()-start
        (HERE/(args.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');np.savez_compressed(HERE/(args.tag+'_predictions.npz'),**arrays);pd.DataFrame(rows).to_csv(HERE/(args.tag+'_predictions.csv'),index=False)
    report['seed_variance_budget_vs_meanonly']={seed:{metric:dict(geometric_gain_percent=float(100*(1-np.exp(np.log(values).mean()))),wins=int(np.sum(np.array(values)<1)),cells=len(values)) for metric,values in metrics.items()} for seed,metrics in seed_rr.items()}
    assert all(hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha for name,sha in frozen.items())
    report['frozen_files_sha_unchanged']=True;report['elapsed_seconds']=time.perf_counter()-start
    (HERE/(args.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 边际均值重参数化后的方差预算增量控制','',
      '只新增mean-only单模型×五市场×seed11/22，CRPS48轮上限，原样复用数据、归一化、inner验证及full-precut refit训练器。旧矩预算模型及全部既有同前缀文件SHA核对保持不变。','',
      'mean-only使用M=mu/p+s/sqrt(p0_static)*U；保持E(Y)=mu，其Var(Y)=(p/p0_static)*s²+(1-p)*mu²/p。原预算模型使用sqrt(p)分母，其Var(Y)=s²+(1-p)*mu²/p。这不是总体方差完全与p无关的模型。缩放sqrt(p/p0)保持梯度，p=p0时严格同初始化，主干与五mark输出相同。','',
      '原模型属于已有two-part分布族的重参数化，mean-only思想不作首创声明；本控制仅识别方差预算缩放约束的贡献。','',
      '| 对照 | MQS几何改善 | 完整CRPS几何改善 |','|---|---:|---:|']
    for label,v in report['summary'].items():lines.append(f"| {label} | {v['MQS']['geometric_gain_percent']:+.5f}% ({v['MQS']['wins']}/20) | {v['CRPS']['geometric_gain_percent']:+.5f}% ({v['CRPS']['wins']}/20) |")
    lines+=['','## 分市场：方差预算 vs mean-only','', '| 市场 | MQS改善 | CRPS改善 |','|---|---:|---:|']
    for market,m in report['markets'].items():
        result={metric:100*(1-np.exp(np.mean([np.log(c['scores']['moment_neural_prior'][metric]/c['scores'][NAME][metric]) for c in m['by_horizon'].values()]))) for metric in ['MQS','CRPS']};lines.append(f"| {market} | {result['MQS']:+.5f}% | {result['CRPS']:+.5f}% |")
    lines+=['','## 两seed自身方向（均保留，不选择）','', '| seed | 方差预算/mean-only MQS改善 | 完整CRPS改善 |','|---|---:|---:|']
    for seed,v in report['seed_variance_budget_vs_meanonly'].items():lines.append(f"| {seed} | {v['MQS']['geometric_gain_percent']:+.5f}% ({v['MQS']['wins']}/20) | {v['CRPS']['geometric_gain_percent']:+.5f}% ({v['CRPS']['wins']}/20) |")
    lines+=['','全部市场/h的MQS、完整CRPS、coverage80/95、Brier和训练轮数均保留在独立JSON，逐原点预测及每seed参数单独保存。原模型与控制共享零atom/两normal预测族，分量均值不解释为正负分类。','',
      '边界：这是历史复用测试上的探索性消融，不证明新颖性、显著性或总体预测成功；预算约束仅为可检验假设，不是经济风险普遍固定的真理。经验基准CRPS用完整训练样本ECDF、MQS用linear样本分位数、Brier用Beta平滑频率，三种经验表示不同。', '',f"本次运行{report['elapsed_seconds']:.2f}秒；初始化/均值/方差/梯度检查及所有旧文件SHA核对通过。"]
    (HERE/(args.tag+'_report_zh.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(report['summary'],indent=2),flush=True);print('seed',json.dumps(report['seed_variance_budget_vs_meanonly']),flush=True);print('elapsed',report['elapsed_seconds'],'frozen sha unchanged',True,flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
