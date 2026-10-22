# -*- coding: utf-8 -*-
"""Round5: sample-size adaptive shrinkage of the forest conditional mark bank.

u(x) = (1-lam(x)) u_forest(x) + lam(x) u_uniform(x)
lam(x) = clip(lam0 * kappa/(kappa+ESS(x)), 0, 1),  ESS = 1/sum_i w_i^2.

No parameters are trained. lam0/kappa (or a fixed lambda) are chosen per market on
inner validation rows only, from a grid fixed before the final evaluation.
Identity transport: the mark distribution IS the shrunken reference.
"""
from __future__ import annotations
import os,sys
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
sys.dont_write_bytecode=True
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
import forest_transport_round5 as F
R=F.R;P=Path(__file__).resolve().parent
GAMMA=.05
LAMS=[.05,.10,.20,.30,.50]
LAM0=[.25,.50,.75,1.0]
KAPPA=[8.,16.,32.,64.,128.]
CONFIG=['fixed_%.2f'%v for v in LAMS]+['ess_l%.2f_k%d'%(a,int(k)) for a in LAM0 for k in KAPPA]
ANCHOR='fixed_0.05'


def lam_vector(name,bank,sids):
    allowed=bank['allowed'].float()[sids];count=bank['count'].float()[sids]
    uniform=allowed/count.clamp_min(1)[:,None]
    forest=((bank['u'][sids]-GAMMA*uniform)/.95).clamp_min(0)
    mass=forest.sum(1,keepdim=True)
    norm=forest/mass.clamp_min(1e-30)
    ess=1/norm.square().sum(1).clamp_min(1e-30)
    if name.startswith('fixed_'):lam=torch.full_like(count,float(name.split('_')[1]))
    else:
        a,k=name.split('_')[1:]
        lam0=float(a[1:]);kappa=float(k[1:]);lam=lam0*kappa/(kappa+ess)
    lam=torch.where(mass[:,0]>0,lam.clamp(0,1),torch.ones_like(lam))
    lam=torch.where(count>0,lam,torch.zeros_like(lam))
    return lam,ess,mass[:,0]


def make_ops(bank,rows,sids,name,X):
    lam,ess,mass=lam_vector(name,bank,sids)
    count=bank['count'].float()[sids];allowed=bank['allowed'].float()[sids]
    uniform=allowed/count.clamp_min(1)[:,None]
    forest=((bank['u'][sids]-GAMMA*uniform)/.95).clamp_min(0)
    u=(1-lam[:,None])*forest+lam[:,None]*uniform
    u=torch.where(count[:,None]>0,u,torch.zeros_like(u))
    p=.8*X['local'][torch.tensor(rows)].float()+.2*bank['p']
    return dict(w=u.double(),p=p.double(),a=torch.ones(len(rows),dtype=torch.float64),b=torch.zeros(len(rows),dtype=torch.float64)),ess


def evaluate(D,X,banks,rows,name):
    """Returns per-horizon metrics and per-row (q,c,p) in z units."""
    y=D['y'][rows];scale=D['scale'][rows];hi=D['hi'][rows]
    Q=np.empty((len(rows),15));C=np.empty(len(rows));PP=np.empty(len(rows))
    for j,bank in enumerate(banks):
        pos=np.flatnonzero(hi==j);idx=rows[pos];sids=X['sid'][torch.tensor(idx)]
        ops,_=make_ops(bank,idx,sids,name,X)
        q,c,p,_,_=R.mixture(torch.tensor(D['z'][idx]),bank['z'],[ops])
        Q[pos]=q.numpy();C[pos]=c.numpy();PP[pos]=p.numpy()
    return Q,C,PP


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',default='ess_shrinkage_round5');a=ap.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);F.activate()
    panel=pd.read_csv(R.T.B.PANEL,parse_dates=['date'])
    old=np.load(P/'forest_transport_round5_predictions.npz');rule=np.load(P/'qrf_round4_probability_control_predictions.npz')
    frozen=[P/'forest_transport_round5.py',P/'forest_transport_round5_predictions.npz']
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen}
    report={'status':'no-training ESS-adaptive reference shrinkage; pre-registered grid, inner-selected one config per market',
            'form':'u=(1-lam)u_forest+lam u_uniform; lam=lam0*kappa/(kappa+ESS), ESS=1/sum w^2; fixed-lambda controls included',
            'anchor':'fixed_0.05 must reproduce the frozen forest_reference_only distribution',
            'grid':CONFIG,'selection':'per market, minimum mean_h log(raw MQS / raw anchor MQS) on inner validation rows only',
            'frozen_before':before,'markets':{}}
    (P/(a.tag+'_manifest.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    arrays={};rows_out=[];start=time.perf_counter()
    for market in R.T.B.CUT:
        d=panel[panel.market==market].sort_values('date').reset_index(drop=True);D=R.S.build(d)
        prepared,_,ci=F.load_market(D,market)
        rows_all=np.arange(len(D['t']))
        vr=D['va'];va_scores={}
        for name in CONFIG:
            Xi,bi=prepared['inner'];Q,C,PP=evaluate(D,Xi,bi,vr,name);sc=D['scale'][vr]
            va_scores[name]=float(np.mean([R.metrics(D['y'][vr][D['hi'][vr]==j],Q[D['hi'][vr]==j]*sc[D['hi'][vr]==j,None],
                                               C[D['hi'][vr]==j]*sc[D['hi'][vr]==j],PP[D['hi'][vr]==j])['MQS'] for j in range(4)]))
        anchor=va_scores[ANCHOR];ratios={k:float(np.log(max(v,1e-30)/max(anchor,1e-30))) for k,v in va_scores.items()}
        chosen=min(CONFIG,key=lambda k:(ratios[k],CONFIG.index(k)))
        te=D['te'];Xf,bf=prepared['final'];pred={}
        for name in [chosen,ANCHOR]:
            Q,C,PP=evaluate(D,Xf,bf,te,name);sc=D['scale'][te]
            pred[name]=(Q*sc[:,None],C*sc,PP)
        anchor_old=old[market+'_forest_reference_only_Q']
        diff=float(np.abs(pred[ANCHOR][0]-anchor_old).max());cdiff=float(np.abs(pred[ANCHOR][1]-old[market+'_forest_reference_only_CRPS']).max())
        assert diff<2e-6 and cdiff<2e-7,(diff,cdiff)
        cells={}
        for j,h in enumerate(R.T.HOR):
            ix=D['hi'][te]==j;cell={}
            for name in [chosen,ANCHOR]:cell[name]=R.metrics(D['y'][te][ix],pred[name][0][ix],pred[name][1][ix],pred[name][2][ix])
            for base,file,key in [('rule_qrf','qrf_round4_probability_control_predictions.npz','qrf_rule_atom'),
                                  ('qrf_original','qrf_round4_probability_control_predictions.npz','qrf_original'),
                                  ('lstm_qr','deep_seq_round4_predictions.npz','lstm_qr'),
                                  ('forest_transport','forest_transport_round5_predictions.npz','forest_transport'),
                                  ('forest_reference_only','forest_transport_round5_predictions.npz','forest_reference_only'),
                                  ('honest_no_readout','prior_readout_transport_round5_predictions.npz','honest_no_readout')]:
                source=np.load(P/file);full=market+'_'+key
                cell[base]=R.metrics(D['y'][te][ix],source[full+'_Q'][ix],source[full+'_CRPS'][ix] if full+'_CRPS' in source else None,
                                     source[full+'_p'][ix] if full+'_p' in source else None)
            cells[str(h)]=cell
            cell['chosen']=cell[chosen]
            for model,metric in [(k,v) for k,v in cell.items() for _ in [0]]:pass
            for name,values in cell.items():
                rows_out.append(dict(market=market,horizon=int(h),model=name,**{k:v for k,v in values.items() if k in ['MQS','CRPS','WIS','MAE','Brier','coverage']}))
        arrays[market+'_origin']=D['t'][te];arrays[market+'_horizon']=R.T.HOR[D['hi'][te]];arrays[market+'_target']=D['y'][te]
        for name in [chosen,ANCHOR]:
            arrays[market+'_'+name+'_Q']=pred[name][0];arrays[market+'_'+name+'_CRPS']=pred[name][1];arrays[market+'_'+name+'_p']=pred[name][2]
        report['markets'][market]=dict(chosen=chosen,anchor=ANCHOR,anchor_reproduction_error=dict(Q=diff,CRPS=cdiff),
                                       inner_validation_MQS=va_scores,inner_log_ratio=ratios,by_horizon=cells,
                                       cache_sha=ci['sha256'],inner_valid_rows=len(vr))
        print(market,'chosen',chosen,'ratio',round(ratios[chosen],6),'elapsed',round(time.perf_counter()-start,1),flush=True)
    df=pd.DataFrame(rows_out);df.to_csv(P/(a.tag+'_cells.csv'),index=False,encoding='utf-8-sig')
    np.savez_compressed(P/(a.tag+'_predictions.npz'),**arrays)
    report['summary']={}
    for base in ['rule_qrf','qrf_original','lstm_qr','forest_reference_only','forest_transport','honest_no_readout',ANCHOR]:
        if base==ANCHOR:continue
        report['summary']['chosen_vs_'+base]={}
        for metric in ['MQS','CRPS']:
            rr=[c['chosen'][metric]/c[base][metric] for m in report['markets'].values() for c in m['by_horizon'].values() if c[base][metric] is not None]
            if rr:report['summary']['chosen_vs_'+base][metric]=dict(geometric_gain_percent=float(100*(1-np.exp(np.log(rr).mean()))),
                                                                   wins=int((np.array(rr)<1).sum()),cells=len(rr))
    report['frozen_after']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen};assert before==report['frozen_after']
    report['elapsed_seconds']=time.perf_counter()-start
    (P/(a.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 样本量自适应参考收缩（无训练）','',f'运行{report["elapsed_seconds"]:.2f}秒。无任何参数训练；每市场仅按inner验证行从预先固定网格选一个配置，final只用该配置与固定0.05锚。','',
           'u=(1-lam)u_forest+lam*u_uniform，lam=lam0*kappa/(kappa+ESS)，ESS=1/sum_i w_i^2取自已成熟森林条件权重；fixed_0.05锚必须复现冻结forest_reference_only。','']
    for market,v in report['markets'].items():
        lines.append(f'- {market}: 选中 `{v["chosen"]}`，inner log比值{v["inner_log_ratio"][v["chosen"]]:+.4f}；锚复现误差Q={v["anchor_reproduction_error"]["Q"]:.2e}，CRPS={v["anchor_reproduction_error"]["CRPS"]:.2e}')
    lines+=['','| 对照 | MQS | 完整CRPS |','|---|---:|---:|']
    for base,d in report['summary'].items():
        q=d['MQS'];c=d.get('CRPS')
        lines.append(f'| {base} | {q["geometric_gain_percent"]:+.4f}% ({q["wins"]}/{q["cells"]}) | '+(f'{c["geometric_gain_percent"]:+.4f}% ({c["wins"]}/{c["cells"]})' if c else '不可识别')+' |')
    lines+=['','正值表示收缩参考更好。这是参考构造实验，不涉及神经残差；历史测试已复用，不证明显著性或发表成功。']
    (P/(a.tag+'_report_zh.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with threadpool_limits(limits=1):main()
