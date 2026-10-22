# -*- coding: utf-8 -*-
"""Round5: does cross-market state information improve the conditional reference itself?

Only the *conditioning information* changes; the reference construction (maturity mask,
uniform reservoir gamma=.05, rule zero atom, 5 seeds, leaf sizes) is untouched.
Arms: base (no new columns), cross (4 other markets x {r5, vol20}), perm (same columns,
time order destroyed) -> controls for "more columns" vs "real cross-market information".
Self-check: arm base must reproduce the frozen forest_reference_only distribution.
"""
from __future__ import annotations
import os,sys
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
sys.dont_write_bytecode=True
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
import forest_prior_round5 as FP
import forest_transport_round5 as F
T=FP.T;Q=FP.Q;R=F.R;P=Path(__file__).resolve().parent
GAMMA=.05;SEEDS=[11,22,33,44,55];ARMS=['base','cross','perm']


def market_features(panel,market):
    d=panel[panel.market==market].sort_values('date').reset_index(drop=True)
    own=d.date.to_numpy();cols=[]
    for other in [m for m in T.B.CUT if m!=market]:
        o=panel[panel.market==other].sort_values('date')
        od=o.date.to_numpy();olp=np.log(o.price.to_numpy(float))
        idx=np.searchsorted(od,own,side='right')-1
        ok=idx>=0;lp=np.where(ok,olp[np.clip(idx,0,len(olp)-1)],np.nan)
        ret=np.diff(np.concatenate([[np.nan],lp]))
        r5=np.full(len(own),np.nan);v20=np.full(len(own),np.nan)
        for i in range(len(own)):
            j=idx[i]
            if j<0 or not np.isfinite(olp[j]):continue
            if j>=5 and np.isfinite(olp[j-5]):r5[i]=olp[j]-olp[j-5]
            lo=max(0,j-20)
            if j-lo>=8:v20[i]=float(np.std(np.diff(olp[lo:j+1])))
        cols+= [r5,v20]
    f=np.column_stack(cols).astype(np.float32)
    f=np.where(np.isfinite(f),f,0.).astype(np.float32)
    return d,f


def augment(X,extra,arm,idx,seed=20261021):
    if arm=='base':return X['x'].numpy()
    e=extra.copy()
    if arm=='perm':
        rng=np.random.default_rng(seed)
        for c in range(e.shape[1]):rng.shuffle(e[:,c])
    return np.column_stack([X['x'].numpy(),e[idx]]).astype(np.float32)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',default='market_info_round5');a=ap.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    started=time.perf_counter()
    panel=pd.read_csv(T.B.PANEL,parse_dates=['date'])
    old=np.load(P/'forest_transport_round5_predictions.npz')
    rules=np.load(P/'qrf_round4_probability_control_predictions.npz')
    frozen={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['forest_transport_round5.py','forest_transport_round5_predictions.npz','forest_prior_round5.py','qrf_round4_probability_control_predictions.npz']}
    report=dict(status='information-set test on the frozen reference construction; no training, no test tuning',
                arms={k:v for k,v in [('base','no appended columns'),('cross','4 markets x {r5,log-return vol20} aligned asof to own date'),
                                       ('perm','identical columns, each permuted across own dates with fixed seed')]},
                selfcheck='arm base must match frozen forest_reference_only quantiles within 2e-6',
                frozen_before=frozen,markets={})
    (P/(a.tag+'_manifest.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    arrays={};rows=[]
    LEAF={m:{int(h):int(v['chosen_minleaf']) for h,v in s2['selection'].items()} for m,s2 in json.loads((P/'qrf_round4.json').read_text(encoding='utf-8'))['markets'].items()}
    with threadpool_limits(limits=2):
        for market in T.B.CUT:
            d,extra=market_features(panel,market)
            D=T.add_context(d,T.E.build(d));D['market']=market
            cache=FP.load_cache(market,'final',D)
            fit=D['rf'];X,_=T.prepare(D,fit,True)
            assert X['x'].shape[0]==len(D['t']),(X['x'].shape,len(D['t']))
            sid_of=cache['sid'].numpy()
            valid=cache['valid_sid_h'].numpy()
            n_sid=len(cache['origin_times'])
            te=D['te'];pred={}
            for arm in ARMS:
                xx=augment(X,extra,arm,D['t'])
                Qs=np.zeros((len(te),15));Cs=np.zeros(len(te));PP=np.zeros(len(te))
                for hi,h in enumerate(T.HOR):
                    h=int(h);bank=cache['banks'][hi]
                    fit_h=fit[D['hi'][fit]==hi];nzids=FP.nonzero_ids(D,fit,hi)
                    lookup={int(i):j for j,i in enumerate(fit_h)}
                    cols=np.array([lookup[int(i)] for i in nzids])
                    query_rows=np.flatnonzero(D['hi']==hi);qsids=sid_of[query_rows]
                    raw=np.zeros((n_sid,len(nzids)),dtype=np.float32)
                    leaf=LEAF[market][int(h)]
                    for seed in SEEDS:
                        forest=FP.RandomForestRegressor(min_samples_leaf=leaf,random_state=seed,**Q.RF_PARAMS)
                        forest.fit(xx[fit_h],D['z'][fit_h])
                        ww,error=Q.leaf_weights(forest,xx[fit_h],xx[query_rows],sanity=(seed==11))
                        raw[qsids]+=ww[:,cols]/len(SEEDS)
                    allowed=bank['allowed'].numpy()
                    u,forest_only,uniform,count,fallback=FP.make_reference(raw,allowed,valid[:,hi])
                    ix=np.flatnonzero(D['hi'][te]==hi);sid=sid_of[te[ix]]
                    w=torch.tensor(u[sid],dtype=torch.float64);w=w/w.sum(1,keepdim=True).clamp_min(1e-30)
                    local=torch.tensor(D['local'][te][ix],dtype=torch.float64)
                    p=.8*local+.2*float(bank['p'])
                    op=dict(w=w,p=p,a=torch.ones(len(ix),dtype=torch.float64),b=torch.zeros(len(ix),dtype=torch.float64))
                    q,c,pe,_,_=R.mixture(torch.tensor(D['z'][te][ix],dtype=torch.float64),bank['z'].double(),[op])
                    sc=D['scale'][te][ix];Qs[ix]=q.numpy()*sc[:,None];Cs[ix]=c.numpy()*sc;PP[ix]=pe.numpy()
                pred[arm]=(Qs,Cs,PP)
                arrays[market+'_'+arm+'_Q']=Qs;arrays[market+'_'+arm+'_CRPS']=Cs;arrays[market+'_'+arm+'_p']=PP
            arrays[market+'_origin']=D['t'][te];arrays[market+'_horizon']=T.HOR[D['hi'][te]];arrays[market+'_target']=D['y'][te]
            diff=float(np.abs(pred['base'][0]-old[market+'_forest_reference_only_Q']).max())
            cdiff=float(np.abs(pred['base'][1]-old[market+'_forest_reference_only_CRPS']).max())
            info=dict(reproduction_error=dict(Q=diff,CRPS=cdiff),cells={})
            exact=diff<2e-6 and cdiff<2e-7
            info['reproduction_exact']=bool(exact)
            info['reproduction_note']=('base bitwise reproduces the frozen forest_reference_only quantiles and CRPS' if exact else
                'base does NOT reproduce the frozen artifact; arms are compared under one identical implementation, so the A/B/C contrast stays internally paired but absolute levels must not be appended to the frozen table')
            assert diff<1.0 and cdiff<0.2,(market,diff,cdiff)
            for hi,h in enumerate(T.HOR):
                ix=D['hi'][te]==hi;cell={}
                for arm in ARMS:cell[arm]=R.metrics(D['y'][te][ix],pred[arm][0][ix],pred[arm][1][ix],pred[arm][2][ix])
                cell['rule_qrf']=R.metrics(D['y'][te][ix],rules[market+'_qrf_rule_atom_Q'][ix],None,None)
                info['cells'][str(int(h))]=cell
                for arm,values in cell.items():
                    rows.append(dict(market=market,horizon=int(h),model=arm,MQS=values['MQS'],CRPS=values.get('CRPS')))
            report['markets'][market]=info
            print(market,'repro',f'{diff:.2e}',f'{cdiff:.2e}','elapsed',round(time.perf_counter()-started,1),flush=True)
    df=pd.DataFrame(rows);df.to_csv(P/(a.tag+'_cells.csv'),index=False,encoding='utf-8-sig')
    np.savez_compressed(P/(a.tag+'_predictions.npz'),**arrays)
    summary={}
    for arm in ARMS:
        summary[arm]={}
        for base in ['rule_qrf','base']:
            if base==arm:continue
            for metric in ['MQS','CRPS']:
                r=[c[arm][metric]/c[base][metric] for m in report['markets'].values() for c in m['cells'].values()
                   if c[arm].get(metric) is not None and c[base].get(metric) is not None]
                summary[arm][base+'_'+metric]=dict(gain_percent=float(100*(1-np.exp(np.log(np.array(r)).mean()))),
                                                   wins=int((np.array(r)<1).sum()),cells=len(r))
    summary['per_market']={m:{arm:float(100*(1-np.exp(np.mean([np.log(c[arm]['MQS']/c['base']['MQS']) for c in v['cells'].values()]))) ) for arm in ['cross','perm']} for m,v in report['markets'].items()}
    report['summary']=summary;report['frozen_after']={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in frozen}
    assert report['frozen_after']==frozen
    report['elapsed_seconds']=time.perf_counter()-started
    (P/(a.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 跨市场信息对条件参考本身的作用（无训练）','',f'{report["elapsed_seconds"]:.1f}秒。仅改变条件信息列，参考构造（成熟掩码、γ=.05均匀库、规则零原子、5种子、叶子大小）全部不变。',
           '','base必须逐位复现冻结forest_reference_only；cross为4个其它市场×{5日收益,20日波动}按as-of对齐到本市场日期；perm为同样列但按本市场日期打乱（控制“只是多了列”）。','']
    for m,v in report['markets'].items():
        lines.append(f'- {m}: 复现误差 Q={v["reproduction_error"]["Q"]:.1e}, CRPS={v["reproduction_error"]["CRPS"]:.1e}；cross MQS相对base {summary["per_market"][m]["cross"]:+.3f}%，perm {summary["per_market"][m]["perm"]:+.3f}%')
    lines+=['','| 对比 | MQS | 完整CRPS |','|---|---:|---:|']
    for arm,d in summary.items():
        if arm=='per_market':continue
        for k,v in d.items():
            lines.append(f'| {arm} vs {k} | {v["gain_percent"]:+.4f}% ({v["wins"]}/{v["cells"]}) |')
    lines+=['','若cross未显著超过perm，则跨市场列没有提供可用信息；若接近0，说明条件参考已饱和于本市场状态。历史测试复用，不证明显著性。']
    (P/(a.tag+'_report_zh.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
