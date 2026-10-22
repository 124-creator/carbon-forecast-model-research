# -*- coding: utf-8 -*-
"""Validation-only global mark mean/scale calibration; no model fitting."""
from __future__ import annotations
import os, sys
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
sys.dont_write_bytecode=True
import hashlib, itertools, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import seq_transport_round4 as R

P=Path(__file__).resolve().parent
TAG='reference_moment_calibration_round5'
MU=[0.,.25,.5,.75,1.,1.25]
SCALE=[.5,.75,1.,1.25,1.5,2.]
GRID=list(itertools.product(MU,SCALE))
MODEL='moment_calibrated_reference'
BASES=['forest_reference_only','rule_qrf','forest_transport_normalized','forest_transport_rawselected','lstm_qr']


def prepare(cache,phase,part):
    D={k:v.numpy() for k,v in cache['rows'].items()}
    rows=D[part];hi=D['hi'][rows];sid=cache['row_sid'][torch.tensor(rows)]
    prepared=[]
    for j,b in enumerate(cache['phases'][phase]['banks']):
        pos=np.flatnonzero(hi==j);ii=rows[pos];sids=sid[torch.tensor(pos)]
        u=b['u'][sids]
        assert torch.all(b['count'][sids]>=32)
        # Match the frozen reference operator's float32 normalization first;
        # mixture normalizes these weights in float64 before defining F.
        w=u/u.sum(1,keepdim=True)
        wd=w.double();wd=wd/wd.sum(1,keepdim=True);z=b['z']
        m0=(wd*z.double()).sum(1);v0=(wd*(z.double()-m0[:,None])**2).sum(1)
        local=cache['rows']['local'][torch.tensor(ii)]
        prob=.8*local+.2*torch.full_like(local,b['p'])
        prepared.append(dict(pos=pos,z=z,w=w,wd=wd,p=prob,m0=m0,v0=v0,
            yz=cache['rows']['z'][torch.tensor(ii)].float(),sc=D['scale'][ii],
            cached_m0_max_difference=float((m0-b['m0'][sids].double()).abs().max())))
    return D,rows,prepared


def evaluate(D,rows,prepared,mu,scale,check=False):
    q=np.empty((len(rows),15));c=np.empty(len(rows));pr=np.empty(len(rows));branch=np.empty(len(rows));bb=np.empty(len(rows));mm=np.empty(len(rows))
    diag=dict(mean_error=0.,variance_error=0.,CDF_CRPS_error=0.,collision_count=0,collision_mass_max=0.,p_effective_identity_error=0.)
    for item in prepared:
        pos=item['pos'];z=item['z'];m0=item['m0'];a=torch.full_like(m0,scale);b=(mu-scale)*m0
        o=dict(w=item['w'],p=item['p'],a=a,b=b)
        qq,cc,pp,values,weights=R.mixture(item['yz'],z,[o])
        q[pos]=qq.numpy()*item['sc'][:,None];c[pos]=cc.numpy()*item['sc'];pr[pos]=pp.numpy()
        branch[pos]=item['p'].numpy();bb[pos]=b.numpy();mm[pos]=m0.numpy()
        mark=a[:,None]*z.double()+b[:,None];mw=(item['wd']*mark).sum(1)
        vw=(item['wd']*(mark-mw[:,None])**2).sum(1)
        diag['mean_error']=max(diag['mean_error'],float((mw-mu*m0).abs().max()))
        diag['variance_error']=max(diag['variance_error'],float((vw-scale**2*item['v0']).abs().max()))
        collision=(mark==0)&(item['wd']>0);collision_mass=item['p'].double()*(item['wd']*collision).sum(1)
        diag['collision_count']+=int(collision.sum());diag['collision_mass_max']=max(diag['collision_mass_max'],float(collision_mass.max()))
        diag['p_effective_identity_error']=max(diag['p_effective_identity_error'],float((pp-(item['p'].double()-collision_mass)).abs().max()))
        if check:
            n=min(3,len(pos));ind=R.cdf_integral(item['yz'][:n].double().numpy(),values[:n].numpy(),weights[:n].numpy())
            diag['CDF_CRPS_error']=max(diag['CDF_CRPS_error'],float(np.max(np.abs(cc[:n].numpy()-ind))))
    assert np.isfinite(q).all() and np.isfinite(c).all() and np.all(np.diff(q,axis=1)>=-1e-12)
    assert diag['mean_error']<1e-10 and diag['variance_error']<1e-9 and diag['p_effective_identity_error']<1e-12
    assert not check or diag['CDF_CRPS_error']<1e-10
    return dict(Q=q,CRPS=c,p=pr,branch_p=branch,a=np.full(len(rows),scale),b=bb,m0=mm,diagnostic=diag)


def horizon_scores(D,rows,result):
    return [float(R.T.B.score(D['y'][rows[D['hi'][rows]==j]],result['Q'][D['hi'][rows]==j]).mean()) for j in range(4)]


def summary(markets):
    cells=[cell['scores'] for m in markets.values() for cell in m['by_horizon'].values()]
    out={}
    for base in BASES:
        out[MODEL+'_vs_'+base]={}
        for metric in ['MQS','CRPS','WIS','MAE','Brier']:
            valid=[c for c in cells if c[base][metric] is not None]
            if not valid:continue
            ratios=np.array([c[MODEL][metric]/c[base][metric] for c in valid])
            out[MODEL+'_vs_'+base][metric]=dict(geometric_gain_percent=float(100*(1-np.exp(np.log(ratios).mean()))),wins=int((ratios<1).sum()),cells=len(ratios))
    return out


def save(report,arrays,cells,valrows):
    report['summary']=summary(report['markets'])
    (P/f'{TAG}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(P/f'{TAG}_predictions.npz',**arrays)
    pd.DataFrame(cells).to_csv(P/f'{TAG}_cells.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(valrows).to_csv(P/f'{TAG}_validation_grid.csv',index=False,encoding='utf-8-sig')


def main():
    started=time.perf_counter();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    names=['seq_transport_round4.py','empirical_tilt_round3.py','neural_restart.py','forest_prior_round5.py',
        'forest_transport_round5.py','forest_transport_round5.json','forest_transport_round5_predictions.npz',
        'forest_transport_rawselect_round5.py','forest_transport_rawselect_round5.json','forest_transport_rawselect_round5_predictions.npz']
    names += [f'forest_prior_round5_{m}.pt' for m in R.T.B.CUT]
    before={name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in names}
    manifest=dict(method='known global conditional mark mean/scale calibration control; not a new model',
        lambda_mu=MU,lambda_s=SCALE,grid=[list(x) for x in GRID],n_candidates=36,
        mapping='M=lambda_s*z+(lambda_mu-lambda_s)*m0; conditional mean=lambda_mu*m0, sd=lambda_s*s0',
        weights='same conditional mark u, reproduce frozen float32 reference normalization then true float64 F',
        moments='m0/s0 computed from the normalized F weights in float64; cached float32 moment discrepancy recorded',
        probability='same phase p_rule; p_eff computed from F with exact mark zero collisions',
        selection='one pair per market, shared across h1/5/10/20, minimizing mean_h log(raw_MQS_inner_Dva_h/identity_anchor_h)',
        tie='exact equal criterion: smaller squared distance to (1,1), then grid order',
        anchor=[1.,1.],final='apply only selected pair plus fixed identity anchor to final reference; never score test grid',
        training=dict(forest_fits=0,NN_fits=0),frozen_sha256_before=before)
    # Freeze the full grid and rule before any candidate is scored.
    (P/f'{TAG}_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    old=np.load(P/'forest_transport_round5_predictions.npz')
    raw=np.load(P/'forest_transport_rawselect_round5_predictions.npz')
    report=dict(manifest=manifest,markets={});arrays={};cells=[];valrows=[]
    with threadpool_limits(limits=2):
        for market in R.T.B.CUT:
            cache=torch.load(P/f'forest_prior_round5_{market}.pt',map_location='cpu',weights_only=False)
            D,va,inner=prepare(cache,'inner','va')
            anchor=evaluate(D,va,inner,1.,1.,True);anchor_h=horizon_scores(D,va,anchor)
            best_key=None;best_result=None;best_pair=None;records=[]
            for number,(mu,scale) in enumerate(GRID):
                result=anchor if (mu,scale)==(1.,1.) else evaluate(D,va,inner,mu,scale)
                mqs=horizon_scores(D,va,result)
                criterion=float(np.log(np.array(mqs)/anchor_h).mean())
                row=dict(market=market,grid_index=number,lambda_mu=mu,lambda_s=scale,raw_macro_log_MQS_ratio=criterion,
                    validation_geometric_gain_percent=float(100*(1-np.exp(criterion))))
                for h,m,a in zip(R.T.HOR,mqs,anchor_h):row[f'MQS_h{int(h)}']=m;row[f'anchor_MQS_h{int(h)}']=a
                row['validation_diagnostic']=result['diagnostic'];records.append(row)
                tie=(criterion,(mu-1)**2+(scale-1)**2,number)
                if best_key is None or tie<best_key:best_key=tie;best_result=result;best_pair=(mu,scale)
                valrows.append({k:v for k,v in row.items() if k!='validation_diagnostic'})
            assert len(records)==36 and best_key[0]<=1e-14
            for row in records:row['selected']=(row['lambda_mu'],row['lambda_s'])==best_pair
            for row in valrows:
                if row['market']==market:row['selected']=(row['lambda_mu'],row['lambda_s'])==best_pair
            print('SELECTED',market,best_pair,'inner_logMQS',best_key[0],flush=True)
            _,te,final=prepare(cache,'final','te')
            fixed=evaluate(D,te,final,1.,1.,True)
            selected=fixed if best_pair==(1.,1.) else evaluate(D,te,final,*best_pair,check=True)
            prefix=market+'_';metadata=[('origin',D['t'][te]),('horizon',R.T.HOR[D['hi'][te]]),('target',D['y'][te]),('scale',D['scale'][te])]
            for key,value in metadata:
                assert np.array_equal(value,old[prefix+key]) and np.array_equal(value,raw[prefix+key]);arrays[prefix+key]=value
            arrays[prefix+'origin_date']=old[prefix+'origin_date']
            pred={MODEL:selected,'forest_reference_only':fixed}
            for model,key in [('rule_qrf','rule_qrf'),('forest_transport_normalized','normalized'),('forest_transport_rawselected','rawselect'),('lstm_qr','lstm_qr')]:
                pred[model]={suffix:raw[prefix+key+'_'+suffix] if prefix+key+'_'+suffix in raw else None for suffix in ['Q','CRPS','p']}
            assert np.array_equal(pred['forest_transport_normalized']['Q'],old[prefix+'forest_transport_Q'])
            assert np.array_equal(raw[prefix+'reference_Q'],old[prefix+'forest_reference_only_Q'])
            refcheck={suffix:float(np.max(np.abs(fixed[suffix]-old[prefix+'forest_reference_only_'+suffix]))) for suffix in ['Q','CRPS','p']}
            assert refcheck['Q']==0 and refcheck['CRPS']<1e-12 and refcheck['p']<1e-12
            assert np.array_equal(fixed['branch_p'],old[prefix+'forest_reference_only_branch_p'])
            by_h={}
            for j,h in enumerate(R.T.HOR):
                pos=np.flatnonzero(D['hi'][te]==j);scores={}
                for model,values in pred.items():
                    score=R.metrics(D['y'][te[pos]],values['Q'][pos],None if values['CRPS'] is None else values['CRPS'][pos],None if values['p'] is None else values['p'][pos])
                    scores[model]=score;cells.append(dict(market=market,horizon=int(h),model=model,n=len(pos),**score))
                by_h[str(int(h))]=dict(n=len(pos),scores=scores)
            for model,values in pred.items():
                for suffix in ['Q','CRPS','p','branch_p','a','b','m0']:
                    if suffix in values and values[suffix] is not None:arrays[prefix+model+'_'+suffix]=values[suffix]
            for suffix,value in [('origin',D['t'][va]),('horizon',R.T.HOR[D['hi'][va]]),('target',D['y'][va]),('scale',D['scale'][va])]:arrays[prefix+'validation_'+suffix]=value
            for model,value in [('selected',best_result),('anchor',anchor)]:
                for suffix in ['Q','CRPS','p']:arrays[prefix+'validation_'+model+'_'+suffix]=value[suffix]
            arrays[prefix+'selected_pair']=np.array(best_pair)
            report['markets'][market]=dict(selection=dict(lambda_mu=best_pair[0],lambda_s=best_pair[1],criterion=best_key[0],anchor_raw_MQS=anchor_h,grid_scores=records),
                fixed_reference_reproduction_max_error=refcheck,selected_test_diagnostic=selected['diagnostic'],by_horizon=by_h,
                normalized_m0_vs_cache_float32_max_error=max(item['cached_m0_max_difference'] for item in inner+final),
                final_selected_pairs_applied=1,final_identity_anchor_evaluated=True,
                forest_fits=0,NN_fits=0,baseline_CRPS='full distributions only; LSTMQR remains None')
            report['elapsed_seconds']=time.perf_counter()-started;save(report,arrays,cells,valrows)
            print('MARKET_DONE',market,'elapsed',round(report['elapsed_seconds'],1),flush=True)
            del cache,inner,final
    after={name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in names}
    assert before==after
    report.update(frozen_sha256_after=after,elapsed_seconds=time.perf_counter()-started)
    save(report,arrays,cells,valrows)
    lines=['# 第五轮：全局reference mean/scale校准控制','',
        '这是已知的低成本校准控制，不是新模型；没有训练森林或NN，不改主NN协议。每市场只在inner D.va从预冻结36组选择一个共享全部h的参数对，final只应用该pair与identity对照。',
        '选择准则是四个h的raw MQS相对于各自固定reference(1,1)的mean log ratio，未按test选择；全部180个val候选分数保留。',
        'conditional mark权重u和p_rule保持固定，仅M=lambda_s*z+(lambda_mu-lambda_s)*m0。真正排序NZ mark+零atom后统一计算Q/完整CRPS/actual p；目标mean=0不强制median=0。', '',
        '| 市场 | lambda_mu | lambda_s | inner几何MQS改善 |', '|---|---:|---:|---:|']
    for market,v in report['markets'].items():
        s=v['selection'];lines.append(f"| {market} | {s['lambda_mu']:g} | {s['lambda_s']:g} | {100*(1-np.exp(s['criterion'])):+.4f}% |")
    lines+=['','| 校准reference 对参照 | 测试MQS改善 | 测试完整CRPS改善 |','|---|---:|---:|']
    for key,value in report['summary'].items():
        q=value['MQS'];c=value.get('CRPS');cc='15q不可识别' if c is None else f"{c['geometric_gain_percent']:+.4f}% ({c['wins']}/20)"
        lines.append(f"| {key.split('_vs_')[1]} | {q['geometric_gain_percent']:+.4f}% ({q['wins']}/20) | {cc} |")
    lines+=['','正值表示校准reference更好；20个market×h格等权几何汇总。LSTMQR没有可用完整CRPS，不由15个分位点构造替代CRPS。',
        '固定(1,1) reference的Q逐值复现冻结forest_reference_only，CRPS/p误差及独立CDF积分核对保存在JSON。',
        'm0用归一后的同一F权重以float64计算，避免缓存float32均值舍入影响目标矩；固定reference权重首先复用原算子的float32归一以严格匹配anchor。',
        'JSON保留36val scores/market、目标矩误差、实际零碰撞、全部20格和源SHA；NPZ保留selected/fixed以及只读基线Q/CRPS/p，另存selected/anchor validation预测。',
        f"运行{report['elapsed_seconds']:.3f}秒；forest fits=0，NN fits=0，所有读取旧源/基线/cache SHA一致。",'',
        '该控制用于判断prior-readout神经模型是否超出简单全局mean/scale校准；不改变已有NN训练或选择协议。']
    (P/f'{TAG}_report_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(report['summary'],ensure_ascii=False,indent=2),flush=True)
    print('ALL_CALIBRATION_DONE',report['elapsed_seconds'],flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
