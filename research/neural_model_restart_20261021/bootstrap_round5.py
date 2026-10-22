# -*- coding: utf-8 -*-
"""Paired moving-origin block bootstrap of frozen true-ensemble predictions.

No fitting, seed selection, distribution reconstruction, or old-file writes.
"""
from __future__ import annotations
import os,sys
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
sys.dont_write_bytecode=True
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

P=Path(__file__).resolve().parent
TAG='bootstrap_round5'
MARKETS=['CEA','GUANGZHOU','SHANGHAI','HUBEI','EU']
HORIZONS=[1,5,10,20]
TAUS=np.array([.025,.05,.10,.15,.20,.30,.40,.50,.60,.70,.80,.85,.90,.95,.975],dtype=np.float32)
SEEDS=[11,22,33,44,55]
B=2000
BLOCKS=[5,10]
RNG_SEED=20261021
SOURCES={'prior':'prior_readout_transport_round5','forest':'forest_transport_round5',
         'additive':'additive_transport_round5','matched':'prefix_transport_controls_round5'}
SPECS=[('rule_qrf','prior','rule_qrf'),('qrf_original','prior','qrf_original'),
       ('honest_no_readout','prior','honest_no_readout'),('forest_reference_only','prior','forest_reference_only'),
       ('rbf_additive','additive','rbf_prior'),('forest_transport','forest','forest_transport')]
MATCHED=['matched_insample_no_readout','matched_honest_no_readout_reweight_only']


def aligned(archive,market,key,canonical):
    origin,horizon,y,scale=canonical
    so=archive[market+'_origin'];sh=archive[market+'_horizon']
    if np.array_equal(so,origin) and np.array_equal(sh,horizon):order=np.arange(len(origin))
    else:
        lookup={(int(o),int(h)):j for j,(o,h) in enumerate(zip(so,sh))}
        assert len(lookup)==len(origin)==len(so)
        order=np.array([lookup[(int(o),int(h))] for o,h in zip(origin,horizon)])
    assert np.array_equal(archive[market+'_target'][order],y)
    assert np.array_equal(archive[market+'_scale'][order],scale)
    q=archive[market+'_'+key+'_Q'][order];c=archive[market+'_'+key+'_CRPS'][order]
    assert q.shape==(len(y),15) and c.shape==(len(y),)
    assert np.isfinite(q).all() and np.isfinite(c).all()
    d=y[:,None]-q
    mqs=(2*np.maximum(TAUS*d,(TAUS-1)*d)).mean(1)
    return np.stack([mqs,c],axis=-1)


def moving_indices(rng,n,length):
    k=(n+length-1)//length
    start=rng.integers(0,n-length+1,size=(B,k),dtype=np.int32)
    indices=(start[:,:,None]+np.arange(length,dtype=np.int32)).reshape(B,-1)[:,:n]
    assert indices.shape==(B,n) and indices.min()>=0 and indices.max()<n
    counts=np.zeros((B,n),dtype=np.int32)
    np.add.at(counts,(np.repeat(np.arange(B),n),indices.ravel()),1)
    assert np.all(counts.sum(1)==n)
    return indices,counts


def describe(draw,point,**keys):
    lo,hi=np.quantile(draw,[.025,.975],method='linear')
    return dict(**keys,point_gain_percent=float(point),lower95_percent=float(lo),upper95_percent=float(hi),
        P_boot_gain_gt_zero=float(np.mean(draw>0)),bootstrap_median_gain_percent=float(np.median(draw)),
        bootstrap_mean_gain_percent=float(np.mean(draw)),B=len(draw))


def main():
    start=time.perf_counter()
    metas={name:json.loads((P/(prefix+'.json')).read_text(encoding='utf-8')) for name,prefix in SOURCES.items()}
    archives={name:np.load(P/(prefix+'_predictions.npz')) for name,prefix in SOURCES.items()}
    specs=list(SPECS);skipped=[]
    matched_seeds=metas['matched']['config']['seeds']
    for name in MATCHED:
        ok=matched_seeds==SEEDS and all(len(metas['matched']['markets'][m]['fits'][name])==5 and
            all(f'{m}_{name}_seed{s}_Q' in archives['matched'] for s in SEEDS) and
            f'{m}_{name}_Q' in archives['matched'] for m in MARKETS)
        if ok:specs.append((name,'matched',name))
        else:skipped.append(dict(model=name,reason='no complete verified five-seed ensemble; single-seed outputs skipped'))
    for name,source,key in specs:
        if name in ['rule_qrf','qrf_original','forest_reference_only']:continue
        assert metas[source]['config']['seeds']==SEEDS
        assert all(len(metas[source]['markets'][m]['fits'][key])==5 for m in MARKETS)
    names=[v[0] for v in specs];candidates=[x for x in names if x not in ['rule_qrf','qrf_original']]
    comparisons=[(model,'rule_qrf') for model in candidates]+[('rule_qrf','qrf_original'),('honest_no_readout','forest_reference_only'),('rbf_additive','forest_reference_only')]
    for control in MATCHED:
        if control in names:comparisons.append(('honest_no_readout',control))
    files=[prefix+suffix for prefix in SOURCES.values() for suffix in ['.py','.json','_predictions.npz']]
    frozen={name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in files}
    manifest=dict(status='conditional uncertainty diagnostics for exploratory frozen ensembles; no significance declaration or candidate switch',
        B=B,block_lengths=BLOCKS,rng_seed=RNG_SEED,bit_generator='PCG64',rng_order='one Generator; block length 5 then10, markets CEA/GUANGZHOU/SHANGHAI/HUBEI/EU',
        resampling='noncircular overlapping moving-origin blocks; ceil(N/L) uniformly sampled starts in [0,N-L], concatenate and truncate to N',
        block_units='observed unique sorted test origins, not calendar days',
        pairing='one sampled-origin index/count matrix per market/length shared by every model and all horizons',
        missing_h='missing end-of-series h rows remain missing; weighted sums divided by resampled valid count in each market/h',
        estimand='per-cell raw mean score then equal-weight market x h geometric ratio; no pooling sample counts across cells',
        market_sampling='markets never resampled or pooled; independent origin-block draws within each market, fixed five markets',
        metrics=['MQS:2*mean15pinball per origin on stored raw Q and canonical y','CRPS:stored raw per-origin score'],
        intervals='95% percentile, np.quantile(method=linear); P_boot(gain>0) is a resample fraction, not a p-value',
        models=specs,comparisons=[list(x) for x in comparisons],skipped=skipped,source_sha256_before=frozen,
        no_training=True,no_seed_selection=True,no_full_F_reconstruction=True)
    (P/f'{TAG}_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    point=np.empty((5,4,len(names),2));data={};checks={};scores_out={};sample_arrays={};draw_arrays={};point_rows=[]
    for mi,market in enumerate(MARKETS):
        prior=archives['prior'];canonical=tuple(prior[market+'_'+field] for field in ['origin','horizon','target','scale'])
        origin,hor,y,scale=canonical
        assert np.all(scale>0) and len(set(zip(origin.tolist(),hor.tolist())))==len(y)
        times,inverse=np.unique(origin,return_inverse=True);valid=np.zeros((len(times),4),dtype=np.float64)
        cube=np.zeros((len(times),4,len(names),2),dtype=np.float64)
        for j,h in enumerate(HORIZONS):valid[inverse[hor==h],j]=1
        for model_index,(name,source,key) in enumerate(specs):
            score=aligned(archives[source],market,key,canonical)
            for j,h in enumerate(HORIZONS):
                mask=hor==h;cube[inverse[mask],j,model_index,:]=score[mask]
                point[mi,j,model_index]=score[mask].mean(0)
                for metric_index,metric in enumerate(['MQS','CRPS']):
                    expected=metas[source]['markets'][market]['by_horizon'][str(h)]
                    expected=expected['scores'][key][metric] if 'scores' in expected else expected[key][metric]
                    assert abs(point[mi,j,model_index,metric_index]-float(expected))<1e-12
                point_rows.append(dict(market=market,horizon=h,model=name,n=int(mask.sum()),MQS=point[mi,j,model_index,0],CRPS=point[mi,j,model_index,1]))
        assert np.all(point[mi]>0)
        n0=valid.sum(0).astype(int)
        data[market]=(cube,valid,times)
        checks[market]=dict(n_origins=len(times),n_by_h={str(h):int(n) for h,n in zip(HORIZONS,n0)},n_queries=len(y),key_alignment_exact=True,resampling={})
        scores_out[market+'_origins']=times;scores_out[market+'_valid_h']=valid.astype(np.uint8);scores_out[market+'_scores']=cube
    scopes=[('all20',[(m,h) for m in range(5) for h in range(4)]),('SH_only',[(2,h) for h in range(4)]),('ex_SH',[(m,h) for m in [0,1,3,4] for h in range(4)])]
    scopes += [('market_'+m,[(mi,h) for h in range(4)]) for mi,m in enumerate(MARKETS)]
    scopes += [('h'+str(h),[(m,j) for m in range(5)]) for j,h in enumerate(HORIZONS)]
    scopes += [('cell_'+m+'_h'+str(h),[(mi,j)]) for mi,m in enumerate(MARKETS) for j,h in enumerate(HORIZONS)]
    rng=np.random.Generator(np.random.PCG64(RNG_SEED));records=[]
    with threadpool_limits(limits=2):
        for length in BLOCKS:
            boot=np.empty((B,5,4,len(names),2))
            for mi,market in enumerate(MARKETS):
                cube,valid,times=data[market];indices,count=moving_indices(rng,len(times),length)
                denominator=count@valid
                assert np.all(denominator>0)
                result=(count.astype(np.float64)@cube.reshape(len(times),-1)).reshape(B,4,len(names),2)
                result/=denominator[:,:,None,None]
                assert np.isfinite(result).all() and np.all(result>0)
                for b in [0,B//2,B-1]:
                    direct=cube[indices[b]].sum(0)/valid[indices[b]].sum(0)[:,None,None]
                    assert np.allclose(result[b],direct,atol=1e-12,rtol=1e-12)
                boot[:,mi]=result
                digest=hashlib.sha256(indices.tobytes()).hexdigest()
                checks[market]['resampling'][str(length)]=dict(B=B,index_shape=list(indices.shape),index_sha256=digest,
                    same_indices_for_all_models_h=True,counts_sum_equals_N=True,direct_weighted_score_check=True,
                    resampled_n_min_by_h=denominator.min(0).astype(int).tolist(),resampled_n_max_by_h=denominator.max(0).astype(int).tolist())
                sample_arrays[f'L{length}_{market}_origin_indices']=indices
                sample_arrays[market+'_canonical_origins']=times
            for candidate,reference in comparisons:
                ci,ri=names.index(candidate),names.index(reference)
                logdraw=np.log(boot[:,:,:,ci,:]/boot[:,:,:,ri,:])
                logpoint=np.log(point[:,:,ci,:]/point[:,:,ri,:])
                for scope,pairs in scopes:
                    mm,hh=zip(*pairs);mm=np.array(mm);hh=np.array(hh)
                    gain=100*(1-np.exp(logdraw[:,mm,hh,:].mean(1)))
                    pgain=100*(1-np.exp(logpoint[mm,hh,:].mean(0)))
                    for j,metric in enumerate(['MQS','CRPS']):
                        records.append(describe(gain[:,j],pgain[j],block_length=length,candidate=candidate,reference=reference,metric=metric,scope=scope,cells=len(pairs)))
                    if scope in ['all20','SH_only','ex_SH'] or scope.startswith('market_'):
                        draw_arrays[f'L{length}_{candidate}_vs_{reference}_{scope}']=gain
            print('BLOCK_DONE',length,'elapsed',round(time.perf_counter()-start,2),flush=True)
    after={name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in files};assert frozen==after
    frame=pd.DataFrame(records);main_frame=frame[frame.scope.isin(['all20','SH_only','ex_SH'])]
    markets_frame=frame[frame.scope.str.startswith('market_')]
    frame.to_csv(P/f'{TAG}_intervals.csv',index=False,encoding='utf-8-sig')
    main_frame.to_csv(P/f'{TAG}_summary.csv',index=False,encoding='utf-8-sig')
    markets_frame.to_csv(P/f'{TAG}_by_market.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(point_rows).to_csv(P/f'{TAG}_point_scores.csv',index=False,encoding='utf-8-sig')
    np.savez_compressed(P/f'{TAG}_origin_indices.npz',**sample_arrays)
    with np.load(P/f'{TAG}_origin_indices.npz') as saved_indices:
        for market in MARKETS:
            for length in BLOCKS:
                value=saved_indices[f'L{length}_{market}_origin_indices']
                assert value.dtype==np.int32 and value.shape[0]==B
                assert hashlib.sha256(value.tobytes()).hexdigest()==checks[market]['resampling'][str(length)]['index_sha256']
                checks[market]['resampling'][str(length)]['serialized_index_hash_readback_pass']=True
    np.savez_compressed(P/f'{TAG}_draws.npz',**draw_arrays)
    np.savez_compressed(P/f'{TAG}_origin_scores.npz',model_names=np.array(names),metric_names=np.array(['MQS','CRPS']),horizons=np.array(HORIZONS),**scores_out)
    report=dict(manifest=manifest,self_checks=checks,results=records,source_sha256_after=after,
        total_query_count=sum(v['n_queries'] for v in checks.values()),elapsed_seconds=time.perf_counter()-start)
    (P/f'{TAG}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 第五轮：冻结ensemble的时间块自助区间','',
        '仅重采样已存预测的逐origin损失，不训练、不挑seed、不改主候选。全部候选使用真实五seed分布ensemble；不平均seed quantile或seed CRPS代替ensemble。',
        '每市场按unique test origin时间顺序进行非循环moving-block bootstrap，重叠block长度5与10个origin，B=2000，PCG64种子20261021。所有模型及该市场全部h严格共用同一重采样索引。',
        '每格按该次抽到的有效origin count计算损失均值，末端缺失h目标保留缺失；主聚合是20格等权几何gain，不按市场样本数重加权、不重抽市场。','',
        '## 全20格：相对rule-QRF','',
        '| 候选 | L | MQS点估计 / 95%区间 / P_boot正向 | CRPS点估计 / 95%区间 / P_boot正向 |','|---|---:|---|---|']
    for candidate in candidates:
        for length in BLOCKS:
            part=main_frame[(main_frame.candidate==candidate)&(main_frame.reference=='rule_qrf')&(main_frame.scope=='all20')&(main_frame.block_length==length)]
            texts=[]
            for metric in ['MQS','CRPS']:
                r=part[part.metric==metric].iloc[0]
                texts.append(f"{r.point_gain_percent:+.4f}% / [{r.lower95_percent:+.4f}, {r.upper95_percent:+.4f}]% / {r.P_boot_gain_gt_zero:.3f}")
            lines.append(f"| {candidate} | {length} | {texts[0]} | {texts[1]} |")
    lines+=['','## 市场依赖：MQS相对rule-QRF','',
        '| 候选 | L | 上海点估计 / 95%区间 | 非上海点估计 / 95%区间 |','|---|---:|---|---|']
    for candidate in candidates:
        for length in BLOCKS:
            texts=[]
            for scope in ['SH_only','ex_SH']:
                r=main_frame[(main_frame.candidate==candidate)&(main_frame.reference=='rule_qrf')&(main_frame.scope==scope)&(main_frame.block_length==length)&(main_frame.metric=='MQS')].iloc[0]
                texts.append(f"{r.point_gain_percent:+.4f}% / [{r.lower95_percent:+.4f}, {r.upper95_percent:+.4f}]%")
            lines.append(f"| {candidate} | {length} | {texts[0]} | {texts[1]} |")
    lines+=['','## 匹配机制与对照尺度','',
        '| 对比 | L | MQS点估计 / 95%区间 / P_boot正向 | CRPS点估计 / 95%区间 / P_boot正向 |','|---|---:|---|---|']
    for candidate,reference in comparisons:
        if reference=='rule_qrf':continue
        for length in BLOCKS:
            texts=[]
            for metric in ['MQS','CRPS']:
                r=main_frame[(main_frame.candidate==candidate)&(main_frame.reference==reference)&(main_frame.scope=='all20')&(main_frame.block_length==length)&(main_frame.metric==metric)].iloc[0]
                texts.append(f"{r.point_gain_percent:+.4f}% / [{r.lower95_percent:+.4f}, {r.upper95_percent:+.4f}]% / {r.P_boot_gain_gt_zero:.3f}")
            lines.append(f"| {candidate} vs {reference} | {length} | {texts[0]} | {texts[1]} |")
    lines+=['','## 匹配差异的上海依赖','','候选均为honest_no_readout，按同一冻结对比分别看上海与非上海。','',
        '| 对照 | 指标 | L | 上海点估计 / 95%区间 | 非上海点估计 / 95%区间 |','|---|---|---:|---|---|']
    for reference in MATCHED:
        if reference not in names:continue
        for metric in ['MQS','CRPS']:
            for length in BLOCKS:
                texts=[]
                for scope in ['SH_only','ex_SH']:
                    r=main_frame[(main_frame.candidate=='honest_no_readout')&(main_frame.reference==reference)&(main_frame.metric==metric)&(main_frame.scope==scope)&(main_frame.block_length==length)].iloc[0]
                    texts.append(f"{r.point_gain_percent:+.4f}% / [{r.lower95_percent:+.4f}, {r.upper95_percent:+.4f}]%")
                lines.append(f"| {reference} | {metric} | {length} | {texts[0]} | {texts[1]} |")
    lines+=['','整体prefix正方向不能解读为普遍市场增量；剔除上海后差异接近0，affine匹配的非上海点估计转负。相对固定参考的两档区间亦跨0。','','## 自检与解释边界','',
        f"共{report['total_query_count']}个共同origin×h键。JSON保存每格n、2000次索引shape/hash、重采样有效n范围、直接索引与count加权均值的一致性。源文件SHA前后一致。运行{report['elapsed_seconds']:.3f}秒。",
        'origin_indices NPZ保存所有重采样索引；origin_scores保存逐origin原始分数/有效h mask；draws保留主聚合及逐市场bootstrap gain。CSV包含全部候选×逐市场以及每h/每格区间，JSON保留全部点估计、区间与B。',
        '这里的95%为预定块长度下的百分位区间。P_boot(gain>0)是自助正向比例，不是Bayesian posterior或统计检验p值；未做多重比较校正，不声明显著性通过。',
        'moving-block方法要求各市场测试期损失序列近似局部平稳、短程依赖可由所选块长描述；长度5/10是有限敏感性分析，长记忆、结构变点与跨市场同期依赖可能使区间偏窄。市场分别抽块隐含条件上市场间抽样独立，未模拟跨市场共同冲击。',
        '市场被固定而非重抽，区间只针对这五个市场内的时间抽样；不涵盖训练样本/seed/选轮/超参/结构选择的不确定性。历史测试被开发反复使用，bootstrap不能恢复新测试集或修正选择偏差。完整不确定性可能更大，结果只是条件于冻结预测的稳定性证据。',
        '保留主候选及次要控制原定位；小正点估计与少数cell或上海驱动的现象必须结合区间和剔除上海结果解读。']
    (P/f'{TAG}_report_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(main_frame[(main_frame.scope=='all20')&(main_frame.metric=='MQS')].to_string(index=False),flush=True)
    print('ALL_BOOTSTRAP_DONE',report['elapsed_seconds'],flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
