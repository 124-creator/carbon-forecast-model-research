# -*- coding: utf-8 -*-
"""Read-only forest result reconstruction and full same-seed/cell gains."""
import sys
sys.dont_write_bytecode=True
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
import shape_split_round5 as S
P=Path(__file__).resolve().parent;R=S.R;T=S.T
torch.set_num_threads(1)
r=json.loads((P/'shape_split_round5.json').read_text(encoding='utf-8'));a=np.load(P/'shape_split_round5_predictions.npz');cells=pd.read_csv(P/'shape_split_round5_cells.csv');panel=pd.read_csv(T.B.PANEL,parse_dates=['date'])
assert r['forest_fits']==600 and set(r['markets'])==set(T.B.CUT) and len(r['anchor_checks'])==20
assert all(all(c[k] for k in ['raw_seed_and_mean_weights_bitwise_equal','gamma0_Q_CRPS_p_bitwise_equal_rule_QRF','gamma05_Q_CRPS_p_bitwise_equal_fixed_reference']) for c in r['anchor_checks'])
assert all(hashlib.sha256((P/name).read_bytes()).hexdigest()==sha for name,sha in r['frozen_sha256'].items())
keys=0;qerr=cerr=perr=alpha_error=0.;maxcdf=0.;bad_inner_kept=[];alpha_range={}
for market,mres in r['markets'].items():
    d=panel[panel.market==market].sort_values('date').reset_index(drop=True);D=R.S.build(d);te=D['te'];keys+=len(te);h=a[market+'_horizon'];branch=a[market+'_rule_qrf_p']
    assert np.array_equal(a[market+'_origin'],D['t'][te]) and np.array_equal(a[market+'_target'],D['y'][te]) and np.array_equal(a[market+'_scale'],D['scale'][te])
    for arm in S.ARMS:
        sel=mres['selection'][arm]
        if arm=='mse_y':assert sel['chosen_leaf'] is None and len(sel['candidates'])==1
        else:
            candidates=sel['candidates'];best=min(candidates,key=lambda item:(item['macro_log_MQS_anchor_ratio'],item['leaf']))
            assert sel['chosen_leaf']==best['leaf'] and sel['tied_best_leaves']==[v['leaf'] for v in candidates if v['macro_log_MQS_anchor_ratio']==best['macro_log_MQS_anchor_ratio']]
            if sel['macro_log_MQS_anchor_ratio']>0:bad_inner_kept.append([market,arm,sel['macro_log_MQS_anchor_ratio']])
        for j,hor in enumerate(T.HOR):
            key=f'{market}_{arm}_h{hor}';pos=a[key+'_test_positions'];z=torch.tensor(a[key+'_support_nz_z']);can=a[key+'_canonical_u'];weights=a[key+'_conditional_seed_mixture_weights'];assert np.allclose(weights.sum(1),1.,atol=1e-12,rtol=0)
            assert np.array_equal(pos,np.flatnonzero(h==hor)) and np.isfinite(can).all() and np.all(can>=0)
            alpha_range.setdefault(arm,[1.,0.]);alpha_range[arm]=[min(alpha_range[arm][0],float(weights.min())),max(alpha_range[arm][1],float(weights.max()))]
            chosen=np.array([0,len(pos)//2,len(pos)-1]);yy=torch.tensor(D['z'][te[pos[chosen]]],dtype=torch.float32);pp=torch.tensor(branch[pos[chosen]],dtype=torch.float32)
            def op(u):
                w=torch.tensor(u);w=w/w.sum(1,keepdim=True)
                return dict(w=w,p=pp,a=torch.ones(len(chosen)),b=torch.zeros(len(chosen)))
            uops=[];weighted=np.zeros_like(can,dtype=np.float64)
            for seed in S.SEEDS:
                u=a[key+f'_seed{seed}_u'];oo=op(u[chosen]);uops.append(oo)
                u32=torch.tensor(u);uw=(u32/u32.sum(1,keepdim=True)).double();uw=uw/uw.sum(1,keepdim=True);weighted+=weights[:,S.SEEDS.index(seed),None]*uw.numpy()
                with torch.no_grad():q,c,p,_,_=R.mixture(yy,z,[oo])
                qerr=max(qerr,float(np.max(np.abs(q.numpy()*D['scale'][te[pos[chosen]],None]-a[f'{market}_{arm}_seed{seed}_Q'][pos[chosen]]))));cerr=max(cerr,float(np.max(np.abs(c.numpy()*D['scale'][te[pos[chosen]]]-a[f'{market}_{arm}_seed{seed}_CRPS'][pos[chosen]]))));perr=max(perr,float(np.max(np.abs(p.numpy()-a[f'{market}_{arm}_seed{seed}_p'][pos[chosen]]))))
            canon=torch.tensor(can);cw=(canon/canon.sum(1,keepdim=True)).double();cw=cw/cw.sum(1,keepdim=True);alpha_error=max(alpha_error,float(np.abs(cw.numpy()-weighted).sum(1).max()))
            for label,ops in [('ensemble',[op(can[chosen])]),('equal_conditional_diagnostic',uops)]:
                with torch.no_grad():q,c,p,_,_=R.mixture(yy,z,ops)
                qerr=max(qerr,float(np.max(np.abs(q.numpy()*D['scale'][te[pos[chosen]],None]-a[f'{market}_{arm}_{label}_Q'][pos[chosen]]))));cerr=max(cerr,float(np.max(np.abs(c.numpy()*D['scale'][te[pos[chosen]]]-a[f'{market}_{arm}_{label}_CRPS'][pos[chosen]]))));perr=max(perr,float(np.max(np.abs(p.numpy()-a[f'{market}_{arm}_{label}_p'][pos[chosen]]))))
            ff=mres['final'][arm][str(hor)];maxcdf=max(maxcdf,ff['canonical']['CDF_CRPS_error'],max(v['CDF_CRPS_error'] for v in ff['seeds'].values()))
    chosen=mres['selected_composite_arm'];assert chosen==min(S.ARMS,key=lambda arm:mres['selection'][arm]['macro_log_MQS_anchor_ratio'])
    for label in ['ensemble',*['seed'+str(s) for s in S.SEEDS]]:
        for suffix in ['Q','CRPS','p']:assert np.array_equal(a[f'{market}_selected_composite_{label}_{suffix}'],a[f'{market}_{chosen}_{label}_{suffix}'])
assert qerr==0 and cerr<1e-12 and perr<1e-12 and alpha_error<3e-6
# Each cell/seed gain explicitly states whether the anchor is same-seed or canonical.
lookup={(row.market,int(row.horizon),row.model,str(row.seed)):row for row in cells.itertuples()};gainrows=[]
for row in cells.itertuples():
    if row.model not in S.ARMS+['selected_composite']:continue
    for base in ['mse_y','rule_qrf','qrf_original','forest_reference_only','honest_no_readout','rbf_additive','lstm_qr','ess_selected']:
        bseed=str(row.seed) if base=='mse_y' else 'fixed';other=lookup.get((row.market,int(row.horizon),base,bseed))
        if other is None:continue
        entry=dict(market=row.market,horizon=int(row.horizon),model=row.model,seed=str(row.seed),baseline=base,baseline_seed=bseed,role='reference_only' if row.model=='selected_composite' else 'predeclared_arm')
        for metric in ['MQS','CRPS','WIS','MAE']:
            v,w=getattr(row,metric),getattr(other,metric);entry[metric+'_gain_percent']=None if pd.isna(v) or pd.isna(w) else 100*(1-v/w)
        gainrows.append(entry)
g=pd.DataFrame(gainrows);g.to_csv(P/'shape_split_round5_gains.csv',index=False)
matched={};market_gain={}
for arm in S.ARMS[1:]:
    matched[arm]={};market_gain[arm]={}
    for seed in map(str,S.SEEDS):
        sub=g[(g.model==arm)&(g.seed==seed)&(g.baseline=='mse_y')]
        matched[arm][seed]={metric:float(100*(1-np.exp(np.log(1-sub[metric+'_gain_percent']/100).mean()))) for metric in ['MQS','CRPS']}
    for market in T.B.CUT:
        sub=g[(g.model==arm)&(g.seed=='canonical')&(g.baseline=='mse_y')&(g.market==market)]
        market_gain[arm][market]={metric:float(100*(1-np.exp(np.log(1-sub[metric+'_gain_percent']/100).mean()))) for metric in ['MQS','CRPS']}
out=dict(forest_fits=600,neural_training_calls=0,common_origin_horizon_keys=keys,frozen_file_count=len(r['frozen_sha256']),all_old_SHA_unchanged=True,all20_anchor_Q_CRPS_p_and_raw_weights_bitwise_match=True,leaf_ties_recorded_smaller_chosen=True,all_inner_worse_arm_market_pairs_still_final=bad_inner_kept,canonical_rawmix_then_condition_order_preserved=True,equal_conditioned_mix_diagnostic_separate=True,composite_reference_only=True,stored_u_replay_Q_max_error=qerr,stored_u_replay_CRPS_max_error=cerr,stored_u_replay_p_max_error=perr,independent_CDF_CRPS_max_error=maxcdf,canonical_vs_rawNZmass_weighted_seedF_max_L1_float_error=alpha_error,conditional_seed_mixture_weight_ranges=alpha_range,same_seed_arm_gains_vs_A=matched,canonical_market_gains_vs_A=market_gain,cell_seed_gain_rows=len(g))
(P/'shape_split_round5_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 分裂响应实验：锚、权重层级和单seed核对','',f'600森林、0 NN训练、{keys}共同原点×期限键，全部旧SHA一致。20个A锚的原始seed/mean权重、γ0 ruleQRF Q/CRPS/p、γ.05 fixedref Q/CRPS/p均逐位一致。','',f'NPZ u重建每模型每h固定3query，Q误差{qerr:.3g}、CRPS误差{cerr:.3g}、p误差{perr:.3g}；独立CDF最大误差{maxcdf:.3g}。canonical与按rawNZ质量加权的seedF在精确算术等价；float32累积/归一化后的最大L1差{alpha_error:.3g}。','',
'内选正log比的臂仍完整final：'+str(bad_inner_kept),'', '逐market/h/seed的显式gain已另存shape_split_round5_gains.csv，A比较为同seed；强基线比较为其冻结canonical。composite仅reference-only，未进原主表。','', '| arm | seed | 相对同seed A的MQS改善 | CRPS改善 |','|---|---:|---:|---:|']
for arm,seeds in matched.items():
    for seed,v in seeds.items():lines.append(f"| {arm} | {seed} | {v['MQS']:+.4f}% | {v['CRPS']:+.4f}% |")
lines+=['','| arm | market | canonical相对A的MQS改善 | CRPS改善 |','|---|---|---:|---:|']
for arm,markets in market_gain.items():
    for market,v in markets.items():lines.append(f"| {arm} | {market} | {v['MQS']:+.4f}% | {v['CRPS']:+.4f}% |")
(P/'shape_split_round5_verification_report_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
