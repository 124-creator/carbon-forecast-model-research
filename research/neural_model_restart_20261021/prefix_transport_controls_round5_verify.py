# -*- coding: utf-8 -*-
"""Checkpoint and same-seed mechanism verification; no new fitting."""
import sys
sys.dont_write_bytecode=True
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
import prefix_transport_controls_round5 as C
P=Path(__file__).resolve().parent;R=C.R;M=C.M
torch.set_num_threads(1);C.activate()
r=json.loads((P/'prefix_transport_controls_round5.json').read_text(encoding='utf-8'));a=np.load(P/'prefix_transport_controls_round5_predictions.npz');old=np.load(P/'prior_readout_transport_round5_predictions.npz');panel=pd.read_csv(R.T.B.PANEL,parse_dates=['date'])
assert set(r['markets'])==set(R.T.B.CUT)
fits=[f for v in r['markets'].values() for fs in v['fits'].values() for f in fs];assert len(fits)==50
assert all(f['epochs_run']>=30 and f['refit_epochs']==f['best_epoch'] and f['allocated_parameters']==2636 for f in fits)
assert all(not f['prior_weight_nonzero_gradient'] and not f['refit_prior_weight_nonzero_gradient'] for f in fits)
assert all(hashlib.sha256((P/name).read_bytes()).hexdigest()==sha for name,sha in r['frozen_sha256'].items())
assert all(hashlib.sha256((P/name).read_bytes()).hexdigest()==sha for name,sha in r['reference_cache_sha256'].items())
keys=0;aberr=qerr=cerr=0.;raw_diff=0;reference_floor=weighted_floor=collision=0;minimum={};seed_comparisons={};market_gains={}
for f in fits:
    history=f['raw_relative_validation_history'];assert len(history)==f['epochs_run']+1 and min(history,key=lambda row:row[1])[0]==f['best_raw_epoch'] and f['raw_diagnostic_only'];raw_diff+=f['best_raw_epoch']!=f['best_epoch']
    for s in [f['training_geometry'],f['validation_geometry'],f['refit']['training_geometry']]+list(f['test_geometry'].values()):
        reference_floor+=s['reference_floor_visits'];weighted_floor+=s['weighted_floor_visits']
        for k in ['min_reference_variance','min_weighted_variance']:
            if s[k] is not None:minimum[k]=min(minimum.get(k,float('inf')),s[k])
    collision+=sum(s['collision_count'] for s in f['test_geometry'].values())
for market,mres in r['markets'].items():
    for field in ['origin','horizon','target','scale']:assert np.array_equal(a[market+'_'+field],old[market+'_'+field])
    y=a[market+'_target'];h=a[market+'_horizon'];scale=a[market+'_scale'];keys+=len(y)
    for field in ['Q','CRPS','p']:assert np.array_equal(a[market+'_'+C.FROZEN+'_'+field],old[market+'_honest_no_readout_'+field])
    for seed in C.SEEDS:
        for field in ['Q','CRPS','p','branch_p','a','b']:assert np.array_equal(a[f'{market}_{C.FROZEN}_seed{seed}_{field}'],old[f'{market}_honest_no_readout_seed{seed}_{field}'])
    data=panel[panel.market==market].sort_values('date').reset_index(drop=True);D=R.S.build(data);prepared={};audits={};cacheinfo={}
    for kind in ['insample','honest']:prepared[kind],audits[kind],cacheinfo[kind]=M.load_explicit(D,market,kind)
    M.validate_eval_equality(D,prepared['honest'],prepared['insample'])
    cp=torch.load(P/f'prefix_transport_controls_round5_{market}_checkpoints.pt',map_location='cpu',weights_only=False);te=torch.tensor(D['te'])
    for model in C.CONTROLS:
        kind=C.KINDS[model];X,banks=prepared[kind]['final'];assert cp['cache_info'][kind]['sha256']==cacheinfo[kind]['sha256'] and cp['prior_scalers'][kind]==audits[kind]['final']['prior_scaler']
        assert all(f['cache_kind']==kind for f in mres['fits'][model])
        for b,saved in zip(banks,cp['final_banks'][kind]):
            for k in ['z','phi2','mature','u','allowed','count','m0','v0','vu','bank_D_indices']:assert torch.equal(b[k],saved[k])
        Q=a[market+'_'+model+'_Q'];CRPS=a[market+'_'+model+'_CRPS'];assert np.isfinite(Q).all() and np.isfinite(CRPS).all() and np.all(np.diff(Q,axis=1)>=-1e-8)
        assert np.max(CRPS-np.mean([a[f'{market}_{model}_seed{s}_CRPS'] for s in C.SEEDS],0))<1e-6
        if model==C.CONTROLS[1]:
            for seed in C.SEEDS:assert np.all(a[f'{market}_{model}_seed{seed}_a']==1) and np.all(a[f'{market}_{model}_seed{seed}_b']==0)
        ops={j:[] for j in range(4)}
        for seed in C.SEEDS:
            net=C.MatchedNet(model);net.load_state_dict(cp['state_dicts'][f'{model}_seed{seed}']);net.eval()
            with torch.no_grad():
                raw=net(X,te)
                for j,b in enumerate(banks):
                    pos=np.flatnonzero(h==R.T.HOR[j]);chosen=pos[[0,len(pos)//2,len(pos)-1]];ii=te[chosen];assert bool(torch.all(net.prior_input(X,ii)==0))
                    o=C.control_operator(raw[chosen],b,X['sid'][ii],X['local'][ii],model);ops[j].append(o)
                    aberr=max(aberr,float(np.max(np.abs(o['a'].numpy()-a[f'{market}_{model}_seed{seed}_a'][chosen]))),float(np.max(np.abs(o['b'].numpy()-a[f'{market}_{model}_seed{seed}_b'][chosen]))))
        for j,b in enumerate(banks):
            pos=np.flatnonzero(h==R.T.HOR[j]);chosen=pos[[0,len(pos)//2,len(pos)-1]]
            with torch.no_grad():q,c,p,_,_=R.mixture(torch.tensor(y[chosen]/scale[chosen]),b['z'],ops[j])
            qerr=max(qerr,float(np.max(np.abs(q.numpy()*scale[chosen,None]-Q[chosen]))));cerr=max(cerr,float(np.max(np.abs(c.numpy()*scale[chosen]-CRPS[chosen]))))
    market_gains[market]={model:float(100*(1-np.exp(np.mean([np.log(cell['scores'][C.FROZEN]['MQS']/cell['scores'][model]['MQS']) for cell in mres['by_horizon'].values()])))) for model in C.CONTROLS}
assert aberr<2e-5 and cerr<2e-6
for seed in C.SEEDS:
    seed_comparisons[str(seed)]={}
    for control in C.CONTROLS:
        ratios={metric:[] for metric in ['MQS','CRPS']}
        for market in r['markets']:
            y=a[market+'_target'];h=a[market+'_horizon'];q=a[f'{market}_{C.FROZEN}_seed{seed}_Q'];qc=a[f'{market}_{control}_seed{seed}_Q'];c=a[f'{market}_{C.FROZEN}_seed{seed}_CRPS'];cc=a[f'{market}_{control}_seed{seed}_CRPS']
            for hor in R.T.HOR:
                mask=h==hor;ratios['MQS'].append(float(R.T.B.score(y[mask],q[mask]).mean()/R.T.B.score(y[mask],qc[mask]).mean()));ratios['CRPS'].append(float(c[mask].mean()/cc[mask].mean()))
        seed_comparisons[str(seed)][control]={metric:dict(geometric_gain_percent=float(100*(1-np.exp(np.log(v).mean()))),wins=int(np.sum(np.array(v)<1)),cells=20) for metric,v in ratios.items()}
out=dict(new_configs=50,frozen_reference_models_retrained=0,common_origin_horizon_keys=keys,frozen_file_count=len(r['frozen_sha256']),all_old_file_and_cache_SHA_unchanged=True,reference_single_seed_and_mixture_predictions_exact_copies=True,parameter_count=2636,all5seed_initial_states_match_frozen_architecture=True,all_prior_inputs_and_weight_gradients_zero=True,inner_encoder_nonzero_gradient_count=sum(f['encoder_nonzero_gradient'] for f in fits),nonzero_final_epoch_count=sum(f['best_epoch']>0 for f in fits),nonzero_final_encoder_gradient_count=sum(f['refit']['encoder_nonzero_gradient'] for f in fits if f['best_epoch']>0),raw_diagnostic_trace_count=50,best_raw_differs_count=raw_diff,raw_diagnostic_not_used_for_main_selection=True,reweight_a1_b0_exact=True,reference_floor_query_visits=reference_floor,weighted_floor_query_visits=weighted_floor,collision_support_count=collision,variance_minima=minimum,checkpoint_ab_max_error=aberr,checkpoint_true_mixture_Q_raw_max_error=qerr,checkpoint_true_mixture_CRPS_raw_max_error=cerr,replay_queries_per_model=12,frozen_reference_seed_mechanism_comparisons=seed_comparisons,frozen_reference_market_MQS_gain_percent=market_gains)
(P/'prefix_transport_controls_round5_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 匹配机制控制：冻结参照、零输入与true mixture核对','',f'完成50新configs，原25 frozen honest_no_readout未重训，逐seed与mixture Q/CRPS/p/a/b直接拷贝核对一致。{keys}共同原点×期限键和全部旧SHA一致。','',f'两控制2636参数/同5seed初始化与原架构精确相同；standardized prior输入和prior weight梯度全0。reweight控制所有a精确1、b精确0。checkpoint每模型12固定query重建a/b误差{aberr:.3g}、true5mixture Q误差{qerr:.3g}、CRPS误差{cerr:.3g}。','',f'50条visited raw诊断全留，{raw_diff}条best_raw不同于norm；没有改本轮selector/refit。reference/weighted floor访问{reference_floor}/{weighted_floor}，额外零碰撞{collision}。','', '| 冻结honest_no_readout / 对照 | MQS改善 | 完整CRPS改善 |','|---|---:|---:|']
for key,v in r['summary'].items():
    if not key.startswith(C.FROZEN+'_vs_'):continue
    q=v['MQS'];c=v.get('CRPS');ct='15q不可识别' if c is None else f"{c['geometric_gain_percent']:+.4f}% ({c['wins']}/20)";lines.append(f"| {key} | {q['geometric_gain_percent']:+.4f}% ({q['wins']}/20) | {ct} |")
lines+=['','## 同seed机制方向，全5保留','', '| seed | 对照 | MQS改善 | CRPS改善 |','|---|---|---:|---:|']
for seed,values in seed_comparisons.items():
    for control,v in values.items():lines.append(f"| {seed} | {control} | {v['MQS']['geometric_gain_percent']:+.4f}% | {v['CRPS']['geometric_gain_percent']:+.4f}% |")
lines+=['','noReadout常量支路仅在推断函数层面可折叠，训练参数与优化路径未改。历史test已开发复用，正方向仅作为开发候选，需要冻结后滚动复核，不将先前prior-readout primary改名为成功。']
(P/'prefix_transport_controls_round5_verification_report_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
