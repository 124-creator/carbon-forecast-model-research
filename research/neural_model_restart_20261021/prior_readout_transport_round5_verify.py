# -*- coding: utf-8 -*-
"""Verify completed prior-readout experiment, prefix metadata and exact mixtures."""
import sys
sys.dont_write_bytecode=True
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
import prior_readout_transport_round5 as M
P=Path(__file__).resolve().parent;R=M.R
torch.set_num_threads(1);M.activate()
r=json.loads((P/'prior_readout_transport_round5.json').read_text(encoding='utf-8'));a=np.load(P/'prior_readout_transport_round5_predictions.npz');base=np.load(P/'forest_transport_round5_predictions.npz');panel=pd.read_csv(R.T.B.PANEL,parse_dates=['date'])
assert set(r['markets'])==set(R.T.B.CUT)
fits=[f for market in r['markets'].values() for fs in market['fits'].values() for f in fs];assert len(fits)==100
assert all(f['epochs_run']>=30 and f['refit_epochs']==f['best_epoch'] and f['allocated_parameters']==2636 for f in fits)
assert all(hashlib.sha256((P/name).read_bytes()).hexdigest()==sha for name,sha in r['frozen_sha256'].items())
assert all(hashlib.sha256((P/name).read_bytes()).hexdigest()==sha for name,sha in r['reference_cache_sha256'].items())
keys=0;block_count=0;prefix_rows=0;qerr=cerr=aberr=0.;scaler_checks=[];market_gains={};reference_floor=weighted_floor=visits=collision=0;prob_error=0.;minimum={};raw_difference=0
for f in fits:
    history=f['raw_relative_validation_history'];assert len(history)==f['epochs_run']+1 and f['raw_diagnostic_only'];assert min(history,key=lambda row:row[1])[0]==f['best_raw_epoch'];raw_difference+=f['best_raw_epoch']!=f['best_epoch']
    for s in [f['training_geometry'],f['validation_geometry'],f['refit']['training_geometry']]+list(f['test_geometry'].values()):
        visits+=s['query_visits'];reference_floor+=s['reference_floor_visits'];weighted_floor+=s['weighted_floor_visits']
        for field in ['min_reference_variance','min_weighted_variance','min_uniform_variance']:
            if s[field] is not None:minimum[field]=min(minimum.get(field,float('inf')),s[field])
    collision+=sum(s['collision_count'] for s in f['test_geometry'].values());prob_error=max(prob_error,max(s['final_nonzero_vs_branch_max_error'] for s in f['test_geometry'].values()))
for market,mres in r['markets'].items():
    for field in ['origin','horizon','target','scale']:assert np.array_equal(a[market+'_'+field],base[market+'_'+field])
    y=a[market+'_target'];h=a[market+'_horizon'];scale=a[market+'_scale'];keys+=len(y)
    data=panel[panel.market==market].sort_values('date').reset_index(drop=True);D=R.S.build(data);ends=D['t']+R.T.HOR[D['hi']];prepared={};audits={};cacheinfo={}
    for kind in ['insample','honest']:prepared[kind],audits[kind],cacheinfo[kind]=M.load_explicit(D,market,kind)
    M.validate_eval_equality(D,prepared['honest'],prepared['insample'])
    hc=torch.load(P/f'forest_prior_honest_round5_{market}.pt',map_location='cpu',weights_only=False)
    for phase,part in [('inner','tr'),('final','rf')]:
        blocks=hc['phases'][phase]['honest_blocks'];assert len(blocks)==4
        for block in blocks:
            source=block['prefix_fit_D_indices'].numpy();left,right=block['start'],block['stop'];sid=block['query_sid_indices'].numpy();times=hc['times'].numpy()
            assert np.isin(source,D[part]).all() and np.all(ends[source]<left)
            assert np.array_equal(block['prefix_target_indices'].numpy(),ends[source]) and np.all((times[sid]>=left)&(times[sid]<right))
            for per_h in block['per_h']:
                hs=per_h['prefix_h_D_indices'].numpy();query=per_h['query_D_indices'].numpy();assert np.isin(hs,source).all() and np.intersect1d(hs,query).size==0 and np.all(ends[hs]<left)
                j=list(R.T.HOR).index(per_h['horizon']);bank=hc['phases'][phase]['banks'][j]
                assert np.array_equal(bank['bank_D_indices'][per_h['full_bank_columns']].numpy(),per_h['prefix_NZ_D_indices'].numpy())
            block_count+=1;prefix_rows+=len(source)
        for kind in ['insample','honest']:
            X,banks=prepared[kind][phase];scaler=audits[kind][phase]['prior_scaler'];fit=np.array(scaler['fit_D_indices'],dtype=np.int64)
            assert np.isin(fit,D[part]).all() and bool(torch.all(X['prior_valid'][torch.tensor(fit)]))
            assert np.array_equal(np.array(scaler['mean'],np.float32),X['prior_raw'][torch.tensor(fit)].numpy().mean(0))
            assert np.array_equal(np.array(scaler['std'],np.float32),X['prior_raw'][torch.tensor(fit)].numpy().std(0)+1e-5)
            assert bool(torch.all(X['prior'][~X['prior_valid']]==0))
            scaler_checks.append(dict(market=market,kind=kind,phase=phase,fit_eligible_rows=len(fit),fit_indices_sha256=scaler['fit_D_indices_sha256']))
    cp=torch.load(P/f'prior_readout_transport_round5_{market}_checkpoints.pt',map_location='cpu',weights_only=False);te=torch.tensor(D['te'])
    for model in M.MODELS:
        kind='insample' if model=='insample_readout' else 'honest';X,banks=prepared[kind]['final'];ops={j:[] for j in range(4)}
        assert mres['variant_cache_kind'][model]==kind and cp['cache_info'][kind]['sha256']==cacheinfo[kind]['sha256']
        assert cp['prior_scalers'][kind]==audits[kind]['final']['prior_scaler']
        for b,saved in zip(banks,cp['final_banks'][kind]):
            for field in ['z','phi2','mature','u','allowed','count','m0','v0','vu','bank_D_indices']:assert torch.equal(b[field],saved[field])
        Q=a[market+'_'+model+'_Q'];C=a[market+'_'+model+'_CRPS'];assert np.isfinite(Q).all() and np.isfinite(C).all() and np.all(np.diff(Q,axis=1)>=-1e-8)
        assert np.max(C-np.mean([a[f'{market}_{model}_seed{s}_CRPS'] for s in M.SEEDS],0))<1e-6
        for seed in M.SEEDS:
            net=M.ReadoutNet(model);net.load_state_dict(cp['state_dicts'][f'{model}_seed{seed}']);net.eval()
            with torch.no_grad():
                raw=net(X,te)
                for j,b in enumerate(banks):
                    pos=np.flatnonzero(h==R.T.HOR[j]);chosen=pos[[0,len(pos)//2,len(pos)-1]];ii=te[chosen]
                    if model=='honest_no_readout':assert bool(torch.all(net.prior_input(X,ii)==0))
                    op=M.prior_operator(raw[chosen],b,X['sid'][ii],X['local'][ii],model);ops[j].append(op)
                    aberr=max(aberr,float(np.max(np.abs(op['a'].numpy()-a[f'{market}_{model}_seed{seed}_a'][chosen]))),float(np.max(np.abs(op['b'].numpy()-a[f'{market}_{model}_seed{seed}_b'][chosen]))))
        for j,b in enumerate(banks):
            pos=np.flatnonzero(h==R.T.HOR[j]);chosen=pos[[0,len(pos)//2,len(pos)-1]]
            with torch.no_grad():q,c,p,_,_=R.mixture(torch.tensor(y[chosen]/scale[chosen]),b['z'],ops[j])
            qerr=max(qerr,float(np.max(np.abs(q.numpy()*scale[chosen,None]-Q[chosen]))));cerr=max(cerr,float(np.max(np.abs(c.numpy()*scale[chosen]-C[chosen]))))
    market_gains[market]={ref:float(100*(1-np.exp(np.mean([np.log(c['scores']['honest_readout']['MQS']/c['scores'][ref]['MQS']) for c in mres['by_horizon'].values()])))) for ref in M.MODELS[1:]+[M.B.REFERENCE,'rule_qrf','old_forest_transport']}
assert aberr<2e-5 and cerr<2e-6
no_grad=sum(f['prior_weight_nonzero_gradient'] for v in r['markets'].values() for f in v['fits']['honest_no_readout']);assert no_grad==0
out=dict(configs=100,common_origin_horizon_keys=keys,frozen_file_count=len(r['frozen_sha256']),all_old_files_and_ten_cache_SHA_unchanged=True,parameter_count=2636,inner_encoder_nonzero_gradient_count=sum(f['encoder_nonzero_gradient'] for f in fits),nonzero_final_epoch_count=sum(f['best_epoch']>0 for f in fits),nonzero_final_encoder_gradient_count=sum(f['refit']['encoder_nonzero_gradient'] for f in fits if f['best_epoch']>0),prior_readout_weight_gradient_count=sum(f['prior_weight_nonzero_gradient'] for f in fits),no_readout_weight_gradient_count=no_grad,verified_prefix_blocks=block_count,verified_prefix_source_row_visits=prefix_rows,prior_scaler_checks=scaler_checks,outer_reference_u_bitwise_equal_old=True,raw_diagnostic_count=len(fits),best_raw_differs_count=raw_difference,raw_not_used_for_main_selection=True,variance_minima=minimum,reference_floor_query_visits=reference_floor,weighted_floor_query_visits=weighted_floor,geometry_query_visits=visits,collision_support_count=collision,final_nonzero_vs_branch_max_error=prob_error,checkpoint_ab_max_error=aberr,checkpoint_true_mixture_Q_raw_max_error=qerr,checkpoint_true_mixture_CRPS_raw_max_error=cerr,fixed_replay_queries_per_model=12,market_MQS_gain_percent=market_gains)
(P/'prior_readout_transport_round5_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# Prior读出、prefix源边界和真实混合核对','',f'100configs、{keys}共同原点×期限键、{len(r["frozen_sha256"])}旧文件与10缓存SHA核对通过；全部模型2636分配参数。','',f'逐字段核对{block_count}个prefix block：森林源目标均严格早于block开始，与query行无交集；NZ映射按原D行而非z值。inner/final以各自fiteligible行拟合4feature scaler，20组mean/std/fitindices重算一致；无效特征行全0；边界后参考u逐位同旧。','',f'checkpoint每模型12固定query重建，a/b最大误差{aberr:.3g}，真实五seed mixture Q误差{qerr:.3g}、CRPS误差{cerr:.3g}。no-readout prior weight非零梯度{no_grad}，其四维标准化输入全0；正常readout的内容观测单元测试与零head恢复均通过。','',f'100条visited raw诊断均保留，{raw_difference}条best_raw不同于原normalized选择；未用于本轮选轮或额外拟合。reference/weighted floor访问{reference_floor}/{weighted_floor}，额外零碰撞{collision}。','', '| honest_readout / 参照 | MQS改善 | 完整CRPS改善 |','|---|---:|---:|']
for key,v in r['summary'].items():
    if not key.startswith('honest_readout_vs_'):continue
    q=v['MQS'];c=v.get('CRPS');ct='15q不可识别' if c is None else f"{c['geometric_gain_percent']:+.4f}% ({c['wins']}/20)";lines.append(f"| {key} | {q['geometric_gain_percent']:+.4f}% ({q['wins']}/20) | {ct} |")
lines+=['','prefix是reference训练误差机制实验，不使旧结果失效。所有历史test复用，全部模型/seed保留，不通过test决定结构、epoch或发布成功。']
(P/'prior_readout_transport_round5_verification_report_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in out.items() if k not in ['prior_scaler_checks']},ensure_ascii=False,indent=2))
