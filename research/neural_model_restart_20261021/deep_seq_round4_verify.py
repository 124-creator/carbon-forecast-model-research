# -*- coding: utf-8 -*-
"""Read-only verification/summary of completed deep_seq_round4 outputs."""
import sys
sys.dont_write_bytecode=True
import json,hashlib
from pathlib import Path
import numpy as np
import neural_restart as B
P=Path(__file__).resolve().parent
R=json.loads((P/'deep_seq_round4.json').read_text(encoding='utf-8'))
A=np.load(P/'deep_seq_round4_predictions.npz')
C=np.load(P/'empirical_tilt_round3_followup_predictions.npz')
models=['lstm_qr','gru_qr','tcn_qr','transformer_qr'];seeds=[11,22,33,44,55]
assert set(R['markets'])==set(B.CUT)
fits=[f for m in R['markets'].values() for fs in m['fits'].values() for f in fs]
assert len(fits)==100 and all(f['inner_encoder_nonzero_gradient'] for f in fits)
assert all(f['inner_epochs']>=30 and 0<=f['best_epoch']<=80 and f['refit_epochs']==f['best_epoch'] for f in fits)
assert all(f['refit_encoder_nonzero_gradient'] and f['refit_encoder_parameter_L2_change']>0 for f in fits if f['best_epoch']>0)
assert all(hashlib.sha256((P/name).read_bytes()).hexdigest()==sha for name,sha in R['frozen_sha256'].items())
keys=0;candidate_ratios=[];seed_summary={};by_market={};max_mean_error=0
for market,mres in R['markets'].items():
    for field in ['origin','horizon','target','scale']:
        assert np.array_equal(A[market+'_'+field],C[market+'_'+field])
    y=A[market+'_target'];h=A[market+'_horizon'];keys+=len(y)
    assert np.array_equal(A[market+'_candidate_probability_mixture_Q'],C[market+'_full_ensemble_Q'])
    cq=np.mean([C[f'{market}_full_seed{s}_Q'] for s in seeds],0)
    assert np.array_equal(A[market+'_candidate_quantile_average_Q'],cq)
    for hor in [1,5,10,20]:
        mask=h==hor;candidate_ratios.append(float(B.score(y[mask],C[market+'_full_ensemble_Q'][mask]).mean()/B.score(y[mask],cq[mask]).mean()))
    by_market[market]={}
    for model in models:
        qs=[A[f'{market}_{model}_seed{s}_Q'] for s in seeds];ens=A[f'{market}_{model}_Q']
        error=float(np.max(np.abs(ens-np.mean(qs,0))));max_mean_error=max(max_mean_error,error);assert error<1e-12
        for q in qs+[ens]:assert np.isfinite(q).all() and np.all(np.diff(q,axis=1)>=-1e-8)
        ratios=[mres['by_horizon'][str(hh)]['scores'][model]['MQS']/mres['by_horizon'][str(hh)]['scores']['candidate_probability_mixture']['MQS'] for hh in [1,5,10,20]]
        by_market[market][model]=dict(MQS_geometric_gain_vs_candidate_probability_mixture=float(100*(1-np.exp(np.log(ratios).mean()))),wins=int(np.sum(np.array(ratios)<1)))
for model in models:
    seed_summary[model]={}
    for seed in seeds:
        ratios=[]
        for market in R['markets']:
            y=A[market+'_target'];h=A[market+'_horizon'];q=A[f'{market}_{model}_seed{seed}_Q'];base=C[f'{market}_full_seed{seed}_Q']
            for hor in [1,5,10,20]:
                mask=h==hor;ratios.append(float(B.score(y[mask],q[mask]).mean()/B.score(y[mask],base[mask]).mean()))
        seed_summary[model][str(seed)]=dict(MQS_gain_vs_same_seed_candidate=float(100*(1-np.exp(np.log(ratios).mean()))),wins=int(np.sum(np.array(ratios)<1)),cells=20)
result=dict(all_checks_passed=True,fit_count=len(fits),test_origin_horizon_keys=keys,frozen_file_count=len(R['frozen_sha256']),all_inner_encoders_have_nonzero_gradient=True,all_selected_nonzero_refits_have_nonzero_encoder_gradient=True,zero_epoch_final_count=sum(f['best_epoch']==0 for f in fits),max_quantile_ensemble_reconstruction_error=max_mean_error,candidate_probability_mixture_vs_quantile_average=dict(MQS_geometric_gain_percent=float(100*(1-np.exp(np.log(candidate_ratios).mean()))),wins=int(np.sum(np.array(candidate_ratios)<1)),cells=20),by_market=by_market,all_five_seed_comparisons=seed_summary,precision='canonical float64 raw targets; exact same origins,h,scale; no exact CRPS inferred')
(P/'deep_seq_round4_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 四模型完整性、集成与逐seed核对','',f'已核对100fit、{keys}个共同原点×期限键、{result["frozen_file_count"]}个冻结文件SHA。全部inner encoder有非零梯度，非零epoch最终refit也有非零encoder梯度。验证选择epoch0的最终模型共{result["zero_epoch_final_count"]}个；它们没有假装训练过最终encoder。','', '| 模型 | 对候选概率混合MQS改善 | 对候选同Qavg表示MQS改善 |','|---|---:|---:|']
for model in models:
    p=R['summary'][model+'_vs_candidate_probability_mixture']['MQS'];q=R['summary'][model+'_vs_candidate_quantile_average']['MQS'];lines.append(f"| {model} | {p['geometric_gain_percent']:+.4f}% ({p['wins']}/20) | {q['geometric_gain_percent']:+.4f}% ({q['wins']}/20) |")
cc=result['candidate_probability_mixture_vs_quantile_average'];lines+=['',f"候选自身概率混合相对其五seed分位平均的MQS改善为{cc['MQS_geometric_gain_percent']:+.4f}%（{cc['wins']}/20）。两种集成不是同一模型定义，比较均保留。",'', '| 市场 | LSTM-QR | GRU-QR | TCN-QR | Transformer-QR |','|---|---:|---:|---:|---:|']
for market,values in by_market.items():lines.append('| '+market+' | '+' | '.join(f"{values[m]['MQS_geometric_gain_vs_candidate_probability_mixture']:+.4f}%" for m in models)+' |')
lines+=['','## 每seed对其同seed冻结候选（全保留）','', '| 模型 | seed | MQS改善 | 胜出/20 |','|---|---:|---:|---:|']
for model,ss in seed_summary.items():
    for seed,v in ss.items():lines.append(f"| {model} | {seed} | {v['MQS_gain_vs_same_seed_candidate']:+.4f}% | {v['wins']}/20 |")
lines+=['','这些是固定小宽度基线和历史复用测试，不能解释为所有LSTM/GRU/TCN/Transformer配置的结论。QR用64步原始序列加同context，候选仅用context摘要和历史响应bank；QR可利用更多时序细节。所有QR supervised train标签均来自inner训练区间，候选另有causal bank的至少32成熟样本条件，因此训练query有效数并非强行相同。没有test选模型、seed或epoch。','', '主脚本完整报告和cells表保留MQS/WIS/MAE/80与95覆盖率。只有15q，未报告完整CRPS。']
(P/'deep_seq_round4_verification_report_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
