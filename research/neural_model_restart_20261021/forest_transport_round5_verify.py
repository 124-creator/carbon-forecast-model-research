# -*- coding: utf-8 -*-
"""Post-run verification of phase-separated forest transport; no refitting."""
import sys
sys.dont_write_bytecode=True
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
import forest_transport_round5 as F
P=Path(__file__).resolve().parent
torch.set_num_threads(1);F.activate();R=F.R
r=json.loads((P/'forest_transport_round5.json').read_text(encoding='utf-8'));a=np.load(P/'forest_transport_round5_predictions.npz');old=np.load(P/'seq_transport_round4_predictions.npz');panel=pd.read_csv(R.T.B.PANEL,parse_dates=['date'])
assert set(r['markets'])==set(R.T.B.CUT)
fits=[f for m in r['markets'].values() for fs in m['fits'].values() for f in fs];assert len(fits)==75
assert all(f['epochs_run']>=30 and f['refit_epochs']==f['best_epoch'] for f in fits)
assert all(hashlib.sha256((P/name).read_bytes()).hexdigest()==sha for name,sha in r['frozen_sha256'].items())
assert all(hashlib.sha256((P/name).read_bytes()).hexdigest()==sha for name,sha in r['reference_cache_sha256'].items())
keys=0;qerr=cerr=aberr=0.;traces=[];mins={};reference_floor=weighted_floor=visits=collisions=0;prob_error=0.
for f in fits:
    assert f['raw_diagnostic_only'] and len(f['raw_relative_validation_history'])==f['epochs_run']+1
    best=min(f['raw_relative_validation_history'],key=lambda row:row[1]);assert best[0]==f['best_raw_epoch']
    traces.append(dict(normalized_best=f['best_epoch'],raw_relative_best=f['best_raw_epoch'],raw_relative_score=best[1]))
    stats=[f['training_geometry'],f['validation_geometry'],f['refit']['training_geometry']]+list(f['test_geometry'].values())
    for s in stats:
        visits+=s['query_visits'];reference_floor+=s['reference_floor_visits'];weighted_floor+=s['weighted_floor_visits']
        for key in ['min_reference_variance','min_weighted_variance','min_uniform_variance','min_reference_uniform_variance_ratio','min_weighted_uniform_variance_ratio']:
            if s[key] is not None:mins[key]=min(mins.get(key,float('inf')),s[key])
    collisions+=sum(s['collision_count'] for s in f['test_geometry'].values())
    prob_error=max(prob_error,max(s['final_nonzero_vs_branch_max_error'] for s in f['test_geometry'].values()))
for market,mres in r['markets'].items():
    for field in ['origin','horizon','target','scale']:assert np.array_equal(a[market+'_'+field],old[market+'_'+field])
    h=a[market+'_horizon'];y=a[market+'_target'];scale=a[market+'_scale'];keys+=len(y)
    for model in F.LEARNED+[F.REFERENCE]:
        for suffix in ['Q','CRPS','p']:assert np.isfinite(a[market+'_'+model+'_'+suffix]).all()
        assert np.all(np.diff(a[market+'_'+model+'_Q'],axis=1)>=-1e-8)
    for model in F.LEARNED:
        mean_c=np.mean([a[f'{market}_{model}_seed{s}_CRPS'] for s in F.SEEDS],0)
        assert np.max(a[market+'_'+model+'_CRPS']-mean_c)<1e-6
    # Load only completed inner/final phases; replay checkpoint final predictions.
    data=panel[panel.market==market].sort_values('date').reset_index(drop=True);D=R.S.build(data);prepared,audit,meta=F.load_market(D,market);X,banks=prepared['final'];te=torch.tensor(D['te'])
    cp=torch.load(P/f'forest_transport_round5_{market}_checkpoints.pt',map_location='cpu',weights_only=False)
    assert cp['reference_cache']['sha256']==meta['sha256'] and torch.equal(cp['reference_cache']['row_sid'],X['sid'])
    for b,saved in zip(banks,cp['banks']):
        for key in ['z','phi2','mature','u','allowed','count','m0','v0','vu']:assert torch.equal(b[key],saved[key])
    for model in F.LEARNED:
        ops={j:[] for j in range(4)}
        for seed in F.SEEDS:
            net=R.TransportNet(model);net.load_state_dict(cp['state_dicts'][f'{model}_seed{seed}']);net.eval()
            with torch.no_grad():
                raw=net(X,te)
                for j,b in enumerate(banks):
                    pos=np.flatnonzero(h==R.T.HOR[j]);chosen=pos[[0,len(pos)//2,len(pos)-1]];ii=te[chosen]
                    o=F.forest_operator(raw[chosen],b,X['sid'][ii],X['local'][ii],model);ops[j].append(o)
                    aberr=max(aberr,float(np.max(np.abs(o['a'].numpy()-a[f'{market}_{model}_seed{seed}_a'][chosen]))),float(np.max(np.abs(o['b'].numpy()-a[f'{market}_{model}_seed{seed}_b'][chosen]))))
        for j,b in enumerate(banks):
            pos=np.flatnonzero(h==R.T.HOR[j]);chosen=pos[[0,len(pos)//2,len(pos)-1]]
            with torch.no_grad():q,c,p,_,_=R.mixture(torch.tensor(y[chosen]/scale[chosen]),b['z'],ops[j])
            qerr=max(qerr,float(np.max(np.abs(q.numpy()*scale[chosen,None]-a[market+'_'+model+'_Q'][chosen]))));cerr=max(cerr,float(np.max(np.abs(c.numpy()*scale[chosen]-a[market+'_'+model+'_CRPS'][chosen]))))
assert cerr<2e-6 and aberr<2e-5
assert mins['min_reference_uniform_variance_ratio']>=.05-1e-5
assert mins['min_weighted_uniform_variance_ratio']>=.05*np.exp(-4.2)-1e-5
out=dict(fits=75,common_origin_horizon_keys=keys,frozen_files=len(r['frozen_sha256']),all_frozen_and_five_prior_cache_SHA_unchanged=True,inner_encoder_nonzero_gradient_count=sum(f['encoder_nonzero_gradient'] for f in fits),positive_final_epochs=sum(f['best_epoch']>0 for f in fits),positive_final_encoder_nonzero_gradient_count=sum(f['refit']['encoder_nonzero_gradient'] for f in fits if f['best_epoch']>0),zero_epoch_final_count=sum(f['best_epoch']==0 for f in fits),raw_diagnostic_trace_count=len(traces),raw_best_differs_from_normalized_count=sum(t['normalized_best']!=t['raw_relative_best'] for t in traces),raw_diagnostic_not_used_for_selection=True,variance_minima=mins,reference_floor_query_visits=reference_floor,weighted_floor_query_visits=weighted_floor,geometry_query_visits=visits,collision_support_count=collisions,final_nonzero_vs_branch_max_error=prob_error,checkpoint_a_b_max_error=aberr,checkpoint_true_mixture_Q_raw_max_error=qerr,checkpoint_true_mixture_CRPS_raw_max_error=cerr,replayed_queries_per_model=12)
(P/'forest_transport_round5_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 森林参考输运：75fit与phase/mixture核对','',f'75配置、{keys}共同原点×期限键、{len(r["frozen_sha256"])}旧文件与5 prior cache SHA保持一致。每phase fit indices、boundary、scaler、bank D行、maturity和query sid严格核对。','',f'checkpoint重建每模型12固定query，a/b最大误差{aberr:.3g}，五seed真实mixture Q误差{qerr:.3g}、CRPS误差{cerr:.3g}。完整精确概率混合满足CRPS凸性，不是seed分数/quantile平均。','',f'全部75条raw-relative validation诊断轨迹已留存，{out["raw_best_differs_from_normalized_count"]}条best_raw_epoch不同于原normalized best；没有用它们改本轮selector或额外refit。仅在既有visited epoch内取诊断best，不宣称新的完整搜索。','',f'最终epoch0 {out["zero_epoch_final_count"]}/75；inner非零encoder梯度{out["inner_encoder_nonzero_gradient_count"]}/75。reference/weighted floor访问{reference_floor}/{weighted_floor}，额外零碰撞{collisions}；实际非零概率与branch差最大{prob_error:.3g}。','', '| 主候选 / 参照 | MQS改善 | 完整CRPS改善 |','|---|---:|---:|']
for key,v in r['summary'].items():
    if not key.startswith('forest_transport_vs_'):continue
    q=v['MQS'];c=v.get('CRPS');cs='15q不可识别' if c is None else f"{c['geometric_gain_percent']:+.4f}% ({c['wins']}/20)";lines.append(f"| {key} | {q['geometric_gain_percent']:+.4f}% ({q['wins']}/20) | {cs} |")
lines+=['','固定forest_reference_only将gamma平滑贡献与NN残差区分；所有三学习组的其他强基线对比在JSON/cells中全保留。所有历史测试已开发复用，不以正结果证明发表成功，也不以负结果否定整个方向。']
(P/'forest_transport_round5_verification_report_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
