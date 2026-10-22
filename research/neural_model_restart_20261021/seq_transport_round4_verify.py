# -*- coding: utf-8 -*-
"""Verify completed transport artifacts, reconstruct true mixtures from checkpoints."""
import sys
sys.dont_write_bytecode=True
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
import seq_transport_round4 as M
P=Path(__file__).resolve().parent
torch.set_num_threads(1)
r=json.loads((P/'seq_transport_round4.json').read_text(encoding='utf-8'));a=np.load(P/'seq_transport_round4_predictions.npz')
base=np.load(P/'empirical_tilt_round3_followup_predictions.npz');lstm=np.load(P/'deep_seq_round4_predictions.npz');panel=pd.read_csv(M.T.B.PANEL,parse_dates=['date'])
assert set(r['markets'])==set(M.T.B.CUT)
fits=[f for market in r['markets'].values() for fs in market['fits'].values() for f in fs];assert len(fits)==100
assert all(f['epochs_run']>=30 and f['refit_epochs']==f['best_epoch'] for f in fits)
assert all(hashlib.sha256((P/name).read_bytes()).hexdigest()==sha for name,sha in r['frozen_sha256'].items())
minv0=[];minvw=[];floor0=floorw=visits=0;collision_native=0
for f in fits:
    stats=[f['training_geometry'],f['validation_geometry'],f['refit']['training_geometry']]+list(f['test_geometry'].values())
    for s in stats:
        if s['min_uniform_variance'] is not None:minv0.append(s['min_uniform_variance'])
        if s['min_weighted_variance'] is not None:minvw.append(s['min_weighted_variance'])
        floor0+=s['uniform_floor_visits'];floorw+=s['weighted_floor_visits'];visits+=s['query_visits']
    collision_native+=sum(s['collision_count'] for s in f['test_geometry'].values())
keys=0;replay_qerr=0.;replay_cerr=0.;ab_error=0.;actual_collision_count=0;seed_summary={};market_summary={}
for market,mres in r['markets'].items():
    for field in ['origin','horizon','target','scale']:assert np.array_equal(a[market+'_'+field],base[market+'_'+field])
    y=a[market+'_target'];h=a[market+'_horizon'];scale=a[market+'_scale'];keys+=len(y)
    for variant in M.VARIANTS:
        for suffix in ['Q','CRPS','p']:
            v=a[market+'_'+variant+'_'+suffix];assert np.isfinite(v).all()
            if suffix=='Q':assert np.all(np.diff(v,axis=1)>=-1e-8)
        mean_c=np.mean([a[f'{market}_{variant}_seed{s}_CRPS'] for s in M.SEEDS],0)
        assert np.max(a[market+'_'+variant+'_CRPS']-mean_c)<1e-6
        for seed in M.SEEDS:
            assert np.all(a[f'{market}_{variant}_seed{seed}_p']<=a[f'{market}_{variant}_seed{seed}_branch_p']+1e-12)
    # Checkpoint reconstruction uses the exact full test encoder batch, then12 fixed queries.
    cp=torch.load(P/f'seq_transport_round4_{market}_checkpoints.pt',weights_only=False)
    data=panel[panel.market==market].sort_values('date').reset_index(drop=True);D=M.S.build(data);X,banks=M.prepare(D,'rf');te=torch.tensor(D['te'])
    assert np.array_equal(cp['context_mean'],D['x_aug'][D['rf']].mean(0)) and np.array_equal(cp['context_std'],D['x_aug'][D['rf']].std(0)+1e-5)
    for bb,saved in zip(banks,cp['banks']):
        for field in ['z','phi2','mature']:assert torch.equal(bb[field],saved[field])
    for variant in M.VARIANTS:
        ops={j:[] for j in range(4)}
        for seed in M.SEEDS:
            net=M.TransportNet(variant);net.load_state_dict(cp['state_dicts'][f'{variant}_seed{seed}']);net.eval()
            with torch.no_grad():
                raw=net(X,te)
                for j,bank in enumerate(banks):
                    positions=np.flatnonzero(h==M.T.HOR[j]);chosen=positions[[0,len(positions)//2,len(positions)-1]];ii=te[chosen]
                    o=M.operator(raw[chosen],bank,X['sid'][ii],X['local'][ii],variant);ops[j].append(o)
                    ab_error=max(ab_error,float(np.max(np.abs(o['a'].numpy()-a[f'{market}_{variant}_seed{seed}_a'][chosen]))),float(np.max(np.abs(o['b'].numpy()-a[f'{market}_{variant}_seed{seed}_b'][chosen]))))
                    marks=o['a'].double()[:,None]*bank['z'].double()+o['b'].double()[:,None];actual_collision_count+=int(((marks==0)&(o['w']>0)).sum())
        for j,bank in enumerate(banks):
            positions=np.flatnonzero(h==M.T.HOR[j]);chosen=positions[[0,len(positions)//2,len(positions)-1]]
            with torch.no_grad():q,c,p,_,_=M.mixture(torch.tensor(y[chosen]/scale[chosen]),bank['z'],ops[j])
            replay_qerr=max(replay_qerr,float(np.max(np.abs(q.numpy()*scale[chosen,None]-a[market+'_'+variant+'_Q'][chosen]))))
            replay_cerr=max(replay_cerr,float(np.max(np.abs(c.numpy()*scale[chosen]-a[market+'_'+variant+'_CRPS'][chosen]))))
    market_summary[market]={}
    for reference in ['frozen_tilt','qrf_original','rule_qrf','lstm_qr','no_sequence_transport','seq_location_scale_only','seq_reweight_only']:
        ratios=[c['scores']['seq_transport']['MQS']/c['scores'][reference]['MQS'] for c in mres['by_horizon'].values()]
        market_summary[market][reference]=float(100*(1-np.exp(np.log(ratios).mean())))
assert replay_cerr<2e-6 and ab_error<2e-5
for seed in M.SEEDS:
    seed_summary[str(seed)]={}
    for reference in ['no_sequence_transport','seq_location_scale_only','seq_reweight_only','frozen_tilt_same_seed','lstm_qr_same_seed']:
        rr=[]
        for market in r['markets']:
            y=a[market+'_target'];h=a[market+'_horizon'];q=a[f'{market}_seq_transport_seed{seed}_Q']
            qb=(base[f'{market}_full_seed{seed}_Q'] if reference=='frozen_tilt_same_seed' else lstm[f'{market}_lstm_qr_seed{seed}_Q'] if reference=='lstm_qr_same_seed' else a[f'{market}_{reference}_seed{seed}_Q'])
            for hor in M.T.HOR:
                mask=h==hor;rr.append(float(M.T.B.score(y[mask],q[mask]).mean()/M.T.B.score(y[mask],qb[mask]).mean()))
        seed_summary[str(seed)][reference]=dict(MQS_geometric_gain_percent=float(100*(1-np.exp(np.log(rr).mean()))),wins=int(np.sum(np.array(rr)<1)),cells=20)
out=dict(complete_fits=100,common_origin_horizon_keys=keys,all_frozen_SHA_unchanged=True,frozen_file_count=len(r['frozen_sha256']),min_actual_uniform_variance=min(minv0),min_actual_weighted_variance=min(minvw),uniform_floor_query_visits=floor0,weighted_floor_query_visits=floorw,total_geometry_query_visits=visits,recorded_final_support_collision_count=collision_native,replayed_float64_collision_support_count_on_fixed_queries=actual_collision_count,checkpoint_ab_max_error=ab_error,checkpoint_true_mixture_Q_raw_max_error=replay_qerr,checkpoint_true_mixture_CRPS_raw_max_error=replay_cerr,checkpoint_replay_queries_per_model=12,seed_MQS_controls=seed_summary,market_MQS_gain_percent=market_summary,epoch0_final_count=sum(f['best_epoch']==0 for f in fits))
(P/'seq_transport_round4_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 条件经验输运：检查点、精确混合与分市场核对','',f'100fit、{keys}个共同原点×期限键核对完成；{len(r["frozen_sha256"])}个旧文件SHA未改。每模型从checkpoint重建固定12个query（每h首/中/末），合并五seed不同支持再算同一分布。','',f'最大a/b复现误差{ab_error:.3g}，mixture Q原始收益单位误差{replay_qerr:.3g}，mixture CRPS误差{replay_cerr:.3g}。所有mixture CRPS满足凸性，不是seedCRPS平均。','',f'全部训练/验证/refit/test几何query访问中，实际最小均匀方差{min(minv0):.6g}、最小倾斜方差{min(minvw):.6g}；uniform floor访问{floor0}、weighted floor访问{floorw}（总访问{visits}，含重复epoch，不是唯一query数）。最终选epoch0共{out["epoch0_final_count"]}/100。','', '| 市场 | 对frozen tilt | 对raw QRF | 对rule QRF | 对LSTM-QR | 对no-seq | 对loc-scale-only | 对reweight-only |','|---|---:|---:|---:|---:|---:|---:|---:|']
for market,values in market_summary.items():lines.append('| '+market+' | '+' | '.join(f'{values[k]:+.4f}%' for k in ['frozen_tilt','qrf_original','rule_qrf','lstm_qr','no_sequence_transport','seq_location_scale_only','seq_reweight_only'])+' |')
lines+=['','## 同seed自身对照（全部保留）','', '| seed | 参照 | MQS改善 | 胜出/20 |','|---|---|---:|---:|']
for seed,values in seed_summary.items():
    for ref,v in values.items():lines.append(f"| {seed} | {ref} | {v['MQS_geometric_gain_percent']:+.4f}% | {v['wins']}/20 |")
lines+=['','正值表示seq_transport更好。固定branch p与最终F非零质量分开保存；Brier使用最终F，零碰撞计数另存。没有宣称退化subset单位方差、总体方差不随频率或新的通用分布族。所有历史测试复用，结论仅为候选模型实验。']
(P/'seq_transport_round4_verification_report_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
