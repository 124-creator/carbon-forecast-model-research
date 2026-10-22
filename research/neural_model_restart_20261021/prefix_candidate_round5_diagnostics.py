# -*- coding: utf-8 -*-
"""Read-only evidence extraction for a prespecified secondary control.

No fits, seed selection, new F construction, or mixture rescoring. Scores come
from frozen cells/NPZ; raw heads come from all-query checkpoint forwards.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import prior_readout_transport_round5 as N

P=Path(__file__).resolve().parent
TAG='prefix_candidate_round5_diagnostics'
SOURCE='prior_readout_transport_round5'
MODEL='honest_no_readout'
SEEDS=[11,22,33,44,55]
BASES=['rule_qrf','forest_reference_only']


def stats(delta,ell,beta,s0,rawscale):
    factor=torch.exp(torch.from_numpy(np.ascontiguousarray(ell))).numpy()
    mean_delta=delta.mean(0)
    # This is implied by common-reference target moments, not an average F.
    mixture_std=np.sqrt((factor**2).mean(0)+delta.var(0))
    return dict(n_queries=delta.shape[1],n_components=delta.shape[0],
        delta_mean=float(delta.mean()),delta_abs_mean=float(np.abs(delta).mean()),delta_rms=float(np.sqrt((delta**2).mean())),
        ell_mean=float(ell.mean()),ell_abs_mean=float(np.abs(ell).mean()),
        expell_mean=float(factor.mean()),expell_abs_deviation_from_one=float(np.abs(factor-1).mean()),
        expell_min=float(factor.min()),expell_max=float(factor.max()),
        beta_rms=float(np.sqrt((beta**2).mean())),beta1_mean=float(beta[:,:,0].mean()),beta2_mean=float(beta[:,:,1].mean()),
        mean_over_seeds_delta_abs_mean=float(np.abs(mean_delta).mean()),
        target_mixture_std_ratio_mean=float(mixture_std.mean()),
        target_mixture_std_ratio_abs_deviation=float(np.abs(mixture_std-1).mean()),
        target_raw_mean_shift_abs_mean=float(np.abs(delta*s0[None,:]*rawscale[None,:]).mean()))


def main():
    start=time.perf_counter();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    files=[SOURCE+'.py',SOURCE+'.json',SOURCE+'_cells.csv',SOURCE+'_predictions.npz',
        SOURCE+'_verification.json','forest_transport_round5.py','seq_transport_round4.py',
        'deep_seq_round4.py','empirical_tilt_round3.py','neural_restart.py']
    files += [f'{SOURCE}_{m}_checkpoints.pt' for m in N.R.T.B.CUT]
    files += [f'forest_prior_honest_round5_{m}.pt' for m in N.R.T.B.CUT]
    before={name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in files}
    panel_hash=hashlib.sha256(N.R.T.B.PANEL.read_bytes()).hexdigest()
    source=json.loads((P/(SOURCE+'.json')).read_text(encoding='utf-8'))
    table=pd.read_csv(P/(SOURCE+'_cells.csv'),dtype={'seed':str},float_precision='round_trip')
    prediction=np.load(P/(SOURCE+'_predictions.npz'))
    panel=pd.read_csv(N.R.T.B.PANEL,parse_dates=['date'])
    manifest=dict(candidate=MODEL,status='prespecified secondary exploratory control; primary honest_readout unchanged',
        seeds=SEEDS,cache_kind='honest',final_context_fit='D.rf',operation='all test query net(X,te) raw head inference only',
        transformed_heads=dict(delta='1.5*tanh(raw0)',ell='1.5*tanh(raw1)',expell='exp(ell)',beta='.35*tanh(raw2:4)'),
        beta_rms='sqrt(mean over queries and two coordinates of beta^2)',
        ensemble_head_statistics='equal-weight component descriptives over all five seed heads; no single ensemble neural head is asserted',
        ensemble_target_moments='mean shift/s0=mean_seed(delta); std/s0=sqrt(mean_seed(exp(2ell))+Var_seed(delta)), with common reference, fixed branch p and inactive floors (source verification)',
        score_source='stored NPZ/cells only; no new full F scoring or checkpoint-mixture validation',
        zero_training=True,no_seed_selection=True,numeric_neutral_tolerance=1e-6,frozen_sha256_before=before,source_panel_sha256=panel_hash)
    (P/f'{TAG}_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    cells=[];epochs=[];heads=[];arrays={};report=dict(manifest=manifest,markets={})
    with threadpool_limits(limits=1),torch.no_grad():
        for market in N.R.T.B.CUT:
            data=panel[panel.market==market].sort_values('date').reset_index(drop=True)
            D=N.R.S.build(data);te=D['te'];hi=D['hi'][te]
            prepared,_,cacheinfo=N.load_explicit(D,market,'honest');Xf,banks=prepared['final']
            assert cacheinfo['sha256']==before[cacheinfo['path']]
            idx=torch.tensor(te);assert bool(torch.all(Xf['prior_valid'][idx]))
            checkpoint=torch.load(P/f'{SOURCE}_{market}_checkpoints.pt',map_location='cpu',weights_only=False)
            info={int(x['seed']):x for x in source['markets'][market]['fits'][MODEL]}
            assert sorted(info)==SEEDS
            for key,val in [('origin',D['t'][te]),('horizon',N.R.T.HOR[hi]),('target',D['y'][te]),('scale',D['scale'][te])]:
                assert np.array_equal(val,prediction[market+'_'+key]);arrays[market+'_'+key]=val
            m0=np.empty(len(te));s0=np.empty(len(te))
            for j,b in enumerate(banks):
                pos=np.flatnonzero(hi==j);sid=Xf['sid'][idx[torch.tensor(pos)]]
                m0[pos]=b['m0'][sid].numpy();s0[pos]=b['v0'][sid].clamp_min(N.R.FLOOR).sqrt().numpy()
            raw_list=[]
            for seed in SEEDS:
                fit=info[seed];epoch=int(fit['refit_epochs']);assert epoch==int(fit['best_epoch'])
                net=N.ReadoutNet(MODEL);net.load_state_dict(checkpoint['state_dicts'][f'{MODEL}_seed{seed}']);net.eval()
                assert bool(torch.all(net.prior_input(Xf,idx)==0))
                raw=net(Xf,idx).detach()
                delta=1.5*torch.tanh(raw[:,0]);ell=1.5*torch.tanh(raw[:,1]);beta=.35*torch.tanh(raw[:,2:4])
                raw_list.append(raw.numpy())
                epochs.append(dict(market=market,seed=seed,selected_epoch=epoch,epoch_zero=epoch==0,
                    nonzero_raw_query_count=int((raw.abs().max(1).values>0).sum()),raw_max_abs=float(raw.abs().max()),
                    zero_raw_all_queries=bool(torch.all(raw==0)),epochs_run_inner=int(fit['epochs_run']),
                    best_raw_epoch_diagnostic_only=int(fit['best_raw_epoch'])))
                for j,h in enumerate(N.R.T.HOR):
                    pos=hi==j
                    st=stats(delta.numpy()[None,pos],ell.numpy()[None,pos],beta.numpy()[None,pos,:],s0[pos],D['scale'][te[pos]])
                    heads.append(dict(market=market,horizon=int(h),seed=str(seed),selected_epoch=epoch,active_seed_count=int(epoch>0),**st))
                del net
            raw=np.stack(raw_list);delta=1.5*np.tanh(raw[:,:,0]);ell=1.5*np.tanh(raw[:,:,1]);beta=.35*np.tanh(raw[:,:,2:4])
            # Use the actual torch transforms for saved diagnostics as well.
            tr=torch.tensor(raw);delta=(1.5*torch.tanh(tr[:,:,0])).numpy();ell=(1.5*torch.tanh(tr[:,:,1])).numpy();beta=(.35*torch.tanh(tr[:,:,2:4])).numpy()
            for name,value in [('raw_heads',raw),('delta',delta),('ell',ell),('expell',torch.exp(torch.tensor(ell)).numpy()),('beta',beta),('reference_m0',m0),('reference_s0',s0)]:arrays[market+'_'+name]=value
            for j,h in enumerate(N.R.T.HOR):
                pos=hi==j
                st=stats(delta[:,pos],ell[:,pos],beta[:,pos,:],s0[pos],D['scale'][te[pos]])
                heads.append(dict(market=market,horizon=int(h),seed='ensemble_components',selected_epoch=None,
                    active_seed_count=sum(int(info[s]['refit_epochs'])>0 for s in SEEDS),**st))
                for seed in [str(s) for s in SEEDS]+['ensemble']:
                    row=table[(table.market==market)&(table.horizon==int(h))&(table.model==MODEL)&(table.seed==seed)]
                    assert len(row)==1;rr=row.iloc[0]
                    key=market+'_'+MODEL+('' if seed=='ensemble' else '_seed'+seed)
                    assert prediction[key+'_Q'][pos].shape==(int(rr['n']),15)
                    assert abs(float(prediction[key+'_CRPS'][pos].mean())-float(rr.CRPS))<1e-12
                    out=dict(market=market,horizon=int(h),seed=seed,n=int(rr['n']),MQS=float(rr.MQS),CRPS=float(rr.CRPS),
                        selected_epoch=None if seed=='ensemble' else int(info[int(seed)]['refit_epochs']))
                    for base in BASES:
                        br=table[(table.market==market)&(table.horizon==int(h))&(table.model==base)&(table.seed=='ensemble')]
                        assert len(br)==1
                        for metric in ['MQS','CRPS']:
                            den=float(br.iloc[0][metric]);ratio=out[metric]/den
                            out[f'{base}_{metric}']=den;out[f'{metric}_ratio_vs_{base}']=ratio
                            out[f'{metric}_gain_percent_vs_{base}']=100*(1-ratio)
                    cells.append(out)
            report['markets'][market]=dict(cache_sha256=cacheinfo['sha256'],epoch_zero=sum(int(info[s]['refit_epochs'])==0 for s in SEEDS),
                epoch_nonzero=sum(int(info[s]['refit_epochs'])>0 for s in SEEDS),selected_epochs={str(s):int(info[s]['refit_epochs']) for s in SEEDS})
            print('HEADS_DONE',market,report['markets'][market]['selected_epochs'],'elapsed',round(time.perf_counter()-start,1),flush=True)
            del checkpoint,prepared,Xf,banks
    cellframe=pd.DataFrame(cells);headframe=pd.DataFrame(heads);epochframe=pd.DataFrame(epochs);summaries=[]
    for seed in [str(s) for s in SEEDS]+['ensemble']:
        part=cellframe[cellframe.seed==seed];assert len(part)==20
        for base in BASES:
            result=dict(seed=seed,reference=base,cells=20)
            for metric in ['MQS','CRPS']:
                r=part[f'{metric}_ratio_vs_{base}'].to_numpy()
                result[metric+'_geometric_gain_percent']=float(100*(1-np.exp(np.log(r).mean())))
                result[metric+'_wins']=int((r<1).sum())
                result[metric+'_near_equal_cells_1e6']=int((np.abs(r-1)<=1e-6).sum())
            summaries.append(result)
    report['seed_summary']=summaries
    report['epochs']=dict(total=len(epochs),zero=int(epochframe.epoch_zero.sum()),nonzero=int((~epochframe.epoch_zero).sum()),
        by_seed={str(s):dict(zero=int(epochframe[epochframe.seed==s].epoch_zero.sum()),nonzero=int((~epochframe[epochframe.seed==s].epoch_zero).sum())) for s in SEEDS})
    columns=['delta_mean','delta_abs_mean','delta_rms','expell_mean','expell_abs_deviation_from_one','ell_abs_mean','beta_rms','target_mixture_std_ratio_mean','target_mixture_std_ratio_abs_deviation']
    allmean=headframe[headframe.seed=='ensemble_components'][columns].mean().to_dict()
    active=headframe[(headframe.seed!='ensemble_components')&(headframe.selected_epoch>0)]
    report['head_macro20_component_statistics']={k:float(v) for k,v in allmean.items()}
    report['head_active_fit_cell_statistics']={k:float(v) for k,v in active[columns].mean().to_dict().items()}
    after={name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in files}
    assert before==after and panel_hash==hashlib.sha256(N.R.T.B.PANEL.read_bytes()).hexdigest()
    report.update(frozen_sha256_after=after,elapsed_seconds=time.perf_counter()-start)
    pd.DataFrame(summaries).to_csv(P/f'{TAG}_seed_summary.csv',index=False,encoding='utf-8-sig')
    cellframe.to_csv(P/f'{TAG}_cells.csv',index=False,encoding='utf-8-sig')
    epochframe.to_csv(P/f'{TAG}_epochs.csv',index=False,encoding='utf-8-sig')
    headframe.to_csv(P/f'{TAG}_heads.csv',index=False,encoding='utf-8-sig')
    np.savez_compressed(P/f'{TAG}_head_queries.npz',**arrays)
    (P/f'{TAG}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Prefix候选只读证据提取','',
        '对象为预设次要控制honest_no_readout；不改主候选honest_readout，不训练、不选seed，不重算full-mixture评分或重复checkpoint-mixture验证。',
        '评分直接读取冻结cells/NPZ；真实head通过N.load_explicit(kind=honest)的final D.rf context与每个checkpoint在全部te上net(X,te)读取。', '',
        '| seed | 参照 | MQS几何改善 | MQS胜格 | CRPS几何改善 | CRPS胜格 |', '|---|---|---:|---:|---:|---:|']
    for row in summaries:lines.append(f"| {row['seed']} | {row['reference']} | {row['MQS_geometric_gain_percent']:+.6f}% | {row['MQS_wins']}/20 | {row['CRPS_geometric_gain_percent']:+.6f}% | {row['CRPS_wins']}/20 |")
    lines+=['','所有seed按预定顺序完整呈现。严格胜格按ratio<1计算，另存1e-6相对误差内的近相等格数，避免把epoch0附近数值差异当机制证据。','',
        '## 最终epoch','',f"共25个market×seed重拟合，epoch0={report['epochs']['zero']}，非零epoch={report['epochs']['nonzero']}。",'',
        '| 市场 | seed11/22/33/44/55最终epoch | epoch0 / 非零 |','|---|---|---|']
    for market,v in report['markets'].items():lines.append(f"| {market} | {'/'.join(str(v['selected_epochs'][str(s)]) for s in SEEDS)} | {v['epoch_zero']} / {v['epoch_nonzero']} |")
    lines+=['','## 真实head描述','',
        'delta=1.5tanh(raw0)，ell=1.5tanh(raw1)，尺度倍数exp(ell)，beta=.35tanh(raw2:4)。beta RMS定义为对query和两个坐标的平方平均开根号。',
        '下列整体数值先在每个market/h内等权汇总五个seed组件，再对20格等权平均；逐seed/market/h完整表另存。', '',
        f"- mean(delta)={allmean['delta_mean']:.7f}；mean|delta|={allmean['delta_abs_mean']:.7f} 个参考标准差。",
        f"- mean exp(ell)={allmean['expell_mean']:.7f}；mean|exp(ell)-1|={allmean['expell_abs_deviation_from_one']:.7f}。",
        f"- beta RMS的20格均值={allmean['beta_rms']:.7f}。",
        f"- 非零epoch的活跃fit-cell中，mean|delta|={report['head_active_fit_cell_statistics']['delta_abs_mean']:.7f}，mean|exp(ell)-1|={report['head_active_fit_cell_statistics']['expell_abs_deviation_from_one']:.7f}。", '',
        '幅度上是位置与尺度的联合小残差，位置相对参考标准差的平均调整略大，两者同阶；不能解释为单纯尺度校准。有符号均值接近0或尺度均值接近1可能掩盖query间抵消。',
        '只有seed11、33在固定reference上的MQS/CRPS同时为正；17/25个最终fit是epoch0，多数市场ensemble包含四个零head组件，不将微小ensemble增量描述为五个单模型都稳定改善。',
        'delta直接控制目标mark均值m0+s0*delta，exp(ell)直接控制目标mark标准差s0*exp(ell)；a/b受倾斜后的mw/sw共同影响，不能单独解释成mean/variance变化。',
        '真实五seedmixture没有单个神经head；ensemble_components行只描述组件等权统计。另存每query seed均值delta的绝对均值，以及共同参考下由目标矩推得的mixture std ratio=sqrt(mean(exp(2ell))+Var(delta))；它包含组件均值差异。',
        '头幅度只能说明参数行为，不能分解MQS/CRPS收益，也不能替代matched位置/尺度/shape机制对照。现有source验证显示floor和额外零碰撞均为0，本次沿用该既有结论。', '',
        '完整cells表有20格×全部五seed及ensemble，epochs表25行，heads表120行，head_queries存所有query raw及实际变换后的heads和参考矩。',
        f"只读诊断耗时{report['elapsed_seconds']:.3f}秒；源文件、checkpoint、honest cache与panel SHA前后一致。"]
    (P/f'{TAG}_report_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(seed_summary=summaries,epochs=report['epochs'],heads=allmean),ensure_ascii=False,indent=2),flush=True)
    print('ALL_DIAGNOSTICS_DONE',report['elapsed_seconds'],flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
