# -*- coding: utf-8 -*-
"""Consolidate the five-seed candidate and deterministic empirical controls."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import neural_restart as B

P=Path(__file__).resolve().parent


def main():
    f=np.load(P/'empirical_tilt_round3_followup_predictions.npz')
    b=np.load(P/'empirical_bench_round3_predictions.npz')
    out=[];cells=[]
    for market in B.CUT:
        pre=market+'_'
        for name in ['origin','horizon','target']:
            assert np.array_equal(f[pre+name],b[pre+name]),(market,name)
        y=f[pre+'target'];h=f[pre+'horizon'];q=f[pre+'full_ensemble_Q']
        cs=f[pre+'full_ensemble_CRPS']
        baselines={
          'uniform_static':(f[pre+'uniform_static_Q'],f[pre+'uniform_static_CRPS']),
          'uniform_rule':(f[pre+'uniform_rule_Q'],f[pre+'uniform_rule_CRPS']),
          'horizon_only':(f[pre+'horizon_only_ensemble_Q'],f[pre+'horizon_only_ensemble_CRPS']),
          'state_only':(f[pre+'state_only_ensemble_Q'],f[pre+'state_only_ensemble_CRPS']),
          'HGB_augmented':(f[pre+'hgb_augmented_Q'],None),
          'HGB_state':(f[pre+'hgb_state_only_Q'],None)}
        for name in ['ewhs','recent','knn']:
            baselines[name]=(b[pre+name],b[pre+name+'_CRPS'])
        for hh in [1,5,10,20]:
            keep=h==hh
            mq=float(B.score(y[keep],q[keep]).mean());cr=float(cs[keep].mean())
            for name,(qb,cb) in baselines.items():
                base=float(B.score(y[keep],qb[keep]).mean())
                cells.append({'market':market,'horizon':hh,'baseline':name,'metric':'MQS',
                              'candidate_score':mq,'baseline_score':base,'ratio':mq/base})
                if cb is not None:
                    base=float(cb[keep].mean())
                    cells.append({'market':market,'horizon':hh,'baseline':name,'metric':'CRPS',
                                  'candidate_score':cr,'baseline_score':base,'ratio':cr/base})
    df=pd.DataFrame(cells)
    for (name,metric),d in df.groupby(['baseline','metric'],sort=False):
        rr=d.ratio.to_numpy()
        out.append({'baseline':name,'metric':metric,'geometric_gain_percent':float(100*(1-np.exp(np.log(rr).mean()))),
                    'wins':int((rr<1).sum()),'cells':len(rr)})
    summary=pd.DataFrame(out)
    df.to_csv(P/'ROUND3_FIVE_SEED_BASELINE_CELLS.csv',index=False,encoding='utf-8-sig')
    summary.to_csv(P/'ROUND3_FIVE_SEED_BASELINE_SUMMARY.csv',index=False,encoding='utf-8-sig')
    def item(name,metric):
        d=summary[(summary.baseline==name)&(summary.metric==metric)]
        return None if d.empty else d.iloc[0]
    labels={'uniform_static':'同表示静态经验分布','uniform_rule':'滚动频率＋均匀mark',
            'horizon_only':'仅期限条件的神经mark','state_only':'仅历史状态的神经mark',
            'ewhs':'验证选衰减的EWHS','recent':'验证选窗的近期经验分布',
            'knn':'同信息状态近邻分布','HGB_augmented':'增强输入HGB','HGB_state':'状态输入HGB'}
    rows=['| 五seed候选的参照 | MQS改善 | 胜出单元 | 完整CRPS改善 |',
          '|---|---:|---:|---:|']
    for name,label in labels.items():
        a=item(name,'MQS');c=item(name,'CRPS')
        cv='—' if c is None else f"{c.geometric_gain_percent:+.4f}%"
        rows.append(f'| {label} | {a.geometric_gain_percent:+.4f}% | {int(a.wins)}/{int(a.cells)} | {cv} |')
    verify=json.loads((P/'empirical_tilt_round3_followup_verification.json').read_text(encoding='utf-8'))
    err=max(x['max_CRPS_integral_error'] for x in verify['markets'].values())
    report=f'''# 第三轮：经验分布神经适应候选及样条对照

日期：2026-10-21。状态：找到有真实小幅增益的候选，继续补齐模型实验；尚未宣告整个目标完成。

## 当前主攻候选

保留“滚动零质量先验＋条件有符号经验分布倾斜网络”。零收益概率采用过去128观测的频率与当前拟合区间频率组合，非零收益的分布由神经网络在历史标准化场景上重新分配概率。主体两个隐藏层24/16，输出5个有界倾斜系数；1407个分配参数。概率头没有可靠净增量，当前候选固定概率规则，避免用更多模块掩盖无增益部分。

模型以按期限保存的历史标准化收益z_i为支持；五个响应基函数分别描述有符号位置、绝对幅度、截断二阶幅度、正负尾部方向及尾部超额幅度。每次拟合只从允许训练标签估计基函数统计，系数为0.35*tanh(NN(x,h))，非零场景权重正比exp(beta·phi(z_i))，再与0质量组合。支持按当前已知波动尺度反变换，无KDE，不强制双Normal形状，也不生成跨期限联合路径。

训练检索只放入t_key+h<=query_origin的历史标签，且要求至少32条已成熟非零场景；模型参数和scaler仍按整个允许拟合区间估计。“因果记忆”仅指检索标签时点，不把它称为每个历史查询都独立冻结参数的完整vintage实验。

## 五seed当前结果

固定seed11/22/33/44/55。每seed训练内选epoch后全部precut重新拟合，最终在同一支持上平均完整概率权重，然后从同一混合分布算分位与完整CRPS；不平均CRPS冒充混合CRPS。

{chr(10).join(rows)}

MQS为15个指定分位点的两倍平均pinball。完整CRPS对模型的离散预测分布精确计算，独立CDF积分最大误差为{err:.2e}。改善为20个市场×期限单元的几何相对改善，正值更好。HGB仅给有限分位，未把网格分数称完整CRPS；其lookback250与候选128不同，不能把全部差异归给结构。

两seed初试曾得到约2.19%的经验基准MQS改善，五seed后下降到约1.24%，应采用较保守的五seed结论。full相对滚动均匀mark的五seed MQS改善约为[+1.968,+1.596,-0.631,+0.244,-0.429]%；部分单seed反向，当前正结论是预定五seed概率混合模型，而不是挑选最好seed。条件mark相对仅期限mark的收益约1.31%，支持继续研究状态条件的分布修正；price/calendar额外输入的收益仅约0.12%，不能作为主要创新。

当前收益并不覆盖所有市场/期限：广州和部分CEA/上海组合更明显，湖北较长期限及EU存在退步。下一轮用冻结配置进行滚动时期复核和正确深度学习基线比较，不把一次历史测试平均值等同于可发表保证。

## 已完成的结构与输入对照

1. [经验mark五seed报告](empirical_tilt_round3_followup_report_zh.md)：full/state/h-only同容量、同频率规则、同支持。逐seed和五seed混合全部保存，原始两seed源码和结果SHA保持一致。
2. [样条模型同输入控制](atom_spline_round3_statecontrols_report_zh.md)：state-only的神经atom相对continuous spline改善MQS0.3400%、完整CRPS0.3460%，两seed方向一致；但神经概率头未超过规则，固定BCE辅助项也没有主评分收益。规则atom相对经验MQS约1.2379%、完整CRPS0.8493%，可保留为替代分布表达。
3. [验证选择输入的样条结果](atom_spline_round3_contextselected_report_zh.md)：严格只按原inner记录选aug/state，各5个market/seed，重新合并同一CDF；相对经验MQS0.5947%、完整CRPS0.1351%。不能按测试较好的全state反向改名最终模型。
4. [输入探针](context_probe_round3_report_zh.md)：HGB对price/calendar联合包有小幅信息收益，长h更明显；无约束Ridge外推不稳。输入包收益不等于模型结构创新。
5. [经验分布强控制](empirical_bench_round3.json)：EWHS衰减、近期窗和KNN的参数只在inner验证选择；拟合边界后所有这些记忆均冻结，不用更晚test标签追加。

## 实现独立核对与近邻边界

独立代码审查未见跨cut标签误用、双重缩放、混合分布/CRPS口径错误或原点偷换。检索maturity mask、epoch选择与full-precut refit均对应真实代码。完整权重还用独立CDF区间积分核对，所有三版本概率分支完全相同，h-only在同h内的条件mark恒定性通过。

条件加权经验分布本身已有明确近邻：[Distributional Random Forests，JMLR2022](https://arxiv.org/html/2005.14458v3)。指数倾斜和CRPS也不是首创。当前候选应围绕碳价零收益频率与有符号标准化经验分布的特定神经修正结构、参数效率和组件实测增量提出贡献，不改名包装通用加权ECDF为新分布族。

## 下一轮

按 [第四轮执行方案](ROUND4_FROZEN_CANDIDATE_PLAN_ZH.md) 冻结五seed主配置，补齐LSTM/GRU/TCN/Transformer分位预测及QRF等适当基线，并在预先记录的滚动时期比较。结构消融聚焦响应位置/幅度统计与额外尾部统计，不继续堆砌没有净增益的概率门控或均值预算。

目前历史测试已被开发使用，全部结果保留探索性质；不宣称未触碰测试、首创通用分布族或所有市场都获益。整体目标保持进行中。
'''
    (P/'ROUND3_MODEL_PROGRESS_ZH.md').write_text(report,encoding='utf-8')
    state={'status':'ongoing','candidate':'fixed_rolling_zero_prior_plus_conditional_empirical_tilt',
           'candidate_source':'empirical_tilt_round3.py','five_seed_source':'empirical_tilt_round3_followup.py',
           'frozen_seeds':[11,22,33,44,55],'frozen_variant':'full',
           'primary_metric':'geometric market-horizon raw-return MQS ratio; exact CRPS secondary',
           'next_plan':'ROUND4_FROZEN_CANDIDATE_PLAN_ZH.md',
           'next':['correct_deep_sequence_quantile_baselines','QRF_conditional_distribution_baseline',
                   'three_rolling_origin_periods_frozen_architecture','location_scale_vs_tail_response_basis_ablation'],
           'comparison_summary':out,'goal_complete':False}
    (P/'ROUND3_NEXT_STATE.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary.to_string(index=False))


if __name__=='__main__':main()
