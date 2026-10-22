# -*- coding: utf-8 -*-
"""Record completed round4 experiments and the next concrete model hypothesis."""
import json
from pathlib import Path
import pandas as pd

P=Path(__file__).resolve().parent

def load(name):return json.loads((P/name).read_text(encoding='utf-8'))


def main():
    seq=load('seq_transport_round4.json');roll=load('rolling_tilt_round4.json')
    kernel=load('kernel_memory_round4.json');basis=load('tilt_basis_round4.json')
    verify=load('seq_transport_round4_verification.json')
    refs=['no_sequence_transport','seq_location_scale_only','seq_reweight_only','frozen_tilt','lstm_qr','qrf_original','rule_qrf','uniform_rule']
    labels=['无序列输运','仅位置和尺度输运','仅响应重加权','第三轮倾斜网络','LSTM-QR','原始QRF','相同规则概率QRF','相同概率的均匀经验分布']
    rows=[]
    for ref,label in zip(refs,labels):
        d=seq['summary']['seq_transport_vs_'+ref]
        a=d['MQS'];c=d.get('CRPS')
        rows.append({'reference':ref,'label':label,'MQS_gain_percent':a['geometric_gain_percent'],
                     'MQS_wins':a['wins'],'cells':a['cells'],
                     'CRPS_gain_percent':None if c is None else c['geometric_gain_percent']})
    pd.DataFrame(rows).to_csv(P/'ROUND4_NEURAL_CANDIDATE_SUMMARY.csv',index=False,encoding='utf-8-sig')
    table=['| 新输运模型的参照 | MQS改善 | 胜格 | 完整CRPS改善 |','|---|---:|---:|---:|']
    for r in rows:
        c='不由15q推定' if r['CRPS_gain_percent'] is None else f"{r['CRPS_gain_percent']:+.4f}%"
        table.append(f"| {r['label']} | {r['MQS_gain_percent']:+.4f}% | {r['MQS_wins']}/{r['cells']} | {c} |")
    rtable=['| 第三轮冻结模型的滚动参照 | MQS改善 | 完整CRPS改善 |','|---|---:|---:|']
    for ref,d in roll['summary'].items():
        rtable.append(f"| {ref} | {d['MQS']['geometric_gain_percent']:+.4f}% | {d['CRPS']['geometric_gain_percent']:+.4f}% |")
    zeros=sum(f['best_epoch']==0 for p in roll['periods'].values() for f in p['fits']['candidate'])
    text=f'''# 第四轮模型实验汇总

日期：2026-10-21。整体目标继续进行；已经取得更好的神经模型，但尚未超过最强条件分布基线。没有转评估类论文，也不把某一候选未胜出推广为ML/DL不可行。

## 本轮最值得继续的神经模型

LSTM条件位置/尺度与矩归一经验形状输运：过去64步收益及不变价标记、36维已知context进入LSTM16和context16，shared16输出4个有界系数。均匀成熟非零历史支持给参考m0/s0，响应重加权给w，输出值为mt+st*(z-mw)/sw，其中mt=m0+s0*delta、st=s0*exp(ell)。因此形状修改与非零部分的条件均值/尺度控制分开。

这不是此前未获额外增益的无条件方差budget。p采用滚动规则；真实非零概率按最后完整F计算。五seed合并各自变化的a*z+b支持及0原子后求完整概率mixture，不平均参数或CRPS冒充混合分布。

{chr(10).join(table)}

改善均按五市场×四期限20格的几何相对比计算，正值表示完整seq_transport更好。完整模型已比LSTM-QR和旧FF倾斜模型好一点，但比rule-QRF仍差约0.524%。广州和EU是主要差距；对rule-QRF，市场MQS改善约CEA+0.97%、广州-7.08%、上海+2.85%、湖北+1.94%、EU-1.61%。不按测试选择市场或删seed。

四组各五市场五seed，共100个内层选择/最终拟合配置，运行663秒。100个内层encoder均有非零梯度，58个非零epoch最终模型实际更新，42个合法选择epoch0。四组均分配2468参数。单seed组件收益有反向，ensemble的内部增量不能称每个seed都稳定有效。

实现核对：4307个原点×期限键和target/scale严格对齐；58个旧文件SHA未变。独立几何、CDF积分和gradcheck通过；检查点每模型12固定query重建a/b、真实mixture Q/CRPS最大误差均0。实际最小均匀/倾斜方差为{verify['min_actual_uniform_variance']:.6f}/{verify['min_actual_weighted_variance']:.6f}，floor触发0次，额外mark零碰撞0。

详见 [输运模型完整报告](seq_transport_round4_report_zh.md)、[检查点及逐seed核对](seq_transport_round4_verification_report_zh.md)。

## 强基线已经正确完成

- [四类序列基线](deep_seq_round4_report_zh.md)：LSTM/GRU/TCN/Transformer-QR共100个配置，全部真实训练、正确pinball、位置编码及encoder更新核对；内选轮后precut重拟合。LSTM比第三轮冻结候选MQS好0.8704%，GRU好0.4305%，不能再说旧模型已胜标准DL。
- [QRF](qrf_round4_report_zh.md)：五seed各80树，leaf只在inner从10/25/50选择，leaf内真实scalar响应给概率，保持零标签；完整CDF评分。其MQS比第三轮候选好1.3500%。
- [QRF相同概率控制](qrf_round4_probability_control_report_zh.md)：不重训，固定成与候选相同p_rule后，QRF优势反而增至MQS1.5220%、完整CRPS1.6594%。所以主要差距仍是条件非零分布，不是零概率估计。

## 三滚动时期

冻结第三轮配置，按每市场观测数65%/75%/85%截止，评估后126观测原点，5seed，15时期/3780个forecast keys。相对静态经验及EWHS有收益，未整体超过状态近邻。{zeros}/75个seed配置选择epoch0，是允许的参考回退，不能当作全部没训练。

{chr(10).join(rtable)}

这些是历史滚动探索，不叫新盲测。原点窗/实际cut日期在 [预先生成的manifest](rolling_tilt_round4_manifest.json)，完整数据在 [滚动结果](rolling_tilt_round4.json)。

## 已实际完成的其他模型尝试

- [响应因子消融](tilt_basis_round4.json)：完整5因子比仅位置/绝对幅度2因子MQS差0.2238%、完整CRPS差0.3390%，不支持继续堆高阶响应因子。
- [神经状态检索](kernel_memory_round4.json)：100个学习配置及固定控制，1078秒。固定p的状态kernel比旧模型MQS好0.0937%，但比response-only控制差0.1604%、比QRF差1.2735%；该设置不够。
- [CDF/零质量/期限共享森林](mixed_cdf_forest_round4_report_zh.md)：175个forest实际拟合。per-h CDF、pooled MSE、pooled CDF、pooled zero-CDF都未超过QRF；matched pooled CDF比pooled MSE差约2.00%MQS。额外zero坐标仅有约0.18%平均增量，不能包装为整体成功。跨h借样质量实际64%–75%，说明需要控制共享程度，不是未发生共享。CDF网格仅用于训练分裂近似；最终完整CRPS对真实weighted ECDF精确评分。

所有分支和seed保留，不把测试后挑出的最好seed/最好市场写成主结果。

## 下一轮具体改进

按 [第五轮森林参考输运方案](ROUND5_FOREST_TRANSPORT_PLAN_ZH.md)，将QRF条件非零参考替换当前均匀参考，再以LSTM做矩归一的分布残差修正。保留小比例uniform reservoir，防止参考和形状方差坍缩，并把reference_only列为明确控制，区分参考平滑与神经输运收益。森林参考阶段和NN阶段使用各自正确fit区间，训练检索继续屏蔽未成熟标签。

这沿用已出现小收益的输运层，针对已测得的条件分布差距继续改模型。可以并行用验证选定的log-horizon kernel限制过强跨期限借样，但主方向不改为评估论文。整体目标仍未完成。
'''
    (P/'ROUND4_MODEL_PROGRESS_ZH.md').write_text(text,encoding='utf-8')
    state=load('ROUND4_ACTIVE_STATE.json')
    state['completed_children']=state.pop('active_children')
    state['active_children']={}
    state['completed']['seq_transport_round4']='100 configs complete and verified; beats LSTM MQS .13663%, still behind rule-QRF .52423%'
    state['completed']['mixed_cdf_forest_round4']='175 forests complete; all four variants behind QRF; zero-CDF internal small improvement .1816%, excessive crossh borrowing 64-75%'
    state['next_plan']='ROUND5_FOREST_TRANSPORT_PLAN_ZH.md'
    state['next']='forest-conditioned reference plus sequence residual empirical transport; optional horizon borrowing kernel'
    state['objective_complete']=False
    (P/'ROUND4_ACTIVE_STATE.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    (P/'ROUND4_NEXT_STATE.json').write_text(json.dumps({'status':'ongoing','goal_round_completed':4,'objective_complete':False,
        'next_plan':state['next_plan'],'candidate':'forest-conditioned sequence empirical transport to implement',
        'best_neural_source':'seq_transport_round4.py','strong_baseline':'qrf_round4_probability_control',
        'parent_running_jobs':[],'active_children':{},'five_seeds':[11,22,33,44,55],
        'do_not':['pivot to evaluation paper','relabel known weighted ECDF as first distribution family','pick test-best seed/market'],
        'last_model_scores':rows},ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()
