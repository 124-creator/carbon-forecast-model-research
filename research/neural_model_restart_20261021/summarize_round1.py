# -*- coding: utf-8 -*-
"""Generate a compact research progress record from actual round-one artifacts."""
import json
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parent
S = json.loads((P/'multiscale_round1.json').read_text(encoding='utf-8'))
E = json.loads((P/'marked_event_round1.json').read_text(encoding='utf-8'))
V = json.loads((P/'verify_routing.json').read_text(encoding='utf-8'))


def summarize():
    all_epochs = [f['best_epoch'] for m in S['markets'].values() for fs in m['fits'].values() for f in fs]
    dual_epochs = [f['best_epoch'] for m in S['markets'].values() for f in m['fits']['dual_route']]
    rows = []
    names = {'empirical_anchor':'经验分位数底座','gaussian_rw':'Gaussian随机游走',
             'ridge_quantile':'Ridge均值+残差分位数','hgb_quantile':'分位数梯度提升',
             'uniform':'固定均匀融合','horizon_only':'仅期限路由',
             'state_only':'仅状态路由','shared_route':'共享位置/间距路由',
             'single_scale':'同参数三分支单尺度'}
    for k,r in S['summary'].items():
        base=k.split('_vs_',1)[1]
        gain=100*(1-np.exp(r['mean_log_MQS_ratio']))
        rows.append(f"| {names.get(base,base)} | {gain:+.4f}% | {r['wins']}/{r['cells']} |")
    erows=[]
    for k,r in E['summary'].items():
        if not k.startswith('dual_head_vs_'):
            continue
        gain=100*(1-np.exp(r['mean_log_MQS_ratio']))
        erows.append(f"| {k.split('_vs_',1)[1]} | {gain:+.4f}% | {r['wins']}/{r['cells']} |")
    report = f'''# 模型创新实验：第一轮结构实现与消融

日期：2026-10-21。状态：继续实验；未宣告可发表成功。

## 多尺度位置—间距双路由

模型见 [multiscale_round1.py](multiscale_round1.py)，指标见 [multiscale_round1.json](multiscale_round1.json)。

每个预测原点读取250步历史，分别以1/5/20步块构成12步输入，即三个分支实际覆盖12/60/240步。这是分辨率与记忆跨度的联合变化，不称互不重叠的频率分量。每个尺度编码器与专家头输出同一h期目标的位置修正和正分位间距因子。

位置路由和间距路由分别由历史状态及期限产生。输出中位数由位置路由决定，正间距由另一组权重决定；从中位数向两侧累加间距，确保分位数不交叉。共享路由是精确嵌套对照，仅把两组权重绑在一起；同尺度控制保留三个分支和相同参数规模，但都读取最细尺度。

五市场、h=1/5/10/20、两种子11/22，六个网络版本共60个种子配置。验证选轮次后从同seed重新拟合所有测试前可用标签；原点、目标及评分在各臂一致。标准比较另有Gaussian随机游走、经验分位数、Ridge残差分位数和HistGradientBoosting分位数。

| 双路由的参照 | MQS几何平均改善 | 更好单元 |
|---|---:|---:|
{chr(10).join(rows)}

改善定义为100×[1-exp(mean(log(MQS_model/MQS_reference)))]，20个市场×期限等权；负值为退步。MQS为所列分位点的两倍平均pinball，不称精确连续CRPS。

对常见学习基线的改善可以保留为当前配置的实测结果，但对强经验底座几乎持平，对共享路由的增量也很小，尚不足将双路由作为已验证主创新。

## 训练诊断

全部60个配置中，{sum(e==0 for e in all_epochs)}个被验证集选为epoch0，双路由10个配置中有{sum(e==0 for e in dual_epochs)}个。双路由所选轮次为{dual_epochs}。因此，部分模型实际退回静态底座，另一些仅更新很少轮数；这不是“所有网络都被充分训练后仍不可能有效”的证据。下一步应检验更稳定的专家预训练、较低学习率、共同与局部结构共享，避免路由在专家尚未形成差异时没有学习信号。

[verify_routing.py](verify_routing.py) 的非平凡随机头检查通过：正间距、权重和为1、两组路由均接收梯度；把两组路由权重绑定后与共享版最大差异为{V['maximum_nested_case_difference']:.2e}。这只证明所称结构被实现，不证明泛化收益。

## 价格非零概率—幅度双解码

另一路代码与方法见 [marked_event_round1.py](marked_event_round1.py)、[marked_event_round1_report_zh.md](marked_event_round1_report_zh.md)，完整结果见 [marked_event_round1.json](marked_event_round1.json)。该模型直接学习h期收益的零质量与两个有符号中心的连续正态分量，不生成跨期限联合路径。h>1的零终点收益不等于期间没有价格变化。

| 动态双头的参照 | MQS几何平均改善 | 更好单元 |
|---|---:|---:|
{chr(10).join(erows)}

这一版本没有显示动态概率头相对固定概率头的主要MQS增量。已从保存参数补算 [Brier与log-loss结果](marked_event_round1_brier.md)：h=1时五市场均有小幅改善，按预测原点合并的Brier改善0.6183%、log-loss改善0.9792%；所有期限19/20单元改善。这支持概率头学到了一点条件信息，但主要MQS仍相对固定概率退步0.0805%，不能据此改写成总体胜利。下一步可改进概率与幅度分支的交互和联合训练目标，检查能否把组件收益传递到价格分布。两个实验使用不同历史窗，跨实验结果不直接归因为网络家族差别。

## 与近邻模型的关系

[Pathformer](https://arxiv.org/html/2402.05956v2) 已有趋势/季节驱动的多尺度路由，[TimeMixer](https://arxiv.org/html/2405.14616v1) 已有各尺度预测器融合，[TFT](https://arxiv.org/html/1912.09363v3) 已有多期限分位数和样本相关门控。这里可检验的结构差异是解除位置与分位间距对同一尺度权重的绑定；不能主张首创多尺度、状态路由或分位数MoE。当前增量仍不足，继续改机制和训练。

## 下一轮实验

1. 保留双路由候选，测试专家独立辅助损失与低学习率，使专家先形成有用差异，再检验解除两组权重绑定的收益。所有设置按训练内验证选择。
2. 概率—幅度模型增加由历史滚动非零比例提供的条件先验，并检验量化评分对齐的训练；概率头已有小收益但存在明显校准漂移，需把组件信息传递到价格分布。
3. 接入已有原始能源、金融因子，研究因素组自适应滞后及其与位置/不确定性任务的交互。数据位置见 [EXOG_SOURCE_CANDIDATES.json](EXOG_SOURCE_CANDIDATES.json)，先查覆盖和时点；仅从原始表重建过去可知特征，不复用历史加工矩阵的回填与划分标签。
4. 可考虑共享标准化规律、市场局部修正的训练方式，解决单市场路由样本不足，同时按统一日期限制跨市场信息时点。

所有新实验和输出继续写当前仓库，历史封存数据只读。旧归档中的“必须转评估论文”定位不作为当前研究方向依据；直接用户目标是模型创新。

本轮测试期在历史开发中已被反复使用，结果为探索性。不得据此宣称未触碰测试、预注册、总体统计显著或实验全部完成。
'''
    (P/'ROUND1_MODEL_PROGRESS_ZH.md').write_text(report,encoding='utf-8')
    meta={'status':'ongoing','multiscale_zero_epoch_configs':sum(e==0 for e in all_epochs),
          'multiscale_total_configs':len(all_epochs),'dual_route_best_epochs':dual_epochs,
          'new_neural_seed_configurations':60+30,
          'next':['expert_auxiliary_loss_and_low_learning_rate',
                  'rolling_probability_prior_and_quantile_aligned_training',
                  'exogenous_group_lag_interactions','date_safe_partial_market_sharing'],
          'exogenous_candidate_manifest':'EXOG_SOURCE_CANDIDATES.json'}
    (P/'ROUND1_NEXT_STATE.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(meta,ensure_ascii=False,indent=2))


if __name__=='__main__':
    summarize()
