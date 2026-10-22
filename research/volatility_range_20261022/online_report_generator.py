from pathlib import Path
import pandas as pd
import json

P=Path(__file__).resolve().parent

def table(d, digits=6):
    d=d.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c]=d[c].map(lambda v:f'{v:.{digits}f}')
    return '| '+' | '.join(map(str,d.columns))+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in d.itertuples(index=False,name=None))


def main():
    a=pd.read_csv(P/'online_full25_equal_asset.csv')
    p=pd.read_csv(P/'online_full25_pairwise.csv')
    w=pd.read_csv(P/'online_weight_summary.csv')
    manifest=json.loads((P/'online_manifest.json').read_text(encoding='utf-8'))
    part=a[(a.period.isin(['2025','2026'])) & (a.horizon>1) & a.model.isin(['PartialFeedback_dynamic','LeadBalanced_partial']) & (a.benchmark=='Online_dynamic')]
    mechanism=part[['period','horizon','model','delta_QLIKE_equal_asset','MSE_ratio_equal_asset','assets_QLIKE_better']]
    focus=a[(a.period.isin(['2025','2026'])) & (a.model=='Online_dynamic')]
    controls=focus[focus.benchmark.isin(['Online_expanding','Equal_geometric','Equal_arithmetic','Frozen2024'])].pivot(index=['period','horizon'],columns='benchmark',values='delta_QLIKE_equal_asset').reset_index()
    strong=focus[focus.benchmark.isin(['MA22','EWMA94','GARCH11_t','QHAR_interaction','N_gate_dynamic'])].pivot(index=['period','horizon'],columns='benchmark',values='delta_QLIKE_equal_asset').reset_index()
    individual=p[(p.period.isin(['2025','2026'])) & (p.horizon>1) & p.model.isin(['PartialFeedback_dynamic','LeadBalanced_partial']) & (p.benchmark=='Online_dynamic')][['asset','period','horizon','model','delta_QLIKE_negative_better','MSE_ratio','NW_normal_p_two_sided']]
    weights=w[(w.year>=2025)&(w.horizon==22)&(w.model=='Online_dynamic')][['asset','year','n','mean_w_EWMA97','mean_w_HAR_return','mean_w_QHAR_joint','mean_w_QHAR_liquidity_plus_scale']]
    text='''# 在线风险反馈组合：固定协议结果

结论：部分兑现反馈存在小幅、方向一致的等资产QLIKE机制信号，但不足以认定模型成功。h=5/22的四个“年份×期限”组合中，普通partial和lead均衡partial相对完整动态反馈均改善，每格仅2/3资产改善；逐资产Newey-West近似检验没有一个达到5%水平。h=22的MSE反而恶化。更根本的是动态full在2026明显落后于等权、冻结权重与GARCH，partial没有扭转这一结论。不能把相对较弱动态基线的小幅改善说成稳定强基线胜利。

2024仅作预热与探索；判断表只用2025和不完整2026。所有年度此前已见，不是未触碰的外部检验集。KRBN/KCCA原始数据到2026-06-08，KEUA到2026-03-20，各期限评分原点还须保留完整且不跨年目标；等资产平均给予每个资产相同质量，不能解释为等日数汇总。

## 部分兑现的机制比较

QLIKE差为候选减基准，负数更好。MSE比是先逐资产取候选/基准再等资产平均，小于1更好；资产改善数分母为3。

'''+table(mechanism)+'''

h=1 full与普通partial独立求解得到逐位相同预测/权重。lead均衡也独立求解，再以权重rtol/atol=1e-9、预测rtol=1e-10/atol=1e-15验证数学等价；没有复制full权重充作验收。h=22普通partial在2025/2026的平均MSE比分别为1.071573/1.206238，lead均衡为1.084623/1.167616。短lead均衡后QLIKE方向仍在，说明观察到的均值改善不完全由短lead质量偏重解释；样本内的小幅差异仍不足以确立迁移性或非平稳无偏性。

## 动态full相对基本组合控制

下表均为动态full减列中基准的等资产QLIKE差：

'''+table(controls)+'''

## 动态full相对全部模型中的强基准

'''+table(strong)+'''

2025相对GARCH各期限三资产均改善；2026 h=5对GARCH三资产均恶化，h=22只有1/3资产改善。2026 h=22普通partial/lead均衡相对GARCH的等资产QLIKE差仍分别为+0.716861/+0.714781。2025对EWMA94虽有负的等资产均值，各期限却只有1/3资产改善，不能只用均值描述普适胜利。完整25模型表同时包括原18模型和7个新组合，保留所有对照及MSE。

## 分资产机制结果与不确定性

Newey-West使用lag=max(5,h−1)，表中p为未调整的双侧正态近似；不把重叠目标、多个比较和短2026样本下的近似值当严格确认性证据。

'''+table(individual)+'''

## 权重的实际行为

以下为动态full的h=22年度平均权重，各行四权重之和为1。完整日级权重同时保留各候选、有效反馈量、共享窗口数、标签截止、拟合时间与状态，可核查变化是否由当时已兑现信息触发。

'''+table(weights)+'''

## 固定协议与审计

四专家固定为EWMA97、HAR_return、QHAR_joint、QHAR_liquidity_plus_scale；预测为log forecast的单纯形组合。full、partial与lead均衡共用最近252个s<t且原scored_common为真的已发布原点；full只取共享窗中的完整标签，不向更老原点补满。三者均以共享窗完整标签至少44个启动，之前等权，半衰期22原生交易日，ridge=.01。原始日收益平方来自既有feature_data，partial只读取s+j<=t的已发生分量，不由未来目标反推。

partial的每原点观测数m=min(h,t−s)，响应knownsum/m、质量decay*m/h，等价于规定的部分QLIKE贡献除以总兑现质量。lead均衡对每j将已兑现原点的衰减质量归一为1/h，再构造a与q/a。各lead质量、sum(a)=1、全m=h时随机及实际窗口的objective/gradient退化full均已断言。

扩展控制用全部成熟标签等权。Frozen2024在首个2025共同原点只用已成熟2024反馈拟合，h1/5/22分别251/247/230条，此后冻结到2026；2024仅等权预热，未回填拟合后的历史权重。

首次尝试在任何权重拟合前被输入目标逐位唯一断言拦下：34个原点的CSV往返表示末位不同，最大3.3881317890172014e−21。核对后统一同原点MA22记录的目标，不改真实目标、原文件或评分样本；训练的完整原始r²均值也与保存目标紧容差核对。随后完成固定协议运行；又按事前新增的数值验收要求将lead均衡h1改为独立求解并验证，参数和h5/h22预测不变。没有依据结果调整专家、窗口、衰减或罚项。

'''+f"最终成功优化{manifest['counts']['optimizations']}次，失败{manifest['counts']['failures']}次；5073个共同原点，7模型35511条新预测，25模型126825条完整对照。最大单纯形一阶最优性间隙{manifest['max_simplex_first_order_gap']:.3g}；输入hash核验未变。\n"+'''
这是已有online aggregation家族中的机制实验；部分风险反馈和lead均衡仅称候选，仍需独立查重，不作新颖性声明。

## 输出

- [固定计划](online_risk_combination_plan.md)、[在线脚本](online_risk_combination.py)、[主manifest](online_manifest.json)。
- [25模型共同预测](online_full25_predictions_common.csv)、[年度和合并期间摘要](online_full25_summary.csv)、[全部配对比较](online_full25_pairwise.csv)、[等资产配对比较](online_full25_equal_asset.csv)、[25模型manifest](online_full25_manifest.json)。
- [原11模型机制表](online_comparison_predictions_common.csv)、[日级权重](online_weights.csv)、[权重年度摘要](online_weight_summary.csv)、[优化诊断](online_fit_diagnostics.csv)。
'''
    (P/'online_risk_combination_report.md').write_text(text,encoding='utf-8')
    print('Saved complete Chinese online report')

if __name__=='__main__':
    main()
