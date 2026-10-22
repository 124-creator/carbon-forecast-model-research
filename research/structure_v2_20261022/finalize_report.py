"""Postprocess the single frozen run; never imports or refits its models."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(frame):
    names=list(frame.columns);lines=['| '+' | '.join(names)+' |','| '+' | '.join(['---']*len(names))+' |']
    for row in frame.itertuples(index=False,name=None):
        values=[('无定义' if pd.isna(x) else f'{x:.6f}') if isinstance(x,(float,np.floating)) else str(x) for x in row]
        lines.append('| '+' | '.join(values)+' |')
    return lines

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    m=json.loads((P/'manifest.json').read_text(encoding='utf-8'));assert m['status']=='completed'
    assert all(sha(P/name)==value for name,value in m['outputs'].items())
    before={name:sha(P/name) for name in m['outputs'] if name not in ['REPORT_ZH.md','A3_anomaly_origins.csv']}
    e=pd.read_csv(P/'pairwise_equal_asset.csv',dtype={'period':str},float_precision='round_trip')
    comp=pd.read_csv(P/'A_components.csv',float_precision='round_trip');mix=pd.read_csv(P/'mix_observations.csv',float_precision='round_trip');multi=pd.read_csv(P/'A_multistart.csv',float_precision='round_trip');shares=pd.read_csv(P/'annual_decomposition_shares.csv',float_precision='round_trip')
    gate=m['decisions_2025'];verdict='；'.join(f"{name}：h5{'通过' if gate[name]['by_horizon']['5'] else '未过'}、h22{'通过' if gate[name]['by_horizon']['22'] else '未过'}，{'保留' if gate[name]['retained'] else '淘汰'}" for name in ['A2_bounded','DM_pair'])
    lines=['# 碳ETF结构实验：固定A2与DM_pair','', '本固定程序未通过预定门槛，淘汰。'+verdict+'。只依据2025的h5/h22逐期限双基准及≥2/3资产门槛。','',f"18个保存基准+6个新配置，{m['counts']['common_origins']}共同原点、{m['counts']['all_predictions']}行。QLIKE差为候选减基准，MSE比<1更好；先逐资产再等资产平均。2024仅预热；2026是按协议单发的已见诊断，不构成确认，也不给淘汰程序二次机会。",'','主模型双基准完整比较（改善数均以3资产为分母）：']
    rows=[]
    for year in ['2025','2026']:
        for h in [5,22]:
            for name in ['A2_bounded','DM_pair']:
                z=e[(e.period==year)&(e.horizon==h)&(e.model==name)].set_index('benchmark');a=z.loc['EWMA97'];b=z.loc['GARCH11_t']
                rows.append({'年':year if year=='2025' else '2026已见诊断','H':h,'候选':name,'ΔQ对EWMA':a.delta_QLIKE_equal_asset,'MSE比EWMA':a.MSE_ratio_equal_asset,'EWMA改善数':int(a.assets_QLIKE_better),'ΔQ对GARCH':b.delta_QLIKE_equal_asset,'MSE比GARCH':b.MSE_ratio_equal_asset,'GARCH改善数':int(b.assets_QLIKE_better)})
    lines+=table(pd.DataFrame(rows));lines+=['','2025结构对照差异（2026已见诊断的全部差异保留于配对文件，不参与确认）：']
    rows=[]
    for h in [5,22]:
        for a,b in [('A2_bounded','K1_direct_CC'),('A2_bounded','A3_free_cross'),('A2_bounded','K2_no_cross'),('DM_pair','K3_no_DM')]:
            r=e[(e.period=='2025')&(e.horizon==h)&(e.model==a)&(e.benchmark==b)].iloc[0]
            rows.append({'年':'2025','H':h,'比较':f'{a}−{b}','ΔQLIKE':r.delta_QLIKE_equal_asset,'MSE比':r.MSE_ratio_equal_asset})
    lines+=table(pd.DataFrame(rows));lines+=['',f"A3越界18原点，其中非正10原点；10点的QLIKE无定义且未删除，含这些点的完整年/期QLIKE均值和配对差记无定义，不参与QLIKE比较，不按剩余点重算。另8个越界但正的预测QLIKE可定义，按原协议保留；MSE保留全部原点。详见[A3逐原点清单](A3_anomaly_origins.csv)。",'A3异常分组（未列出的资产/年/H均为0）：']
    counts=comp.groupby(['asset','year','horizon'])[['A3_bound_violation','A3_nonpositive']].sum().reset_index();counts=counts[counts.A3_bound_violation>0].rename(columns={'asset':'资产','year':'年','horizon':'H','A3_bound_violation':'越界','A3_nonpositive':'非正'})
    lines+=table(counts);lines+=['',f"A2全部f>0且满足界；rho夹取{int(comp.rho_clipped.sum())}个、|rho|>0.999饱和{int(comp.rho_saturated.sum())}个。246组固定零/全1/4两起点中，目标差>1e−8的2组如下（均选uniform）；所有目标和梯度已存，不声称全局最优："]
    for r in multi[multi.absolute_difference>1e-8].itertuples():lines.append(f"- {r.asset} h{r.horizon}，重训原点{r.refit_origin_index}（{r.refit_date}）：零起点目标{r.objective_zero:.12f}，uniform目标{r.objective_uniform:.12f}，差{r.absolute_difference:.12f}。")
    lines+=['','年度份额（隔夜²/日内²/cross，sum分量/sum CC方差，三项和1；完整年份另存）：']
    for asset,g in shares[shares.year.isin([2024,2025,2026])].groupby('asset',sort=False):lines.append(asset+'：'+'；'.join(f'{r.year} {r.gap:.6f}/{r.id:.6f}/{r.cross:.6f}' for r in g.itertuples())+'。')
    lines+=['','κ窗口分布（当前refit eps下各目标资产评分日期；完整分资产/年分布另存）：']
    rows=[]
    for w,g in mix.groupby('window'):
        v=g.kappa;rows.append({'w':int(w),'最小':v.min(),'P05':v.quantile(.05),'中位':v.median(),'P95':v.quantile(.95),'最大':v.max()})
    lines+=table(pd.DataFrame(rows));lines+=['', '解释边界：rho是归一化交叉二阶矩，非Pearson条件相关系数。两阶段用同一训练集的分量拟合值估theta，是有意的插件估计；存在生成回归元/在样本乐观，不是严格预序列stacking，也不声称联合最优。rho误设会使总方差有偏。','A2与A3同时改变边界、函数族、尺度、损失及惩罚几何，表中是整套交叉项建模策略差异，不能归因纯一致性约束。A3采用QLIKE分量+平方误差cross的混合损失。K2仅作置零诊断。','正分量的c=E[Y/expη]为训练期QLIKE乘性尺度精确最优解；真实目标不加eps。h1只验证已实现代数恒等式，不要求预测等于未来实现平方或预测均值之和的平方。','DM_pair固定KRBN/KCCA组成，是双资产幅度混合特征，仅比K3多3个斜率且共享训练行；κ为不含收益符号的幅度同步性，不是相关系数。自含项、敞口重叠及ridge收缩几何改变妨碍唯一新信息归因，不构成外生识别。','先例：Andersen, Bollerslev & Huang (2011), Journal of Econometrics 160(1):176–189，已分别建模日内连续、跳跃和隔夜方差并合成总波动预测。主任务只核验作者页摘要，未读全文。本实验仅定位“碳ETF应用+有界合成约束”，不声称提出分解或新框架。','2026不完整且此前已见；KEUA末段不能无条件当成持续EUA期货暴露。NW只作未调整探索性诊断，完整h1/2024、逐资产及全部对照正反结果均保留。',f"验收：全部{m['independent_validation']['all_saved_coefficient_predictions_rebuilt']}新预测经保存系数重建，最大相对差{m['independent_validation']['max_relative_forecast_difference']:.3g}<预定1e−9；1476训练记录通过。合成梯度最大误差{max(m['checks']['gradient_errors']):.3g}；原点/标签/κ/旧输入hash均通过，无重拟合。",'','- [完整预测](predictions_common.csv)、[逐模型摘要](summary.csv)、[逐资产配对](pairwise.csv)、[等资产配对](pairwise_equal_asset.csv)。','- [分量及rho](A_components.csv)、[两起点](A_multistart.csv)、[年度份额](annual_decomposition_shares.csv)、[κ分布](kappa_distribution.csv)、[独立重建验收](independent_validation.json)。','- [固定协议](PROTOCOL_ZH.md)、[单次实验脚本](structure_experiment.py)、[manifest](manifest.json)。']
    assert len(lines)<=80,len(lines)
    anomalies=comp[comp.A3_bound_violation|comp.A3_nonpositive].copy();anomalies['QLIKE_defined']=anomalies.A3>0;anomalies.to_csv(P/'A3_anomaly_origins.csv',index=False,float_format='%.17g',encoding='utf-8-sig')
    (P/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    assert before=={name:sha(P/name) for name in before}
    m['finalizer_sha256']=sha(Path(__file__));m['outputs']['REPORT_ZH.md']=sha(P/'REPORT_ZH.md');m['outputs']['A3_anomaly_origins.csv']=sha(P/'A3_anomaly_origins.csv');m['report_lines']=len(lines)
    if 'multistart_differences_gt_1e8' in m['counts']:m['counts']['multistart_differences_gt_1e_minus8']=m['counts'].pop('multistart_differences_gt_1e8')
    m['multistart_objective_difference_threshold']=1e-8
    (P/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
    print(verdict);print('Report lines:',len(lines));print('Manifest SHA256:',sha(P/'manifest.json'))
if __name__=='__main__':main()
