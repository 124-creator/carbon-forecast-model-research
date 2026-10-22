"""Finalize five-expert tables/report only; never refit or alter predictions."""
from pathlib import Path
import json
import pandas as pd
import online_five_experiment as five
from online_report_generator import table

P=Path(__file__).resolve().parent
ALIASES={'Five_dynamic_full':'动态full','Five_dynamic_partial':'动态partial','Five_dynamic_LB':'动态LB',
         'Five_expanding_full':'扩展full','Five_expanding_partial':'扩展partial','Five_expanding_LB':'扩展LB',
         'Five_equal_geometric':'五等权几何','Five_equal_arithmetic':'五等权算术'}


def main():
    manifest_path=P/'online_five_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    before={name:five.previous.digest(P/name) for name in ['online_five_predictions.csv','online_five_full35_predictions.csv','online_five_weights.csv','online_five_fit_diagnostics.csv']}
    allp=pd.read_csv(P/'online_five_full35_predictions.csv',float_precision='round_trip')
    summary,pairs,a=five.summarize(allp)
    for name,frame in [('online_five_summary.csv',summary),('online_five_pairwise.csv',pairs),('online_five_equal_asset.csv',a)]:
        frame.to_csv(P/name,index=False,float_format='%.17g',encoding='utf-8-sig')
    assert before=={name:five.previous.digest(P/name) for name in before}
    assert manifest['input_hashes']=={name:five.previous.digest(P/name) for name in manifest['input_hashes']}
    f=a[a.period.isin(['2025','2026'])&(a.horizon>1)].copy()
    ownmask=(f.model.isin(['Five_dynamic_partial','Five_dynamic_LB'])&(f.benchmark=='Five_dynamic_full'))|(f.model.isin(['Five_expanding_partial','Five_expanding_LB'])&(f.benchmark=='Five_expanding_full'))
    own=f[ownmask][['period','horizon','model','delta_QLIKE_equal_asset','MSE_ratio_equal_asset','assets_QLIKE_better']].copy()
    own['model']=own.model.map(ALIASES)
    own.columns=['年','H','候选','QLIKE差对各自full','MSE比对各自full','改善资产数']
    strong=[]
    for (year,h,model),g in f.groupby(['period','horizon','model']):
        by=g.set_index('benchmark')
        def delta(b):
            return 0. if model==b else float(by.loc[b,'delta_QLIKE_equal_asset'])
        strong.append({'年':year,'H':h,'配置':ALIASES[model],'QLIKE差对GARCH':delta('GARCH11_t'),
                       'MSE比对GARCH':float(by.loc['GARCH11_t','MSE_ratio_equal_asset']),
                       'QLIKE差对五等权几何':delta('Five_equal_geometric'),'QLIKE差对五等权算术':delta('Five_equal_arithmetic')})
    text='''# 五专家最后固定实验：机制信号未成为稳健模型优势

结论：加入已保存GARCH11_t后，2026 h22部分反馈相对full的改善增强，四个partial/LB候选均改善3/3资产；但2025出现明确反例，五专家学习组合在2026仍不敌GARCH及五专家等权组合。专家可用性没有自动转化为样本外优势。本轮结果不支持把该机制认定为跨年份、跨学习器的稳定增益，更不支持宣称成功战胜强基准。

事前只固定8配置：扩展窗full/partial/LB、动态窗full/partial/LB、等权几何/算术。专家为原EWMA97、HAR_return、QHAR_joint、QHAR_liquidity_plus_scale加GARCH11_t；只复用保存预测。扩展窗使用全部已发布s<t原点，无衰减；动态三者共用最近252个已发布原点，半衰期22原生交易日，不给full补旧标签。三者各自共享44条完整反馈启动，单纯形五维权重，正则.01/2‖w−(1/5)‖²。

partial令m=min(H,t−s)，训练响应knownsum/m、质量d_s m/H。LB令Z_j=Σ_{s:s+j<=t}d_s，mass_sj=d_s/(H Z_j)，a_s=Σ_j mass_sj，响应Σ_j mass_sj r²[s+j]/a_s；只读取已发生分量。没有改其他参数、启动或评分样本。

下面QLIKE差为候选减基准，负数更好；MSE比小于1更好。均先逐资产计算再等资产平均，MSE比随分母变化，不同基准下的平均比值不具传递性；改善资产数分母3。

'''+table(own)+'''

2025动态h5两候选均恶化、0/3资产改善；2025动态h22的LB也恶化，扩展h22两候选的等资产均值都恶化。2026 h22动态partial/LB虽改善QLIKE，MSE比却是1.067083/1.051123；扩展版本则分别0.985093/0.986436。不能只挑2026 h22给出正面结论。

强基准与五等权如下；所有其他原模型、四专家组合、MSE和h1逐资产明细完整保留在35模型CSV。

'''+table(pd.DataFrame(strong))+'''

加第五专家本身也不构成稳定改善。2026 h22，五专家动态full相对四专家动态full的QLIKE差+0.019638；五专家扩展full相对四专家扩展full为+0.049915、MSE比1.263022。五专家动态partial相对四专家partial改善−0.054925，而五专家扩展partial相对四专家扩展partial恶化+0.039929。五等权算术相对原四等权算术h22的QLIKE改善−0.129300，却有MSE比1.074321。没有一种简单的“增加GARCH必胜”解释。

权重上，2026 h22动态full的GARCH等资产年度平均权重约0.1251，动态partial约0.2600；扩展full/partial约0.1708/0.1619。部分反馈确实改变了权重，但更大权重或更早反馈本身不能代替强基准检验。

统计边界：在2025/2026 h5/h22各候选对本学习器full的比较中，有实质差异的未调整p<.05仅在扩展候选的KCCA 2026 h22（partial/LB约.0081/.0125）和KEUA 2026 h5（约.0180/.0178）。每候选12个“资产×年×期限”格即使仅做Bonferroni校正，最小p也不能过.05。动态候选没有实质差异p<.05。NW沿用lag=max(5,H−1)、双侧正态近似，未当成确认性证据。

后处理另发现KCCA 2026 h5扩展候选与full的预测相对差仅3.6e−15/5.3e−15、损失差约−3e−16，却因极小标准误产生原始p=.039/.033。已明确逐日损失差max≤1e−12为numerical_tie，统计量记0、p记1，并在CSV保留原始统计列；没有改预测、权重、拟合或参数，也没有重训。

'''+f"验收：{manifest['counts']['common_origins']}共同原点，8配置{manifest['counts']['new_predictions']}条新预测，35模型{manifest['counts']['all_predictions']}行；{manifest['counts']['optimizations']}次优化、0失败，最大单纯形一阶间隙{manifest['max_simplex_first_order_gap']:.3g}。五维h1各学习器的full/partial/LB均独立优化；扩展最大权重差{manifest['h1_max_errors']['expanding']['max_weight_absolute']:.3g}、预测相对差{manifest['h1_max_errors']['expanding']['max_forecast_relative']:.3g}，动态分别{manifest['h1_max_errors']['dynamic']['max_weight_absolute']:.3g}/{manifest['h1_max_errors']['dynamic']['max_forecast_relative']:.3g}。已通过无未来分量、共享启动、lead质量、sum a=1、全成熟目标/梯度一致、所有35模型同原点和旧输出hash不变验收。\n"+'''

2024只作预热；2025及不完整2026此前已见，资产末端不齐。此前两轮负结果与报告保留；本轮为当前ETF池最后一次改动，停止遍历专家池、温度或半衰期。不作新颖性、非平稳无偏性或稳健优势声明。

KEUA暴露限制：[KraneShares官方页面](https://kraneshares.com/keua/)当前展示的Top 10 Holdings标注Data as of 03/13/2026、Cash 100.00%、资产4,565,861美元，Growth图也止于03/13/2026。这是官方页面现状，不是已认证的历史时点可得记录；未核实清盘公告，不据此声称清盘。KEUA 2026末段不能无条件解释为持续EUA期货暴露。全部原样本及负结果保留，没有事后剔除KEUA来改善成绩。

- [事前计划](online_five_plan.md)、[实验脚本](online_five_experiment.py)、[无重训后处理](online_five_finalize.py)、[manifest](online_five_manifest.json)。
- [35模型共同预测](online_five_full35_predictions.csv)、[完整QLIKE/MSE摘要](online_five_summary.csv)、[全部逐资产配对](online_five_pairwise.csv)、[等资产配对](online_five_equal_asset.csv)。
- [日级权重](online_five_weights.csv)、[权重年度摘要](online_five_weight_summary.csv)、[优化诊断](online_five_fit_diagnostics.csv)。
- [原25模型报告](online_risk_combination_report.md)、[四专家扩展迁移报告](online_expanding_transfer_report.md)。
'''
    (P/'online_five_report.md').write_text(text,encoding='utf-8')
    manifest.setdefault('fit_script_sha256',manifest['script_sha256'])
    manifest['script_sha256']=five.previous.digest(five.__file__)
    manifest['finalizer_sha256']=five.previous.digest(__file__)
    manifest['numerical_tie_postprocessing']={'rule':'max absolute daily QLIKE difference <=1e-12: statistic0 p1, raw retained','predictions_weights_fits_unchanged':True,'reason':'Floating-point ties otherwise generated false small p values; no refit or parameter change'}
    manifest['outputs'].update({name:five.previous.digest(P/name) for name in ['online_five_summary.csv','online_five_pairwise.csv','online_five_equal_asset.csv','online_five_report.md']})
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FIVE FINALIZED: report saved; numerical ties flagged; predictions and old outputs unchanged')


if __name__=='__main__':
    main()
