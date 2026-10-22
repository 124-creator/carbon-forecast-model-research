"""Generate the round report from persisted results and verify saved gate values."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import corrected_risk_experiment as b
import gated_risk_experiment as g
root=Path(__file__).resolve().parent
base=pd.read_csv(root/'summary.csv')
s=pd.read_csv(root/'gate_summary.csv')
pair=pd.read_csv(root/'gate_pairwise.csv')
agg=pd.read_csv(root/'gate_pairwise_equal_asset.csv')
params=pd.read_csv(root/'gate_parameters.csv')
pred=pd.read_csv(root/'gate_predictions_all.csv')
weights=pd.read_csv(root/'gate_weights.csv')
q=pd.read_csv(root/'gate_weight_summary.csv')
checks=0
for asset in b.ASSETS:
    d,r,r2,eps,xs,ma,ew=b.feature_data(asset)
    x=xs['QHAR_liquidity']
    for h in b.HS:
        for model in g.NAMES:
            ff=params[(params.asset==asset)&(params.horizon==h)&(params.model==model)].sort_values('origin_index')
            for f in ff.itertuples():
                pp=pred[(pred.asset==asset)&(pred.horizon==h)&(pred.model==model)&(pred.refit_origin_index==f.origin_index)].sort_values('origin_index')
                ids=pp.origin_index.to_numpy()
                scaler=tuple(np.array(json.loads(v)) for v in [f.shared_center,f.shared_sd,f.direct_center,f.direct_sd])
                own,rng,direct=g.transform(x[ids],scaler)
                eta,ww,_,_=g.components(np.array(json.loads(f.parameters)),own,rng,direct,model.endswith('dynamic'))
                np.testing.assert_allclose(np.exp(eta),pp.forecast,rtol=1e-10,atol=1e-15)
                saved=weights[(weights.asset==asset)&(weights.horizon==h)&(weights.model==model)&weights.origin_index.isin(ids)].sort_values('origin_index')
                np.testing.assert_allclose(ww,saved.gate_weight,rtol=1e-10,atol=1e-15)
                checks+=len(pp)
assert checks==len(pred)
(root/'gate_validation.json').write_text(json.dumps({'saved_gate_and_forecast_reconstruction':True,'checked_prediction_rows':checks},indent=2),encoding='utf-8')

def md(df):
    lines=['| '+' | '.join(str(c) for c in df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    for row in df.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.5f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(lines)

mean=s.groupby(['year','horizon','model']).delta_MA22.mean().unstack('model')
view=mean[['EWMA97','GARCH11_t','HAR_return','QHAR_return','QHAR_joint','QHAR_liquidity','QHAR_interaction','QHAR_gate_static','QHAR_gate_dynamic']].reset_index()
view.columns=['年','h','EWMA97','GARCH-t','HAR','QHAR-return','joint','liquidity','interaction','static gate','dynamic gate']
ab=agg[agg.model=='QHAR_gate_dynamic'].pivot(index=['year','horizon'],columns='benchmark',values='delta_QLIKE_equal_asset')[['EWMA97','QHAR_gate_static','QHAR_interaction','QHAR_liquidity']].reset_index()
ab.columns=['年','h','动态−EWMA97','动态−固定门','动态−交互','动态−liquidity']
per=s[s.model=='QHAR_gate_dynamic'][['asset','year','horizon','n','delta_MA22','MSE_pct_MA22']].copy()
for bench,col in [('EWMA97','delta_EWMA97'),('QHAR_gate_static','delta_static')]:
    v=pair[(pair.model=='QHAR_gate_dynamic')&(pair.benchmark==bench)][['asset','year','horizon','delta_QLIKE','NW_t']]
    per=per.merge(v.rename(columns={'delta_QLIKE':col,'NW_t':'NW_'+bench}),on=['asset','year','horizon'])
per=per[['asset','year','horizon','n','delta_MA22','delta_EWMA97','delta_static','NW_QHAR_gate_static','MSE_pct_MA22']]
per.columns=['资产','年','h','n','ΔMA22','ΔEWMA97','Δ固定门','NW t 对固定门','MSE%MA22']
per.to_csv(root/'gate_focused_asset_results.csv',index=False)
gq=q[(q.year==2025)&(q.model=='QHAR_gate_dynamic')][['asset','horizon','q05','median','q95']].copy();gq.columns=['资产','h','门权P05','门权中位数','门权P95']
base22=base[(base.year==2025)&(base.horizon==22)&base.model.isin(['EWMA97','QHAR_interaction'])][['asset','model','delta_MA22','MSE_pct_MA22']]
gate22=s[(s.year==2025)&(s.horizon==22)&(s.model=='QHAR_gate_dynamic')][['asset','model','delta_MA22','MSE_pct_MA22']]
check22=pd.concat([base22,gate22]).sort_values(['asset','model']);check22.columns=['资产','模型','ΔMA22','MSE%MA22']
path=lambda f:f'[{f}]({f})'
text='''# 风险模型研究：2026-10-22 本轮运行报告

后续追加授权的尺度归一化与自由尺度系数对照也已完成，详见[归一化补充报告](NORMALIZED_REPORT_ZH.md)。本页保留此前九模型与门控这一轮的结果与当时计划。

本轮已完成独立修正版九模型实验，以及新增共享斜率动态门控、固定门控消融。普通交互和动态门控均未形成跨阶段、跨期限稳定增益；这是当前候选的结果，不能推出价格不可预测或其他架构无效，也不表示整体研究已完成。

## 对象、目标与时间边界

KRBN、KEUA、KCCA 的 Yahoo OHLC/adjusted-close/volume；KRBN/KCCA 数据止于 2026-06-08，KEUA 止于 2026-03-20。各资产独立交易日期，不做日期 union。目标为 y(t,h)=mean(r[t+1]^2,...,r[t+h]^2)，r 为 adjusted-close 对数收益，h=1/5/22。原始目标不填充、不替换，零合法。

2024 为已看过 validation，2025 为已看过 development，2026 为已看过 diagnostic；没有干净 test，2026 不是全年。全部模型从 2024-01-02 首个可用原点开始，每 22 个资产原点 expanding refit；特征含当天 t，训练标签必须满足 s+h≤refit_t。跨日历年标签从训练和计分都剔除。评估原点包括 close 后所有当日信息，因此不是开盘前预测。

九模型 47,925 条总预测，45,657 条模型计分记录，对应 **5,073 个资产×期限×日期共同原点**，不能称为 4.6 万独立样本。加入两门控模型后共同计分记录为 55,803=5,073×11；期限之间、同期限重叠目标及资产之间也不独立。计分保留零目标：KRBN/KCCA 的 h1 各4个，KEUA 的 h1 66个、h5 2个。

## 模型与候选公式

基准：恰22日 MA22；λ=.94/.97 EWMA；真实 arch zero-mean GARCH(1,1)-Student-t，以 return×100 拟合、22原点期间固定参数过滤、解析平均未来h期条件方差；log HAR_return 用训练残差 Duan smearing。QHAR 用正 log-link 和 mean QLIKE + .01/2×标准化斜率平方和，截距不惩罚。

own_k=log(rolling mean r²+eps)，range_k=log(rolling mean Parkinson+eps)，k=1/5/22；eps 仅由 pre2024 mean r²×1e-4 冻结。joint 添加 range 三尺度及 log(5日 adjusted overnight gap²均值+eps)；liquidity 再添 rel=tanh(log((vol+1)/(rolling22medianvol+1)))、flat=1(high=low)；interaction 添加 rel×(range_k−own_k) 三项。

新增门控：

`log f = b0 + Σ_k b_k[(1−w_t)own_k+w_t range_k] + bg gap + bq rel + bf flat`

`w_t = sigmoid(a0+a1 rel+a2 flat)`；固定门消融设 a1=a2=0，但 a0 仍学习，q的直接项保留。每个训练窗内 own/range 同尺度共用中心和SD，其余输入分别标准化；标准化后的 rel/flat 也进入门。b0/a0 不惩罚，其余斜率 ridge=.01。初始门权 .2/.5/.8 三组仅按训练惩罚目标选；不按评估结果选。

## 三资产等权阶段结果

下表是各资产内同原点 QLIKE 差后再三资产等权平均，Δ=模型−MA22，负数更好。MA22 自身差为0。不是损失百分比，也没有跨资产合并推断。

'''+md(view)+'''

动态门控机制消融：下表负数代表动态门控较相应基准好。动态−固定门在9个阶段期限格子中仅3格为负；2025 h5/h22 仍不及 EWMA97，2026 全部期限不及普通交互。不能据此确立状态门机制。

'''+md(ab)+'''

## 逐资产、阶段与期限结果

MSE%MA22=100×MSE模型/MSE_MA22，100为相等；如356.60应解释为MSE高256.60%，不是高356.60%。NW t 用 lag=max(5,h−1)，符号与QLIKE差相同。未做多重检验调整；2026 KEUA h22仅32原点且lag=21，推断尤其不稳定。

'''+md(per)+'''

2025 h22 的大MSE比率主要出现在 KRBN、KEUA；KCCA 明显不同，不应用等资产均值概括所有资产：

'''+md(check22)+'''

## 门权与实现核验

2025 动态门权分位数如下。门权只是预测混合系数，不能用其曲线声称测量误差已识别。全阶段参数、逐日门权与分位数均已保存。

'''+md(gq)+'''

九模型1,312次拟合、门控1,476次初值拟合/492次选定拟合，均0优化失败flag；门控出现一条pandas未来版本兼容性弃用警告，不影响当前数字。门控非凸，三起点不保证全局最优。解析梯度有限差分误差小于2.5e-7。独立由原始数据重建全部目标与MA22、重算loss/summary、核验训练界限和共同原点；82个GARCH refit区间中段×3期限（246组）与arch重新固定参数过滤一致；保存参数重建全部10,650条门控预测和门权也通过。

数据限制：adjusted close修订历史不可得，无法证明历史调整因子是当时可见版本；日收益平方只是噪声较大的风险代理；flat不等于零收益；gap/range特征采用对数；cross-year训练剔除属于本轮明确的保守约定。独立核验 flat 日零成交量：KEUA 67/79=84.81%，KRBN 0/4、KCCA 0/1。

## 下一步假说（未运行）

先检查2025 KRBN/KEUA长期风险是否存在历史高波动与当前尺度失配，再考虑以每个历史原点当时可得的 EWMA97 s_s 归一化 y_s/s_s、own/range/gap，并令 f_t=s_t exp(g_t)。QLIKE仅相差log(s_t)，不是更换损失。需先查 scaled HAR/volatility normalization 等既有研究；不先称新颖，不保证会改进。本轮没有继续启动该实验，计划独立于门控胜负记录。

## 复现与产物

在本目录运行（指定Python环境，Python3.11.15/arch8.0.0）：

```powershell
& 'C:\\Users\\Dreamboat\\AppData\\Local\\hermes\\hermes-agent-v0.21.5\\venv\\Scripts\\python.exe' corrected_risk_experiment.py
& 'C:\\Users\\Dreamboat\\AppData\\Local\\hermes\\hermes-agent-v0.21.5\\venv\\Scripts\\python.exe' gated_risk_experiment.py
& 'C:\\Users\\Dreamboat\\AppData\\Local\\hermes\\hermes-agent-v0.21.5\\venv\\Scripts\\python.exe' validate_corrected_results.py
& 'C:\\Users\\Dreamboat\\AppData\\Local\\hermes\\hermes-agent-v0.21.5\\venv\\Scripts\\python.exe' build_round_report.py
```

'''
for desc,files in [
 ('原始九模型完整结果',['corrected_risk_experiment.py','predictions_all.csv','predictions_scored_common.csv','summary.csv','summary_equal_asset.csv','manifest.json']),
 ('门控与全部强基准比较',['gated_risk_experiment.py','gate_comparison_predictions_common.csv','gate_summary.csv','gate_pairwise.csv','gate_pairwise_equal_asset.csv','gate_manifest.json']),
 ('收敛、参数与门权',['fit_diagnostics.csv','gate_fit_diagnostics.csv','gate_parameters.csv','gate_weights.csv','gate_weight_summary.csv']),
 ('核验、更正与后续假说',['sample_audit.csv','independent_validation.json','gate_validation.json','CORRECTIONS_20261022.md','NEXT_CANDIDATE_NOTES.md'])]:
    text+='- '+desc+'：'+ '、'.join(path(f) for f in files)+'。\n'
(root/'ROUND_REPORT_ZH.md').write_text(text,encoding='utf-8')
print('REPORT COMPLETE; reconstructed gate rows',checks)
print(per.to_string(index=False))
print(gq.to_string(index=False))
