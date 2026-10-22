"""Verify normalization/reconstruction and build a separate supplemental report."""
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import numpy as np
import pandas as pd
import corrected_risk_experiment as b
import gated_risk_experiment as g
import normalized_risk_experiment as nn
root=Path(__file__).resolve().parent
new=pd.read_csv(root/'normalized_predictions_all.csv')
summary=pd.read_csv(root/'normalized_summary.csv')
pair=pd.read_csv(root/'normalized_pairwise.csv')
agg=pd.read_csv(root/'normalized_pairwise_equal_asset.csv')
rp=pd.read_csv(root/'normalized_regression_parameters.csv')
gp=pd.read_csv(root/'normalized_gate_parameters.csv')
old=pd.read_csv(root/'predictions_all.csv')
scored=new[new.scored_common]
np.testing.assert_allclose(scored.qlike,np.log(scored.forecast)+scored.target/scored.forecast,rtol=1e-10,atol=1e-11)
np.testing.assert_allclose(scored.squared_error,(scored.forecast-scored.target)**2,rtol=1e-10,atol=1e-14)
prior=old[old.model=='MA22'][['asset','origin_index','horizon','target','scored_common']]
comp=new.merge(prior,on=['asset','origin_index','horizon'],suffixes=('','_old'),validate='many_to_one')
np.testing.assert_allclose(comp.target,comp.target_old,rtol=5e-15,atol=1e-20,equal_nan=True)
assert comp.target.eq(0).equals(comp.target_old.eq(0))
max_target_csv_roundtrip_difference=float(np.nanmax(np.abs(comp.target-comp.target_old)))
assert comp.scored_common.equals(comp.scored_common_old)
groups={key:frame.sort_values('origin_index') for key,frame in new.groupby(['asset','horizon','model','refit_origin_index'])}
checked=0;earlyzero={}
for asset in b.ASSETS:
    d,r,r2,eps,xs,ma,ew=b.feature_data(asset)
    scale=ew[.97];xx=nn.normalized_features(xs,scale)
    earlyzero[asset]=np.flatnonzero(scale==0).tolist()
    finite=np.logical_and.reduce([np.isfinite(v).all(axis=1) for v in xs.values()])
    assert (scale[finite]>0).all()
    for f in rp[rp.asset==asset].itertuples():
        pp=groups[(asset,f.horizon,f.model,f.origin_index)];ids=pp.origin_index.to_numpy()
        mu=np.array(json.loads(f.mu));sd=np.array(json.loads(f.sd));beta=np.array(json.loads(f.parameters))
        zz=np.column_stack([np.ones(len(ids)),(xx[f.model][ids]-mu)/sd])
        pred=np.exp(zz@beta)
        if f.model!=nn.CONTROL:pred*=scale[ids]
        np.testing.assert_allclose(pred,pp.forecast,rtol=1e-10,atol=1e-14)
        np.testing.assert_allclose(scale[ids],pp.risk_scale,rtol=1e-10,atol=1e-14)
        checked+=len(ids)
    for f in gp[gp.asset==asset].itertuples():
        pp=groups[(asset,f.horizon,f.model,f.origin_index)];ids=pp.origin_index.to_numpy()
        scaler=tuple(np.array(json.loads(v)) for v in [f.shared_center,f.shared_sd,f.direct_center,f.direct_sd])
        own,rng,direct=g.transform(xx['NQHAR_liquidity'][ids],scaler)
        eta,weight,_,_=g.components(np.array(json.loads(f.parameters)),own,rng,direct,f.model.endswith('dynamic'))
        np.testing.assert_allclose(scale[ids]*np.exp(eta),pp.forecast,rtol=1e-10,atol=1e-14)
        checked+=len(ids)
assert checked==len(new)
check={'saved_all_normalized_forecasts_reconstructed':True,'checked_rows':checked,'original_targets_numerically_unchanged':True,'max_target_CSV_roundtrip_abs_difference':max_target_csv_roundtrip_difference,'zero_target_mask_exactly_unchanged':True,'original_scoring_flags_exactly_unchanged':True,'early_zero_EWMA_indices':earlyzero,'zero_scale_origins_with_valid_original_features':0,'original_loss_recomputed':True}
(root/'normalized_validation.json').write_text(json.dumps(check,indent=2),encoding='utf-8')

w=pd.concat([pd.read_csv(root/'gate_weights.csv'),pd.read_csv(root/'normalized_gate_weights.csv')],ignore_index=True)
w=w[w.scored_common]
saturation=w.groupby(['asset','year','horizon','model']).gate_weight.agg(n='size',q05=lambda a:a.quantile(.05),median='median',q95=lambda a:a.quantile(.95),fraction_lt_001=lambda a:(a<.01).mean(),fraction_gt_099=lambda a:(a>.99).mean(),fraction_extreme=lambda a:((a<.01)|(a>.99)).mean()).reset_index()
saturation.to_csv(root/'all_gate_saturation.csv',index=False)

def md(df):
    lines=['| '+' | '.join(str(c) for c in df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    for row in df.itertuples(index=False,name=None):lines.append('| '+' | '.join(f'{v:.5f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(lines)

view=summary.groupby(['year','horizon','model']).delta_MA22.mean().unstack()[['EWMA97','QHAR_liquidity_plus_scale','NQHAR_return','NQHAR_joint','NQHAR_liquidity','NQHAR_interaction','N_gate_static','N_gate_dynamic']].reset_index()
view.columns=['年','h','EWMA97','plus_scale','N-return','N-joint','N-liquidity','N-interaction','N-static','N-dynamic']
checks=[]
for year in [2024,2025,2026]:
    for h in [1,5,22]:
        def val(model,bench):return float(agg[(agg.year==year)&(agg.horizon==h)&(agg.model==model)&(agg.benchmark==bench)].delta_QLIKE_equal_asset.iloc[0])
        checks.append({'年':year,'h':h,'N-liq−EWMA97':val('NQHAR_liquidity','EWMA97'),'N-liq−原liq':val('NQHAR_liquidity','QHAR_liquidity'),'N-liq−plus_scale':val('NQHAR_liquidity',nn.CONTROL),'N-dynamic−N-static':val('N_gate_dynamic','N_gate_static'),'N-dynamic−原dynamic':val('N_gate_dynamic','QHAR_gate_dynamic')})
checks=pd.DataFrame(checks)
focus=summary[(summary.year==2025)&(summary.horizon==22)&summary.model.isin(['EWMA97','QHAR_liquidity','QHAR_liquidity_plus_scale','QHAR_gate_dynamic']+nn.NAMES)].drop_duplicates(['asset','model'])[['asset','model','n','delta_MA22','MSE_pct_MA22']].copy()
focus.columns=['资产','模型','n','ΔMA22','MSE%MA22']
full=pair[((pair.model=='NQHAR_liquidity')&pair.benchmark.isin(['EWMA97',nn.CONTROL]))|((pair.model=='N_gate_dynamic')&(pair.benchmark=='N_gate_static'))].copy()
full.to_csv(root/'normalized_focused_asset_results.csv',index=False)
q=saturation[(saturation.year==2025)&saturation.model.isin(['QHAR_gate_dynamic','N_gate_dynamic'])][['asset','horizon','model','q05','median','q95','fraction_lt_001','fraction_gt_099']].copy()
q.columns=['资产','h','模型','P05','中位','P95','w<.01比例','w>.99比例']
summary2025=summary[(summary.year==2025)&(summary.horizon==22)&summary.model.isin(['EWMA97','QHAR_liquidity','QHAR_liquidity_plus_scale','NQHAR_liquidity','N_gate_dynamic'])]
print('VALIDATION '+json.dumps(check),flush=True)
print(checks.to_string(index=False),flush=True)
print(summary2025[['asset','model','delta_MA22','MSE_pct_MA22']].to_string(index=False),flush=True)
print(q.to_string(index=False),flush=True)
text='''# 尺度归一化追加实验：2026-10-22

本报告补充[前一轮完整报告](ROUND_REPORT_ZH.md)。在不覆盖九模型与门控结果的前提下，新增四个归一化QHAR、两种归一化门控、一个自由尺度系数公平对照；没有查看或下载新日期的数据。

## 当前判断

归一化缓解了2025 h22的部分MSE失配，但未形成稳定收益：原liquidity→N-liquidity的MSE%MA22，KRBN从539.02降到345.01，KEUA从791.86降到147.37，KCCA却从55.80升到79.59。自由尺度系数对照在同三资产分别为327.40、168.41、54.60，表明收益不能只归因于齐次约束。N-liquidity对plus_scale的等资产QLIKE差在9个阶段期限格中8格为正；2026 h22差+0.26324，约束缺乏稳定优势。

归一化动态门对同归一化固定门9格仅2格QLIKE差为负。2025 h22动态门对EWMA97等资产差−0.00261，资产内却是KRBN +0.04267（NW t=2.814）、KEUA +0.01776（t=1.125）、KCCA −0.06826（t=−1.056）；三个资产MSE均比EWMA97高，分别高64.37%、12.49%、27.60%。2026 h22动态门对EWMA97等资产差+0.07461，对原动态门差+0.57682，跨阶段代价明显。

归一化后2025 KRBN三个期限w>.99的比例均100%；KEUA h22 w<.01比例100%、h5为77.05%；KCCA h22 w>.99为90.35%。这主要是极端选择代理，不能宣传动态可靠性融合。固定门消融不支持动态增益，**应撤下本轮流动性gate的核心机制主张**，保留失败候选和诊断以追溯，不宣告所有风险模型研究结束。极小的1e−7量级静动态差可能仍给出较大NW t，属于几乎相同饱和预测，不解释为经济上有意义的机制证据。

## 参数化与公平性

每个历史原点s使用当时可得的EWMA97风险尺度S_s，训练响应Y_s/S_s；own/range/gap的原log(mean+eps)减log(S_s)，rel/flat不变，interaction中range−own差不变；最终f_t=S_t exp(g_t)。目标CSV始终保留原始Y，零目标没有替换，尺度也没有floor。

QLIKE log(f)+Y/f 与归一化目标的QLIKE仅相差已知log(S)，但这不意味着和原模型同一假设类。按未标准化系数展开，liquidity候选是 `log f=b0+Σ_{全部7个log代理} b_k log(V_k)+(1−Σ_{全部7个log代理} b_k)log(S)+q直接项`，求和包括gap，归一化带来了offset/齐次系数约束。标准化后ridge几何也改变，不能把全部差异归于单一机制。共享斜率动态门控在含gap系数时同样遵循这种展开。

公平对照 QHAR_liquidity_plus_scale 使用原9个liquidity特征再加log(S)，在原始Y上QLIKE训练，无固定offset、尺度系数自由。NQHAR_liquidity与它的比较有助区分新增EWMA尺度信息和约束的作用。N_gate_dynamic须与同归一化N_gate_static比较，不能只比较旧模型。

沿用同一expanding窗口、每22个原点refit、训练标签s+h≤refit_t、跨年训练与计分剔除、预2024 epsilon、训练内标准化、ridge=.01与三组门权初始化。2024 validation、2025 development、2026 diagnostic均已看过，没有干净测试。

## 三资产等权QLIKE差

Δ=模型−MA22，负数更好，不报QLIKE百分比。每格先算各资产共同原点内差，再三资产等权平均。

'''+md(view)+'''

机制对照差如下，仍是负数更好：

'''+md(checks)+'''

## 2025 h22：逐资产检查尺度失配

MSE%MA22为100×比率；应减100才是相对变化百分比。不同资产必须分别解释，不能用均值代替资产结论。

'''+md(focus)+'''

## 门控饱和诊断

以下列出2025原动态门与归一化动态门的分位数和极端比例；全部阶段、两种静态门也保存在[完整饱和诊断](all_gate_saturation.csv)。门权接近1意味着几乎只选range、接近0意味着几乎只选own；不自动支持融合或状态自适应。原KRBN2024三个h、KEUA2024/25 h5/h22、KCCA2026 h22有明显饱和。没有事后给a0加惩罚来制造曲线；参数图形不识别测量误差。

'''+md(q)+'''

## 实现与核验

第一次执行完成KRBN后，被对“全部早期日期尺度>0”的过严断言中断。KEUA最早索引1/2/3的EWMA为0，但没有完整22日特征，从未进入原训练/评分样本。修正为在原共同有效特征日期断言尺度>0，其他早期logscale保持无效NaN；没有新增删样本、改目标或补尺度。修复后重跑，原计分标志和零目标标志完全一致；目标CSV往返最大绝对差3.39e-21，为浮点表示误差，没有实质变动。

新增7模型37,275条总预测；18模型91,314条计分记录=5,073个资产×期限×日期共同原点×18，不是9万独立样本。1,230次线性QHAR/自由尺度对照拟合、1,476次门控初值拟合、492次最终门控拟合。优化flag计数见[manifest](normalized_manifest.json)；全部37,275条新预测由保存参数独立重建通过，QLIKE/MSE重算通过，归一化QLIKE恒等式通过。NW lag=max(5,h−1)，未做多重比较调整，2026样本尤其有限。

## 研究解释与下一步边界

尺度×修正的构造已有component multiplicative-error、volatility component、GARCH-MIDAS等相关原理，不能称新family，也不能仅靠relative-risk改名声称创新。当前epsilon来自pre2024 mean r²，整体伸缩时应随风险单位变动；不要把固定epsilon版本宣称严格等变。归一化后是否保留流动性gate，应依据同归一化静态消融和饱和诊断，而不是模型名字。

KRBN/KCCA档案截至2026-06-08，而当前是2026-10-22。若之后真实数据可取得，可在冻结候选和比较规则之后构成新增时间外验证；目前没有下载或查看这些日期，也不保证可取得。KEUA档案止3月，不能假设能补齐。应先完成当前诊断和候选冻结，再决定拉取新数据。

## 复现与产物

先保留上一轮两脚本的结果，再在同一目录与同一Python环境运行：

```powershell
& 'C:\\Users\\Dreamboat\\AppData\\Local\\hermes\\hermes-agent-v0.21.5\\venv\\Scripts\\python.exe' normalized_risk_experiment.py
& 'C:\\Users\\Dreamboat\\AppData\\Local\\hermes\\hermes-agent-v0.21.5\\venv\\Scripts\\python.exe' build_normalized_report.py
```

- [新实验脚本](normalized_risk_experiment.py)、[manifest](normalized_manifest.json)、[新模型全部预测](normalized_predictions_all.csv)、[18模型共同计分预测](normalized_comparison_predictions_common.csv)。
- [全部18模型资产/年/期限汇总](normalized_summary.csv)、[逐基准差与NW/MSE比率](normalized_pairwise.csv)、[等资产描述汇总](normalized_pairwise_equal_asset.csv)、[关键资产消融](normalized_focused_asset_results.csv)。
- [线性拟合诊断](normalized_regression_fit_diagnostics.csv)、[线性参数](normalized_regression_parameters.csv)、[门控拟合诊断](normalized_gate_fit_diagnostics.csv)、[门控参数](normalized_gate_parameters.csv)、[逐日门权](normalized_gate_weights.csv)、[饱和诊断](all_gate_saturation.csv)、[独立核验](normalized_validation.json)。
'''
(root/'NORMALIZED_REPORT_ZH.md').write_text(text,encoding='utf-8')
