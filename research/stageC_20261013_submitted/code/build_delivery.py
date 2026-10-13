from pathlib import Path
import json,hashlib,sys,platform,shutil,csv
import numpy as np,pandas as pd,scipy,sklearn,torch
R=Path(__file__).resolve().parents[1];O=R/'outputs'
summary=pd.read_csv(O/'check_summary.csv').set_index('model');per=pd.read_csv(O/'check_per_year_horizon.csv');syn=pd.read_csv(O/'synthetic_summary.csv');audit=json.loads((R/'metadata/acquisition_audit.json').read_text());quality=json.loads((O/'quality_audit.json').read_text())
name={'RW':'随机游走','drift':'局部漂移（240步）','Ridge_level':'相对水平 Ridge','Ridge_diff':'差分 Ridge','SVR':'RBF-SVR','Holt':'阻尼 Holt','SES':'简单指数平滑','HGBT':'梯度提升树','free':'同容量普通 MLP','global':'全局共享衰减参数','conditional':'状态依赖衰减候选','ablation_fixed':'固定 τ=10 消融','ablation_no_gap':'删除历史间隔消融','ablation_no_drift':'删除线性项消融'}
lines=['|方法|平均相对MSE↓|相对RW改善%|2021相对MSE|2022相对MSE|','|---|---:|---:|---:|---:|']
for k,n in name.items():
 r=summary.loc[k];lines.append(f'|{n}|{r.relative_mse:.6f}|{r.skill_vs_RW_pct:+.4f}|{r.ratio_2021:.6f}|{r.ratio_2022:.6f}|')
mt='\n'.join(lines)
sl=['|人工过程|RW|Ridge|普通网络|全局衰减|状态依赖衰减|有额外状态信息的oracle|','|---|---:|---:|---:|---:|---:|---:|']
for kind,ch in [('RW','无信号随机游走'),('constant','固定持续率的趋势+暂态'),('varying','时变持续率的趋势+暂态')]:
 d=syn[syn.process==kind].set_index('model')['mean'];sl.append('|'+ch+'|'+'|'.join(f'{d[k]:.6f}' for k in ['RW','Ridge_diff','free','global','conditional','latent_state_oracle'])+'|')
st='\n'.join(sl)
ledger=[
 dict(id='L01',title='ETSformer: Exponential Smoothing Transformers for Time-series Forecasting',source='https://arxiv.org/html/2202.01381v2',date='v1 2022-02-03; v2 2022-06-20',level='A:全文HTML可访问，核读§3、§4.1.3、§4.2相关公式；不是复现实验',evidence='已有几何阻尼、多头可学习阻尼、水平/增长/季节结构。指数核不构成单独创新。'),
 dict(id='L02',title='Modelling carbon price dynamics with jump-diffusion and higher-order hidden Markov switching',source='https://www.sciencedirect.com/science/article/pii/S0307904X26004646',date='2026条目；具体first-online/卷期日期本轮未核实',level='B:搜索返回出版社摘要片段；直接打开403；未读全文',evidence='跳跃扩散+高阶HMM，碳价状态记忆已有直接近邻。作者仓库提供GCP.csv。未核其全文协议或复现代码。'),
 dict(id='L03',title='CarbonTimer: Overcoming data scarcity in carbon price forecasting with a large time series model',source='https://www.sciencedirect.com/science/article/pii/S0959652626017889',date='2026条目；本轮不重新宣称精确first-online',level='B:搜索返回出版社摘要；直接打开403；未读全文',evidence='Timer监督微调用于广东/湖北小样本；未核预训练覆盖、完整划分，不引用提升数字作本轮成绩。'),
 dict(id='L04',title='Forecasting carbon prices with high-frequency data and irregular shocks: A hybrid P-NN-MIDAS framework',source='https://www.sciencedirect.com/science/article/pii/S1544612326003582',date='2026条目；具体first-online未核实',level='B:搜索返回出版社摘要片段；直接打开403；未读全文',evidence='混频及不规则冲击已有近邻；本文没有所需实时外生数据，不拼接事后指数。'),
 dict(id='L05',title='KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics',source='https://arxiv.org/abs/2107.10043',date='v1 2021-07-21; v3 2022-03-11; DOI 10.1109/TSP.2022.3158588',level='B:官方摘要/版本/相关DOI已读；本轮未核读全文',evidence='结构化状态空间+神经滤波已存在，不能把神经网络加Kalman当作首创。'),
 dict(id='L06',title='Neural Kalman Filtering for Unknown Dynamics: Task-Aware Learning with a Koopman Backbone',source='https://arxiv.org/abs/2609.24763',date='2026新条目；精确提交日未核实',level='B/C:检索返回官方arXiv摘要片段；直接打开DisabledError；未读全文',evidence='未知动力学、Koopman提升与神经滤波是需要后续排除的近邻，不据此声称已完成新颖性审查。'),
 dict(id='L07',title='EM-KalmanNet: Learned Expectation-Maximization for Adaptive Tracking in Partially Known, Block-Wise Time-Varying State-Space Models',source='https://arxiv.org/abs/2608.24404',date='2026新条目；精确提交日未核实',level='C:官方arXiv检索题名线索；直接打开DisabledError；方法未核读',evidence='待取得合法全文的时变状态模型近邻；不能仅凭题名断言与新候选等价。'),
 dict(id='L08',title='Intraday and Daily Price Discovery in Carbon Markets: EUA Futures Versus Carbon ETFs',source='https://onlinelibrary.wiley.com/doi/10.1111/irfi.70099',date='出版社检索显示first-published 2026-08-26',level='B:出版社检索摘要片段；非全文',evidence='小时级EUA期货/ETF的有效价格和暂态噪声分离；任务为价格发现，不作为本研究转向实证的理由。')]
(R/'literature').mkdir(exist_ok=True)
with (R/'literature/literature_ledger.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(ledger[0]));w.writeheader();w.writerows(ledger)
(R/'literature/READING_LIMITS.md').write_text('本轮只对L01的相关全文方法部分进行了核读；其他层级逐条见台账。未运行这些论文原版代码，未宣称击败SOTA。403/DisabledError未通过受限镜像绕过。精确发表日期不从URL推断。\n',encoding='utf8')
report=r'''# 新GDEA数据、模型实验与方向裁决｜Stage C

研究日期：2026-10-13。任务：下载新数据、寻找预测模型/学习方法增量；不回到同期关联论文。

## 1. 运行状态与最终裁决

用户提醒卡顿后，现场发现梯度提升树任务CPU约100%、日志持续刷新，没有死锁。耗时集中在强基线的多次滚动重估。随后补完全部81组配置及消融、汇总评分、完成质量核查；交付时没有训练进程继续运行。

**当前状态依赖衰减预测头不通过。它在新的作者GDEA序列上明显落后于随机游走；本轮没有找到可以据此撰写结果论文的新模型。** 不把这个结论扩大为“碳价不可预测”，也不把已有基线的小幅胜出冒充创新。

研究入口仍可限定为“低自由度、可辨识的持续/暂态状态学习”，但这只是下一阶段的问题方向。状态空间、指数阻尼、神经滤波各自已有直接近邻；本轮没有批准任何未经检验的新算法名称。

## 2. 新数据实际获取情况

读取用户仓库主线为 `7b619967d189a33f46ddd7acaf2123d0d5f4f120`。交易所HTTP接口和本环境直接文件下载未成功，具体DNS失败记录保存在metadata；没有宣称下载到了交易所官方完整行情。

成功路线是通过GitHub连接器逐段读取公开作者仓库 `Chen-zhongxiu/Carbon-price-data`，固定版本 `6e3ea533832f82fd1c8b4859bb2af8c33bd19a31`，文件 `GCP.csv`。日期/价格文本转入计算环境，重新生成类型及原文件四舍五入的对数列，随后整份文件Git blob哈希与源站精确一致。训练前有一处转录笔误已据原文纠正，直到完整哈希一致才使用。不是仅用首尾样例、拼凑行数或按预期价格补值。

- 原始文件：62,539字节，1,921行数值记录；2014-03-20—2023-02-20。
- Git blob SHA-1：`23b1ffc5c68ded3c7b3c03dd46242788f0db52f7`。
- SHA-256：`bedc757f7edf307f26daa752105c209b7a3c1f1b4a46e5f6dc7abd2a513b7e41`。
- 重复日期0、价格缺失0、非正价0；相邻同价91次、周末记录19条、最大日期间隔27天。
- 字段为date/type/price/log_price，type标为GDEA；币种、收盘/VWAP、历史筛选规则、首次可得时刻未独立认证。没有成交量。

来源README说明它服务于《Modelling carbon price dynamics with jump-diffusion and higher-order hidden Markov switching: Theory, filtering, and forecasting》；这不等于本文读完该论文或官方认证了价格。作者仓库根目录未见许可证；本交付不另授予原数据许可、不宣称可商用。

本轮目标是作者报告价格的冻结快照顺序评价，不能写作交易所认证GDEA收盘价实时回测。保留周末、缺口和重复价，不将缺失日期插值，不把同价或无数量当成无成交。

## 3. 事前冻结的时间合同

|区间|观测数|用途|共同评分起点|
|---|---:|---|---:|
|2014—2018|933|初始历史、冻结损失尺度；实际每次拟合仅用最近512个完整成熟起点|不作报告主评分|
|2019|232|调参|212|
|2020|240|调参|220|
|2021|242|一次性历史检查|222|
|2022|244|一次性历史检查|224|
|2023|30|在进入模型/指标前截断|未评分|

旧EU序列及其2017—2018保留段没有参与本轮训练或评价。新GDEA的2021—2022本轮已打开，后续不能再说是未接触的检验集。虽然首次成绩前冻结了本轮协议，但这是本地冻结，不是外部预注册；历史数据已经可见，且市场身份受限，所以不是独立官方市场确认。

预测步长为1/5/20个观测步。所有预测起点及三个目标均在同一评分年；每个时点先观察价格，再以i+20≤当前索引筛选成熟标签，更新后发出预测。所有拟合型竞争方法使用相同完整标签支持和20步重估节奏；简单平滑基线按其定义更新状态。已发预测不回算。

**主指标是年份和步长等权的平均相对MSE：**

\[
Q(M)=\frac{1}{2\times3}\sum_{y\in\{2021,2022\}}\sum_{h\in\{1,5,20\}}
\frac{\operatorname{MSE}_{y,h}(M)}{\operatorname{MSE}_{y,h}(RW)}.
\]

RW恒为1。改善百分比为100(1−Q)。这是平均相对平方误差，不是RMSE改善比例。这个口径在新数据成绩出现前冻结，不与上一轮EU的训练尺度NMSE直接拼接。另报原始MAE/RMSE及训练尺度NMSE。神经种子在损失上平均，不先集成预测。

## 4. 真实实现的候选

设q_t为只由过去60个价格变化得到的RMS尺度，含训练期确定的小幅下限；x_t包括60个相对现价差/q_t及一个过去日期间隔log(1+gap)。候选为

\[
\widehat p_{t+h|t}=p_t+q_t\left[\frac h{20}u_\theta(x_t)
+\left(1-e^{-h/\tau_\theta(x_t)}\right)v_\theta(x_t)\right],
\qquad \tau_\theta(x)=0.5+39.5\sigma(g_\theta(x)).
\]

这是直接预测原始价格变化的模型，能改变方向，不是上一轮0到1的乘法收缩。线性项和暂态项是预测参数化，未经潜在状态恢复证明，不能解释成已经识别出的经济永久冲击/暂态噪声。

骨干61→32→16，GELU；条件衰减候选与自由三输出MLP均2,563参数。全局衰减对照2,547参数，固定τ对照2,546参数，不能声称后二者与候选严格等参数。学习率0.001/0.003，weight decay0.001/0.01，每族4组；种子11/29/47。初始200次更新，之后每20观测步更新20次，batch≤128，无未来标签和目标平滑。

原始多步平方误差按冻结训练尺度标准化。各模型族只根据2019—2020选参；候选最终为lr=0.001、wd=0.01。自由MLP和全局衰减独立选参恰好也选中同一设置，所以独立/匹配对照数值相同。

实际预算：69组传统基线配置、12组神经配置×3种子=36条主神经轨迹、3项固定参数消融×3种子=9条轨迹。另有3种人工过程×5条路径×3种神经结构=45条人工训练轨迹。所有81组配置的预测档案均完整读取通过。各族网格上限相近不代表计算预算或容量完全相同。

## 5. 一次性历史检查结果

__MAIN_TABLE__

候选的平均相对MSE为1.819708，相对RW恶化81.9708%；2021恶化76.8702%、2022恶化87.0714%。这不只是未达到2%门槛，而是方向性失败。同容量普通网络亦明显落后于RW；这一点与小样本估计不稳定/过拟合相容，但本轮没有独立分离优化、非平稳性和表示偏置各自的贡献，不能把失败唯一归因于其中一个。

调参区预先选出的非候选总比较对象C*是Holt（调参Q=0.923713），检查区Q=1.095678。检查区成绩最好的已选参基线是240步漂移，Q=0.992383，但2022仍恶化5.4162%；这是描述性排序，不是据测试重新选择生产模型，也不是可发表的创新结果。

不能事后改用训练尺度NMSE替换主指标：例如SVR的训练尺度NMSE略小于RW，但按事前主指标仍恶化6.9928%。全部指标都保留，不能择优口径报告。

## 6. 机制与可辨识性检查

### 6.1 状态依赖机制没有提供增量

条件τ候选比全局τ对照恶化约10.05%；比同容量自由MLP也更差。固定τ=10反而优于条件τ，但仍比RW差53.18%。因此不能拿某个较好的消融顶替事前主候选并写成功。

2021年种子11中95.5%的τ小于1，种子29约68.0%，种子47约43.7%；不同种子的时间尺度明显不一致。边界聚集不自动证明经济制度变化或真实冲击半衰期，更不能用它给模型添经济解释。

### 6.2 指数核等价于已有几何阻尼

对ρ=exp(−1/τ)，

\[
1-e^{-h/\tau}=1-\rho^h=\frac{1-\rho}{\rho}\sum_{j=1}^{h}\rho^j.
\]

ETSformer全文§3式(2)、§4.1.3已有几何阻尼及多头可学习阻尼[L01]。本轮对多组τ计算验证上述恒等式。不是说整个条件参数模型与ETSformer完全相同，而是指数加载本身没有新颖性；唯一可能的剩余差异是依赖当前状态的持续时间学习，实际没有增益支持。

### 6.3 三个端点不保证时间尺度稳定可识别

当v=0时，预测与τ无关。τ较大时，1−exp(−h/τ)≈h/τ，暂态加载与线性项近乎共线。对(u,v,logτ)的三端点Jacobian，v=1工作坐标下τ=40时条件数约1.43万；τ=400（范围外的纯数学诊断）时约1.21千万。这说明参数化存在病态边界，不是坐标不变的完整识别定理，也不是对真实市场潜状态的认证。

后续若研究低维状态模型，应该先解决原始风险下的状态恢复和弱识别控制，不宜继续把自由τ曲线当作创新证据。

## 7. 人工过程：有信号不等于当前网络学得到

每种过程5条独立路径，网络固定seed=11、lr=0.001、wd=0.01，不依据市场检查成绩挑参数。

__SYN_TABLE__

人工过程包含无信号RW、局部趋势叠加固定暂态AR、局部趋势叠加预定时变暂态AR。Ridge在两种有信号过程中取得一定改善，但当前三种网络均未稳定转化该空间。Oracle使用真实隐含趋势、暂态和预定制度切换日程，信息比拟合模型更多，仅作诊断参考，不能当公平市场比较或有限样本保证。

本轮没有以失败模型“证明碳价无法预测”；更具体的发现是当前网络/训练配方尚未通过基本信号恢复检查。

## 8. 不确定性、实现和卡顿防护

在各检查年内进行配对移动块重抽样，主块长40，20/60敏感性，每档2,000次。全部模型和步长使用同一重抽样日期。主块长下，候选相对RW改善百分比区间约[−188.85%,−51.63%]；相对全局τ区间跨零。它们未校正模型选择/多重比较，也不保证结构变化过程的标称覆盖。

12项原单元测试在普通和python -O下通过，覆盖源文件哈希、未来扰动不改变过去特征/预测、成熟标签、参数边界、容量匹配等；共同年内评分另在质量审计中检查。另完成质量审计：81档案可读且预测齐全；2023未进入模型；冻结协议未改；HGBT批量预测与原逐行预测在固定测试夹具上精确一致。

代表性重放范围是候选seed11完整轨迹、自由MLP seed11完整轨迹、相对水平Ridge完整档案，逐值一致。**没有声称把全部81组完整重跑。** HGBT在首次运行期间曾将同一重估块的预测改为批量计算；没有改变拟合或超参数。质量审计验证了对应路径的数值等价。

交付后运行器增加缓存形状/有限值/完成收据检查；重放入口每10秒更新run_status.json，写明PID、当前阶段、完成档案数和日志位置。它不修改预测算法。原始结果和所有变更说明均保留。

## 9. 文献边界与下一方向

本轮新检索8条重点记录，逐条见literature/literature_ledger.csv。只有ETSformer的相关全文方法已经核读；CarbonTimer、跳跃扩散高阶HMM、P-NN-MIDAS主要是出版社检索摘要，页面直接访问403；KalmanNet核实官方摘要和版本。2026年的Koopman/EM-KalmanNet新条目为摘要片段或题名线索，直接访问失败，没有声称已读全文。

**潜在问题方向：可辨识度控制的低维持续/暂态状态学习。** 示意模型p_t=ℓ_t+z_t，ℓ携带持久水平/趋势，z满足有界稳定持续率。先以因果滤波得到少数状态；只有多步敏感度提供足够信息时，才允许相对全局时间尺度作小幅状态依赖修正。弱信息时约束新增自由度，而不是用大MLP重新拟合每个τ。潜在贡献应落在具体“何时允许适配”的学习规则和可验证的弱识别边界，不是滤波/阻尼/神经网络的简单叠加。

这个示意方向尚未实现、尚未证明新颖、尚未取得市场增益。本轮不另造名字或把它当第四个已经成功的模型。KalmanNet及2026时变神经滤波已构成直接近邻[L05-L07]；碳价永久/暂态分离也不是未经研究的概念[L08]。下一次独立实验必须在新的预先固定数据/区间上进行；2021—2022不能继续充当未见测试集。

发表层面的三个条件仍分别未满足：官方/有定义主样本的可靠性；相对强基线稳定增量；可被近邻排除的具体机制贡献。当前完成的是更完整的预测方法开发与否证实验，不是可提交的结果论文，更不是同期关联实证替代品。

## 10. 文件使用

- 原始作者数据：raw_private/GCP.csv，已验证整文件hash；来源与边界见DATA_NOTICE.md。
- 新下载脚本：code/download_data.py。网络可用时直接下载固定版本并验证hash；已有正确副本时不重新下载。
- 全部配置、家族选择、逐年步长误差、逐样本预测、模拟、区间与重放审计：outputs/。
- 运行 `python reproduce.py --destination 新目录`，会复制输入、代码与冻结协议，重新训练，不覆盖交付结果。中断后同目录加 `--resume`。部分模拟若中断会整阶段重跑，市场配置按完整档案断点续跑。
- 本轮未推送远端GitHub，未调用用户本机GPU。源作者数据没有被重新标为本代码许可。
'''.replace('__MAIN_TABLE__',mt).replace('__SYN_TABLE__',st)
(R/'REPORT_ZH.md').write_text(report,encoding='utf8')
(R/'DATA_NOTICE.md').write_text('''# 数据来源与使用边界

本地副本来自Chen-zhongxiu/Carbon-price-data，固定提交6e3ea533832f82fd1c8b4859bb2af8c33bd19a31，GCP.csv。源站README称其服务于Modelling carbon price dynamics with jump-diffusion and higher-order hidden Markov switching: Theory, filtering, and forecasting。

数据是作者发布的数值快照，不是本轮重新从交易所下载的官方认证行情。根目录未见明确数据许可证；本交付不授予或声称具备商业利用/公开再分发许可，不以研究代码许可覆盖原始数据。保留副本用于用户请求的分析与验证，任何后续对外发布须单独处理数据来源条件。

原文件价格含义、计价单位、筛选规则及历史首次可得时刻仍有缺口。不得标成已经认证的收盘价，或把观测步擅自叫正式交易日。下载渠道失败日志及源文件哈希在metadata中，不能把Git哈希一致升级为市场含义认证。

来源：https://github.com/Chen-zhongxiu/Carbon-price-data/tree/6e3ea533832f82fd1c8b4859bb2af8c33bd19a31
''',encoding='utf8')
(R/'README.md').write_text('''# Stage C: Carbon forecasting research

Read REPORT_ZH.md first. Candidate failed the predeclared continuation gate; this is not a successful new-model manuscript.

Contains a pinned author-reported GDEA snapshot, 81 configuration archives, a conditional horizon-response decoder and controls, 3 fixed-parameter ablations, 15 synthetic paths, and audit evidence. No original EU results reused. No official exchange-vintage claim.

## Replay

Use Python 3.13 and requirements.txt (CPU PyTorch build is sufficient).

```
python reproduce.py --destination ./stageC_replay
```

A fresh directory is required. To resume a reproduction use the same destination plus --resume. run_status.json records live progress; per-configuration checkpoints are validated before skipping. The delivered outputs are never overwritten by this command. Do not treat a short checkpoint pause as deadlock: inspect process activity and recent log/checkpoint updates.

Data notice: DATA_NOTICE.md. Source hash and failed direct download attempts: metadata/.
''',encoding='utf8')
env={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'torch':torch.__version__,'gpu_used':False,'threads':1}
(R/'metadata/environment.json').write_text(json.dumps(env,indent=2))
(R/'requirements.txt').write_text(f'numpy=={np.__version__}\npandas=={pd.__version__}\nscipy=={scipy.__version__}\nscikit-learn=={sklearn.__version__}\ntorch=={torch.__version__.split("+")[0]}\n')
changes=[{'stage':'before training','change':'A transcription mismatch was corrected against retrieved source; full source hash matched before use.','effect_on_source':'none: matches pinned original'}, {'stage':'before historical evaluation','change':'HGBT predictions in each fixed-refit block batched for runtime; same fitted estimators/inputs/hyperparameters.','verification':'quality_audit HGBT fixture exact equality'}, {'stage':'after evaluation','change':'Added checkpoint shape/finite/receipt validation and standalone 10-second progress monitor.','effect_on_algorithm':'none'}, {'stage':'after evaluation','change':'Added representative replays and algebraic conditioning/damping diagnostics.','effect_on_hyperparameter_selection':'none'}]
(R/'metadata/implementation_change_log.json').write_text(json.dumps(changes,indent=2,ensure_ascii=False))
files={str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in R.rglob('*') if f.is_file() and '__pycache__' not in str(f) and f.name!='delivery_manifest.json'}
(R/'metadata/delivery_manifest.json').write_text(json.dumps({'files':files,'count':len(files),'all_market_configurations':81,'evaluation_complete':True},indent=2))
shutil.copyfile(R/'REPORT_ZH.md','/mnt/data/carbon_stageC_20261013_REPORT_ZH.md')
shutil.copyfile(R/'raw_private/GCP.csv','/mnt/data/GDEA_author_GCP_verified.csv')
print('Report and manifests written. pandas',pd.__version__,'files',len(files))
