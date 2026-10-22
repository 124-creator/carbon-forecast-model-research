# 碳预测模型创新：当前研究状态

更新：2026-10-22。任务继续，尚未宣告达到可发表的模型创新与实验成功标准。用户目标是模型创新和可信正向实验；不转阴性或评估类论文，不投MDPI。

## 2026-10-22 两项决定性结果

**一、评测工具的分辨率低于一直在追的效应量。** 对已冻结的R20归档做只读跨期审计：14个模型在2025与2026各切3个连续块，跨期排序Spearman仅0.170（原假设标准差0.277，z=+0.61），2026期内部分块一致性为−0.247。真正改写排序的量（各模型跨期位移的离散度）是0.0847 raw QLIKE，而本项目已报效应为0.0004–0.0025，相差34–212倍。配对6块×3种子=18格的符号检验：R20 joint vs continued 13/18、R20 joint vs C18_GRU 11/18、density_only vs continued 9/18，**全部未确立**。结论：单窗口胜负不构成有效推断。详见[跨期稳定性审计](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/PERIOD_STABILITY_ZH.md>)。

**二、观测算子这条线按预注册规则关闭。** 在GRN（零range代理342/1581）上做噪声梯度检验，规则在见到任何数字前写死。结果：C1（gamma锚 vs EWMA锚，9对）**0/9**，差+0.102至+0.113，远超0.002–0.005的种子波动；C2（路径耦合，6对）**2/6**，且在唯一有竞争力的ewma_anchor家族里0/3；绝对门槛（QLIKE低于EWMA22的1.127771且MSE不劣于1.321107e-8）**24个配置0个通过**——ewma_anchor家族12个全部QLIKE更优但MSE全部更差。诊断确认gamma在h=22高估41%，但单一尺度校正只收回0.054，校正后仍差0.0945，说明是预测形态错而非常数搬运错误。判定：**不再以观测算子/测量误差作为论文贡献主张**。详见[结果与判定](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/GRN_NOISE_GRADIENT_RESULT_ZH.md>)与[预注册](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/GRN_NOISE_GRADIENT_PREREGISTRATION_ZH.md>)。

未受这两项影响的事实：KRBN/KCCA的2026仍为已用诊断期；GRN2026仍未打开；GRN零range是报价不可获得的缺失机制，不是加性测量误差。

## 2026-10-22 区间（分布）实验：一条假设确认，一条否证

点预测轴已被证明到顶，转向唯一有量级的方向：h日均值的预测区间。描述性计算给出GRN原始代理上h=22的方差膨胀**3.84倍**（解析值3.80），独立假设会把区间标准差低估约1.96倍。预注册固定位置为EWMA22、只比离散度，比较iid假设（A）、全局经验池（B）、状态条件近邻池（C）。

**H1（聚合效应）强确认。** iid假设把95%区间覆盖率压到h=5的0.855、h=22的**0.553**（对数空间）；乘法空间更差（0.425、**0.325**）。两个空间、两种构造、量级远超抽样噪声（228起点上二项标准误约0.014）。

**H2（状态条件形状）被否证。** 加了关键对照B2——全局池但直接剔除2019–2020起点、不做任何条件化。h=22：C对B为−1.69e-06（t=−4.81），但**C对B2为+7.9e-07（t=+3.17）**，方向反转且显著。C的全部表面优势来自顺手剔除2020，不是来自条件化。近邻数k从50增到200时结果单调靠向B2，与之一致。

**本轮最好的简单方法是B2**：pinball/mean(y)=0.1626、覆盖0.956、中位相对sd 0.862，四项里最优。但它只是"用对了残差池和聚合"，**不构成方法创新**。

必须记录的两个方法问题：一是首次预注册运行用乘法残差`y_h/L_s`，而2020占池15.1%且`L_s`中位数仅2.90e-06，使B的h=22相对sd达6.37倍水平——覆盖0.982是"宽到无信息"换来的，不是校准；该次运行B无效、C不可信。二是三次运行都继承EWMA22在2025的**位置偏差**：验证期对数残差中位数−0.243，即水平系统性高估约27%，且`L_va`中位数只有训练期的38%。宽度正确也补不回中心偏移。详见[结果](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/INTERVAL_FORECAST_RESULT_ZH.md>)、[预注册](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/INTERVAL_FORECAST_PREREGISTRATION_ZH.md>)、[设计效应](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/AGGREGATION_DESIGN_EFFECT.json>)。

**水平偏差不可利用（同日补查）。** 上述−0.243是**中位数**效应而非均值效应：h=22的QLIKE最优尺度c*=1.0613（仅6%偏高），oracle校正增益−0.0018；h=1的c*=1.1576、增益−0.0113。且偏差逐年漂移（h=22的mean(y/L)：2021年1.489、2022年1.226、2023年1.042、2024年1.105、2025年1.061），用训练期尺度1.2182反而使h=22变差+0.0073。**关键：连oracle全局尺度校正的最大增益0.0113也低于跨期审计的可复现下限0.0847**，因此即使做对也无法证明。

## 数据与确认边界

- KRBN/KCCA的原2026保留期已在第20轮打开并完成一次冻结确认。后续同一时期只能作诊断，不能重新称为全新独立确认。
- 原冻结快照末日为2026-10-20。共同评分为每资产179个起点，2026-01-02—09-18，22日标签末日不晚于10-20。
- GRN是Barclays Series B Carbon ETN，不是ETF。现有历史从2019-09-18开始，缺口不填；信用风险、split口径和与原ETF底层敞口重叠均需披露。
- GRN截至2025年底的开发数据已用于训练；隔离的GRN2026原始数据只做过质量检查，尚未构造模型目标、预测或评分。相关产品的外部复现不等于独立宏观市场确认。
- 日内range目标是未来Parkinson proxy均值，不是已识别的IV/QV。H/L缺失不得补零；真实报告的零range必须保留。GRN含零目标的normalized QLIKE不定义，主QLIKE仍计算所有行。

## 已取得的结果

第20轮joint相对同训练/同输出头continued的normalized QLIKE改善仅0.0821%，三个seed方向一致，但同步区块区间跨0。整体仍落后MEM。全部selected alpha_N=0，因此密度机制未获支持。详见[冻结确认审计](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/ROUND20_HOLDOUT_ZH.md>)。

第21轮改用正值方差Markov状态与Gamma观测、完整shape与radial后验。开发期NN ensemble为1.080559，相对fixed mixture的normalized QLIKE改善0.4626%，但仍落后更强的raw/GRU。两个seed有改善，一个精确回退；M后验确有变化，但包括reference/future-factor耦合，不能解释为真实采样次数识别。[开发审计](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/ROUND21_DEVELOPMENT_ZH.md>)。

第21轮已用2026诊断：global_dense known-scale QLIKE 1.155654，MEM stationary 1.159433；神经density为1.184949，fixed mixture为1.184680。基础正值模型有小幅点优势，神经密度增量未泛化；不能事后把global换成原主候选或宣告独立成功。[完整描述性结果](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/round21_frozen_diagnostic_used2026_report.json>)。

第21轮GRN独立开发训练也已完成：3个base及3seed NN，fixed mixture选到epoch80，需要如实标预算上限；NN选38/59/33。所有原始零目标参与评分；尚待强基准完整比较，不据局部提升宣告成功。

## 当前执行的创新实验

第22轮学习未来风险强度S与非零区间指示Z的联合预测路径。Z仅定义为raw range>0，不代表真实交易次数或无交易。预测用E_q[SZ]，包含风险与区间状态的协方差；神经势由已知历史生成，KL约束相对固定参考路径的修正。Feynman–Kac、Doob与KL理论本身已知，新性候选是具体受约束的预测结构。

已完成数学核枚举、梯度、零锚点、因果性与标签成熟检查。真实开发训练已启动，三组为coupled_path(137参数)、independent_path(110)、direct_marked_control(137)。直接控制的a/c不除桶长；各组输出惩罚均按active channel平方求和。所有base参数、预处理和缓存冻结。[模型源](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/round22_predictive_path.py>)。

GRN强控制拟合同时启动：MEM stationary/integrated、原HAR/HGB、HGB Poisson及直接QLIKE回归，全部同日期/同原始目标、不删除零目标。[基准源](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/grn_strong_baselines.py>)。

当前主线程后台job：R22为pwsh-15e9fbe05d9a4af1；GRN基准为pwsh-a71c7e83cbf147b4。完成后收集并独立比较，不重复启动同一实验。

## 真前瞻记录与数值范围

基于2026-10-21完成bar的原模型预测已于2026-10-21 22:06:33 UTC封存，接受receipt为true；R21新模型也在下一NYSE开盘截止前独立封存。未知未来目标留空，不回填预测时间、不覆盖旧记录。目前未设置每日自动任务，也不能用一条前瞻记录宣告成功。

条件Gamma的分箱/收缩后矩与global矩有近似闭合差；状态网格插值保持内部均值，不保持方差。256/512纯检查敏感度不是市场预测误差界，边界单行截断率也不是实际posterior加权截断率。详见[数值审计](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/ROUND21_NUMERICAL_AUDIT_ZH.md>)。

早期HAR-D的7–10%及全部显著等结论已撤回。第13—19轮结果保留为历史探索，不能替代新确认。CEA昨日收盘价及挂牌/大宗口径仍有边界，未作为可靠新独立数据集使用。
