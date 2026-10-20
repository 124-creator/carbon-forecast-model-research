# 唯一外生信息增量隔离试验：结果

已按预写CONTRACT执行一次固定配置，没有追加截距/alpha/核形状轮次，没有commit或push。

## 主结果
2020/2021同年451起点（224/227），年度×h1/5/20等权relative RW MSE，越小越好：

|模型|主分数|相对匹配price-only|
|---|---:|---:|
|原MLE480|0.949793230121|不适用|
|同池price-only受惩罚残差校准|1.041722637988|0|
|固定price校准+煤diff增量|1.062464239081|+0.020741601093|
|固定price校准+FX return增量|1.041590974486|−0.000131663502|
|historical strongstate，仅历史参照|0.942705|不是本次重训练|

煤当前diff信息路线未提供总体增益，应停止。FX存在很小的总体数值改善，不能写成“完全无增量”：2020 h1/h20改善、h5恶化；2021三个h均恶化。该结果不构成稳定/显著/跨年度的有效信息证据，也远未超过原MLE或历史strongstate。保留这项微小条件信息观察，不继续本轮截距或alpha搜索；不据此设计新算法或升级neural。若下一步讨论FX设计，须先明确这只是薄弱信息证据，不能称稳健成功。

## 全部year×h原始分数
|模型/年份|h1|h5|h20|
|---|---:|---:|---:|
|MLE 2020|.873708713|.910254771|.901040385|
|MLE 2021|.992252281|1.009778240|1.011724991|
|PRICE 2020|.862243520|1.039321316|1.130928831|
|PRICE 2021|1.031824036|1.054447489|1.131570637|
|COAL 2020|.865650705|1.074408203|1.218352979|
|COAL 2021|1.032461739|1.055390449|1.128521359|
|FX 2020|.860776039|1.044274031|1.106140644|
|FX 2021|1.039427534|1.063700593|1.135227007|

## 实际方法及公平性
price状态沿用diagnose.py的8块价格变化widths=[1,1,1,2,5,10,20,20]，训练标准化后附加常数列，常数和状态都受同一个n*alpha惩罚。每h独立系数，但alpha按三个h平均inner MSE统一选择。price-only先选择zero/.1/10，source随后只对已固定校准的残差拟合。煤只用原feature column1的相邻公告价格差；FX只用原FXstrong column0/1的1/5条ECB记录log ratio变化。

所有模型使用每块完全相同的成熟训练池：480窗中煤+FX均有效起点的交集；最后80个共同起点验证，最大h20标签严格早于验证边界。这里共同池比全价格池更窄，price-only成绩不能冒称全价格池最佳校准。源标准化与向price-state ridge投影均在inner train估计，再以零独立截距readout估计增量。投影和readout使用同一source alpha候选.1/10；源zero也参加验证。选定后在完整共同成熟池实际重新拟合。

这是sequential regularized partialling，不是joint orthogonal regression、不保证正交、不是DML或新算法。ridge投影的截距属于训练source nuisance，不是另加残差常数校正。价格校准固定，源readout不再中心化U或残差、不加入常数列。

12个刷新块选择全部保留：
|通道|zero|alpha .1|alpha 10|
|---|---:|---:|---:|
|PRICE|5|6|1|
|COAL|6|2|4|
|FX|6|4|2|
zero是内验证候选，不是外层安全保证。TRAINING_LOGS含每块三组候选损失、共同n、purge边界及煤diff不同取值数量；持有值起点不是独立事件样本。

煤发行缺失40个起点，实际保存预测逐值等于匹配PRICE而不是MLE。FX本样本无发行缺失，因此FX真实缺失分支检查在此样本不提供非空证据；代码相同匹配PRICE回退规则。

## 来源和旧结果核查
指定工作区原本没有newrepo或煤/FX训练输入。输入只读来自实际归档repo E:/碳价格/独立数据归档/carbon-forecast-model-research；输出严格位于当前工作区newrepo/exogenous_information_route_20261014/increment_isolation，不修改旧实验。

已读取coal_failure_diagnostic/REPORT_ZH.md、diagnose.py、coal_training/run.py及run receipt，FXstrong/run.py、报告及run receipt；SOURCE_SCHEMA_AUDIT实读CSV columns、123公告与123索引、1025×2 ECB记录、全部原NPZ键/shape/dtype/hash。旧煤columns [level,diff,age,missing]；旧linear采用未受惩罚残差均值，旧kernel也带共享未受惩罚均值，所以不能把原失败直接当无信息结论。

煤+1沿用publication_displayed+1自然日；FX lag2沿用原函数date-(lag+1)，即date-3自然日截点，不偷换成date-2。FX log(CNY/EUR÷USD/EUR)=log(CNY/USD)，使用已经下载的数据，不认证实时首版。原123煤事件包括价格前缀之后的公告，但按可用日排除，未制造假面板。950价格使用原逐字节定额读取器，到2021-12-31即停止，不读取2022价格或下一行。

## 验证与限制
实际259次ridge线性求解，包括候选、部署重拟合及代表扰动重拟合；12个刷新块。代表initial、initial+40、initial+200分别扰动future coal、FX和price，共9项实际完整重拟合：所有拟合参数、alpha/zero选择和发行预测差均0。price未来扰动从发行点+1开始，实际重新计算price状态、MLE vintage、标签和全部校准/增量拟合，未使用缓存预测冒充refit。

独立verify.py在normal和python -O运行，两份JSON完全相等，核查451、MLE复现、有限预测、成熟界线、9个refit及匹配缺失回退；不是用assert。没有对259全部fit逐个futureperturb认证，也没有新统计显著性或独立事件bootstrap结论。

输出：CONTRACT.md、run.py、verify.py、INPUT_AUDIT.json、SOURCE_SCHEMA_AUDIT.json、PREDICTIONS_LOCAL_ONLY.npz、TRAINING_LOGS.json、FUTURE_REFITS.json、RUN_RECEIPT.json、VERIFICATION_NORMAL/OPT.json、YEAR_H_SCORES.csv及本报告。实际训练运行0.375秒；整轮在15分钟预算内。预测仅本地保存，无联网、无push。参数和source选择基于inner段，不按外层年份调参。
