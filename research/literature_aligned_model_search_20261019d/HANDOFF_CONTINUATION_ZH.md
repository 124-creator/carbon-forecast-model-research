# 续接上下文：文献对齐与新模型实验，2026-10-19d

## 1. 用户目标与授权

用户要求继续碳价预测实验，了解已发表文献怎样执行，实际取得需要的数据，找到有可检验贡献的模型；实验支持后再论文。用户明确允许借鉴文献中合理的训练信息使用，不接受把训练未来标签、历史窗口内双向处理、作者清洗基准等一律排除。既有下载、研究仓库上传与草稿 PR 授权持续有效，不重复询问。

所有回答应使用中文，直接说明做了什么、什么支持创新及什么仍不支持。不能承诺正结果或录用，不能为了达标改写原主候选、选最好测试种子、隐去固定控制，也不能只写新的审计计划而不进行市场实验。

## 2. 当前完成状态

**本轮五批市场模型运行全部完成，无训练任务挂起。**另完成一批文献信息范围诊断。所有输出已经有独立只读复算；模型实现者另做初始化/参数学习机制回放，二者身份在各报告中区分。

最终状态：`PREDICTIVE_STRUCTURE_USEFUL_LEARNED_INCREMENT_UNCONFIRMED`。没有开始声称新模型成功的论文。末端 MLP/固定响应仍保留为后续强结构基线；量逐步分配在部分市场确实学习改善，但量本身的跨时期增量没有确认。

研究分支：`research/literature-aligned-model-search-20261019d`。

该分支叠放于前一研究分支 `research/forecast-geometry-filter-20261019c`（旧 Draft PR24），不修改或合并旧 PR。新研究目录之外的旧文件保持原内容。

## 3. 五批结果与主次身份

| 批次 | 评分前固定身份 | 核心结果 MSE/RW | 解读 |
|---|---|---|---|
| V1，4市场×2022/2023 | 主 allocation_full；readout 原为同容量控制 | full1.001207；readout0.975430；constant0.984378 | readout 是结果后发现，其相对RW正向、相对constant不确定；净学习高度集中上海2023 |
| V2，广州/湖北×两年 | 主 liquidity_full | full0.974008；去量0.987823 | 95.6%总体RW收益集中湖北2023；原主块区间跨零，量增量不稳 |
| V3，ScienceDB新增市场 | 主liquidity_full；次allocation_readout；BJ/TJ平均，CQ另列 | BJ/TJ full0.810168、readout0.790512、constant0.805094 | full北京0.559604、天津1.060731；六项整体主/次区间都跨零。readout两市场都从自身init改善 |
| SWEWT补充 | 在V3部分结果后单独记录，非原V3注册家族 | price0.809787、full0.801339 BJ/TJ均值 | 同目标、同预算的多步控制适配，不是原作者数值复现 |
| V4，官方天津较近尾段 | 在前四批后、该新尾段分数前：主readout，次liquidity_full | readout=constant0.959915；full0.957621；去量0.940117；SWEWTfull0.991564；priceRidge0.901624 | 六结构全部epoch0；readout对RW和SWEWT区间正，但对constant逐值完全无增量，强验证基线Ridge更好 |

V4预指定五项主20观测块99%区间（五项约同时95%），单位RW百分点：readout对RW+4.008546[1.305597,7.018662]；对constant0[0,0]；对SWEWTfull+3.164942[.683396,5.932099]；liquidityfull对RW+4.237930[1.265488,7.609692]；对shock−1.750350[−3.928017,.312550]。40块敏感性完整保留，不选择更有利块长。

V4六结构12次选择训练共900正梯度步骤，回滚前参数确实改变；18份最终结构模型均选择epoch0。30 checkpoint与同seed初始化逐值相同，所有993可构造起点的最终预测与自身初始化相同。这不等于没有尝试训练，也不等于架构在一切可能训练方案下永远无效。

总计560神经选择/构造任务，440有正梯度；224选择、336最终构造，其中216最终有更新、120 epoch0。执行13,254epoch。五批经典估计器528fit，独立EWT诊断84fit，共612。不能把同一原点在不同模型中出现算成新的独立样本。

## 4. 当前模型的准确数学边界

价格新息是统计术语，不是“已经原创”的证明。r_t 是原训练价格RMS归一化算术变化；能源u_t各用自身训练RMS。两有序衰减rho初始化.5/.9，暂时分配初始.05/.05，其余永久；未来响应由同一状态决定H1/5/20。窗口64、状态每窗口从零重置，不是终身在线滤波。

V1 full与readout各169参数、同输入与初始预测；readout历史用固定分配，仅终点以w_proposed/w_base乘两响应。constant7、price-only118。V2full/readout/unconstrained11、shock9；full/readout初始函数不同，full/unconstrained初始函数相同但raw系数映射导数不同。10.508倍指初始化的链式导数/raw梯度比例，不是有效系数损失梯度或Adam实际步长整体比。

孤立正新息从零状态出发只能保留/回撤，不能生成正向持续动量；不能把状态识别为经济真实永久成分。条件对称与未来能源零均值等额外条件才支持原状态过程均值解释；readout不自动继承它。已有永久暂时分解、神经状态空间、学习滤波增益、神经条件响应先例，不宣称这些首次。

GRU/LSTM20输出仅H1/5/20有损失，另17×33=561头参数无任务梯度。名义量输入GRU4500/潜在任务有效3939；SWEWT全输入4788/4227。所有神经预算2LR(.003/.01)、max80/min15/patience12、seed17选择、17/29/43重新拟合；主先平均seed损失，集成另列。

## 5. 数据、当前暴露与许可证

既有四开发系列2022/2023已被多次性能开发，不能重新称盲测。ScienceDB原平台V1，文件名v6；2019作者已清洗每日成交均价，CC BY4.0。北京673、天津457、重庆240按旧60/20/20切分；过去仅QC不是过去已拟合，但这次V3后它们已完成绩效评价。

新天津官方 canonical：

`/tmp/carbon-next-market-data-20261019d/TIANJIN_OFFICIAL_MEAN.csv`

SHA256 `85ca07648855db00fbf67c4e1c083db715f62ab829f74c143c3b3f67da21252a`；1077条，2013-12-26—2026-09-30，表头date,price,volume_tons,turnover_yuan；官方TJEA首张表的当日汇总，保留两位小数价格、全部行，无改价/填充/删行。A646/B861，test从2023-04-19、共同原点196至2026-07-02，末H20标签2026-09-30，步长是观测记录。EIA最大回溯年龄17–20天，源覆盖2026-09-29，非陈旧到2023的能源序列。

官方入口 https://www.chinatcx.com.cn/tshq ；Body.js普通公开URL由该页HTML链接，获取/哈希/复建见provenance。历史版本非逐日报告vintage。此前没有该后段模型分数，但现在V4已评分，所以以后是已暴露开发数据。旧天津457日期都存在于新源，178有源值差异；没有据此筛选，新test与旧源全部日期零重叠。

作者无明确许可的工作簿、辅助量、天津原件/逐行canonical、含相应价格路径的NPZ/CSV以及binary.pt保持/tmp，不加公开repo。ScienceDB许可包公开；项目自训numeric state_dict明确获授权公开，诊断JSON可保留，不能再用“所有参数都必须私有”删掉它们。Library旧原件包曾被自动审查拒绝，不绕过限制重新上传。

## 6. 运行位置与复现

运行根：`/tmp/carbon-literature-aligned-runs-20261019d/`，子目录为 `main_v1`、`liquidity_v2`、`new_market_v3`、`swewt_gru_supplement`、`official_tianjin_v4`。

解释器：`/tmp/carbon-research-env/bin/python`。EWT依赖路径 `/tmp/carbon-forecast-20261019d-deps`。Python3.12.14、torch2.14.1+cpu等准确版本见provenance/RUNTIME.json及运行协议。完整命令见REPRODUCE_ZH.md；新的官方runner用`--data-file`，不是`--data-root`。复跑任何preflight必须新`--preflight-report`，勿覆盖原回执。正式out需要新空目录。

EWT信息范围诊断逐点CSV已移到 `/tmp/carbon-literature-aligned-runs-20261019d/information_diagnostic/PREDICTIONS.csv`；SHA d21795622bbb037772ff39f63c79b689ada5ad21721740219ca38efa7c660715，审计复跑显式`--predictions`。公开只留汇总、完整来源/选择、迁移收据。此诊断仅单步Ridge，不能冒充神经原论文复现。

## 7. 本地冻结与远端镜像

代码提交与自身拟合的先后证据见provenance/LOCAL_PHASE_FREEZES.json；五个本地冻结提交依次20dee519、72010f6e、ba048cb2、a0627389、9c7d9044。它们远端内容树镜像有不同commit SHA，逐项映射见provenance/REMOTE_PHASE_MIRRORS.json。GitHub上传在本地运行开始后，不能称外部预注册。

本地原parent230ccf574cc3046c382f18e091e3d64e62b6af29与远端原parentd018af027470dbbe51f254c769ebb3ff7c45b0e4具有同树e47b03d091e5312c4a99debcab5a3ceed0be640c。不能不经处理直接把两种commit身份混用。最终交付远端HEAD与本地HEAD内容树应精确一致；PR保持草稿。

## 8. 已闭合旧方向与下一步边界

旧predictive-decomposition坐标不变性/GRU重参数、96fit分离预测、endpoint教师ridge、SMP零年龄、RRC/SOLID成熟反馈、OHLC、条件持续性/分布几何均保留原结论。上一CDF几何5600配置与1%资源门有其特定对象，不能推广为本轮所有期刊的阈值。

下一研究问题应直接针对现有证据：允许价格新息的延续和回撤；新增学习部分先在历史验证中改善自身初始化，再比较价格Ridge及固定响应。这还只是可检验设计问题，不是已经实现或确认的创新。只扩大超参数直到测试好看不能形成新的独立证据。可以将当前数据用于开发，但需要另行记录的未评分时期/来源来作下一次确认；不复用“confirmation_sealed”历史目录名宣称现阶段仍未暴露。

用户要求实验达标后论文，目前模块增量未达标。因此交付完整研究报告、代码和记录；不写一篇声称模型已成功的正结果论文，也不将任务改成用户未请求的纯泄露批评论文。

