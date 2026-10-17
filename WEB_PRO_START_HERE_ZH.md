# 网页端Pro最新研究交接入口

更新日期：2026-10-17。研究结果基准提交：26b4f403c566ee44e5af79cdcfc5b83bc21bd27b。

本交接包整合已经完成的实验、当前发表困境和下一轮执行要求；没有声称新增市场训练、作者模型复现或已成立的创新。旧START_HERE_PRO_ZH.md及README下方更早的“尚未训练”等文字属于历史，以本页和所列固定结果为准。

## 直接读取GitHub，不需要用户上传附件

交接方式已按用户要求修正：研究资料由本机助手上传GitHub，网页Pro直接读取下面的公开网页和文件。用户只需发送本页链接，不需要下载、解压或上传ZIP。仓库中的ZIP只是上一提交保留的离线快照，不是当前交接前提；其中旧的附件交接要求不再适用。

先读当前完整执行提示词：
https://github.com/124-creator/carbon-forecast-model-research/blob/main/research/web_pro_handoff_20261017/EXECUTION_PROMPT_ZH.txt

随后按网页读取索引访问资料。索引为每个文件提供固定提交、GitHub网页链接、公开原始文本链接、字节数和SHA-256；不要求读取整个仓库：
https://github.com/124-creator/carbon-forecast-model-research/blob/main/research/web_pro_handoff_20261017/WEB_READ_INDEX.json

第一批必读网页：

当前研究状态：
https://github.com/124-creator/carbon-forecast-model-research/blob/c78846539e79dc459d3559ae5d0a410ba15fc0d5/research/web_pro_handoff_20261017/RESEARCH_STATE_ZH.md

数据范围（继承数据约束，不继承旧的ZIP交接步骤）：
https://github.com/124-creator/carbon-forecast-model-research/blob/c78846539e79dc459d3559ae5d0a410ba15fc0d5/research/web_pro_handoff_20261017/DATA_SCOPE_ZH.md

结构化证据索引：
https://github.com/124-creator/carbon-forecast-model-research/blob/c78846539e79dc459d3559ae5d0a410ba15fc0d5/research/web_pro_handoff_20261017/EVIDENCE_INDEX.json

最新实验报告：
https://github.com/124-creator/carbon-forecast-model-research/blob/c78846539e79dc459d3559ae5d0a410ba15fc0d5/research/recurrent_response_20261014/hindsight_diagnostic_20261017/REPORT_ZH.md

完整98行配置/对照表：
https://github.com/124-creator/carbon-forecast-model-research/blob/c78846539e79dc459d3559ae5d0a410ba15fc0d5/research/recurrent_response_20261014/hindsight_diagnostic_20261017/ALL_CONFIGS.csv

批准的广州950条独立数据前缀（不是含后年数据的旧完整CSV）：
https://github.com/124-creator/carbon-forecast-model-research/blob/c78846539e79dc459d3559ae5d0a410ba15fc0d5/research/web_pro_handoff_20261017/data/guangzhou_approved_prefix_950.csv

必须报告实际读到的文件，不能因为看到目录或链接就声称读完。某个链接打不开时，列出确切链接和实际错误，先基于可读材料推进；不要改为要求用户上传ZIP，不绕过安全或授权拒绝。用户尚未通过本助手在网页Pro发起任务，本页不宣称Pro已执行。

## 当前事实

最新反向诊断完成96配置、192次正式拟合，另5次计时与3次边界复训。有限空间的后验冠军c003=0.9138619946970724；此前验证区选配置后合计=1.0203741624182203，统计参考=0.9512831004028871，RW=1。全部98行配置/对照结果保留。它是已暴露广州2020/2021上的诊断，不是新算法、全局最优或独立测试。

此前多尺度竞争和残差效用记忆两个固定候选都未达到晋级条件；不能只引用它们略好于某个消融的数字而略去更强GRU/LSTM对照，也不能把不同刷新策略的分数直接排名。

现在缺的是有证据支持的方法贡献，而不是论文写法。下一轮优先从一个可核实、可实现的完整已发表预测方案出发，真正重新训练，再围绕具体不足提出结构或学习规则改进。完整GRU/LSTM、必要时分解或外生输入仍可考虑；不能要求所有部件首创，也不能把拼接现有模块自动称为创新。

## GitHub资料可以做什么

上述数据、基础代码、最新98行结果和关键报告均有GitHub独立文件，不只存在于ZIP中。数据前缀为2018-01-03至2021-12-31，原样29204字节；只使用索引列出的批准前缀，不读取2022/2023、私有逐点预测或旧权重。历史核验报告是已完成证据，不是网页Pro的新执行结果。

有实际Python/依赖执行权限时，Pro可以直接从索引列出的公开文件取得源码和批准CSV，在自己的独立临时工作区按runtime_path恢复目录后运行；不需要用户从本机提供文件。没有运行权限时，直接交付本机可运行脚本和明确命令，不可虚构训练。此前资料包测试成功不等于本轮192次重新训练。数据规则见DATA_SCOPE_ZH.md，本页的直接GitHub交接方式优先于旧离线步骤。

## 请Pro交付

1. 读到的材料及访问/运行能力清单。
2. 至多三个有直接近邻和实现依据的候选，选一个主方案、一个备选。
3. 主方案具体结构/学习规则、与最近邻的差异、数据需求、强基线与消融。
4. 实际可运行代码及真实小型验证；如无法运行，明确未执行并给本机交接。
5. 当前能否形成论文贡献的诚实判断。不转阴性论文或同期关联实证，不承诺录用。

本次网页读取修订清单：research/web_pro_handoff_20261017/WEB_READ_PUBLICATION_MANIFEST.json，按Git index字节核对，排除自身。上一提交的PUBLICATION_MANIFEST.json和ZIP保留为历史快照，不代表本页修改后的字节。训练结果、检索访问、可用实现、实际执行、新颖性和远端发布是不同证据层级。
