# 实际读取、工具能力与证据边界

日期：2026-10-17。入口固定提交 `1c67e562e721dd82fff4e4143b97a798a1cf44fa`；研究结果基准 `26b4f403c566ee44e5af79cdcfc5b83bc21bd27b`；材料快照 `c78846539e79dc459d3559ae5d0a410ba15fc0d5`。

## 批准材料：21项全部取得且逐字节核验

通过连接的GitHub工具逐文件读取入口、完整执行提示和索引，再按runtime_path恢复下列21项。所有文件的字节数与SHA-256均和固定索引相同。没有声称读取整个仓库；额外目录树元数据用于确认是否存在适用的AGENTS.md/CLAUDE.md，不等于读取那些目录内全部内容。

| 实际恢复路径 | 字节数 | 固定原文 |
|---|---:|---|
| `RESEARCH_STATE_ZH.md` | 7798 | [来源](https://github.com/124-creator/carbon-forecast-model-research/blob/c78846539e79dc459d3559ae5d0a410ba15fc0d5/research/web_pro_handoff_20261017/RESEARCH_STATE_ZH.md) |
| `DATA_SCOPE_ZH.md` | 4162 | [来源](https://github.com/124-creator/carbon-forecast-model-research/blob/c78846539e79dc459d3559ae5d0a410ba15fc0d5/research/web_pro_handoff_20261017/DATA_SCOPE_ZH.md) |
| `EVIDENCE_INDEX.json` | 13444 | [来源](https://github.com/124-creator/carbon-forecast-model-research/blob/c78846539e79dc459d3559ae5d0a410ba15fc0d5/research/web_pro_handoff_20261017/EVIDENCE_INDEX.json) |
| `DATA_ATTRIBUTION.md` | 1665 | [来源](https://github.com/124-creator/carbon-forecast-model-research/blob/c78846539e79dc459d3559ae5d0a410ba15fc0d5/research/web_pro_handoff_20261017/DATA_ATTRIBUTION.md) |
| `data/parsed/figshare_28682348/guangzhou_reported.csv` | 29204 | [来源](https://github.com/124-creator/carbon-forecast-model-research/blob/c78846539e79dc459d3559ae5d0a410ba15fc0d5/research/web_pro_handoff_20261017/data/guangzhou_approved_prefix_950.csv) |
| `research/recurrent_response_20261014/hindsight_diagnostic_20261017/REPORT_ZH.md` | 8057 | [来源](https://github.com/124-creator/carbon-forecast-model-research/blob/c78846539e79dc459d3559ae5d0a410ba15fc0d5/research/recurrent_response_20261014/hindsight_diagnostic_20261017/REPORT_ZH.md) |
| `research/recurrent_response_20261014/hindsight_diagnostic_20261017/ALL_CONFIGS.csv` | 25325 | [来源](https://github.com/124-creator/carbon-forecast-model-research/blob/c78846539e79dc459d3559ae5d0a410ba15fc0d5/research/recurrent_response_20261014/hindsight_diagnostic_20261017/ALL_CONFIGS.csv) |
| `research/recurrent_response_20261014/hindsight_diagnostic_20261017/PROTOCOL.md` | 7127 | [来源](https://github.com/124-creator/carbon-forecast-model-research/blob/c78846539e79dc459d3559ae5d0a410ba15fc0d5/research/recurrent_response_20261014/hindsight_diagnostic_20261017/PROTOCOL.md) |
| `research/recurrent_response_20261014/hindsight_diagnostic_20261017/hd_core.py` | 2419 | [来源](https://github.com/124-creator/carbon-forecast-model-research/blob/c78846539e79dc459d3559ae5d0a410ba15fc0d5/research/recurrent_response_20261014/hindsight_diagnostic_20261017/hd_core.py) |
| `research/recurrent_response_20261014/hindsight_diagnostic_20261017/hd_fit.py` | 7243 | [来源](https://github.com/124-creator/carbon-forecast-model-research/blob/c78846539e79dc459d3559ae5d0a410ba15fc0d5/research/recurrent_response_20261014/hindsight_diagnostic_20261017/hd_fit.py) |
| `research/recurrent_response_20261014/hindsight_diagnostic_20261017/hd_run.py` | 10500 | [来源](https://github.com/124-creator/carbon-forecast-model-research/blob/c78846539e79dc459d3559ae5d0a410ba15fc0d5/research/recurrent_response_20261014/hindsight_diagnostic_20261017/hd_run.py) |
| `research/recurrent_response_20261014/hindsight_diagnostic_20261017/hd_analysis.py` | 8345 | [来源](https://github.com/124-creator/carbon-forecast-model-research/blob/c78846539e79dc459d3559ae5d0a410ba15fc0d5/research/recurrent_response_20261014/hindsight_diagnostic_20261017/hd_analysis.py) |
| `research/recurrent_response_20261014/hindsight_diagnostic_20261017/hd_verify.py` | 13620 | [来源](https://github.com/124-creator/carbon-forecast-model-research/blob/c78846539e79dc459d3559ae5d0a410ba15fc0d5/research/recurrent_response_20261014/hindsight_diagnostic_20261017/hd_verify.py) |
| `research/recurrent_response_20261014/hindsight_diagnostic_20261017/test_hd.py` | 6697 | [来源](https://github.com/124-creator/carbon-forecast-model-research/blob/c78846539e79dc459d3559ae5d0a410ba15fc0d5/research/recurrent_response_20261014/hindsight_diagnostic_20261017/test_hd.py) |
| `research/recurrent_response_20261014/hindsight_diagnostic_20261017/TIMING.json` | 2141 | [来源](https://github.com/124-creator/carbon-forecast-model-research/blob/c78846539e79dc459d3559ae5d0a410ba15fc0d5/research/recurrent_response_20261014/hindsight_diagnostic_20261017/TIMING.json) |
| `research/recurrent_response_20261014/fair_recurrent_tuning_20261016/REPORT_ZH.md` | 5930 | [来源](https://github.com/124-creator/carbon-forecast-model-research/blob/c78846539e79dc459d3559ae5d0a410ba15fc0d5/research/recurrent_response_20261014/fair_recurrent_tuning_20261016/REPORT_ZH.md) |
| `research/recurrent_response_20261014/fair_recurrent_tuning_20261016/fair_core.py` | 4193 | [来源](https://github.com/124-creator/carbon-forecast-model-research/blob/c78846539e79dc459d3559ae5d0a410ba15fc0d5/research/recurrent_response_20261014/fair_recurrent_tuning_20261016/fair_core.py) |
| `research/recurrent_response_20261014/fair_recurrent_tuning_20261016/fair_data.py` | 1991 | [来源](https://github.com/124-creator/carbon-forecast-model-research/blob/c78846539e79dc459d3559ae5d0a410ba15fc0d5/research/recurrent_response_20261014/fair_recurrent_tuning_20261016/fair_data.py) |
| `research/recurrent_response_20261014/fair_recurrent_tuning_20261016/requirements.lock.txt` | 219 | [来源](https://github.com/124-creator/carbon-forecast-model-research/blob/c78846539e79dc459d3559ae5d0a410ba15fc0d5/research/recurrent_response_20261014/fair_recurrent_tuning_20261016/requirements.lock.txt) |
| `research/recurrent_response_20261014/horizon_scale_competition_20261016/REPORT_ZH.md` | 5925 | [来源](https://github.com/124-creator/carbon-forecast-model-research/blob/c78846539e79dc459d3559ae5d0a410ba15fc0d5/research/recurrent_response_20261014/horizon_scale_competition_20261016/REPORT_ZH.md) |
| `research/recurrent_response_20261014/residual_utility_memory_20261017/REPORT_ZH.md` | 6424 | [来源](https://github.com/124-creator/carbon-forecast-model-research/blob/c78846539e79dc459d3559ae5d0a410ba15fc0d5/research/recurrent_response_20261014/residual_utility_memory_20261017/REPORT_ZH.md) |

完整哈希、原仓库路径和固定URL保存于[provenance/SOURCE_READ_MANIFEST.json](provenance/SOURCE_READ_MANIFEST.json)。入口和实际执行提示保存于同目录。

## 阅读深度与实际执行

- 研究状态、数据边界、证据索引、最新报告、四组历史报告和完整98行配置/对照表已用于理解既有证据与限制。
- fair_data、fair_core、hd_core、hd_fit、hd_run和hd_analysis的实际信息路径、成熟标签、选模和评分相关实现已核查。hd_verify及test_hd作为历史验证源码取得并保留；本轮没有冒称重新执行全部历史测试或192次旧拟合。
- 作者实现阅读范围限定为LTSF-Linear的10个指定文件、PROCEED的16个指定文件及FITS模型/训练器/数据工厂/入口/配置/许可证等指定文件。各自源码来源清单随literature和vendor保存。
- NLinear、DLinear原模型类和完整PROCEED核心由新的项目训练入口实际执行；原论文整套基准没有复现。FITS只执行合成形状/算子测试，没有市场训练。最终拟合与在线更新数量以各阶段RUN_COMPLETE和审计文件为准。
- 独立审计不通过再次训练来复算分数；它独立重建批准CSV标签、起点、标准化与统计参考，再核验本轮新产生的局部状态和预测。审计的证据范围与不足明确写入独立文件。

## 可用能力及环境

可用连接GitHub逐文件读取和新分支写入；可检索原论文、正式proceedings与作者代码；可在独立临时工作区执行Python、NumPy和CPU PyTorch。GitHub写权限经工具查得当前连接账户124-creator对目标仓库有admin权限；本轮交付采用独立研究目录与可审查分支，不改写历史结果。

初始环境缺少torch。第一次本地安装因默认缓存目录只读而失败，切换到本次工作区缓存；沙箱网络连接失败后，通过环境提供的正常自动审批取得CPU依赖。随后为作者PROCEED核心安装transformers4.40.2。没有请求密码、令牌或私有模型，没有下载预训练权重。

正式训练环境为Python3.12.14、NumPy2.5.4、torch2.3.1+cpu，固定单训练进程、float64及数学线程1；StageB另用transformers4.40.2。NumPy在正式StageA启动前随依赖安装升级；24次正式拟合在同一进程中记录该版本。StageA计时进程未记录NumPy版本，此项缺口见STAGE_A_ENVIRONMENT_NOTE.json，不用事后猜测填补。

## 访问错误与尚未验证的内容

| 链接/对象 | 实际情况 | 处理与证据限制 |
|---|---|---|
| [本仓库固定提交根AGENTS.md](https://github.com/124-creator/carbon-forecast-model-research/blob/1c67e562e721dd82fff4e4143b97a798a1cf44fa/AGENTS.md) | GitHub 404；固定目录树中也未发现AGENTS.md/CLAUDE.md | 可选约束文件不存在；21项必需材料没有缺项 |
| [PROCEED ACM页](https://dl.acm.org/doi/10.1145/3690624.3709210) | 浏览工具返回Internal Error | 使用公开作者全文v5；不声称从ACM页读取全文 |
| FITS的OpenReview入口（原记录未保留精确URL） | 审查记录为“浏览器验证”，没有保留完整错误栈 | 读取可正常打开的ICLR正式proceedings全文；没有绕过验证，也不猜测补写失败URL |
| [FITS PDF的执行器直接下载](https://proceedings.iclr.cc/paper_files/paper/2024/file/701251e1db4a2e4dd2ef23f5265d5936-Paper-Conference.pdf) | 审查记录为“执行环境网络限制”，完整错误栈未保留 | 浏览工具已能读取正式全文；未保存本地PDF，不伪称下载成功 |
| [N-BEATS官方OpenReview页](https://openreview.net/forum?id=r1ecqn4YwB) | 浏览器验证 | 只把已核代码/成熟基展开范式作为近邻线索，不声称该页全文已读 |
| [LLapDiff arXiv访问](https://arxiv.org/abs/2605.19805) | 一个审查代理得到DisabledError；负责人只取得摘要层线索 | 不以其全文建立创新否定结论，已读S4D数学近邻足够 |
| NeurIPS 2024 Correlated Errors正式PDF（原记录未保留精确失败URL） | 一个审查代理请求超时 | 已核[正式题录](https://proceedings.neurips.cc/paper_files/paper/2024/hash/619b8e3ead58dce90bc615f2a7d5d102-Abstract-Conference.html)，实际读[作者arXiv方法文本](https://arxiv.org/html/2402.01000v4)；未执行其代码 |

TEFL仅确认公开预印本及方法披露。PADRE的负责人和线性审查读取正式摘要，学习机制审查另读取[正式9页PDF](https://www.ijcai.org/proceedings/2026/0503.pdf)相关方法与主要实验；证据层级以各自原始记录为准，未核代码。Label Horizon Paradox核正式题录与作者方法文本。这些近邻核查不等于运行它们的完整预测方案，不增加本轮完整候选与市场拟合数。

没有验证任何方法在未暴露数据上的最终泛化，没有进行期刊分区或订阅政策的当前核查，也没有认证新颖性、录用资格或交易收益。三区/四区传统订阅仍是用户目标；本轮判断聚焦是否已经形成可署名的方法贡献。

## 唯一市场数据范围

只读取广州2018-01-03至2021-12-31的950条批准CSV，29204字节，SHA-256：

`1366ca4814dcdb8aa43296eb64768d1336d6f0fc9088648d6d7ccdbe43318070`

未获取旧完整CSV的2022/2023后缀、其他市场、外生资料、旧逐点预测或旧权重。保留原价、原日期、周末、同价和来源列；来源列只用于溯源，不进入模型。公开包保留代码、训练轨迹和聚合分数；本轮逐点预测及权重保存在local_only，仅供此次核验，未加入远端交付。
