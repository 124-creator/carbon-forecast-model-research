# 碳价格项目既有路线与数据独立接续审计

审计日期：2026-10-19（Asia/Shanghai）。审计人：本轮并行只读审计代理。

## 1. 实际仓库基线与读取范围

通过当前 GitHub connector 直接核实：

| 对象 | 当前身份 | 状态 |
|---|---|---|
| `main` | `1c67e562e721dd82fff4e4143b97a798a1cf44fa` | 仍为 10 月 17 日 handoff，README 不是当前全部研究状态 |
| PR #19 | `research/prior-transport-learning-20261018`，`b05939a45252ec65ba3fcf3b454dfc1804850972` | Open、Draft；基于 `research/distribution-shape-20261017` |
| PR #20 | `research/auction-clearing-innovation-20261018`，`4cc0919a749e5263bba717f4bfdbafb26a3df77c` | Open、Draft；叠加在 PR #19 上，尚未合并 |

当前工作副本为主代理下载的 `carbon-research/`，位于 PR #20 固定提交。本审计先通过 connector 读取正文/代码，随后只读本地副本。该提交递归 Git tree 共 3,120 个条目；未发现仓库内 `AGENTS.md` 或 `CLAUDE.md`，main 根 `AGENTS.md` 请求亦返回 404。不能把早期报告引用的 Windows 祖先目录 `E:/碳价格/AGENTS.md` 当作本工作区已经存在的约束文件。

读取了用户交接文本、PR 元数据、现行结果/方法/来源说明、相关模型源码及四个明确截至 2023 年的开发 CSV。没有打开 EEX 2024/2025、全国 LNCEA 2024 或 ScienceDB 尾段的报价、预测、权重；没有重算任何确认分数，没有修改既有实验文件或向 GitHub 写入。PR 及旧主报告中的既有确认用途陈述属于已公开研究历史，和新读取确认原始价格不同。

## 2. 对上轮 SDMC/FDMC 的独立代码理解

PR #20 不是一条已成功的模型路线。公开部分新增 119 个研究文件，只含代码、冻结协议、汇总与图表；prepared 事件输入、逐起点预测和权重不在该提交中。`reproduction/reproduce.py` 需要独立完整证据包，不能凭全新 Git 克隆声称已回放 63 个最终模型。

本轮核读 `prepare_examples.py` 及 `scale_separated_revision/factorized_models.py`，确认以下关键语义：

- 预测在上一拍卖日后一天的 00:00 UTC 签发，输入允许上一日及以前结果。这是基于历史最终报表的可得时点工作假设，尚不是逐场首发版本认证。
- h=1/5/20 是签发时已知完整日历版本中的第 h 个计划拍卖事件。代码先固定目标日期/平台，随后查询结果，未把“下一成功拍卖”偷换成选择规则。
- 原始 SDMC 使用过去 20 个拍卖事件的 39 个特征及 16 个 context；未来实际投标量只作为标签。官方 Bid Median 未被认证为数量加权分位点或二级市场价格。
- FDMC 用官方同场统计量 M 和清算比率 Q=P/M；未来 M 只进入辅助标签。`prepare_factorized_examples.py` 用过去 60 场的 |P−M| 中位数构造微观尺度，所有新尺度和 M0/P0 context 也提供给对照。
- FDMC 在给定编码状态下独立组合共同 M 分布与微观 Q 分布。`FactorDirect` 解除 Gamma 清算但仍在当前有限分解族内，不是任意自由联合分布。
- 标准 SDMC/FDMC 主训练目标用各起点波动尺度归一化后的 MSE，选型和主报告则使用原价年度×期限相对 MSE。不同风险权重已在更早审查中明确，不宜再次把普通 WLS 修正命名为创新。

FDMC 相对 RW 主分数 1.004985，六项原门槛未通过；尺度消融虽有支持，不能证明随机清算必要。关键新路线 `docs/SINGLE_ROUTE_PROPOSAL_ZH.md` 明确仍未落实两类新增观测：同一 EUA 资产且时间/合约可核实的二级价格锚；同一有效投标池的数量加权分位或深度约束。重用 mean、median、总投标量不自动补足这两个缺口。

## 3. 稀疏交易、有效价格、随机时钟方向其实已做过什么

| 方向与证据路径 | 已真正执行的内容 | 结论及不能重包装的内容 |
|---|---|---|
| `deep_direction_review_20261013/calibration/calibrate.py`、`REPORT_ZH.md` | `filter_bank` 为 p=m+z：m 永久随机游走，z 为 AR(1) 暂态；31 个 rho×永久方差份额的有限网格，Gaussian 创新方差剖面消去；120/240/480 窗口，状态逐观测过滤 | 已有真实低维潜在持久/暂态价格基线；MLE480 广州旧开发分数约 0.949793。它不是从未试过的新有效价格模型；经济潜状态未被识别 |
| `local_candidate_next_20261013/DESIGN_ZH.md`、`REPORT_ZH.md` | 随机斜率+隐含刷新仅提议未实现；实际执行 8 个趋势、同价持续性、已发生日期间隔与暂态交互 Ridge 探针 | 报价刷新不可仅由日期/价格认证；2021 同价组只有 2 个评价起点。不能将同价自动解释为未交易/陈旧报价 |
| `recurrent_response_20261014/predictive_observer_market/REPORT_ZH.md` | 72 次真实拟合：共同块输入 GRU、RKN 常数与条件观测方差；另 3 次真实后缀扰动重训 | RKN 常数 0.9903408，条件 0.9897715，机制 gate=false。条件方差不能解释成已校准的市场微观噪声 |
| `recurrent_response_20261014/innovation_state_learning/REPORT_ZH.md` | 149 个拟合块，候选为 20 维移位创新状态+GRU 学习增益；部分 LSTM 未完成 | 候选 0.9563259，取消反馈 0.9535541，取消移位 0.9540572，常增益 0.9527890，普通 GRU 0.900363。必要部件没有优势 |
| `cross_market_innovation_20261017/methods/business_clock/THEORY_DECISION_ZH.md` | 随机业务时钟+条件半群的理论核查，没有该模型的真实拟合 | 独立 OU+指数时钟的事件步条件均值严格退化 AR(1)；内生有限状态时钟是已有 MAP 核；Latent ODE、Neural Jump SDE 为直接近邻。不能靠换名和新损失权重主张创新 |
| `structural_innovation_20261017b/REPORT_ZH.md`、`methods/SMP_IMPLEMENTATION_ZH.md` | 实际五序列 SMP：符号×零变化年龄共 9 个名义状态、Markov reward 多步传播、GRU、hurdle 辅助，以及 no-clock 等消融 | SMP 全模型 1.323787，one-step 1.035092；EEX 全模型 2.392421，h20 4.411899。活动可预测不等于有符号均值可预测，不能重提年龄门控即可提升 |
| `asof_information_probe_20261013/REPORT_ZH.md` | 仅明确发布时序的合成：已知参数/噪声的状态增广 Kalman；不是真实外生市场实验 | 没有真实信息增量证据；未知方差学习亦不能直接冒称新机制 |
| `recurrent_response_20261014/release_event_learning/REPORT_ZH.md` | 广州950前缀，123份有发布日期支持的煤价公告；price/clock/hold/event四方案两种子完整，其他方案中止 | 仅碳价 0.900363，新公告单次注入 0.944243；事件注入固定实现停止。不能宣称所有异步外生信号已经被否定 |

上述数值分别属于不同实验协议，不能跨行做统一优胜排名。本审计核实这些事实与代码结构，没有从旧逐点预测重新计算这些成绩。

**仍然存在的判断边界：** 仓库没有证明所有状态空间、有效价格或随机时钟方案不可能成功；它已证明多项具体实现无收益，且为宽泛命名给出了数学近邻。若新候选只是局部趋势/滤波器+时间门控，应先写出它和以上模型的实际学习算子差别，不能用“以前失败所以需要更复杂”充当依据。

## 4. 当前可以直接读取的开发 CSV

本审计本地读取以下四个明确的开发文件，核对记录数、日期、列名和 SHA256；未加载原始三份含 2024 后缀的全量 CSV。来源均为作者存储库的 CC BY 4.0 声明，价格语义和历史版本仍须分别说明。

| 路径（相对仓库） | 行数 | 日期范围 | SHA256 |
|---|---:|---|---|
| `research/cross_market_innovation_20261017/data/GUANGZHOU_development.csv` | 1,434 | 2018-01-03—2023-12-29 | `8d78a4f13c5a799519546fd3bfa0e0ac5eaa52b0441a983a95ef06f5e5e34051` |
| `research/cross_market_innovation_20261017/data/HUBEI_development.csv` | 1,408 | 2018-01-02—2023-12-29 | `04655d59c1b585f6c35c02a43469a965fe3589eb613d7d8d8e46996751b7d797` |
| `research/cross_market_innovation_20261017/data/EU_REPORTED_development.csv` | 1,515 | 2018-01-02—2023-12-29 | `ca7c192380897076ef57516a11ea2ccefda6ba92ab9823d9aa8f2bb281cefedc` |
| `research/cross_market_innovation_20261017/data/SHANGHAI_D4_development.csv` | 1,507 | 2013-12-19—2023-12-29 | `425424353bb3ac09e3d96b291680b4179e86f180da626a5d19e5c1e58ed960a1` |

合计 5,864 条作者报告价格，不是 5,864 个独立市场实验。广州与官方 endPrice 全来源逐值相同；湖北主要对应 Latest 但有4日期差异和23日期官方快照不可用；EU_REPORTED 品种、货币、期货连续构造未核实；上海单位和作者预处理尚未完全认证。

其他可复用资源：

- EEX 官方档案：2012–2023 有2,269已完成场，另3取消；旧日聚合2,255日。原件及派生报价不在公开仓库，`event_feedback_innovation_20261017/data_reproduction/reproduce_eex_data.py` 可恢复全档案，但会机械解析确认年份。若本轮继续隔离确认，优先复用 PR20 开发专用 `parse_auction_events.py`，仅打开2012–2023成员，不直接执行全档案准备器。
- 全国 LNCEA：Mendeley `10.17632/5skvz8sgrz.1` 有760作者处理后对数记录，598开发、162留存。多变量尚未进入主要真实模型，但只有处理后列，不能把反指数输出叫原始未经处理价格。
- PLOS `10.1371/journal.pone.0296105.s001`：深圳1,893有效作者收盘值、398带日期空单元格；广东1,706、湖北1,868、上海1,198。上轮仅机械核验，未导出/拟合。这是可下载新来源，但不自动是全新市场时期或未经处理官方行情。开发前须决定纳入和外部验证切分，不能看成绩挑来源。
- ScienceDB 7市场4,196日均价：上游整月异常识别及替换规则使其不适合作严格实时主证据；844条旧尾段继续单列。
- README 提到的 4,861 行 EUA 原来源已经定位：`research/stageC_20261013_review/NEW_EUA_ACQUISITION_AND_AUDIT.json` 指向 `Chen-zhongxiu/Carbon-price-data` 的固定提交 `6e3ea533832f82fd1c8b4859bb2af8c33bd19a31`、`EUA.csv`，2005-05-19—2024-04-08，166,738字节，SHA256 `86b24f12d006e7c3a0a79ab9035f7d9750b4755f1b69071a3425c41a1c9bed16`。旧审计只确认单价格及 type=EU ETS，没有认证 OHLC；仓库无许可声明，spot/futures/venue/roll/units亦未解决。作者 `Section 7-forecasting/EU-one step.R` 引用此表，但上轮所读单份代码未给定义。可作为继续核查来源论文的线索，不能把它直接称为4,861条已认证OHLC或新官方连续期货。

## 5. 确认状态必须沿用实际用途

`auction_clearing_innovation_20261018/docs/CONFIRMATION_EXPOSURE_STATUS_ZH.md` 是最新核读的用途链：

- 上海2024–2025共261条已经用于 BCM 确认拟合、评分和公开交付，不能再次称未见确认；本工作区虽包含对应文件，本审计未打开其价格。
- EEX2024/2025各213个日聚合观察尚未用于模型评分。原档案被机械解析过，因此只能称“未参与本项目选模/评分”，不能称从未被任何程序看到。
- 全国LNCEA2024、ScienceDB预留尾段尚未用于项目模型评分；来源处理、跨源时期重复等限制仍在。
- 原广州/湖北/EU作者序列的2024后缀共178条按旧协议保守归类，不是新的独立确认资源。

SDMC/FDMC旧门槛未通过，因此不能以继续实验的概括授权为理由自动重跑它们并打开2024/2025救分。若真正改变目标/模型问题，需写成独立新协议并诚实披露既有暴露，而非追改旧冻结协议。

## 6. 一个最小科学诊断建议

**优先检验“真实新增的历史供体价格是否改变可预测条件均值”，不要立刻再实现报价冻结或有效价格潜状态。** 这是进入模型创新前的一次可停止的信息诊断，不把诊断本身算作新算法。

可立即从四个已在仓库的合法开发 CSV 开始，保留原价目标和原始观察步 h=1/5/20。固定按目标日期 q 只接纳供体 d<q 的记录，传入最后可用增量、历史短窗口统计、年龄和缺失mask；不得将最终数据库快照包装成逐时点实时版本。先选择一个目标或事前固定三国内目标，不让EU身份不明成为经济结构约束。

最少比较：目标历史 Ridge；同预算目标+供体 Ridge；只加供体日期/年龄/mask的对照；同信息普通 GRU。评价按预定年度×期限的原价相对MSE，并保留所有格。年龄/异步对齐本身已有理论和源码近邻，故唯一进入新模型的条件是供体价格在日期对照之外仍有稳定的可检验条件均值增量，而且灵活普通模型没有完全吸收拟主张的新约束收益。

这项诊断不同于旧 BCM（主要是训练共享，不是活供体预测输入），也不同于已有煤价公告实验；已有 `structural_innovation_20261017b/literature/literature_crossmarket/CROSSMARKET_REVIEW_ZH.md` 只完成其文献与源码审查，明确未实际拟合。但该审查已经指出 GRU-ODE-Bayes、CRU、神经ECM都是直接近邻，因此新方法不能只是这三者的拼接或重命名。

如果主代理已能取得身份清楚、覆盖确认期的 EUA 二级价格，应将这个诊断换成 EEX 目标的同资产价格锚测试，先比较最后可用二级价和简单 ARX，再决定联合清算是否还剩下可估计的贡献。只超过拍卖随机游走、没有超过直接复制二级价，不构成模型创新。部分数量联合约束仍需新的数量数据；没有此类观测时不建议推进旧提案的似然和联合潜变量实现。

## 7. 本轮可复用代码位置与应保留的边界

| 用途 | 文件 | 复用方式 |
|---|---|---|
| 明确开发事件解析 | `research/auction_clearing_innovation_20261018/parse_auction_events.py` | 年份严格2012–2023，保留取消，不复制确认读取路径 |
| 历史日历纠正 | `calendar_correction_20261018/parse_calendar_vintages_corrected.py` | 继承已核实2020脚注特例，不再用原错误解析器 |
| 因果目标与输入构造 | `prepare_examples.py`、`verify_example_causality.py` | 若目标仍同一事件任务可复用，其他任务应另写新协议 |
| FDMC正确尺度和同标签对照 | `scale_separated_revision/prepare_factorized_examples.py`、`factorized_models.py` | 只能作为已失败可复算基线，不自动再推广候选 |
| 四市场已有普通训练和评分 | `cross_market_innovation_20261017/source/data_core.py`、`fit_core.py`、`statistical_reference.py` | 克隆后改新目录/入口，不覆盖原冻结源码与输出 |
| 低维永久/暂态基线 | `deep_direction_review_20261013/calibration/calibrate.py::filter_bank` | 可移植为明确已知基线；旧runner有Windows归档路径，不能直接当便携程序运行 |
| 标准异步近邻 | `structural_innovation_20261017b/literature/literature_crossmarket/author_gru_ode_bayes/` | 源码静态参考；其论文/代码KL方向差异已记录，未完整复现训练依赖 |

所有旧STOP、预算中止、结果暴露与校验记录应保留；新研究写入新增目录。此次审计没有给出新模型的成功结果或新颖性清除证明。
