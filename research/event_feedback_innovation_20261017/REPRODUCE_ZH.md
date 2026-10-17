# 复现入口与真实执行范围

本目录包含研究者自己的 RRC 实现、必要对照、冻结协议、聚合输出、独立审核及 EEX 来源准备程序。原始历史结果没有被重写。RRC 已按原规则停止；重放用于核验，不是重启候选或重新获得一次进入确认区的机会。

## 1. 最小目录与依赖

保留当前目录完整结构。四个旧开发 CSV 在前一研究的 `research/cross_market_innovation_20261017/data/` 中，固定基准为 [提交 ee732bdb2a500c5b3f9f3ab3c2a8f251aa530516](https://github.com/124-creator/carbon-forecast-model-research/tree/ee732bdb2a500c5b3f9f3ab3c2a8f251aa530516/research/cross_market_innovation_20261017)。当前 GitHub 研究分支继承它们；不要将旧上海确认 CSV 当成新测试。

训练实际使用 Python 3.12.14、NumPy 2.5.4、PyTorch 2.3.1+cpu，单线程 float64。CPU 模型不需要 GPU、外部 API、私人权重或密钥。EEX 解析需要 xlrd 2.0.1、openpyxl 3.1.5、et_xmlfile 2.0.0；固定清单为 `data_reproduction/requirements-eex.txt`。绘图使用独立现成环境的 matplotlib 3.10.8、NumPy 2.3.5，详见图表回执。

模型核心未依赖 pandas、scikit-learn 或网络服务。`requirements-model.txt` 记录实际模型环境版本；这不是已经在第二台机器重装成功的保证。不同 PyTorch/BLAS 版本可能产生浮点尾差，比较应保留来源、设置及差异，不要用重跑挑选有利结果。

## 2. 只核对公开聚合结果：不下载或读取报价

从任意工作目录执行，输出必须是此前不存在的目录：

```bash
python /path/to/event_feedback_innovation_20261017/source/reproduce_rrc.py public-audit --output /path/to/new_public_audit
```

入口核对原 D1 协议和源代码哈希，再实际调用独立选择程序。它复算所有族内 2022 年选择、2023 年门槛和配参消融，读取的是公开的 2,280 格指标及完成回执，不读取协议内记录的本地报价路径。原始结果应为 `advance=false`。

本轮已经从 `/tmp` 的不同工作目录实际运行该入口，复算结果一致。完整 380 份逐起点预测的独立复算此前已经执行；这一步的便携验证不冒充又进行一次市场训练或第三方重训。

## 3. 来源数据准备

EEX 原报价及派生价格数组不随公共包分发。使用 [数据方法说明](data_reproduction/EEX_DATA_METHODS_ZH.md) 的固定官方 ZIP 入口，或者提供本地已有的同哈希 ZIP。程序仅接受预定 SHA256，官网将来替换同名文件时会停止，不会自动换版本。

```bash
python /path/to/event_feedback_innovation_20261017/data_reproduction/reproduce_eex_data.py check

python /path/to/event_feedback_innovation_20261017/data_reproduction/reproduce_eex_data.py prepare --download --output /path/to/new_eex_workspace
```

开发 CSV 位于新工作区的 `local_only/prepared/development/EEX_EUA_T3PA_daily_development.csv`。2024/2025 按原来源规则机械拆分到独立子目录；数据准备不等于打开确认评价。下面所有模型命令只需要开发 CSV，绝不要求确认 CSV。

四个旧开发文件为：`GUANGZHOU_development.csv`、`HUBEI_development.csv`、`SHANGHAI_D4_development.csv`、`EU_REPORTED_development.csv`。它们的许可、出处和哈希沿用前一研究的文件；不能把它们的许可套用到 EEX。

## 4. 完整 RRC 数值重放

先运行 dry-run 验证全部开发 CSV、源代码和固定设计；不创建输出目录、不拟合。

```bash
python /path/to/event_feedback_innovation_20261017/source/reproduce_rrc.py replay --previous-study /path/to/cross_market_innovation_20261017 --eex-development /path/to/new_eex_workspace/local_only/prepared/development/EEX_EUA_T3PA_daily_development.csv --output /path/to/new_rrc_replication --budget-minutes 40 --dry-run
```

去掉 `--dry-run` 即在新目录执行原来相同的 20 次共同预训练、10 条教师流、260 条学生流和 90 条校准流，再计算原 38 配置的全部结果。输出目录已存在、源文件哈希变化、非有限值或截止时间到达均会停止，不会自动重试。

入口只重定位数据/源码路径并设定这次复核的有限截止时间，保留原始协议字节副本；新 `REPLAY_PROTOCOL.json` 明确标记为事后复现。独立选择程序的原 CLI 只接受原始完成回执，新重放直接复用它未修改的纯评分函数，不伪造历史协议哈希。40 分钟是本轮原预算量级；如果另一台机器更慢，应在运行前确定自己的复核预算。

本轮真实执行的是原 `run_rrc_development.py` 完整 D1；新增便携包装器的 dry-run 和公开审计入口已经验证。没有为了检查这个包装器而重复整个 D1 网格，因此不能声称这个新入口已经在第二环境完成全量重训。

## 5. 合成及独立数学审核

这些检查只使用合成数组，不是市场有效性的替代：

```bash
python /path/to/event_feedback_innovation_20261017/audits/run_audits.py --case core --output-dir /path/to/new_core_audit
python /path/to/event_feedback_innovation_20261017/audits/run_audits.py --case targets --output-dir /path/to/new_target_audit
python /path/to/event_feedback_innovation_20261017/audits/run_audits.py --case events --output-dir /path/to/new_event_audit
```

RRC 原始检查中的一次标签成熟边界 fixture 错误已记录：代码正确拒绝了超出可见边界的 H20 请求，修正合成起点后通过；没有借此修改真实模型或删除正式失败运行。D1 正式实验本身 0 失败。

## 6. 原作者 DSOF 执行锚

`source/author_adapter.py` 和 `source/run_author_anchor.py` 是本方因果接口；它们调用固定作者提交 `11cebd980a1f77ef0e1db9ac313dbfe6d37984e9` 中未修改的模型、ER、TD 和梯度更新方法。需要的 12 个文件及 Git blob/SHA256 在 `provenance/AUTHOR_DSOF_SOURCE_MANIFEST.json`。

作者仓库当前没有声明可用源码再分发许可证，因此其原源码不在公共包中。自行取得该固定作者源码到自己的本地引用目录后，可按 `provenance/AUTHOR_EXECUTION_PROTOCOL.json` 重定位作者路径、开发路径及新输出位置，在新的复核协议下运行；不要覆写历史 receipt。

本轮实际完成 1 次计时和 1 次正式作者执行，正式锚为 2022/2023 六格平均 1.118697。它使用作者 DLinear/MLP 默认核心及原价训练前缀标准化；不是原完整 benchmark 驱动器或原数据集复现，也不是与 GRU 匹配的主优势基线。

## 7. SOLID 备用参照

原作者固定版本、MIT LICENSE 及必要源码在 `backup_review/author_solid/`，选择与非 AMP 更新的精确行为见 `backup_review/SOLID_THEORETICAL_REVIEW_ZH.md`。相应适配器、有限运行协议和实际输出单独记录，绝不与 RRC 主实验的运行数、选择时点或停止决定混写。最终执行与未覆盖范围以 `RESULTS_SOLID_ZH.md` 为准。

完成上述 D1 重放后，使用新入口核对该次离线初态并准备 SOLID 新复核目录：

```bash
python /path/to/event_feedback_innovation_20261017/source/reproduce_solid.py prepare --rrc-replay /path/to/new_rrc_replication --previous-study /path/to/cross_market_innovation_20261017 --eex-development /path/to/new_eex_workspace/local_only/prepared/development/EEX_EUA_T3PA_daily_development.csv --output /path/to/new_solid_replication --budget-minutes 40

python /path/to/new_solid_replication/source/reproduce_solid.py run --protocol /path/to/new_solid_replication/provenance/SOLID_REPLAY_PROTOCOL.json
```

`prepare` 只核验与准备，`run` 才执行原作者适配。可以先用 `--paths-only` 仅检查路径，或 `--dry-run` 完整核验但不创建输出。入口只引用按原离线验证规则选出的 teacher，分别记录 NPZ 文件哈希、教师参数哈希及与原 B1 的异同；不会读取最后在线状态来替换离线初态，也不会用 B1 成绩重新选择。

新入口已通过 20 项纯合成检查、现有目录 60 个必要文件路径检查和独立静态审阅，没有额外进行市场适配。新重放需要自己的逐预测独立审核，不能继承原 B1 的 PASS。预算从准备时开始计时；完整限制与身份规则见 [SOLID 可移植重放说明](backup_review/REPRODUCE_SOLID_ZH.md)。

## 8. 公开文件与本地精确记录

公共包提供全部聚合成绩、配置/协议、研究者代码、独立审核和统计科学图。EEX ZIP、原表、逐日/逐场报价、逐起点绝对价格预测、教师缓存和训练权重不公开分发。完整本地记录可在取得来源后重新生成；代码成功并不授予 EEX 数据再分发许可。

运行截止是每轮/事件检查的协作式截止；Python 执行器会等待正在进行的算子或 I/O 退出，并非到点强制杀死子进程。原 D1 已在预算内完成。这个限制不影响已报告的完成数，但后续复核不应称其为硬实时保护。
