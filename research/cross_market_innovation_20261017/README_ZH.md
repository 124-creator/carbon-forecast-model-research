# 扩展数据与 BCM-GRU 研究包

**先读 [REPORT_ZH.md](REPORT_ZH.md)。本轮没有得到已验证的可发表模型创新。** 已完成四市场开发、冻结后的上海确认、机制消融、学习曲线及独立复算。BCM-GRU 相对普通共享 GRU 的确认分数改善约 1.39%，但匹配的常规训练增强数值近似相同且点估计略优；两项主要优越性检验均未通过。完整报告保留这一结论，不将候选名称视为创新成立的证据。

本目录是对用户“扩大数据、提出创新模型并实际实验”要求的新增研究交付。基于主仓库固定提交 `1c67e562e721dd82fff4e4143b97a798a1cf44fa`，与上一轮 `research/web_pro_execution_20261017` 及其草稿 PR 分开保存。

## 1. 已完成的工作

| 项目 | 实际规模与状态 |
|---|---|
| 四条主序列来源库存 | 6,303 条；其中 178 条既有 2024 后段未进入本轮 |
| 实际进入本轮的数据 | 6,125 条＝5,864 开发＋261 上海确认 |
| 正式预测器拟合 | D1 141＋D2 158＋学习诊断 24＋确认 62＝385 |
| 历史补全器 | 开发 9＋确认 6＝15 次；每次 500 步 |
| 额外计时 | 3 次真实市场短训练，单列记录 |
| 确认起点 | 2024 年 143、2025 年 78，共 221；同年完整 20 步目标 |
| 正式拟合失败与重试 | 0 |
| 确认结论 | 新增条件矩机制的净增益未得到支持 |

其他实际取得的全国对数价格 760 条、七试点成交均价 4,196 条有独立来源与导出代码，未混入主实验或计作主模型训练数据。

## 2. 主要文件

| 文件或目录 | 用途 |
|---|---|
| [REPORT_ZH.md](REPORT_ZH.md) | 数据、创新假设、全部关键结果、统计解释和发表判断 |
| [methods/METHODS_ZH.md](methods/METHODS_ZH.md) | 可用于后续论文写作的方法定义及数学边界 |
| [methods/RELATED_WORK_ZH.md](methods/RELATED_WORK_ZH.md) | 直接近邻、已有成分、理论反例及候选贡献边界 |
| [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md)、`data/` | 五个实际输入 CSV、作者署名、CC BY 4.0 来源和变换 |
| `data_provenance/` | 新数据来源、原件哈希、提取器、25 CSV 导出审查、历史封存记录 |
| `frozen_d1_source/`、`d2_source/`、`confirmation_source/` | 各阶段原样冻结的模型与训练源码 |
| `source/` | 初始开发源码快照，按原样保留；正式 D1 复现使用 `frozen_d1_source/` |
| [results/ALL_CONFIGS.csv](results/ALL_CONFIGS.csv) | 49 个开发聚合配置组，含各种子、市场、年份与期限 |
| [results/ALL_FITS.csv](results/ALL_FITS.csv) | 全部 385 正式预测器的配置、任务、训练与收据索引 |
| [results/ALL_METRICS.csv](results/ALL_METRICS.csv) | 7,698 个单元指标行，含各视图与确定性控制，不把它们计为独立训练 |
| [results/CONFIRMATION_MODEL_SUMMARY.csv](results/CONFIRMATION_MODEL_SUMMARY.csv) | 13 个确认模型的整洁汇总 |
| [results/CONFIRMATION_FORECASTS_H1_H5_H20.csv](results/CONFIRMATION_FORECASTS_H1_H5_H20.csv) | 21,879 行逐点确认预测；确定性控制不伪造三个随机种子 |
| [results/CONFIRMATION_PRIMARY_INPUTS.npz](results/CONFIRMATION_PRIMARY_INPUTS.npz) | 三个主要模型、三种子、20 期及真实目标的统计输入 |
| [CONFIRMATION_INFERENCE.json](CONFIRMATION_INFERENCE.json) | 原主检验、20/60 块长敏感性、边际区间和次要 MSE 差 |
| [results/LEARNING_CURVES.csv](results/LEARNING_CURVES.csv)、`figures/` | 两种时间取样方式的学习曲线、确认图；SVG 与 PNG |
| `audits/` | 独立 NumPy 前向及统计复算、来源和范围说明 |
| [reproduction/REPRODUCE_PORTABLE_ZH.md](reproduction/REPRODUCE_PORTABLE_ZH.md) | 创建新隔离目录并重新训练的便携入口 |
| [PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json) | 公开文件的大小、SHA-256 与 Git blob SHA；不包含自身以避免自引用 |

`data_provenance/` 中“继续封存”等文字是数据取得阶段的历史记录；本轮确认已经结束，实际用于确认的 CSV 现已公开。这些文件没有倒改成事后状态。不得将再次运行公开数据称为另一次新的盲确认。

## 3. 原始训练记录与确认数组

为了保留原始字节并控制可浏览文件数量，逐拟合训练收据与大数组放在两份归档中：

| 归档 | 精确保留内容 | 文件数 | 字节数 |
|---|---|---:|---:|
| [records/RUN_RECORDS.tar.gz](records/RUN_RECORDS.tar.gz) | 运行前计划、开始/完成、全部训练收据与轨迹、完整 D1/D2/C/学习结果、计时与修正记录 | 447 | 2,629,381 |
| [records/CONFIRMATION_ARRAYS.tar.gz](records/CONFIRMATION_ARRAYS.tar.gz) | 62 个确认预测器状态、62 份全部视图/20 期预测数组、6 个确认补全器状态 | 130 | 7,177,927 |

逐成员相对路径和 SHA-256 分别在相邻的 `*_MANIFEST.json` 中。解压会恢复原研究相对布局，包括 `d1_receipts/` 等与 `local_only/confirmation/`。在研究目录下操作：

```bash
tar -xzf records/RUN_RECORDS.tar.gz
tar -xzf records/CONFIRMATION_ARRAYS.tar.gz
```

建议先按清单核对归档 SHA-256，并在一个研究目录副本中解压。归档成员均为本研究的已知相对路径；没有符号链接。根目录可浏览版本未重复存放最大的 `D1_RESULTS.json`、`D2_RESULTS.json`，它们的原始字节包含在第一份归档中，解压后恢复。

开发和学习阶段较大的原始预测数组/权重不包含在公开数组归档中。已发布对应冻结代码、全部数据、种子、配置、收据和独立审核输出，可通过新工作区重跑生成。**不能声称只下载本包就已经拥有全部开发权重，也不能把独立保存状态复算冒称完整重训练复现。**

## 4. 重新运行

完整指令及直接依赖固定版本见 [REPRODUCE_PORTABLE_ZH.md](reproduction/REPRODUCE_PORTABLE_ZH.md)。最直接的数据入口是本包 `data/` 中五个按字节核对的文件，不需要重新联网下载。

示例中的路径应替换为实际克隆位置和新的空目录：

```bash
python -B /path/to/repo/research/cross_market_innovation_20261017/reproduction/prepare_reproduction_workdir.py \
  --study-dir /path/to/repo \
  --data-dir /path/to/repo/research/cross_market_innovation_20261017/data \
  --output /your/path/new_carbon_study
```

准备副本不训练。后续按便携说明依次运行 D1、D2、开发汇总、学习诊断和 C，以及确认汇总；学习诊断不进入确认配置选择。原始研究模型在当前环境已经完成，便携启动器已做源码、合成数据和路径测试；发布前另用这五份公开数据成功准备了一个新的工作区，核对复制了 65 个文件，冻结来源保持一致。**没有再做第二轮端到端真实市场训练**。

历史 `requirements.txt` 保留研究所用 CPU 依赖目录的已安装包集合，其中包含此前安装但当前四市场模型无需使用的包。便携复现优先使用 `reproduction/requirements-portable.txt`。训练和独立审核使用 NumPy 2.5.4，原始最终统计与绘图使用 NumPy 2.3.5；这一差异在报告中明确保留。

保存状态的独立复算说明见 `audits/`。审查器包含各自运行期路径和依赖约定，须按该目录说明配置；确认数组已经全部提供，完整开发前向核查还需重建未归档的开发数组。

## 5. 研究完整性与当前边界

确认前已固定的主要比较是 BCM 对开发选出的普通共享 GRU、BCM 对匹配增强；不是从 13 个模型事后选一个最弱对手。主评分对年份与期限等权，平均种子损失，不是种子预测集成。主检验块长 40，10,000 次，两个主要比较 Holm 校正；20/60 块长只作敏感性。

上海确认只来自一个新增作者来源，2025 年仅 78 个相关起点，40 步块长的有效长度约 1.95 块；年度更新也会引入跨年训练依赖。因此结果不支持任意市场、任意状态的泛化断言。数据量增大、训练次数增多、方法公式完整和数值审核通过，都不能替代新增机制本身的有效性证据。

研究目标仍是模型或学习算法创新。下一步需有不同且可验证的机制假设，以及更可靠、尚未用于方法选择的外部资料；本轮确认数据已见，不可反复调参后继续标记为独立未见测试。
