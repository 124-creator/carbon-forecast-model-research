# EWT 信息诊断的公开结果与私有逐点文件

本目录公开固定诊断协议、输入来源哈希、12 个选中配置的聚合指标、未来扰动结果和总汇总。完整逐点预测曾用于独立复算，3,510 条记录全部通过核对。独立审查详见 `../../audits/INDEPENDENT_EWT_DIAGNOSTIC_ZH.md`。

逐点文件包含各 origin 的观测价格变化。按本轮交付范围，已在独立审查完成后复制到：

`/tmp/carbon-literature-aligned-runs-20261019d/information_diagnostic/PREDICTIONS.csv`

复制前后 SHA-256 均为 `d21795622bbb037772ff39f63c79b689ada5ad21721740219ca38efa7c660715`，共 281,843 bytes。确认私有副本逐字节一致后，已移除仓库中的逐点副本。[迁移回执](PRIVATE_PREDICTIONS_MANIFEST.json) 保留原位置、当前私有位置和独立审查身份。

复算独立审查时，以 `--predictions` 显式指定该文件或同哈希的个人本地副本，并让 `--out` 指向一个不存在的新收据文件。源代码与原 JSON 回执记录的是原运行和审查时的位置，不因迁移改写。

`FULL_SERIES_RETROSPECTIVE` 有意使用未来数据，是信息边界对照；它的聚合误差不作为可部署预测能力。该诊断没有重新解释为作者原论文的精确复现。
