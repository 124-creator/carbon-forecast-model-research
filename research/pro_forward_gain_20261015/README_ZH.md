# Stage O：前推预测收益子空间学习

科学状态：STOP_DEVELOPMENT_GATE_FAILED。本轮仍以碳价格预测模型创新为目标，不转同期关联实证或写阴性结果论文。候选GAIN主相对MSE0.9618004973，普通RANK6为0.8812064943；新增机制未通过。

已完成每环境336次优化器训练+48次冻结编码器解析拟合，共384拟合任务和29预测配置。GitHub运行38717964759完整执行；results_remote/849文件持久进入Git。448个NPZ的5399数组、58主/集成指标和384份数值trace与本地首次完全一致，8份执行源码与前缀相同；不称第三方科学复现或新时间段确认。

入口：PROTOCOL.md、MATHEMATICS_ZH.md、REPORT_ZH.md、LITERATURE_SCOPE_ZH.md、REMOTE_COMPARISON.json。首轮全配置在local_first_summary/all_scores.csv；首轮完整参数/日志与完整远端记录在对话下载ZIP分别保留。results_remote不是限时artifact替代。

只读既有广州前950条截至2021-12-31，源行号/Excel格式非特征。不修改原始价格、不读后续市场。研究分支不合并main。

## 独立环境复现

本轮使用Python3.13.5、numpy2.3.5、scipy1.17.0、scikit-learn1.8.0、torch2.10.0+cpu。请在独立环境安装requirements及CPU PyTorch，不改变其他项目环境。

    python research/pro_forward_gain_20261015/run_all.py --out /path/to/NEW_results --workers 3

输出目录必须不存在。新克隆从已存在仓库CSV定长读取许可前缀，不依赖旧模型参数。全部模型由此重新生成。core.py为唯一新规则，legacy.py复用已检查基础设施及FIXED6/RANK6/GRU，不表示其中未运行的旧模式被本轮重测。

独立算术与代表未来扰动重训：

    python research/pro_forward_gain_20261015/audit.py --out /path/to/NEW_results

结果后几何诊断（无训练）和两端比较：

    python research/pro_forward_gain_20261015/describe_saved.py --out /path/to/results --destination /path/to/POSTHOC_GEOMETRY.json
    python research/pro_forward_gain_20261015/compare_runs.py /path/to/local /path/to/remote --destination /path/to/comparison

结果后诊断不用于选模，完整详细记录在下载包中并可按上面脚本再生。时间日志/环境字符串不要求字节相同；数值一致不等于预测机制有效。
