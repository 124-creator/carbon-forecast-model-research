# Stage N：自适应期限节点的完整实验材料

**科学状态：STOP_DEVELOPMENT_GATE_FAILED。** 不是已成立新算法或可发表结果论文。主报告为REPORT_ZH.md，执行/远端差异见FINAL_STATUS_ZH.md与REMOTE_COMPARISON.json。

本地384次神经拟合，19组传统配置及两种组合，29预测档案；远端按同一固定协议完整重训。只读既有广州前950条、截至2021-12-31。不新下载数据，不打开2022以后，不将源行号/Excel类型作为特征。

目录包括core.py、experiment.py、baselines.py、analyze.py、audit.py、test_core.py与run_all.py；PROTOCOL.md先于新训练提交。results_remote/持久存储849文件，含384份数值模型参数/预测、384份逐轮日志、各模式完整预测、参考、评价目标、组合权重、环境和审计。不是只有限时artifact。

local_first_summary/all_scores.csv为本地首次全部分数。完整本地首次参数与日志另外保留在对话下载包results_local，不以远端更好的结果覆盖首轮。比较工具compare_runs.py需要这两个独立目录；逐数组详细差异在下载包中，也可再生。

## 复现

在独立CPU环境安装requirements.txt对应包，PyTorch使用官方CPU索引。保持原始仓库数据不变，从仓库根运行：

    python research/pro_adaptive_horizon_20261015/run_all.py --out /path/to/NEW_results --workers 3

输出目录必须不存在，程序不会隐式覆盖/续跑旧实验。默认只取既有原CSV的29204许可字节，完整哈希检查后解析950条；任何不一致即中止，不尝试读取后缀。

    python research/pro_adaptive_horizon_20261015/compare_runs.py --local FIRST_RESULTS --remote SECOND_RESULTS --destination NEW_COMPARISON.json
    python research/pro_adaptive_horizon_20261015/describe_saved.py --source FIRST_RESULTS --destination NEW_DESCRIPTION.json

后两个脚本只做记录比较和结果后固定参数描述，不训练、不选第二候选、不变更主成绩。

## 解释边界

候选对RW改善10.76%，但对固定节点仅0.20%，不胜单参数TILT；不能据此宣布新模型创新。平均种子损失与预测集成须区分。文献范围在LITERATURE_SCOPE_ZH.md，不把已有自由节点、插值和神经系数回归称首创。

GitHub工作流完整执行成功，但跨环境不完全逐值一致；数据前缀和七份训练/评分源码一致，具体环境差异原因未被隔离。两端科学裁决相同不等于独立时间段确认。原main未合并、仓库可见性未修改、第三方全文未提交。
