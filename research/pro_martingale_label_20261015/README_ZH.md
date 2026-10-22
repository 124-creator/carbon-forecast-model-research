# Stage P：保持均值标签平滑的分布式多期限预测

**科学状态：STOP_DEVELOPMENT_GATE_FAILED。** 模型创新仍是目标，但本轮候选未胜普通Gaussian标签、同结构MSE和强基函数模型，不进入结果论文。

阅读顺序：REPORT_ZH.md → MATHEMATICS_ZH.md → LITERATURE_SCOPE_ZH.md → PROTOCOL.md。完整源码8份、前置测试和全部数值材料；远端执行是否完成及与首轮差异只以FINAL_STATUS_ZH.md/REMOTE_COMPARISON.json为准。

在完整仓库根目录、新环境中安装requirements.txt，然后执行：

    python research/pro_martingale_label_20261015/run_all.py --out /absolute/path/to/NEW_results --workers 3

必须指定不存在的输出目录，防止覆盖。默认只读取既有data/parsed/figshare_28682348/guangzhou_reported.csv前29204字节；也可用--prefix指向本交付包的approved_prefix.csv。输出30组完整预测、432份模型参数及日志、指标和审计。仅读到2021-12-31；不下载价格/权重，不读取后续年份或其他市场。

仅预测点价格；概率表不是已经校准的价格密度。生成器、辅助损失和共享期限机制的已有边界详见数学报告。失败不意味着论文可以用阴性实证替代，当前只是研发归档。

results_remote为GitHub新环境完整重训；首轮全数组单独保留在下载包results_local。远端若与首轮不一致，必须披露，不能选择较好的一组替换。
