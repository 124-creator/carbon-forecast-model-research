# 2026-10-20 k阶段第一次正式调用失败回执

首次正式运行的训练已在首个asof_close/2018开发单元执行，但保存每原点预测时抛出 `TypeError: numpy.savez_compressed() got multiple values for keyword argument 'rw'`，原因是同时传入显式rw=和包含rw字段的forecast字典。未出现已提交的评分结果；`formal_run/PROTOCOL.json`是该次的本地输出。这是输出封装错误，不是模型公式、目标、验证协议、格点或数据更改。

唯一修复：保存时从forecast展开中排除字符串键rw，rw仍单独保存，与之前在内存计算的RW评分一致。执行前source sha256=0298ae2f37258e6b71239599639d00ac19744e1e287d87fc67e4ba777c36accd；修复后SHA另由哈希命令核验。保留失败目录，修复后向新目录 `formal_run_fixed/`重跑整个预定矩阵，不在测试后改任何配置、模型或数据。

2018-2023历年早已为开发数据，未声称该次断点提供独立确认。这个故障与修复不应从公开报告中删除。

第二次重新运行也在保存首单元时因输出目录 `formal_run_fixed/private/` 未创建而发生 `FileNotFoundError`。这是同一保存层的另一个错误；新增两行 `mkdir(exist_ok=True,parents=True)`，分别确保 `private/` 与 `cells/` 存在。未改变模型、特征、测试、调参或分割。本次再次使用新目录`formal_run_fixed2`，保留失败目录及完整控制台失败日志。本批均为已暴露开发期，修复过程需要如实记录。
