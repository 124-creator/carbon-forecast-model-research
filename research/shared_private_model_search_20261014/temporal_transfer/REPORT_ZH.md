# 合成 temporal_transfer 结果

本轮不支持创新或性能达标，停止该路线，不调阈值。执行次序有错误：两次训练均早于实际检索，因此不能标成“先检索后训练”的完整合规实验。所有原始结果保留。

|方法|seed17 test均值|seed29 test均值|
|---|---:|---:|
|SP|0.901444|0.859279|
|PCGrad|0.799883|0.769615|
|adaptive_weights|0.911774|0.920756|
|temporal_transfer|0.951459|0.918977|
|random_tail|0.875195|0.914866|
|private_only|0.994079|0.949670|

完整各任务 core、inner_tail、outer_validation、test MSE和checkpoint在 results.json。候选与同信息adaptive weights差异仅共享坐标定位，不能证明真实任务差异识别。PCGrad两seed均更好，chronological tail没有胜过随机tail，机制增量不成立。
分隔：核心0:48；内部时间尾48:64仅更新；外层验证64:128仅各任务checkpoint；test128:512仅训练结束后评价。归一化核心拟合。第4任务无共同机制，多个方法选择初始checkpoint，应明确报告未学会而非安全通过。
每fit固定1200步、9600梯度调用，12fit，600秒总上限；并非每方法训练满10分钟或同FLOPs。原pilot同340参数和合成生成器，但训练核心缩短，不能与历史分数直接比较。不是严格一阶未来预测，属于时间泛化。
已实际HTTP读取PCGrad/CAGrad/NashMTL/TAG/Ren validation weighting/time-series shared attention，共6个主源HTTP200。搜索有403失败，原始回执保留。
没有市场数据读取、提交或push。