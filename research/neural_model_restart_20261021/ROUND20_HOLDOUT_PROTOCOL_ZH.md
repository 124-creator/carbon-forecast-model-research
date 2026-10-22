# Round20 2026保留期确认协议

本协议在首次构造2026模型目标和评分之前固定。当前2025是反复使用的开发期，不能作为独立确认。原始ETF快照已取得，但未生成2026模型预测或评分。

## 模型与数据冻结

- 主候选固定为Round20 joint的三seed预测均值；seed11/22/33已选新训练epoch分别10/12/2。旧GRU warmstart分别来自26/26/28，不再重选。
- 主匹配控制是continued：同一warmstart、同样scalar与三个正bucket head、同样优化规则，epoch11/13/2。clock_only与density_only为预定消融，不能在看到测试后替换主候选。
- 所有市场训练参数只用2024及以前标签；2025用于选epoch。2026不重训，不再估计正常化、bank、风险状态参数或volume AR参数。允许使用每个预测起点已经公开的OHLC/volume推进因果状态。
- 使用冻结的KRBN/KCCA原快照及1/5/22个后续交易观测的原始range目标。共同origin必须有成熟的22日标签，origin日期从2026开始，label_end不超过源快照末日。
- builder预处理digest为85bd63bb036677e8cd6d2c2941bd1ddbc34d86df15f9a8441f5b09fabd15a513。运行前另存selection JSON，绑定runner、数据、源代码、校准参考、checkpoint与比较输入的SHA256；本协议不自行创建允许测试的selection。

## 预定比较

除四个Round20组外，报告原GRU、R16 raw/fixed滤波、MEM stationary/integrated、同信息HAR/HGB、GK/RS测量滤波与EWMA22。所有模型使用同日期与同目标。凡重新构建的旧基准，必须在开发段复现其已存预测，不允许借未来数据调参或改用较弱定义。

## 指标和判断

主指标为两个资产×三个期限等权的QLIKE=log(v)+y/v。主差值固定为joint减continued，负数表示算子候选较好。附所有资产/期限、三个seed配对方向、normalized QLIKE及MSE。先平均seed预测再计算损失，seed不算独立数据集。

方向性确认要求主差值为负、至少两个配对seed同向，并公开每资产/期限方向，不能以单格最优或改用MSE挽救主指标失败。是否具备实际方法价值还须结合效应幅度、最强非算子基准和消融，不把数值噪声或普通GRU优化收益算成算子创新。M的市场增量目前未成立；即使joint胜出，也不能据此自动声称M有贡献。

使用同步日期非循环移动区块bootstrap，B=2000、block22/44，两资产/三期限/各比较共享每轮日期索引。区间反映有限保留期不确定性；跨0时只能称点估计改善，不能写显著或稳定优势。两ETF有共同CCA敞口，不当作两个独立市场确认。不对负raw QLIKE均值直接取百分比。

无论结果好坏，保留全比较；不根据2026重选seed、epoch、模型、期限或资产。若主候选未改善匹配控制，如实判该次确认未通过，继续独立新研究，不把消融或其他指标的最好数值改成主结论。

## 实现前提与已知范围

runner默认只做development-only重放，真实test需明确selection文件。模型参数和decoder/AR/bank保持固定，仅延展观测数据buffers。先产生封存预测，再统一评分；source hash运行前后核对，输出拒覆盖。

创新候选限于受约束观测算子及其参考矩、未来目标倍率和公共尺度的联动，不是新的time-change、有限采样定理、Kalman或GRU。Gaussian条件矩与未来volume/risk创新独立是工作假设；不声称识别真实交易次数、活动时钟、测量噪声或IV。基础实验的完成和论文可发表程度分开判断，不能保证期刊接收。
