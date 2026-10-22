# Round 17：未来前瞻 logging 的输入准备记录

本次只取得已完成的行情输入，不包含任何预测、未来目标、标签、损失或评分，也没有设置自动执行任务。

## 新输入及 as-of

- [KRBN 2026-10-21 完成 bar](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/data_acquisition_round13/KRBN_new_completed_bar_20261021_round17.csv>)：Yahoo query1/chart，抓取于2026-10-21T21:08:15.119010Z，即 New York 17:08:15。
- [KCCA 2026-10-21 完成 bar](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/data_acquisition_round13/KCCA_new_completed_bar_20261021_round17.csv>)：同一公开来源，抓取于2026-10-21T21:08:16.879224Z。

完成规则为 exchange-local date=2026-10-21，抓取晚于当日 America/New_York 16:00常规收盘；API exclusive period2=2026-10-22T00:00Z。没有纳入10月22日日线。响应包含10月20/21两日，但新bar文件只保留21日那一行；raw响应原样另存，未在旧CSV后追加。

两 bar 均字段完整、正价、包络有效、range非零、volume正。完整浮点OHLC、adj_close与volume来自供应商原始响应；没有自行复权、平滑、拼接或填补。

[本次输入manifest](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/data_acquisition_round13/KRBN_KCCA_latest_complete_inputs_20261021_round17_manifest.json>)保留URL、响应头、UTC as-of、原始响应与新bar的SHA，以及原两份2026-10-20冻结CSV前后相同的SHA。

## 其他产品的准入状态

GRN 有可读原始行情，法律上是 Barclays Series B Carbon ETN；当前可见窗口从2019-09-18开始，晚于其实际首售/预期挂牌日期。完整成立期覆盖未通过，零range和早期零量必须保留并明确处理；2026 raw已单独隔离为质量数据。本轮不据此运行模型。

CARB.TO 官方已于2025年终止；XCO2.AX 有2025年终止公告镜像线索但原件核验未完成，两者本次Yahoo均404。没有补造当前行情或创建其CSV。

这些产品与KRBN/KCCA的底层碳期货风险可能重叠，不能当作独立宏观市场；产品身份、币种与交易机制不同也不等于经济风险独立。详细身份与来源见[三产品说明](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/data_acquisition_round13/ADDITIONAL_CARBON_PRODUCTS_ROUND17_ZH.md>)。

## 后续真实前瞻记录所需边界

未来若另行获准生成预测，应在预测起点保存真实生成时间、数据输入as-of及SHA、模型与配置版本、预测对象及期限。未来观察发生后才另行记录观察值；不能把本次已发生的10月21日行情回称当日的事前预测。本轮没有执行这些预测步骤，也没有预承诺每日自动运行。
