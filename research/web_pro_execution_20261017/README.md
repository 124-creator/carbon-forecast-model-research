# 2026-10-17 网页入口独立研究执行

**本轮已完成实际运行和独立核验；模型创新尚未成立。**

42次正式新拟合、6次真实短计时和4,239次成熟标签在线更新均完成。只使用批准的广州950条前缀，全部结果属于已暴露开发区。主方案NLinear/DLinear与备选PROCEED的固定版本都未晋级，没有扩大搜索或改写既有结果。

## 先读这些文件

| 内容 | 入口 |
|---|---|
| A—E完整交付：阅读、候选、方法、结果与论文判断 | [REPORT_ZH.md](REPORT_ZH.md) |
| 方法公式、信息路径、直接近邻与未成立的创新 | [NOVELTY_AND_METHOD_ZH.md](NOVELTY_AND_METHOD_ZH.md) |
| 21项来源、精确哈希、访问限制与执行能力 | [ACCESS_AND_EVIDENCE_ZH.md](ACCESS_AND_EVIDENCE_ZH.md) |
| 只取得批准文件并在新目录复现 | [REPRODUCE_ZH.md](REPRODUCE_ZH.md) |
| Stage A冻结协议与完整结果 | [协议](STAGE_A_PROTOCOL.md)、[结果](STAGE_A_RESULTS.json)、[独立审核](audit/AUDIT_STAGE_A_ZH.md) |
| Stage B冻结协议与完整结果 | [协议](STAGE_B_PROTOCOL.md)、[结果](STAGE_B_RESULTS.json)、[独立审核](audit/AUDIT_STAGE_B_ZH.md) |

主风险为451起点上、2020/2021×h1/5/20六格等权MSE/RW_MSE，越低越好，RW=1。

| 阶段 | 方法 | 三种子平均主风险 |
|---|---|---:|
| A：年度冻结 | NLinear | 1.196356 |
| A：年度冻结 | DLinear | 1.786029 |
| A：年度冻结 | c003_GRU | 0.915578 |
| A：年度冻结 | c025_LSTM | 0.944931 |
| B：逐观察在线 | 普通NLinear（Adam） | 1.113662 |
| B：逐观察在线 | 完整PROCEED | 1.125524 |
| B：逐观察在线 | PROCEED_no_drift | 1.100613 |

两阶段的刷新、训练排列和验证适配有差异，不按跨阶段差值归因收益。统计参考为0.951283。完整PROCEED未胜普通在线、no_drift、RW或统计参考。

所有阶段原始协议、源码哈希、全部种子/六格分数、训练轨迹和独立审计保留。作者模型和许可证在vendor及literature中；本次逐点预测与权重不在公开包中。复现脚本只准备新隔离目录，不覆盖历史结果，训练器在PRE_RUN已存在时拒绝重启。
