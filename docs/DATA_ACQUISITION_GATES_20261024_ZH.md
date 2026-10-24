# 中国试点/CEA/CCER 数据采集状态与准入门（2026-10-24）

> **结论：数据获取尚未完成，也尚不具备新的确认性预测实验资格。** 本文是对已经存在的资料和 [Draft PR #32](https://github.com/124-creator/carbon-forecast-model-research/pull/32) 的定位；没有下载新原始报价、没有新增拟合/预测得分。事前已见的 2025、2026 年份均不能重新充当确认期。

## 1 数据分层（绝不当成一张可合并的「碳市场面板」）

| 层 | 对象和来源 | 已核验的边界 | 模型准入 |
|---|---|---|---|
| 美股 ETF 风险 | KRBN、KEUA、KCCA（[冻结入口](../START_HERE_PRO_20261022_ZH.md)） | ETF 价格的未来 H 个观察收益平方均值，非碳配额期货/现货。2025 判定、2026 仅已见诊断 | 旧五路径已依协议结束；不能以下载中国数据改写旧结论 |
| 地方市场转载历史 | 四川联合环境交易所（SCEEX）历史转发表；北京/上海/天津/广东/湖北/深圳/重庆/福建产品分别审核 | 2016—17 2,823 原行、2018—25 21,430 原行、2026 至 10-13 1,459 原行，合计 **25,712 原行**，不等于 25,712 个合格交易日 | 仅来源/字段审核；非一手认证的可比日行情 |
| 湖北 HBEA / 广东 GDEA | 固定 2018—2024 历史复核 | HBEA 1,703 原行、其中 1,637 行满足预设机械筛查；GDEA 1,700/1,649；16/24 个字段一致性诊断标记，湖北 4 个代码错配及 4 个重复日期组、广东 1 个重复日期组 | **缺历史官方分渠道规则及字段语义认证**，异常留原样不删 |
| 湖北官方公共服务页面 | 已取得原页 1—5 页，45 行 2026-08-04—10-13 | 45 行均显示 `交易价格=600000`；不得猜单位/缩放；网页 177 页并未全部取得 | 不可当元/吨价格；不补 10-14 之后缺口 |
| 全国 CEA | 全国履约配额，不与地方试点代码拼接 | **2026-01-05 综合价格口径切换为 CEA25 权重 1、旧年权重 0**；需确认期次与合约语义 | 跨定义断点不得无条件计算收益序列 |
| 全国 CCER | 自愿减排交易，与 CEA、地方配额各自成表 | 2025 官网 203 索引、202 原文 HTTP200、1 原文 HTTP404（3月20）；缺页值仅后日累计推导。2026 192 原文 HTTP200（其中 6 份显式无交易） | 原档哈希/摘要可复核，逐日原文和再分发权利不齐 |
| 政策/制度 | 官方交易规则、清缴、方法学与报告 | **19 候选 URL，17 HTTP200，2 湖北 HTTP412，历史第一公开版本 0/19 认证** | 历史 as-of 政策特征 **0 个合格**；签发日≠发布日≠施行日 |

以上地方和政策计数来自 [PR #32 的数据停机规则](https://github.com/124-creator/carbon-forecast-model-research/blob/data/china-pilots-policy-20261023/research/china_market_expansion_20261023/DATA_READINESS_STOP_RULE_20261023_ZH.md) 与 [湖北广东源审计](https://github.com/124-creator/carbon-forecast-model-research/blob/data/china-pilots-policy-20261023/research/china_market_expansion_20261023/PRICE_SEMANTICS_HUBEI_GUANGDONG_20261023_ZH.md)。本页只是文档整理，不重新计算这些数字。

## 2 公开与私人来源边界

- **可以公开**：自写采集器、验证程序、来源链接、摘要字段、哈希、失败/异常计数、复现说明（是否授权第三方代码另核）。
- **目前不随公共 GitHub 批量发布**：未获可再分发许可的交易所逐日原始行情、CCER 原 HTML、地方政府源 HTML、第三方 PDF/Excel、私人逐原点预测及内部私有归档。
- **许可证未决**：仓库公开可读，但根目录无 `LICENSE`；本次不替作者选择 MIT/Apache 等协议，也不声称原始数据因此开源。
- **来源失败必须保留**：HTTP 403/412/521/503、CCER 404、年报私有归档 `transfer_failed` 等，不能变成「零交易」「已下载」或无声剔除。

## 3 何时停止采集、何时能够讨论新模型

1. **可以停止无收益的镜像重复抓取**：25,712 原行仍卡在产品/报价单位/成交通道定义与授权，不是再多抓几页转载就能解决。网站 403/412/CloudWAF 时停止，不试图绕过。
2. **优先 P0 证据**：湖北和广东 2018—2024 交易所授权历史导出（分挂牌/协议、品种、期次、结算/收盘/均价字段、量额、异常标记），各版本规则及正式发布时间，允许复现的许可证/授权文本。详见 [对外索取清单](https://github.com/124-creator/carbon-forecast-model-research/blob/data/china-pilots-policy-20261023/research/china_market_expansion_20261023/PRIMARY_EXCHANGE_EXPORT_REQUEST_20261023_ZH.md)。
3. **如选择只预测价格、不用政策**：可暂不等全部政策原文，但仍须满足价格/交易通道/数据权利/固定日期覆盖条件；不用政策必须事先写定，不能据得分改变。
4. **如选择政策增强模型**：需独立证明每条政策的版本、历史首次可得时间、发布时间及对应市场，当前 0/19 不达标。
5. **实验开始之前**：先交不超过 60 行的冻结方案和拟合预算（资产、目标、h=1/5/22 中事前选定期限、EWMA97/K1/GARCH 等适用基准、判定区间、资产等权 ΔQLIKE/逐资产改善、停止条件），等用户再次确认；2025/2026 已被检视，不得标记新确认。
6. 仍不能满足一手来源/权限，就明确记为 **BLOCKED—需作者提供合法导出或待未来真正未见时期**，不通过换市场、删异常或调整窗口“追指标”。

## 4 精确入口

- [#32 数据 PR](https://github.com/124-creator/carbon-forecast-model-research/pull/32) —— **Draft、未合并 main**。
- [数据说明卡](https://github.com/124-creator/carbon-forecast-model-research/blob/data/china-pilots-policy-20261023/research/china_market_expansion_20261023/OPEN_SOURCE_DATASET_CARD_20261023_ZH.md) —— 公共代码/审计元数据的复用边界。
- [数据停机规则](https://github.com/124-creator/carbon-forecast-model-research/blob/data/china-pilots-policy-20261023/research/china_market_expansion_20261023/DATA_READINESS_STOP_RULE_20261023_ZH.md) —— 可否称数据「齐备」的逐项条件。
- [历史 HBEA/GDEA 价格定义](https://github.com/124-creator/carbon-forecast-model-research/blob/data/china-pilots-policy-20261023/research/china_market_expansion_20261023/PRICE_SEMANTICS_HUBEI_GUANGDONG_20261023_ZH.md) —— 数字与成交通道问题。
- [一手交易所授权清单](https://github.com/124-creator/carbon-forecast-model-research/blob/data/china-pilots-policy-20261023/research/china_market_expansion_20261023/PRIMARY_EXCHANGE_EXPORT_REQUEST_20261023_ZH.md) —— 外部阻断项。
