# 中国试点行情／政策事件的可用性与建模前数据合同（2026-10-23）

状态：**数据资格冻结草案；不等于预测实验事前注册。** 仅审核数据能否成为输入，保留已揭示的 2025、2026，不能据其新出结果宣称确认成功。

## A. 三个强制门槛

1. **来源**：每条原始记录保留 `market_id, institution_as_reported, product_code, allowance_vintage, trade_channel, session_date, source_url, retrieved_utc, raw_sha256, source_row_index`；不以交易所更名自动等价产品。
2. **价格与金额**：`open, high, low, reported_mean, reported_close, quantity, turnover` 是不同字段。只有量额原单位和同一成交集合得到官方认证，才衍生 `transaction_vwap = turnover / quantity`；`reported_close` 永不默认等于它。
3. **首次可得**：任何外生政策在对应预测原点前须有可复查的确切版本、首次公开时间和适用市场/配额年份；页面当前写的发布日期、签署日、生效日、履约截止日不能相互代替。

## B. 观察状态与异常（互斥地存储，不删除）

- `OBSERVED_POSITIVE`：来源记录真实可解析的正价格/成交量，仍不代表渠道资格已认证。
- `REPORTED_ZERO`：来源字面为零；不能自动解释为该市场休市，且不计算零量的比值。
- `SOURCE_DASH`：`-` / `--` / 空值，含义为未知或未上报；不得将其填零或插值当成真实交易。
- `EXPLICIT_NO_TRADE`：仅当一手文本明示无成交；保存原句来源，不能由缺行推断。
- `HTTP_ERROR`：保留 403/412/503/521、超时及未抓取，写明日期与请求 URL。
- `SCHEMA_CONFLICT`：例如湖北公共交易平台 2026-09-23—10-13 九行的 `交易价格=600000`；不能凭量额比值默默改字段。
- `VINTAGE_OR_CHANNEL_AMBIGUITY`：品种年份、挂牌/协议、大宗/竞价未明，不拼成单一时间序列。
- `DUPLICATE_KEY`：同日同市场同品种多行完整保留，明确多行来源与拆并理由；不任取第一条。
- `DERIVED_NOT_ORIGINAL`：如 CCER 2025-03-20 用后日累计值推导，禁止称为该日已发布事实。

## C. 政策事件表（可使用前必须逐事件证实）

每个事件保存：`policy_id, jurisdiction, affected_product, allocation_vintage, title, official_url, signed_at, announced_at_displayed, first_public_at_verified, version_first_seen_utc, downloaded_sha256, effective_from, compliance_deadline, quantitative_value, value_denominator, unit, source_quote_locator, revision_status, PIT_eligible`。

- 只按经过**独立且适用于精确文字版本**的证据填写 `first_public_at_verified`；缺失则 `PIT_eligible=false`，不能凭推测滞后 1 天宣称已知。
- 如果官方披露日期/时间及其版本得到证实，而精确盘中时刻缺失，则只允许自之后的**下一当地交易日收盘预测原点**起使用；不能用于公告当天开盘前预测。
- 对修订/撤销政策创建新 `policy_id + version` 行，不覆盖先前事件；在历史原点只选该时刻已知的版本。
- 数值必须带基数和适用对象，例如北京与深圳政策都出现 70% 但分母不同，不能归为同一无量纲强度。
- 2026-10-23 已审核的 15 个候选公告，13 个可下载、2 个 HTTP412；**15/15 尚无历史精确首发版本凭证，模型 PIT 合格 0**。该现状不得通过想象修改。

## D. 各市场当前口径的阻断记录

| 产品 | 已核事实 | 禁止推断 |
|---|---|---|
| 湖北 HBEA | 政府公共交易页面 9 行显示“协商议价”，其中 `成交总价/成交总量` 对 `日收益价` 的显示相容；最新 2026-10-13 | `交易价格=600000` 不是已认证的元/吨价格；政府页面不等于交易所逐笔全渠道原件 |
| 广东 GDEA | 旧官网规则区分挂牌与协议转让，且 2017、2018 收盘生成规则存在差异 | 不能将任意 `Latest`、`Close`、日成交量额比与全部通道均价互换；缺 10/14—23 官方日表 |
| 北京 BEA | 官网搜索引擎可见 10/14、10/15 候选，但本轮原网页 HTTP521 | 搜索摘要不等于网页原档，不能宣布 10/14—23 已完成；北京 CCER 信息不可代替 BEA |
| 全国 CCER | 官方 2025 目录 203 条、成功保存原件 202，另 1 个 404 | 算术推导的 2025-03-20 不得作为当时已知输入 |
| 全国 CEA | 官方月报可核 2026 年 1—9 月合计，但不是日价格 | 不从月统计逆推日级价格或交易日 |

## E. 成为模型输入／评分数据前的验收

1. 同一来源版本所有页面与 CSV 行数一一对应、原 SHA 匹配；失败和越界按资产、日、原因逐项计数。
2. 一手交易所或政府表头须明确量额单位、价格通道、配额期次，来源间数字对应不能替代语义认证。
3. 首次可得政策按上文白名单审查，未通过只允许描述性附件，不可进入任何历史预测。
4. 任何研究者启动**新的模型拟合**之前另行提交 ≤60 行、含预算的冻结方案，明确目标/资产/期限/基准/评分起止/淘汰/统计方法，等待作者批准；既有 A2/EWMA97 事前裁决不得重写。
5. 评分若未来获批，每一项比较均须等资产 ΔQLIKE 与逐资产改善数、损失差置信区间（有推断时）、原点和异常计数；负数损失水平不作百分比变化。
6. **2025、2026 已查看，禁止将其改名为新的确认检验。** 真正的未触及确认样本必须事先定义并在开始前完成版本封存；当下训练、预测原点和 QLIKE 均为 0。

相关机读证据：`audit/PRIMARY_PILOTS_POLICY_GATE_20261023.json`、`policies/policy_event_ledger.json`、`audit/HISTORICAL_RELAY_CANONICAL_FILE_ERRATUM_20261023.md`。
