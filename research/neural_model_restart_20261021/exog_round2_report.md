# 第二轮外生候选输入准备

只读取三份归档 CSV 的日期、当期价格审计列和指定九个 daily lag level 列；未执行或编辑旧脚本。目标日期和价格完全取 canonical 面板。来源可用性未独立验证，本产物仅用于候选实验。

## 三源与 canonical 价格核对

|市场|原始行数|原始日期范围|重复日期行|canonical行数|交集行数|价格不一致行|最大绝对差|
|---|---:|---|---:|---:|---:|---:|---:|
|HUBEI|2792|2014-04-02 至 2025-12-31|0|1466|1455|37|3.3599999999999994|
|GUANGZHOU|2622|2013-12-19 至 2026-01-13|0|2839|2615|42|8.5|
|EU|3426|2013-01-02 至 2026-04-20|0|1577|1570|1570|4.59|

## 九因子原值覆盖、正负与起止

|市场|因子|有值/总行|缺失|零/负数|最小值|最大值|首个有值日|最后有值日|
|---|---|---:|---:|---:|---:|---:|---|---|
|HUBEI|brent_oil_lag1|2792/2792|0|0/0|19.33|127.98|2014-04-02|2025-12-31|
|HUBEI|wti_oil_lag1|2792/2792|0|0/1|-37.630001068115234|123.6999969482422|2014-04-02|2025-12-31|
|HUBEI|natural_gas_lag1|2792/2792|0|0/0|1.21|13.2|2014-04-02|2025-12-31|
|HUBEI|csi300_lag1|2792/2792|0|0/0|2115.14|5807.72|2014-04-02|2025-12-31|
|HUBEI|dax_lag1|2792/2792|0|0/0|8571.9501953125|24611.25|2014-04-02|2025-12-31|
|HUBEI|sp500_lag1|2792/2792|0|0/0|1815.68994140625|6932.0498046875|2014-04-02|2025-12-31|
|HUBEI|vix_VIXCLS_lag1|2792/2792|0|0/0|9.14|65.54|2014-04-02|2025-12-31|
|HUBEI|usdcny_lag1|2792/2792|0|0/0|610.79|725.55|2014-04-02|2025-12-31|
|HUBEI|eurcny_lag1|2792/2792|0|0/0|648.52|857.56|2014-04-02|2025-12-31|
|GUANGZHOU|brent_oil_lag1|2578/2622|44|0/0|19.33|127.98|2013-12-19|2026-01-13|
|GUANGZHOU|wti_oil_lag1|2482/2622|140|0/1|-37.6300010681|123.6999969482|2013-12-19|2025-11-28|
|GUANGZHOU|natural_gas_lag1|2483/2622|139|0/0|1.4950000048|9.6800003052|2013-12-19|2025-11-28|
|GUANGZHOU|csi300_lag1|2597/2622|25|0/0|2086.97|5807.72|2013-12-19|2026-01-13|
|GUANGZHOU|dax_lag1|0/2622|2622|0/0|None|None|None|None|
|GUANGZHOU|sp500_lag1|2482/2622|140|0/0|1809.5999755859|6890.8901367188|2013-12-19|2025-11-28|
|GUANGZHOU|vix_VIXCLS_lag1|2482/2622|140|0/0|9.1400003433|82.6900024414|2013-12-19|2025-11-28|
|GUANGZHOU|usdcny_lag1|2336/2622|286|0/0|6.0713|7.35|2013-12-19|2026-01-13|
|GUANGZHOU|eurcny_lag1|2264/2622|358|0/0|6.5584001541|8.6726999283|2013-12-19|2025-11-28|
|EU|brent_oil_lag1|0/3426|3426|0/0|None|None|None|None|
|EU|wti_oil_lag1|0/3426|3426|0/0|None|None|None|None|
|EU|natural_gas_lag1|0/3426|3426|0/0|None|None|None|None|
|EU|csi300_lag1|0/3426|3426|0/0|None|None|None|None|
|EU|dax_lag1|0/3426|3426|0/0|None|None|None|None|
|EU|sp500_lag1|0/3426|3426|0/0|None|None|None|None|
|EU|vix_VIXCLS_lag1|0/3426|3426|0/0|None|None|None|None|
|EU|usdcny_lag1|0/3426|3426|0/0|None|None|None|None|
|EU|eurcny_lag1|0/3426|3426|0/0|None|None|None|None|

## 对齐后的真正覆盖

|市场|canonical行数|至少一因子|九因子齐全|最大来源age/天|age>7天行|age>30天行|
|---|---:|---:|---:|---:|---:|---:|
|HUBEI|1466|1466|1466|60|20|1|
|GUANGZHOU|2839|2821|0|260|197|146|
|EU|1577|0|0|5|0|0|

|市场|因子|对齐有值/总行|原点首日|原点末日|最大age/天|
|---|---|---:|---|---|---:|
|HUBEI|brent_oil_lag1|1466/1466|2018-01-02|2024-03-29|60|
|HUBEI|wti_oil_lag1|1466/1466|2018-01-02|2024-03-29|60|
|HUBEI|natural_gas_lag1|1466/1466|2018-01-02|2024-03-29|60|
|HUBEI|csi300_lag1|1466/1466|2018-01-02|2024-03-29|60|
|HUBEI|dax_lag1|1466/1466|2018-01-02|2024-03-29|60|
|HUBEI|sp500_lag1|1466/1466|2018-01-02|2024-03-29|60|
|HUBEI|vix_VIXCLS_lag1|1466/1466|2018-01-02|2024-03-29|60|
|HUBEI|usdcny_lag1|1466/1466|2018-01-02|2024-03-29|60|
|HUBEI|eurcny_lag1|1466/1466|2018-01-02|2024-03-29|60|
|GUANGZHOU|brent_oil_lag1|2793/2839|2013-12-20|2026-09-30|260|
|GUANGZHOU|wti_oil_lag1|2531/2839|2013-12-20|2025-12-01|81|
|GUANGZHOU|natural_gas_lag1|2532/2839|2013-12-20|2025-12-01|81|
|GUANGZHOU|csi300_lag1|2812/2839|2013-12-20|2026-09-30|260|
|GUANGZHOU|dax_lag1|0/2839|None|None|None|
|GUANGZHOU|sp500_lag1|2531/2839|2013-12-20|2025-12-01|81|
|GUANGZHOU|vix_VIXCLS_lag1|2531/2839|2013-12-20|2025-12-01|81|
|GUANGZHOU|usdcny_lag1|2540/2839|2013-12-20|2026-09-30|260|
|GUANGZHOU|eurcny_lag1|2298/2839|2013-12-20|2025-12-01|81|
|EU|brent_oil_lag1|0/1577|None|None|None|
|EU|wti_oil_lag1|0/1577|None|None|None|
|EU|natural_gas_lag1|0/1577|None|None|None|
|EU|csi300_lag1|0/1577|None|None|None|
|EU|dax_lag1|0/1577|None|None|None|
|EU|sp500_lag1|0/1577|None|None|None|
|EU|vix_VIXCLS_lag1|0/1577|None|None|None|
|EU|usdcny_lag1|0/1577|None|None|None|
|EU|eurcny_lag1|0/1577|None|None|None|

## 同日跨源一致性

直接比较同名原始列，容差 rtol=1e-6、atol=1e-6。ratio 为 A/B 中位数，不作为自动转换依据；完整计数见 manifest。

|来源A/B|因子|同日双方有值|同量纲数值相等比例|A/B中位数|
|---|---|---:|---:|---:|
|HUBEI__GUANGZHOU|brent_oil_lag1|2497|0.003604325190228274|0.9980620155038759|
|HUBEI__GUANGZHOU|wti_oil_lag1|2408|0.03696013289036545|0.9996946608940727|
|HUBEI__GUANGZHOU|natural_gas_lag1|2409|0.0037359900373599006|0.9908643863313235|
|HUBEI__GUANGZHOU|csi300_lag1|2516|0.0|0.9997243186056279|
|HUBEI__GUANGZHOU|dax_lag1|0|None|None|
|HUBEI__GUANGZHOU|sp500_lag1|2408|0.0340531561461794|0.999631731964155|
|HUBEI__GUANGZHOU|vix_VIXCLS_lag1|2408|0.03612956810631229|1.0036215964779192|
|HUBEI__GUANGZHOU|usdcny_lag1|2248|0.0|99.87645934988362|
|HUBEI__GUANGZHOU|eurcny_lag1|2180|0.0|100.06624214205196|
|HUBEI__EU|brent_oil_lag1|0|None|None|
|HUBEI__EU|wti_oil_lag1|0|None|None|
|HUBEI__EU|natural_gas_lag1|0|None|None|
|HUBEI__EU|csi300_lag1|0|None|None|
|HUBEI__EU|dax_lag1|0|None|None|
|HUBEI__EU|sp500_lag1|0|None|None|
|HUBEI__EU|vix_VIXCLS_lag1|0|None|None|
|HUBEI__EU|usdcny_lag1|0|None|None|
|HUBEI__EU|eurcny_lag1|0|None|None|
|GUANGZHOU__EU|brent_oil_lag1|0|None|None|
|GUANGZHOU__EU|wti_oil_lag1|0|None|None|
|GUANGZHOU__EU|natural_gas_lag1|0|None|None|
|GUANGZHOU__EU|csi300_lag1|0|None|None|
|GUANGZHOU__EU|dax_lag1|0|None|None|
|GUANGZHOU__EU|sp500_lag1|0|None|None|
|GUANGZHOU__EU|vix_VIXCLS_lag1|0|None|None|
|GUANGZHOU__EU|usdcny_lag1|0|None|None|
|GUANGZHOU__EU|eurcny_lag1|0|None|None|

## 使用约束与验证

- 仅交付 HUBEI、GUANGZHOU、EU；CEA/SH 无对应源，跨源公共因子未通过整体验证，不拼接其他市场历史。
- 数组逐行以 market/date/canonical_row_index 对回 canonical，levels 为 N×9 原值，observed_mask 为相同形状；缺失保留 NaN。source_date/source_age_days 为最近严格更早原始行，feature_source_date/feature_source_age_days 仅在该因子有值时有效；无值用 NaT/-1。
- 日期匹配使用 backward asof 且禁止同日匹配；不从未来填值，不后向填充，不把 NA 变为 0。最近行的因子缺失不会搜索更老的非缺失记录。
- lag1 仅是旧列名，不保证发布日。额外要求来源日严格早于原点，但不能排除旧表上游填补、修订、时间区/收盘时间不明。mask 代表归档表中非缺失，不代表上游真实发布或真实观测。
- 保留每市场原始量纲；HUBEI 与 GZ 汇率显著存在约100倍尺度差，股票/能源也可能不同合约或对齐来源，不可直接混用。油/气单位和合约未独立查证。下游仅在训练集拟合缩放；非正值不可直接取log。
- 已验证来源日严格早于原点、market/date无重复、canonical日期价格逐项相等、NaN与mask完全对应、NPZ无pickle精确回读、三源/canonical/候选清单输入SHA256前后未变。
- 完整输入SHA256、原值范围/正负、因子覆盖和逐市场价格审计见 exog_round2_manifest.json。未复用旧评估论文的目标定位或结果主张。
