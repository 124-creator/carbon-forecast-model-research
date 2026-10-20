# 天津官方长历史快照：标准化、切分与复现核查

**PASS_DATUM_IDENTITY_AND_FIXED_PARTITION_QC_NO_MODEL_FIT；PASS_FIXED_RAW_AND_CANONICAL_BYTES。** 这是数据获取后的标准化补充，保留先前的 [可得性调查](NEXT_MARKET_DATA_AVAILABILITY_ZH.md) 原样；本文件明确后续模型所用的四列 canonical 身份。没有运行市场拟合或预测评分。

## 固定数据身份

| 项目 | 值 |
|---|---|
| 官方入口 | https://www.chinatcx.com.cn/tshq |
| 实际输入 | `/tmp/carbon-next-market-data-20261019d/tianjin_body.js` |
| 原件 SHA-256 | `c3e4c075832f6e5087cf416adecbae9d3c1b61859a6ad09fdf84294197393238` |
| 四列输入 | `/tmp/carbon-next-market-data-20261019d/TIANJIN_OFFICIAL_MEAN.csv` |
| CSV SHA-256 | `85ca07648855db00fbf67c4e1c083db715f62ab829f74c143c3b3f67da21252a` |
| CSV 大小／行数 | 34,757 bytes；1,077 数据行、1,077 唯一日期 |
| schema | `date,price,volume_tons,turnover_yuan` |
| 首末日 | 2013-12-26—2026-09-30 |
| 目标 | TJEA 当日汇总成交均价，元／吨；不解释成收盘价 |
| 量额 | 同一行当日总成交量（吨）、当日总成交额（元） |

第一张主表两个表头已精确校验，字段索引为日期 0、品种 1、当日汇总量 3、当日汇总额 5、均价 7。所有品种均为 TJEA；不是累计字段，也不是挂牌竞价子列。全部总字段有限数值，均价和总量均为正、总额非负，无占位／缺失／重复日期。77 行挂牌子列空白不影响总量字段，全部保留。只去除数值中的千位逗号并按日期升序；未删行、补值、前向填充、改写均价或按模型表现筛选。

## 固定 60/20/20 与标签成熟

`N=1077; A=floor(0.6N)=646; B=A+floor(0.2N)=861`，使用零基索引、左闭右开分区。

| 分区 | 原始记录索引 | 原始记录日期 |
|---|---|---|
| 训练 | `[0,646)` | 2013-12-26—2020-10-22 |
| 验证 | `[646,861)` | 2020-10-23—2023-04-18 |
| 后段测试 | `[861,1077)` | 2023-04-19—2026-09-30 |

同 64 窗口与 H1/5/20 共同成熟条件下，训练原点为 64…625，验证为 646…840，测试为 **861…1056，196 个原点，2023-04-19—2026-07-02**。这些期限按保留的观测交易记录计算；全表最大日期间隔 319 个自然日，不能把 H20 改称恒定 20 个自然日。

## 与旧天津作者基准的机械对应

旧 ScienceDB 天津 457 个日期在官方表中全部存在，但不应无检验地拼接目标：456 行量完全一致，279 行额完全一致，144 行价格数值完全一致；434 行价格在官方两位小数显示的半分钱以内。279 行同时满足量、额完全一致及价差不超过半分钱，178 行记为来源值差异。这里没有推断每一差异的成因，也没有据此删除或调整任何新表记录。新后段日期与旧基准全部 457 个日期的交集为 **0**。[TIANJIN_OFFICIAL_CANONICAL_QC.json](TIANJIN_OFFICIAL_CANONICAL_QC.json) 保存完整数量与来源绑定。

## 可执行复现

在本 study 目录执行，使用 Python 标准库，无须安装数据分析依赖。已有原件的纯离线复现：

```bash
python provenance/prepare_tianjin_official_snapshot.py \
  --raw-file /tmp/carbon-next-market-data-20261019d/tianjin_body.js \
  --out-dir /tmp/tianjin-official-rebuild-fresh
```

如无原件，省略 `--raw-file` 可对已固定的官方公开 Body.js URL 作一次正常 GET；`--out-dir` 必须为新目录，不能置于当前研究仓库。省略输出参数时自动创建临时目录。下载后必须匹配原件 SHA；若官网静态对象内容改变，脚本会明确失败，不以新数据悄悄替代本轮输入。

上述离线流程已经实际执行，重建 34,757 bytes CSV 与模型输入逐字节一致。[TIANJIN_OFFICIAL_REPRODUCTION_CHECK.json](TIANJIN_OFFICIAL_REPRODUCTION_CHECK.json) 状态为 PASS，记录解析脚本 SHA `a00b7a3be5a894f9ef3b4833941cd5d64fd091e2f20e703851a0688190800745`。网络获取发生在初次调查；本次 CLI 检查是离线重建，不伪称额外的网络复现测试。

官网未发现明确再分发许可，因此只有自编解析代码、来源说明和汇总／哈希收据放入仓库；原件、CSV 与按日来源差异表均留临时运行目录。后续模型应另行冻结新阶段协议，并保留此前早期作者数据上的天津结果；本数据核查不构成预测成功结论。
