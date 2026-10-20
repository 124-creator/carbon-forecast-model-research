# 扩大数据与单步预测方法研究（阶段i/j/k）

**阅读顺序：** REPORT_ZH.md → FINAL_STATUS.json → results/* → 每阶段 METHOD_*.md / source/*.py / audits/*。

研究性质：2018—2023等历史数据上的**开发实验**。三个阶段绝不属于同一种资产/任务的单一分母；阶段k的高指标属于同日外部价格**条件信息**，源的真实首次可用时间仍未核验。

## 运行所需私有输入（不公开再分发）

- `carbon_distribution_update_private_run_20261019.zip`：先前用户实验归档，`inputs/development/EUA_LONG_development.csv` 等，完整 zip 含受限或私人资料，勿上传仓库。
- `carbon_research_prior_transport_20261018.zip`：早前的额外欧洲及试点开发数据，原始报价不重新公开。
- `carbon_prior_transport_private_evidence_20261018.zip`：`local_only/inputs/EEX_EUA_T3PA_daily_development.csv`，从EEX 2012-2025官方交易所Excel合法还原的2012-2023开发子集。年度报告官方链接见原报告。不同产品与价格字段必须保持隔离。
- `stage_i`的CEA和CCER公开整理来源，可按`stage_i/source/download_secondary_sources.py`取得并记录哈希，不将CCER称为CEA配额。

## 运行环境

Python 3.13 / numpy 2.3 / pandas 2.2 / scipy 1.17 / scikit-learn 1.8 / PyTorch 2.10 CPU；建议限制OpenBLAS/OMP到1线程。不同软件版本不保证位级训练相同。

```bash
# 实验阶段 i：四市场/作者报价单步+小样本CCER，复现实验前先恢复历史归档和官方作者数据
python stage_i/source/run_single_step.py --data-dir /path/to/archives --new-data-dir /path/to/other_public_sources --out /tmp/i_new
python stage_i/source/replay_and_statistics.py --run /tmp/i_new --out /tmp/i_audit --reps 5000

# 实验阶段 j：官方 EEX T3PA 2012-2023 开发事件，严格只用过去/当前已完成拍卖
python stage_j/source/test_auction_preflight.py --archive /path/to/carbon_prior_transport_private_evidence_20261018.zip
python stage_j/source/run_auction_mechanism.py --archive /path/to/carbon_prior_transport_private_evidence_20261018.zip --out /tmp/j_new
python stage_j/source/audit_auction_mechanism.py --run /tmp/j_new --out /tmp/j_audit

# 实验阶段 k：两种同一报告价日期边界必须都评价
python stage_k/source/test_asof_temporal.py
python stage_k/source/run_dual_price.py --primary-archive /path/to/carbon_prior_transport_private_evidence_20261018.zip --aux-archive /path/to/carbon_distribution_update_private_run_20261019.zip --out /tmp/k_new
python stage_k/source/audit_dual_price.py --run /tmp/k_new --out /tmp/k_audit
```

注意上述所有输出目录应新建，不能覆盖旧物证。阶段k第一次运行的两处纯输出层故障有明确冻结提交及结果前回执；修正后的原始训练源码在stage_k/source/run_dual_price.py，唯一正式结果来自`formal_run_fixed2`。

如果取得官方2024/25完整年度报表，先由作者评估版权/保存边界和新时期角色，再按相同产品字段、固定时间分割确认。2026年只是进行中的年度，不应当伪造完整年度评分。新数据不得与旧报价直接混合或统一伪装为某一市场的单一期货价格。
