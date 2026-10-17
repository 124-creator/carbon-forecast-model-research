# D2 PatchTST 拟合接口与机制对比口径

日期：2026-10-17。本文件是供 Root 接入的说明，不改变 D1 冻结源或扩大 D2 冻结训练数量。

## 1. PatchTST 真实训练入口

```python
import fit_core
from patchtst_fit import (
    fit_patchtst, predict_patchtst, predict_patchtst_normalized,
    serialize_patchtst_state,
)

# 父级从已授权内存数据构造，不在helper内加载市场文件：
# train['levels'][i] = market.price[origin_i-20:origin_i+1]
# val/evaluation的levels同样只截止到自己的origin。
# levels必须为原币种价格，scale必须保持原data_core训练尺度[N]。

model, receipt = fit_patchtst(
    train, val,
    config={'family':'patchtst', 'd_model':16, 'shrink':0.1},
    seed=17, max_epochs=60, deadline=absolute_monotonic_deadline,
    common=fit_core,
)
if not receipt['eligible_for_selection']:
    # 记录未完成；不得混入足预算模型的选择。
    ...

forecast_change = predict_patchtst(model, evaluation)  # 原币种累计变化
forecast_normalized = predict_patchtst_normalized(model, evaluation)
state = serialize_patchtst_state(model, common=fit_core)
```

如果不显式传 `common`，函数在调用时 `import fit_core`，要求调用进程已经能够导入该模块。使用的原公共函数为 `configure`、`group_weights`、`validation_score`、`state_arrays`、`state_hash`；没有复制一套新的评分或状态哈希算法替代它们。

返回的 `model` 是字典，包含 `net`、`config`、`market_names`、完整 `state` 和 `state_sha256`；不是直接的裸 `nn.Module`。receipt 使用与 D1 相同的 `epochs`、`seconds`、`selected_epoch`、`validation_score` 等字段。PatchTST 必须使用专用 predict 函数，不能把价格水平送进 D1 的增量 GRU 输入函数。

## 2. 固定学习规则

- float64、CPU 单线程、调用公共 deterministic 配置。
- 原作者随机初始化保留；没有把 GRU 的 ridge-head 初始化挪到 PatchTST 不同结构的 flatten head。应在论文实现表中披露这一点。
- 原价格 21 节点，作者输出未来 20 个价格水平；外部换算为 `(forecast-current)/(train_scale*sqrt(h))`。
- full-batch `train()`；验证、最终预测均为 `eval()`。
- AdamW 学习率 0.001，所有可训练参数 weight decay 0.001，梯度范数裁剪 1。
- 训练目标是等市场加权的归一化未来变化 MSE，加 `shrink ×` 同样加权的归一化预测平方；shrink 仅允许 0.1 或 1。
- 最多 60 epochs，patience=12，改进阈值 `1e-6`。
- epoch0 的验证分数参与选择；每次变好保存**完整 state_dict**，包含 BatchNorm 的 running mean/variance 和 batch 计数。
- 验证分数直接复用公共 `validation_score`：市场等权、H=1/5/20 等权的相对 RW MSE。

输入的 `levels[N,21]`、`scale[N]` 和 `full[N,20,C]` 在拟合前核对：`diff(levels)/scale` 应等于第一列价格增量。如果调用方误传已归一化的水平、错位 origin 或 `[N,1]` scale，接口会报错。这个检查避免期限归一化中出现静默的广播或重复缩放错误。

## 3. 预算与未完成状态

deadline 是绝对 `time.monotonic()` 截止值。进入已过期调用时直接报 TimeoutError；运行中在新的训练步之前、训练步之后及验证之后检查。

如果一个完整 tensor 操作跨过 deadline，无法在该运算中途安全抢占；下一检查会停止后续工作。未验证的新状态不会被当成最佳状态。函数恢复此前最佳完整状态，receipt 标记 `status='deadline'`、具体 `stop_reason`，并设置 `eligible_for_selection=False`。Root 应保存这条未完成记录并排除正常选择，不能悄悄使用部分预算模型。

数值非有限值直接报错，没有自动改学习率、重启或额外试验。实际发生的优化器步骤与未验证截止行均计入 trace/receipt。

## 4. 合成检查已完成的范围

`synthetic_fit_check.py` 已通过，结果在 `FIT_SYNTHETIC_RECEIPT.json`：真实市场文件读取 0、真实市场拟合 0，只有 3 个合成控制例共 15 步优化器更新。

1. 正常两步训练：独立重算公共 H1/5/20 验证分数，核对逐行训练尺度广播与完整状态哈希。
2. 人工指定后续验证分数始终变差：真实执行 12 个合成训练步骤后早停，选择 epoch0，完整状态 SHA 与初始状态相同，BatchNorm 计数恢复为 0。
3. 人工时钟在一次训练更新后、验证前到期：没有额外梯度更新，恢复 epoch0，标记未完成且不允许选用。

这些是 API/学习规则检查，不是额外真实市场结果，也不是 PatchTST 作者 benchmark 复现。

## 5. KCM 对 MCSSL：必须配对的比较

Root 交接的 D1 结果中，KCM32/shrink1/λ1 为约 0.985380，同配置 augmentation 为约 0.985440；差约 `−0.000060`，相对幅度约 0.006%。最佳 pooled price GRU16/shrink0.1 约为 0.983237。这些数值来自 Root 交接，本代理没有独立重算 D1 成绩。如此小的匹配差异只能提出继续核查的理由，不能本身作为机制有效的强证据。

D2 应以其冻结清单内的**同一 backbone、同一市场输入、同一监督视图比例、同一训练支持、同一初始化/seed、同一 shrink**为比较单元：

- augmentation：完整/粗视图监督，机制系数为 0。
- KCM：在上述全部条件上加入既定固定核、非重叠样本对、正部条件矩。
- MCSSL 论文式适配：同一 GRU 及监督目标，加入两次 q 条件补全的预测差乘积，并按预先声明的逐组/期限正部稳定化。

不能用某个宽度/shrink 下的 KCM 最佳值，减去另一个宽度/shrink 下的 augmentation/MCSSL 分数，再称为正则的净贡献。实际冻结的 D1 与 D2 均使用广州、湖北、欧盟和上海四市场、相同的 5,864 条开发数据输入；D1 的 141 次拟合从一开始就包含上海。较早三市场、4,357 条数据的提案属于冻结前准备，不是 D1 实验结果。跨 D1/D2 或更早试验比较时仍须核对训练协议，并在相同市场和完全相同 origins 上配对。

需要同时保留两种问题：

1. **机制增量**：同配置 KCM−augmentation、MCSSL−augmentation、KCM−MCSSL 的逐 seed、逐 market/year/H 差异。
2. **实际预测效用**：模型是否超过本轮可选的最佳简单模型、RW 和强结构对照。机制正则仅略优于自己的匹配基线，不代表它优于所有实际可选方法。

两类结论应分别报告。选择λ、宽度、shrink和early-stopping均只允许依照已冻结开发验证规则；不能因看到 D2 评估结果再给某臂额外预算。

## 6. 核对正则究竟做了什么

在冻结模型的同一评估起点上，用同一 full/short/sparse 输入保存以下量即可，不需要新的市场训练：

- 原始 full 视图的真实预测误差，以及两个人为粗视图的误差，明确人工视图是干预。
- 预测修正 `d=f(full)-f(coarse)` 的平方均值、固定核的未截断矩、正部激活比例。
- 非重叠有效对数与核权重和；负 U 或低激活率不能被写成已证明一致。
- MCSSL 两份补全输入/预测的差异、缺失历史价格重建误差。MSE 补全器可能忽略噪声，必须记录其是否退化。

正则值降低并不自动意味着预测改善；`d` 一律变小也可能只是有用细信息被抹掉。合成中合法细信息的例子、可抑制的粗条件偏移例子及无法发现跨期反号的例子应保留，防止把条件均值约束误写为一般因果/OOD 保证。

若进行不确定性估计，使用保持模型预测配对的连续时间块，而不是把高度重叠的每日 origins 当作独立样本。可报告预先固定的 41/82 个原始观测步块长敏感性；非重叠仍不保证金融时间序列独立，跨市场相关性和仅四市场的局限必须说明。随机 seed 是优化随机性重复，不是新的独立市场。

本项目的 MCSSL 是独立补全器预训后冻结的适配。若预训 q 后完全不用它也不用 MC 项，预测器就与 augmentation 相同，因此不必另增一个重复市场训练来称作“imputation-only”对照。作者论文中 imputation 辅助任务与共享编码器联合训练的情形与此不同，不能把这两者混称。

## 7. 结论边界

若同配置的稳定增量不能复现，或者只在人工稀疏视图上出现而完整真实历史预测没有收益，结论应是该候选尚未获得机制证据。若仅更大的共享 GRU、更多真实训练样本或更合适的 shrink 带来优势，贡献应归给这些因素。

如有稳定改善，仍应使用已核实的有限贡献表述：**已有鞅一致性与核条件矩思想在缺乏可靠条件补全器的跨市场价格预测中的一种可实现训练设计及其经验证效果**。不得把该实验写成首次提出 tower property、KCM、一般缺失数据学习或一般跨期稳健性。
