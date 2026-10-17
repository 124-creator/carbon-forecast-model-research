# PatchTST 作者核心对照：D2 接口与来源核验

日期：2026-10-17。状态：**作者监督预测 core 已固定、原字节保存并通过合成核验；本代理真实市场训练为 0。**

## 1. 可运行交付

```python
from patchtst_adapter import LocalPatchTST

model = LocalPatchTST(d_model=16, context_length=21).double()

# 同一市场 origin=t 时，输入 p[t-20:t+1]，共21个价格水平。
future_levels = model(levels)  # levels: [B,21] 或 [B,21,1] -> [B,20]

# 如果 levels 为原币种价格，training_scale=s_m；
# 如果先除过该市场训练尺度，training_scale=1。
normalized_change = model.normalized_delta(levels, training_scale=s_m)
# = (future_levels - current_price) / (s_m * sqrt(h)), h=1,...,20
```

薄 adapter 只设置作者已有参数、输入/输出形状和项目目标的外部换算。它直接执行完整的作者 `models/PatchTST.py` 与三个依赖层文件；没有重写 attention、归一化、patching、位置编码或预测头。

默认 context=21 是 Root 在冻结前选定的配置，用于与共享 GRU 的 20 个价格增量所依赖的 21 个价格节点对齐。最初的 context=20 也完成过合成形状检查，随后按这一决定改为 21；两者均为作者已有 `seq_len/context_window` 参数，不是源码长度补丁。

必须将 `vendor/` 相对目录和 adapter 一同保存/复制。作者绝对 `layers.*` 导入通过临时命名空间别名载入，随后恢复调用进程已有的通用 `layers` 绑定；这是导入隔离，不修改原始源码。合成检查验证了外部已有 `layers` 模块不被覆盖。

## 2. 固定配置及实际 patch 数

| 项目 | 当前配置 |
|---|---:|
| 单个实例的价格通道 | 1 |
| 价格水平历史长度 | 21 |
| 输出未来价格水平 | 20 |
| patch 长度 / stride | 5 / 5 |
| end padding | 复制末值 5 次，作者原实现 |
| 实际 patch 数 | 5 |
| Transformer 层数 / heads | 1 / 2 |
| d_model | 16 或 32 |
| d_ff | 64 |
| dropout / head_dropout / fc_dropout | 0 / 0 / 0 |
| RevIN / affine / subtract_last | true / false / false |
| BatchNorm / residual attention | 作者默认保留 |
| 位置编码 /预测头 | 作者可学习位置编码 / flatten + linear |
| decomposition / individual | false / false |

作者末尾复制 padding 后，patch 数为

\[
N=\left\lfloor\frac{L-P}{S}\right\rfloor+2.
\]

L=21、P=S=5 时 N=5。最后一个 patch 是当前价格加四个复制价格，第五个追加 padding 点未被 unfold 使用；整个行为来自原作者代码。L=20 时也有五个 patch，最后一组全是末值复制。

无 affine 的 RevIN 与 `affine=0` 的作者运行脚本默认一致；其输入窗口均值、总体方差及反归一化仍完整保留。当前运行时 torch=2.3.1+cpu，而作者 requirements 写 `torch==1.11.0`。本交付只验证所需模型内核在现有批准运行时的兼容性，没有宣称还原作者原环境。四个 core 文件只需要 torch、numpy；完整作者项目的 pandas、matplotlib、scikit-learn 不是此薄接口的额外运行依赖。

参数量如下。总量中的一个是作者 attention scale 的 `requires_grad=False` 参数，不应误记为参与优化。

| d_model | 总参数 | 可训练参数 |
|---:|---:|---:|
| 16 | 5077 | 5076 |
| 32 | 12117 | 12116 |

各市场应独立训练实例，市场身份自然固定。不能把广州、湖北、EU 的不同日期记录直接排列为同一个同时刻多通道输入。所有训练/验证/评估使用原始观测支持，外部尺度只能来自相应训练切分。

BatchNorm 的训练状态、running mean/variance 必须随 state_dict 保存；验证和最终预测使用 `eval()`。作者训练模式中的 BatchNorm 跨 batch 计算统计，不能在评估时让其他未来起点改变某个起点的预测。

## 3. 这项对照能够支持的结论

论文为 Yuqi Nie、Nam H. Nguyen、Phanwadee Sinthong、Jayant Kalagnanam 的 *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*，作者 arXiv 元数据明确标注 ICLR 2023。其主要结构包括子序列 patch、逐通道共享 Transformer、输入实例归一化和 flatten 预测头。作者全文 §3.1 已读取。

- 作者论文：https://arxiv.org/abs/2211.14730
- 作者全文：https://arxiv.org/html/2211.14730v2
- DOI：https://doi.org/10.48550/arXiv.2211.14730
- 作者仓库：https://github.com/yuqinie98/PatchTST
- 固定提交：https://github.com/yuqinie98/PatchTST/commit/204c21efe0b39603ad6e2ca640ef5896646ab1a9
- 许可证：https://github.com/yuqinie98/PatchTST/blob/204c21efe0b39603ad6e2ca640ef5896646ab1a9/LICENSE

实验标签应写成“**PatchTST 作者预测内核，按本项目统一协议训练的单市场配置**”。这是完整的现代预测结构对照，而不是只挑选一个 attention 部件。当前短历史、小宽度和单市场训练方案与作者论文 PatchTST/42、PatchTST/64 的原 benchmark 不同；不应声称复现了其论文精度，也不能将这个特定配置的失败外推为原论文全部方案无效。

## 4. 原字节核验

`SOURCE_MANIFEST.json` 保存固定提交、每文件 Git blob SHA、实际文件 SHA256、字节数和来源 URL。下列原文件均通过 Git blob 算法逐字节匹配，未做源代码修改：

- `PatchTST_supervised/models/PatchTST.py`
- `PatchTST_supervised/layers/PatchTST_backbone.py`
- `PatchTST_supervised/layers/PatchTST_layers.py`
- `PatchTST_supervised/layers/RevIN.py`
- `PatchTST_supervised/requirements.txt`
- 仓库 `LICENSE` 和 `README.md`

RevIN 文件保留作者原有的上游 attribution。仓库 Apache-2.0 全文已保存，需随源代码分发。

读取 OpenReview 的 `https://openreview.net/forum?id=Jbdc0vTOcol` 时返回验证页，重定向为 `https://openreview.net/challenge?redirect=/forum?id%3DJbdc0vTOcol`。本代理没有继续尝试该站挑战或其他 OpenReview 获取方式。作者 arXiv 页面本身可正常读取；具体访问结果已记入 manifest。

## 5. 合成验证范围

`synthetic_check.py` 和 `SYNTHETIC_RECEIPT.json` 覆盖两个宽度的最终 context=21 配置：

- 21 输入、20 输出和五个 patch 实际形状正确，无长度修补。
- 薄 wrapper 与直接原作者 core 的输出、所有可训练参数梯度逐元素一致，最大误差为 0。
- 原尺度价格变化至项目归一化目标的外部换算一致。
- 常数价格窗口和训练梯度均为有限值。
- eval 模式下，更改同 batch 的其他样本不影响指定样本输出。
- 作者源字节匹配，通用导入命名空间被恢复。

真实市场读取 0、真实市场 fit 0、优化器步骤 0。不会将这些合成检查记为市场重复训练或第三方 benchmark 复现。

最终 adapter SHA256：`1fcfa34eb8431d60cee9e0e7c99e22649ad534ed3a63bab1b4905ade23545840`。Root 在 D2 独立目录内执行正式市场训练，本交付不改 D1 冻结源或前轮冻结文件。
