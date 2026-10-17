# PROCEED 候选与实现独立核查（2026-10-17）

## 结论与状态

保留为本轮第三候选、建议备选。它是已发表的参数适应学习机制，实质不同于 HSC 的期限竞争和 RUM 的残差效用检索。本次只读论文及源码，未对广州训练 PROCEED，也没有宣称其为我们的模型创新。

## 论文与访问层级

- 论文：Lifan Zhao and Yanyan Shen, *Proactive Model Adaptation Against Concept Drift for Online Time Series Forecasting*, KDD 2025, 2020–2031。
- 正式 DOI：https://doi.org/10.1145/3690624.3709210
- 作者全文：https://arxiv.org/html/2412.08435v5 。本次已读方法、实验设置、附录 A、B、C、D、E；v5 声明修正 Eq.(3) 的拼写。
- 正式 ACM 页面 https://dl.acm.org/doi/10.1145/3690624.3709210 本次打开返回 Internal Error；未冒称读到 ACM 版本 PDF。KDD 接收信息由作者 arXiv、作者主页及 DOI 检索互证。
- 作者主页：https://lifan-zhao.github.io/publication/proceed/
- 作者仓库：https://github.com/SJTU-DMTai/OnlineTSF
- 固定版本：0a8646199f6dcbf5346f933508b33c5e35b5708b；作者提交日期 2025-12-25。
- 作者 pyproject.toml 明确 license = "Apache-2.0"；树中未见单独 LICENSE。应如实写成“pyproject 声明 Apache-2.0”，不能写成无许可证或自行声称每个继承文件均已独立权利核验。
- 另有作者 Zenodo preview 归档 DOI 10.5281/zenodo.14554744，本次只读记录页，未下载归档。

以上事实的出处也可固定到：
https://github.com/SJTU-DMTai/OnlineTSF/blob/0a8646199f6dcbf5346f933508b33c5e35b5708b/pyproject.toml

## 作者机制（简述）

当前输入与最近成熟样本分别编码，概念差向量经瓶颈生成器生成逐层参数缩放。在线时只用到时点 t 已成熟的样本（完整 H 输出时 s <= t-H），参数的当前临时适配不永久累积；基础权重继续由成熟反馈更新。训练时仅在历史训练区内打乱样本以合成概念差。附录 D 特别说明：成熟样本的反馈预测也应经过适配器，不能将正文示意简化为仅裸骨干更新。

其直接近邻为 OneNet/FSNet；OneNet 是 NeurIPS 2023，不是 ICML 2024，也不是该仓库 README 写的 NeurIPS 2024。这里只将二者用于边界比较，不增加完整候选。

## 固定源码与依赖

源码目录：/workspace/scratch/b31c5edbd0de/audit_learning_proceed/author_source

SHA-256 与 GitHub blob SHA 见 SOURCE_MANIFEST.json。16 个文件按 GitHub 返回 UTF-8 文本原样保存。核心适配器及 Linear/NLinear/DLinear 所需仓库内 import 闭包已保存：

- adapter/proceed.py
- adapter/module/base.py
- adapter/module/generator.py
- adapter/module/down_up.py
- adapter/module/up.py
- adapter/module/ssf.py
- layers/RevIN.py
- models/Linear.py、models/NLinear.py、models/DLinear.py

附加保存 README.md、pyproject.toml、run.py、exp/exp_proceed.py、data_provider/data_loader.py 与 models/TCN.py，用于核查。没有保存完整实验框架；TCN 的 TS2Vec 层传递依赖未补齐，不能据此直接 import TCN。

核心外部 import：torch、numpy、transformers，以及 Python 标准库。作者 pyproject 声明 Python >= 3.10、torch >= 2.0；核心代码直接引用 transformers.Conv1D，实际运行环境应检查该导出存在，不可以未核验就称兼容任何 transformers 版本。完整作者 runner 另外依赖 pandas、scikit-learn、einops、matplotlib 等。

建议独立入口使用：

    sys.path.insert(0, AUTHOR_SOURCE)
    from adapter.proceed import Proceed
    from models.NLinear import Model as NLinear
    backbone = NLinear(configs)
    model = Proceed(backbone, args)

所需 args 包括 seq_len、pred_len、enc_in、individual、freeze、concept_dim、act、bottleneck_dim、individual_generator、ema、tune_mode、merge_weights、do_predict、wo_clip。设 tune_mode="down_up" 时只适配已支持的线性/卷积层。主干输入为 [B,L,C]，recent_batch 为 [B_previous,L+H,C]，输出 [B,H,C]。recent_batch 中输入和标签必须用一致的观测单位和因果标准化，不能把价格增量输入和另一种统计锚残差标签不加说明地拼成“原序列”。

## 已定位的实质复现陷阱

1. **最新 runner 禁用了训练。** 当前提交 run.py 第439–440行注释了 exp.train(...)；提交 diff：
   https://github.com/SJTU-DMTai/OnlineTSF/commit/0a8646199f6dcbf5346f933508b33c5e35b5708b
   所以“运行最新作者入口”不等于“作者模型已从零重训”。可用明确透明的独立训练入口，或固定父提交 3070e64543d88c6ab552fa1b28e3ff8dbdebd77e 并显式补必要兼容修复。本次没有修改作者源码。
2. **现有项目 SequenceNet 无可自动适配层。** 作者 add_adapters_ 支持 nn.Linear、nn.Conv1d、transformers.Conv1D（另模式支持部分归一化层）；不支持 nn.GRU/LSTM。项目 hd_fit.py 的 readout 又是裸 nn.Parameter。因此将当前 SequenceNet 原样传进去会产生零适配层，不能称完整 GRU-PROCEED。
3. **作者完整实验默认切分不同。** README 默认 20:5:75；广州必须保持年度边界、成熟清除区、训练标准化和 451 起点主评分，不套默认 loader 直接计算分数。代码 Dataset_Recent 的 gap=H 返回历史与当前成对样本，思路正确但适配边界要重建。
4. **“冻结适配器”要按源码说清。** freeze_adapter 冻结两个编码器与生成器 weights，但在 more_bias=True 时生成器末级偏置仍可能可训练；不能笼统宣称所有适配器张量测试期绝不更新。若独立简化入口另行冻结全部，应标记迁移实现差异。
5. **正文与代码参数化有细节差异。** 当前 Linear 适配层实际为输入缩放、输出缩放及独立加性 bias shift；用代码形式写定义最清楚：F_t(x)=(1+a_out)⊙[W((1+a_in)⊙x)]+b+c。不要直接照 HTML 中维度不一致的矩阵转置排版抄实现。

## 广州 950 条可行性判断（本审查的推断）

项目 fair_data.support 在 end=459/703 时分别约 299/400 个训练起点、60 个验证起点，且窗口重叠；不应按 950 个独立样本衡量模型容量。作者论文主实验远大于广州，默认 d_c=200、r=32 不宜未经预算与验证直接复制。

如果仅做备选机制的最小迁移，作者 NLinear（L=60、H=20、C=1）加 d_c=8、r=4 的小适配器可约有 3036 个参数：NLinear 1220，两个概念编码器 1280，生成器 536（根据当前代码参数形状计算，未以训练结果认定最优）。这是缩小维度后的机制迁移，不是原论文表格复现，更不是新模型。

最可能失败的原因：短样本覆盖不了概念漂移与最佳参数变化间的可学习映射；1步在线噪声使基础权重过拟合；共同价格趋势已被 NLinear/统计锚吸收，适配器只增加估计方差。没有任何数据证据保证此候选优于 GRU/LSTM。

## 最小否证边界

如后续启动本备选，先固定超参数和预算，然后成对比较相同骨干的静态版本、同成熟标签/同梯度预算的在线 GD、完整适配器和去概念差条件的等容量适配器。强 GRU/LSTM、统计参考与 RW 的作用必须保留；不同刷新策略不能只按历史点分数相排。

以下任一情况足以停止该固定版本，不允许继续改名称救分：

- 不能通过后缀扰动不影响过去预测、标签成熟和零适配恒等性；
- 只好于静态弱骨干，却不优于同预算在线 GD；
- 把概念差置零/打乱后收益基本不变，说明收益来自容量或普通更新；
- 主要收益由单年度或单一期限承担，另一年度出现明显损害；具体数值门槛在跑前冻结；
- 未优于原始直接近邻却把一般缩小维度、输出正则或低秩结构宣称创新。

## 理论引用限制（独立推理）

附录 A 通过参数更接近最优点来论证预测可改善；其中“连续性使局部 Lipschitz 上下界比趋于1”不是普遍成立的命题。反例：F(θ)=(100θ1,θ2)、θ*=0，θ_old=(0,2)，θ_new=(1,0)。新参数距离更短，但预测误差从2变100。可引用作者实证与机制，不能把该附录用成碳市场无条件收益保证。

## 本次实际项目源码阅读范围

已读 WEB_PRO_START_HERE_ZH.md、EXECUTION_PROMPT_ZH.txt、fair_data.py、hd_core.py、hd_fit.py、hd_analysis.py；对 hd_run.py 做了关键词和相关片段检查。未读取私有权重或旧逐点预测，未读取 2022/2023，未重跑历史192次拟合。

