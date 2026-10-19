# 已拟合方程独立 NumPy 复放

状态：**PASS_INDEPENDENT_NUMPY_FITTED_EQUATIONS**。

本程序重新计算已完成 checkpoint 的方程、读出、损失和预测；不增加超参数搜索或优化器训练。EWT 的已固定分析特征读取自带 SHA 的私有缓存；原始特征、成熟掩码与 20 步监督标签独立重建。基底、VP 和 Ridge 均使用 NumPy 显式公式，没有调用 Torch 模型前向或 sklearn 估计器。

## 覆盖

| 类型 | 选择 checkpoint | 最终 checkpoint |
|---|---:|---:|
| EWT 响应 | 384 | 24 |
| Dense-path 响应 | 128 | 8 |
| EWT Ridge | 48 | 6 |

另核验全部选择表的固定 tie 顺序，逐 checkpoint 的 epoch 容差选择、梯度更新计数、最终拟合步数及实际数值参数变化。Dense-path 的训练方差为原始训练期 H1–20 二阶矩；选择与主评价使用原始 H1/H5/H20 方差，未以 20 步平均替代 3 步主评价。

## 数值差

| 核验项 | 最大绝对差/残差 |
|---|---:|
| `response_head_coefficients` | 8.66862137627e-12 |
| `response_head_column_scale` | 3.89022147829e-13 |
| `response_saved_head_kkt` | 8.99280649946e-15 |
| `response_training_data_loss` | 6.2172489379e-15 |
| `response_training_profiled_objective` | 5.55111512313e-16 |
| `response_validation_loss` | 1.33226762955e-15 |
| `ridge_coefficients` | 2.06585326623e-09 |
| `ridge_saved_coefficients_kkt` | 6.61870558361e-11 |
| `ridge_validation_loss` | 5.59130057809e-08 |
| `native_test_predictions` | 3.02406988339e-11 |
| `native_dense_path_predictions` | 1.24344978758e-14 |
| `summary_Q` | 1.94122495856e-13 |

## 学习与预算

32 个响应最终 checkpoint 中，25 个存在非零的已学习非线性参数变化；16 个所选 epoch 达到 320 上限。完整分市场、设置和家族明细在 JSON 中保留。

这些检查验证保存的参数确实满足相应训练读出方程，并复现已发表在运行收据中的损失和预测。它们不重演 Adam 的完整参数轨迹，不证明非线性全局最优，也不把达到预算上限解释为已充分收敛。最佳 checkpoint 没有保存 trial 最后一次尝试参数，故该最终尝试向量的变化量只保留原收据，未声称独立重建。

所有输出仅含误差上界、计数、模型级参数变化和文件身份，不含逐起点数据或预测。信息设置及开发证据角色沿用冻结方案；本审计不改变科研结论。

审计程序 SHA256：`3e829579cffc778ba8674697d3ccddaf85d1fbd87efe64b517ce320ba809088e`。

冻结方案 SHA256：`78893370f6862b1295ba9cfcc5e71fc89776944aeae39346de788a4e09cc8880`。
