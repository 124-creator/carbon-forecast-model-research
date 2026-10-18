# R2D2 作者核心与当前 Torch 兼容性核查

日期：2026-10-18。本说明先记录接入前的作者源码/API 核查，再单列后续本轮接入范围；论文全文由根代理负责。

固定仓库与提交：[bertinetto/r2d2](https://github.com/bertinetto/r2d2/tree/fc0c13ec991bb9f84395cb12de57cb150ce76f8d)，`fc0c13ec991bb9f84395cb12de57cb150ce76f8d`。

[作者目录](../../vendor/r2d2/) 是五文件原样源子集：MIT LICENSE、README、environment.yml、`fewshots/models/r2d2.py`、`fewshots/models/adjust.py`。各文件 bytes、SHA-256 和 Git blob SHA-1 见 [SOURCE_MANIFEST.json](../../vendor/r2d2/SOURCE_MANIFEST.json)。全部 blob 与同提交 tree 一致，没有修改作者函数。MIT 文本要求保留版权及许可；本子集已保留。没有把 README 的其他数据下载链接执行。

## 实际 ridge 实现

`RRNet.rr_standard` 与 `RRNet.rr_woodbury` 提供 primal/dual 解，支持显式逆矩阵或 `gesv` 线性求解。令 \(N=n_{\rm way}n_{\rm shot}n_{\rm augment}\)，作者先将设计矩阵与 one-hot 标签除以 \(\sqrt N\)。对应解为：

\[
W=(X^\top X/N+\lambda I)^{-1}X^\top Y/N.
\]

full `loss` 会拼接常数列，原实现的 identity 包含该列，所以此实现中的 bias 同样受 ridge 约束。不能默认其截距不惩罚。预测还经过可学习 `AdjustLayer` 缩放/偏置，外层是分类 CrossEntropyLoss。改成连续多期回归损失是任务适配，不是原作者图像分类 benchmark。

固定代码的 `LambdaLayer` 是标量或指数参数化标量乘输入 I。根代理已阅读的论文讨论范围更广；不能把论文中的对角正则、IRLS 或其他变体全部说成本文件已运行。

## 接入前的环境检查记录

实际运行 `import torch` 并检查属性，得到 Python 3.12.14、Torch 2.3.1+cpu：

- `torch.gesv` 不存在；
- `torch.linalg.solve` 存在；
- CUDA 不可用。

原 `r2d2.py` 无条件导入 `gesv`，因此即使只选择 inverse 分支，也不能在该环境直接导入原模块。原 `adjust.py` 构造器、完整 episode loss 和 queries 辅助函数还写死了 `.cuda()`；`loss.data[0]` 也是旧标量接口。作者环境文件指定 Python 3.5、PyTorch 0.3、CUDA 9.1。

所以，在这一步源码接入前的检查时，**原 core 已取得、许可已核、API 障碍已观察；作者 core 尚未 import、forward/backward 或训练**。五文件子集本来就不是可安装的原完整包。这一段是历史时点记录，不表示后续本轮仍未执行数值核。

## 后续本轮实际接入范围

后续 [author_kernel.py](../../source/author_kernel.py) 读取并核对原 `r2d2.py` 字节，以 AST 提取未改写的 `rr_standard/rr_woodbury` 函数，提供独立的现代 CPU `gesv` 等价返回接口。没有修改原函数节点；调用时用 `x.clone()` 隔离作者原地缩放，标签仍按作者规则除以样本数平方根。

该兼容层的合成检查已经执行 primal、dual 和梯度比较，收据嵌在 [SYNTHETIC_SMOKE.json](../../provenance/SYNTHETIC_SMOKE.json) 的 `source_author_check`；这里不重新执行检查或读取正式市场结果。详见 [R2D2 本轮接入说明](R2D2_REVIEW_ZH.md)。执行的是数值求解核心，不是原作者图像分类数据、CNN、输出校准、CUDA 训练驱动或论文 benchmark；不能把后者计为已经复现。本分工没有为该兼容性说明另装依赖或修改旧实验。

