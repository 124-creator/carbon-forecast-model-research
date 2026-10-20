# G2决定性消融补充

结论：固定G2生成器、两seed、现有训练协议下，没有达到年龄和证据各自至少2%的独立增量门。不得推论所有年龄/证据机制无效；这里只评价本实现、本模拟器和两个seed。当前机制不放行，原G0 seed17 null失败保持原判。

## 执行范围

8个目标均达到累计500epoch/500steps：seed7 full仅重放原完成checkpoint，六个缺失配置各fresh训练500steps；seed7 constant从原last状态181epoch继续319steps。原checkpoint只有last/selected，没有Adam状态，因此constant恢复时重置Adam moments，明确不是与fresh500等价的连续训练。未重复训练任何已有完整配置，未挑选恢复起点。所有评价仍按原validation最低checkpoint；本次8项选定epoch均为500。没有运行small/MLE/G5等剩余网格，也不能宣称48配置整体完成。

600秒额外执行上限内，profile、训练、诊断及结果保存耗时347.094秒。另做独立保存预测重放、分数复算和工程检查；没有额外训练。原模型、损失、数据生成器、种子、学习率、成熟支持及validation cadence保持原样；没有追加候选或修改原results。

## 结果（等权三h测试MSE；正数为full改善）

|seed|full相对age|full相对evidence|full相对constant|
|---|---:|---:|---:|
|7|+0.047879%|-0.033013%|-1.789346%（constant为不等价resume）|
|17|-0.109994%|-0.034777%|+6.822355%|

两seed均未达到相对age/evidence各2%的门。seed17相对constant改善不能证明年龄独立贡献：full仍略差于去掉年龄的evidence，也略差于age。最强全部可训练控制含small/MLE尚未补充，因此这里只作已比较控制范围的判定，不宣称完整G2强基线门通过。G0 full/RW seed17=1.027213406>1.01仍失败，哪怕G2改善也不得开放市场。

## 耗时与工程

实际三次forward/backward profiling表明不存在Python三角求解循环，调用原生torch.linalg.solve_triangular。此短profile中addmm占self CPU 50.87%，三角求解3.52%；包含启动/测量开销，不能将该比例外推所有epoch。没有执行向量化/缓存修改，故不存在需要证明的改写数值等价性。

每个run已独立重放selected预测及三h分数，并对发行点1810后的全部模拟价格进行扰动，预测差严格为0。原工程全坐标/方向梯度、未成熟标签、叠加性、核平台和危险逆放大控制另在VERIFICATION.json保存。所有新增run逆谱认证上界<=1.713，重构误差<=8.89e-16；这些不证明优化充分或状态可识别。

## 文件与失败

complete.py、verify.py；BUDGET/CHECKPOINT_SCHEMA/PROFILE/RESULTS/COMPLETION/DECISION/VERIFICATION/FAILURES.json；Python/Torch实际profile；每个新增run checkpoint（含last、selected和本次optimizer）、epoch trace、预测npz；EXECUTION及EXECUTION_TORCH日志。

失败完整保留：初次脚本写入出现f-string语法错误，已修正并通过语法检查；首次系统python执行缺少torch，见EXECUTION.log，随后使用原RUN_START记录的torch环境成功执行；训练阶段FAILURES.json为空。静态编辑器系统环境缺torch提示不代表原venv运行失败。没有读取市场/2022，不提交、不push。
