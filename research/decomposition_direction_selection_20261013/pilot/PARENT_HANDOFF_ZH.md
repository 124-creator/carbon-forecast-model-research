# 父验证与结果摘要

## 判断
末端修订确实存在，但当前广州开发数据未建立“历史成熟修订可靠性能够稳定改善未来原价预测”的机制。半宽5/10/20均H1通过、H2/H3失败；不增加注意力。

- 开发评分（越低越好）：RW=1，MLE480=0.94979323，RAW16最佳=1.01461535，RAW8最佳=1.01479773，分解最佳=1.03386076（w10/alpha1），成熟修订对照最佳=1.03961905（w5/alpha1）。这些最佳值是开发集事后选择，不能作为未见泛化结论。
- 修订RMS/单步变化RMS：2020约0.46—0.80，2021约0.77—2.20。大修订不等于可预测的误差。
- w10/alpha1的历史成熟修订RMS与1/5/20未来绝对误差Pearson相关：2020为0.150/0.355/0.603，2021为-0.025/-0.085/-0.153。年份间不稳定。每个步长均检查0.1门槛；即使用步长平均，结论也不变。
- w5/alpha1成熟修订对照相对同配置分解改善不足1%，且仍差于RW/MLE；其相对MLE差值20点块bootstrap区间包含0且上界大于0。所有窗口均未通过H3。

## 复核命令
在仓库根目录执行（bash）：

```bash
python research/decomposition_direction_selection_20261013/pilot/test_pilot.py
python research/decomposition_direction_selection_20261013/pilot/run.py
python research/decomposition_direction_selection_20261013/pilot/verify.py
python -O research/decomposition_direction_selection_20261013/pilot/verify.py
python -O research/decomposition_direction_selection_20261013/pilot/test_pilot.py
```

使用numpy/scipy；calibrate.py仅导入定义，未调用其读取归档的main。run.py依赖仓库既有calibration源码与本地许可价格前缀。父复核不应给数据CSV做全文件哈希，避免越过2022禁读边界。

## 文件
运行前假设DESIGN_ZH.md；源码run.py/test_pilot.py/verify.py；完整配置CONFIGURATIONS.json；ALL_RESULTS.json（六单元逐项原价MSE）；TRAINING_LOGS.json（训练边界、参数和标准化）；DIAGNOSTICS.json（逐年修订、相关和500次bootstrap）；RUN_RECEIPT.json；INPUT_AUDIT.json（仅消费前缀哈希）；PREDICTIONS_LOCAL_ONLY.npz（仅本地审核，勿自动发布）；REPORT_ZH.md；正常/优化模式VERIFICATION.json与VERIFICATION_OPTIMIZED.json。

## 核验与限制
正常与python -O模式均通过451起点、26配置、312个训练块参数预测重放、原价损失恒等、9组未来特征和预测扰动、前缀vintage复算与成熟边界检查。TDD首个检查在源码缺失时确实失败，源码完成后通过。未读取2022或下一边界记录。实验计算约1.3秒，预算900秒；未安装torch，未修改旧结果，未提交推送。

相同Ridge骨干、相同16维不意味着完全匹配输入信息：截断平滑最大回溯8+w，RAW16回溯16，RAW8仅作重构恒等的补充基线。因此不能把模型差异全部解释为修订机制；本轮没有获益，不再扩大模型。诊断关联可能是波动状态的代理，没有证明修订信息独立于波动。分解是截断对称滚动均值，不是VMD/CEEMDAN、不是新算法；本结果不否定其他分解或其他市场。
