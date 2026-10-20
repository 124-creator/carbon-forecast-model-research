# 只读研究复评包

没有新的市场训练成绩。REASSESSMENT_ZH.md 为研究建议；STRUCTURAL_DIAGNOSTIC.json 是保存参数上的符号对称性与不可见历史方向检查。OLD_RESULTS_BY_HORIZON.csv 只重聚合既有阶段g指标。

## 重演结构检查
将先前交付的 carbon_stage_g_evidence_20261019.zip 解压到本目录下的 stage_g_evidence/，再执行：

```bash
OPENBLAS_NUM_THREADS=1 python structural_probe.py
```

仅需NumPy。程序不读取原始市场序列、不训练、不寻找更优测试配置。

该包不包含原始行情、模型权重或逐原点预测。旧参数仍需从先前已授权交付的证据包取得。
