# 强 price-only 骨干与煤外生接入契约（实际执行）

## 结果
- 950条广州原价许可前缀，最后2021-12-31；读取原始源码 `calibrate.read_prefix`：无缓冲逐字节，恰好header+950记录，从不请求下一行。先验计数来自历史 INPUT_AUDIT；不能声称整个项目历史从未解码2022边界。
- 初始459行；统一同年完整h=1/5/20目标451起点（2020:224，2021:227）。独立以 `p[t]+prediction_changes` 计算原价MSE，再对每年/每h除RW MSE、六格等权：**0.9497932301207621**。
- 对历史 `state/PREDICTIONS_LOCAL_ONLY.npz:MLE480` 最大差异0；独立重建 `state.history` 的发行参数选择也逐位相同。没有pickle/旧权重，没有运行历史main/全训练网格，没有修改历史产物，没有push。
- normal和python -O验证均通过。代表边界459/499/659各实际执行原始filter+MLE参数选择两次，对可观测末端后的真实价格数组副本施加扰动，预测差异0；另检查初始发行459后的整个未来。此为代表点fresh refit因果性检查，不把保存数组复评分称为全量重训。

## 实际协议、成熟与scale
执行真实依赖 `E:/碳价格/独立数据归档/carbon-forecast-model-research/research/deep_direction_review_20261013/calibration/calibrate.py`，不是重命名弱ridge。MLE为历史31个固定(rho,q)参数候选的永久RW+AR(1)暂态模型；本次只复现固定MLE480，不做新模型/新训练配置搜索。每40个观察点重新选择，fit_end独占；窗口 `[max(1,fit_end-480),fit_end)`，最晚参数价格fit_end-1，初始458创新，以后480。滤波状态可更新到当前发行t。一步创新训练支持不等于每h成熟多步残差支持。

MLE使用原价，无输入/输出标准化、无沿用0.625142379239414或sqrt(h)缩放。Gaussian公共创新方差由历史 `a0+n*log(max(a1/n,1e-14))` 剖面消去。原价评分不标准化；RW分母只是评分归一化。逐次边界、支持、选中rho/q见 REFIT_PROTOCOL.json。

## 源码追踪与verify含义
- `local_mature_feedback_20261013/prototype.py:66-97` 为另一个弱神经实验：first=300，广州scale=0.625142379239414，增量除scale、目标再除sqrt(h)，471评分起点，未要求跨年共同完整目标；不能拿它代替451强MLE。本次未执行其边界解码reader。
- 该目录 `verify.py` 是60预测文件/360单元复评分、有限差分梯度和合成未成熟标签扰动；不代表MLE重训。
- `multi_angle_model_search_20261013/state/run.py:15-28` 的history使用真实calibrate.filter_bank和发行参数40步刷新；`:43-54` 是多步残差修正，成熟idx终点fit-h-1，最多480，RMS列缩放含惩罚截距。本次不接入该修正，不混入historical linear 0.9427054678157569。
- 该 `state/verify.py` 重建history、correct及原价评分；只明示这些重新计算，不代表所有历史实验重训。
- 可查 usages：`multi_angle_model_search_20261013/learning/run.py:62` 调用base.predict(MLE480)，`segmented_price_backbone_20261014/rolling/run.py:57` 复用历史issuance_mle；`local_coverage_model_20261014/run.py`、`coverage_interface_audit_20261014/audit.py`、`next_model_route_20261014/pilot/run.py`、`persistent_residual_search_20261013/run.py` 导入state。源码hash固定在本次验证JSON。

## 固定coal接口，不声明performance
`forecast(p,fit_end,origins,panel=None)` 返回原始变化预测和receipt；noextras完全不改变baseline，all-newinputs缺失逐位回退。`coal_features(panel,origins)` 固定三列：煤已发布价格、过去五次发布变化、以碳观察点计的发布年龄；辅助available_at为真实发布日期映射的碳session index。输入必须已经发行对齐，同源真实release panel，available_at<=origin、age>=0，未满足/缺失列只标记不可用，不填假值、不拿未来数据补齐。

尚无完整真实panel，因此外生修正明确禁用；即使有完整字段，forecast仍返回骨干并报告pending。接口仅暴露features与有效行mask供后续冻结估计器接入，没有训练coal权重、没有coal成绩、没有新算法声明。可用panel首次对齐与发布语义需另行验证。

## 重放
```
python research/exogenous_information_route_20261014/strong_backbone/test_interface.py
python research/exogenous_information_route_20261014/strong_backbone/verify.py
python -O research/exogenous_information_route_20261014/strong_backbone/verify.py
```

本目录依赖本机真实历史源码/许可数据，不是standalone发布包。PRICE_ONLY_LOCAL_ONLY.npz含原价变化信息仅本地。2020/2021反复开发使用，绝非新测试；2022禁止。无新市场源，无traininggrid。RED/GREEN日志记录缺失接口失败与真实接口通过。
