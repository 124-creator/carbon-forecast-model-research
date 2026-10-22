# 一次前瞻记录脚本：代码与开发回放交付

本次完成 **code + purechecks + 2025 prediction-only replay**。没有执行真实2026前瞻入口，没有产生以2026-10-21为origin的预测、claim或commit文件；没有读取GRN的2026目标/评分，没有训练、evaluate、评分或安排每日任务。旧CSV、checkpoint、frozen源码及报告未修改。

## 1. 审读入口与源码哈希

[prospective_forecast_once.py](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/prospective_forecast_once.py>) 共539行，SHA256：

`89c192ffe1b997438051bb5abfc032af4be11b5fabb63238effbc2cd2c5b9a40`

关键位置：第56行时间门；第79行原source校验；第127行development gate及冻结checkpoint加载；第238行无标签观测前缀；第250行直接query forward；第285/308行严格append与输入as-of；第335行私有frames组装及data-buffer扩展；第372行exclusive输出/commit；第425行真实前瞻入口；第463行purechecks。

[交付manifest](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/prospective_forecast_once_delivery_manifest.json>)集中列出源码/检查/回放SHA和验收结果。

## 2. 本轮实际执行的两个入口

```powershell
python -B prospective_forecast_once.py --check-only
python -B prospective_forecast_once.py --development-replay
```

使用已有Python环境及冻结runner的实际API。入口默认仅purechecks，没有时间覆盖参数。`sys.dont_write_bytecode=True`和`-B`避免改写已有模块缓存。

### Purechecks

[纯检查记录](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/prospective_forecast_once_purechecks.json>)实际UTC为2026-10-21T21:47:26.048945Z，源码哈希与交付一致。通过内容包括：

- 截止时刻相等和晚于截止均拒绝，输入fetch处于未来或早于完成收盘也拒绝。
- synthetic旧prefix篡改和重复追加bar拒绝。
- synthetic模型在完全不提供y的情况下，对最后观测索引给出正的h1/5/22预测。
- synthetic扩展只改变m0/P0/K0/observed/cells数据buffers，旧prefix逐值exact；learned θ、bank、decoder、AR不变。
- 不打开真实市场CSV、归档NPZ或checkpoint，不训练、不evaluate、不评分。

### 2025 prediction-only replay

[开发回放JSON](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/prospective_forecast_once_development_replay.json>)的实际生成时间为 **2026-10-21T21:47:59.451968Z**，两资产历史origin都是 **2025-11-28**。它明确标记 `DEVELOPMENT_REPLAY_NOT_FORWARD` / `is_forward=false`，不能计作新前瞻确认。

每个资产25组预测：四个R20变体各3 seeds及ensemble、oldGRU 3 seeds及ensemble、R16 raw/frozen、MEM stationary/integrated、EWMA22。所有预测都是有限正值，目标为null，没有计算任何loss或score。

- 50项完整2025开发预测比较，最大绝对误差 **1.4094628242311558e-18**。
- 50项历史query单行比较，最大绝对误差 **2.303929616531697e-19**。
- 沿用原runner `rtol=1e-11, atol=1e-14`，没有放宽门槛或替代失败模型。
- 52项冻结源/CSV/参数/checkpoint及selection哈希在回放后重新核对一致。
- AST审查未发现fit/train/backward/step/evaluate/score_predictions/collect_predictions属性调用。

[回放NPZ](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/prospective_forecast_once_development_replay.npz>)不含target或label数组；所有预测数组与JSON逐值一致。[开发commit](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/prospective_forecast_once_development_replay.commit.json>)只证明开发产物完整性，`is_forward=false`。

## 3. 模型与数据接口边界

没有调用原runner的collect_predictions，因为其HAR/HGB路径包含拟合。新脚本只复用其checkpoint接口、prefix buffer扩展、预测对比容差和fixed-KF预测函数；HAR/HGB/GK/RS按授权省略。

R20/oldGRU只加载原state_dict与固定selected epoch，MEM只读取原fitparams，R16只读取原dynamics。pre2025 normalizer/bank由HB的私有development reference复现，并要求原preprocessing digest exact；没有拟合新的AR、θ、normalizer、bank、neural权重或MEM参数。原模块global SOURCE/RESERVED/SOURCES在检查后保持不变，无旧源码改写或global monkeypatch。

推理张量只包含已观测log_proxy/even/cell_b/cell_R/F17，不含y。对query建立无target/split标签的私有观测前缀；R20/oldGRU只过滤到query+1。R16私有query索引数组只是forward地址，不含对应未来y。MEM/EWMA同样只读取截至query的数据。

目标定义保持HB的Parkinson代理：后续1/5/22条有效日线记录的代理均值，**包括已完成但真实零range的bar，并非只数正range日**。过滤器的observed标记不改变这一目标时钟。

## 4. 待审读的真实前瞻入口：本轮未执行

脚本另提供 `--record-prospective`，其执行授权留给主线程审读后处理。本交付没有调用该入口。

其固定条件：

- origin=2026-10-21；真实主机UTC必须早于 **2026-10-22T13:30:00Z**，等于截止也拒绝。记录真实started/created/commit时间，不接受传入now、origin或deadline来回填。
- 绑定旧frozen-selection字节SHA `d39a08d0db8f99eefb37e0fbac75c0821eee9578719d8d6264c785a387785617`，复用原HB/RR source校验和checkpoint selected epoch合同；这仅复用完整性门，不把旧allow_test合同当作新的2026确认。
- 绑定[最新完成输入manifest](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/data_acquisition_round13/KRBN_KCCA_latest_complete_inputs_20261021_round17_manifest.json>)的SHA `fc34fb4b1e0fa1d22ce87a4b36849ad7f89494b6a277a5136374ceb44c1a1ef3`；检查新单bar和raw响应的SHA、fetch时间及完成收盘时点。
- 原2026-10-20快照与新单bar只在私有frames/new snapshot拼接；旧prefix每列exact，新增日期必须恰为2026-10-21。新snapshot按17位有效数字保存并round-trip重读逐值核对，原CSV不追加、不替换。
- 先通过开发gate，再扩展data buffers，最后直接query `n-1`。HB兼容helper只能构造截至当前已成熟的历史target，并断言全部成熟于最后输入日；最后query不在其target-origin数组中。未知Oct22+标签不构造、不写入NPZ，JSON target=null。
- horizon是后续观测session序号，不伪造未来交易日或结果；所有实际未来观察只能另写actuals文件，不修改原预测。
- JSON/NPZ/claim/commit均exclusive；已有输出即拒绝。commit绑定JSON与NPZ哈希；缺少accepted receipt的entry不得视为有效记录，若跨截止/主机时间倒退则receipt拒绝，不回写时间或重写payload。

这套真实append/forecast路径尚未针对2026实际输入执行；本轮通过的是纯检查和已有2025origin的prediction-only回放。它不预承诺未来N个样本成功，也不把旧2026测试重新当成新确认。
