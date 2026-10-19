# 阶段 e 复现说明

## 1. 复现对象

本轮包含原始 max80 实验和一次后加 max320 预算敏感性。原始模型实现与运行器在本地 `eff4540abfeb63e599859459d0d188d65fd637a3` 冻结；预算运行器及补充计划在 `c12443a7adeb36f48683290287f85ac7760ee833` 冻结。后来整理的报告、图表、导出助手和复现入口不属于更早的训练源冻结。

根目录是完整 `carbon-research` checkout。阶段 e 通过哈希引用了阶段 d 的若干模型和预处理源；不能只复制阶段 e 子文件夹后删除旧阶段。读取旧实现不改变其结果。

## 2. 运行环境

实际使用 Python 3.12.14、NumPy 2.3.5、SciPy 1.17.0、PyTorch 2.14.1+cpu、pandas 2.2.3、scikit-learn 1.8.0、ewtpy 0.2 与 PyWavelets 1.8.0。绘图使用 Matplotlib。版本是本轮环境记录，不代表要求读者无条件升级其已有项目。

本会话的 Python 位于 `/tmp/carbon-research-env/bin/python`；额外 EWT 依赖位于 `/tmp/carbon-forecast-20261019d-deps`。每个进程限制 BLAS、OpenMP 和 Torch 线程，核心模型使用四个进程、GRU 使用两个进程。

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=/tmp/carbon-forecast-20261019d-deps
```

其他环境应将已安装的 EWT 包加入自己的 Python 环境；不能假定上述临时目录存在。

## 3. 数据取得与固定字节重建

原始行情页为 `https://www.chinatcx.com.cn/tshq`。页面链接的 Body.js 包含天津与 CEA 历史表。官方原始行没有被收入公开代码树，两个提取脚本只接受本轮固定字节；若服务端历史内容更改，会明确报哈希不符，不静默以更新内容替代本轮数据。

已有本轮原始文件时，分别在全新目录重建：

```bash
/tmp/carbon-research-env/bin/python \
  research/literature_aligned_model_search_20261019d/provenance/prepare_tianjin_official_snapshot.py \
  --raw-file /tmp/carbon-next-market-data-20261019d/tianjin_body.js \
  --out-dir /tmp/carbon-stage-e-reproduce-tianjin

/tmp/carbon-research-env/bin/python \
  research/signed_response_search_20261019e/provenance/prepare_cea_official_snapshot.py \
  --raw-file /tmp/carbon-next-market-data-20261019d/tianjin_body.js \
  --out-dir /tmp/carbon-stage-e-reproduce-cea
```

没有原始文件时，可省略 `--raw-file`，脚本会执行一次普通官方公开 GET，并核对返回字节。上游服务不可用或内容变更时，该精确快照不能靠伪造旧哈希来恢复；应记录缺口或单独登记新快照复现实验。提取器不执行网页 JavaScript。

把生成的 `TIANJIN_OFFICIAL_MEAN.csv` 和 `CEA_OFFICIAL_MEAN.csv` 放在一个研究用目录，供下面 `--data-root` 使用。核对标准文件 SHA256：

| 文件 | SHA256 |
|---|---|
| TIANJIN_OFFICIAL_MEAN.csv | `85ca07648855db00fbf67c4e1c083db715f62ab829f74c143c3b3f67da21252a` |
| CEA_OFFICIAL_MEAN.csv | `f9f0e09b7500b2bdb4ec8b16c61d5d5fa2b64de63a24c0cf7bd1d269aa2be75d` |

不补交易日，不重算公布均价，不用挂牌价或收盘价替换目标。目录必须在 checkout 之外，脚本会防止把这些逐日原值写入公开代码树。

## 4. 新环境的一键复现实验

使用 [reproduce_stage_e.py](source/reproduce_stage_e.py)。它核查冻结模型、计划和数据哈希，只重定位文件路径；预检不训练、不计算预测成绩。`--run-root` 必须不存在或为空，重用有结果的目录会被拒绝。

```bash
/tmp/carbon-research-env/bin/python \
  research/signed_response_search_20261019e/source/reproduce_stage_e.py \
  --data-root /tmp/carbon-stage-e-reproduce-data \
  --run-root /tmp/carbon-stage-e-reproduce-preflight \
  --with-budget320 --preflight-only
```

实际重训使用另一个全新目录：

```bash
/tmp/carbon-research-env/bin/python \
  research/signed_response_search_20261019e/source/reproduce_stage_e.py \
  --data-root /tmp/carbon-stage-e-reproduce-data \
  --run-root /tmp/carbon-stage-e-reproduce-full \
  --with-budget320
```

该命令执行原始 244 拟合；加 `--with-budget320` 后再执行补充 170 拟合。省略该参数即可只复现原始实验。它不改模型源、优化网格、切分或指标，也不自动发布远端内容。复现不会把已经评分的数据重新变成未见确认；运行上下文明确记录这是已完成实验的 replication。

**预算计划的关联哈希需要区分科学设置与本次运行记录。** 原始 core 协议包含创建时间与数据路径，因此全新运行的协议字节不可能自动等于本轮旧协议。薄封装生成一份注明“复现”的补充计划副本，只更新新 core 协议哈希及路径元数据，保留全部科学设置，计算新副本 SHA 后传给冻结预算运行器。原始计划不被改写，也不伪造旧时间戳。

## 5. 本会话实际执行命令

以下命令已经完成，列出是为了说明真实执行方式。**不要向同一个有结果的目录再运行。** 原始统计计划 SHA 为 `b6f33c9c328f37f9a283275d6cc36e4e48153bc3322d45e2b8deb74e3d1d987c`。

```bash
/tmp/carbon-research-env/bin/python \
  research/signed_response_search_20261019e/source/run_signed_response_search.py \
  --datasets research/signed_response_search_20261019e/provenance/RUN_DATASETS.json \
  --freeze-plan research/signed_response_search_20261019e/audits/STATISTICAL_PLAN_BEFORE_FIT.json \
  --freeze-plan-sha b6f33c9c328f37f9a283275d6cc36e4e48153bc3322d45e2b8deb74e3d1d987c \
  --out /tmp/carbon-signed-response-runs-20261019e/core --workers 4

/tmp/carbon-research-env/bin/python \
  research/signed_response_search_20261019e/source/run_neural_references.py \
  --datasets research/signed_response_search_20261019e/provenance/RUN_DATASETS.json \
  --plan research/signed_response_search_20261019e/audits/STATISTICAL_PLAN_BEFORE_FIT.json \
  --expected-plan-sha256 b6f33c9c328f37f9a283275d6cc36e4e48153bc3322d45e2b8deb74e3d1d987c \
  --out /tmp/carbon-signed-response-runs-20261019e/neural_references --workers 2

/tmp/carbon-research-env/bin/python \
  research/signed_response_search_20261019e/source/run_budget_sensitivity.py \
  --original-core-out /tmp/carbon-signed-response-runs-20261019e/core \
  --addendum research/signed_response_search_20261019e/audits/BUDGET_SENSITIVITY_BEFORE_RUN.json \
  --expected-addendum-sha ac1117d24f395cb7b3331ffd2e4de3d6e936f9b9891748204c4aa8df394de72c \
  --out /tmp/carbon-signed-response-runs-20261019e/budget320 --workers 4
```

## 6. 独立复算与统计

原始审计器读取真实逐原点 NPZ、选择/最终 JSON 记录与二进制状态，用独立 NumPy 计算核心输入、线性方程、预测与指标；GRU 为冻结框架只读回放。

```bash
/tmp/carbon-research-env/bin/python \
  research/signed_response_search_20261019e/audits/independent_stage_e_statistics.py \
  --core-run /tmp/carbon-signed-response-runs-20261019e/core \
  --neural-run /tmp/carbon-signed-response-runs-20261019e/neural_references \
  --out /tmp/carbon-stage-e-independent-replay
```

新环境可增加 `--data-root` 指向两个同名标准数据文件，并将 core/neural 参数改为本次复现输出目录。统计计划保持原始哈希。审计包括每市场各 50,000 次、20/40 两种块长、8 项核心家族区间与 2 项次要参考边际区间；随机种子和索引共享顺序见冻结计划，不根据结果改动。

原预算审计器绑定本轮原计划与协议字节，用于验证本轮物证：

```bash
/tmp/carbon-research-env/bin/python \
  research/signed_response_search_20261019e/audits/audit_budget_sensitivity.py \
  --original /tmp/carbon-signed-response-runs-20261019e/core \
  --budget /tmp/carbon-signed-response-runs-20261019e/budget320 \
  --report /tmp/carbon-stage-e-budget-replay.json
```

该物证专用审计不能把复现的新时间戳/新路径假装成本轮旧记录。新复现实验的数值可对照公开选择和汇总结果，但新运行身份、原始数据哈希与科学设置应分别记录。原始记录的审计已经完成，无需为生成图表再训练模型。

## 7. 导出与绘图

```bash
/tmp/carbon-research-env/bin/python \
  research/signed_response_search_20261019e/source/export_stage_e_results.py \
  --run-root /tmp/carbon-signed-response-runs-20261019e --include-budget320

/tmp/carbon-research-env/bin/python \
  research/signed_response_search_20261019e/source/make_stage_e_figures.py
```

导出只接受已完成且源哈希未变的运行，并使用 JSON 文件名允许列表。自行训练的数值参数有意保留；逐原点数组、原始行情和二进制权重不被拷入公开树。若目标文件已存在且内容不同，会拒绝覆盖本轮证据。绘图只使用公开汇总值，输出 PNG 与 PDF，不重新访问市场数据。

## 8. 预期核对值

原始主模型 Q：天津 `0.920781189956941`、CEA `0.878516089937736`；补充主模型 Q：天津 `0.9222501584181703`、CEA `0.8714230126152055`。原始共 244 拟合、11,885 正更新，补充共 170 拟合、32,683 正更新。不同库或硬件的浮点/优化差异需要报告，不通过修改数据或选择规则强求旧小数完全一致。
