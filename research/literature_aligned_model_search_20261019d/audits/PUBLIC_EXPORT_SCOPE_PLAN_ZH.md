# 最终公开范围增量审查准备

**PENDING_ROOT_FINAL_DOCS**。这是执行清单，不是最终通过回执。主代理通知最终报告、图表及结果导出定稿后，才运行 [audit_public_export_scope.py](audit_public_export_scope.py) 写新快照；原 [PUBLIC_SCOPE_PREFLIGHT.json](PUBLIC_SCOPE_PREFLIGHT.json) 与其脚本保持原样。

## 冻结的范围决策

主代理明确授权公开自研候选与结构对照的数值 `state_dict` JSON，用于机制与实际 forward 复核；它们不能再被笼统归为全部私有权重。完整 torch 二进制 checkpoint、逐点预测路径、无明确再分发许可的新作者原件／逐日原价量及天津官方 Body.js／canonical CSV 仍留运行目录。ScienceDB CC BY 4.0 原 XLS、12 个分区／完整 CSV 与归属资料按已核实许可公开。

## 增量检查内容

1. 校验原 132 文件审查回执 SHA-256 仍为 `0e854dcdc4617ef8b4c6f8c1e1fadded35961da8e7044fddf5617cf0fc481d36`，其原脚本哈希也保持一致。
2. 用原脚本在新的临时目录复核原规则：其他旧研究相对基线无修改，EWT 私有复制件哈希正确、仓库逐点副本不存在，自研数值参数允许公开。
3. 加入天津原 HTML、Body.js、解码 HTML、解析表、四列／五列 CSV、GitHub SQLite 及作者解析源的实际内容哈希；同时加入各阶段公开预测文件清单和二进制 checkpoint 清单中的全部哈希。
4. ScienceDB 数据例外按已保存包清单的精确文件 SHA 判断，不能把未知原件仅改名移入该目录便当作有许可。
5. 检查 CSV／TSV 日期与价量／预测字段、JSON 逐日观测对象及长序列字段、原件／checkpoint 格式与 SQLite 魔数。明确允许的自研 `state_dict` 单独记录。
6. 对原快照列出新增、移除、修改及不变文件数量；核对读取前后内容和文件清单，避免在导出仍进行时写成最终 PASS。
7. 若提供实际交付目录或 zip，同时逐文件检查该副本并固定哈希；不上传、不解压执行、不改动结果。

## 调用方式

在主代理发出定稿通知后，从本 study 目录运行：

```bash
python audits/audit_public_export_scope.py \
  --repo /workspace/scratch/562d14c1693a/carbon-research \
  --study /workspace/scratch/562d14c1693a/carbon-research/research/literature_aligned_model_search_20261019d \
  --out audits/PUBLIC_EXPORT_SCOPE_FINAL.json
```

`--out` 必须不存在；若该名称已用，换一个新回执名。可以加 `--export-dir` 或 `--archive` 指定已组装的交付副本。当前只做语法检查，没有调用上述正式审查命令，也没有根据未来结果修改公开规则。

检查仍是已知哈希、字段与文件格式的可复核审查，不是证明任意文本或图片不能隐藏编码数据。最终回执自身在快照之后生成，不做不可能的自包含内容哈希；之后新增或修改文件仍需要再写新回执。
