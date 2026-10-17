# 新公开数据获取、分区与核验

本入口复现三个独立公开来源，实际下载四个原始文件，共 2,522,422 bytes。其中 ScienceDB 两个版本重复，只计一个观察队列。仅需 Python 与 Excel 读取库，不运行 Torch、模型训练或评分。

## 已完成的实际执行

本轮四个来源文件已通过发布页给出的公开下载 URL 实际下载成功。Mendeley 的 SHA-256/字节数与公开 API 相符，ScienceDB 的 MD5/字节数与公开元数据相符；本轮另记录全部 SHA-256。上海 XLSX 只从固定 ZIP 中取指定成员，并验证其 SHA-256。

`prepare_independent_data.py`、`prepare_sciencedb_data.py` 和 `verify_data_exports.py` 均实际执行。25 个 CSV / 6,724 条记录的原件对应与分区审计通过。

`acquire_independent_sources.py` 是在实际下载之后整理的便利入口，已实测干跑及拒绝覆盖既有目录；没有为了测试这个包装器重复下载数据。该区别记录在 `provenance/REPRODUCER_VALIDATION.json`，不能把干跑说成第二次完整联网复现。

## 1. 准备隔离环境

实际解析环境为 Python 3.12.14、openpyxl 3.1.5、et_xmlfile 2.0.0、xlrd 2.0.1。已有这些依赖时直接使用，不必重装。本轮没有安装或修改环境。

从空白环境复现时，可先安装 Python 3.12.14，然后建立独立虚拟环境并安装本目录 `requirements-data.txt`。例如：

```bash
python3.12 -m venv /absolute/new/carbon-data-venv
/absolute/new/carbon-data-venv/bin/python -m pip install -r /absolute/source/data_expansion_20261017/requirements-data.txt
```

这里提供联网安装途径；这条新虚拟环境路径未在本轮另行创建。

## 2. 查看固定获取计划

从包含本目录代码和冻结协议的完整交付目录执行：

```bash
python -B /absolute/source/data_expansion_20261017/acquire_independent_sources.py \
  --output /absolute/new/carbon-data-reproduction --dry-run
```

`--output` 必须是此前不存在的新目录。即使干跑，已有目录也会被拒绝。干跑不联网、不创建输出目录、不读取价格。

## 3. 实际获取

```bash
python -B /absolute/source/data_expansion_20261017/acquire_independent_sources.py \
  --output /absolute/new/carbon-data-reproduction
```

入口只使用代码中固定的四条发布方公开 URL，验证固定字节数和 SHA-256；无认证、无重试、无镜像。只允许原发布域的正常 HTTPS 重定向。若对方拒绝、来源发生变化或哈希不符，会保留失败收据并停止，不覆盖或继续使用不匹配的数据。

入口把两个准备脚本、独立审计器及 `INDEPENDENT_SOURCE_SCOPE.json` 复制到新目录。原始响应与工作簿留在 `local_only/raw/`；生成 `.gitignore` 防止默认提交这些文件。ZIP 只展开指定碳价 XLSX，检查成员路径、重复成员、链接类型、总展开预算和精确身份。能源、天气、证券及期货文件不进入模型输入。

## 4. 生成开发区与封存区

```bash
cd /absolute/new/carbon-data-reproduction
python -B prepare_independent_data.py
python -B prepare_sciencedb_data.py
python -B verify_data_exports.py
```

这些入口遇到既有清单或同名 CSV 会拒绝覆盖。上海与全国序列以 2023-12-31 / 2024-01-01 为分界，确认截止 2025-12-31；ScienceDB 使用下载前冻结的每市场 60%/20%/剩余顺序划分。所有输出仍在 `local_only`，没有未来日期补齐、价格填充、平滑、异常修正或逆对数重建。

预期新目录产生：

| 文件组 | 目标列 | 文件数 | 总行数 |
|---|---|---:|---:|
| `local_only/development/SHANGHAI_D4.csv` 与对应确认文件 | `price` | 2 | 1,768 |
| 全国 `CEA_LIAO_LOG.csv` 开发与确认文件 | `ln_price` | 2 | 760 |
| `local_only/sciencedb_v6/{train,validation,confirmation_sealed}/` | `mean_price` | 21 | 4,196 |

三个目标定义不同。准备好文件不等于授权自动纳入同一个实验池；root 的最终模型协议决定实际使用队列。

## 5. 应保存的复现证据

保留 `provenance/REPRODUCTION_ACQUISITION.json`、两份数据清单、`DATA_EXPORT_AUDIT.json`、冻结协议和执行代码。新运行时间与清单文件哈希会变化；相同固定来源与版本的原件和 CSV 哈希应与本轮清单一致。文件写入时间并非外部可信时间戳，因此用前置协议、源身份、执行顺序和代码共同说明封存过程。

既有官方交易所接口的失败属于另一条已结束的获取路径，不是此复现流程的依赖。本入口不重新尝试它们，也不读取上一轮市场结果、权重或预测。
