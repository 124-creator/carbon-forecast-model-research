# 固定作者来源的本地复现

`download_public_author_data.py` 使用 Python 标准库从两个固定 GitHub commit 获取文件，并核对字节数、SHA-256 和 Git blob SHA-1。默认下载六个文件，共 1,478,671 bytes：广州作者价格工作簿、对应 VMD 分量表，以及 Zzysty 的广东、湖北、上海和外部因素四表。不下载作者单独的预测成绩 CSV。

| 上游仓库 | 固定 commit |
|---|---|
| StriveOrange/TGSINFO-VMD-NEQQLattice | `3e8a155f034f38cb20430e601c4edb13c22ed7fc` |
| Zzysty/carbon_dataset | `8654ea586188634a519310d88a8c582e9685667d` |

两库在审查版本中均未声明 LICENSE。复现命令将原件留在私有本地目录，不打印市场原值、不上传、不调用收费服务，也不执行作者 notebook 或任何模型。下载成功不构成再发布许可。部分原表含 2024 年记录；整文件字节获取与预测评分分开记账，随后对齐脚本只导出 2024 年前的既定开发日期。

## 下载或只核验已有文件

在本研究目录执行：

```bash
python provenance/download_public_author_data.py --out-dir /tmp/carbon-author-reproduction-new
```

省略 `--out-dir` 会生成一个新的系统临时目录。文件仅在大小与两种哈希全匹配后保存；同名文件若已存在，只核验、复用，不覆盖。不匹配会停止。默认收据 `AUTHOR_DATA_DOWNLOAD_RECEIPT.json` 若已存在，需要用 `--receipt` 指定新的位置，不覆盖历史收据。

```bash
python provenance/download_public_author_data.py \
  --out-dir /tmp/carbon-author-reproduction-new --verify-only \
  --receipt /tmp/carbon-author-reproduction-check-new.json
```

`--verify-only` 完全不联网。本次新增 CLI 已用此前实际下载的六份原件执行这一模式，全部字节与哈希通过。未为了验证包装器重复下载六个原件；不能把离线核验说成第二次联网复现。需要读作者实现时可额外传入 `--include-notebook`，只下载固定 notebook，不执行或打印其输出。MATLAB 文本审查沿用另外的文献来源收据，不在这个六文件包内。

## 生成成交量辅助文件

解析需要 `xlrd==2.0.1` 与 `openpyxl==3.1.5`。使用已有这些依赖的解释器，并把仓库根目录、私有原件根目录和全新输出目录明确传入：

```bash
python provenance/prepare_auxiliary_alignment.py \
  --raw-root /tmp/carbon-author-reproduction-new \
  --repo-root /absolute/path/carbon-forecast-model-research \
  --out-dir /tmp/carbon-volume-prepared-new \
  --receipt /tmp/carbon-volume-alignment-new.json
```

输出保持旧广州／湖北全部价格日期，新增 `date,volume,turnover,price_match,valid_volume`。只有作者日期唯一、价格精确匹配、量有限且非负时标记有效；缺失、重复、价格冲突不填造量。收据与输出拒绝覆盖已有文件。相同固定原件和旧 canonical CSV 应得到本轮哈希：

| 文件 | 行数 | SHA-256 |
|---|---:|---|
| GUANGZHOU_auxiliary.csv | 1,434 | `77f93f3c0579d350196975719aeb2e2e12684bf2e946933db74a07854c2afa00` |
| HUBEI_auxiliary.csv | 1,408 | `15d33910cc9c35d3f51a8043d6d909d5e045aa2c11ee0e55482c2cb5cdcf571d` |

这六份作者文件包含价格来源比较与文献预处理审计材料；它们不是六个独立碳市场，也不是新的盲确认数据。
