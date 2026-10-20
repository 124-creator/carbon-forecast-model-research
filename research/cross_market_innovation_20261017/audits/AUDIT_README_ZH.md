# 独立审查器：运行与发布边界

本目录保存四市场研究的独立前向、计分和导出审查代码，及原运行生成的审查收据。代码用于复核，不训练新模型。完整结论见 [合并审查报告](AUDIT_D2_CONFIRMATION_ZH.md)。

## 1. 可以复核到哪一层

| 任务 | 需要的材料 | 是否仅凭公开聚合文件完成 |
|---|---|---|
| 重算主要确认统计 | CONFIRMATION_PRIMARY_INPUTS.npz、冻结 paired_statistics.py | 是；不需要模型权重 |
| 重算所有确认模型的分数与 CSV | 确认预测包、原收据、汇总、导出文件 | 解压确认数组及记录归档后可进行 |
| 独立前向核对全部 62 个确认预测器 | 确认权重/预测包、四市场训练前缀、固定确认分区、冻结源和计划、前置审查收据 | 还需准备经哈希验证的数据与路径映射 |
| 独立前向核对 D1/D2/学习曲线 | 对应全部状态和预测包、数据、冻结计划及收据 | 否；原开发数组不随 Git 发布，需按复现入口重跑生成 |
| 检查核矩、信息边界、统计公式的合成样例 | 指定冻结代码、NumPy；部分检查需要 PyTorch | 不读取市场数值，不需要原训练数组 |

发布包的 records/RUN_RECORDS.tar.gz 保留 447 个原运行计划、收据等文件；records/CONFIRMATION_ARRAYS.tar.gz 保留 62 个预测器状态、62 个预测包和 6 个补全器状态，共 130 个文件。按发布包清单解压后恢复研究目录下的原相对路径。归档的内容身份和数量由发布包清单负责；本目录的数值收据核对的是其中对应的原文件身份。

没有公开的 D1/D2/learning 原数组不能从聚合表逆向恢复。重新训练得到的新数组应配套其自身的计划、源哈希与运行收据；不能把新数组塞进旧收据，遇到旧哈希失败后删除检查。原始审查收据证明原运行曾完成哪些核验；新运行应另存结果，不覆盖这份证据。

## 2. 代码入口与本目录依赖

所有数值前向审查器使用 NumPy 与 Python 标准库，不导入 PyTorch 或调用作者模型前向。

| 入口 | 本目录直接依赖 | 必要原运行输入 |
|---|---|---|
| verify_d1.py | statistical_reference.py | D1 冻结源、计划、数据注册表、全部 D1 收据/状态/预测 |
| finalize_d1.py | verify_d1.py | 两个 PARTIAL_141 审查文件、D1 原数组、最终 D1 结果与汇总 |
| verify_d2.py | verify_d1.py、numpy_patchtst.py | D2 冻结源、计划、收据、状态、预测及补全器 |
| verify_development_selection.py | verify_d1.py、verify_d2.py | D1/D2 完整重算文件、开发聚合、24 个学习曲线状态/预测/收据 |
| verify_confirmation.py | verify_d1.py、verify_d2.py、verify_development_selection.py、numpy_patchtst.py | 确认源/计划/规则、62 个状态/预测、6 个补全器、原确认与开发数据、选型审查收据 |
| verify_confirmation_export.py | verify_d1.py | 两个完整确认审查文件、确认预测包、公开 CSV |
| check_statistics.py | statistics.py | 仅合成数据 |

verify_confirmation.py 会先检查 CONFIRMATION_RUN_COMPLETE.json 的全部 62 次拟合已结束，再读取确认分区。该脚本将冻结统计源加载为独立模块；本目录 statistics.py 与冻结 confirmation_source/paired_statistics.py 的原字节相同，SHA256 为 c0003775786ad89aead6a9007d3bf28dc7b01aa647df5be6eaece7b91d4bcd81。

其余合成检查依赖如下：

| 入口 | 附加依赖 | 原脚本定位的代码 |
|---|---|---|
| audit_information_mechanism.py | PyTorch | information_model.py 的原 model.py 路径及 data_core.py |
| audit_mcssl_control.py | PyTorch | mcssl_control.py |
| check_confirmation_dataflow.py | 确认 runner 的导入依赖，包括 PyTorch | confirmation_source/run_confirmation.py 及其冻结依赖 |
| check_statistical_reference.py | NumPy、标准库 AST | 旧 fair_core.py 中 reference 函数文本；本目录 statistical_reference.py |

这些合成检查不新增市场拟合。它们与只用 NumPy 的完整前向复算是不同层级，不能把所有检查描述成完全没有训练框架依赖。

## 3. 原运行目录的命令

原工作区为 /workspace/scratch/b31c5edbd0de，原审查代码位于其 scratch/design_audit_20261017，研究位于 carbon_workspace/research/cross_market_innovation_20261017。下列命令要求该布局及所需原输入已经存在；不要把它们描述成任意 clone 位置的通用命令。

~~~bash
python -B scratch/design_audit_20261017/verify_d1.py --partial
python -B scratch/design_audit_20261017/finalize_d1.py
python -B scratch/design_audit_20261017/verify_d2.py
python -B scratch/design_audit_20261017/verify_development_selection.py
python -B scratch/design_audit_20261017/verify_confirmation.py
python -B scratch/design_audit_20261017/verify_confirmation_export.py
~~~

原 D1 完整收据由完整 PARTIAL_141 前向核验加 finalize_d1.py 最终聚合产生，故清单保留两个名字仍为 PARTIAL_141 的中间证据文件。它们实际已经覆盖全部 141 次前向；finalize 额外验证最终完成记录与公开汇总。verify_d1.py 不带参数也能直接完成前向，但不会自动执行本次 finalizer 的全部最终聚合工作。其余入口不接受 --root、--output 或 --data 参数；不要在使用文档中虚构这些 CLI 参数。

原训练/审查环境使用 Python 3.12.14、NumPy 2.5.4；训练和需要训练模块的合成检查使用 PyTorch 2.3.1+cpu。原聚合、统计及绘图运行使用 NumPy 2.3.5。确认统计已在 2.5.4 审查环境复算并与原数值吻合，但两种版本仍分别披露。原审查显式限制 OMP_NUM_THREADS、OPENBLAS_NUM_THREADS 和 MKL_NUM_THREADS 为 1；设置环境应在导入 NumPy 前完成。

## 4. 换目录运行的要求

原审查器保留原研究布局及来源注册表哈希，未经修改地放进另一个目录不会自动获得可移植性。发布包的便携入口负责准备数据、恢复归档和映射路径；这里明确以下边界：

1. 在导入依赖它的其他审查模块前，把 verify_d1.ROOT 绑定到新研究目录，把 verify_d1.HERE 绑定到新审查目录，并更新 verify_d1.OUT 为对应 d1_audit 子目录。其他模块在导入时取这些常量；导入后再改父模块不会自动更新已有绑定。
2. DEVELOPMENT_REGISTRY.json 的原 path 字段和确认计划的 sealed_path 是历史绝对路径。只改 ROOT 不足以解决。应由明确的包装层将这些路径映射到**字节哈希一致、分区规则一致**的已准备文件，并保留原注册表及原计划供身份核验。
3. 不得为绕过路径问题静默删掉哈希、日期、已完成门禁或成熟边界断言；修改原 JSON 会改变冻结哈希。若包装层需要替换读取函数，应另保存其代码、路径映射和校验收据，明确它不是原审查程序未经变动的一键运行。
4. 合成预检的 MODEL_PATH、CONTROL、SOURCE 等还指向原工作区源码；可以明确映射到发布的同哈希冻结副本。旧统计参考的 AST 对照需旧 fair_core.py 原函数文本；它不属于核心前向复算的运行依赖。
5. 重跑审查会写新的审查时间及输出收据。应在独立复制目录运行，保留已发布原收据以核对原实验。统计重算不重新训练，也不改变比较对象或冻结的抽样规则。

公开确认主统计重算可直接调用冻结 statistics 函数或研究的 aggregate_confirmation.py 入口；后者还依赖完整确认预测包及记录。只验证主要三个模型时，输入字典必须恰好包含已注册的三个名称，不应把完整 13 模型表传给该接口。

## 5. 精确发布清单

AUDIT_RELEASE_ALLOWLIST.json 是本目录逐文件白名单，包含相对路径、类别、字节数和 SHA256。它不授权递归发布整个 scratch 目录。D1/D2/learning 原数组、原市场文件以及未列出的临时文件均不在本白名单中；确认数组归档由研究发布包的独立清单控制。

清单还记录最终审查结论、环境差异和外部依赖。发布时保持所列相对层级，可保留本报告中的相对链接。清单自身不进行自引用哈希；其 SHA256 随父发布清单单独记录。
