# SOLID B1 的可移植重放

本入口生成并执行一次**新的 SOLID 适配参照复现**。它使用用户已经重放的 RRC D1 离线状态，不追加 D1 骨干拟合。原 B1 的历史协议、源文件与结果保持原样；新运行不是原始执行、外部预注册或未见确认实验，也不能改变 RRC 已停止的决定。

入口为 source/reproduce_solid.py，SHA-256：

~~~text
71864c8422a7bb64d6b9520193ef7e410d36ee1b88a290681bdcc9da6fa373df
~~~

## 1. 需要的输入

需要完整的公开研究代码包，以及三个用户本地位置：

| 参数 | 内容 |
|---|---|
| --rrc-replay | 已运行 D1 的目录；包含 provenance/REPLAY_PROTOCOL.json、原字节 D1 代码和 local_only/D1/{系列}_s{种子}/。也接受仍持有的原 D1 目录，但会明确记录它不是新 D1 复现。 |
| --previous-study | 前一跨市场研究目录；其 data 子目录包含四个原开发 CSV。 |
| --eex-development | 单独准备的 EEX_EUA_T3PA_daily_development.csv；只接受原开发文件哈希。 |
| --output | 尚不存在的新 SOLID 输出目录。 |

四个旧开发文件名固定为 GUANGZHOU_development.csv、HUBEI_development.csv、SHANGHAI_D4_development.csv 和 EU_REPORTED_development.csv。入口没有确认数据参数。

每个 D1 作业需要 STARTED.json、PRETRAIN_SELECTION.json，以及 shrink=0.1、1.0 的两份 pretrain JSON 和 NPZ。只使用原选择规则已经选出的 **offline teacher**：先按两份离线验证分数选择，精确并列时取较大的 shrink，再核对 PRETRAIN_SELECTION.json。不读取 teacher_final.npz，也不重新用 2022 或 2023 的 SOLID 结果选择初始化。

## 2. 先检查路径

从公开研究目录执行；以下命令不读取 CSV 或 NPZ 内容，不创建输出目录，不进行适配：

~~~sh
python source/reproduce_solid.py prepare \
  --rrc-replay "/path/to/rrc-replay" \
  --previous-study "/path/to/previous-study" \
  --eex-development "/path/to/EEX_EUA_T3PA_daily_development.csv" \
  --output "/path/to/new-solid-replay" \
  --paths-only
~~~

此模式会校验代码和协议元数据、确认所需路径存在，并明确打印 data_hashes_verified=false、state_hashes_verified=false。路径检查通过不能替代数据和状态的完整身份核验。

把 --paths-only 换成 --dry-run，可在不创建输出、不执行适配的情况下完成完整身份核验。该模式读取五个开发文件的字节来校验 SHA-256，但不解析价格；同时读取选中的离线 NPZ 来计算状态哈希。它不会运行网络 forward、反传或优化器。

## 3. 准备新协议，再显式执行

去掉上述检查模式参数即可准备新目录：

~~~sh
python source/reproduce_solid.py prepare \
  --rrc-replay "/path/to/rrc-replay" \
  --previous-study "/path/to/previous-study" \
  --eex-development "/path/to/EEX_EUA_T3PA_daily_development.csv" \
  --output "/path/to/new-solid-replay" \
  --budget-minutes 40
~~~

prepare 只准备文件。它保留以下五份实际运行源的原字节及原 SHA-256：

~~~text
source/rrc_core.py
source/run_solid_reference.py
source/solid_author_adapter.py
backup_review/author_solid/exp/exp_main.py
backup_review/author_solid/LICENSE
~~~

作者源码 MIT 许可证随同保留，无需网络下载。新入口本身也复制到新目录。历史结果、预测、在线状态和模型权重不会被复制到新目录；所选 D1 离线检查点作为用户明确提供的输入被引用。

新协议为 provenance/SOLID_REPLAY_PROTOCOL.json，phase 为 SOLID_REPLAY。它保留原 8 个配置、10 个系列种子作业、20 个 recency 精确等价派生关系和 2022 选择/2023 评价规则。新建的 study_root、开发文件路径、离线检查点路径、初始状态身份及截止时间只写在新协议中。原协议中的历史绝对路径仅用于留档，入口不打开它们。

随后显式运行：

~~~sh
python "/path/to/new-solid-replay/source/reproduce_solid.py" run \
  --protocol "/path/to/new-solid-replay/provenance/SOLID_REPLAY_PROTOCOL.json"
~~~

run 先核对准备收据、新协议和五份原运行文件，再调用**未修改的** run_solid_reference.py。这是唯一会执行真实逐 query 头部适配的入口。结果在新目录的 local_only/SOLID_REPLAY 与 results/SOLID_REPLAY 下；完整时校验 60 个实际流、20 个派生系列、25332 次头更新和 480 个评分格，最后写 REPLAY_COMPLETE.json。

截止时间从 prepare 开始计算，预算可在 1–240 分钟之间明确设置。过期会停止；不会修改协议续期或自动重试。需要再次准备时使用另一个全新的目录。若移动了输入或准备后的目录，也应重新准备新的协议，让实际运行路径再次被明确记录。

## 4. 文件身份与参数身份分别记录

INITIAL_STATE_COMPARISON.json 对每个作业保留：

| 字段 | 含义 |
|---|---|
| checkpoint_sha256 | 用户所选离线 NPZ 的实际文件 SHA-256。 |
| teacher_initial_state_sha256 | T.* 去前缀后，按固定键、形状、dtype 和数组字节计算的教师状态哈希，与原 rrc_core.digest_arrays 完全同口径。 |
| joint_pretrain_state_sha256 | 整份 T.*＋S.* 数组哈希；必须等于相应离线预训练收据的 final_state_sha256。 |
| same_checkpoint_file_as_original_B1 | NPZ 文件字节是否与原 B1 一样。 |
| same_teacher_state_as_original_B1 | 真正用于 SOLID 初始化的教师参数是否与原 B1 一样。 |
| same_selected_shrink_as_original_B1 | 新 D1 的原规则选择是否得到原 B1 使用的 shrink。 |

NPZ 压缩或容器元数据可以使文件哈希不同，而参数仍完全相同。入口不把这两种身份混为一谈。若用户的 D1 复現产生不同教师或不同离线选择，默认如实记为新的初始化，不偷偷加载历史权重补齐一致性。希望只运行逐参数相同的初始化时，在 prepare 或 --dry-run 加 --require-original-state；只要一个教师不同即停止。

PORTABLE_INPUT_REGISTRY.json 另记录相对于用户 D1/旧研究目录的角色和路径规则。最终执行协议按原 runner 的接口保存解析后的本地绝对路径；可移植性来自在目标机器上重新构建新协议，而非改写已冻结的历史协议。

## 5. 本轮验证的准确范围

backup_review/check_solid_portability.py 的 20 项纯合成检查已通过。它检验路径模式不打开 CSV/NPZ、数据与离线状态身份拒绝、验证选择拒绝、NPZ 文件身份和状态身份分离、五份源原字节复制、覆盖拒绝、协议/源码变更拒绝、截止时间拒绝，以及只有显式 run 才到达原 runner 调用。测试在 subprocess 调用处拦截，实际运行源没有执行。

另对本工作区的现有原 D1 做过一次 --paths-only：60 个必要离线文件路径存在，CSV/NPZ 内容读取为 0、模型拟合为 0、头适配为 0。它不等于第三方机器上已经完成市场复现。分别见 SOLID_PORTABILITY_SYNTHETIC.json 与 SOLID_PORTABLE_PATH_CHECK.json。

独立静态审查核对了原/新 D1 身份兼容、旧路径不被打开、离线状态选择、源文件身份、准备与执行分离、预算和新结果隔离；结论见 SOLID_PORTABILITY_STATIC_REVIEW.json。该审查未读取市场或检查点、未运行入口。

## 6. 新运行需要自己的结果审核

此入口完成新运行并核对有限执行数量，不自动调用原 B1 的独立审核 CLI。原独立审核明确绑定原 B1 协议 SHA、phase 和原 D1 指标身份；不能伪造这些身份让新重放冒充原运行，也不能把原历史 PASS 直接复制成新 SOLID_REPLAY 的审核结论。

新协议仍明确指定：各 family 只用 2022 年的等系列、等期限、等种子 MSE/RW 选择配置，精确并列时按配置 ID；再报告同一配置的 2023 年结果。进一步的独立逐预测复算与该选择聚合应为新运行另存审核记录。主入口写出的 480 格和私有预测可供此审核，不是已经完成新独立审核的声明。
