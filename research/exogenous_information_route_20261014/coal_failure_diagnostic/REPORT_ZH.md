# 煤信息失效机制诊断：停止当前核，不否定全部煤价

## 决策
不保留已失败的固定共享核，也不继续调核形状/alpha。仅保留一个尚未验证的下一步假设：**价格偏置校准与煤的条件增量信息分开估计**。这是成熟训练上的部分线性/残差化 ridge，对应既有 forecast encompassing/orthogonalized nuisance estimation，不是新算法、不是神经模型，当前结果不支持 neural 升级。

## 实际执行范围
读取 coal_training/run.py、CONTRACT、saved predictions、training logs、scores；使用原发行历史 MLE 残差，40观察点刷新、同年451起点、h=1/5/20、seed=None。950条价格仅通过既有逐字节定额读取器读到2021-12-31，绝不读2022或下一行。保留原四组预测和原日志选定alpha，没有新training grid，没有新阈值或超参选择。每块完全成熟支撑与原索引一致；inner训练最晚h20目标严格早于原最后80起点验证段。

48块原参数重建；24个year×h×model单元；144个inner方向单元。正常与-O均通过，原校正预测最大差1.776e-15，MSE贡献分解恒等式最大差3.109e-15。源码的read_prefix审计输出被重定向到本目录，未覆盖原实验审计。价格残差的8维price nuisance仅为诊断，采用已有分块增量和原alpha，不是历史strong state-corrected .942705的复现，也不是新部署比较模型。

## 原实现真正估计了什么
- Linear每h有不惩罚截距；Kernel将三个h残差flatten，估计一个不惩罚共享截距，而level、age按h放大，shock为diff×[k(age_sessions+h)-k(age_sessions)]。共享截距不等价于每h偏置校准。
- 有效发行及拟合都排除了missing，故missing列常零：中心化设计秩3/4。不是求解失败：active condition 1.105–2.149；加正则全矩阵condition最大17.282。零列没有新增信息，ridge仍可解。
- 每块n=366–452是观察起点数，不是独立煤发布次数；训练level/diff变化段仅15–44，重叠h标签与重复持有值进一步削弱独立支撑。不得把n当独立样本量。完整scale、mean、beta、intercept均保存在BLOCK_SUPPORT_SCALES.json。
- 48块中40块alpha=10、8块alpha=.1。标准化ridge方向沿特征值d缩小为d/(d+n alpha)，但截距完全不缩小。强正则使模型接近残差均值回补，而不是接近原MLE零校正。这里没有比较新alpha，不能断言减小正则会解决。
- two-event raw diff不是识别出的煤意外创新；训练块diff均值约-4.19至+3.61，存在有限样本漂移，不能作为结构性创新零均值。且age计时单位不同：输入age为自然天，核为观察点age。无证据把固定12观察点核认作实际煤冲击响应。

## 可加和误差账本
设e=truth−MLE、c=煤校正，ΔMSE=E[c²]−2E[ce]。对截距/level/shock/age/missing各项c_j，分配E[c_j c]−2E[c_j e]，其和严格等于ΔMSE；交叉项如此分配，不是唯一因果归因，也不是干预后的重训练消融。

primary lag+1：
| 模型 | 年份 | h1 ΔrelativeRW | h5 | h20 |
|---|---|---:|---:|---:|
| Linear | 2020 | +.0494 | +.2383 | +.5951 |
| Linear | 2021 | +.0008 | +.0271 | +.0709 |
| Kernel | 2020 | +.4855 | +.3644 | +.3579 |
| Kernel | 2021 | +.0389 | −.0784 | −.1224 |

2020 Linear h20：Δ原单位MSE .51560，其中截距贡献+.5184、level−.0029、shock−.0007、age+.0008。平均截距校正+.5915，但原MLE残差均值−.00762；校正能量.4604，alignment−.0276。主要损失不是shock过大，而是无惩罚历史残差均值负迁移。h1、h5也主要为截距贡献。

2021 Linear h20：截距贡献−1.4128（有帮助）、level+2.7583（有害）、shock−.2479、age+.0010；校正能量4.5644，2alignment=3.4659，净损失+1.0985。这里不是简单“方向全错”，而是level暴露与幅度/分布变化压过帮助。shock在该单元是净有益的，不能宣称煤冲击完全无信息。

Kernel 2020 h1：截距+.1002、level−.0101、shock+.0015、age−.0015，净损失+.09016；共享截距明显挤压短h。Kernel 2021 h20则有所改善（Δrelative −.1224），但不能用这个事后单元保留核或重选训练方案。lag+2的完整24单元见YEAR_H_SUMMARY.csv/JSON；不据其未来成绩挑发行滞后。

## 历史训练/inner purged方向能否判无信息
价格8维nuisance用原purged train估计，煤level/diff/age同样向price nuisance做正则部分残差化，再用原alpha拟合煤增量。所有诊断输出不回灌原预测；没有根据外层future outcomes选训练。这里的alpha本来由同一80验证段选择，所以inner诊断也不是独立未参与选择的测试证据，不能据此给显著性或新gate。

lag+1 linear：12块中煤增量inner ΔMSE为害h1=9、h5=6、h20=8；validation alignment为正h1=6、h5=6、h20=5。price-only校正为害分别7、5、6块。原完整煤校正为害分别7、5、5块。h20煤正交增量平均ΔMSE却为−.1386，说明少数块幅度影响均值：既非一致增益，也不是全无信息。lag+1 kernel继承其原alpha后对应煤增量为害8/7/8块；共享核训练先验不是这项纯线性诊断的一部分。

结论是：成熟历史方向不稳定，价格偏置校准本身也不稳定，无法用此次诊断区分“煤本身无条件信息”与“价状态条件化不足/事件支撑不足/核错误/冲击非创新”。条件相关性是诊断，不是因果煤效应；inner未提升也不能否定所有非线性信息。没有独立事件bootstrap或依赖稳健encompassing检验，不报告p值。

## 唯一下一机制设计（尚不训练）
目标：检验煤是否包含超过价格偏置的**目标相关条件信息**，而不是学习一条先验响应核。

固定已有发行ledger与+1主滞后、无missing精确回退、原MLE issuance-vintage、每h成熟残差支撑、480窗/40刷新、原80段h20 purge及原alpha候选/选择流程不变。价格端承担每h截距/状态校准；煤端只使用训练价格状态条件化后的diff innovation u=diff−m_price(past_state)，并以相同价格nuisance得到v=residual−g_price(past_state)，估计一个每h的source增量β_h u。煤端无独立常数项，不能把残差均值当煤信息；不加入核、共享h约束、neural或新阈值。价格校准与price+煤只作同支撑成对比较，不用外层年份损失选择训练路线。所有nuisance在各purged训练段内估计，部署时只成熟训练；不能用验证段拟合nuisance。当前正则部分投影并不严格保证正交，因此要称regularized partialling，而不是DML理论已成立。

可证伪预测：当煤不是信息而只是价状态代理时，条件diff增量在既有purged验证上没有稳定方向/净能量收益；若分离后仍只有截距回补贡献而煤增量无收益，则不保留该煤路线。下一实验只报告原评价规则下成对全年/h结果与既有inner损失，不新建阈值、不做traininggrid，也不按2020/2021外层结果选择alpha或门控。设计只是对当前诊断的低容量估计假设，不是声称已成功。

只有先获得控制price-only、同发行信息、同成熟支撑下可重复的nonlinear signal证据（固定小非线性超过线性，优化充分，跨inner方向有一致性，并保留无信号控制），才有理由讨论neural。当前不存在该证据。

## 近邻与流程限制
四条primary检索总上限已用完；第一encompassing query服务失败，第四找到Clark–McCracken primary NBER manuscript与publisher记录；其他检索找到Granger–Ramanathan 1984 publisher摘要、arXiv:1802.08667v6正交/Riesz估计primary摘要。精确原始响应见PRIMARY_SEARCH*.json。方法正文fetch均失败（工具报告private/internal network address），未绕过，不能宣称全文审阅。

**未满足“先检索再诊断”顺序**：初始URL抓取失败后先执行了诊断，成功检索后才形成最终机制设计。保留失败收据并明确这一流程gate失败；不要把此报告描述为完整通过literature-before-execution。NEIGHBOR_AUDIT只支持既有机制邻近性，不认证新颖性。

没有读2022价格、没有修改原coal预测/训练、没有重选超参，没有commit/push。asof迁移快照依然不是历史首版认证。整个执行在15分钟限额内；单次数值诊断小于1秒。
