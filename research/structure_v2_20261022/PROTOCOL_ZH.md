# 结构候选A与DM_pair：拟合前固定协议

本协议按主任务最终指令冻结于任何候选拟合或结果生成之前。2026只执行本固定方案，不根据结果回头修改。此前未冻结草案中的自由cross主模型、三资产ragged方案及B取消方案全部被本版替代；它们未运行、未产生结果。所有新文件仅写本目录，原数据、原脚本、原结果和报告不修改。

## 数据、时间线与评分

1. KRBN/KEUA/KCCA，h=1/5/22，2024预热，2025淘汰判定，2026一次诊断。原生记录顺序与索引不变，不补日、不填零、不沿用缺失资产值。各资产从2024首原点起每22个原生原点重训。扩展训练；s+h<=refit_t；训练和评分均排除跨年目标。
2. 共同评分键完全复制normalized_comparison_predictions_common.csv的5073个asset/h/origin键；18个已存基准只复用保存预测（含EWMA97、GARCH11_t），不重训。以每键MA22行target评分；原生r²重算必须在rtol=1e-10、atol=1e-15内一致。QLIKE为log f+Y/f，真实零目标保留，不往目标加eps。
3. 本轮恰好6个新模型：A2_bounded（主）、A3_free_cross（诊断）、K1_direct_CC（对照）、K2_no_cross（诊断）、DM_pair（探索性）、K3_no_DM（对照）。不增加期限、特征、专家、池、半衰期或调参。
4. 标准化仅用当前有效训练行均值/标准差；std<=1e-12置1。所有斜率ridge目标项.005*sum(beta²)，截距不罚。QLIKE loglink沿用corrected_risk_experiment.py机制：截距初值log(mean Y)，斜率0；L-BFGS-B maxiter2000、ftol1e-12、gtol1e-7、maxls50。优化失败/非有限值记录并停止，不静默切换优化器。保存最大梯度，正常成功仍须max|grad|<=1e-4。

## A：分量、校正与对照

5. oa_t=open_t*(adjclose_t/close_t)，gap_t=log(oa_t/adjclose_(t-1))，id_t=log(adjclose_t/oa_t)。对有效行断言gap+id与log(adjclose_t/adjclose_(t-1))误差<=1e-12。open/close缺失或非正行不构造分量，保存原始索引；不得删评分原点。若需要的原点评分特征无法形成，记录失败而不是回退或删点。
6. 分别对gap²、id²建立直接多期模型：特征log(rolling mean x+eps)的1/5/22项，含t；目标未来h个原生观察的分量均值。eps沿用该资产pre2024 mean(r²)*1e-4，固定且只加在特征对数。组件和K1共享全部组件/CC/cross特征有效且标签完整的训练行。
7. 正分量与K1训练目标mean(eta+Y*exp(-eta))+.005*sum(beta_slope²)。训练后固定乘性校正c=mean(Y/exp eta)，零目标ratio=0；预测c*exp eta。它是QLIKE下乘性尺度的精确最优解：d/dc[log c+E(Y/exp eta)/c]=0，不称Duan校正或测试期校准。未惩罚截距下c理论近1，逐fit记录。
8. K1_direct_CC使用r_cc²自身的相同log均值1/5/22结构、相同QLIKE/乘性校正与训练行，直接预测同一CC目标。K2_no_cross复用Ghat+Ihat，仅作诊断，不作主模型。按资产、年份报告sum(gap²)/sum(r²)、sum(id²)/sum(r²)、sum(2gap*id)/sum(r²)，检查三项和为1，记录漂移；不利用份额改模型。
9. h1只检验已实现r²=gap²+id²+2gap*id的代数恒等式，不要求预测等于未来实现平方，也不要求预测等于(预测gap均值+预测id均值)²。后者会遗漏条件方差。

## A2：有界归一化交叉二阶矩

10. 两阶段插件估计：先估Ghat、Ihat；使用同一训练行上的分量拟合值固定它们，再用总Y估theta。z=theta0+theta·X_CC，X_CC为K1的三项log均值1/5/22特征、按当前训练行标准化。rho称“归一化交叉二阶矩”，不称Pearson条件相关系数。
11. rho=tanh(z)，显式夹至[-1+delta,1-delta]，delta=1e-6预先固定。稳定计算1+rho：未夹时为2*sigmoid(2z)，夹取等价在[delta,2-delta]；f=(sqrt(Ghat)-sqrt(Ihat))²+2*sqrt(Ghat*Ihat)*(1+rho)。没有事后方差floor，不改变delta。保存夹取次数及|rho|>0.999次数。
12. theta损失mean(log f+Y/f)+.005*sum(theta_slope²)，截距不罚。两个固定初值：4维全零、4维全1/4，各独立优化一次，方法/容差同第4条。两者均须成功、有限且最大梯度<=1e-4，选择训练目标较低者（完全相同选全零）；不按2025/2026损失选择。逐fit保存两目标、参数、梯度和差异，报告目标差>1e-8的次数/最大差，不能只看success。预先用合成数组做中央差分梯度抽查，含未夹及夹取区域，绝对误差<=1e-5。
13. 所有评分原点断言f>0且|f-Ghat-Ihat|<=2*sqrt(Ghat*Ihat)*(1+1e-8)。两阶段有意避开自由三元组的联合不识别和额外非凸优化，但第二阶段本身仍非凸，不声称联合最优。rho误设会造成偏差；同训练集估计分量与theta存在生成回归元/在样本乐观，不称严格预序列stacking。

## A3：自由signed cross诊断

14. cross目标为未来h个2gap*id均值；特征是gap*id带符号滚动均值1/5/22，identity link，训练标准化，闭式ridge最小二乘.5*MSE+.005*sum(slope²)，截距不罚，允许负cross。分量用QLIKE、cross用平方误差，属于混合损失，为表达负值而采用，不声称联合最优。
15. A3=Ghat+Ihat+Chat。保留每个共同评分原点，报告|Chat|>2sqrt(Ghat*Ihat)*(1+1e-8)和A3<=0数量；非正A3的QLIKE标为无定义，不裁剪、不删点、不用剩余点冒充完整共同样本。涉及此组的年/期QLIKE均值及配对差记NaN并报告无定义数量；MSE仍完整可算。A3无定义本身是诊断，不触发对A2或B的技术失败。
16. A2与A3同时改变边界、函数族、尺度、损失及惩罚几何，差值只称“整套交叉项建模策略差异”，不能解释为纯二阶矩一致性约束代价。

## B：固定双资产幅度混合特征DM_pair

17. 只使用固定集合{KRBN,KCCA}，没有三资产ragged分支。每次refit前，eps_i=严格date<refit的正r²均值*1e-4，记录；对数底层a=r²_KRBN+eps_K、b=r²_KCCA+eps_C。仅在两资产当日都有真实观察的日期构造G=sqrt(a*b)，不填充。按共同有效日期的观察步数形成log(rolling mean G)的1/5/22三项，再精确对齐每个目标资产native日期。
18. DM_pair=原QHAR_liquidity全部特征+上述三项；K3_no_DM仅去掉三项。两者共享DM可用的同一训练行、标签、ridge和训练标准化，只有3个额外斜率参数。B没有A的额外乘性校正。原QHAR_liquidity内部eps维持原pre2024定义；混合特征eps在每次refit更新并在下次refit前固定。
19. 每个w记录kappa_w=sum_w sqrt(a*b)/sqrt(sum_w a*sum_w b)，检验0<kappa<=1（浮点容差1e-12）及w1的kappa=1；检验log(mean_w G)=.5[log(mean_w a)+log(mean_w b)]+log kappa。保存各窗口分布及资产/年分布，未使用未来数据。kappa只衡量幅度同步性，不含收益符号，不是相关系数。
20. 正式名称“双资产幅度混合特征DM_pair”；不称共同因子、共同冲击或独立跨市场冲击。KRBN/KCCA自身项与混合特征机械重叠，KRBN又包含相关碳市场敞口，不构成外生识别；即使预测增益存在，也不能唯一归因新信息，因为ridge收缩几何同时变化。

## 淘汰、先例、输出与失败

21. A2和DM_pair分别按2025判断：h5、h22各自都要满足等资产ΔQLIKE对EWMA97和GARCH11_t均<0，且对每个基准>=2/3资产改善；两H都通过才保留。h1只报告不设门，不做跨期限平均，不根据2026改变淘汰结论。性能淘汰仍完成另一个已固定候选；真正数值/技术失败保存诊断并停止。
22. 主任务已核验作者页摘要所述先例：Andersen, Bollerslev & Huang (2011), Journal of Econometrics 160(1):176–189，已有日内连续/日内跳跃/隔夜方差分别建模再合成总波动预测；本轮未读全文。因此定位仅为“碳ETF应用+有界合成约束”，不声称提出分解或新框架。
23. 完整输出predictions_common.csv（18旧+6新，共24模型）、summary.csv、pairwise_equal_asset.csv、pairwise.csv、manifest.json及<=80行REPORT_ZH.md，另保存拟合/原点分量/rho/双资产组成与kappa/代数审计。QLIKE差负数更好，MSE比<1更好；先逐资产后等资产平均。NW lag=max(5,h-1)，正态双侧仅探索性；逐日损失差最大<=1e-12标为数值持平，不将浮点残差解释为显著性。
24. manifest包含协议/脚本/输入/base脚本hash、计数、断言清单和阶段状态。冻结协议hash先回报主任务，再运行A，A完整跑通后运行B。若技术失败，保存failure.json、失败manifest和简短报告，不能将残缺输出标成完整共同样本；不修改旧文件或据结果修参数。A3的QLIKE无定义按第15条透明保留，不自动淘汰原点。

冻结依据：主任务最后确认的A2稳定表达+delta=1e-6、两初值诊断、A3保留无定义原点、恢复固定DM_pair及kappa诊断，和h5/h22逐期限淘汰线；所有结果尚未生成。
