# next 实际模型测试规范（不是本次训练结果）

前置条件：全文与作者代码门 BLOCKED，允许执行证伪性合成测试，不允许启动市场实验或宣称新算法。只有下述程序实现且实际通过才可报告模型测试成功；本次没有实施或训练模型。

## 输入与防泄漏

首轮只生成合成数据，不加载市场文件。建议 Python+NumPy+pytest；next 建立 fit_corrected_rrr(X,Y,R,rank,lambda,D) 和 predict(X)，严格使用 DESIGN.md 的闭式解。保存每次 split 的 fit_end、label_end、R来源、随机种子和参数。共同时间网格，lags=4，horizons=[1,3,6]，3个异质输出；目标 y 按目标×步长堆叠。训练/验证/测试依次 60%/20%/20%，在每条边界 purge=max(horizon)=6，仅在训练标签全部可见时允许使用样本。每个 origin 的滤波只读≤origin；测试禁止 smoother 或重估标准化。

## 合成生成器（固定，不看测试调整）

seed=0..19，T=1200，latent_dim=2，observed_dim=12，output_dim=3。
f[t]=diag(0.8,0.3) f[t−1]+eta[t]，eta~N(0,diag(0.3,0.2))；固定随机载荷 L~N(0,1)，每列单位范数。x*[t]=L f[t]；x[t]=x*[t]+e[t]。y[t]=C f[t]+epsilon[t]，C=[[1,0],[0,1],[1,−1]]，epsilon~N(0,0.05I)。生成三类情形：
A. R=0；B. 独立测量误差 R=diag([0.01]*6+[0.3]*6)；C. 压力测试，e 与 f 相关（e的前两列增加0.5*f），经典EIV假设失效。另做负对照：把训练未来 y 随机置换，只用于揭露无信息时过拟合，不能当时间序列主结果。
R的首轮来源是生成器真值，属于合成已知噪声假设，不是现实可识别性证明。第二轮使用两个独立观测副本 x1/x2，R估计为 Cov(x1−x2)/2（仅训练段）；估计误差和泄漏单独记录。不提供副本或外部噪声标定时禁止从普通波动反推 R。

## 必须实现的同信息对照

所有方法使用相同历史、同标签、同 origin 和验证预算：
1. own_history ridge 直接多步，不读取其他来源。
2. joint_history ridge 直接多步（主分母；不是只比自身历史）。
3. PCA factors+ridge direct head，训练X重构因子，rank候选[1,2,4]。
4. DFM state space：相同rank，训练段估计 L,F,Q,R，测试使用 Kalman filter 而非未来 smoother；对每个 h，预测 C F^h filtered_state。若使用 oracle 参数，独立标记 oracle_DFM，不混入可实现基线。
5. ordinary DMD 与 TLS-DMD：以共同lag向量相邻快照识别算子，h步算子推进后接训练读出头；TLS用堆叠[X;Y]截断右子空间再识别算子。记录 rank、谱半径、爆炸预测数，不对测试静默裁剪。
6. PLS2 direct：训练历史到堆叠未来响应，component候选[1,2,4]，缩放只由训练段决定。
7. predictive ridge-RRR direct，等同候选目标 R=0。
8. corrected EIV-RRR direct，使用已知/重复观测估计 R；invalid Gram 候选明确跳过。
9. 已有 clean-input oracle：同 ridge-RRR 直接头但输入 x*，仅用于 oracle_gap，不参与可实现模型胜负。

λ统一候选[0,0.001,0.01,0.1,1,10]，训练标准化后计算；rank≤min(p,q)。按验证集目标平均 MSE 选一次，测试不调参。条件均值输入收缩只作为已有处理方式消融，和 noisy x 部署结果分开报告。

## 单元验证与否决标准

- R=0 时与独立实现的 ridge-RRR B及预测一致，rtol=1e−7、atol=1e−9；λ=0满秩时等于OLS。
- 构造极大 R，G 非正定必须返回明确 invalid，不能悄悄投影成另一目标。
- 目标函数与随机 rank-r 可行 B 比较，闭式解不高于候选，浮点误差≤1e−8相对尺度。
- clean/noisy风险分开：训练去衰减不保证部署收益；情形C允许输，并明确假设失效。
- 记录 scaled MSE、MAE、逐输出逐h损失；20种子差值均值和配对bootstrap 95%区间（2000次，seed=20261014）。
- 主效应 Δ=MSE(joint_history ridge)−MSE(candidate)，再比较强对照 PLS/RRR/DFM/TLS。只胜自身历史不算增量证据。
- oracle_gap=MSE(joint_history ridge)−MSE(clean oracle)；只有 gap>0 才报告 gap_closure=Δ/gap，否则写 undefined；数值为负/超过1照实报告。
- R=0候选退化通过即确认证明与已有RRR重合，不能把该情形性能称创新。优势在正确R下存在、R失配时消失，只支持测量误差假设下定制用途。

## next 产物与运行约定

next 必须实际写 estimator.py、test_estimator.py、run_synthetic.py 后执行：
`python -m pytest causal_multimarket_forecaster_20261014/predictive_factor_design/test_estimator.py -q`
`python causal_multimarket_forecaster_20261014/predictive_factor_design/run_synthetic.py --config synthetic_config.json --output synthetic_results`
这两个脚本当前不存在，命令是下一次实现后的验收规范，不是已运行记录。产物至少含 config、split_manifest、各模型逐origin预测、metrics、invalid_configs、环境版本、真实命令退出码。任何不可运行模型必须列 failed/skipped，不能自动用ridge替代且保留原模型标签。
