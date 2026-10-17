# 既有机制的只读排除清单

读取日期：2026-10-17。固定仓库：`124-creator/carbon-forecast-model-research`；固定 main commit：`1c67e562e721dd82fff4e4143b97a798a1cf44fa`。

本轮只读取递归树元数据和以下六份 Markdown 报告，没有读取旧权重、调用旧训练器或重新计算旧市场预测。这里的数值是旧报告已披露结果，不是新确认。只据已完成对照排除具体机制，不能据此宣称整个方法家族不可能有效。

| 历史目录/报告 | 已有机制及结果 | 本轮排除边界 |
|---|---|---|
| `innovation_state_learning/REPORT_ZH.md` | GRU 学习增益、20 维移位创新状态、一阶误差反馈；完整对比轨迹中候选约 0.95633，no-feedback 约 0.95355、no-shift 约 0.95407、固定增益约 0.95279；多次选择 epoch 0 | 不把学习 Kalman 式增益、固定可逆解码坐标、移位创新状态换名重启。低分对照不是全家族无效定理。 |
| `predictive_decomposition_model/REPORT_ZH.md` | 四个守恒分量自由混入同一 GRU；输入可逆基变换可被首层吸收，数值等价约 5.55e−17；预算未全部完成 | 不能把该自由分解作为已识别的新函数类或完整实验胜利。 |
| `separated_predictive_decomposition/REPORT_ZH.md` | 四个分量分开状态、共享标量 GRUCell、状态调制分解速率；候选约 1.03086，普通 GRU 约 0.97537 | 不将相同分离状态/动态分解重新包装成当前机制；旧结果限于原开发年份。 |
| `predictive_observer_market/REPORT_ZH.md` | 作者 RKNCell，条件测量方差与固定方差对照；约 0.98977 对 0.99034，2021 变差且 seed 波动大 | 不把条件方差重新命名为已识别的报价测量噪声；没有测量证据便不能赋予该经济解释。 |
| `release_event_learning/REPORT_ZH.md` | 煤价公告只在新发布时更新与持有输入；price-only 约 0.90036，new-event 约 0.94424；部分预算未完成 | 不复用发布日期事件输入作为新创新，不把文档日期加一天等同认证首次发布版本。 |
| `research/broader_direction_scan_20261013/audit/AUDIT_ZH.md` | 线性闭式 2×2：共同 H20/各期限成熟支持 × 40/1 步刷新；分数约 1.03827、1.03662、1.03921、1.03279；正规方程残差约 8.95e−16 | 标签成熟/刷新更快已经被有限检验，不能只换口号；但函数类等价不自动否决估计与优化贡献。 |

前轮还已明确失败或否决 HSC 固定期限竞争、RUM 残差效用记忆、目标端多期限成熟误差对齐，以及稳定极点低秩 head 的创新主张。本轮主候选是在同一预测起点构造嵌套的过去信息视图；不是重启这些残差记忆、在线专家选择或创新状态机制。但它有 MCSSL/KCM 的新外部直接近邻，所以“不同于旧尝试”仍不足以证明原创性。

## 读取文件身份

以下路径前五项均位于 `research/recurrent_response_20261014/`；最后一项如表所列。

| 文件 | Git blob SHA |
|---|---|
| `innovation_state_learning/REPORT_ZH.md` | `1f6d0c967616ce05156be34a4e986e3f393e5f0a` |
| `predictive_decomposition_model/REPORT_ZH.md` | `05eb0191856811fd3f87f227991440e6a4caeeef` |
| `predictive_observer_market/REPORT_ZH.md` | `0b004f194230f01be13ee9d1254aea85422a6dfc` |
| `release_event_learning/REPORT_ZH.md` | `54195c3c21a4ac1aa45f2b286c385949aac1a4b4` |
| `separated_predictive_decomposition/REPORT_ZH.md` | `ef75762e7cbb8706d37b86464f02c33f1c9015a2` |
| `research/broader_direction_scan_20261013/audit/AUDIT_ZH.md` | `dd1685a562017e30c046dd6a13af9ccbfdd0128f` |

源文件链接可由上述固定 commit、完整路径构造。这里没有复制或执行旧报告内的命令，也没有访问历史安全阻断链接。
