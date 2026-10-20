# 单次固定配置：外生信息增量隔离
执行前冻结。只测试 coal diff (+1 calendar day) 与 FX log(CNY/USD) 1/5-observation returns (lag2 implementation: date-3 calendar days cutoff)，不调整配置。原煤123公告，不构造面板。
原 issuance-vintage MLE480；950价格前缀至2021-12-31，451同年起点，h1/5/20，480成熟窗，40刷新。每块共同训练池为原成熟窗中煤与FX均有效起点（同时用于price-only、coal、FX），验证最后80起点，h20严格purge。
价格状态使用已有8块变动汇总widths=[1,1,1,2,5,10,20,20]。标准化后附加常数列，全部参数含截距同样受到 n*alpha 惩罚；price-only在inner训练选择[zero,.1,10]，并固定该已选择校准后才估计source增量。源向同价格状态做训练内ridge部分残差化；source readout没有独立截距，不再中心化部分残差。源projection及readout alpha共同选[.1,10]，增量zero允许参与验证。零候选只是一种选择，不保证外层安全。
这明确是sequential regularized partialling，不是joint orthogonal、不是DML、不是新算法。价格校准固定后源仅拟合剩余残差；所有nuisance/scales均在purged训练内部估计。外层源缺失则精确回退该块匹配price校准，不回退原MLE。全部451起点公布原MLE、同池price-only、coal增量、FX增量；历史strongstate .942705仅历史参照，不假称同配置重训练。
公布全部alpha/zero次数、year*h成绩和共同支撑/独立煤事件数。代表刷新点执行未来源值与未来价格扰动的实际完整重拟合，对比参数、选择和发行预测。保存后独立normal/-O重评分，禁止assert作为验证。只此一轮，若无增量停止两源当前路线；若有只是信息证据，下步才设计。无commit/push。
