# 运行前固定协议

只实施 design/DESIGN_ZH.md §2 的30参数模型，不使用另一个阻尼幅度pilot。固定 seeds=(7,17)，计算总预算1200秒，CPU单线程float64。窗口60，L12，H=(1,5,10)，长度2400，train1200/val600/test600；每h独立成熟支持，切分前10个origin purge。归一化与特征裁剪仅用训练前缀确定：绝对冲击99.5%分位，其余证据范围[-4,4]。初始暂态零。

本次预先缩减原设计5seeds为2seeds，固定Adam 500epochs lr=.001，full batch，每10epoch记录validation并保存所有训练轨迹。没有追加初始化、调参、改变生成器。若总预算到达保留已有与未完成标记，不补跑挑选结果。按验证loss最低checkpoint评价测试；验证末5次loss相对跨度≤2%、训练loss下降且finite为稳定学习门；末梯度必须记录但不宣称充分优化。稳定性不等于最优性。

方法：full(30)、constant(6，同核常数路由)、age(30，证据置零)、evidence(30，年龄置零)、same-information small MLP(四输入逐冲击共享4隐单元+三h加权汇聚；35参数)、ordinary Gaussian exact covariance MLE(有限核+永久/暂态/测量三个方差，共8参数，学习未知参数；不传truth)、random walk。MLE最大100 Adam epochs .03，独立单步差分联合Gaussian likelihood，不把重叠多步target当likelihood。其验证模型选择同其他方法。线性oracle仅在G1 diagnostic，给已知真实协方差但不给任何可训练方法。

生成器固定：G0永久标准差1，无暂态无测量；G1 Gaussian永久1/暂态.35/测量.1，肩部核节点(1,.95,.85,.45,.2,.05,0)；G2永久1，加Bernoulli(.05)*Normal(0,1)冲击；隐藏AR(.8)状态的sigmoid混合概率决定该冲击为永久或暂态，隐状态不交给任何学习者；测量.1。G5同G1，暂态.6、测量.2，生成前240点burnin含前缀外暂态和相邻反向冲击。每个测试前缀独立零初态，与120lag线性oracle诊断比较。不因阴性结果改变参数。

工程：核端点/单调/平台，三角重构；有限差分所有30坐标及direction；未来suffix扰动和未成熟标签扰动；固定q叠加性；inverse谱范数/condition/max|e|，small perturbation forecast放大。训练val/test每个origin都审计inverse infinity norm(谱范数上界)，最坏谱norm>10停止；额外q=0近不稳定控制。回构通过不意味着正确识别。

机制门：G2两seed分别full相对最强可训练控制至少3%，相对age/evidence分别至少2%（年龄与证据独立增量）；G0相对RW恶化≤1%，G1相对oracle≤5%，G5inverse≤10且学习门稳定。未通过停止当前实现，不停止研究，不访问任何市场/2022/禁用边界。known：q固定后线性innovations、有限FIR/shift-register SSM与永久暂态均已知；年龄/证据增量待证，不声称首创。不commit/push。
