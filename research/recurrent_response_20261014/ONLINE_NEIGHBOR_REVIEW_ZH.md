# 在线候选必须面对的直接近邻及暂定研究问题

父实际执行gh搜索online time series forecasting、读取两官方README，并完整读取DSOF src/trainers/trainerBaseERTD.py。未下载论文全文，不称全面查新。

已核近邻：yyalau/iclr2025_dsof（README自述ICLR2025）：slow完整标签experience replay，fast temporal difference、不完整标签；代码td方法使用curr预测无梯度并调用backward_tdlossFT，ER方法待完整标签入池。SJTU-DMTai/OnlineTSF（README自述KDD2025 PROCEED）支持已观测标签、SOLID/OneNet/FSNet/DER++/ER/naive；预先没有读这些各模型核心实现，不声称全部核完。

撤回把‘快速短反馈＋慢速完整标签’或‘没有未来泄漏的滚动更新’单独作为新贡献的建议。DSOF直接覆盖此问题。

暂定不同问题（NOT_ACCEPTED_AS_NOVEL）：以真实反馈到达时点对(发行起点,期限)训练对分组，显式处理同一未来价格在多个期限损失中重复出现造成的梯度相关。仅使用已成熟真实标签，不用当前网络预测未来作标签。候选更新是由历史成熟发行误差估计的收缩相关结构预条件化当前共享GRU梯度；不能把误差协方差当测量误差/潜在信号方差，不能称统计安全。

定义A_T={(o,h):o+h=T,h∈1..20}，发行输入只含o当时历史。组内当前损失梯度g_{o,h}，历史相关结构C仅由更新前已成熟发行误差计算；估计收缩到对角/identity，防少样本逆不稳。具体公式与优化准则尚未冻结，未经机制试验，‘普通GLS/自然梯度/组梯度降相关’已有可能性须再核。重复出现同一个y_T不证明所有梯度冗余，也不证明组相关白化有效。

强对照必须有：同骨干普通成熟mask在线GD、经验回放、标签到达分组但无相关处理、同支持的仅对角期限权重、DSOF完整作者机制、普通整体缩小学习率/梯度范数匹配。若收益只因更小步长或相同优化预算偏差、或C不可识别/逆不稳，停止，不改名称冒创新。

拟定研究范围完整GRU/LSTM+online learning，不外生补丁；煤FX目前无稳定开发增量。还未找到达标新算法，不保证创新发表。没有市场新训练或2022price访问，没有commitpush。
