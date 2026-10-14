# 完整循环模型与响应机制实际结果

本轮父助手亲自实现和执行，无子agent。120正式fits=5模型×2seed×12滚动块；完整451起点。5首块代表未来价格扰动真实重训参数/预测一致；normal/O verify独立重算scores、purge、epochs通过。训练成功fit记录237.8567秒，不含fixture和额外扰动训练，不能将此数称全部总运行时间。

主score：GRUfree .9753725253647837；LSTMfree 1.0127552710017886；GRUfixed .9907953923463548；GRUglobal .9932772484767555；GRUadaptive 1.0083031563500418。无新增机制超过freeGRU或强MLE .949793/state .942705（历史参考非本轮重训）。不保留状态条件tau为贡献，不救tau范围或新增核数。

GRU两seed1.007930512885899/.942814537843669；LSTM .957702196960224/1.067808345043353。不同seed结果分散，不挑有利seed当结果。报告均值是平均损失，不是预测集成；不能混称。GRU可保留为真实骨干，尚未超过强对照。适应响应丢弃不否定全部神经响应路线。

参数数1252/1556/946/947/963；不同head参数不同，训练更新预算相同不是同FLOPs。17/120选择step0，1/120到200上限；patience非收敛失败也非stationarity证明。完整输入60差分+原价净变化全20loss；共享一个训练创新std保持20步原价误差比例，不各h另缩放。noexogenous、不分解。

先实际GitHub检索和NeuralLaplace作者README读取后写protocol再code；作者README是范围证据非全文方法核收。gh得到HEAD4cc654161f61c776054c594c4f545a954083b25a，但README先从默认ref取得，未验证其gitblob完全对应此commit，不宣称完整固定源码核验。状态参数化时间响应有近邻，不将tau head称首创。

所有2020—21已开发，不是独立验证。未读2022price、不读checkpoint、不改变官方包；数值新权重NPZ仅本地，未commitpush。下一贡献必须面对种子稳定性和高效原价学习，而不能挑seed或拼普通ensemble宣称新算法。
