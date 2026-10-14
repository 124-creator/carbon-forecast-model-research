# 反馈到达分组的在线GRU先导契约

已有父实际DSOF/PROCEED作者README及DSOF ER/TD core核读，source先于本设计。完整标签回放、部分标签TD已有；本机制仍可能普通GLS/预条件，不首创。只做筛查，不替代DSOF最终对照。

初始GRUfree来自recurrent_response同协议459fit仅seed17/29新生成数值NPZ，allow_pickle=False读取自己的state，禁止pickle/旧checkpoint。initial固定尺度=该fit训练创新sd，在线不更新scale。初始模型复现首块发行prediction必须一致。

到达T=459..929，先观察p_T，再仅利用过去已发行且o+h=T,h1..20的真实标签更新，随后发行T的20步预测；未来预言标签不进入更新。每到达一步AdamW lr1e-4 wd.001 clip1，固定seed，同一组真实标签及sameoptimizersteps。原发行预测存档不回填。输入60差分，初始模型1280?参数数以实际1252为准。

ordinary：组误差平方均值；diagonal：用历史已成熟发行误差的各期限方差逆加权；correlated：用同一过去误差组估计C，固定.75对角收缩及训练尺度内ridge1e-4，W=(.75diag(C)+.25C+ridgeI)^-1，归一trace(W)=当前组标签数。loss=e^TWe/group_n。这是普通二次加权损失邻域，不能认定新贡献；区分非对角相关作用。

C由已成熟且实际保存的发行误差估计，使用过去最多120完整到达组，至少60组，否则W=I。选择W只使用T之前的完整组，当前T组在更新/发行完成后才加入池。没有把误差协方差称测量噪声、未见标签、经济因果参数。误差均值去除后估计协方差，少样本缩小且检查有限条件数。

相同451起点所有loss、states数值local保留，按年h1/5/20平均两seed损失，记录stepnorm/gradnorm/clipcount/实际条件数、ordinary/diagonal/correlated更新差。不能sameupdates称sameFLOPs。固定40重训GRU旧.975372是不同训练算力历史参考，不作为同预算消融；MLE.949793/state.942705历史参考未重训。

初始预训练复现门、未来suffix真实在线重放T=600之前参数与已发行预测bitwise一致。执行前将边界从539改为600，确保相关结构已启用后仍检验防泄漏，不改变训练超参。normal/O独立score。期限随机perm或新收缩率不追加搜索，结果不胜匹配对照则不继续救阈值；有开发增量也不first/独立验证。no2022price/no commitpush。
