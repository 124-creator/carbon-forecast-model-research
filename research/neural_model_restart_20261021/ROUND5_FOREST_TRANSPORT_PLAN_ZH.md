# 第五轮：条件森林参考上的神经经验输运

状态：第四轮已验证经验输运内部有小幅收益，完整模型MQS比LSTM-QR好约0.137%、比第三轮倾斜网络好约1.006%，但仍落后同概率rule-QRF约0.524%。目标保持模型创新与成功实验，不转评估论文。

## 主攻假设

当前输运以所有已成熟非零历史响应的均匀分布作为参考。QRF提供更强的状态条件参考，LSTM提供过去64步的顺序信息。下一版将QRF条件参考嵌入经验输运层，由LSTM学习参考分布的条件均值、尺度和形状残差；不是事后按测试市场选择模型，也不是平均两种最终预测。

保留第四轮已核对的条件矩与O(M)完整CRPS，优先只改变参考权重，避免同时改损失、输入和网络预算后无法归因。

## 建议的两项并行工作

### A：forest_prior_round5，仅构建参考缓存

1. 读 qrf_round4.py 的实际forest参数构造器，以及 qrf_round4.json 已由inner验证选择的每market/h leaf设置。不要重新按test调leaf。森林仍为seed11/22/33/44/55、各80树的概率均值，参数与强baseline一致。
2. inner阶段在D.tr拟合scaler和森林；final阶段在D.rf拟合scaler和森林，严格每条标签t_i+h_i在相应boundary之前。需要重建森林以取得训练查询参考，因为旧实验仅存测试weights、未存树；这是为新模型取得新预测，不增加测试调参。
3. 用所有合法query context计算leaf概率，再映射到T.make_bank同序的非零响应。训练查询将尚未成熟的响应权重严格置0，条件归一；若成熟非零leaf质量为0，fallback到当前成熟非零场景的均匀分布。validation和test支持仅当前fit partition的标签，因此均已成熟。
4. 设参考 u=(1-gamma)*u_forest+gamma*u_uniform，先固定gamma=.05，不按test改。uniform只在当前成熟非零支持上，保证保留历史尾部且参考方差不因稀疏leaf坍缩。
5. 存每market两phase的相同D键、scaler、非零bank顺序、query原点到参考矩阵行的映射、u、成熟mask/count及概率p_static。query sid要与seq_transport的全体unique origin索引一致；不存在该h目标的末端sid可标明不用，不能错配行。
6. final测试上的未平滑森林条件mark应与冻结QRF weights重排/去零后相同；gamma=0且按rule p组合，应重现rule-QRF完整CRPS，允许浮点CDF边界选择差异但报告最大误差。

### B：forest_transport_round5，新神经模型

复用只读seq_transport_round4的正确sequence/context、损失和真实异支持mixture几何，新增独立源码与输出；不改任何旧文件。若A尚未完成，可先写模型及用人工参考测试，再消费缓存。

每query参考u来自A：

- beta=.35*tanh(shape_head)，w_i正比u_i*exp(beta·phi2(z_i))；phi2仍是现有有符号位置与绝对幅度响应，不增加未被支持的高阶因子。
- m0=sum(u_i*z_i)，s0²=sum(u_i*(z_i-m0)²)。
- mw=sum(w_i*z_i)，sw²=sum(w_i*(z_i-mw)²)。
- delta=1.5*tanh(location_head)，ell=1.5*tanh(scale_head)。
- mt=m0+s0*delta，st=s0*exp(ell)。
- mark_i=mt+st*(z_i-mw)/sw=a*z_i+b。
- Y=(1-p_rule)*delta_0+p_rule*sum(w_i delta_mark_i)。p_rule仍为0.8local+0.2fit期频率。实际非零概率按完整F计算并记录额外零碰撞。

四个对照即可，预算集中在有用模型：

1. forest_reference_only：仅平滑森林参考，无NN更新；分离gamma平滑自身的收益。
2. forest_transport：完整LSTM位置/尺度/矩归一形状输运，主候选。
3. forest_location_scale_only：beta置0，其余不变。
4. forest_reweight_only：不做affine输运，mark=z。

三种学习版本×五市场×五seed=75个内层选择/最终重拟合组合，max64/min30/patience16、同现有AdamW/正则和normalized macro pinball选择，不挑seed。encoder/头维持2468分配参数；全部h从首轮共同训练。保持源数据、CUT、原点、target、scale与旧实验完全一致。已有uniform-reference seq_transport作为外部结构对照，不重训冒充新证据。

五seed最终支持各自a_s*z+b_s不同，必须合并其概率及0 atom后从同一F算Q与完整CRPS，不能平均参数/CRPS/Q冒充概率mixture。保留每seed、真正mixture、checkpoint和参考cache。对比rule-QRF、原QRF、LSTM-QR和uniform-reference seq_transport；LSTM仅15q，不给其捏造完整CRPS。

## 参考保留的数学意义

当uniform非零支持方差V_uniform>0，u_i>=gamma/N，故Var_u>=gamma*V_uniform。两个phi坐标均在[-3,3]，beta各在[-.35,.35]，能量在[-2.1,2.1]，因此归一后w_i>=exp(-4.2)*u_i，Var_w>=gamma*exp(-4.2)*V_uniform>0。该下界保证形状归一的分母有正支持；实际仍记录数值floor触发，不把退化数据强行称单位方差。

这是参考正则和稳定性设计的解释，不宣称这些通用不等式为首创。是否提高预测分数由reference_only与其他对照检验。

## 已知实现边界

- 森林参数/phi/scaler使用整个允许拟合区间；成熟mask仅限制检索响应，不称每个训练查询都使用当时冻结的全部参数。
- 训练森林leaf partition由拟合区间监督学习；新模型不能把validation/test响应放进参考支持。不要用旧final森林给inner validation制造未来训练标签。
- 0收益是h步端点价格相同，不等于h步内无价格更新，更不等于没有交易。
- 输运后的mark可能恰好为0，固定的是branch mixing p，最终Brier必须从完整预测分布算真实零质量；第四轮此碰撞为0，新版仍需记录。
- 经验条件分布、QRF、位置尺度变换、CRPS均已有方法。贡献应落在条件森林参考、保留尾部的非退化约束与序列驱动的矩归一残差输运这一具体碳价模型，并给出匹配组件增量。

## 可并行的次要实验：期限相似度约束

第四轮pooled森林约64%–75%的质量来自其它h，提示h输入没有充分限制借样。可独立前缀 horizon_kernel_round5，用已有pooled_MSE作为基础，不继续叠加未有效的CDF分裂。

对每query h，重分配每个历史响应权重 w_i' 正比 w_i*exp(-abs(log(h/source_h_i))/tau)。在inner tr拟合一个seed11/80树的相同pooled_MSE，仅用validation在tau=[.25,.5,1,2,infinity]选每市场一个值，不按h或test挑最优；final直接对已冻结的五seed pooled_MSE ensemble weights做相同后处理，不需重训final森林。另可无重训报告规则零概率重平衡版本。保留tau=infinity原始pool作对照，并比较per-h QRF与rule-QRF。全过程源标签仍按自己的t_i+h_i在fit边界前成熟，推断反变换按query的sigma*sqrt(h)。

这检验借样强度与期限条件化，不把后处理称为新通用分布族。主工作仍是上面的森林参考神经输运。

若这版仍未优于强基线，继续改模型：可独立试预先记录的参考偏离正则或有界price-gap/horizon位置残差，但每版独立存储，不静默改旧结果，不把一次未胜出推广为整个ML/DL方向无效。
