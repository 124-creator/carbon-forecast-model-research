# 分裂响应实验：锚、权重层级和单seed核对

600森林、0 NN训练、4307共同原点×期限键，全部旧SHA一致。20个A锚的原始seed/mean权重、γ0 ruleQRF Q/CRPS/p、γ.05 fixedref Q/CRPS/p均逐位一致。

NPZ u重建每模型每h固定3query，Q误差0、CRPS误差0、p误差0；独立CDF最大误差6.88e-15。canonical与按rawNZ质量加权的seedF在精确算术等价；float32累积/归一化后的最大L1差6.63e-08。

内选正log比的臂仍完整final：[['CEA', 'rank_score', 0.0029844823561361125], ['CEA', 'mse_z2', 0.0013905451530898969], ['GUANGZHOU', 'mse_abs', 0.01863664672177735], ['GUANGZHOU', 'mse_z2', 0.02148451348634907], ['SHANGHAI', 'mse_abs', 0.0555954429474589], ['SHANGHAI', 'rank_score', 0.018520044492618494], ['SHANGHAI', 'mse_z2', 0.053446056917804095], ['HUBEI', 'mse_abs', 0.0034400831798796425], ['HUBEI', 'rank_score', 0.017817198589684523], ['EU', 'rank_score', 0.008473090710969992]]

逐market/h/seed的显式gain已另存shape_split_round5_gains.csv，A比较为同seed；强基线比较为其冻结canonical。composite仅reference-only，未进原主表。

| arm | seed | 相对同seed A的MQS改善 | CRPS改善 |
|---|---:|---:|---:|
| mse_abs | 11 | -1.0379% | -1.0644% |
| mse_abs | 22 | -1.2716% | -1.2607% |
| mse_abs | 33 | -1.0238% | -1.0339% |
| mse_abs | 44 | -0.8874% | -0.9708% |
| mse_abs | 55 | -0.9757% | -0.9559% |
| rank_score | 11 | -1.1256% | -0.9480% |
| rank_score | 22 | -1.0017% | -0.9570% |
| rank_score | 33 | -0.9778% | -0.8791% |
| rank_score | 44 | -1.0552% | -1.0141% |
| rank_score | 55 | -0.9483% | -0.8420% |
| mse_z2 | 11 | -1.3630% | -1.3262% |
| mse_z2 | 22 | -1.5397% | -1.4917% |
| mse_z2 | 33 | -1.1865% | -1.2981% |
| mse_z2 | 44 | -1.2606% | -1.3517% |
| mse_z2 | 55 | -1.2006% | -1.2045% |

| arm | market | canonical相对A的MQS改善 | CRPS改善 |
|---|---|---:|---:|
| mse_abs | CEA | +1.0158% | +0.7103% |
| mse_abs | GUANGZHOU | -4.1911% | -3.4216% |
| mse_abs | SHANGHAI | +0.7003% | +0.7540% |
| mse_abs | HUBEI | -1.3098% | -1.0536% |
| mse_abs | EU | -1.7670% | -2.5370% |
| rank_score | CEA | +0.4741% | +0.3792% |
| rank_score | GUANGZHOU | -0.0934% | -0.0231% |
| rank_score | SHANGHAI | -0.9631% | -0.9190% |
| rank_score | HUBEI | -3.8081% | -3.8375% |
| rank_score | EU | -0.6932% | -0.3393% |
| mse_z2 | CEA | +0.8953% | +0.5606% |
| mse_z2 | GUANGZHOU | -5.3690% | -4.4657% |
| mse_z2 | SHANGHAI | +0.2133% | +0.5130% |
| mse_z2 | HUBEI | -0.7593% | -1.0532% |
| mse_z2 | EU | -1.8409% | -2.5315% |
