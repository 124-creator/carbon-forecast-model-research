# Act-Now作者代码实际核读与单变量适用性

此记录只基于本轮真实gh工具输出，不继承同轮五个web失败目标的任何内容。gh search online forecasting返回Anoise/Act-Now，真实读取README、完整models/lade.py与完整utils/buffer.py，以及exp/exp_small.py模型调用上下文。固定commit2d88789c92b4ef1d365cda4ef387db8ef34ea757。未读取其原论文正文，未安装、执行或复现作者完整框架。未确认项目再分发许可，buffer版权头提示依赖rootLICENSE，不推代码。

README明确快流buffer partial/pseudo labels、慢流complete labels，以及Lade统计流/归一化流独立预测。[6] 本轮再次确认不能以“双流+标签分解”首创；作者性能百分数不采信为我们数据预期。

实际lade.py中输入沿最后维求mean/variance；预测mean/std送入decom_model之前detach；统计head以目标统计监督训练，预测head拟合原标签，两个optimizer分别更新。[7] 这属于特定已有实现，不证明其他版本相同。fast/slow缓冲部分仅README，不冒称看过完整在线成熟协议；普通buffer.py只有rehearsal/reservoir代码，不是完成FSB/SSB逻辑审计。

重要单变量门：若输入按小数据runner接口为B×T×1，沿最后维求均值将每个值减自身，归一化流严格为0，std为sqrt(1e-7)；原统计流仍随时间变化，不能说整模型输出恒定或所有训练无梯度。若作者其他路径实际为B×N×T，则需另审，不跨路径泛化。本轮只做这个明确shape下的数值反例，非市场结果。这提示单变量直接移植Lade不能完整测试论文所说的两个信息分支；若改轴则是本项目适配版而不是原实现复现，且不构成创新。

因此本轮不盲跑新的Act-Now广州基线；需明确变量维度、统计流定义以及许可证后才可能用作适配对照。已审核普通GRU/快慢集成仍保留，当前没有达标创新。

## Sources
[6] https://github.com/Anoise/Act-Now/blob/2d88789c92b4ef1d365cda4ef387db8ef34ea757/README.md
[7] https://github.com/Anoise/Act-Now/blob/2d88789c92b4ef1d365cda4ef387db8ef34ea757/models/lade.py
