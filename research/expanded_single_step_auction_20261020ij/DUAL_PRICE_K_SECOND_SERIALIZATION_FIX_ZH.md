# 阶段k二次保存失败回执（正式单元输出之前）

第一次`np.savez_compressed`参数rw重复后，已按冻结回执修复并在新的输出目录重跑。第二次因从未创建`formal_run_fixed/private`目录，发生`FileNotFoundError`，仍是在首个2018年单元的保存层而非训练、评分、时间切分上失败，未完成已输出的评价单元。

再次**只新增两个mkdir**，确保`private/`和`cells/`目录存在；其余实验公式、源数据、信息条件、模型、调参网格不变。修复后run_dual_price.py的SHA256=`c553136948e6ffa137cbdd78013c4fa38c1d1c3f066dc066be7b5a44ab7bde03`（前一SHA=`b4260418c6d7eda2639f294bedf358d5bdb523d19b6694a8b43f5e89ff25d75a`）。旧失败目录保存，不覆盖。新正式目录`formal_run_fixed2`。这两次均是输出代码实现错误，不应声称已完成阶段k拟合评价，不能隐去。