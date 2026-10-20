# 阶段k：正式运行故障与代码修正（结果前）

第一次运行进入2018年asof_close开发单元训练和评分后，在`numpy.savez_compressed`保存时失败：`TypeError: numpy.savez_compressed() got multiple values for keyword argument 'rw'`。没有完成任何一个单元的文件导出，也没有任何最终研究汇总。保留原`formal_run/`失败目录。

本次**唯一修改**是保存NPZ时的`**forecast`改为`**{k:v for k,v in forecast.items() if k!='rw'}`：原字典的`rw`仍通过明确的`rw=rw`单独保存。模型公式、超参搜索、数据、验证协议、统计方法和评价方法均没有改变。未因首次计算看到不利数据而调整模型。

源代码SHA256（旧）：`0298ae2f37258e6b71239599639d00ac19744e1e287d87fc67e4ba777c36accd`。
修复后的SHA256：`b4260418c6d7eda2639f294bedf358d5bdb523d19b6694a8b43f5e89ff25d75a`。

现在使用新的输出目录`formal_run_fixed`完整重跑全部2018—2023开发单元和两种信息条件，不覆盖失败记录。禁止将这些已暴露年度声称为新盲测试。