# 复核入口与交付范围

使用同仓库research/torch_segmented_backbone_20261014/.venv中的Python3.11.15、NumPy1.26.4、Torch2.3.1+cpu。没有新增安装或预训练权重；CUDA关闭、float64、单线程数学库。依赖见相邻fair_recurrent_tuning_20261016/requirements.lock.txt。

在本目录、选定上述环境后运行：

    python -B -m unittest test_rum -v
    python -B -O -m unittest test_rum -v

有本地保存预测的工作区可执行：

    python -B rum_verify.py --output NEW_VERIFY.json
    python -B -O rum_verify.py --output NEW_VERIFY_O.json

输出使用独占创建，不覆盖原核验。代码不加载pickle、旧checkpoint或第三方模型。独立评分器不导入rum_core/rum_fit/rum_run，只读取原价批准前缀与本轮数值NPZ，allow_pickle=False。

执行源、协议、单测及共同基础代码摘要冻结在PRE_RUN.json。已完成目录禁止再次init、batch或aggregate覆盖。重新训练必须是新目录和新运行记录，不能冒充原始保存结果；原绝对期限不重置。真实边界复训的2次额外拟合记录在BOUNDARY*.json，与32次正式拟合分开。

共同基础仅为相邻fair_core.py、fair_data.py；GRU匹配测试另调用相邻fit_recurrent.py做合成fresh-fit比较，没有读取旧参数或历史预测。run内部只使用最多3个CPU训练进程，不是子agent。

数据仍引用data/parsed/figshare_28682348/guangzhou_reported.csv的已核29204字节前缀。不要做完整文件hash或扩大读取到下一行。每个键的完整H20未来必须早于查询的60步窗口，推理不使用查询未来。

公开范围仅本方实现/测试/协议、聚合训练记录/分数和局限；排除local_only、NPZ逐点预测/参数、输入缓冲、venv、第三方作者源码、长检索响应、论文全文、凭据。代码/训练记录可阅读，单测可运行；因为保存预测不公开，公开包不是下载后直接重放全部已保存预测的完整归档。SOURCES.json只是索引；来源等级和未完成的查新见PROTOCOL，不能用本地引用脚本认证全文访问。

没有子agent、第三方独立代码审查或作者模型复现。SELF_REVIEW表示主助手自审，独立评分表示计算实现分离。最终同步以仓库外REMOTE_READBACK为准，不以本文充当推送成功凭据。
