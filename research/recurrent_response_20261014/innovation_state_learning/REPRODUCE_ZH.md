# 复运行与证据范围

本目录保留首轮协议和结果，runner故意拒绝覆盖PRE_RUN.json或local_only。需要重新训练时，在同一父目录创建全新兄弟目录，复制本目录全部.py、PROTOCOL.md、.gitignore，但不复制JSON收据、旧结果或local_only。这样ROOT解析仍指向同一数据仓库，首轮记录也不会被覆盖。

使用独立环境中与ENVIRONMENT.json一致的NumPy/Torch版本，不使用Hermes自己的Python环境。先在新目录运行：

    python -B -m unittest discover -p 'test_*.py' -v
    python -B -O -m unittest discover -p 'test_*.py' -v
    python -B run_innovation.py --workers 2

完整训练结束后，分别运行下列命令，把stdout保存在不同的新核验文件中：

    python -B verify_innovation.py
    python -B -O verify_innovation.py

公开档案不含逐点预测、参数、作者代码或复制的价格前缀，因此仅凭公开的汇总不能再次验分；必须在本地重新训练得到local_only，或者使用本机首次训练保存的local_only。数据读取只请求许可表前950条及其表头，验证截至2021-12-31的既有前缀身份，不请求下一条。

本轮实际在预算约1201.47秒时停止，149个块已完成保存，但只有11条完整轨迹，LSTM两条轨迹未完成，因此没有RESULTS.json，也没有执行后续6次真实价格边界复训。不能把本轮称作完整156-fit实验。部分验分使用：

    python -B -c "import json; from verify_innovation import replay_incomplete; print(json.dumps(replay_incomplete()))"

PARTIAL_REPLAY_NORMAL/OPTIMIZED只评分完整轨迹，永远返回acceptance_passed=false；候选已经败给完整完成的普通GRU对照，因此足够违反原门槛的必要条件，但不能认证完整实验通过。不要补齐LSTM后改写本次中止记录，也不要把缺失预测填成RW。

训练与验分是不同证据层。verify_innovation独立读取原价、构造目标及重算分数，不导入模型、trainer或runner，但不是另一名研究者的独立实现重训。FINAL_TESTS中的训练边界检查使用合成价格；本轮没有CAUSAL.json，不能声称已经做过真实价格完整边界复训。

全文查新没有通过。AUTHOR_SOURCE_RECEIPTS只证明取得哪些作者源码；SEARCH_RECEIPT只是实际检索元数据。SEARCH_ACTUAL的长片段及逐点数值保留本地，不再分发。数学边界另见MATH_BOUNDARIES_ZH，不把固定decoder、Kalman式更新或加权范数不等式独立称为创新。
