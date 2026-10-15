# 执行与核验

已核验的隔离解释器：research/torch_segmented_backbone_20261014/.venv/Scripts/python.exe；Python3.11.15、NumPy1.26.4、PyTorch2.3.1+cpu。不要在Hermes自身环境安装或升级训练依赖。

在本目录执行：

    <该隔离Python> -B -m unittest discover -p 'test_*.py' -v
    <该隔离Python> -B -O -m unittest discover -p 'test_*.py' -v
    <该隔离Python> -B verify_events.py --output <新的评分JSON>
    <该隔离Python> -B -O verify_events.py --output <另一个新的评分JSON>

评分需要本地未公开的local_only输入和预测。单元测试含清楚命名的合成夹具，不作为市场结果。不要重新运行run_events.py覆盖已结束的不完整实验；本目录原中止预算不延长。
