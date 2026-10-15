# 运行及范围

经工具执行确认：Python3.11.15，NumPy1.26.4，PyTorch2.3.1+cpu；完整pip freeze见requirements.lock.txt。已有隔离解释器位于research/torch_segmented_backbone_20261014/.venv/Scripts/python.exe，禁止修改Hermes运行环境。锁文件记录实际Windows CPU环境，不保证任意平台可直接安装。

在本目录，以该解释器执行：

    python -B -m unittest discover -p 'test_*.py' -v
    python -B -O -m unittest discover -p 'test_*.py' -v
    python -B run_fair.py --timing
    python -B run_fair.py --workers 3
    python -B boundary_audit.py
    python -B verify_fair.py --output <新的回放文件>
    python -B -O verify_fair.py --output <另一个新的回放文件>

这里的python应替换为上述隔离解释器。run_fair.py和boundary_audit.py拒绝覆盖旧结果；复运行必须在单独新目录／副本执行，并保留本轮冻结输出。不在同目录删除local_only后冒充首次运行。

普通公开克隆有源码、协议、已归因作者CSV和汇总，但没有本轮逐点预测／参数，不能直接完成保存结果回放。新训练会创建本地数值包；数值相同、同环境重训、独立实现、第三方复现是不同证据层级。

PRE_RUN中的源摘要来自执行工作树。交付清单以Git index blob为准，不能拿Windows文本归一化后的发布摘要偷偷替换执行摘要。公开清单排除自身摘要，所有数据数组、模型参数和外部原始响应均不入Git。
