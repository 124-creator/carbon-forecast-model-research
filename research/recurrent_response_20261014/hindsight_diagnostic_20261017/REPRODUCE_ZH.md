# 复算、重新训练及文件范围

本次研究为已暴露广州2020/2021的后验配置诊断。请先阅读PROTOCOL.md和REPORT_ZH.md，不把oracle当作可部署选择器。

## 本机已有结果复算

实验目录：research/recurrent_response_20261014/hindsight_diagnostic_20261017。
已实际使用的解释器为research/torch_segmented_backbone_20261014/.venv/Scripts/python.exe。环境见PRE_RUN.json：Python 3.11.15、numpy 1.26.4、torch 2.3.1+cpu，float64，单数学线程；依赖锁定表在相邻fair_recurrent_tuning_20261016/requirements.lock.txt。不要修改Hermes环境或安装不必要GPU包。

在该目录执行以下命令，<PYTHON>表示上述解释器的绝对路径：

    <PYTHON> -B -m unittest test_hd -v
    <PYTHON> -B -O -m unittest test_hd -v
    <PYTHON> -B hd_verify.py --output 一个尚不存在的核验文件名.json
    <PYTHON> -B -O hd_verify.py --output 另一个尚不存在的核验文件名.json

验分器独立构建标签和尺度，不导入训练器或特征生成器。它只读取广州CSV的29204字节批准前缀；不请求第951条记录，不做全文件hash。输出文件采用排他创建，不能用已有的VERIFY_NORMAL.json覆盖历史核验。9项单测使用合成数据检查新拟合等行为，不代表额外192次市场复训。

## 独立目录重新训练

不要在本次完整结果目录重新运行init/batch/aggregate/boundary：这些操作有意拒绝覆盖。另建完整独立研究根目录，保留以下相对层级：

    data/parsed/figshare_28682348/guangzhou_reported.csv
    research/recurrent_response_20261014/fair_recurrent_tuning_20261016/fair_core.py
    research/recurrent_response_20261014/fair_recurrent_tuning_20261016/fair_data.py
    research/recurrent_response_20261014/hindsight_diagnostic_20261017/hd_core.py
    research/recurrent_response_20261014/hindsight_diagnostic_20261017/hd_fit.py
    research/recurrent_response_20261014/hindsight_diagnostic_20261017/hd_run.py
    research/recurrent_response_20261014/hindsight_diagnostic_20261017/hd_analysis.py
    research/recurrent_response_20261014/hindsight_diagnostic_20261017/hd_verify.py
    research/recurrent_response_20261014/hindsight_diagnostic_20261017/test_hd.py
    research/recurrent_response_20261014/hindsight_diagnostic_20261017/PROTOCOL.md
    research/recurrent_response_20261014/hindsight_diagnostic_20261017/TIMING.json

数据可只复制已经批准的29204字节完整前缀，必须保持原字节/列名/950条记录；不需要读取或复制后续年份。TIMING.json记录本轮已经执行的5次历史计时，只能作为固定预算的历史依据，不得把复制这个文件说成复训时又执行了5次计时。不要复制本轮PRE_RUN.json、RUN_COMPLETE.json、BATCH_*、BOUNDARY*或local_only；新运行会创建自己的记录。

保持原源码字节（本目录.gitattributes声明LF），在新实验目录依次执行：

    <PYTHON> -B hd_run.py init
    <PYTHON> -B hd_run.py batch

每次batch最多8次拟合，最多2个CPU工作进程。192/8=24批；从init开始的2700秒截止不重置，必须连续安排。每批末输出实际INVENTORY，不要在超时后修改期限继续原运行。完整后依次执行：

    <PYTHON> -B hd_run.py aggregate
    <PYTHON> -B hd_verify.py --output VERIFY_NORMAL.json
    <PYTHON> -B -O hd_verify.py --output VERIFY_OPTIMIZED.json
    <PYTHON> -B hd_run.py boundary

这是192次正式加3次代表性边界拟合；复制的历史计时不算新拟合。不同硬件/库可能有数值差异；本次结果不代表第三方已经复现。

## 公开结果对应关系

PRE_RUN.json保存96组配置、192个任务、源hash、环境、批准数据前缀和预算；BATCH_*保存每批的启动与完成。RUN_COMPLETE.json只在全部配对保存后生成。

RESULTS.json/ALL_CONFIGS.csv含96个训练配置加RW/stat两对照，共98行结果，六格风险与前期验证风险均保留。TRAINING_SUMMARY.json含192个拟合摘要，没有完整训练轨迹。FACTOR_EFFECTS.json只比较48组交叉中除指定因素外完全相同的配置。

POSTHOC_DIAGNOSTICS.json在结果出现后计算。其初始化/上限计数来自TRAINING_SUMMARY.json；各输入组最佳值来自96个训练配置；年度排名只在96个训练配置内比较，Spearman为验证分数与下一年分数的平均并列秩的Pearson相关。它没有参与训练、冻结配置或主评分。

PUBLICATION_MANIFEST.json的hash/字节数以Git index blob为准，排除自身以避免自引用；旧交付的manifest是历史记录，不随本轮README更新而重写。远端回读凭据存仓库外，提交成功和实际远端回读分开。

公开包没有local_only中的预测NPZ、输入缓冲及完整轨迹，因此不能仅凭公开聚合表重复本机的原始逐点验分；需要重新训练生成。没有发布旧实验、凭据、第三方源码或论文全文。
