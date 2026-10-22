# Stage M：持续时间响应与低秩对照

科学状态：**STOP_DEVELOPMENT_GATE_FAILED**。唯一候选RENEWAL没有超过MARKOV、低秩或固定基函数控制。不要把更好的普通控制改名为创新。

先读REPORT_ZH.md，再读PROTOCOL.md、LITERATURE_SCOPE_ZH.md、REMOTE_COMPARISON.json。7份训练/评分/审计源码、22项测试和完整results_remote均已归档。results_remote为完整GitHub重训，local_first_summary为本地首轮分数；下载包另保留完整results_local，两个版本互不覆盖。

## 复现

使用独立CPU环境，安装requirements.txt中版本；PyTorch使用官方CPU安装源。不修改已有项目环境，不下载预训练权重。数据依赖项目已有data/parsed/figshare_28682348/guangzhou_reported.csv；读取器只请求前29204字节/950记录，核对哈希，不读取下一字节。

    python -B -m unittest discover -s research/pro_renewal_response_20261015 -p test_core.py -v
    python -B -O -m unittest discover -s research/pro_renewal_response_20261015 -p test_core.py -v
    python research/pro_renewal_response_20261015/run_all.py --out /path/to/NEW_results --workers 3

输出目录必须不存在，以免覆盖首轮。独立算术和代表性重训：

    python research/pro_renewal_response_20261015/audit.py --out /path/to/NEW_results

本轮workflow已完成从头远端重训并提交849结果文件。不要对已存在的results_remote再次执行固定workflow；常规复现使用上面新的输出目录。GitHub绿色状态不代表机制或严格数值复现通过。

## 差异比较

    python research/pro_renewal_response_20261015/compare_runs.py --local PATH_A --remote PATH_B --out /path/to/comparison.json
    python research/pro_renewal_response_20261015/describe_saved.py --results PATH_A --out /path/to/descriptive.json

两脚本不训练、不修改输入结果、不寻找更优配置。两端并非逐值一致，参见REMOTE_COMPARISON.json；候选主分数和停止结论相同不能抹去其他轨迹差异。

数据为作者发布快照，非实时vintage认证。模型只用价格变化和同历史参考，不用源行号/Excel格式。2020—2021均为反复开发，禁止默认读取新年份。没有新颖性、经济状态识别、确认性显著优势或录用保证。完整后验诊断和来源限制保留，未合并main。
