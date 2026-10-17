# 独立数学与实现审查

本目录保存自有数学分析、合成脚本、原始收据的精确副本，以及适配新研究目录结构的入口。没有市场逐点价格、已训练市场权重或未获得再分发许可的作者源码。

## 读哪些文件

- DSOF_REVISION_REVIEW_ZH.md：当前教师修订目标、随机游走双计反例、持续漂移、标准条件回归等价性、教师—学生自举及明确否证条件。
- RRC_CORE_AUDIT_ZH.md：新实现的因果、尺度、缓存、ridge 事件权重与梯度前审；对应简要收据 RRC_CORE_AUDIT_SUMMARY.json，完整收据 RRC_CORE_SYNTHETIC_AUDIT.json。
- RRC_RUNNER_STATIC_AUDIT.json：正式 D1 协议与调度源码的静态检查，包括准确配置数、仅开发路径和合作式超时的边界。
- NEXT_DESIGN_AUDIT_ZH.md：旧 BCM 的公开开发诊断，以及已停止作为独立创新的事件秩一分析。它不能改变旧确认结论。
- MATHEMATICAL_AUDIT_SOURCE_MANIFEST.json：从审查工作目录复制的 12 个原文件的字节数和 SHA256。

## 可移植复跑

保留原脚本原样，使旧收据中的脚本 SHA256 仍可核验。它们原有的绝对工作目录假设由 run_audits.py 在调用前覆盖。**移动目录后请用此入口，不要直接假定原脚本的默认研究路径仍正确。**

在本研究目录下运行：

~~~bash
python audits/run_audits.py --case events --output-dir /tmp/event_audit_rerun
python audits/run_audits.py --case targets --output-dir /tmp/target_audit_rerun
python audits/run_audits.py --case core --output-dir /tmp/core_audit_rerun
python audits/audit_rrc_runner.py --output-dir /tmp/runner_audit_rerun
~~~

events 和 targets 只需要 Python 标准库及 NumPy；core 还需 PyTorch。原合成运行采用 NumPy 2.5.4、PyTorch 2.3.1+cpu。入口不安装依赖、不访问网络、不加载市场文件；core 会训练小型合成模型来核对执行等价性。请使用空输出目录，避免覆盖原始收据。

旧 BCM 日志诊断还需要前一研究的公开收据，不需要其市场预测或权重：

~~~bash
python audits/run_audits.py --case logs \
  --previous-study ../cross_market_innovation_20261017 \
  --output-dir /tmp/bcm_log_audit_rerun
~~~

前一研究应已有 d1_receipts、d2_receipts、D1_SUMMARY.json、DEVELOPMENT_AGGREGATE.json；如果公开包将收据装在归档中，应先按原相对路径解压。仅复制本目录不足以重做日志诊断，但足以读取已经保存的诊断结果。其数据依赖是旧公开收据，不是新增确认数据。

所有重跑输出记录实际代码 SHA256 和相应环境版本；新生成的时间戳自然不同。数学与合成通过不等于市场有效性通过；实际评分、固定选择规则、推进门槛及任何新确认须阅读各自独立审查。
