# 网页端资料包：数据、权限与复训范围

## 只提供批准前缀

本目录data/guangzhou_approved_prefix_950.csv以及ZIP中data/parsed/figshare_28682348/guangzhou_reported.csv是同一组原样前缀字节：950条、2018-01-03至2021-12-31、29204字节；sha256为1366ca4814dcdb8aa43296eb64768d1336d6f0fc9088648d6d7ccdbe43318070。列名原样保留：date、author_reported_price、source_excel_row、source_date_kind。后两列是追溯元数据，不得当作有预测意义的市场变量。

导出只对原文件执行无缓冲read(29204)，没有请求下一字节/下一条，也没有做完整原文件hash。来源是已经署名发布的作者价格CSV，不是旧模型输出。原始存储/排序变换由既有归档说明；本交接仅截取批准的完整记录前缀，没有改价、改日期、插值或删周末/同价。

署名：Yingying Fan (2025)，Daily Carbon Price Data from Hubei, Guangzhou, and EU Emissions Trading Systems (2018–2024)，DOI 10.6084/m9.figshare.28682348.v1，仓库声明CC BY 4.0。完整署名限制见本目录DATA_ATTRIBUTION.md及仓库data/raw/figshare_28682348/ATTRIBUTION.md。原表sheet误标湖北而广州日期/价格与已持官方GDEA endPrice比对一致的事实保留；这不认证所有报价时点、历史vintage和真实交易日。作者许可不是交易所权利担保。

本包不含2022/2023，也不含其他碳市场、外生序列、旧权重、私有逐点预测或长论文正文。不得为了复算而下载仓库其他完整CSV/XLSX后年后缀。若新机制确需新增资料，先列最小数据需求和合法来源，等待明确数据范围，不把缺失值或模拟数据伪装成真实外生信息。

## 资料包结构与能力边界

ZIP解压后保留research/...与data/...层级，可在独立目录运行批准前缀读取器、合成单测和新的拟合。六个HD Python文件和两个fair基础模块取自固定结果提交26b4f403c566ee44e5af79cdcfc5b83bc21bd27b；它们是本方新实现，不是第三方作者复现。包内evidence/是历史报告/聚合数据，不能当作Pro自己运行的输出。

有可用解释器时，先检查Python/numpy/torch和实际CPU资源。既有环境是Python3.11.15、numpy1.26.4、torch2.3.1+cpu、float64、数学线程1；不假设网页端具备相同环境。依赖表放在research/recurrent_response_20261014/fair_recurrent_tuning_20261016/requirements.lock.txt。缺少运行权限或库时应返回可执行本机代码与依赖说明，不捏造安装/训练成功，不下载大型GPU包或模型权重。

在research/recurrent_response_20261014/hindsight_diagnostic_20261017下，可先执行：

    python -B -m unittest test_hd -v
    python -B -O -m unittest test_hd -v

临时文件应放在你实际获准的scratch目录，并设置TMPDIR。不能将上面命令中的python当作所有平台都已安装/配置的保证。

重新运行旧HD网格不是本次Pro任务的默认内容。确需复训时，只在独立解压目录执行hd_run.py init，然后连续执行batch至完整192次，最后aggregate、hd_verify.py及boundary；不要复制evidence里的PRE_RUN/RUN_COMPLETE到运行目录。TIMING.json是原研究五次历史计时的记录，复制它不等于重新执行了五次计时。第一次init只建立输入、参考和任务清单，不代表192次训练完成。历史独立验分需要未公开预测NPZ；没有它们时应先实际重新训练生成，而不是复制历史VERIFY结果。

本次打包验证只核对成员/CRC/hash、读取前缀、运行合成单测及初始化兼容性；不宣称在发布交接包时重训192次，更没有在网页Pro实际启动任务。真实打包验证输出见PACKET_VALIDATION.json。

## 科学状态不因打包改变

2020/2021仍为已暴露开发区，后验选参不能重命名为独立测试；评分、原价和旧实验结论不重写。下一轮可在此做有限探索，但必须保留所有配置和真实失败。历史训练结果、统计核验、模型机制、新颖性与发表资格分别论证，不让审计替代模型开发，也不让微小调参增益替代创新。
