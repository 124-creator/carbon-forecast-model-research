# 数据来源与使用边界

本地副本来自Chen-zhongxiu/Carbon-price-data，固定提交6e3ea533832f82fd1c8b4859bb2af8c33bd19a31，GCP.csv。源站README称其服务于Modelling carbon price dynamics with jump-diffusion and higher-order hidden Markov switching: Theory, filtering, and forecasting。

数据是作者发布的数值快照，不是本轮重新从交易所下载的官方认证行情。根目录未见明确数据许可证；本交付不授予或声称具备商业利用/公开再分发许可，不以研究代码许可覆盖原始数据。保留副本用于用户请求的分析与验证，任何后续对外发布须单独处理数据来源条件。

原文件价格含义、计价单位、筛选规则及历史首次可得时刻仍有缺口。不得标成已经认证的收盘价，或把观测步擅自叫正式交易日。下载渠道失败日志及源文件哈希在metadata中，不能把Git哈希一致升级为市场含义认证。

来源：https://github.com/Chen-zhongxiu/Carbon-price-data/tree/6e3ea533832f82fd1c8b4859bb2af8c33bd19a31
