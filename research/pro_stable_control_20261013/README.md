# 低维稳定估计与单一学习规则：Stage D

使用 `research/pro_data_packet_20261013` 的既有三份CSV，只在广州2020—2021调参区进行本轮市场评分。2022验证和2023历史段不自动打开。

先执行低维合成控制，再检验一条基于剖面似然集合的联合预测投影规则。候选的通用最小范数凸包优化与Magging等价；本轮最优阈值为0，与普通低维最大似然预测逐值相同，未获得独立创新增量，停止在调参区。

运行：
```bash
python -m pip install -r research/pro_stable_control_20261013/requirements.txt
python research/pro_stable_control_20261013/run_all.py --root . --out /tmp/stageD-replay
```
输出目录必须不存在。完整配置、预测数组、测试日志和阶段状态会写入该目录。GitHub Actions提供独立远端重放与私有结果artifact。

三份价格来自仓库注明的Yingying Fan (2025), DOI 10.6084/m9.figshare.28682348.v1, CC BY 4.0。原始价格不修改；source_excel_row和source_date_kind只用于溯源，绝不作为预测特征。当前工作不认证交易所历史定价规则或实时版本。

最终报告和重放核验随后同分支归档。合成控制通过不等于方法新颖性或市场预测验证通过。
