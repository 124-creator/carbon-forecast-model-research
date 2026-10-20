# 固定煤外生强基线实际结果

没有真实开发增量；停止本次固定共享煤事件核，不扩展调参，不推广为煤信息无效或阴性论文。

|方法|发布延迟|等年等h相对RW MSE|较MLE差值|
|---|---|---:|---:|
|MLE480|价格only|0.949793230120762|0|
|线性煤残差校正|+1自然日|1.113393469581482|+0.163600239460720|
|共享response核|+1自然日|1.124093707803371|+0.174300477682609|
|线性煤残差校正|+2自然日|1.116806863368486|+0.167013633247724|
|共享response核|+2自然日|1.082105882434503|+0.132312652313741|

全部451同年完整h1/5/20 origins：2020为224、2021为227；完整逐年逐h原价MSE及相对RW MSE见RESULTS.json。+1有411有效煤origins、40精确回退MLE；+2有410、41。没有按missing删评分样本。123事件原panel，尾2022-01-04不可进入2021features；碳价格严格950行，最后2021-12-31，不读下一记录。

训练前CONTRACT.md固定两候选、alpha=[.1,10]、最大30自然日有效期、两past-events差分、训练内h20 purge、latest480共同pool、历史发行版本MLE成熟残差；40观察点refresh。发布年切换亦不carry上一年事件差分。保留显式missing及age，missing强制回退。

核可追踪到既有structured_response_search_20261014/design/DESIGN_ZH.md §2.1/2.4：uniform节点质量的固定三角k(a)=max(1-a/12,0)，事件diff乘[k(a+h)-k(a)]。这里只做一个共享系数约束的distributed-lag/generalized ridge估计假设，不是新算法，也不是原神经innovations observer复现。自然日过期与核观察点age分开记录。

真实future-coal扰动并重新选择alpha/拟合：48次；代表459/499/659未来price扰动并实际重拟合：12次；参数和当次预测差0。normal和python -O分别重评分全部5行，score差0、fallback检查通过。先见test缺实现失败，后真实端到端test通过；CSV BOM造成一次索引验证失败，改utf-8-sig后通过，未改数据。

产物全部仅coal_training；读取已审核strong_backbone接口后直接复用其相同calibrate.py真实MLE实现，无需复制骨干、未修改旧骨干。run.py、test_run.py、verify.py、CONTRACT.md、RESULTS.json、TRAINING_LOGS.json、CAUSAL_REFITS.json、COVERAGE.json、RUN_RECEIPT.json、normal/O verification、本地预测及gitignore。无commit/push，无price写入。

局限：未认证历史首版offline snapshot，21旬仍缺；验证证明本实现因果边界，不证明当年可得版本。2020—2021是已用开发集。核2021个别h改善不能抵消2020退化或替换主评分；禁止事后选择年/步长。
