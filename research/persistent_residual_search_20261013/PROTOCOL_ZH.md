# 冻结协议
仅广州950条前缀，2020—2021同年完整451签发点，原价变化，六个年份/步长相对RW MSE等权。900秒预算，一次候选，不追加配置。
源码核实：旧linear不是动态偏差；逐h每40点刷新，以480个成熟发行残差回归[1,MLE预测,tanh价格水平状态,20日/120日波动比]，列RMS缩放，所有系数含截距零中心岭alpha=10，目标严格<fit。
机制：归一化残差e_h/sqrt(h)含共享、短期变化的发行时点偏差。唯一候选为发行年龄加权的共享标量估计：权重exp(-(fit-1-s)/30)，每h独立成熟s+h<fit，不截齐h20，480发行窗口，零中心20个单位伪观测收缩。观测噪声等权，不能视为独立真实观测，区间采用时间块。
比较：MLE、旧linear精确复现、每h常数、每h EWMA（成熟到达时间权重）、共享成熟到达EWMA、独立发行年龄消融、常规每h随机游走Kalman（Q=.01,R=1，按到达残差更新）以及唯一共享发行年龄候选。全部修正在40点刷新冻结；历史MLE保持原发行参数，不回填。
门槛：候选须优于全部对照及独立消融、两年份均优于旧linear，候选减旧linear的年份内长度40循环移动块bootstrap（2000次，seed=73）95%区间上界<0。否则停止。共享不成立不升级。
动态偏差、Kalman、EWMA、DMA、延迟观测和指数年龄衰减均已有，候选只测试固定估计差异，不声称算法创新。文献依据：2018 Exploring the Exponentially Decaying Merit of an Out-of-Sequence Observation；2020 Weighted Information Filtering, Smoothing, and Out-of-Sequence Measurement Processing (arXiv:2009.02659)。
验证必须显式异常、不依赖assert；普通/-O，逐h未成熟标签扰动、回放重评分，原价未来扰动，结果交由父核独立复核。不得读取2022/边界，不提交推送。
