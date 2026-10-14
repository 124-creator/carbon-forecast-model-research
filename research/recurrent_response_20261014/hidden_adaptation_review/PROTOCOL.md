# 最后隐藏状态适配的等价性检查

本轮真实gh取得ADAPT-Z README和adapt_z.py相关核心上下文，固定commitf5b70e1127daeb0e9e07b72bf6f4c8393776bb3b，再写本契约。代码adapter读取feature与historical gradient形成z；在线路径延迟选取旧样本；未完整审查loader与全部base模型，不认证其全部防泄漏。load和主程序有torch.load，本轮未执行，也不下载checkpoint。licenseAPI404，未确认再分发许可，不复制发布作者源码。论文正文及发表身份不以未取得工具内容支持。

数学资格不是预测实验：对本项目最终状态16维、20h线性head，pred=W h+b，若适配只加最终hidden量z，pred'=W(h+z)+b=pred+Wz。适配器可以是任意条件函数z=A(x,g)，恒等式仍成立。固定随机double precision输入和head检查前向、两个独立梯度方向及rank，normal/O重复。没有新GRU市场训练或实际预测，不选随机seed成绩，不以这个特例否定所有深层适配或ADAPT-Z。

若等价，不能把这种最终hidden适配称不同于条件残差校正的模型创新；其输出增量受W列空间约束，可能是有用已有低秩正则但需匹配同函数类残差对照。改变早期recurrent动态不自动等价，具体方案仍需要区别真实直接近邻，不能借这个反例声称已找到新创新。
