# 四稳定极点响应头：有限试验建议与新颖性边界

日期：2026-10-17。状态：只做数学/近邻审查；未实施该头、未训练，不构成第四个完整候选。

## 严格建议

本轮暂不启动所述18次拟合来寻找模型创新。现有结构是标准对角稳定状态空间解码器的一个小维实例，有限验证即使有利，也只能证明这种归纳偏置在已暴露广州区间有用。主流程已经真实重训 NLinear/DLinear 时，无须为了完成“小型真实验证”而追加一个新颖性尚无差异的实验族。

如果负责人另有明确诊断目的，它可以是合法的结构消融；应只称“稳定指数基响应头”，不创造新算法名，不声称论文方法贡献已成立。

## 精确数学归约

拟议结构为：

    a_t = [GRU(x_t),1] W
    rho_k = 0.99 tanh(eta_k)
    c_(t,h) = sum_k a_(t,k) sum_(j=0)^(h-1) rho_k^j

令 D=diag(rho_1,...,rho_4)，z_(t,0)=a_t，z_(t,j+1)=Dz_(t,j)，则

    c_(t,h) = sum_(j=0)^(h-1) 1^T z_(t,j)
    c_(t,h+1)-c_(t,h) = 1^T D^h a_t。

因此它恰是“GRU生成初态 + 对角稳定线性状态递推 + 累加观测”的解码器。一阶差分响应属于四指数/Vandermonde基；整个20期限输出属于至多4维全局共享子空间。学习eta是学习全局极点，tanh约束离散谱半径；两者均为常见稳定SSM参数化。加原统计参考不改变此等价。

全局rank≤4不是无条件“去方差”结论：参数更少仅缩小假设空间，偏差可能增加；极点聚集时基可能病态并产生幅值抵消。正负极点允许衰减与符号交替，但不会自动带来市场经济解释。参数稳定也不是预测误差稳定或收益保证。

## 最近数学与架构近邻

- **S4D，NeurIPS 2022**：Albert Gu、Ankit Gupta、Karan Goel、Christopher Ré，*On the Parameterization and Initialization of Diagonal State Space Models*。官方全文已读相关理论与参数化节：
  https://proceedings.neurips.cc/paper_files/paper/2022/file/e9a32fade47b906de908431991440f7c-Paper-Conference.pdf
  Eq.(7)将对角SSM核写成Vandermonde基加权和；3.3节讨论稳定谱的有界/单侧参数化。拟议头是此类动力结构的输出解码用途，不等于整套S4D，但不能称新发现的稳定极点学习机制。

- **N-BEATS，ICLR 2020**：神经网络产生系数再投到预测基这一范式已成熟；目前所提是选择指数累加基并使用GRU编码器。作者代码：
  https://github.com/ServiceNow/N-BEATS/blob/master/models/nbeats.py
  https://github.com/ServiceNow/N-BEATS/blob/master/experiments/model.py
  本次官方OpenReview页面遭遇浏览器验证，未冒称从该页读到全文。

- **Koopa，NeurIPS 2023**：Yong Liu等，*Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors*。官方论文页已读：
  https://proceedings.neurips.cc/paper_files/paper/2023/hash/28b3dc0970fa4624a63278a4268de997-Abstract-Conference.html
  其方法已有端到端学习嵌入与线性动力预测思路；不代表拟议简化头与完整Koopa相同，但形成直接方法近邻。

- 负责人发现LLapDiff arXiv:2605.19805。本次独立打开abs链接返回DisabledError，故本审查不依据其全文，也不把负责人提供的摘要转述冒称本人读过全文。仅上述已发表近邻已足以限制新颖性主张。

## 为什么18次的机制归因尚未闭合

必须区分四臂：

1. 原自由稠密20维头；
2. 自由rank4因子头；
3. 相同初始四极点且固定的基；
4. 学习四极点的响应头。

1对2回答输出低秩是否有效；2对3回答特定指数形状是否有效；3对4回答学习极点是否有效。如果18次只包含3变体×2cuts×3seeds，且稠密头没有同种子/同初始化/同训练口径的可复用对照，就不能同时回答这三层问题。历史c003仅单种子成绩不能补齐其他两个种子的控制。

即使四臂齐全并显示4最好，也只是此固定结构有开发区证据。要形成方法贡献，还需一个相对稳定SSM/已知基展开模型明确而具体的新增学习原则；当前没有。NLinear失效本身也不能证明GRU的自由head就是主要病因。

