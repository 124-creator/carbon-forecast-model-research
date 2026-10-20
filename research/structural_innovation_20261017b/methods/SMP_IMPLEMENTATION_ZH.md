# SMP 实现合同与范围

2026-10-17。本文件说明本方有限状态 signed 均值传播实现；没有市场实验结果。Markov reward 的均值递推是标准数学，本文件不把它称新理论。

## 接口与单位

```python
net = SMPModel(input_dim=C, hidden_size=32, horizon=20, variant='full').double()
initialization_receipt = net.initialize_head(q0, mean_abs_nonzero)
m, aux = net(x, observed_state, return_aux=True)
auxiliary = hurdle_nll(aux, next_delta_over_s, event=event_from_raw_price)
```

`x` 为 `[B,20,C]`（实现接受任意正长度）；`observed_state` 为整数 `[B]`；`m` 为 `[B,20]`，单位是 $\Delta p/s_t$，没有 $\sqrt h$。恢复原价、固定训练支持的风险分母、事件阈值和信息边界全部由 runner 管理。`aux` 至少有 `q[B,9,3]`、`b[B,9,2]`、`q_current[B,3]`、`b_current[B,2]`，另返回稳定的 `log_q` 与 `log_q_current`。

GRU 是单层、隐藏维32、无 dropout，离线可训练。每次发行预测时只编码当前真实历史一次；接下来的20步状态传播固定这一上下文，不向 GRU 注入模拟或真实未来信息。这不是冻结 GRU 参数。

## 状态和递推

状态为 $(s+1)\times3+\min(a,2)$，$s\in\{-1,0,+1\}$ 是上一非零净变化符号，$a$ 是连续零净变化观测数。**符号必须由整个可见历史前缀继承**，不能只在最近20步中查找后将未找到的旧符号重置为0。全历史从未非零时才使用符号0。缺失日期不自动计成零变化观测。

事件顺序固定为 `zero, negative, positive`，幅度顺序是 `negative, positive`。无变化的下一状态为 `[1,2,2,4,5,5,7,8,8]`；负变化到状态0，正变化到状态6。

$$
F_0(z)=0,\qquad
F_{h+1}(z)=\sum_{e\in\{0,-,+\}}q_e(z)
\left[r_e(z)+F_h\bigl(n_e(z)\bigr)\right],
\qquad r(z)=(0,-b_-(z),b_+(z)).
$$

输出为 $F_h$ 在实际起始状态的值。`propagate_mean(q,b,observed_state,horizon=20)` 独立提供这一函数，不重新归一化输入概率，也不改变梯度。

## 真正的参数绑定

| variant | q 自由组 | b 自由组 | 约束 |
|---|---:|---:|---|
| full | 9 | 9 | 每源状态一组 |
| no_clock | 3 | 3 | 每种上一符号一组，沿年龄绑定 |
| no_sign | 3 | 3 | 每种年龄一组，沿上一符号绑定 |
| pooled_mark | 9 | 1 | 仅幅度沿全部源状态绑定 |
| iid_state | 1 | 1 | 概率与幅度均不随未来源状态改变 |
| one_step | 9 | 9 | 网络与 full 完全相同；runner 只用 h1 主损失训练 |

各变体收到相同 `x` 和当前状态。先从自由组产生输出再索引扩为9行，不创建多余可训练副本再平均。`no_clock` / `no_sign` 仍可从固定当前上下文知道当前历史；约束的是未来状态反馈。`iid_state` 的未来均值退化为同一当前条件漂移的线性累计，是重要的简单结构控制。

GRU 参数数为 $96C+3264$；full/one_step、no_clock/no_sign、pooled_mark、iid_state 的 head 参数分别为1485、495、957、165。C=7仅为本方合成检查示例，总数分别5421、4431、4893、4101。应使用最终真实C另行计数。

9状态合同包含符号0且年龄0/1的状态3/4；在正常已观察20步的发行起点中，这两状态及其后续路径通常不可达。因此上述为**名义总参数数**，不能声称所有head参数都在真实训练中得到非零梯度。需要由实际支持报告可达状态与覆盖。代码没有擅自删除合同规定的状态。

## 初始化与辅助 NLL

`initialize_head` 将两个 head 的权重设0，不改 GRU；$q_0$ 剪裁到 `[.001,.95]`，$q_-=q_+=(1-q_0)/2$，所有状态共用同一初始幅度。因此所有20期均值严格等于0，即 RW。幅度为 `softplus(raw_b)+1e-4`。若请求幅度小于等于数值下界，用 `1e-4+1e-8`，返回收据明确记载应用值，避免无声修改。初始head权重为0时，第一次反向传播的encoder梯度为0是预期初始化性质。

一阶 signed exponential hurdle NLL 为

$$
\ell(\delta,e)=
\begin{cases}
-\log q_0,&e=0,\\
-\log q_{\operatorname{sign}(\delta)}+\log b_{\operatorname{sign}(\delta)}+|\delta|/b_{\operatorname{sign}(\delta)},&e=1.
\end{cases}
$$

这里 $\delta=\Delta p/s_t$。`event` 必须优先由 runner 在原价上使用固定 $10^{-10}$ 阈值决定。若不传，函数只按归一化值是否精确非零推断，**不会**把原价阈值错误施加到归一化值。显式 event 为真而归一化变化恰为0时抛错，因为符号未定义。`reduction` 支持 `none/mean/sum`。

对任一固定当前历史，NLL 的 $s_t$ Jacobian 与模型参数无关；若报告跨市场原价密度的数值，应在 runner 明确恢复相应 $\log s_t$ 常数。用于学习的这个常数不改变模型梯度。

## 本方检查与独立审查分工

`source/check_smp_implementation.py` 只使用合成输入，核对6变体形状、严格RW初始化、概率单纯形、NLL有限反向梯度，未做任何优化器更新。收据为 `audits/SMP_IMPLEMENTATION_CHECK.json`。没有市场或确认文件读取，也没有作者模型执行。

路径枚举、矩阵均值独立实现、有限差分以及状态绑定数学由独立审计员另行完成；本方 smoke check 不冒称这些独立审计已经通过。查新与已知等价的完整说明在本轮 `scratch/new_research_20261017_phase2/literature_renewal/RENEWAL_REVIEW_ZH.md`，其中先前CF候选未被采用，SMP仍以点MSE作为主指标。
