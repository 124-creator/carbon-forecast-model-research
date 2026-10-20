# -*- coding: utf-8 -*-
"""
修正文档：
 1) §24.34 已被 §24.33 的指针段取代，3 处引用失效 —— 改指 §24.36。
 2) 订正 38/39/40 目前只在 §24.37 的索引里以 "→ N" 形式出现，
    §24.35/§24.36 的正文里没有正式定义 —— 在各自章节补上正式定义。
"""
import os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
DOC = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020\DIRECTION_ZH.md"
t = open(DOC, encoding="utf-8").read()

# ---- 1) 修 §24.34 引用 ----
t = t.replace("完整链条跨 5 个切分独立重跑（§24.34）", "完整链条跨 5 个切分独立重跑（§24.36）")
t = t.replace("**校准后的替代证据见 §24.34**", "**校准后的替代证据见 §24.36**")
t = t.replace("> **⚠⚠ A15 取代 A14（§24.33–§24.34）：第一个零 oracle 依赖的模型。**",
              "> **⚠⚠ [已撤回] A15 取代 A14（§24.33）：第一个零 oracle 依赖的模型。**")

# ---- 2) 在 §24.35 / §24.36 正文补上 38/39/40 的正式定义 ----
A1 = "**A15 显著差于 A14。** §24.33 的\"A15 取代 A14\"这个结论是错的，本节正式撤回。"
N1 = ("""**A15 显著差于 A14。** §24.33 的"A15 取代 A14"这个结论是错的，本节正式撤回。

> **(38) 撤回"§24.33 声称 A15 取代 A14"。** 密期限网格显示 $h_{\\text{sat}}=20$ "
> 使形状订正在 $h\\ge20$ 完全失效，**A15 相对 A14 显著更差**"
> （合并 $\\log$ 比 $+0.00531$，校准 $p=0.9917$；差距最大处正是 $h=20$）。""")
assert A1 in t
t = t.replace(A1, N1, 1)

# A2 无需改动（引号是排版引号），跳过

A3 = "**这消除了 A14 的最后一个可攻击点。** A14 的规格可以原样保留，"
N3 = ("""> **(39) §24.33 的选择规则有缺陷。** "训练段校准 p 最小"在 p 并列时随机打破，
> 而并列区间恰是效果最好的区间。已改为**规则 R**（§24.36.2）：
> 在校准 $p\\le0.05$ 的候选中取训练段效应量最大者。

**这消除了 A14 的最后一个可攻击点。** A14 的规格可以原样保留，""")
assert A3 in t
t = t.replace(A3, N3, 1)

A4 = "### 24.36.5 最终规格"
N4 = ("""> **(40) A14 的 $h_{\\text{sat}}=50$ 不是 oracle 读数。**
> 事前固定的规则 R 在切分 0.45 上正好选出 50，
> 在 5 个切分上均给出 5/5 显著的测试段结果，
> 且与测试段 oracle 最优的差距仅 0.43%。
> **§24.27 的披露应改写为"$h_{\\text{sat}}$ 由事前固定的训练段规则选出"。**

### 24.36.5 最终规格""")
assert A4 in t
t = t.replace(A4, N4, 1)

open(DOC, "w", encoding="utf-8", newline="\n").write(t)

# 复查
nums = {}
for m in re.finditer(r"\*\*(?:\((\d+)\)|→ (\d+)（定义于 [^）]+）)\*\*", t):
    n = int(m.group(1) or m.group(2))
    nums.setdefault(n, []).append(t[:m.start()].count("\n") + 1)
dups = {k: v for k, v in nums.items() if len(v) > 1}
subs_all = set(re.findall(r"^### (\d+\.\d+)", t, flags=re.M))
secs = set(re.findall(r"^## (\d+\.\d+)", t, flags=re.M))
refs = sorted({x for x in re.findall(r"§(\d+\.\d+)", t)})
bad = [r for r in refs if r not in (secs | subs_all)]
print(f"重复定义: {dups if dups else '无'}")
print(f"失效引用: {bad if bad else '无'}")
print(f"订正编号定义处: {sorted(nums)}")
print(f"文档 {len(t.splitlines())} 行")
