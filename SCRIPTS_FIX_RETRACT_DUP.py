# -*- coding: utf-8 -*-
"""
修正文档：订正编号 33–40 在各自章节里已定义，§24.37 的汇总清单重复定义了它们。
把汇总清单改成**引用索引**（"→ 33（定义于 §24.32.1）"），
使每个编号在全文档中**只被定义一次**。
"""
import os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
DOC = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020\DIRECTION_ZH.md"
t = open(DOC, encoding="utf-8").read()

# §24.37 的汇总清单：把 "(33) ... **" 这种"定义式"改成"→ 33（定义于 §24.32.1）... "
i0 = t.index("## 24.37 本轮（§24.32–§24.36）的完整订正清单")
i1 = t.index("## 24.38 安慰剂结论的校准重做")
sec = t[i0:i1]

# 找到每条 (N) 所在的段落，映射到其定义章节
DEF = {
    33: "§24.32.1", 34: "§24.32.6", 35: "§24.32.5", 36: "§24.32.5",
    37: "§24.32.4", 38: "§24.35.2", 39: "§24.36.1", 40: "§24.36.4",
}
def repl(m):
    n = int(m.group(1))
    body = m.group(2)
    if n in DEF:
        return f"**→ {n}（定义于 {DEF[n]}）**{body}"
    return m.group(0)

new_sec = re.sub(r"\*\*\((\d+)\)([^\n]*)", repl, sec)
t = t[:i0] + new_sec + t[i1:]
open(DOC, "w", encoding="utf-8", newline="\n").write(t)

# 复查
nums = {}
for m in re.finditer(r"\*\*(?:\((\d+)\)|→ (\d+)（定义于 [^）]+）)\*\*", t):
    n = int(m.group(1) or m.group(2))
    nums.setdefault(n, []).append(t[:m.start()].count("\n") + 1)
dups = {k: v for k, v in nums.items() if len(v) > 1}
print(f"定义处出现多次的编号: {dups if dups else '无'}")
print(f"文档 {len(t.splitlines())} 行")
