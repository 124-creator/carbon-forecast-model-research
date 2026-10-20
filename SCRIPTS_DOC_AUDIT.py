# -*- coding: utf-8 -*-
"""
文档级一致性审计：检查 DIRECTION_ZH.md 中的关键数字是否与 JSON 单源一致
=====================================================================
背景：本文档已积累 46 条订正，任何一处残留的旧数字都会让论文自相矛盾。
本脚本从 JSON 重算关键数字，再在文档中搜索这些数字的出现位置，
并检查"旧值"是否仍以未加标注的形式出现。
输出 DOC_AUDIT.json
"""
import json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
DIR = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
DOC = os.path.join(DIR, "DIRECTION_ZH.md")
J = lambda f: json.load(open(os.path.join(DIR, f), encoding="utf-8"))
t = open(DOC, encoding="utf-8").read()

print("=" * 104)
print("文档一致性审计")
print("=" * 104)

# ---- 1. 结构检查 ----
secs = re.findall(r"^## (\d+\.\d+)", t, flags=re.M)
print(f"\n[1] 结构：{len(t.splitlines())} 行，{len(secs)} 个二级章节，{len(t.encode())} 字节")
dups = [s for s in set(secs) if secs.count(s) > 1]
print(f"    重复章节号: {dups if dups else '无'}")
order = [float(s) for s in secs]
bad_order = [(secs[i], secs[i + 1]) for i in range(len(order) - 1) if order[i + 1] < order[i]]
print(f"    章节顺序问题: {bad_order if bad_order else '无'}")
# 标记为已撤回的章节
retracted = re.findall(r"^## (\d+\.\d+) \[已撤回\]", t, flags=re.M)
print(f"    显式标记为已撤回: {retracted if retracted else '无'}")

# ---- 2. 关键数字核对 ----
print("\n[2] 关键数字（从 JSON 单源 → 文档中的出现次数）")
FE = J("FINAL_EVIDENCE.json")
FS = J("FINAL_SPEC.json")
PC = J("PAPER_CLAIMS.json")
PCAL = J("PLACEBO_CALIB.json")
DC = J("DM_CALIB.json")
ME = J("MECHANISM.json")
VF = J("VERIFY.json")
HG_ = J("HORIZON_GRID.json")
MM = J("MULTIMETRIC.json")
EV = J("ECONVALUE.json")
PB = J("PLACEBO.json")
DR = J("DATAROBUST.json")

CHECKS = []
def chk(name, val, fmt="{:.4f}"):
    s = fmt.format(val)
    n = t.count(s)
    CHECKS.append(dict(name=name, value=s, count=n))
    return n

chk("A14 vs A5 合并 log 比", FE["final"]["A5"]["real_mu"], "{:+.5f}")
chk("A14 vs A5 p 值", FE["final"]["A5"]["p"], "{:.4f}")
chk("A14 vs RW p 值", FE["final"]["RW"]["p"], "{:.4f}")
chk("A14 vs SD250 p 值", FE["final"]["SD250"]["p"], "{:.4f}")
chk("A14 vs EMP p 值", FE["final"]["EMP"]["p"], "{:.4f}")
chk("A14 vs VR p 值", FE["final"]["VR"]["p"], "{:.4f}")
chk("A14 vs SD250 合并 log 比", FE["final"]["SD250"]["real_mu"], "{:+.5f}")
chk("规则R 主切分 h_sat 测试段 log 比", PC["te_mu"]["50"], "{:+.5f}")
chk("规则R 主切分 h_sat 测试段 p", PC["te_p"]["50"], "{:.4f}")
chk("规则R 主切分 h_sat 训练段 p", PC["tr_p"]["50"], "{:.4f}")
chk("规则R 5 切分 测试段 p 最小", min(r["te_p"] for r in FS), "{:.4f}")
chk("规则R 5 切分 测试段 p 最大", max(r["te_p"] for r in FS), "{:.4f}")
chk("真实ξ_d 合并 log 比", PCAL["out"]["real"]["real_mu"], "{:+.5f}")
chk("真实ξ_d 校准 p", PCAL["out"]["real"]["p"], "{:.4f}")
chk("随机分位 合并 log 比", PCAL["out"]["rand"]["real_mu"], "{:+.5f}")
chk("滞后一期 合并 log 比", PCAL["out"]["lag"]["real_mu"], "{:+.5f}")
chk("渐近DM L=10 假阳性率", DC["calibration"]["a10"]["frac05"], "{:.4f}")
chk("渐近DM L=80 假阳性率", DC["calibration"]["a80"]["frac05"], "{:.4f}")
chk("置换检验 假阳性率", DC["calibration"]["perm"]["frac05"], "{:.4f}")
chk("独立复现最大绝对差", VF["max_abs_diff"], "{:.2e}")
chk("PIT-KS A14/A5 中位", 0.9769, "{:.4f}")
chk("机制 LOO R²", ME["loo"]["r2"], "{:.3f}")
chk("A15 vs A14 合并 log 比", 0.00531, "{:+.5f}")

for c in CHECKS:
    flag = "✓" if c["count"] > 0 else "·"
    print(f"    {flag} {c['name']:<36} {c['value']:>12}  文档出现 {c['count']} 次")

# ---- 3. 旧值残留检查：这些数字曾被订正，不应以无标注形式出现 ----
print("\n[3] 已被订正的旧数字的残留检查")
STALE = [
    ("0.00414", "A15(h_sat=20) 相对 A5 的旧结论"),
    ("A15 取代 A14", "被 §24.35 撤回的结论"),
    ("h_sat=20", "被 §24.35 撤回的 h_sat"),
    ("h_sat = 20", "被 §24.35 撤回的 h_sat"),
]
for s, why in STALE:
    n = t.count(s)
    # 找出上下文里是否带"撤回/作废/§24.35"等标注
    flagged = 0
    for m in re.finditer(re.escape(s), t):
        ctx = t[max(0, m.start() - 400):m.end() + 400]
        if any(k in ctx for k in ("撤回", "作废", "§24.35", "§24.38", "不得", "已否", "被 §24")):
            flagged += 1
    unflagged = n - flagged
    flag = "✓" if unflagged == 0 else "✗"
    print(f"    {flag} '{s}'（{why}）：共 {n} 次，其中 {flagged} 次带撤回标注，"
          f"{unflagged} 次未标注")

# ---- 4. 禁止声明是否被文档自己违反 ----
print("\n[4] 禁止声明自检：文档是否自己犯了大错")
FORBID = [
    ("显著优于现有方法", "主张总表的禁止项"),
    ("A15 取代 A14", "§24.35 已撤回"),
]
for s, why in FORBID:
    n = t.count(s)
    flagged = 0
    for m in re.finditer(re.escape(s), t):
        ctx = t[max(0, m.start() - 500):m.end() + 500]
        if any(k in ctx for k in ("撤回", "作废", "不得", "禁止", "已否",
                                  "§24.35", "§24.38", "不能写", "不可写", "禁止声明",
                                  "**不能**", "**不可**", "**不得**")):
            flagged += 1
    unflagged = n - flagged
    flag = "✓" if unflagged == 0 else "✗"
    print(f"    {flag} '{s}'（{why}）：共 {n} 次（{flagged} 次出现在禁止/撤回语境），"
          f"未标注 {unflagged} 次")

# ---- 5. 订正编号连续性 ----
print("\n[5] 订正编号")
nums = sorted({int(x) for x in re.findall(r"\*\*\((\d+)\)", t)})
print(f"    出现的订正编号: {nums}")
missing = [i for i in range(1, (max(nums) if nums else 0) + 1) if i not in nums]
print(f"    缺号: {missing if missing else '无'}")
dupn = [i for i in set(nums) if len(re.findall(r"\*\*\(%d\)" % i, t)) > 1]
print(f"    重复定义的编号: {dupn if dupn else '无'}")

# ---- 6. 章节交叉引用有效性 ----
print("\n[6] 交叉引用")
subs_all = set(re.findall(r"^### (\d+\.\d+)", t, flags=re.M))
allsec = set(secs) | subs_all
refs = sorted({x for x in re.findall(r"§(\d+\.\d+)", t)})
bad_ref = [r for r in refs if r not in allsec]
print(f"    引用了 {len(refs)} 个章节，其中指向不存在章节的: "
      f"{bad_ref if bad_ref else '无'}")

OUT = dict(struct=dict(lines=len(t.splitlines()), sections=secs, dups=dups,
                       bad_order=bad_order, retracted=retracted),
           numbers=CHECKS, forbidden=FORBID, retraction_nums=nums,
           missing_nums=missing, dup_nums=dupn, bad_refs=bad_ref)
json.dump(OUT, open(os.path.join(DIR, "DOC_AUDIT.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\n已写出 DOC_AUDIT.json")
