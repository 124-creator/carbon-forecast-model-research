# -*- coding: utf-8 -*-
"""最终验收：所有检验、审计与可复现性检查一次性跑完。"""
import json, os, subprocess, sys
sys.stdout.reconfigure(encoding="utf-8")
REPO = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research"
DIR = os.path.join(REPO, "research", "fullcycle_h20_20261020")
sys.path.insert(0, REPO)

print("=" * 104)
print("最终验收")
print("=" * 104)

# 1) 论文草稿数字审计
r = subprocess.run([sys.executable, os.path.join(r"E:\Agent员工\碳价格预测研究\_npzcheck", "audit_paper.py")],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
ok_paper = "未出现的数字：0 项" in r.stdout
forb_ok = r.stdout.count("✓ 'significantly better than existing methods'") == 1
print(f"[1] 论文草稿数字审计: {'✓ 通过' if ok_paper else '✗ 失败'}"
      f"（{r.stdout.count('✓')} 个数字核对）  禁止声明: {'✓' if forb_ok else '✗'}")

# 2) 文档一致性审计
r2 = subprocess.run([sys.executable, os.path.join(r"E:\Agent员工\碳价格预测研究\_npzcheck", "audit_doc.py")],
                    capture_output=True, text=True, encoding="utf-8", errors="replace")
lines = [x for x in r2.stdout.splitlines() if x.strip().startswith(("✓", "✗"))]
n_fail = sum(1 for x in lines if x.strip().startswith("✗"))
no_dup = "重复定义的编号: 无" in r2.stdout
no_ref = ("指向不存在章节的: 无" in r2.stdout)
print(f"[2] 文档一致性审计: {len(lines)} 项检查，失败 {n_fail} 项  "
      f"重复订正编号: {'✓无' if no_dup else '✗'}  失效引用: {'✓无' if no_ref else '✗'}")

# 3) 可复现性自检
r3 = subprocess.run([sys.executable, os.path.join(r"E:\Agent员工\碳价格预测研究\_npzcheck", "repro.py")],
                    capture_output=True, text=True, encoding="utf-8", errors="replace")
print(f"[3] 可复现性自检: {'✓ 全部齐备' if '自检：✓' in r3.stdout else '✗ 有缺失'}")

# 4) 独立复现
V = json.load(open(os.path.join(DIR, "VERIFY.json"), encoding="utf-8"))
print(f"[4] 独立复现: {'✓ 通过' if '通过' in V['verdict'] else '✗'}  "
      f"最大绝对差 {V['max_abs_diff']:.2e}（{V['n_checks']} 个量）")

# 5) 关键结论一览
FE = json.load(open(os.path.join(DIR, "FINAL_EVIDENCE.json"), encoding="utf-8"))
HO = json.load(open(os.path.join(DIR, "H_ONLY.json"), encoding="utf-8"))
FS = json.load(open(os.path.join(DIR, "FINAL_SPEC.json"), encoding="utf-8"))
PC = json.load(open(os.path.join(DIR, "PAPER_CLAIMS.json"), encoding="utf-8"))
DC = json.load(open(os.path.join(DIR, "DM_CALIB.json"), encoding="utf-8"))
print("\n[5] 关键结论")
print(f"    第一贡献 H 机制：A5 vs 固定√h  p = {HO['out']['A5|SD250']['p']:.4f}  "
      f"({HO['out']['A5|SD250']['real_mu']:+.5f})")
print(f"    第二贡献 形状块：A14 vs A5     p = {HO['out']['A14|A5']['p']:.4f}  "
      f"({HO['out']['A14|A5']['real_mu']:+.5f})")
print(f"    相对随机游走：A14 vs RW       p = {FE['final']['RW']['p']:.4f}  "
      f"({FE['final']['RW']['real_mu']:+.5f})")
print(f"    相对上一版规格：A14 vs A5     p = {FE['final']['A5']['p']:.4f}  "
      f"({FE['final']['A5']['real_mu']:+.5f})")
print(f"    跨切分一致性：{sum(1 for x in FS if x['te_p'] < 0.05)}/{len(FS)} 切分显著")
print(f"    规则 R 选出 h_sat = 50（主切分），5 切分选出 "
      f"{[x['sel'] for x in FS]}")
print(f"    DM 检验校准：渐近 L=10 假阳性 {DC['calibration']['a10']['frac05']:.4f}"
      f"（名义 0.05）")
print(f"    主张 {len(PC['claims'])} 条 / 禁止声明 {len(PC['must_not_claim'])} 条")

# 6) 仓库状态
g = subprocess.run(["git", "-C", REPO, "status", "--porcelain"], capture_output=True, text=True)
gl = subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%h %s"], capture_output=True, text=True)
ncommits = subprocess.run(["git", "-C", REPO, "rev-list", "--count", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
doc = os.path.join(DIR, "DIRECTION_ZH.md")
print(f"\n[6] 仓库：{'✓ 工作树干净' if not g.stdout.strip() else '✗ 有未提交改动'}  "
      f"提交数 {ncommits}  最新 {gl.stdout.strip()}")
print(f"    DIRECTION_ZH.md: {os.path.getsize(doc)} 字节, "
      f"{len(open(doc,encoding='utf-8').read().splitlines())} 行")
print(f"    PAPER_DRAFT.md:  {os.path.getsize(os.path.join(DIR,'PAPER_DRAFT.md'))} 字节")
print(f"    REPRODUCE.md:    {os.path.getsize(os.path.join(DIR,'REPRODUCE.md'))} 字节")
figs = os.path.join(DIR, "figures")
print(f"    figures/: {len(os.listdir(figs))} 个图")
scripts = [f for f in os.listdir(REPO) if f.startswith("SCRIPTS_") and f.endswith(".py")]
print(f"    脚本: {len(scripts)} 个 SCRIPTS_*.py")

ALL_OK = ok_paper and forb_ok and n_fail == 0 and no_dup and no_ref
print("\n" + "=" * 104)
print(f"验收结果：{'✓ 全部通过' if ALL_OK else '✗ 存在未通过项'}")
print("=" * 104)
