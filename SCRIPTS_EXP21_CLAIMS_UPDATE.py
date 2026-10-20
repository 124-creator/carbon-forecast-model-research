# -*- coding: utf-8 -*-
"""更新论文主张总表：加入文献标准基准的结论与新的禁止声明。"""
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8")
DIR = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
PC = json.load(open(os.path.join(DIR, "PAPER_CLAIMS.json"), encoding="utf-8"))
LB = json.load(open(os.path.join(DIR, "LIT_BENCH.json"), encoding="utf-8"))
FE = json.load(open(os.path.join(DIR, "FINAL_EVIDENCE.json"), encoding="utf-8"))
f = lambda b: FE["final"][b]

C = PC["claims"]
def add(cid, claim, test, number, source, caveat=""):
    C.append(dict(id=cid, claim=claim, test=test, number=number, source=source,
                  caveat=caveat))

add("C17", "相对文献标准统计基准（ARIMA/GARCH/AR/EMA）不显著占优",
    "校准合并检验，30 次打乱（每变体各自建零分布）",
    "ARIMA p = 1.0000、GARCH p = 1.0000、AR p = 1.0000、EMA p = 1.0000",
    "LIT_BENCH.json",
    "真实 CRPS 上 A15 在 5 市场中的 4 个优于全部统计基准；但零分布中心显示"
    "打乱序列上 ARIMA 比 A15 好 26%，故比较无信息量")
add("C18", "GARCH_H（嫁接本文 h^H 标度）与 A15 无法区分",
    "同上", "GARCH_H p = 0.3333", "LIT_BENCH.json",
    "两者共享 h^H，差别只在位置块与形状块")
add("C19", "H 是唯一在多条独立证据链上都稳健的贡献",
    "SEL/EVAL 分离、逐市场分析、安慰剂、机制诊断、GARCH_H 对照五条链",
    f"h^H vs 固定√h：p = {f('SD250')['p']:.4f}；其余各条均不显著",
    "FINAL_EVIDENCE.json, PLACEBO_CALIB.json, MECHANISM.json, LIT_BENCH.json")
add("C20", "形状块是信息敏感组件",
    "打乱序列上 A15 显著差于 ARIMA",
    "零分布中心 -0.261", "LIT_BENCH.json",
    "有信息时更好、无信息时更差，是形状块在利用信息的间接证据")

NEG = PC["must_not_claim"]
NEG.append("不得声称优于文献标准统计基准（ARIMA/GARCH/AR/EMA，校准 p 全部 ≥ 0.333）")
NEG.append("不得声称位置块与形状块有可检测的增量贡献（五条证据链一致不显著）")
NEG.append("不得把 GARCH_H 的结果读成\"GARCH 也行\"或\"本文也行\"——它测的是位置/形状块，"
           "不是 H")

PC["out"] = dict(LIT_BENCH=LB["out"], FINAL_EVIDENCE=FE["final"])
json.dump(PC, open(os.path.join(DIR, "PAPER_CLAIMS.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

print("=" * 100)
print("更新后的论文主张总表")
print("=" * 100)
for c in C:
    print(f"{c['id']:<5}{c['claim'][:48]:<50}{c['number'][:44]}")
print("\n禁止声明清单：")
for x in NEG:
    print("  ✗ " + x)
print(f"\n共 {len(C)} 条主张，{len(NEG)} 条禁止声明")
