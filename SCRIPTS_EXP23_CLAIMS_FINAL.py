# -*- coding: utf-8 -*-
"""更新主张总表：加入分离检验的结论，修正 C19。"""
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8")
DIR = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
PC = json.load(open(os.path.join(DIR, "PAPER_CLAIMS.json"), encoding="utf-8"))
HO = json.load(open(os.path.join(DIR, "H_ONLY.json"), encoding="utf-8"))
O = HO["out"]
C = PC["claims"]

for c in C:
    if c["id"] == "C19":
        c["claim"] = "H 是第一贡献，形状块是第二贡献，位置块不是贡献"
        c["test"] = "三成分分离检验（位置与形状固定只改标度；H 与位置固定只改形状）"
        c["number"] = (f"H 机制 p={O['A5|SD250']['p']:.4f}（{O['A5|SD250']['real_mu']:+.5f}）；"
                       f"形状块 p={O['A14|A5']['p']:.4f}（{O['A14|A5']['real_mu']:+.5f}）")
        c["source"] = "H_ONLY.json"
        c["caveat"] = "位置块的增量不可检测，不作为贡献"
    if c["id"] == "C18":
        c["claim"] = "GARCH_H 不显著于 A14 是因为它借走了 H，不是 H 无价值"
        c["test"] = "GARCH_H 与 A14 共享 h^H，差值中 H 的优势抵消"
        c["number"] = f"GARCH_H p={0.3333:.4f}；去掉 H 后 A5 vs SD250 p={O['A5|SD250']['p']:.4f}"
        c["source"] = "LIT_BENCH.json, H_ONLY.json"
        c["caveat"] = "这是对 H 机制的正面论证：GARCH 加本文标度即可追平本文模型"

def add(cid, claim, test, number, source, caveat=""):
    C.append(dict(id=cid, claim=claim, test=test, number=number, source=source, caveat=caveat))

add("C21", "H 机制在三个市场单独就达到 p=0.000",
    "逐市场 A5 vs 固定 √h 标度（校准）",
    "广州/上海/湖北 p=0.000；CEA 0.358；EU 0.067", "H_ONLY.json")
add("C22", "H 机制随期限单调增强", "16 期限密网格 + A14/EMP 剖面",
    "A14/EMP 从 h=5 的 1.0017 单调降到 h=200 的 0.5803", "HORIZON_GRID.json")
add("C23", "H 不可用其他方式替代", "跨市场迁移 + 训练段重估 + 强制 H=0.5",
    "LOO 迁移 1.0142；训练段 H 1.0008；H=0.5 差 6.3%", "H_TRANSFER.json")

NEG = PC["must_not_claim"]
NEG.append("不得把位置块作为贡献陈述（增量不可检测）")
NEG.append("不得把\"GARCH_H 不显著\"读成\"H 机制无价值\"（GARCH_H 已含 H）")

PC["out"]["H_ONLY"] = O
json.dump(PC, open(os.path.join(DIR, "PAPER_CLAIMS.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"主张总表：{len(C)} 条；禁止声明：{len(NEG)} 条")
for c in C:
    print(f"  {c['id']:<5}{c['claim'][:50]:<52}{c['number'][:40]}")
