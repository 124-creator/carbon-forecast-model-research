# -*- coding: utf-8 -*-
"""在 §19/§20/§21/§22 标题下方插入 §23 撤回警示条。"""
import sys, pathlib
sys.stdout.reconfigure(encoding="utf-8")

DOC = pathlib.Path(r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020\DIRECTION_ZH.md")
p = DOC.read_text(encoding="utf-8-sig")

BAN = {
"## §19 第十版：机制 #18/#19 —— 两个日频锚定模型，两处 MCS 唯一存活":
"> **⚠ 本节结论已被 §23 撤回。** 脚本未在切分点 `t0` 截断测试原点，测试集 100% 包含训练原点。去泄漏重跑后 DAT 在 25 格中只胜 2 格，两处 MCS singleton 不再成立。",

"## §20 加固：位置项消融、三切分稳健性、n_eff 预言的确认":
"> **⚠ 本节数值已被 §23 减弱。** 去泄漏后合并斜率由 −7.46% 降为 −5.51%、`r` 由 −0.9427 降为 −0.7892；只有 CEA 与 SH 仍支持“优势随 h 上升”，广州符号相反。",

"## 21 基准层与经济价值：DAT0 被自己的族超越":
"> **⚠ 本节结论已被 §23 推翻。** 去泄漏后获胜计数为 `{EMP 6, EMP0 4, MSM1 4, HARh 4, SD250 1, HAR1 1}`——窗口类 10/20、日频族 10/20，`MLh`/`ML1`/`EW97` 各 0 胜，广州由窗口获胜。“DAT0 被自己的族超越”与“最优日频成员对窗口 20/20”均撤回。本节的 CRPS 表与经济价值表请以 §23.3 为准。",

"## 22 第二十个机制：期限依赖的收缩估计量——被否（且是本轮最锐利的失败）":
"> **⚠ 本节的数值受 §23 影响，但结论反而加强。** 该泄漏只会帮助被检验的机制（拟合量在训练原点上被评分），而它仍然失败，因此否证成立。去泄漏后“条件化上限约 3%”被 §23.3 独立复现（HARh −3.12%、MLh +4.24%、MSM1 −1.80%）。",
}

for head, banner in BAN.items():
    if head not in p:
        print(f"[MISS] {head[:40]}")
        continue
    if banner in p:
        print(f"[SKIP] 已存在: {head[:40]}")
        continue
    i = p.index(head) + len(head)
    p = p[:i] + "\n\n" + banner + p[i:]
    print(f"[OK] {head[:40]}")

DOC.write_text(p, encoding="utf-8", newline="\n")
print(f"total lines = {p.count(chr(10))}")
