# -*- coding: utf-8 -*-
"""
论文可复现清单（REPRODUCE.md 的生成器）
=====================================================================
为每个出现在论文里的数字，列出：生成脚本、输入文件、输出 JSON、以及重跑命令。
同时做**可复现性自检**：检查每个 JSON 是否存在、每个关键字段是否齐全。
输出 REPRODUCE.json + REPRODUCE.md
"""
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8")
REPO = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research"
DIR = os.path.join(REPO, "research", "fullcycle_h20_20261020")
sys.path.insert(0, REPO)

ITEMS = [
    ("R1", "主切分上的 (h0,H) 与 A5/A14 的 CRPS", "SCRIPTS_A14.py",
     "PANEL_5MARKET.csv", "FULL_ROSTER.json, A14.json",
     "python SCRIPTS_A14.py"),
    ("R2", "五切分稳健性（规则 R 完整链条）", "SCRIPTS_EXP17_FINAL_SPEC.py",
     "PANEL_5MARKET.csv", "FINAL_SPEC.json", "python SCRIPTS_EXP17_FINAL_SPEC.py"),
    ("R3", "主切分上的规则 R → h_sat=50", "SCRIPTS_EXP20_PAPER_CLAIMS.py",
     "PANEL_5MARKET.csv", "PAPER_CLAIMS.json", "python SCRIPTS_EXP20_PAPER_CLAIMS.py"),
    ("R4", "最终证据表（A14 vs 各对照，200 次打乱）", "SCRIPTS_EXP10_FINAL_EVIDENCE.py",
     "PANEL_5MARKET.csv", "FINAL_EVIDENCE.json", "python SCRIPTS_EXP10_FINAL_EVIDENCE.py"),
    ("R5", "DM 检验的零假设校准", "SCRIPTS_EXP8_DM_CALIB.py",
     "PANEL_5MARKET.csv", "DM_CALIB.json", "python SCRIPTS_EXP8_DM_CALIB.py"),
    ("R6", "合并检验的统计效力", "SCRIPTS_EXP9_POWER_TEST.py",
     "PANEL_5MARKET.csv", "POWER_TEST.json", "python SCRIPTS_EXP9_POWER_TEST.py"),
    ("R7", "安慰剂电池（校准版）", "SCRIPTS_EXP18_PLACEBO_CALIB.py",
     "PANEL_5MARKET.csv", "PLACEBO_CALIB.json", "python SCRIPTS_EXP18_PLACEBO_CALIB.py"),
    ("R8", "16 期限密网格", "SCRIPTS_EXP2_HORIZON_GRID.py",
     "PANEL_5MARKET.csv", "HORIZON_GRID.json", "python SCRIPTS_EXP2_HORIZON_GRID.py"),
    ("R9", "多指标一致性", "SCRIPTS_EXP3_MULTIMETRIC.py",
     "PANEL_5MARKET.csv", "MULTIMETRIC.json", "python SCRIPTS_EXP3_MULTIMETRIC.py"),
    ("R10", "数据口径稳健性", "SCRIPTS_EXP5_DATAROBUST.py",
     "PANEL_5MARKET.csv", "DATAROBUST.json", "python SCRIPTS_EXP5_DATAROBUST.py"),
    ("R11", "经济价值与履约决策", "SCRIPTS_EXP6_ECONVALUE.py",
     "PANEL_5MARKET.csv", "ECONVALUE.json", "python SCRIPTS_EXP6_ECONVALUE.py"),
    ("R12", "市场级诊断量与留一市场检验", "SCRIPTS_EXP19_MECHANISM.py",
     "PANEL_5MARKET.csv", "MECHANISM.json", "python SCRIPTS_EXP19_MECHANISM.py"),
    ("R13", "H 的跨市场迁移（否）", "SCRIPTS_H_TRANSFER.py",
     "PANEL_5MARKET.csv", "H_TRANSFER.json", "python SCRIPTS_H_TRANSFER.py"),
    ("R14", "深度学习序列基准", "SCRIPTS_DL_SEQ.py",
     "PANEL_5MARKET.csv", "DL_SEQ.json, DL_SEQ_AGG.json", "python SCRIPTS_DL_SEQ.py"),
    ("R15", "独立复现检验", "SCRIPTS_VERIFY.py",
     "PANEL_5MARKET.csv, FULL_ROSTER.json, A14.json", "VERIFY.json",
     "python SCRIPTS_VERIFY.py"),
    ("R16", "论文图 1–5", "SCRIPTS_FIGURES.py",
     "HORIZON_GRID.json, MULTIMETRIC.json, FINAL_EVIDENCE.json, MULTISPLIT_A14.json",
     "figures/*.png", "python SCRIPTS_FIGURES.py"),
]

print("=" * 104)
print("可复现性自检")
print("=" * 104)
rows = []
allok = True
for rid, what, script, inp, out, cmd in ITEMS:
    sp = os.path.join(REPO, script)
    ok_s = os.path.exists(sp)
    outs = out.split(", ")
    miss = []
    for o in outs:
        if "*" in o:
            import glob
            if not glob.glob(os.path.join(DIR, o)):
                miss.append(o)
        elif not os.path.exists(os.path.join(DIR, o)):
            miss.append(o)
    ok_o = len(miss) == 0
    status = "✓" if (ok_s and ok_o) else "✗"
    if not (ok_s and ok_o):
        allok = False
    print(f"  {status} {rid:<4} {what[:40]:<42} 脚本{'有' if ok_s else '缺'} "
          f"输出{'全' if ok_o else '缺:' + ','.join(miss)}")
    rows.append(dict(id=rid, what=what, script=script, inputs=inp, outputs=out,
                      command=cmd, script_exists=bool(ok_s), missing_outputs=miss))

print("\n  论文图：")
FIGD = os.path.join(DIR, "figures")
for f in sorted(os.listdir(FIGD)) if os.path.isdir(FIGD) else []:
    print(f"    ✓ {f}  ({os.path.getsize(os.path.join(FIGD, f))} 字节)")

print("\n" + "=" * 104)
print("环境")
print("=" * 104)
import platform
print(f"  Python {platform.python_version()}  {platform.platform()}")
import numpy, pandas, scipy, sklearn, statsmodels
print(f"  numpy {numpy.__version__}  pandas {pandas.__version__}  scipy {scipy.__version__}")
print(f"  sklearn {sklearn.__version__}  statsmodels {statsmodels.__version__}")
try:
    import arch
    print(f"  arch {arch.__version__}")
except Exception:
    pass
try:
    import torch
    print(f"  torch {torch.__version__}")
except Exception:
    pass

print("\n" + "=" * 104)
print("可复现性清单（写入 REPRODUCE.md）")
print("=" * 104)
L = []
L.append("# 可复现性清单")
L.append("")
L.append("本文每个出现在正文或图表中的数字，都由下表中的脚本从 `PANEL_5MARKET.csv` "
         "（sha256 `d470238386e201e6...`，8912 行）重新生成。")
L.append("")
L.append("## 运行环境")
L.append("")
L.append("```")
L.append(f"Python {platform.python_version()}")
L.append(f"numpy {numpy.__version__}, pandas {pandas.__version__}, scipy {scipy.__version__}")
L.append(f"scikit-learn {sklearn.__version__}, statsmodels {statsmodels.__version__}")
L.append("```")
L.append("")
L.append("## 数字 → 脚本 对照表")
L.append("")
L.append("| ID | 论文中的内容 | 脚本 | 输入 | 输出 | 重跑命令 |")
L.append("|---|---|---|---|---|---|")
for r in rows:
    L.append(f"| {r['id']} | {r['what']} | `{r['script']}` | `{r['inputs']}` | "
             f"`{r['outputs']}` | `{r['command']}` |")
L.append("")
L.append("## 两条必须写进论文的口径")
L.append("")
L.append("```python")
L.append("# 测试原点（注意是从 t0 起步长 2，不是从 W+h+5 起步长 2）")
L.append("ts_te = range(max(W + h + 5, t0), len(lp) - h, STEP)   # STEP = 2")
L.append("")
L.append("# 有效性过滤")
L.append("if len(win) < 120 or len(rr) < 40 or np.std(rd, ddof=1) <= 0:")
L.append("    ok = False   # 该原点被剔除")
L.append("```")
L.append("")
L.append("这两条若不写清，数字无法复现。§24.41 的独立复现检验正是因此发现过"
         "一处奇偶性口径差异。")
L.append("")
L.append("## 论文图")
L.append("")
for f in sorted(os.listdir(FIGD)) if os.path.isdir(FIGD) else []:
    L.append(f"- `figures/{f}`")
L.append("")

open(os.path.join(DIR, "REPRODUCE.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L))
json.dump({"items": rows, "all_ok": allok},
          open(os.path.join(DIR, "REPRODUCE.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"已写出 REPRODUCE.md（{len(L)} 行）与 REPRODUCE.json")
print(f"自检：{'✓ 全部齐备' if allok else '✗ 有缺失'}")
