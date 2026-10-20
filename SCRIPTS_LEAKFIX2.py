# -*- coding: utf-8 -*-
"""修补剩余两个带 BOM 的脚本（就地去掉 BOM 并修复 ts_te 截断）。"""
import sys, pathlib
sys.stdout.reconfigure(encoding="utf-8")

ROOT = pathlib.Path(r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research")
FILES = ["SCRIPTS_ECON_VALUE.py", "SCRIPTS_SHRINK_JOINT.py"]
OLD = "range(W + h + 5, len(lp) - h, STEP)"
NEW = "range(max(W + h + 5, t0), len(lp) - h, STEP)"

for fn in FILES:
    p = ROOT / fn
    raw = p.read_text(encoding="utf-8-sig")          # 吃掉 BOM
    bomb = raw.lstrip("\ufeff")
    n = bomb.count(OLD)
    patch = bomb.replace(OLD, NEW)
    idx = patch.find("ts_te = [t for t in " + NEW)
    ok = patch.rfind("t0 = int(", 0, idx) != -1 if idx != -1 else False
    try:
        compile(patch, fn, "exec")
        syn = "OK"
    except SyntaxError as e:
        print(f"[FAIL] {fn}: {e}"); continue
    p.write_text(patch, encoding="utf-8", newline="\n")   # 不带 BOM 写出
    print(f"[PATCH] {fn}: 替换 {n} 处  去BOM  t0前置={ok}  {syn}")

print("\n--- 复核（旧模式残留应为 0）---")
BOM = b"\xef\xbb\xbf"
for fn in FILES:
    s = (ROOT / fn).read_text(encoding="utf-8-sig")
    head = (ROOT / fn).open("rb").read(3)
    has_bom = (head == BOM)
    print(f"{fn:32s} 旧={s.count(OLD)} 新={s.count(NEW)} BOM={has_bom}")
