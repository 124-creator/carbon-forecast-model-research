# -*- coding: utf-8 -*-
"""修补 ts_te 未在 t0 截断的泄漏：range(W+h+5, len(lp)-h, STEP) -> range(max(W+h+5, t0), ...)"""
import sys, io, re, pathlib
sys.stdout.reconfigure(encoding="utf-8")

ROOT = pathlib.Path(r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research")
FILES = ["SCRIPTS_DAT.py", "SCRIPTS_SCALE_ABLATION.py", "SCRIPTS_SCALE_FINAL.py",
         "SCRIPTS_HARDEN.py", "SCRIPTS_BENCH_FAMILY.py", "SCRIPTS_ECON_VALUE.py",
         "SCRIPTS_SHRINK_PERCELL.py", "SCRIPTS_SHRINK_JOINT.py"]
OLD = "range(W + h + 5, len(lp) - h, STEP)"
NEW = "range(max(W + h + 5, t0), len(lp) - h, STEP)"

for fn in FILES:
    p = ROOT / fn
    src = p.read_text(encoding="utf-8")
    n = src.count(OLD)
    if n == 0:
        print(f"[SKIP] {fn}: 未找到目标串")
        continue
    # 确认 t0 在该行之前已定义
    idx = src.find("ts_te = [t for t in " + OLD)
    t0pos = src.rfind("t0 = int(", 0, idx)
    patch = src.replace(OLD, NEW)
    ok = t0pos != -1
    # 语法与补齐校验
    try:
        compile(patch, fn, "exec")
        syn = "OK"
    except SyntaxError as e:
        syn = f"SYNTAX-ERR {e}"
        print(f"[FAIL] {fn}: {syn}"); continue
    if not ok:
        print(f"[WARN] {fn}: 替换 {n} 处，但未在 ts_te 之前找到 t0 定义，请人工核对")
    p.write_text(patch, encoding="utf-8", newline="\n")
    print(f"[PATCH] {fn}: 替换 {n} 处  t0前置={ok}  {syn}")

print("\n--- 复核 ---")
for fn in FILES:
    s = (ROOT / fn).read_text(encoding="utf-8")
    bad = s.count(OLD)
    good = s.count(NEW)
    print(f"{fn:32s} 旧={bad} 新={good}")
