# -*- coding: utf-8 -*-
"""FULL_ROSTER.json 的汇总：论文表一 + 聚合 + 显著性 + A5 vs 每格名册最优。"""
import numpy as np, os, sys, json
sys.stdout.reconfigure(encoding="utf-8")
OUT = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
D = json.load(open(os.path.join(OUT, "FULL_ROSTER.json"), encoding="utf-8"))
RES = {m: {int(h): v for h, v in RESM.items()} for m, RESM in D["res"].items()}
MODELS = D["models"]; MK = list(RES.keys())
HOR = sorted({h for m in RES for h in RES[m]})
cells = [(m, h) for m in MK for h in HOR if h in RES[m]]
med = lambda a: float(np.median(a))
print("=" * 172)
print(f"表一（未校准 CRPS，粗体为该格最优）  共 {len(cells)} 格")
print("=" * 172)
print(f"{'市场':<12}{'h':>5}{'n':>6}" + "".join(f"{n:>8}" for n in MODELS) + f"{'最优':>8}{'A5排':>7}")
for m, h in cells:
    r = RES[m][h]; c = r["crps"]
    bs = min(c, key=c.get)
    row = f"{m:<12}{h:>5}{r['n']:>6}"
    for n in MODELS:
        row += (f"{c[n]:>8.5f}" if n in c else f"{'-':>8}")
    rk = sorted([k for k in c], key=c.get).index("A5") + 1
    print(row + f"{bs:>8}{rk:>5}/{len(c)}")

print("\n" + "=" * 172)
print("聚合（A5 作为被检验模型）")
print("=" * 172)
print(f"{'模型':<9}{'CRPS/EMP':>10}{'胜EMP':>8}{'A5/该':>9}{'A5胜':>7}"
      f"{'A5显著胜':>9}{'A5显著负':>9}{'DM中位p':>9}")
for n in MODELS:
    if n == "A5":
        continue
    ok = [(m, h) for m, h in cells if n in RES[m][h]["crps"]]
    r = np.array([RES[m][h]["crps"][n] / RES[m][h]["crps"]["EMP"] for m, h in ok])
    a = np.array([RES[m][h]["crps"]["A5"] / RES[m][h]["crps"][n] for m, h in ok])
    p = np.array([RES[m][h]["p_vs_A5"][n] for m, h in ok])
    print(f"{n:<9}{med(r):>10.3f}{int((r<1).sum()):>5}/{len(r):<2}{med(a):>9.3f}"
          f"{int((a<1).sum()):>5}/{len(a):<2}{int(((p<0.05)&(a<1)).sum()):>9}"
          f"{int(((p<0.05)&(a>1)).sum()):>9}{med(p):>9.3f}")

print("\nA5 逐期限")
print(f"{'h':>5}{'vs EMP':>9}{'胜':>6}{'显著胜':>7}{'显著负':>7}"
      f"{'vs名册最优':>11}{'胜':>6}{'vs WIS/EMP':>12}")
for h in HOR:
    ks = [(m, h) for m in MK if h in RES[m]]
    r = np.array([RES[m][h]["crps"]["A5"] / RES[m][h]["crps"]["EMP"] for m, _ in ks])
    p = np.array([RES[m][h]["p_vs_A5"]["EMP"] for m, _ in ks])
    bo = []
    for m, _ in ks:
        c = RES[m][h]["crps"]
        alt = {k: v for k, v in c.items() if k != "A5"}
        bo.append(c["A5"] / min(alt.values()))
    bo = np.array(bo)
    w = np.array([RES[m][h]["wis"]["A5"] / RES[m][h]["wis"]["EMP"] for m, _ in ks])
    print(f"{h:>5}{med(r):>9.3f}{int((r<1).sum()):>4}/{len(r)}"
          f"{int(((p<0.05)&(r<1)).sum()):>7}{int(((p<0.05)&(r>1)).sum()):>7}"
          f"{med(bo):>11.3f}{int((bo<1).sum()):>4}/{len(bo)}{med(w):>12.3f}")

allr = np.array([RES[m][h]["crps"]["A5"] / RES[m][h]["crps"]["EMP"] for m, h in cells])
allw = np.array([RES[m][h]["wis"]["A5"] / RES[m][h]["wis"]["EMP"] for m, h in cells])
print(f"\n全 {len(cells)} 格：A5/EMP 中位 {med(allr):.3f} 胜 {int((allr<1).sum())}/{len(allr)}"
      f" | A5/EMP(WIS) 中位 {med(allw):.3f} 胜 {int((allw<1).sum())}/{len(allw)}")

print("\n" + "=" * 172)
print("族 F 的未校准结论（相对 EMP）")
print("=" * 172)
FAM = ["SD250", "EW97", "HAR1", "MSM1", "ML1", "VR", "EMP0", "HARh", "MLh"]
for n in FAM:
    ok = [(m, h) for m, h in cells if n in RES[m][h]["crps"]]
    if not ok:
        print(f"{n:<8} 无数据"); continue
    r = np.array([(RES[m][h]["crps"][n] / RES[m][h]["crps"]["EMP"] - 1) * 100 for m, h in ok])
    print(f"{n:<8} 中位 {med(r):+7.2f}%  更好 {int((r<0).sum()):>2}/{len(r)}  "
          f"最好 {r.min():+7.2f}%  最差 {r.max():+7.2f}%")

print("\nA5 覆盖率（未校准，逐期限中位）")
print(f"{'h':>5}{'c80':>8}{'c95':>8}")
for h in HOR:
    ks = [(m, h) for m in MK if h in RES[m]]
    print(f"{h:>5}{med([RES[m][h]['c80'] for m, _ in ks]):>8.3f}"
          f"{med([RES[m][h]['c95'] for m, _ in ks]):>8.3f}")
print("\n审计重叠总数 =", sum(a[4] for a in D["audit"]))
