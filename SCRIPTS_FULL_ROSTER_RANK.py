# -*- coding: utf-8 -*-
"""追加：A5 vs 每格最优的整体、排名分布、以及 SD250(=H 设为 0.5) 的干净检验。"""
import numpy as np, os, sys, json
sys.stdout.reconfigure(encoding="utf-8")
OUT = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
D = json.load(open(os.path.join(OUT, "FULL_ROSTER.json"), encoding="utf-8"))
RES = {m: {int(h): v for h, v in R.items()} for m, R in D["res"].items()}
MK = list(RES.keys()); HOR = sorted({h for m in RES for h in RES[m]})
cells = [(m, h) for m in MK for h in HOR if h in RES[m]]
med = lambda a: float(np.median(a))

bo, rank, tot = [], [], []
for m, h in cells:
    c = RES[m][h]["crps"]
    alt = {k: v for k, v in c.items() if k != "A5"}
    bo.append(c["A5"] / min(alt.values()))
    rk = sorted(c, key=c.get).index("A5") + 1
    rank.append(rk); tot.append(len(c))
bo = np.array(bo); rank = np.array(rank)
print("A5 相对每格名册最优（比值 <1 表示 A5 就是最优）")
print(f"  中位 {med(bo):.4f}   胜 {int((bo<1).sum())}/{len(bo)}   "
      f"最差 {bo.max():.4f}   最好 {bo.min():.4f}")
print(f"  排名分布：第1名 {(rank==1).sum()} 格，前三 {(rank<=3).sum()} 格，"
      f"后三分之一 {(rank>np.array(tot)*2/3).sum()} 格")
print(f"  平均排名 {rank.mean():.2f}（名册规模 {int(np.mean(tot))}）")
print("\n  逐格落后于最优的幅度 >5% 的：")
for (m, h), r, b in zip(cells, rank, bo):
    if b > 1.05:
        best = min({k: v for k, v in RES[m][h]["crps"].items() if k != "A5"}.items(),
                   key=lambda x: x[1])
        print(f"    {m:<12}h={h:<4} A5 {RES[m][h]['crps']['A5']:.5f}  "
              f"最优 {best[0]} {best[1]:.5f}  +{(b-1)*100:.1f}%  排名 {r}")

print("\n" + "=" * 104)
print("H 指数的两个干净检验（A5 = A5 规格，SD250 = 同一规格但 H 固定 0.5）")
print("=" * 104)
print(f"{'对照':<26}{'A5/该':>9}{'胜':>7}{'显著胜':>8}{'显著负':>8}{'DM中位p':>9}")
for n, lab in (("SD250", "A5 vs SD250 (H=0.5)"), ("VR", "A5 vs VR (滚动估 H)")):
    ok = [(m, h) for m, h in cells if n in RES[m][h]["crps"]]
    a = np.array([RES[m][h]["crps"]["A5"] / RES[m][h]["crps"][n] for m, h in ok])
    p = np.array([RES[m][h]["p_vs_A5"][n] for m, h in ok])
    print(f"{lab:<26}{med(a):>9.3f}{int((a<1).sum()):>5}/{len(a):<2}"
          f"{int(((p<0.05)&(a<1)).sum()):>8}{int(((p<0.05)&(a>1)).sum()):>8}{med(p):>9.3f}")
print("\n  SD250 与 VR 各自相对 EMP（若 VR ≈ SD250，说明滚动估 H 等于没估）")
for n in ("SD250", "VR", "A5"):
    r = np.array([RES[m][h]["crps"][n] / RES[m][h]["crps"]["EMP"] for m, h in cells])
    print(f"    {n:<8} /EMP 中位 {med(r):.3f}  胜 {int((r<1).sum())}/25")
d = np.array([RES[m][h]["crps"]["VR"] / RES[m][h]["crps"]["SD250"] for m, h in cells])
print(f"    VR/SD250 中位 {med(d):.3f}  胜 {int((d<1).sum())}/25  "
      f"=> 滚动 VR 估的 H 相对 H=0.5 {'有' if med(d)<0.98 else '没有'}实质增益")

print("\n" + "=" * 104)
print("逐期限：A5 vs SD250（H 的贡献是否随期限变化）")
print("=" * 104)
print(f"{'h':>5}{'A5/SD250':>11}{'胜':>7}{'显著胜':>8}")
for h in HOR:
    ks = [(m, h) for m in MK if h in RES[m]]
    a = np.array([RES[m][h]["crps"]["A5"] / RES[m][h]["crps"]["SD250"] for m, _ in ks])
    p = np.array([RES[m][h]["p_vs_A5"]["SD250"] for m, _ in ks])
    print(f"{h:>5}{med(a):>11.3f}{int((a<1).sum()):>5}/{len(a):<2}{int(((p<0.05)&(a<1)).sum()):>8}")

print("\n" + "=" * 104)
print("缺测披露")
print("=" * 104)
for n in ("GARCH",):
    ok = [(m, h) for m, h in cells if n in RES[m][h]["crps"]]
    print(f"  {n}: 仅 {len(ok)}/25 格（{sorted({m for m,_ in ok})}）——其余市场 persist>=1 被拒")
print("  MSM1: CEA 的 m0 拟合到识别下界 0.30，该格 MSM 参数不可信")
