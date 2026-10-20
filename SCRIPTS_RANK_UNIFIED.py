# -*- coding: utf-8 -*-
"""
统一排名池：所有模型（10 个旧名册 + 3 个序列 + A5 + A14）同处一池，
每个候选比较时只把自己剔除。上一版 agg28 让 DL 的池里不含 A14、
A14 的池里含 DL，不对称，会高估 DL 的第 1 名。本脚本修正这一点。
输出 DL_SEQ_AGG.json（覆盖）。
"""
import json, os, sys, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
OUT = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
J = lambda f: json.load(open(os.path.join(OUT, f), encoding="utf-8"))
FR, DL, A14J = J("FULL_ROSTER.json")["res"], J("DL_SEQ.json"), J("A14.json")
MK = ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]
HOR = [5, 20, 40, 60, 120]
DLK = ["LSTM", "TCN", "TRF"]
cells = [(m, h) for m in MK for h in HOR]
med = lambda a: float(np.median(a))
OLD = sorted(k for k in FR["CEA"]["5"]["crps"] if k != "A5")
ALL = OLD + DLK + ["A5", "A14"]

VAL = {}
for m, h in cells:
    for k in OLD:
        if k in FR[m][str(h)]["crps"]:
            VAL[(m, h, k)] = FR[m][str(h)]["crps"][k]
    for k in DLK:
        VAL[(m, h, k)] = DL[m][str(h)][k]
    VAL[(m, h, "A5")] = FR[m][str(h)]["crps"]["A5"]
    VAL[(m, h, "A14")] = A14J["cells"][f"{m}|{h}"]["A14"]


def rivals(m, h, me):
    """同池对手：全部模型减去自己；GARCH 只在 EU 可估，故池大小逐格变化。"""
    return [VAL[(m, h, x)] for x in ALL if x != me and (m, h, x) in VAL]


def rank_of(me):
    bo, rk = [], []
    for m, h in cells:
        v = rivals(m, h, me)
        bo.append(VAL[(m, h, me)] / min(v))
        rk.append(sum(1 for x in v if x < VAL[(m, h, me)]) + 1)
    bo, rk = np.array(bo), np.array(rk)
    return dict(bo=bo.tolist(), rk=rk.tolist(), mean_rank=float(rk.mean()),
                firsts=int((rk == 1).sum()), top3=int((rk <= 3).sum()),
                wins=int((bo <= 1.0).sum()), worst=float(bo.max()))


CANDS = ["LSTM", "TCN", "TRF", "A5", "A14"]
RANK = {c: rank_of(c) for c in CANDS}

print("=" * 110)
print(f"统一池 = {len(ALL)} 个模型（{len(OLD)} 旧名册 + 3 序列 + A5 + A14），候选各自剔除自己")
print("=" * 110)
print(f"{'模型':<8}{'/池最优中位':>13}{'胜(=最优)':>11}{'平均排名':>10}{'第1名':>8}{'前三':>7}{'最差':>9}")
for c in CANDS:
    r = RANK[c]
    print(f"{c:<8}{med(r['bo']):>13.4f}{r['wins']:>8}/25{r['mean_rank']:>10.2f}"
          f"{r['firsts']:>8}{r['top3']:>7}{r['worst']:>9.4f}")

for c in CANDS:
    print(f"\n{c} 的第 1 名 {RANK[c]['firsts']} 格: "
          f"{[f'{MN}' for MN in [f'{m[:3]} h={h}' for m,h in cells if RANK[c]['rk'][cells.index((m,h))]==1]]}")

print("\n各格最优模型归属：")
win = {}
for m, h in cells:
    best = min([x for x in ALL if (m, h, x) in VAL], key=lambda x: VAL[(m, h, x)])
    win.setdefault(best, []).append(f"{m[:3]}h{h}")
for k, v in sorted(win.items(), key=lambda x: -len(x[1])):
    print(f"  {k:<8}{len(v):>2} 格: {' '.join(v)}")

print("\n逐期限 LSTM/A5 与 A14/LSTM：")
for h in HOR:
    print(f"  h={h:<4}LSTM/A5 {med([VAL[(m,h,'LSTM')]/VAL[(m,h,'A5')] for m in MK]):.4f}   "
          f"A14/LSTM {med([VAL[(m,h,'A14')]/VAL[(m,h,'LSTM')] for m in MK]):.4f}   "
          f"A14 胜 LSTM {sum(1 for m in MK if VAL[(m,h,'A14')]<VAL[(m,h,'LSTM')])}/5")

json.dump({"rank": RANK, "pool": ALL, "dl_beats_a5":
           [[m, h, k, float(DL[m][str(h)]["p_" + k])] for m, h in cells for k in DLK
            if DL[m][str(h)]["p_" + k] < .05 and VAL[(m, h, k)] < VAL[(m, h, "A5")]],
           "spread": {k: med([DL[m][str(h)]["spread_" + k] for m, h in cells]) for k in DLK}},
          open(os.path.join(OUT, "DL_SEQ_AGG.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已覆盖写出 DL_SEQ_AGG.json")
