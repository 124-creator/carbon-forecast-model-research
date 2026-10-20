# -*- coding: utf-8 -*-
"""论文草稿的数字核对：逐个从 JSON 重算，确认草稿里写的是同一个数。"""
import json, os, re, sys, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
DIR = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
J = lambda f: json.load(open(os.path.join(DIR, f), encoding="utf-8"))
P = open(os.path.join(DIR, "PAPER_DRAFT.md"), encoding="utf-8").read()
FE = J("FINAL_EVIDENCE.json"); FS = J("FINAL_SPEC.json"); PC = J("PAPER_CLAIMS.json")
PCAL = J("PLACEBO_CALIB.json"); DC = J("DM_CALIB.json"); ME = J("MECHANISM.json")
VF = J("VERIFY.json"); EV = J("ECONVALUE.json")

CHECKS = []
def c(name, s, must=True):
    n = P.count(str(s))
    CHECKS.append((name, str(s), n, must))
    return n

f = lambda b: FE["final"][b]
c("A14 vs A5 log ratio", f"{f('A5')['real_mu']:+.5f}")
c("A14 vs A5 p", f"{f('A5')['p']:.4f}")
c("A14 vs EMP p", f"{f('EMP')['p']:.4f}")
c("A14 vs RW log ratio", f"{f('RW')['real_mu']:+.5f}")
c("A14 vs RW p", f"{f('RW')['p']:.4f}")
c("A14 vs SD250 log ratio", f"{f('SD250')['real_mu']:+.5f}")
c("A14 vs SD250 p", f"{f('SD250')['p']:.4f}")
c("A14 vs VR p", f"{f('VR')['p']:.4f}")
c("h_sat=50 test log ratio", f"{PC['te_mu']['50']:+.5f}")
c("h_sat=50 test p", f"{PC['te_p']['50']:.4f}")
c("oracle gap", f"{PC['te_mu']['50']-PC['te_mu'][str(PC['oracle'])]:+.5f}")
c("oracle hsat", PC["oracle"])
c("DM L=10 false positive", f"{DC['calibration']['a10']['frac05']:.4f}")
c("DM L=10 KS p", f"{DC['calibration']['a10']['ksp']:.4f}")
c("perm false positive", f"{DC['calibration']['perm']['frac05']:.4f}")
c("real shape log ratio", f"{PCAL['out']['real']['real_mu']:+.5f}")
c("random shape log ratio", f"{PCAL['out']['rand']['real_mu']:+.5f}")
c("lagged shape log ratio", f"{PCAL['out']['lag']['real_mu']:+.5f}")
c("real shape p", f"{PCAL['out']['real']['p']:.4f}")
c("rand shape p", f"{PCAL['out']['rand']['p']:.4f}")
c("lag shape p", f"{PCAL['out']['lag']['p']:.4f}")
c("rnd1 p", f"{PCAL['out']['rnd1']['p']:.4f}")
c("rnd2 p", f"{PCAL['out']['rnd2']['p']:.4f}")
c("rnd3 p", f"{PCAL['out']['rnd3']['p']:.4f}")
c("LOO R2", f"{ME['loo']['r2']:.3f}")
c("verify max diff", f"{VF['max_abs_diff']:.2e}")
c("verify n checks", VF["n_checks"])
for m in MK if False else ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]:
    c(f"market p {m}", f"{f('A5')['by_market_p'][m]:.3f}")
for r in FS:
    c(f"split {r['frac']} hsat", r["sel"])
    c(f"split {r['frac']} p", f"{r['te_p']:.4f}")
for h in [5, 20, 40, 60, 120]:
    s = [r for r in EV if r["h"] == h]
    c(f"econ h={h} A14", f"{np.mean([r['e1_A14'] for r in s])*100:+.3f}%")

print("=" * 100)
print("论文草稿数字核对")
print("=" * 100)
bad = 0
for name, s, n, must in CHECKS:
    ok = (n > 0) if must else True
    if not ok:
        bad += 1
    print(f"  {'✓' if ok else '✗'} {name:<32} {s:>14}  草稿出现 {n} 次")
print(f"\n  未出现的数字：{bad} 项")

# 检查草稿里是否有不该有的声明
print("\n禁止声明检查：")
for bad_s in ("significantly better than existing methods",
              "A15 replaces A14",
              "A15 取代 A14"):
    n = P.count(bad_s)
    print(f"  {'✓' if n == 0 else '✗'} '{bad_s}' 出现 {n} 次")

# 草稿中所有百分数/小数的格式一致性
print("\n草稿中的裸数字抽查（可能手打错的）：")
sus = re.findall(r"(?<![\w.])-?\d+\.\d{4,}(?![\d])", P)
print(f"  4 位以上小数的出现：{len(sus)} 处，去重后 {len(set(sus))} 个不同值")
print(f"  {'  '.join(sorted(set(sus))[:18])}")
