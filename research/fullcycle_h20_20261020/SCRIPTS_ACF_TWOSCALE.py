# -*- coding: utf-8 -*-
"""
ACF 形状诊断：碳价收益率是否存在「双时间尺度」结构？
------------------------------------------------
起因：hrec1.py 发现 rho1 恒为负，但 sum(rho_2..40) 可为正。
若成立，则单一 H 的幂律在原理上无法描述，必须用两分量叠加。

对每个市场输出：
  - ACF 在 1,2,3,5,10,20,40,60,120,250 处的值
  - 累积 VR 曲线 VR(k) = 1 + 2*sum_{d<k}(1-d/k)rho_d
  - 分段符号：短端(1-5)、中端(6-40)、长端(41-250) 的 rho 均值
  - 双分量拟合：VR(h) = w*VR_fast(h; phi) + (1-w)*VR_slow(h; H)
"""
import pandas as pd, numpy as np, os, sys, warnings
warnings.filterwarnings("ignore")
from scipy.optimize import minimize
sys.stdout.reconfigure(encoding="utf-8")

OUT = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(OUT, "PANEL_5MARKET.csv"), parse_dates=["date"])
LAGS = [1, 2, 3, 5, 10, 20, 40, 60, 120, 250]


def acf_full(x, nlags):
    x = np.asarray(x, float) - np.mean(x)
    v = np.dot(x, x)
    return np.array([np.dot(x[:-k], x[k:]) / v for k in range(1, nlags + 1)])


def rho_acf(r, ks=(2, 3, 5, 10, 20, 40, 60, 120, 250)):
    """ACF 与 VR 都用价格水平的 k 期差，与 hrec1/hurst10 口径一致"""
    lv = r
    dv = np.var(np.diff(lv), ddof=1)
    out = {}
    for k in ks:
        if len(lv) <= k + 5:
            out[k] = np.nan
            continue
        out[k] = np.var(lv[k:] - lv[:-k], ddof=1) / (k * dv)
    return out


print("=" * 118)
print("A. 收益率 ACF 的形状（负号出现在哪里？）")
print("=" * 118)
hdr = f"{'市场':<11}{'期':<12}{'n':>6}" + "".join(f"{'r'+str(k):>8}" for k in LAGS)
print(hdr)
ACC = {}
for mk in ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]:
    g = p[p.market == mk].sort_values("date").reset_index(drop=True)
    lp = np.log(g.price.to_numpy(float)).astype(float)
    dt = g.date.to_numpy()
    cut = int(np.searchsorted(dt, np.datetime64("2024-01-01")))
    for nm, (a, b) in {"全样本": (0, len(lp)), "2024+": (cut, len(lp))}.items():
        lv = lp[a:b]
        if len(lv) < 120:
            continue
        r = np.diff(lv)
        rho = acf_full(r, 250)
        ACC[(mk, nm)] = rho
        line = f"{mk:<11}{nm:<12}{len(lv):>6}"
        for k in LAGS:
            if k <= len(rho):
                line += f"{rho[k-1]:>8.3f}"
            else:
                line += f"{'--':>8}"
        print(line)

print()
print("=" * 118)
print("B. 分段 ACF 均值：短端 vs 长端 是否异号？")
print("=" * 118)
print(f"{'市场':<11}{'期':<12}{'rho_1-5':>10}{'rho_6-40':>10}{'rho_41-120':>12}{'rho_121-250':>13}{'判定':>12}")
for (mk, nm), rho in ACC.items():
    s1 = rho[:5].mean()
    s2 = rho[5:40].mean()
    s3 = rho[40:120].mean() if len(rho) >= 120 else np.nan
    s4 = rho[120:250].mean() if len(rho) >= 250 else np.nan
    flag = "短负长正" if (s1 < 0 < s2) else ("短负长负" if (s1 < 0 and s2 < 0) else "其他")
    print(f"{mk:<11}{nm:<12}{s1:>10.3f}{s2:>10.3f}{s3:>12.3f}{s4:>13.3f}{flag:>12}")

print()
print("=" * 118)
print("C. 双分量叠加拟合  VR(h) = w*VR_fast(h;phi) + (1-w)*VR_slow(h;H)")
print("   VR_fast: OU/AR(1) 型, VR_fast(h)=1+2*sum_{d<h}(1-d/h)phi^d")
print("   VR_slow: 分数阶,   VR_slow(h)=h^(2H-1)")
print("=" * 118)


def vr_fast(h, phi):
    d = np.arange(1, h)
    return 1 + 2 * np.sum((1 - d / h) * phi ** d)


def model(hs, w, phi, H):
    return np.array([w * vr_fast(int(h), phi) + (1 - w) * h ** (2 * H - 1) for h in hs])


HS = np.array([2, 3, 5, 10, 20, 40, 60, 120, 250], float)
print(f"{'市场':<11}{'期':<12}{'w':>7}{'phi':>8}{'H':>8}{'RMSE':>9}{'R2':>8}   "
      f"{'单分量H':>9}{'RMSE':>9}")
FITS = {}
for (mk, nm), rho in ACC.items():
    g = p[p.market == mk].sort_values("date").reset_index(drop=True)
    lp = np.log(g.price.to_numpy(float)).astype(float)
    dt = g.date.to_numpy()
    cut = int(np.searchsorted(dt, np.datetime64("2024-01-01")))
    a, b = (0, len(lp)) if nm == "全样本" else (cut, len(lp))
    lv = lp[a:b]
    vr = np.array([rho_acf(lv, (int(h),))[int(h)] for h in HS])
    ok = np.isfinite(vr) & (vr > 1e-6)
    if ok.sum() < 5:
        continue
    x, y = HS[ok], vr[ok]

    def loss(t):
        w, phi, H = t
        if not (0.01 < w < 0.99 and -0.98 < phi < 0.99 and 0.05 < H < 0.95):
            return 1e6
        return np.mean((np.log(model(x, w, phi, H)) - np.log(y)) ** 2)

    best = None
    for w0 in (0.2, 0.5, 0.8):
        for phi0 in (-0.5, -0.2, 0.2, 0.6):
            for H0 in (0.2, 0.4, 0.6):
                r = minimize(loss, [w0, phi0, H0], method="Nelder-Mead",
                             options=dict(maxiter=4000, xatol=1e-8, fatol=1e-12))
                if best is None or r.fun < best.fun:
                    best = r
    w, phi, H = best.x
    pred = model(x, w, phi, H)
    ss = 1 - np.sum((np.log(y) - np.log(pred)) ** 2) / np.sum((np.log(y) - np.log(y).mean()) ** 2)
    # 单分量
    lg = np.log(x)
    Hs = (np.sum(lg * np.log(y)) / np.sum(lg ** 2) + 1) / 2
    pred1 = x ** (2 * Hs - 1)
    ss1 = 1 - np.sum((np.log(y) - np.log(pred1)) ** 2) / np.sum((np.log(y) - np.log(y).mean()) ** 2)
    FITS[(mk, nm)] = dict(w=w, phi=phi, H=H, r2=ss, r2_single=ss1, H_single=Hs,
                          rmse=float(np.sqrt(best.fun)), hs=x.tolist(), vr=y.tolist())
    print(f"{mk:<11}{nm:<12}{w:>7.3f}{phi:>8.3f}{H:>8.3f}{np.sqrt(best.fun):>9.4f}{ss:>8.4f}   "
          f"{Hs:>9.3f}{np.sqrt(np.mean((np.log(y)-np.log(pred1))**2)):>9.4f}")

print()
print("=" * 118)
print("D. 拟合优度对比汇总：双分量 vs 单分量幂律")
print("=" * 118)
print(f"{'市场':<11}{'期':<12}{'R2(双)':>9}{'R2(单)':>9}{'提升':>9}{'phi(短期回复)':>15}{'H(长期)':>9}")
for (mk, nm), f in FITS.items():
    print(f"{mk:<11}{nm:<12}{f['r2']:>9.4f}{f['r2_single']:>9.4f}{f['r2']-f['r2_single']:>+9.4f}"
          f"{f['phi']:>15.3f}{f['H']:>9.3f}")

import json
json.dump({f"{k[0]}_{k[1]}": v for k, v in FITS.items()},
          open(os.path.join(OUT, "ACF_TWOSCALE.json"), "w"), indent=1, ensure_ascii=False)
print("\n已写出 ACF_TWOSCALE.json")
