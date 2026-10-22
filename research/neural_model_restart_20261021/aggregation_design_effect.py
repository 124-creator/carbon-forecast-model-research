"""Design effect: how much larger is Var(mean of h daily proxies) than the
independence assumption Var(P)/h?  Descriptive only, training period, no model.
"""

import numpy as np
import os

d = os.path.dirname(os.path.abspath(__file__))
g = dict(np.load(os.path.join(d, "grn_range_development.npz"), allow_pickle=False))
# GRN_proxy_raw is the Parkinson proxy. NOTE: GRN_log_proxy encodes a zero range
# as log 0 -> sentinel 0.0, so exp(GRN_log_proxy) would turn 342 missing days into
# 1.0, three orders of magnitude above the real maximum. Use the raw array.
P = g["GRN_proxy_raw"].astype(np.float64)
cut = int(g["GRN_cut"])                                    # first row of 2025
dates = g["GRN_date"]
print("series len", P.shape, "pre-2025 rows", cut, "date range", dates[0], dates[-1])
print("raw proxy: zeros pre-2025 = %d / %d ; full series zeros = %d / %d"
      % (int((P[:cut] == 0).sum()), cut, int((P == 0).sum()), P.size))

p = P[:cut]
p = p[np.isfinite(p)]
print("positive train proxy n =", p.size, " mean=%.6e  var=%.6e" % (p.mean(), p.var(ddof=1)))

print("\n%-4s %8s %14s %14s %10s %10s" % ("h", "n_win", "var(window mean)", "var(P)/h", "design", "sd ratio"))
for h in (1, 2, 5, 10, 22, 44):
    if h == 1:
        wm = p
    else:
        n = p.size - h + 1
        if n <= 2:
            continue
        cs = np.concatenate(([0.0], np.cumsum(p)))
        wm = (cs[h:] - cs[:-h]) / h
    indep = p.var(ddof=1) / h
    design = wm.var(ddof=1) / indep
    print("%-4d %8d %14.6e %14.6e %10.3f %10.3f"
          % (h, wm.size, wm.var(ddof=1), indep, design, np.sqrt(design)))

# implied nominal-95% interval width ratio for the h-day mean
print("\n若按独立假设构造区间，22日均值的 95% 区间宽度会被低估约 "
      f"{np.sqrt(1.0):.2f} 倍 -> 下面给出实测倍率：")
cs = np.concatenate(([0.0], np.cumsum(p)))
h = 22
wm = (cs[h:] - cs[:-h]) / h
print("  实测方差膨胀 = %.3f 倍, 宽度低估 = %.3f 倍" %
      (wm.var(ddof=1) / (p.var(ddof=1) / h),
       np.sqrt(wm.var(ddof=1) / (p.var(ddof=1) / h))))

# empirical lag-k autocorrelation of the proxy, weighted 1-k/h
x = p - p.mean()
den = np.dot(x, x)
print("\nlag  rho_k   1-k/22")
acc = 1.0
for k in range(1, 22):
    r = float(np.dot(x[:-k], x[k:]) / den)
    acc += 2 * (1 - k / 22) * r
    if k <= 5 or k % 5 == 0:
        print("%3d %+.4f  %.4f" % (k, r, 1 - k / 22))
print("方差膨胀的解析值 = %.3f" % acc)

# ---- persist ----
import json
rows = []
p_all = P[:cut]
p_all = p_all[np.isfinite(p_all)]
cs_all = np.concatenate(([0.0], np.cumsum(p_all)))
for h in (1, 2, 5, 10, 22, 44):
    if h == 1:
        wm = p_all
    else:
        n = p_all.size - h + 1
        if n <= 2:
            continue
        wm = (cs_all[h:] - cs_all[: -h]) / h
    indep = p_all.var(ddof=1) / h
    rows.append({"h": h, "n_windows": int(wm.size),
                 "var_window_mean": float(wm.var(ddof=1)),
                 "var_independence": float(indep),
                 "design_effect": float(wm.var(ddof=1) / indep),
                 "sd_understatement": float(np.sqrt(wm.var(ddof=1) / indep))})
x_all = p_all - p_all.mean()
den_all = float(np.dot(x_all, x_all))
rhos = [{"lag": k, "rho": float(np.dot(x_all[:-k], x_all[k:]) / den_all)}
        for k in range(1, 22)]
payload = {
    "scope": "descriptive; GRN raw Parkinson proxy; pre-2025 rows only; no model",
    "source": "grn_range_development.npz:GRN_proxy_raw",
    "sentinel_warning": ("GRN_log_proxy stores a zero range as 0.0, so exp() maps "
                         "missing days to 1.0, far above the real maximum. Never "
                         "exp() this array."),
    "n_obs": int(p_all.size), "n_zero_pre2025": int((p_all == 0).sum()),
    "n_zero_full": int((P == 0).sum()), "n_zero_2025": int((P[cut:] == 0).sum()),
    "mean": float(p_all.mean()), "var": float(p_all.var(ddof=1)),
    "design_effect_by_h": rows,
    "design_effect_h22_analytic": float(acc),
    "autocorrelation": rhos,
    "reading": ("averaging h days reduces the sd of the target far less than sqrt(h); "
                "at h=22 an independence assumption understates the sd by about 1.96x"),
}
with open(os.path.join(d, "AGGREGATION_DESIGN_EFFECT.json"), "w", encoding="utf-8") as fh:
    json.dump(payload, fh, ensure_ascii=False, indent=2)
print("wrote AGGREGATION_DESIGN_EFFECT.json")
