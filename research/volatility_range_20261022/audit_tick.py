"""Is the range measuring volatility, or the tick?

Carbon ETFs are quoted in cents. For a $20 fund one tick is 5 basis points, and on
a quiet day the whole high-low span can be one or two ticks. A range estimator
then reports the tick size, not the volatility, which biases quiet periods upward
through a floor and makes QLIKE (which diverges as RV/forecast -> 0) unstable.

Diagnostics:
  D1  what is the modal high-low span, in ticks?
  D2  what share of days sit at 1, 2, 3 ticks?
  D3  is the span grid-aligned (integer multiples of the tick)?
  D4  do 1-tick days have the same RV distribution as 3-tick days, i.e. is the
      floor binding rather than informative?
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SRC = (r"E:\Agent员工\碳价格预测研究\独立数据归档\_legacy_202605_archive"
       r"\主要实验推进_审计加固版_CLEAN_20260528\08-新实验"
       r"\22_structural_ohlc_q2_route_20260609\03_raw_data\normalized_yahoo_ohlc")
OUT = os.path.dirname(os.path.abspath(__file__))

for s in ("KRBN", "KEUA", "KCCA"):
    d = pd.read_csv(os.path.join(SRC, "%s_ohlc.csv" % s), parse_dates=["date"])
    d = d.dropna(subset=["open", "high", "low", "close"]).sort_values("date")
    d = d[(d["high"] >= d["low"]) & (d["close"] > 0)]
    h, l, c = (d[k].to_numpy(float) for k in ("high", "low", "close"))

    tick = 0.01
    span = (h - l) / tick                     # in ticks
    span_r = np.round(span, 6)
    grid = np.mean(np.isclose(span, span_r, atol=1e-4)) * 100

    print("=== %s  n=%d  median close=%.2f ===" % (s, len(d), np.median(c)))
    print("  price granularity: share of closes on the 0.01 grid = %.2f%%"
          % (100 * np.mean(np.isclose(c * 100, np.round(c * 100), atol=1e-6))))
    print("  span grid alignment = %.2f%%" % grid)
    vc = pd.Series(span_r).value_counts().sort_index()
    small = vc[vc.index <= 6]
    print("  span (ticks) counts 1..6: " +
          "  ".join("%d:%.1f%%" % (k, 100 * v / len(d))
                    for k, v in small.items()))
    le2 = float(np.mean(span_r <= 2)) * 100
    le1 = float(np.mean(span_r <= 1)) * 100
    print("  share <=1 tick = %.1f%%   <=2 ticks = %.1f%%" % (le1, le2))

    lr = np.log(np.maximum(h, l * (1 + 1e-12)) / l)
    park = lr ** 2 / (4 * np.log(2))
    ret = np.r_[np.nan, np.diff(np.log(c))]
    for k in (1, 2, 3, 4):
        m = span_r == k
        if m.sum() > 5:
            print("   span=%d ticks: n=%4d  median Parkinson=%.3e  "
                  "median ret^2=%.3e  ratio=%.2f"
                  % (k, m.sum(), np.median(park[m]),
                     np.nanmedian(ret[m] ** 2),
                     np.median(park[m]) / max(np.nanmedian(ret[m] ** 2), 1e-20)))
    print()
