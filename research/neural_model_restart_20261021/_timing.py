import time

import numpy as np
import pandas as pd
from arch import arch_model

P = (r"E:\Agent员工\碳价格预测研究\独立数据归档\_legacy_202605_archive"
     r"\主要实验推进_审计加固版_CLEAN_20260528\08-新实验\01_数据"
     r"\EUA_原始数据_完整版.csv")

d = pd.read_csv(P)
c = [x for x in d.columns if "close" in x.lower()][0]
r = np.log(d[c].astype(float)).diff().dropna().values * 100
print("n =", len(r))

t = time.time()
am = arch_model(r[:2000], mean="Zero", vol="Garch", p=1, o=0, q=1, dist="normal")
res = am.fit(disp="off")
print(f"fit 2000 obs: {time.time()-t:.2f}s")
print(dict(res.params))

t = time.time()
for j in range(2000, 2022):
    a = arch_model(r[:j], mean="Zero", vol="Garch", p=1, o=0, q=1, dist="normal")
    a.fix(res.params)
print(f"22 fixed-parameter filters: {time.time()-t:.2f}s")

t = time.time()
f = res.forecast(horizon=22, start=2000, reindex=False)
print("forecast shape", f.variance.shape, f"{time.time()-t:.2f}s")
