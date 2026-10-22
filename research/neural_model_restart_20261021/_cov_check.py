import pandas as pd

f = (r"E:\Agent员工\碳价格预测研究\独立数据归档\_legacy_202605_archive"
     r"\主要实验推进_审计加固版_CLEAN_20260528\08-新实验\01_数据\EUA_原始数据_完整版.csv")
cols = ["eua_close_t", "brent_oil_volatility20", "natural_gas_volatility20",
        "newcastle_coal_volatility20", "ttf_gas_volatility10", "vix_VIXCLS_lag1",
        "china_epu_lag1", "international_carbon_attention_index_lag1",
        "stoxx600_return_lag1", "eurusd_return_lag1",
        "brent_price", "ttf_gas_price", "stoxx600_price"]
d = pd.read_csv(f, usecols=["date"] + cols)
d["date"] = pd.to_datetime(d["date"])
d = d.sort_values("date").reset_index(drop=True)
print("total rows", len(d), d["date"].min().date(), "->", d["date"].max().date())
print()
print("column".ljust(44), "nonNaN".rjust(7), "first".rjust(12), "last".rjust(12), "frac_2019p".rjust(10))
for c in cols:
    s = d[["date", c]].dropna()
    if len(s) == 0:
        print(c.ljust(44), "0".rjust(7), "ALL EMPTY")
        continue
    after = d.loc[d["date"] >= "2019-01-01", c].notna().mean()
    print(c.ljust(44), str(len(s)).rjust(7),
          str(s["date"].min().date()).rjust(12), str(s["date"].max().date()).rjust(12),
          ("%.3f" % after).rjust(10))
