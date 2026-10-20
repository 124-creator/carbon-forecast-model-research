# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, os, sys, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
from arch import arch_model
OUT = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(OUT,"PANEL_5MARKET.csv"), parse_dates=["date"])
CUT={"CEA":"2024-01-01","GUANGZHOU":"2024-01-01","SHANGHAI":"2024-01-01","HUBEI":"2023-01-01","EU":"2023-01-01"}
print(f"{'市场':<11}{'样本':<7}{'dist':<8}{'mean':<8}{'omega':>11}{'alpha':>7}{'beta':>7}{'persist':>9}{'nu':>7}")
for m,cut in CUT.items():
    d=p[p.market==m].sort_values("date").reset_index(drop=True)
    lp=np.log(d.price.to_numpy(float)); ra=np.diff(lp)
    dt=pd.to_datetime(d.date).to_numpy(); t0=int(np.searchsorted(dt,np.datetime64(cut)))
    for tag,rr in (("train",ra[:t0]),("full",ra)):
        for dist in ("normal","t"):
            for mn in ("Zero","Constant"):
                try:
                    am=arch_model(rr*100.0, mean=mn, vol="Garch", p=1, q=1, dist=dist)
                    r_=am.fit(disp="off", show_warning=False)
                    al=float(r_.params["alpha[1]"]); be=float(r_.params["beta[1]"])
                    om=float(r_.params["omega"])/1e4
                    nu=float(r_.params["nu"]) if dist=="t" else np.nan
                    print(f"{m:<11}{tag:<7}{dist:<8}{mn:<8}{om:>11.3e}{al:>7.3f}{be:>7.3f}{al+be:>9.4f}{nu:>7.2f}")
                except Exception as e:
                    print(f"{m:<11}{tag:<7}{dist:<8}{mn:<8}  FAIL {type(e).__name__}")
