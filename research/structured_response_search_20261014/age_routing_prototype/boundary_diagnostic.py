# Known-linear oracle diagnostics only; no fitted method receives these parameters.
import json
from pathlib import Path
import numpy as np
import torch
from run import generate,dataset,masks,TRUE_K
from core import gaussian_predict
ROOT=Path(__file__).parent
out=[]
for seed in [7,17]:
    y,truth=generate('G5',seed);scale=float(np.std(np.diff(y[:1200])));cap=float(np.quantile(np.abs(np.diff(y[:1200]))/scale,.995));v=torch.tensor(truth['variances'])/scale**2;k=torch.tensor(TRUE_K);predictions={}
    for window in [60,120]:
        o,d,f,t,m=dataset(y,scale,cap,window);ms=masks(o,m);p=gaussian_predict(d,k,v).detach().numpy();predictions[window]=(o,p,t.numpy(),ms['test'].numpy())
        loss=[float(np.mean((p[:,j][ms['test'][:,j]]-t[:,j].numpy()[ms['test'][:,j]])**2)) for j in range(3)]
        out.append({'seed':seed,'window':window,'oracle_test_h':loss,'diagnostic_only':True,'exact_G5_DGP_oracle':False,'limitation':'Gaussian-core covariance omits deliberately correlated adjacent opposite shocks'})
    o60,p60,_,_=predictions[60];o120,p120,t120,m120=predictions[120];i=np.searchsorted(o60,o120)
    delta=float(np.max(np.abs(p60[i][m120]-p120[m120])));out.append({'seed':seed,'60_120_forecast_max_delta_on_common_test_support':delta})
    np.savez_compressed(ROOT/f'G5_{seed}_boundary_ORACLE_DIAGNOSTIC.npz',origins=o120,p60=p60[i],p120=p120,targets=t120,mask=m120,y=y)
(ROOT/'BOUNDARY_ORACLE_DIAGNOSTIC.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
