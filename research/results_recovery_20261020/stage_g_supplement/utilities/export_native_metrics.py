"""Descriptive original-price metrics, never used to select a model."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--data-dir',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
rows=[]
for p in Path(a.run).glob('*/RESULTS.json'):
 d=json.load(open(p));z=np.load(p.parent/'private/predictions.npz');base=pd.read_csv(Path(a.data_dir)/f"{d['meta']['market']}_development.csv").price.to_numpy();o=z['origins'];truth=z['observed'];target=base[o,None]+truth
 for model in d['scores']:
  pred=z[model];err=pred-truth[None];se=err*err;sm=se[0]+(se-se[0]).mean(0)
  for hi,h in enumerate((1,5,20)):
   mse=sm[:,hi].mean();vt=np.var(target[:,hi])
   rows.append({'cell':p.parent.name,'model':model,'horizon_records':h,'n_origins':len(o),'MSE':mse,'RMSE_sqrt_mean_seed_MSE':np.sqrt(mse),'MAE':np.abs(err[:,:,hi]).mean(),'MAPE_percent':np.mean(np.abs(err[:,:,hi])/target[None,:,hi])*100,'R2_price_level':1-mse/vt if vt>0 else None,'Q':mse/np.mean(truth[:,hi]**2)})
pd.DataFrame(rows).to_csv(a.out,index=False)
