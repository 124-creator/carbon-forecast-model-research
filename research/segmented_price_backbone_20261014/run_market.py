import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import csv,json,time,hashlib
import numpy as np
from pathlib import Path
from model import train,predict,H
OUT=Path(__file__).parent

def read_prefix():
 path=OUT.parents[1]/'data/parsed/figshare_28682348/guangzhou_reported.csv'; lines=[]
 with path.open('rb',buffering=0) as f:
  for _ in range(951):
   line=bytearray()
   while True:
    b=f.read(1)
    if not b: raise ValueError('short prefix')
    line+=b
    if b==b'\n':break
   lines.append(bytes(line))
 raw=b''.join(lines); rows=list(csv.DictReader(raw.decode('utf-8-sig').splitlines())); d=np.array([r['date'] for r in rows]);p=np.array([float(r['author_reported_price']) for r in rows])
 if len(p)!=950 or d[-1]!='2021-12-31':raise ValueError('prefix')
 return d,p,dict(rows=950,last_date=str(d[-1]),boundary_requested=False,prefix_sha256=hashlib.sha256(raw).hexdigest())
def main():
 if (OUT/'MARKET_RESULTS.json').exists():raise RuntimeError('first-run preservation')
 if not json.loads((OUT/'GATES.json').read_text())['synthetic_pass']:raise RuntimeError('synthetic gate')
 start=time.monotonic();remain=900-json.loads((OUT/'GATES.json').read_text())['elapsed_seconds']; deadline=start+remain
 d,p,audit=read_prefix();end=int(np.sum(d<'2020-01-01'));orig=np.array([t for t in range(end,930) if d[t][:4]==d[t+20][:4]])
 if end!=459 or len(orig)!=451:raise ValueError('protocol')
 idx=np.arange(59,end-1); x=np.stack([p[t-59:t+1] for t in idx]); y=np.full((len(idx),3),np.nan)
 for j,h in enumerate(H):
  ok=idx+h<end;y[ok,j]=p[idx[ok]+h]-p[idx[ok]]
 sx=max(np.std(x-x[:,-1:]),.05);sy=np.maximum(np.nanstd(y,axis=0),.05);xx=x/sx;yy=y/sy
 X=np.column_stack([np.ones(len(idx)),xx-xx[:,-1:]]);test=np.stack([p[t-59:t+1] for t in orig])/sx;TX=np.column_stack([np.ones(len(orig)),test-test[:,-1:]])
 truth=p[orig[:,None]+H]-p[orig,None]; years=np.array([s[:4] for s in d[orig]])
 coef=[]
 for j in range(3):
  ok=np.isfinite(yy[:,j]);pen=np.eye(61);pen[0,0]=0;coef.append(np.linalg.solve(X[ok].T@X[ok]+pen,X[ok].T@yy[ok,j]))
 rows=[]
 for mode in ['linear','segmented','unsegmented']:
  for seed in [7,17]:
   if mode=='linear':pred=TX@np.stack(coef,1)*sy;log={}
   else:
    pp,log=train(xx,yy,mode,seed,deadline);pred=predict(pp,test,mode)*sy
   cells=[]
   for year in ['2020','2021']:
    ok=years==year;cells.append(dict(year=year,relative_mse_h=(np.mean((pred[ok]-truth[ok])**2,0)/np.mean(truth[ok]**2,0)).tolist()))
   row=dict(model=mode,seed=seed,score=float(np.mean([c['relative_mse_h'] for c in cells])),cells=cells,**log);rows.append(row)
   np.savez(OUT/f'market_{seed}_{mode}.npz',prediction=pred,target=truth,origins=orig,years=years,input_scale=sx,target_scale=sy)
   (OUT/'MARKET_RESULTS.json').write_text(json.dumps(rows,indent=2));print(mode,seed,row['score'],flush=True)
 audit.update(initial_fit_end_exclusive=end,per_h_mature_counts=np.isfinite(y).sum(0).tolist(),max_target_index=[int(np.max(idx[np.isfinite(y[:,j])]+h)) for j,h in enumerate(H)],parameter_refresh='none; single initial fit, fixed 60 epochs/480 updates; 8 batches of 49/50',seconds=time.monotonic()-start,development_only=True)
 (OUT/'MARKET_AUDIT.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':main()
