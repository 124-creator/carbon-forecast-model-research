import sys,json,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parents[1]/'torch_segmented_backbone_20261014'))
from prefix_loader import read_prefix
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from threadpoolctl import threadpool_limits
H=np.array([1,5,20])

def features(p,ts,kind):
    if kind.startswith('raw'):
        lag=int(kind[3:]);return np.array([np.diff(p[t-lag:t+1]) for t in ts])
    a=[]
    for t in ts:
        d=np.diff(p[t-120:t+1]);sd=max(float(d.std()),1e-8)
        a.append([np.tanh((p[t]-p[t-59:t+1].mean())/(sd*np.sqrt(60))),d[-20:].std()/sd,d[-1],d[-5:].mean(),d[-20:].mean(),d[-60:].std()])
    return np.array(a)

def fit(x,y):
    with threadpool_limits(limits=1):return MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=100,learning_rate=.05,max_leaf_nodes=7,min_samples_leaf=40,l2_regularization=.1,early_stopping=False,random_state=17),n_jobs=1).fit(x,y)

rows,audit=read_prefix(R.parents[2]/'data/parsed/figshare_28682348/guangzhou_reported.csv',950,'2021-12-31')
p=np.array([float(r['author_reported_price']) for r in rows]);dates=np.array([r['date'] for r in rows]);orig=np.array([t for t in range(459,930) if dates[t][:4]==dates[t+20][:4]])
truth=p[orig[:,None]+H]-p[orig,None];years=np.array([d[:4] for d in dates[orig]])
preds={k:np.full(truth.shape,np.nan) for k in ['raw60','raw120','compact']};receipts=[];checks=[]
for block in range(12):
    end=459+block*40;ts=np.arange(max(120,end-20-480),end-20);ids=np.where((orig-459)//40==block)[0];y=p[ts[:,None]+H]-p[ts,None]
    for kind in preds:
        x=features(p,ts,kind);start=time.perf_counter();m=fit(x,y);preds[kind][ids]=m.predict(features(p,orig[ids],kind));elapsed=time.perf_counter()-start
        receipts.append(dict(block=block,kind=kind,fit_end=end,latest_label=int(ts[-1]+20),n=len(ts),seconds=elapsed))
        if block==0:
            q=p.copy();q[end:]+=10000;xx=features(q,ts,kind);yy=q[ts[:,None]+H]-q[ts,None];n=fit(xx,yy)
            if not np.array_equal(x,xx) or not np.array_equal(y,yy) or not np.array_equal(m.predict(x[-3:]),n.predict(x[-3:])):raise ValueError('future refit')
            checks.append(dict(kind=kind,passed=True,fit_end=end))
result={}
for kind,pred in preds.items():
    if not np.isfinite(pred).all():raise ValueError('incomplete')
    cells=[]
    for year in ['2020','2021']:
        mask=years==year;rat=np.mean((pred[mask]-truth[mask])**2,axis=0)/np.mean(truth[mask]**2,axis=0)
        cells.append(dict(year=year,n=int(mask.sum()),ratios=rat.tolist()))
    result[kind]=dict(score=float(np.mean([v for c in cells for v in c['ratios']])),cells=cells)
local=R/'local_only';local.mkdir(exist_ok=True)
np.savez_compressed(local/'PREDICTIONS.npz',origins=orig,years=years,truth=truth,**preds)
(R/'RESULTS.json').write_text(json.dumps(dict(results=result,receipts=receipts,causal=checks,scope='development baseline representation, not novelty'),indent=2))
print(json.dumps(result,indent=2))
