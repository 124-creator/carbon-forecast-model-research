import sys,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parents[2]/'torch_segmented_backbone_20261014'))
from prefix_loader import read_prefix

def features(p,ids):
    x=np.array([np.diff(p[t-60:t+1]) for t in ids]);v=np.column_stack([abs(x[:,-1]),x[:,-5:].std(axis=1),x[:,-20:].std(axis=1),x.std(axis=1)])
    return x,v

def targets(p,ids):
    provisional=np.array([(p[t-4:t+1].sum()+4*p[t])/9 for t in ids]);mature=np.array([p[t-4:t+5].mean() for t in ids]);direct=np.array([(p[t+1:t+5]-p[t]).sum()/9 for t in ids])
    return mature-provisional,direct

def ridge(x,y):
    reg=10*np.eye(x.shape[1]);reg[0,0]=0
    return np.linalg.solve(x.T@x+reg,x.T@y)

def fit(p,end):
    ids=np.arange(max(60,end-4-480),end-4);tr=ids[:-64];va=ids[-60:]
    x,v=features(p,tr);sd=max(float(x.std()),1e-6);y,_=targets(p,tr);b=ridge(np.column_stack([np.ones(len(x)),x/sd]),y/sd)
    xv,vv=features(p,va);yv,_=targets(p,va);err=yv/sd-np.column_stack([np.ones(len(xv)),xv/sd])@b;c=max(float(np.mean(err**2)),1e-8)
    vm=vv.mean(axis=0);vs=np.maximum(vv.std(axis=0),1e-6);a=ridge(np.column_stack([np.ones(len(vv)),(vv-vm)/vs]),err**2)
    return (b,a,sd,c,vm,vs),dict(end=end,n_mean=len(tr),n_variance=len(va),max_mean_target=int(tr[-1]+4),variance_first=int(va[0]),max_variance_target=int(va[-1]+4))

def predict(state,p,ids):
    b,a,sd,c,vm,vs=state;x,v=features(p,ids);mu=np.column_stack([np.ones(len(x)),x/sd])@b*sd
    var=np.clip(np.column_stack([np.ones(len(v)),(v-vm)/vs])@a,.1*c,10*c)*sd**2
    return mu,var,np.full(len(x),c*sd**2)

def main():
    rows,_=read_prefix(R.parents[3]/'data/parsed/figshare_28682348/guangzhou_reported.csv',950,'2021-12-31');p=np.array([float(v['author_reported_price']) for v in rows]);dates=np.array([v['date'] for v in rows]);orig=np.array([t for t in range(459,930) if dates[t][:4]==dates[t+20][:4]])
    y,direct=targets(p,orig);err=float(np.max(abs(y-direct)))
    if err>1e-12:raise ValueError('algebra identity')
    mu=np.zeros(len(orig));var=mu.copy();const=mu.copy();logs=[]
    for block in range(12):
        end=459+40*block;ids=np.where((orig-459)//40==block)[0];state,log=fit(p,end);mu[ids],var[ids],const[ids]=predict(state,p,orig[ids]);logs.append(log)
    q=p.copy();q[600:]+=10000;a,_=fit(p,579);b,_=fit(q,579)
    if not all(np.array_equal(x,y) for x,y in zip(a,b)) or not all(np.array_equal(x,y) for x,y in zip(predict(a,p,[579]),predict(b,q,[579]))):raise ValueError('future refit')
    years=np.array([v[:4] for v in dates[orig]]);results=[]
    for yr in ['all','2020','2021']:
        mask=np.ones(len(orig),dtype=bool) if yr=='all' else years==yr
        e=(y[mask]-mu[mask])**2
        results.append(dict(year=yr,n=int(mask.sum()),mean_mse_ratio=float(np.mean(e)/np.mean(y[mask]**2)),constant_nll=float(np.mean(.5*(np.log(2*np.pi*const[mask])+e/const[mask]))),conditional_nll=float(np.mean(.5*(np.log(2*np.pi*var[mask])+e/var[mask]))),constant_residual_second_moment=float(np.mean(e/const[mask])),conditional_residual_second_moment=float(np.mean(e/var[mask]))))
    local=R/'local_only';local.mkdir(exist_ok=True);np.savez_compressed(local/'PREDICTIONS.npz',origins=orig,years=years,truth=y,direct=direct,mean=mu,variance=var,constant=const)
    (R/'RESULTS.json').write_text(json.dumps(dict(results=results,fits=logs,identity_max_error=err,future_refit_passed=True,origins=len(orig)),indent=2));print(json.dumps(results,indent=2))
if __name__=='__main__':main()
