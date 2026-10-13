"""Predeclared independent controls; fixed hyperparameters, not market tuning."""
from run_market import R,O,P
from models import *
import json,time

def generate(kind,seed,n=1600):
 rng=np.random.default_rng(seed);b=np.zeros(n);ell=np.zeros(n);v=np.zeros(n);rho=np.full(n,.6)
 if kind=='varying':rho=np.where((np.arange(n)//120)%2==0,.2,.9)
 p=np.zeros(n)
 for t in range(1,n):
  if kind=='RW':p[t]=p[t-1]+rng.normal();continue
  b[t]=.98*b[t-1]+.02*rng.normal();ell[t]=ell[t-1]+b[t]+.05*rng.normal();v[t]=rho[t]*v[t-1]+rng.normal();p[t]=ell[t]+v[t]
 p+=100;oracle=np.full((n,3),np.nan)
 for t in range(60,n-20):
  for j,h in enumerate(H):oracle[t,j]=0 if kind=='RW' else b[t]*sum(.98**k for k in range(1,h+1))+(np.prod(rho[t+1:t+h+1])-1)*v[t]
 return p,oracle

def main():
 records=[];details=[]
 for kind in ['RW','constant','varying']:
  for seed in [101,202,303,404,505]:
   pp,oracle=generate(kind,seed);d=make_data(pp,pd.date_range('2000-01-01',periods=len(pp)).values,800);ids=np.arange(800,len(pp)-20);rw=np.mean(d.y[ids]**2,axis=0)
   for method in ['RW','latent_state_oracle','Ridge_diff','free','global','conditional']:
    if method=='RW':pred=np.zeros_like(d.y)
    elif method=='latent_state_oracle':pred=oracle
    elif method=='Ridge_diff':pred=baseline_forecasts(d,method,dict(lag=20,alpha=.1))
    else:pred,_,_=neural_forecasts(d,method,.001,.01,11)
    mse=np.mean((pred[ids]-d.y[ids])**2,axis=0)
    records.append(dict(process=kind,path_seed=seed,model=method,relative_mse=float(np.mean(mse/rw)),**{f'mse_h{h}':float(mse[j]) for j,h in enumerate(H)}))
   print(kind,seed,'completed',flush=True)
   pd.DataFrame(records).to_csv(O/'synthetic_per_path.csv',index=False)
 (O/'synthetic_complete.json').write_text(json.dumps({'completed':True,'paths':15,'neural_trajectories':45,'latent_oracle_information':'known latent drift and transient state plus predetermined regime transition schedule; not available to fitted models; diagnostic lower bound, not market comparator'},indent=2))
if __name__=='__main__':main()
