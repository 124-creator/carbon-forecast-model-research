import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import ast,json,time,pathlib
import numpy as np
from scipy.linalg import solve
from probe import BASE,SCR,features,target,H
# Only three fully inspected pure numpy functions from Stage F, no torch/weights.
source=SCR/'F/carbon_stageF_20261013/prior/lowdim.py'
tree=ast.parse(source.read_text(encoding='utf8'))
tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['bank_parameters','filter_bank','profile_nll']]
namespace={'np':np,'H':H};exec(compile(tree,str(source),'exec'),namespace)
def kernel(x,z,b):return np.exp(-np.maximum((x*x).sum(1)[:,None]+(z*z).sum(1)[None,:]-2*x@z.T,0)/(2*b*b))+1
start=time.monotonic();rows=[];configs=[]
for item in json.loads((BASE/'input_audit.json').read_text(encoding='utf8')):
 path=pathlib.Path(item['path'])
 with np.load(path,allow_pickle=False) as z:p=z[item['read_array']]
 tr=np.arange(60,300);va=np.arange(340,580);ev=np.arange(620,len(p)-20);hs=np.sqrt(np.mean(target(p,tr)**2,0));Y=target(p,tr);yt=target(p,va);ye=target(p,ev)
 bp,ll,_=namespace['filter_bank'](p);j=int(np.argmin(namespace['profile_nll'](ll,1,320)))
 ps={'LOWDIM_MLE':bp[ev,j]}; chosen={'LOWDIM_MLE':{'selected_bank_index':j,'fit_end_exclusive':320}}
 f=features(p);std=f[tr].std(0);std[std<1e-10]=1;f=f/std;best=None
 for b in [.5,2.,8.]:
  K=kernel(f[tr],f[tr],b);Kv=kernel(f[va],f[tr],b)
  for a in [.001,.01,.1,1.]:
   coef=solve(K+a*len(tr)*np.eye(len(tr)),Y,assume_a='pos');loss=float(np.mean(((Kv@coef-yt)/hs)**2));configs.append({'path':path.stem,'bandwidth':b,'alpha':a,'tune_loss':loss})
   if best is None or loss<best[0]:best=(loss,b,a,coef)
 _,b,a,coef=best;ps['RBF_KRR']=kernel(f[ev],f[tr],b)@coef;chosen['RBF_KRR']={'bandwidth':b,'alpha':a}
 for fam,pr in ps.items():
  rr=np.mean((pr-ye)**2,0)/np.mean(ye**2,0);rows.append({'stage':path.parts[-5],'path':path.stem,'family':fam,'relative_mse':float(rr.mean()),'relative_mse_h':rr.tolist(),'selected':chosen[fam]})
 np.savez_compressed(BASE/('strong_predictions_'+path.parts[-5]+'_'+path.stem+'.npz'),origins=ev,**ps)
summary={}
for stage in sorted(set(r['stage'] for r in rows)):
 summary[stage]={}
 for proc in sorted(set(r['path'].rsplit('_',1)[0] for r in rows if r['stage']==stage)):
  summary[stage][proc]={fam:float(np.mean([r['relative_mse'] for r in rows if r['stage']==stage and r['path'].rsplit('_',1)[0]==proc and r['family']==fam])) for fam in ps}
result={'summary':summary,'rows':rows,'configurations':configs,'seconds':time.monotonic()-start,'path_count':len(rows)//2}
(BASE/'STRONG_CONTROLS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({'summary':summary,'path_count':len(rows)//2,'seconds':result['seconds']},indent=2))
