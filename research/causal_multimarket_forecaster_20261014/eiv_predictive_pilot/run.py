import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import numpy as np
import importlib.util,json,csv,time,hashlib
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'joint_history/matched_information/run.py'
spec=importlib.util.spec_from_file_location('matched',SOURCE); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
j=m.j; b=j.b
LAM=[.01,.1,1,10,100,1000]; RANKS=[1,2,4]
def corrected(x,y,lam,rank,R=None):
 n=len(x); G=x.T@x/n+lam*np.eye(x.shape[1])
 if R is not None: G-=R
 e,v=np.linalg.eigh((G+G.T)/2)
 if e.min()<=1e-8*max(1,float(e.max())): raise ValueError('invalid Gram; no clipping')
 gi=(v*(1/np.sqrt(e)))@v.T
 u,s,vt=np.linalg.svd(gi@(x.T@y/n),full_matrices=False)
 return gi@(u[:,:rank]*s[:rank])@vt[:rank]
def pls(x,y,k):
 a=x.copy(); z=y.copy(); ws=[]; ps=[]; qs=[]
 for _ in range(k):
  u,_,_=np.linalg.svd(a.T@z,full_matrices=False); w=u[:,0]; t=a@w; den=t@t
  if den<1e-15: break
  p=a.T@t/den; q=z.T@t/den
  a-=np.outer(t,p); z-=np.outer(t,q); ws.append(w); ps.append(p); qs.append(q)
 W=np.column_stack(ws); P=np.column_stack(ps); Q=np.column_stack(qs)
 return W@np.linalg.pinv(P.T@W)@Q.T
def fit(data,method):
 x,y,_=data['train']; mu=x.mean(0); sd=np.maximum(x.std(0),1e-8); ym=y.mean(0)
 xx={k:(v[0]-mu)/sd for k,v in data.items()}; yc=y-ym; n=len(x); hist=[]; best=None
 _,_,vt=np.linalg.svd(xx['train'],full_matrices=False)
 configs=[(k,None) for k in RANKS] if method=='PLS' else [(k,l) for k in (RANKS if method!='jointlinear' else [9]) for l in LAM]
 for k,l in configs:
  try:
   if method=='PLS': B=pls(xx['train'],yc,k)
   elif method=='factor':
    V=vt[:k].T; a=xx['train']@V; B=V@np.linalg.solve(a.T@a+l*np.eye(k),a.T@yc)
   else: B=corrected(xx['train'],yc,l/n,k)
   loss=float(np.mean((xx['val']@B+ym-data['val'][1])**2))
   hist.append({'rank':k,'penalty':l,'val_mse':loss,'train_mse':float(np.mean((xx['train']@B+ym-y)**2)),'status':'valid'})
   if best is None or loss<best[0]: best=(loss,B,k,l)
  except ValueError as e: hist.append({'rank':k,'penalty':l,'status':'invalid','reason':str(e)})
 if best is None: raise RuntimeError('no valid configs')
 cp={'B':best[1],'mu':mu,'sd':sd,'ym':ym}
 return xx['test']@best[1]+ym,{'history':hist,'selected_rank':best[2],'selected_penalty':best[3]},cp

def main():
 start=time.monotonic(); rows=[]; checks=[]; splits=[]
 # Independently compare closed-form R=0 with fitted-response SVD RRR.
 rng=np.random.default_rng(7); x=rng.normal(size=(90,8)); y=rng.normal(size=(90,5)); lam=.2
 G=x.T@x/len(x)+lam*np.eye(8); full=np.linalg.solve(G,x.T@y/len(x)); e,v=np.linalg.eigh(G); gh=(v*np.sqrt(e))@v.T
 _,_,vt=np.linalg.svd(gh@full,full_matrices=False); other=full@vt[:2].T@vt[:2]
 eq=float(np.max(np.abs(corrected(x,y,lam,2)-other)))
 ols=float(np.max(np.abs(corrected(x,y,0,5)-np.linalg.lstsq(x,y,rcond=None)[0])))
 if eq>1e-9 or ols>1e-9: raise RuntimeError('equivalence failure')
 try: corrected(x,y,0,2,np.eye(8)*100)
 except ValueError: invalid=True
 else: raise RuntimeError('PSD check failed')
 for seed in (17,43):
  for shared in (True,False):
   cond='shared' if shared else 'no_shared'; p,_=b.generate(seed,shared); data,_=j.dataset(p); ts=data['test'][2]; cut=int(ts[0]); pp=p.copy(); pp[cut+1:]+=rng.normal(10000,1000,pp[cut+1:].shape); pd,_=j.dataset(pp)
   tr=data['train'][2]; va=data['val'][2]
   if not (tr[-1]+max(b.H)<va[0] and va[-1]+max(b.H)<ts[0]): raise RuntimeError('label purge failure')
   splits.append({'seed':seed,'condition':cond,'train_last_label':int(tr[-1]+max(b.H)),'val_first_origin':int(va[0]),'val_last_label':int(va[-1]+max(b.H)),'test_first_origin':cut})
   out=ROOT/f'seed_{seed}_{cond}'; out.mkdir(exist_ok=True)
   target=np.stack([p[t+b.H] for t in ts]).transpose(0,2,1)
   for method in ['independentlinear','jointlinear_matched','jointlinear','RRR','PLS','factor']:
    if method in ('independentlinear','jointlinear_matched'):
     ds=[]; ds2=[]; cps=[]; cps2=[]; infos=[]
     for market in range(3):
      interface='independent_increment' if method=='independentlinear' else 'joint_matched'
      d=m.design(p,seed,interface,market); d2=m.design(pp,seed,interface,market)
      a,info,cp=j.ridge(d); a2,info2,cp2=j.ridge(d2); ds.append(a); ds2.append(a2); cps.append(cp); cps2.append(cp2); infos.append(info)
     delta=np.concatenate(ds,axis=1); delta2=np.concatenate(ds2,axis=1); cp={f'{i}_{k}':v for i,c in enumerate(cps) for k,v in c.items()}; cp2={f'{i}_{k}':v for i,c in enumerate(cps2) for k,v in c.items()}; info={'markets':infos}
    else: delta,info,cp=fit(data,method); delta2,info2,cp2=fit(pd,method)
    equal=all(np.array_equal(cp[k],cp2[k]) for k in cp) and np.array_equal(delta[0],delta2[0])
    if not equal: raise RuntimeError('future perturbation failure '+method)
    checks.append({'seed':seed,'condition':cond,'model':method,'future_cut':cut,'exact_refit_first_forecast_equal':equal})
    pred=p[ts,:,None]+delta.reshape(len(ts),3,3)
    np.savez_compressed(out/f'{method}_predictions.npz',prediction=pred,target=target,origins=ts)
    np.savez_compressed(out/f'{method}_checkpoint.npz',**cp)
    (out/f'{method}_configs.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    mse=np.mean((pred-target)**2,axis=0)
    for market in range(3):
     for hi,h in enumerate(b.H): rows.append({'seed':seed,'condition':cond,'model':method,'market':market,'horizon':int(h),'mse':float(mse[market,hi]),'mae':float(np.mean(np.abs(pred[:,market,hi]-target[:,market,hi])))})
    print(seed,cond,method,float(mse.mean()),flush=True)
    if time.monotonic()-start>600: raise RuntimeError('600 second budget exceeded; keep all files')
 with (ROOT/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
 saved=list(csv.DictReader((ROOT/'metrics.csv').open(encoding='utf-8')))
 if len(saved)!=216: raise RuntimeError('count')
 for r in saved:
  a=np.load(ROOT/f"seed_{r['seed']}_{r['condition']}"/f"{r['model']}_predictions.npz",allow_pickle=False); z=a['prediction'][:,int(r['market']),list(b.H).index(int(r['horizon']))]-a['target'][:,int(r['market']),list(b.H).index(int(r['horizon']))]
  if not np.isclose(np.mean(z*z),float(r['mse']),rtol=1e-12): raise RuntimeError('metric reload')
 agg={c:{k:float(np.mean([r['mse'] for r in rows if r['condition']==c and r['model']==k])) for k in ['independentlinear','jointlinear_matched','jointlinear','RRR','PLS','factor']} for c in ['shared','no_shared']}
 result={'identification_gate':'FAIL','novelty_gate':'BLOCKED','candidate':'not estimated: R not identified','no_correction_ablation':'RRR alias','oracle':'not run: no identified measurement error decomposition','aggregates':agg,'metric_rows':len(saved),'all_metrics_reloaded':True,'future_checks':checks,'splits':splits,'R0_equivalence_max_abs':eq,'OLS_max_abs':ols,'nonPSD_rejected':invalid,'seconds':time.monotonic()-start,'numpy':np.__version__,'source':str(SOURCE),'generator_source':str(j.SOURCE),'source_hashes':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [SOURCE,SOURCE.parent.parent/'run.py',j.SOURCE,ROOT/'DESIGN.md',ROOT/'NEXT_TEST_SPEC.md',ROOT/'PROTOCOL.md']}}
 (ROOT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 (ROOT/'RESULTS.md').write_text('# Identification failed; ordinary low-rank diagnostics only\n\n'+json.dumps(result,indent=2)+'\n\nNo innovation evidence. Price transients cannot be relabeled measurement error. RRR is the exact no-correction degeneration, not a successfully identified EIV candidate. PLS/PCA are ordinary comparators. Search budgets differ: ridge 6, RRR/PCA 18, PLS 3 per fit; independent ridge 3 separate market fits. No market access, commits or push. All configurations and failed/null outcomes retained.\n',encoding='utf-8')
 print(json.dumps(agg,indent=2)); print('verified',len(saved),'seconds',result['seconds'])
if __name__=='__main__': main()
